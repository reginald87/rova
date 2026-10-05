"""Driver telemetry ingestion.

This is the only path by which a GPS fix enters the system, and it is written so
that a driver device can be hostile without becoming a security or availability
problem. The order of the checks is the design:

1. **Identity** is derived from the authenticated connection. The frame never
   names a driver — see the schema module.
2. **Shape** was already validated by the schema (finite numbers, in range).
3. **Policy** is applied here: rate limit, then minimum-interval coalescing.
4. **The fix is stored** in Redis with the *server's* receipt time, and only then
   broadcast to the audiences derived from authoritative state.

Two deliberate consequences:

* **No database write per fix.** ``DriverLocation`` mirroring is off by default;
  a driver moving at 1 Hz would otherwise produce 86 400 upserts a day, and none
  of them is history ROVA keeps.
* **The client clock is diagnostics only.** A phone whose clock is an hour fast
  would otherwise poison every freshness calculation in dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from django.conf import settings

from apps.geography.services.geography_service import (
    GeographyError,
    InvalidCoordinateError,
    calculate_distance,
    validate_coordinate,
)
from apps.mobility.choices import TripStatus

from ..errors import RateLimitedError
from ..logging import log_event, log_warning_event
from .events import publish_driver_location
from .location_store import (
    DriverLocationRecord,
    LocationStoreUnavailableError,
    LocationUpdateTooFrequentError,
    resolve_location_store,
)

# Trip statuses during which a driver is "on" a trip for tracking purposes. A
# position reaches the trip's channel only while the trip is genuinely in one of
# these states; before assignment and after completion there is no rider watching a
# marker, so the trip audience is dropped and only operations (which legitimately
# watch the whole fleet) receives the fix. Derived from the Phase 8 choices rather
# than spelled out, so a new in-trip status cannot silently go untracked.
TRACKED_TRIP_STATUSES = (
    TripStatus.DRIVER_ASSIGNED,
    TripStatus.DRIVER_ACCEPTED,
    TripStatus.DRIVER_ARRIVING,
    TripStatus.DRIVER_AT_PICKUP,
    TripStatus.RIDER_ONBOARD,
    TripStatus.TRIP_STARTED,
)

# While en route the rider is watching "how far away is my driver", so the
# pickup distance is worth computing; after the rider is aboard it is noise.
PRE_PICKUP_TRIP_STATUSES = (
    TripStatus.DRIVER_ASSIGNED,
    TripStatus.DRIVER_ACCEPTED,
    TripStatus.DRIVER_ARRIVING,
)


@dataclass(frozen=True, slots=True)
class LocationIngestOutcome:
    """What happened to one submitted fix."""

    accepted: bool
    record: DriverLocationRecord | None = None
    trip_id: str | None = None
    distance_to_pickup_meters: int | None = None
    reason: str = ""


def _server_now() -> datetime:
    return datetime.now(UTC)


def _rate_limit(name: str) -> tuple[int, int]:
    limits: dict[str, tuple[int, int]] = getattr(settings, "REALTIME_RATE_LIMITS", {})
    return limits.get(name, (0, 0))


def check_driver_location_rate_limit(driver_id: object) -> None:
    """Fixed-window backstop above the coalescing rule.

    A driver streaming as fast as the network allows is not useful, but it is
    cheap to serve; the window limit exists to bound a single connection's Redis
    and broadcast cost, not to ration a legitimate 0.5 Hz stream.
    """
    from apps.accounts.services.ratelimit import RateLimitExceeded, check_rate_limit

    limit, period = _rate_limit("ws_location")
    if limit <= 0:
        return
    try:
        check_rate_limit("ws:driver:location", str(driver_id), limit=limit, period=period)
    except RateLimitExceeded as exc:
        log_warning_event(
            "realtime_rate_limited", scope="driver_location", driver_id=str(driver_id)
        )
        raise RateLimitedError(
            "Too many location updates. Send them more slowly.",
        ) from exc


def resolve_tracked_trip(driver_id: object) -> Any | None:
    """The trip this driver is currently driving, or ``None``.

    Expensive enough (one indexed query) that callers cache it per connection
    rather than calling it per fix — see :class:`apps.realtime.consumers.
    DriverConsumer`.
    """
    from apps.mobility.models import Trip

    # ``ride_request`` is joined rather than lazily loaded: the pre-pickup pickup
    # distance reads its origin on every fix, and a deferred relation here would
    # turn each of those into an extra query per GPS point.
    return (
        Trip.objects.filter(
            driver_id=driver_id,  # type: ignore[misc]
            status__in=TRACKED_TRIP_STATUSES,
        )
        .select_related("ride_request")
        .only(
            "id",
            "status",
            "ride_request_id",
            "ride_request__origin_latitude",
            "ride_request__origin_longitude",
        )
        .order_by("-assigned_at")
        .first()
    )


def _pickup_distance_meters(record: DriverLocationRecord, trip) -> int | None:
    """Distance from the driver to the pickup, when it is worth computing.

    Only meaningful before the rider is in the car; afterwards the number is
    noise. The trip's origin comes from the ride request, which is selected by
    the caller so this stays a geometry calculation rather than another query.
    """
    if trip is None or trip.status not in PRE_PICKUP_TRIP_STATUSES:
        return None
    ride_request = getattr(trip, "ride_request", None)
    if ride_request is None:
        return None
    try:
        origin = validate_coordinate(
            float(ride_request.origin_latitude), float(ride_request.origin_longitude)
        )
        current = validate_coordinate(record.latitude, record.longitude)
        return int(round(calculate_distance(current, origin)))
    except (InvalidCoordinateError, GeographyError, TypeError, ValueError):
        # A pickup distance is a convenience field. If the geometry cannot be
        # computed the position is still valid and still broadcast.
        return None


def ingest_driver_location(
    *,
    driver_id: object,
    message,
    trip=None,
) -> LocationIngestOutcome:
    """Validate, store and fan out one ``location.update`` frame.

    ``trip`` is the caller's cached view of the driver's active trip, or ``None``.
    Passing it in keeps the telemetry path free of per-fix database queries; the
    caller refreshes it on trip events and on a TTL.

    Coalesced fixes return ``accepted=False`` and are dropped *before* any write
    or broadcast. That is the whole point of coalescing: the excess never costs a
    Redis round trip, and the client simply sees a position that updates at the
    configured cadence instead of faster.
    """
    check_driver_location_rate_limit(driver_id)

    store = resolve_location_store()
    try:
        record = store.set_location(
            driver_id=driver_id,
            latitude=message.latitude,
            longitude=message.longitude,
            accuracy=message.accuracy,
            heading=message.heading,
            speed=message.speed,
            client_timestamp=message.client_timestamp,
        )
    except LocationUpdateTooFrequentError:
        # Expected, not an error: the device is faster than the configured
        # cadence. Nothing is written and nothing is sent.
        log_event(
            "location_update_coalesced",
            driver_id=str(driver_id),
            min_interval_ms=getattr(settings, "DRIVER_LOCATION_MIN_INTERVAL_MS", 0),
        )
        return LocationIngestOutcome(accepted=False, reason="coalesced")
    except LocationStoreUnavailableError as exc:
        log_warning_event("location_store_unavailable", driver_id=str(driver_id))
        raise exc

    trip_id = str(trip.pk) if trip is not None else None
    distance_to_pickup = _pickup_distance_meters(record, trip)
    publish_driver_location(
        record=record,
        trip=trip,
        distance_to_pickup_meters=distance_to_pickup,
    )
    return LocationIngestOutcome(
        accepted=True,
        record=record,
        trip_id=trip_id,
        distance_to_pickup_meters=distance_to_pickup,
        reason="stored",
    )


def assert_trip_scope(message_trip_id: str | None, trip) -> None:
    """Reject a frame that claims a trip the driver is not driving.

    Treated as non-fatal: the device raced the server's state machine, and the
    remedy is a fresh snapshot on the next fix rather than a closed socket.
    """
    from ..errors import TripScopeMismatchError

    if message_trip_id is None:
        return
    if trip is not None and str(trip.pk) == str(message_trip_id):
        return
    log_warning_event(
        "location_trip_scope_mismatch",
        claimed_trip_id=str(message_trip_id),
        actual_trip_id=str(trip.pk) if trip is not None else None,
    )
    raise TripScopeMismatchError("This trip is not the trip you are currently driving.")
