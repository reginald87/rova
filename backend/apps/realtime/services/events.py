"""Realtime event publication.

One rule shapes this module: **a frame is published only after the database
transaction that produced it has committed.** Every public ``publish_*`` helper
defers with ``transaction.on_commit`` when it is called inside an atomic block, so
a client can never be told a trip was cancelled by a transaction that then rolled
back. Business truth is PostgreSQL; a frame is a notification about it, and the
notification is allowed to be lost — never to lie.

The inverse is also true and is the reason publication is wrapped defensively: a
broadcast failure (Redis down, no consumer, serialisation surprise) is logged and
swallowed. A completed trip must not be rolled back because nobody was listening
on a WebSocket.

Event envelope
--------------
Every frame carries the same four identifying fields so a mobile client can
deduplicate and order without special-casing each event type::

    {
      "event_id": "uuid4",        # unique per frame; the dedup key
      "type": "trip.status",      # versioned event name
      "timestamp": "...Z",        # server emit time, UTC
      "sequence": 7,              # per-trip TripEvent ordinal (trip events only)
      ... type-specific payload
    }
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, overload

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.db import transaction

from apps.dispatch import choices as dispatch_choices
from apps.mobility.choices import TripStatus

from ..choices import RealtimeEventType
from ..logging import log_event, log_warning_event
from .groups import driver_group, operations_groups, trip_group

# Bumped only for a breaking change to an existing event's shape. Additive
# fields do not bump it.
EVENT_ENVELOPE_VERSION = 1

# Terminal states that every client renders as "this trip is over", grouped by the
# frame they share. Derived from the Phase 8 choices so a new terminal status
# cannot be forgotten here — it simply would not get a dedicated event type, and
# would fall through to ``trip.status``.
CANCELLED_TRIP_STATUSES = (
    TripStatus.CANCELLED_BY_RIDER,
    TripStatus.CANCELLED_BY_DRIVER,
    TripStatus.ADMIN_CANCELLED,
    TripStatus.DRIVER_NO_SHOW,
    TripStatus.RIDER_NO_SHOW,
    TripStatus.EXPIRED,
)
COMPLETED_TRIP_STATUSES = (TripStatus.TRIP_COMPLETED,)


@overload
def _isoformat(value: datetime) -> str: ...


@overload
def _isoformat(value: None) -> None: ...


@overload
def _isoformat(value: datetime | None) -> str | None: ...


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def server_now_isoformat() -> str:
    """Server clock, in the same format every frame's ``timestamp`` uses."""
    return _isoformat(datetime.now(UTC))


def build_event(event_type: str, *, sequence: int | None = None, **payload: Any) -> dict:
    """Assemble one event frame.

    ``event_id`` is generated here rather than by the caller so that no code path
    can publish two frames that share an id — the property a client relies on to
    drop duplicates after a reconnect.
    """
    frame: dict[str, Any] = {
        "event_id": str(uuid.uuid4()),
        "type": str(event_type),
        "version": EVENT_ENVELOPE_VERSION,
        "timestamp": server_now_isoformat(),
    }
    if sequence is not None:
        frame["sequence"] = int(sequence)
    for key, value in payload.items():
        if value is not None:
            frame[key] = value
    return frame


def _after_commit(action) -> None:
    """Run ``action`` after the current transaction commits, or immediately.

    Django runs ``on_commit`` callbacks outside the transaction, which is exactly
    the guarantee needed. Outside any atomic block there is nothing to wait for,
    so the work happens inline rather than being deferred to a request that never
    commits.
    """
    if transaction.get_connection().in_atomic_block:
        transaction.on_commit(action)
    else:
        action()


def _group_send(group: str, frame: dict) -> None:
    """Send one frame to one group, never raising into business code."""
    try:
        channel_layer = get_channel_layer()
        if channel_layer is None:
            log_warning_event("realtime_broadcast_skipped reason=no_channel_layer", group=group)
            return
        async_to_sync(channel_layer.group_send)(group, {"type": "realtime.frame", "frame": frame})
        log_event("realtime_broadcast", group=group, event_type=frame.get("type"))
    except Exception:
        # Redis hiccup, serialization surprise, or an unhealthy channel layer.
        # PostgreSQL is already authoritative at this point; losing a frame is
        # recoverable by reconnecting, so it must not unwind committed work.
        log_warning_event("realtime_broadcast_failed", group=group, event_type=frame.get("type"))


def _broadcast(groups: Iterable[str], frame: dict) -> None:
    for group in groups:
        _group_send(group, frame)


def _trip_audience(trip) -> list[str]:
    """Who is entitled to hear about a change to this trip.

    The participants plus operations. The assigned driver's own group is included
    as well: the driver app needs to know it has been given a trip, and that is
    also what lets a streaming socket resolve its active trip without a database
    query per GPS fix.
    """
    groups = [trip_group(trip.pk), *operations_groups()]
    if trip.driver_id:
        groups.append(driver_group(trip.driver_id))
    return groups


# ---------------------------------------------------------------------------
# Trip events (Phase 8 state machine → clients)
# ---------------------------------------------------------------------------


def trip_sequence(trip) -> int | None:
    """Ordinal of the trip's newest lifecycle event.

    Reuses the Phase 8 ``TripEvent`` history rather than keeping a second
    sequence of its own, so a client that reconnects and walks ``TripEvent``s
    lands on the same numbering the live stream used.
    """
    from apps.mobility.models import TripEvent

    try:
        return TripEvent.objects.filter(trip_id=trip.pk).count() or None
    except Exception:  # pragma: no cover - sequence is a nicety, never fatal
        return None


def _cached_fk(instance, field_name: str):
    """Read a forward relation only if the caller already loaded it.

    Realtime publication runs inside request/worker paths that have no business
    issuing extra queries per broadcast. Whatever the caller preloaded is
    included; anything absent is simply omitted from the frame and stays
    available from the REST endpoint the client already has.
    """
    return getattr(instance._state, "fields_cache", {}).get(field_name)


def trip_payload(trip) -> dict[str, Any]:
    """The trip projection carried by every trip event.

    Public because snapshots and live events must agree: a client that renders a
    reconnect snapshot and then an incremental event has to see the same keys, or
    its model has two shapes to reconcile.

    Deliberately not a serialized ``Trip``: a status broadcast happens on every
    lifecycle transition and an operations feed on every one of them, so it must
    not drag pricing snapshots, cancellation fees and city relations along with
    it. Keys follow the Phase 4/8 REST vocabulary so one client model serves
    both transports, and relations are read only when the caller preloaded them.
    """
    ride_request = _cached_fk(trip, "ride_request")
    estimate = _cached_fk(trip, "estimate_fare")
    final = _cached_fk(trip, "final_fare")
    category = _cached_fk(trip, "vehicle_category")
    origin_city = _cached_fk(trip, "origin_city")
    destination_city = _cached_fk(trip, "destination_city")
    origin_zone = _cached_fk(trip, "origin_service_zone")
    destination_zone = _cached_fk(trip, "destination_service_zone")
    return {
        "trip_id": str(trip.pk),
        "status": str(trip.status),
        "driver_id": str(trip.driver_id) if trip.driver_id else None,
        "vehicle_id": str(trip.vehicle_id) if getattr(trip, "vehicle_id", None) else None,
        "vehicle_category_key": category.key if category is not None else None,
        "rider_id": str(trip.rider_id) if trip.rider_id else None,
        "ride_request_id": str(trip.ride_request_id) if trip.ride_request_id else None,
        "origin_address": getattr(ride_request, "origin_address", None) or None,
        "destination_address": getattr(ride_request, "destination_address", None) or None,
        "origin_city_slug": origin_city.slug if origin_city is not None else None,
        "destination_city_slug": destination_city.slug if destination_city is not None else None,
        "origin_service_zone_slug": origin_zone.slug if origin_zone is not None else None,
        "destination_service_zone_slug": (
            destination_zone.slug if destination_zone is not None else None
        ),
        "estimate_total": str(estimate.total) if estimate is not None else None,
        "final_total": str(final.total) if final is not None else None,
        "cancel_reason": trip.cancel_reason or None,
    }


def publish_trip_status(trip) -> None:
    """Broadcast an authoritative trip status change.

    Called after the Phase 8 state machine has committed. ``trip`` is read for
    its committed values only.
    """

    def _emit() -> None:
        frame = build_event(
            RealtimeEventType.TRIP_STATUS,
            sequence=trip_sequence(trip),
            **trip_payload(trip),
        )
        log_event("trip_event_broadcast", trip_id=trip.pk, status=trip.status)
        _broadcast(_trip_audience(trip), frame)

    _after_commit(_emit)


def publish_trip_assigned(trip) -> None:
    """A driver has been attached to the trip."""

    def _emit() -> None:
        frame = build_event(
            RealtimeEventType.TRIP_ASSIGNED,
            sequence=trip_sequence(trip),
            **trip_payload(trip),
        )
        log_event("trip_event_broadcast", trip_id=trip.pk, event_type="trip.assigned")
        _broadcast(_trip_audience(trip), frame)

    _after_commit(_emit)


def publish_trip_cancelled(trip) -> None:
    """The trip reached one of its terminal cancellation states."""

    def _emit() -> None:
        frame = build_event(
            RealtimeEventType.TRIP_CANCELLED,
            sequence=trip_sequence(trip),
            **trip_payload(trip),
        )
        log_event("trip_event_broadcast", trip_id=trip.pk, event_type="trip.cancelled")
        _broadcast(_trip_audience(trip), frame)

    _after_commit(_emit)


def publish_trip_completed(trip) -> None:
    """The trip reached ``TRIP_COMPLETED``."""

    def _emit() -> None:
        payload = trip_payload(trip)
        frame = build_event(
            RealtimeEventType.TRIP_COMPLETED,
            sequence=trip_sequence(trip),
            distance_meters=trip.distance_meters,
            duration_seconds=trip.duration_seconds,
            **payload,
        )
        log_event("trip_event_broadcast", trip_id=trip.pk, event_type="trip.completed")
        _broadcast(_trip_audience(trip), frame)

    _after_commit(_emit)


def publish_trip_status_change(trip) -> None:
    """Publish the one event type that matches ``trip``'s committed status.

    Convenience entry point for the mobility state machine: it maps the status to
    the event that clients actually distinguish, without the domain service
    needing to know the realtime vocabulary. Statuses with no dedicated event type
    fall through to ``trip.status``, which is why the vocabulary stays small.
    """
    status = trip.status
    if status == TripStatus.DRIVER_ASSIGNED:
        publish_trip_assigned(trip)
    elif status in CANCELLED_TRIP_STATUSES:
        publish_trip_cancelled(trip)
    elif status in COMPLETED_TRIP_STATUSES:
        publish_trip_completed(trip)
    else:
        publish_trip_status(trip)


# ---------------------------------------------------------------------------
# Driver events (Phase 9 dispatch → the driver who is affected)
# ---------------------------------------------------------------------------


def _offer_payload(offer) -> dict[str, Any]:
    """What a driver needs to answer an offer.

    The same thin projection the REST offer endpoint returns: the ride, the
    pickup, the fare and the deadline. Ranking internals, the search radius and
    the other candidates are server-side decisions and stay server-side.
    """
    ride_request = offer.ride_request
    fare = ride_request.fare
    vehicle = _cached_fk(offer, "vehicle")
    category = _cached_fk(vehicle, "category") if vehicle is not None else None
    return {
        "offer_id": str(offer.pk),
        "trip_id": str(offer.trip_id),
        "ride_request_id": str(offer.ride_request_id),
        "driver_id": str(offer.driver_id),
        "vehicle_id": str(offer.vehicle_id),
        "vehicle_category_key": category.key if category is not None else None,
        "status": str(offer.status),
        "origin_address": ride_request.origin_address,
        "destination_address": ride_request.destination_address,
        "origin_latitude": float(ride_request.origin_latitude),
        "origin_longitude": float(ride_request.origin_longitude),
        "distance_to_pickup_meters": offer.distance_to_pickup_meters,
        "expires_at": _isoformat(offer.expires_at),
        "estimated_fare_total": str(fare.total) if fare is not None else None,
        "currency": ride_request.currency or None,
    }


def publish_driver_offer(offer, *, outcome: str) -> None:
    """Broadcast an offer lifecycle outcome to the driver it was extended to.

    ``outcome`` is one of ``created`` / ``expired`` / ``declined`` /
    ``superseded`` / ``cancelled``. Where an outcome and the offer's committed
    status disagree, the status wins: a driver is told what the database says
    happened, not what a caller believed was happening.
    """
    status = str(offer.status)
    by_outcome: dict[str, RealtimeEventType] = {
        "created": RealtimeEventType.DRIVER_OFFER_CREATED,
        "expired": RealtimeEventType.DRIVER_OFFER_EXPIRED,
        "declined": RealtimeEventType.DRIVER_OFFER_DECLINED,
        "superseded": RealtimeEventType.DRIVER_OFFER_SUPERSEDED,
        "cancelled": RealtimeEventType.DRIVER_OFFER_CANCELLED,
    }
    event_type = by_outcome.get(outcome, RealtimeEventType.DRIVER_OFFER_CREATED)
    if outcome == "created" and status != dispatch_choices.DriverOfferStatus.PENDING:
        # A pending offer that is no longer pending is not an offer any more;
        # announce the outcome that actually applies. Keyed by `str` because
        # `status` is a string above: the choice members only happen to compare
        # equal to their values because Django derives each name from it.
        by_status: dict[str, RealtimeEventType] = {
            dispatch_choices.DriverOfferStatus.EXPIRED: RealtimeEventType.DRIVER_OFFER_EXPIRED,
            dispatch_choices.DriverOfferStatus.DECLINED: RealtimeEventType.DRIVER_OFFER_DECLINED,
            dispatch_choices.DriverOfferStatus.SUPERSEDED: (
                RealtimeEventType.DRIVER_OFFER_SUPERSEDED
            ),
            dispatch_choices.DriverOfferStatus.CANCELLED: (
                RealtimeEventType.DRIVER_OFFER_CANCELLED
            ),
        }
        event_type = by_status.get(status, event_type)

    driver_id = offer.driver_id

    def _emit() -> None:
        frame = build_event(event_type, **_offer_payload(offer))
        log_event("driver_event_broadcast", driver_id=driver_id, event_type=str(event_type))
        _broadcast([driver_group(driver_id)], frame)

    _after_commit(_emit)


def publish_driver_availability(driver) -> None:
    """Reflect an authoritative availability change.

    ``DriverProfile.availability_status`` remains the single business truth; this
    only tells a driver's own socket and the operations stream that it moved.
    A dropped WebSocket never calls this — a lost connection is not a business
    state change.
    """
    driver_id = driver.pk

    def _emit() -> None:
        frame = build_event(
            RealtimeEventType.DRIVER_AVAILABILITY,
            driver_id=str(driver_id),
            availability=str(driver.availability_status),
            available_since=_isoformat(driver.available_since),
        )
        log_event(
            "driver_event_broadcast",
            driver_id=driver_id,
            event_type=str(RealtimeEventType.DRIVER_AVAILABILITY),
        )
        _broadcast([driver_group(driver_id), *operations_groups()], frame)

    _after_commit(_emit)


# ---------------------------------------------------------------------------
# Driver location (realtime ingest → trip + operations)
# ---------------------------------------------------------------------------


def publish_driver_location(
    *, record, trip=None, distance_to_pickup_meters: int | None = None
) -> None:
    """Broadcast an accepted position.

    Audience is decided by the caller from authoritative state, never by the
    client: the trip group only when the driver is genuinely on that trip, plus
    the operations stream. There is deliberately no "all nearby drivers" group —
    a rider's socket is never a candidate's socket.
    """
    driver_id = record.driver_id
    payload = record.to_client_payload()
    frame = build_event(
        RealtimeEventType.DRIVER_LOCATION,
        trip_id=str(trip.pk) if trip is not None else None,
        driver_id=str(driver_id),
        driver=payload,
        distance_to_pickup_meters=distance_to_pickup_meters,
    )

    groups = [trip_group(trip.pk)] if trip is not None else []
    groups.extend(operations_groups())
    log_event(
        "location_broadcast",
        driver_id=driver_id,
        trip_id=getattr(trip, "pk", None),
        recipients=len(groups),
    )
    _broadcast(groups, frame)


__all__ = [
    "EVENT_ENVELOPE_VERSION",
    "build_event",
    "publish_driver_availability",
    "publish_driver_location",
    "publish_driver_offer",
    "publish_trip_assigned",
    "publish_trip_cancelled",
    "publish_trip_completed",
    "publish_trip_status",
    "publish_trip_status_change",
    "trip_sequence",
]
