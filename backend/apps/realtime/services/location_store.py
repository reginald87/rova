"""Redis-backed current driver locations.

This is the one place in ROVA that answers "where is this driver right now". It
satisfies the same ``DriverLocationStore`` protocol dispatch (Phase 9) already
depends on, so candidate search and live rider tracking read the *same* source
instead of two sources that can disagree.

Design
------
A position is two Redis keys:

* ``rova:rt:driver-location:{driver_id}`` — a hash with the position itself and
  a server-stamped receipt time, carrying a TTL.
* ``rova:rt:driver-geo`` — one GEO sorted set used for radius search.

The hash is the record; the sorted set is only an index. That split is what makes
eviction safe: Redis expires the hash on its own, so an abandoned driver's
position disappears without any sweeper. The sorted set cannot expire per member,
so members whose hash has gone are pruned lazily on read and by a periodic task —
otherwise the index would keep returning drivers who left hours ago.

Freshness is a *computed* property, never stored: ``server_received_at`` lives in
the hash and the window lives in settings, so changing the policy takes effect
immediately instead of after a TTL cycle. A position that is present but too old
is reported ``STALE`` and is never treated as a current location.

Client clocks are recorded but never trusted. ``client_timestamp`` is kept for
diagnostics; ``server_received_at`` — stamped by the server at receipt — is the
only timestamp freshness is computed from.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from django.conf import settings
from django.contrib.gis.geos import Point
from django.core.cache import cache
from django.utils import timezone
from redis.exceptions import RedisError

from apps.dispatch.services.location_service import DriverPosition

from ..choices import LocationFreshness
from ..logging import log_warning_event

# Hash fields. Short names: a hash is rewritten on every GPS fix, and mobile
# bandwidth is billed per byte.
_F_DRIVER = "d"
_F_LATITUDE = "lat"
_F_LONGITUDE = "lon"
_F_CLIENT_AT = "cat"
_F_RECEIVED_AT = "rat"
_F_ACCURACY = "acc"
_F_HEADING = "hdg"
_F_SPEED = "spd"
_F_TRIP = "trip"

# Client clock skew beyond this is discarded outright rather than stored as an
# implausible value; the record is already useless for freshness at that point.
_MAX_CLIENT_SKEW_SECONDS = 86_400


@dataclass(frozen=True, slots=True)
class DriverLocationRecord:
    """One accepted position, as the realtime layer sees it.

    ``server_received_at`` is authoritative. ``client_timestamp`` is whatever the
    device claimed and may be ``None``; it is never used to decide freshness.
    """

    driver_id: str
    latitude: float
    longitude: float
    server_received_at: datetime
    client_timestamp: datetime | None = None
    accuracy: float | None = None
    heading: float | None = None
    speed: float | None = None
    trip_id: str | None = None

    def age_seconds(self, *, now: datetime | None = None) -> float:
        reference = now or timezone.now()
        return (reference - self.server_received_at).total_seconds()

    def freshness(self, *, max_age_seconds: int | None = None, now: datetime | None = None) -> str:
        window = int(
            max_age_seconds
            if max_age_seconds is not None
            else settings.DRIVER_LOCATION_MAX_AGE_SECONDS
        )
        if self.age_seconds(now=now) > window:
            return LocationFreshness.STALE
        return LocationFreshness.FRESH

    def is_fresh(self, *, max_age_seconds: int | None = None, now: datetime | None = None) -> bool:
        return self.freshness(max_age_seconds=max_age_seconds, now=now) == LocationFreshness.FRESH

    def as_position(self) -> DriverPosition:
        """Project onto the Phase 9 read model used by candidate search.

        The driver id is coerced back to the same type the PostGIS store yields
        (a ``UUID``) so candidate search can use one position interchangeably
        regardless of which backing store produced it.
        """
        return DriverPosition(
            driver_id=_driver_pk(self.driver_id),
            latitude=self.latitude,
            longitude=self.longitude,
            recorded_at=self.server_received_at,
        )

    def to_client_payload(self) -> dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "accuracy": self.accuracy,
            "heading": self.heading,
            "speed": self.speed,
            "recorded_at": _isoformat(self.server_received_at),
        }


class LocationStoreUnavailableError(RuntimeError):
    """Redis could not be reached.

    Raised rather than swallowed so the caller can decide: the realtime ingest
    path treats it as a dropped frame (PostgreSQL is untouched and stays
    authoritative), while dispatch treats it as "no positions available".
    """


class LocationUpdateTooFrequentError(RuntimeError):
    """The previous accepted fix is newer than the minimum accepted interval.

    Not a failure: the device is simply faster than the configured cadence. The
    caller drops the frame without writing or broadcasting, so the excess costs
    nothing. ``retry_after_ms`` tells a well-behaved client when to try again.
    """

    def __init__(self, retry_after_ms: int) -> None:
        super().__init__("Location updates are arriving faster than the accepted interval.")
        self.retry_after_ms = max(0, int(retry_after_ms))


def _key_prefix() -> str:
    return f"{cache.key_prefix or 'rova'}:rt"


def _location_key(driver_id: object) -> str:
    return f"{_key_prefix()}:driver-location:{driver_id}"


def _geo_key() -> str:
    return f"{_key_prefix()}:driver-geo"


def location_ttl_seconds() -> int:
    return int(getattr(settings, "DRIVER_LOCATION_TTL_SECONDS", 600))


def freshness_window_seconds() -> int:
    return int(getattr(settings, "DRIVER_LOCATION_MAX_AGE_SECONDS", 120))


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _client() -> Any:
    """The pooled Redis client backing the cache.

    Reused rather than dialled per message: a GPS fix must not open a TCP
    connection, and the pool is sized for exactly this (see ``CACHES``).
    """
    return cache.client.get_client(write=True)  # type: ignore[attr-defined]


class RedisDriverLocationStore:
    """Redis implementation of the Phase 9 ``DriverLocationStore`` protocol.

    Implements both the dispatch-facing protocol (``record_position`` /
    ``get_position`` / ``find_positions_within``) and the richer realtime API
    (``set_location`` / ``get_location`` / ``get_many_locations`` /
    ``delete_location`` / ``is_location_fresh``). Callers outside this module
    depend on the protocol, not on Redis.
    """

    # --- realtime API ---------------------------------------------------------

    def set_location(
        self,
        *,
        driver_id: object,
        latitude: float,
        longitude: float,
        client_timestamp: datetime | float | None = None,
        accuracy: float | None = None,
        heading: float | None = None,
        speed: float | None = None,
        trip_id: object | None = None,
        received_at: datetime | None = None,
    ) -> DriverLocationRecord:
        """Store one accepted position and return it.

        ``received_at`` exists for trusted server-side callers that already have
        an authoritative timestamp (administrative placement, replay in tests).
        The realtime ingest path never passes it, so a device clock cannot age
        its own position into freshness — or out of it.

        Passing ``received_at`` also exempts the caller from the minimum-interval
        coalescing: a caller with an authoritative timestamp is placing a driver,
        not streaming one.
        """
        stamp = received_at or timezone.now()
        driver_key = str(driver_id)
        if received_at is None:
            self._reject_if_too_soon(driver_key, stamp)
        record = DriverLocationRecord(
            driver_id=driver_key,
            latitude=float(latitude),
            longitude=float(longitude),
            server_received_at=stamp,
            client_timestamp=_normalize_client_timestamp(client_timestamp),
            accuracy=_optional_float(accuracy),
            heading=_optional_float(heading),
            speed=_optional_float(speed),
            trip_id=str(trip_id) if trip_id else None,
        )
        payload = {
            _F_DRIVER: record.driver_id,
            _F_LATITUDE: record.latitude,
            _F_LONGITUDE: record.longitude,
            _F_RECEIVED_AT: record.server_received_at.timestamp(),
        }
        # Optional fields are omitted rather than written as null: a Redis hash has
        # no null, and storing a sentinel for "absent" would make every read have
        # to distinguish it from a real value.
        optional = {
            _F_CLIENT_AT: _isoformat(record.client_timestamp),
            _F_ACCURACY: record.accuracy,
            _F_HEADING: record.heading,
            _F_SPEED: record.speed,
            _F_TRIP: record.trip_id,
        }
        payload.update({field: value for field, value in optional.items() if value is not None})
        try:
            pipe = _client().pipeline(transaction=False)
            pipe.hset(_location_key(driver_key), mapping=payload)
            pipe.geoadd(_geo_key(), (record.longitude, record.latitude, driver_key))
            pipe.expire(_location_key(driver_key), location_ttl_seconds())
            pipe.execute()
        except RedisError as exc:
            raise LocationStoreUnavailableError("Redis is unavailable.") from exc
        return record

    def get_location(self, *, driver_id: object) -> DriverLocationRecord | None:
        """Return the stored position, or ``None`` when nothing is stored."""
        raw = self._hgetall(driver_id)
        return _decode(raw) if raw else None

    def get_many_locations(
        self, *, driver_ids: Iterable[object]
    ) -> dict[str, DriverLocationRecord]:
        """Bulk read used by snapshots and operations views.

        One round trip for the whole set: an operations dashboard must not issue
        one Redis command per driver.
        """
        wanted = [str(driver_id) for driver_id in driver_ids]
        if not wanted:
            return {}
        pipe = _client().pipeline(transaction=False)
        for driver_id in wanted:
            pipe.hgetall(_location_key(driver_id))
        try:
            rows = pipe.execute()
        except RedisError as exc:
            raise LocationStoreUnavailableError("Redis is unavailable.") from exc
        results: dict[str, DriverLocationRecord] = {}
        for driver_id, row in zip(wanted, rows, strict=True):
            record = _decode(row)
            if record is not None:
                results[driver_id] = record
        return results

    def delete_location(self, *, driver_id: object) -> None:
        """Forget a driver's position (go-offline, deactivation, tests)."""
        driver_key = str(driver_id)
        try:
            pipe = _client().pipeline(transaction=False)
            pipe.delete(_location_key(driver_key))
            pipe.zrem(_geo_key(), driver_key)
            pipe.execute()
        except RedisError as exc:
            raise LocationStoreUnavailableError("Redis is unavailable.") from exc

    def is_location_fresh(self, *, driver_id: object, max_age_seconds: int | None = None) -> bool:
        record = self.get_location(driver_id=driver_id)
        return record is not None and record.is_fresh(max_age_seconds=max_age_seconds)

    def location_freshness(self, *, driver_id: object, max_age_seconds: int | None = None) -> str:
        """``FRESH`` / ``STALE`` / ``MISSING`` for one driver."""
        record = self.get_location(driver_id=driver_id)
        if record is None:
            return LocationFreshness.MISSING
        return record.freshness(max_age_seconds=max_age_seconds)

    # --- Phase 9 protocol -----------------------------------------------------

    def record_position(
        self,
        *,
        driver_id: object,
        latitude: float,
        longitude: float,
        recorded_at: datetime | None = None,
    ) -> DriverPosition:
        """Protocol entry point used by dispatch and administrative callers.

        ``recorded_at`` is a *server* timestamp supplied by the caller, so it is
        honoured as the receipt time. That is what lets a test or an operator
        place a driver, and what lets a stale position be represented as stale.
        """
        record = self.set_location(
            driver_id=driver_id,
            latitude=latitude,
            longitude=longitude,
            received_at=recorded_at,
        )
        self._maybe_persist(record)
        return record.as_position()

    def get_position(self, *, driver_id: object) -> DriverPosition | None:
        record = self.get_location(driver_id=driver_id)
        return record.as_position() if record is not None else None

    def find_positions_within(
        self, *, origin: Point, radius_meters: int, limit: int
    ) -> list[DriverPosition]:
        """Positions inside ``radius_meters`` of ``origin``, nearest first.

        The radius filter runs inside Redis against the GEO index, so the number
        of position records pulled into this process is bounded by ``limit`` and
        does not grow with the number of drivers online in the city.

        Ordering is distance first, then driver id — the same deterministic
        tie-break the PostGIS implementation uses, so a replayed search returns
        the same order.
        """
        limit = max(1, int(limit))
        try:
            rows = _client().execute_command(
                "GEOSEARCH",
                _geo_key(),
                "FROMLONLAT",
                float(origin.x),
                float(origin.y),
                "BYRADIUS",
                float(radius_meters),
                "m",
                "ASC",
                "COUNT",
                limit,
                "WITHDIST",
            )
        except RedisError as exc:
            raise LocationStoreUnavailableError("Redis is unavailable.") from exc

        distances = {_text(member): float(distance) for member, distance in (rows or [])}
        if not distances:
            return []

        records = self.get_many_locations(driver_ids=distances.keys())
        self._prune_index_members(set(distances) - set(records))

        positions = [
            DriverPosition(
                driver_id=_driver_pk(record.driver_id),
                latitude=record.latitude,
                longitude=record.longitude,
                recorded_at=record.server_received_at,
                distance_meters=int(round(distances.get(record.driver_id, 0.0))),
            )
            for record in records.values()
            if record.driver_id in distances
        ]
        positions.sort(key=lambda item: (item.distance_meters, str(item.driver_id)))
        return positions

    # --- internals ------------------------------------------------------------

    def _reject_if_too_soon(self, driver_id: str, stamp: datetime) -> None:
        """Refuse a fix that arrived inside the minimum accepted interval.

        Enforced at the store rather than in the consumer so the rule holds for
        every writer — a driver who reconnects on a new socket, or a second device
        signed into the same account, cannot multiply ROVA's write rate by simply
        opening another connection.
        """
        min_interval_ms = int(getattr(settings, "DRIVER_LOCATION_MIN_INTERVAL_MS", 0))
        if min_interval_ms <= 0:
            return
        try:
            raw_last = _client().hget(_location_key(driver_id), _F_RECEIVED_AT)
        except RedisError as exc:
            raise LocationStoreUnavailableError("Redis is unavailable.") from exc
        if raw_last is None:
            return
        if isinstance(raw_last, bytes):
            raw_last = raw_last.decode()
        try:
            last_at = float(raw_last)
        except (TypeError, ValueError):  # pragma: no cover - hand-edited hash
            return
        elapsed_ms = (stamp.timestamp() - last_at) * 1000.0
        if elapsed_ms < min_interval_ms:
            raise LocationUpdateTooFrequentError(int(min_interval_ms - elapsed_ms))

    def _hgetall(self, driver_id: object) -> dict:
        try:
            row = _client().hgetall(_location_key(driver_id))
        except RedisError as exc:
            raise LocationStoreUnavailableError("Redis is unavailable.") from exc
        if not row:
            return {}
        if isinstance(next(iter(row.values())), bytes):  # pragma: no cover - decode=True default
            row = {
                (key.decode() if isinstance(key, bytes) else key): (
                    value.decode() if isinstance(value, bytes) else value
                )
                for key, value in row.items()
            }
        return row

    def _prune_index_members(self, driver_ids: Iterable[str]) -> None:
        """Drop GEO members whose position record has already expired.

        Without this the sorted set grows without bound and radius searches keep
        paying for drivers who are long gone. Runs only for members a search
        actually returned, so the cost is bounded by ``limit``.
        """
        stale_members = [driver_id for driver_id in driver_ids if driver_id]
        if not stale_members:
            return
        try:
            pipe = _client().pipeline(transaction=False)
            for driver_id in stale_members:
                pipe.zrem(_geo_key(), driver_id)
            pipe.execute()
        except RedisError:  # pragma: no cover - best-effort housekeeping
            log_warning_event("realtime_location_index_prune_failed", members=len(stale_members))

    def _maybe_persist(self, record: DriverLocationRecord) -> None:
        """Optionally mirror the last known position into the durable row.

        Disabled by default: the telemetry path is Redis-only. When enabled, a
        position is written to PostgreSQL at most once per
        ``DRIVER_LOCATION_PERSIST_INTERVAL_SECONDS`` (and always on the driver's
        first position), so a trip produces a handful of rows rather than one per
        GPS fix. Failures here are logged and swallowed — the durable mirror is
        an optimisation, never a precondition for serving a live position.
        """
        interval = int(getattr(settings, "DRIVER_LOCATION_PERSIST_INTERVAL_SECONDS", 0))
        if interval <= 0:
            return
        try:
            from apps.dispatch.models import DriverLocation

            last = DriverLocation.objects.filter(
                driver_id=record.driver_id  # type: ignore[misc]
            ).values_list("recorded_at", flat=True)[:1]
            last_at = last[0] if last else None
            if last_at is not None:
                if (timezone.now() - last_at) < timedelta(seconds=interval):
                    return
            DriverLocation.objects.update_or_create(
                driver_id=record.driver_id,
                defaults={
                    "point": Point(record.longitude, record.latitude, srid=4326),
                    "recorded_at": record.server_received_at,
                    "source": "REDIS",
                },
            )
        except Exception:
            log_warning_event("realtime_location_persist_failed", driver=record.driver_id)


def _text(value: Any) -> str:
    """Decode a Redis reply to text.

    ``decode_responses`` is not guaranteed by the cache backend — django-redis
    hands back bytes by default — and ``str(b'abc')`` yields ``"b'abc'"``, which
    silently becomes a different driver id rather than an error.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _driver_pk(value: str) -> Any:
    """Redis stores the driver id as a string; the domain uses ``UUID``.

    Dispatch compares positions against ``DriverProfile.pk`` and against the ids
    recorded on dispatch attempts, so the projection has to carry the same type
    the PostGIS store would have produced.
    """
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return value


def _normalize_client_timestamp(value: datetime | float | None) -> datetime | None:
    """Accept a device clock only if it is not wildly implausible.

    The value is diagnostic. An absurd one (a device set to 1970, or to a year in
    the future) is worse than no value at all, because it would be reported to
    operators as if it meant something. Mobile clients send epoch seconds; a
    caller that already holds a ``datetime`` is passed through.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value if timezone.is_aware(value) else value.replace(tzinfo=UTC)
    else:
        try:
            moment = datetime.fromtimestamp(float(value), tz=UTC)
        except (TypeError, ValueError, OverflowError, OSError):
            return None
    delta = (timezone.now() - moment).total_seconds()
    if abs(delta) > _MAX_CLIENT_SKEW_SECONDS:
        return None
    return moment


def _decode(raw: dict) -> DriverLocationRecord | None:
    """Rebuild a record from a Redis hash, or ``None`` when unusable.

    A partially written or hand-edited hash yields ``None`` rather than a record
    with invented fields. Keys are normalised here rather than at each call site:
    django-redis replies with ``bytes`` by default, and a lookup of ``"rat"`` in a
    dict keyed ``b"rat"`` misses silently — which would turn every stored position
    into "no position" instead of into an error.
    """
    if not raw:
        return None
    fields = {_text(key): value for key, value in raw.items()}
    try:
        received_at = float(fields.get(_F_RECEIVED_AT) or 0)
        if received_at <= 0:
            return None
        client_at = fields.get(_F_CLIENT_AT)
        return DriverLocationRecord(
            driver_id=_text(fields.get(_F_DRIVER) or ""),
            latitude=float(fields[_F_LATITUDE]),
            longitude=float(fields[_F_LONGITUDE]),
            server_received_at=datetime.fromtimestamp(received_at, tz=UTC),
            client_timestamp=_parse_client_timestamp(client_at),
            accuracy=_optional_float(fields.get(_F_ACCURACY)),
            heading=_optional_float(fields.get(_F_HEADING)),
            speed=_optional_float(fields.get(_F_SPEED)),
            trip_id=(_text(fields[_F_TRIP]) if fields.get(_F_TRIP) else None),
        )
    except (TypeError, ValueError, KeyError, OverflowError, OSError):
        return None


def _parse_client_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if timezone.is_aware(parsed) else parsed.replace(tzinfo=UTC)


def resolve_location_store():
    """Return the configured ``DriverLocationStore`` implementation.

    One abstraction, two backings. ``redis`` (default) is the ephemeral stream
    used by dispatch and live tracking alike; ``postgres`` is the durable
    PostGIS fallback kept for environments without Redis and for debugging.
    """
    from apps.dispatch.services.location_service import (
        InMemoryDriverLocationStore,
        PostgresDriverLocationStore,
    )

    choice = str(getattr(settings, "DRIVER_LOCATION_STORE", "redis")).strip().lower()
    if choice == "postgres":
        return PostgresDriverLocationStore()
    if choice == "memory":
        return InMemoryDriverLocationStore()
    return RedisDriverLocationStore()


__all__ = [
    "DriverLocationRecord",
    "LocationStoreUnavailableError",
    "LocationUpdateTooFrequentError",
    "RedisDriverLocationStore",
    "freshness_window_seconds",
    "location_ttl_seconds",
    "resolve_location_store",
]
