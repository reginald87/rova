"""Driver location storage — the read side of candidate search.

Dispatch needs to answer "where is every eligible driver right now". Since Phase
10 that question is answered from Redis, because the positions themselves now
arrive over a WebSocket at a rate that has no business touching PostgreSQL.
Everything above this module talks to :class:`DriverLocationStore`, so the backing
implementation is a deployment choice (``DRIVER_LOCATION_STORE``) rather than a
call-site decision — realtime tracking and dispatch cannot drift apart because
there is only ever one implementation in play.

Design notes
------------
* **Freshness is a filter, not a warning.** A stale position counts as no
  position: dispatch would rather skip a driver than offer a ride based on where
  they were half an hour ago.
* **The spatial predicate is index-friendly.** ``ST_DWithin`` is emitted (rather
  than ``ST_Distance(...) <= r``) so PostgreSQL can satisfy the search from the
  GiST index on ``point``. The Redis implementation uses ``GEOSEARCH`` for the
  same reason.
* **Radius widening is a separate query per configured radius** rather than one
  unbounded search: candidate ranking needs an ordered set anyway, and bounding
  each query keeps the SQL predictable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from django.contrib.gis.db.models.functions import Distance
from django.contrib.gis.geos import Point
from django.db.models import BooleanField, FloatField, Value
from django.utils import timezone

from ..models import DriverLocation
from .errors import StaleLocationError


class DWithin(Distance):
    """``ST_DWithin(geography, geography, meters)``.

    Subclassing :class:`Distance` reuses Django's geography casting, so the
    origin argument is wrapped as geography exactly as it is for ``ST_Distance``.
    ``output_field`` is overridden to a boolean because ``ST_DWithin`` is a
    predicate; Django then emits the bare ``ST_DWithin(...)`` expression in the
    ``WHERE`` clause, which is what lets the planner match it against the GiST
    index.
    """

    function = "ST_DWithin"
    # Deliberately a boolean where `DistanceResultMixin` declares a distance
    # field: this is a predicate, not a measurement. The docstring above explains
    # why the narrower type is what makes Django emit a bare `ST_DWithin(...)`.
    output_field = BooleanField()  # type: ignore[assignment, var-annotated]

    def __init__(self, expression, point, radius_meters: float, **kwargs):
        super().__init__(expression, point, **kwargs)
        self.source_expressions.append(Value(radius_meters, output_field=FloatField()))


@dataclass(frozen=True, slots=True)
class DriverPosition:
    """A driver's last known position and how far it is from the search origin."""

    driver_id: object
    latitude: float
    longitude: float
    recorded_at: datetime
    distance_meters: int = 0

    @property
    def age_seconds(self) -> float:
        return (timezone.now() - self.recorded_at).total_seconds()

    def is_stale(self, max_age_seconds: int) -> bool:
        return self.age_seconds > max_age_seconds

    def require_fresh(self, max_age_seconds: int) -> None:
        if self.is_stale(max_age_seconds):
            raise StaleLocationError(
                f"Driver location is {int(self.age_seconds)}s old (maximum {max_age_seconds}s)."
            )


@runtime_checkable
class DriverLocationStore(Protocol):
    """Read/write access to driver positions."""

    def record_position(
        self,
        *,
        driver_id: object,
        latitude: float,
        longitude: float,
        recorded_at: datetime | None = None,
    ) -> DriverLocation: ...

    def get_position(self, *, driver_id: object) -> DriverPosition | None: ...

    def find_positions_within(
        self, *, origin: Point, radius_meters: int, limit: int
    ) -> list[DriverPosition]: ...


class PostgresDriverLocationStore:
    """PostGIS-backed store: one upserted row per driver, GiST indexed."""

    def record_position(
        self,
        *,
        driver_id: object,
        latitude: float,
        longitude: float,
        recorded_at: datetime | None = None,
    ) -> DriverLocation:
        location, _ = DriverLocation.objects.update_or_create(
            driver_id=driver_id,
            defaults={
                "point": Point(float(longitude), float(latitude), srid=4326),
                "recorded_at": recorded_at or timezone.now(),
            },
        )
        return location

    def find_positions_within(
        self, *, origin: Point, radius_meters: int, limit: int
    ) -> list[DriverPosition]:
        """Return up to ``limit`` positions within ``radius_meters``, nearest first.

        The ``(distance_meters, driver_id)`` ordering is intentional: distance is
        the business rule and ``driver_id`` is only a stable tie-breaker, so
        repeating a search for the same pickup yields the same order.
        """
        rows = (
            DriverLocation.objects.annotate(
                within_radius=DWithin("point", origin, float(radius_meters)),
                distance_meters=Distance("point", origin),
            )
            .filter(within_radius=True)
            .order_by("distance_meters", "driver_id")
            .only("driver", "point", "recorded_at")[:limit]
        )
        return [
            DriverPosition(
                driver_id=row.driver_id,
                latitude=float(row.point.y),
                longitude=float(row.point.x),
                recorded_at=row.recorded_at,
                distance_meters=int(_meters(row.distance_meters)),
            )
            for row in rows
        ]

    def get_position(self, *, driver_id: object) -> DriverPosition | None:
        row = (
            DriverLocation.objects.filter(driver_id=driver_id)  # type: ignore[misc]
            .only("driver", "point", "recorded_at")[:1]
            .first()
        )
        if row is None:
            return None
        return DriverPosition(
            driver_id=row.driver_id,
            latitude=float(row.point.y),
            longitude=float(row.point.x),
            recorded_at=row.recorded_at,
        )


class InMemoryDriverLocationStore:
    """In-process store with the same semantics, for unit tests that do not need PostGIS.

    Freshness uses a fixed generous window rather than settings, because a unit
    test of ranking should not need a settings override to place a driver.
    """

    MAX_AGE_SECONDS = 86_400

    def __init__(self) -> None:
        self._positions: dict[object, DriverPosition] = {}

    def record_position(
        self,
        *,
        driver_id: object,
        latitude: float,
        longitude: float,
        recorded_at: datetime | None = None,
    ) -> DriverPosition:
        position = DriverPosition(
            driver_id=driver_id,
            latitude=float(latitude),
            longitude=float(longitude),
            recorded_at=recorded_at or timezone.now(),
        )
        self._positions[driver_id] = position
        return position

    def get_position(self, *, driver_id: object) -> DriverPosition | None:
        return self._positions.get(driver_id)

    def find_positions_within(
        self, *, origin: Point, radius_meters: int, limit: int
    ) -> list[DriverPosition]:
        results = []
        for position in self._positions.values():
            if position.is_stale(self.MAX_AGE_SECONDS):
                continue
            distance_meters = _great_circle_meters(origin, position)
            if distance_meters <= radius_meters:
                results.append(
                    DriverPosition(
                        driver_id=position.driver_id,
                        latitude=position.latitude,
                        longitude=position.longitude,
                        recorded_at=position.recorded_at,
                        distance_meters=distance_meters,
                    )
                )
        results.sort(key=lambda item: (item.distance_meters, str(item.driver_id)))
        return results[:limit]


def _great_circle_meters(origin: Point, position: DriverPosition) -> int:
    from apps.geography.services.geography_service import calculate_distance

    return int(
        round(calculate_distance(origin, Point(position.longitude, position.latitude, srid=4326)))
    )


def _meters(value) -> float:
    """Coerce a spatial distance result to a float number of meters.

    Django's GIS backend wraps ``ST_Distance`` output in a
    ``django.contrib.gis.measure.Distance``, which exposes the meter value as
    ``.m``; a plain float is accepted too so the helper is safe to reuse.
    """
    if value is None:
        return 0.0
    return float(getattr(value, "m", value))


_store: DriverLocationStore | None = None


def get_location_store() -> DriverLocationStore:
    """Return the process-wide location store.

    The single indirection that keeps exactly one source of truth for "where is
    this driver now". Phase 10 backed it with a Redis implementation chosen by
    ``DRIVER_LOCATION_STORE``; the PostgreSQL implementation remains the durable
    fallback for environments without Redis.
    """
    global _store
    if _store is None:
        from apps.realtime.services.location_store import resolve_location_store

        _store = resolve_location_store()
    return _store


def set_location_store(store: DriverLocationStore | None) -> None:
    """Replace the location store. Tests only; ``None`` restores the default."""
    global _store
    _store = store


def freshness_cutoff(max_age_seconds: int) -> datetime:
    """Oldest ``recorded_at`` still considered current."""
    return timezone.now() - timedelta(seconds=max_age_seconds)
