"""Candidate search and ranking — who may be offered this ride, and in what order.

Dispatch never matches on price, rating, or ETA. It matches on three rules, in
this order, and nothing else:

1. **Proximity.** A driver is only considered if their position is inside the
   current search radius and fresh enough to trust.
2. **Fairness.** Among equally close drivers, the one who has been idle longest
   (``DriverProfile.available_since``) is offered first. This is what stops the
   nearest driver from hoarding every ride on a busy street.
3. **Stability.** Ties are broken by driver id so replaying a search for the same
   pickup always produces the same order.

The radius escalates outward only while nothing offerable has been found, so a
quiet night does not ping drivers across the whole city while plenty of drivers
are sitting right there.

Vehicle/driver eligibility is delegated to the owning domains
(``driver_service.can_go_online`` / ``vehicle_service.is_vehicle_eligible``) —
dispatch never re-implements those rules.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from django.conf import settings
from django.contrib.gis.geos import Point

from apps.drivers.models import DriverProfile
from apps.drivers.services.driver_service import can_go_online
from apps.mobility.models import Trip
from apps.vehicles.models import Vehicle
from apps.vehicles.services.vehicle_service import is_vehicle_eligible

from .. import choices
from ..models import DispatchAttempt, DriverOffer
from .location_service import DriverPosition, get_location_store
from .ranking_service import candidate_sort_key, rank_candidates

# A driver holding any of these is on a live ride and must not be offered another.
ACTIVE_TRIP_STATUSES = frozenset(
    {
        "DRIVER_ASSIGNED",
        "DRIVER_ACCEPTED",
        "DRIVER_ARRIVING",
        "DRIVER_AT_PICKUP",
        "RIDER_ONBOARD",
        "TRIP_STARTED",
        "PAYMENT_PENDING",
        "PAYMENT_COMPLETED",
    }
)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One offerable driver for one ride request."""

    driver: DriverProfile
    vehicle: Vehicle
    position: DriverPosition

    @property
    def driver_id(self) -> object:
        return self.driver.pk

    @property
    def distance_meters(self) -> int:
        return self.position.distance_meters

    @property
    def available_since(self):
        return self.driver.available_since

    @property
    def sort_key(self) -> tuple:
        return candidate_sort_key(self)


@dataclass(slots=True)
class CandidateSearch:
    """Outcome of one candidate search, including why it came back empty."""

    candidates: list[Candidate] = field(default_factory=list)
    radius_meters: int = 0
    positions_examined: int = 0
    positions_stale: int = 0
    drivers_ineligible: int = 0
    drivers_unavailable: int = 0
    drivers_previously_offered: int = 0
    radii_searched: list[int] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.candidates


def search_radii() -> list[int]:
    """Configured radii, ascending, with the largest as the effective ceiling."""
    radii = sorted({int(radius) for radius in settings.DISPATCH_SEARCH_RADII_METERS if radius > 0})
    return radii or [10_000]


def origin_point_for(ride_request) -> Point:
    return Point(
        float(ride_request.origin_longitude),
        float(ride_request.origin_latitude),
        srid=4326,
    )


def _active_driver_ids(driver_ids: Iterable[object]) -> set:
    if not driver_ids:
        return set()
    return set(
        Trip.objects.filter(
            driver_id__in=list(driver_ids),
            status__in=ACTIVE_TRIP_STATUSES,
        ).values_list("driver_id", flat=True)
    )


def _previously_attempted_driver_ids(ride_request_id: object) -> set:
    return set(
        DispatchAttempt.objects.filter(
            ride_request_id=ride_request_id,  # type: ignore[misc]
            driver__isnull=False,
        ).values_list("driver_id", flat=True)
    )


def _drivers_with_pending_offers(driver_ids: Iterable[object]) -> set:
    if not driver_ids:
        return set()
    return set(
        DriverOffer.objects.filter(
            driver_id__in=list(driver_ids),
            status=choices.DriverOfferStatus.PENDING,
        ).values_list("driver_id", flat=True)
    )


def find_candidates(
    *,
    ride_request,
    exclude_driver_ids: Iterable[object] = (),
    limit: int | None = None,
) -> CandidateSearch:
    """Return the ranked, offerable drivers for ``ride_request``."""
    search = CandidateSearch()
    limit = int(limit or settings.DISPATCH_MAX_CANDIDATES)
    max_age_seconds = int(settings.DISPATCH_DRIVER_LOCATION_MAX_AGE_SECONDS)
    store = get_location_store()
    origin = origin_point_for(ride_request)
    excluded = set(exclude_driver_ids)

    for radius in search_radii():
        search.radii_searched.append(radius)
        positions = store.find_positions_within(origin=origin, radius_meters=radius, limit=limit)
        search.positions_examined = len(positions)
        search.radius_meters = radius
        if not positions:
            continue

        fresh_positions: dict[object, DriverPosition] = {}
        for position in positions:
            if position.is_stale(max_age_seconds):
                search.positions_stale += 1
                continue
            if position.driver_id in excluded:
                search.drivers_previously_offered += 1
                continue
            fresh_positions[position.driver_id] = position

        if not fresh_positions:
            continue

        drivers = (
            DriverProfile.objects.filter(pk__in=list(fresh_positions))
            .select_related("active_vehicle", "user")
            .prefetch_related("active_vehicle__documents")
        )

        previously_offered = _previously_attempted_driver_ids(ride_request.pk)
        pending_offer = _drivers_with_pending_offers(fresh_positions)
        busy = _active_driver_ids(fresh_positions)

        candidates: list[Candidate] = []
        for driver in drivers:
            if driver.pk in excluded or driver.pk in previously_offered:
                search.drivers_previously_offered += 1
                continue
            if driver.pk in pending_offer or driver.pk in busy:
                search.drivers_unavailable += 1
                continue
            if driver.availability_status != "ONLINE":
                search.drivers_unavailable += 1
                continue
            eligible, _reason = can_go_online(driver)
            vehicle = driver.active_vehicle
            if not eligible or vehicle is None:
                search.drivers_ineligible += 1
                continue
            if vehicle.category_id != ride_request.vehicle_category_id:
                search.drivers_ineligible += 1
                continue
            vehicle_ok, _vehicle_reason = is_vehicle_eligible(driver, vehicle)
            if not vehicle_ok:
                search.drivers_ineligible += 1
                continue
            candidates.append(
                Candidate(
                    driver=driver,
                    vehicle=vehicle,
                    position=fresh_positions[driver.pk],
                )
            )

        if candidates:
            candidates = rank_candidates(candidates)
            search.candidates = candidates[:limit]
            return search

    return search
