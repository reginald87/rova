"""The dispatch pass — one unit of "who gets this ride next".

A pass is deliberately narrow: it either extends exactly one offer, or it records
why it could not. It never loops over candidates handing out offers, because
sequential dispatch means one live offer per trip; the next pass happens after a
decline, an expiry, or a retry.

Dispatch owns no trip state of its own. It calls the mobility service to move a
request into ``SEARCHING`` and, on the far side of an accepted offer, to assign
the driver. Everything it writes itself lives in the dispatch history.

Failing to find a driver is not a failure. A pass that finds nothing records a
``FAILED`` attempt with a structured reason and asks for a retry until
``DISPATCH_MAX_ATTEMPTS`` is reached, after which mobility projects the request
to ``NO_DRIVER_FOUND``. That ordering matters operationally: a driver who comes
online twenty seconds late should still catch the ride.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from apps.mobility.models import RideRequest, Trip
from apps.mobility.services import mobility_service

from .. import choices
from ..models import DispatchAttempt, DriverOffer
from .candidate_service import CandidateSearch, find_candidates
from .errors import InvalidDispatchStateError, RideRequestNotFoundError
from .offer_service import create_offer, expire_offer

# Trip states that dispatch is allowed to act on. Anything else (assigned,
# cancelled, completed, expired) means the ride is out of dispatch's hands.
DISPATCHABLE_TRIP_STATUSES = frozenset({"REQUESTED", "SEARCHING"})


@dataclass(slots=True)
class DispatchOutcome:
    """What one dispatch pass did."""

    ride_request: RideRequest
    trip: Trip
    offer: DriverOffer | None = None
    attempt: DispatchAttempt | None = None
    candidate_count: int = 0
    radius_meters: int = 0
    reason: str = ""
    exhausted: bool = False
    skipped: bool = False

    @property
    def offered(self) -> bool:
        return self.offer is not None

    @property
    def should_retry(self) -> bool:
        return self.reason != "" and not self.exhausted and not self.offered and not self.skipped

    @property
    def attempt_count(self) -> int:
        if self.attempt:
            return int(self.attempt.attempt_number)
        return 0


def _coerce_uuid(value):
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise RideRequestNotFoundError("Ride request not found.") from exc


def _attempted_driver_ids(ride_request_id) -> set:
    return set(
        DispatchAttempt.objects.filter(
            ride_request_id=ride_request_id, driver__isnull=False
        ).values_list("driver_id", flat=True)
    )


def _attempt_count(ride_request_id) -> int:
    return DispatchAttempt.objects.filter(ride_request_id=ride_request_id).count()


def _next_attempt_number(ride_request_id) -> int:
    return _attempt_count(ride_request_id) + 1


def _load_trip(ride_request_id) -> tuple[RideRequest, Trip]:
    request = (
        RideRequest.objects.select_related(
            "vehicle_category", "rider__user", "origin_city", "origin_service_zone"
        )
        .filter(pk=ride_request_id)
        .first()
    )
    if request is None:
        raise RideRequestNotFoundError("Ride request not found.")
    trip = (
        Trip.objects.select_related("ride_request", "vehicle_category")
        .filter(ride_request_id=request.pk)
        .first()
    )
    if trip is None:
        raise InvalidDispatchStateError("This ride request has no trip record.")
    return request, trip


def _classify_empty_search(search: CandidateSearch) -> tuple[str, str]:
    """Name the reason a search came back empty.

    Ordered from most specific to least: a data-quality problem and a coverage
    problem need different fixes, and an operator reading the dispatch history
    should not have to guess which one happened.
    """
    if search.positions_examined == 0:
        return (
            choices.NoDriverReason.NO_DRIVERS_WITHIN_RADIUS,
            f"No driver positions within {search.radius_meters}m "
            f"(radii searched: {', '.join(str(r) for r in search.radii_searched)}m).",
        )
    if search.positions_stale >= search.positions_examined:
        return (
            choices.NoDriverReason.LOCATION_DATA_STALE,
            f"All {search.positions_examined} nearby positions are older than "
            f"{settings.DISPATCH_DRIVER_LOCATION_MAX_AGE_SECONDS}s.",
        )
    if (
        search.drivers_previously_offered > 0
        and search.drivers_ineligible == 0
        and search.drivers_unavailable == 0
    ):
        return (
            choices.NoDriverReason.ALL_CANDIDATES_EXHAUSTED,
            f"{search.drivers_previously_offered} nearby driver(s) already had this ride.",
        )
    return (
        choices.NoDriverReason.NO_ELIGIBLE_DRIVERS,
        f"{search.positions_examined} position(s) examined: "
        f"{search.drivers_ineligible} ineligible, "
        f"{search.drivers_unavailable} unavailable, "
        f"{search.drivers_previously_offered} already attempted.",
    )


def dispatch_ride(
    *,
    ride_request_id: str,
    trigger: str = choices.DispatchTrigger.RIDE_REQUEST,
) -> DispatchOutcome:
    """Run one dispatch pass for a ride request.

    Idempotent per trip: if the ride already has a live offer, or has already
    been assigned or closed, the pass is a no-op. Two workers racing on the same
    request therefore produce one offer, not two.
    """
    request_id = _coerce_uuid(ride_request_id)
    request, trip = _load_trip(request_id)

    if trip.driver_id is not None or trip.status not in DISPATCHABLE_TRIP_STATUSES:
        return DispatchOutcome(ride_request=request, trip=trip, skipped=True)

    if DriverOffer.objects.filter(trip=trip, status=choices.DriverOfferStatus.PENDING).exists():
        return DispatchOutcome(ride_request=request, trip=trip, skipped=True)

    excluded = _attempted_driver_ids(request.pk)
    search = find_candidates(ride_request=request, exclude_driver_ids=excluded)

    if search.candidates:
        return _extend_offer(
            request=request,
            trip=trip,
            search=search,
            trigger=trigger,
            already_offered=excluded,
        )
    return _record_empty_search(request=request, trip=trip, search=search, trigger=trigger)


def _extend_offer(
    *,
    request: RideRequest,
    trip: Trip,
    search: CandidateSearch,
    trigger: str,
    already_offered: set,
) -> DispatchOutcome:
    candidate = search.candidates[0]
    with transaction.atomic():
        locked = Trip.objects.select_for_update(of=("self",)).get(pk=trip.pk)
        if locked.driver_id is not None or locked.status not in DISPATCHABLE_TRIP_STATUSES:
            return DispatchOutcome(ride_request=request, trip=locked, skipped=True)
        if DriverOffer.objects.filter(
            trip=locked, status=choices.DriverOfferStatus.PENDING
        ).exists():
            return DispatchOutcome(ride_request=request, trip=locked, skipped=True)

        if locked.status == "REQUESTED":
            mobility_service.begin_search(ride_request_id=str(request.pk))

        offer, attempt = create_offer(
            ride_request=request,
            trip=locked,
            candidate=candidate,
            attempt_number=_next_attempt_number(request.pk),
            trigger=trigger,
            radius_meters=search.radius_meters,
            candidate_rank=1,
        )
        attempt.metadata = {
            **(attempt.metadata or {}),
            "already_offered_count": len(already_offered),
            "candidate_count": len(search.candidates),
        }
        attempt.save(update_fields=["metadata", "updated_at"])

    return DispatchOutcome(
        ride_request=request,
        trip=trip,
        offer=offer,
        attempt=attempt,
        candidate_count=len(search.candidates),
        radius_meters=search.radius_meters,
    )


def _record_empty_search(
    *, request: RideRequest, trip: Trip, search: CandidateSearch, trigger: str
) -> DispatchOutcome:
    reason, detail = _classify_empty_search(search)
    with transaction.atomic():
        locked = Trip.objects.select_for_update(of=("self",)).get(pk=trip.pk)
        if locked.driver_id is not None or locked.status not in DISPATCHABLE_TRIP_STATUSES:
            return DispatchOutcome(ride_request=request, trip=locked, skipped=True)

        attempt = DispatchAttempt.objects.create(
            ride_request=request,
            trip=locked,
            driver=None,
            vehicle=None,
            attempt_number=_next_attempt_number(request.pk),
            status=choices.DispatchAttemptStatus.FAILED,
            search_radius_meters=search.radius_meters,
            reason=reason,
            detail=detail[:255],
            offered_at=None,
            metadata={
                "trigger": trigger,
                "positions_examined": search.positions_examined,
                "positions_stale": search.positions_stale,
                "drivers_ineligible": search.drivers_ineligible,
                "drivers_unavailable": search.drivers_unavailable,
                "drivers_previously_offered": search.drivers_previously_offered,
                "radii_searched_meters": search.radii_searched,
            },
        )
        exhausted = attempt.attempt_number >= int(settings.DISPATCH_MAX_ATTEMPTS)
        if exhausted:
            attempt.detail = f"{detail} Attempts exhausted.".strip()[:255]
            attempt.save(update_fields=["detail", "updated_at"])
            mobility_service.mark_no_driver_found(ride_request_id=str(request.pk))

    return DispatchOutcome(
        ride_request=request,
        trip=trip,
        attempt=attempt,
        candidate_count=0,
        radius_meters=search.radius_meters,
        reason=reason,
        exhausted=exhausted,
    )


def reassign_after_response(*, offer: DriverOffer, trigger: str) -> DispatchOutcome | None:
    """Move a declined or expired offer onward to the next candidate.

    The driver who just declined (or let the offer lapse) is excluded from the
    search, and so is everyone already attempted for this ride, so reassignment
    walks deterministically down the ranked list instead of retrying the same
    driver.
    """
    now = timezone.now()
    if offer.status == choices.DriverOfferStatus.PENDING and offer.expires_at <= now:
        offer = expire_offer(offer=offer, now=now)

    if offer.status not in (
        choices.DriverOfferStatus.DECLINED,
        choices.DriverOfferStatus.EXPIRED,
    ):
        return None

    return dispatch_ride(ride_request_id=str(offer.ride_request_id), trigger=trigger)


def enqueue_pass(*, ride_request_id: str, trigger: str) -> DispatchOutcome | None:
    """Run a dispatch pass on a worker, or inline when dispatch is synchronous.

    Both paths go through the same ``dispatch_ride`` function, so ``DISPATCH_ASYNC``
    changes only *where* the work runs, never what it does.
    """
    if bool(getattr(settings, "DISPATCH_ASYNC", True)):
        from ..tasks import dispatch_ride_task

        dispatch_ride_task.apply_async(
            kwargs={"ride_request_id": str(ride_request_id), "trigger": trigger}
        )
        return None
    return dispatch_ride(ride_request_id=ride_request_id, trigger=trigger)


def reassign_after_decline(*, offer: DriverOffer) -> DispatchOutcome | None:
    """Move a declined offer to the next candidate.

    Wrapped so the API can hand the work to a worker instead of making the driver
    wait on another driver's eligibility check.
    """
    return enqueue_pass(
        ride_request_id=str(offer.ride_request_id), trigger=choices.DispatchTrigger.REASSIGNMENT
    )


def expire_offers_and_reassign(*, now=None, limit: int = 100) -> list[DispatchOutcome]:
    """Sweep expired offers and re-dispatch the rides they left open.

    Returns one outcome per ride that was re-dispatched, for logging. Runs on a
    Celery beat schedule; the trip row lock inside each pass is what keeps this
    sweep from double-offering when two sweepers overlap.
    """
    now = now or timezone.now()
    expired_ids = list(
        DriverOffer.objects.filter(status=choices.DriverOfferStatus.PENDING, expires_at__lte=now)
        .order_by("expires_at")
        .values_list("id", flat=True)[:limit]
    )
    outcomes: list[DispatchOutcome] = []
    for offer_id in expired_ids:
        offer = DriverOffer.objects.select_related("trip", "ride_request").get(pk=offer_id)
        expired = expire_offer(offer=offer, now=now)
        outcome = reassign_after_response(offer=expired, trigger=choices.DispatchTrigger.EXPIRATION)
        if outcome is not None:
            outcomes.append(outcome)
    return outcomes


def dispatch_history(*, ride_request_id: str) -> list[DispatchAttempt]:
    """Full attempt history for a ride request, oldest first."""
    request_id = _coerce_uuid(ride_request_id)
    return list(
        DispatchAttempt.objects.select_related("driver__user", "vehicle__category")
        .filter(ride_request_id=request_id)
        .order_by("attempt_number")
    )
