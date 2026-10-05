"""Driver offers — creation, listing, and the accept/decline responses.

An offer is a promise with a deadline: "this ride, that driver, this many
seconds". Three database guarantees make that promise safe under concurrency, and
they do the work that application-level checks cannot:

* at most one ``PENDING`` offer per driver,
* at most one ``PENDING`` offer per trip,
* at most one dispatch attempt per (ride request, driver).

Response handling therefore never asks "is this offer still pending?" before
writing — it locks the offer row and lets the unique constraints and the
conditional update settle any race. Two accepts for the same ride request cannot
both win, and a driver cannot be accepting one ride while another offer arrives.

Every outcome is also announced on the driver's realtime socket. A driver who is
holding an offer sheet has no reason to poll for its fate: the offer was created,
taken by somebody else, expired, or cancelled with the ride, and in each case the
app must stop showing it. Those publications are deferred to ``on_commit`` so an
outcome that was rolled back is never announced.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

from django.conf import settings
from django.db import IntegrityError, transaction
from django.utils import timezone

from apps.drivers.models import DriverProfile
from apps.drivers.services.driver_service import can_go_online
from apps.mobility.services import mobility_service
from apps.realtime.services.events import publish_driver_offer
from apps.vehicles.models import Vehicle

from .. import choices
from ..models import DispatchAttempt, DriverOffer
from .errors import (
    DispatchError,
    DriverNotEligibleError,
    OfferAlreadyRespondedError,
    OfferExpiredError,
    OfferNotFoundError,
    OfferNotOwnedError,
    RideAlreadyAssignedError,
)


def _coerce_uuid(value):
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise OfferNotFoundError("Offer not found.") from exc


def get_offer(*, driver: DriverProfile, offer_id: str) -> DriverOffer:
    """Fetch one of ``driver``'s offers.

    Ownership is checked by the lookup itself, so a driver probing another
    driver's offer id gets the same 404 as a non-existent id.
    """
    pk = _coerce_uuid(offer_id)
    offer = (
        DriverOffer.objects.select_related(
            "ride_request", "trip", "vehicle__category", "driver__user"
        )
        .filter(pk=pk, driver=driver)
        .first()
    )
    if offer is None:
        raise OfferNotFoundError("Offer not found.")
    return offer


def list_pending_offers(*, driver: DriverProfile) -> list[DriverOffer]:
    """The driver's open offers, newest first."""
    return list(
        DriverOffer.objects.select_related("ride_request", "trip", "vehicle__category")
        .filter(driver=driver, status=choices.DriverOfferStatus.PENDING)
        .order_by("-created_at")
    )


def create_offer(
    *,
    ride_request,
    trip,
    candidate,
    attempt_number: int,
    trigger: str,
    radius_meters: int,
    candidate_rank: int,
) -> tuple[DriverOffer, DispatchAttempt]:
    """Record one dispatch attempt and hand the ride to exactly one driver.

    Returns ``(offer, attempt)``. Both rows are written in one transaction, so an
    attempt never exists without the offer it produced (or vice versa).
    """
    now = timezone.now()
    expires_at = now + timedelta(seconds=int(settings.DISPATCH_OFFER_TIMEOUT_SECONDS))

    try:
        with transaction.atomic():
            attempt = DispatchAttempt.objects.create(
                ride_request=ride_request,
                trip=trip,
                driver=candidate.driver,
                vehicle=candidate.vehicle,
                attempt_number=attempt_number,
                status=choices.DispatchAttemptStatus.OFFERED,
                search_radius_meters=radius_meters,
                candidate_distance_meters=candidate.distance_meters,
                candidate_rank=candidate_rank,
                offered_at=now,
                metadata={
                    "trigger": trigger,
                    "available_since": _isoformat(candidate.available_since),
                },
            )
            offer = DriverOffer.objects.create(
                ride_request=ride_request,
                trip=trip,
                driver=candidate.driver,
                vehicle=candidate.vehicle,
                dispatch_attempt=attempt,
                status=choices.DriverOfferStatus.PENDING,
                distance_to_pickup_meters=candidate.distance_meters,
                search_radius_meters=radius_meters,
                expires_at=expires_at,
                metadata={"candidate_rank": candidate_rank},
            )
    except IntegrityError as exc:
        # A partial unique constraint fired: either the driver already has an
        # open offer or this trip is already on offer. Both mean "someone else
        # got there first", which is a normal outcome of concurrent dispatch, not
        # a server error.
        raise DispatchError("This ride or driver already has an open offer.") from exc

    # Notify the driver on their own socket once the offer is durably recorded.
    # Deferred to ``on_commit``: an offer that was rolled back must never appear
    # in a driver's app, and a driver who never saw it can simply be offered again.
    publish_driver_offer(offer, outcome="created")
    return offer, attempt


def _isoformat(value) -> str | None:
    return value.isoformat() if value is not None else None


def _lock_pending_offer(*, offer_id: str) -> DriverOffer:
    """Lock the offer row and re-check that it is still actionable.

    Lock first, then decide: the lock is what makes a concurrent
    accept/decline/expire serialize instead of both passing a pre-check.
    """
    pk = _coerce_uuid(offer_id)
    offer = DriverOffer.objects.select_for_update(of=("self",)).filter(pk=pk).first()
    if offer is None:
        raise OfferNotFoundError("Offer not found.")
    return offer


def _settle_attempt(offer: DriverOffer, *, status: str, reason: str = "", detail: str = "") -> None:
    attempt = offer.dispatch_attempt
    attempt.status = status
    attempt.responded_at = timezone.now()
    if reason:
        attempt.reason = reason
    if detail:
        attempt.detail = detail
    attempt.save(update_fields=["status", "responded_at", "reason", "detail", "updated_at"])


def accept_offer(*, driver: DriverProfile, offer_id: str) -> DriverOffer:
    """Accept an offer and assign the driver to the trip.

    The offer row is locked first, then the trip, then the driver — the same
    order the mobility service uses internally, so a dispatch-driven assignment
    and a lifecycle command cannot deadlock against each other. Mobility remains
    the single authority on assignment: if it refuses, no offer is consumed and
    the driver is offered onward.
    """
    now = timezone.now()
    with transaction.atomic():
        # Lock order is offer -> trip -> driver. Only the offer is locked
        # here; mobility_service owns the trip and driver locks and must not
        # be preceded by a driver lock, or an offer response could deadlock
        # against a lifecycle command that locks trip then driver.
        offer = _lock_pending_offer(offer_id=offer_id)
        if offer.driver_id != driver.pk:
            raise OfferNotOwnedError("This offer does not belong to the authenticated driver.")
        if offer.status != choices.DriverOfferStatus.PENDING:
            raise OfferAlreadyRespondedError("This offer has already been responded to.")
        if offer.expires_at <= now:
            raise OfferExpiredError("This offer has expired.")

        # Advisory only: mobility re-validates everything under its own locks,
        # and it is the authority on whether this assignment happens.
        eligible, reason = _driver_still_eligible(driver)
        if not eligible:
            raise DriverNotEligibleError(reason)

        vehicle = Vehicle.objects.get(pk=offer.vehicle_id)
        try:
            mobility_service.assign_driver_to_trip(
                ride_request_id=str(offer.ride_request_id),
                driver=driver,
                vehicle=vehicle,
                actor=driver.user,
            )
        except mobility_service.TripAlreadyAssignedError as exc:
            raise RideAlreadyAssignedError(str(exc.message)) from exc
        except mobility_service.MobilityError as exc:
            raise DriverNotEligibleError(exc.message) from exc

        offer.status = choices.DriverOfferStatus.ACCEPTED
        offer.responded_at = now
        offer.save(update_fields=["status", "responded_at", "updated_at"])
        _settle_attempt(offer, status=choices.DispatchAttemptStatus.ACCEPTED)

        # Retire any other open offer on this trip. Under the current schema this
        # matches nothing, and that is worth being explicit about rather than
        # implying a race can leave two open:
        #
        # `uq_driver_offer_one_pending_per_trip` is a partial unique index over
        # `trip` where `status = PENDING`, so a trip can never hold two open
        # offers. This transaction's own write above is uncommitted, so to every
        # other transaction `offer` is still PENDING and the index still blocks a
        # second one. The query can therefore only ever return `offer`, which is
        # excluded below.
        #
        # It is kept because it is the correct handling *if* that invariant is
        # ever relaxed to let several drivers hold an offer at once, and because a
        # driver left holding a stale offer sheet is a real harm. Do not read it
        # as evidence that concurrent dispatch can produce that state; it cannot.
        superseded = list(
            DriverOffer.objects.select_for_update(of=("self",))
            .filter(
                trip_id=offer.trip_id,
                status=choices.DriverOfferStatus.PENDING,
            )
            .exclude(pk=offer.pk)
        )
        for other in superseded:
            other.status = choices.DriverOfferStatus.SUPERSEDED
            other.responded_at = now
            other.save(update_fields=["status", "responded_at", "updated_at"])
            # Those drivers are still holding an offer sheet for a ride that now
            # belongs to somebody else; say so rather than let them accept a dead
            # offer and fail later.
            publish_driver_offer(other, outcome="superseded")
        return offer


def decline_offer(
    *, driver: DriverProfile, offer_id: str, reason: str = "", note: str = ""
) -> DriverOffer:
    """Decline an offer, leaving the ride open for reassignment."""
    now = timezone.now()
    with transaction.atomic():
        offer = _lock_pending_offer(offer_id=offer_id)
        if offer.driver_id != driver.pk:
            raise OfferNotOwnedError("This offer does not belong to the authenticated driver.")
        if offer.status != choices.DriverOfferStatus.PENDING:
            raise OfferAlreadyRespondedError("This offer has already been responded to.")
        if offer.expires_at <= now:
            raise OfferExpiredError("This offer has expired.")

        offer.status = choices.DriverOfferStatus.DECLINED
        offer.responded_at = now
        offer.decline_reason = (reason or "")[:64]
        offer.decline_note = (note or "")[:255]
        offer.save(
            update_fields=["status", "responded_at", "decline_reason", "decline_note", "updated_at"]
        )
        _settle_attempt(
            offer,
            status=choices.DispatchAttemptStatus.DECLINED,
            detail=(reason or note or "declined")[:255],
        )
        # The driver already knows they declined; the point of the frame is to let
        # their app clear the offer sheet without polling.
        publish_driver_offer(offer, outcome="declined")
        return offer


def expire_offer(*, offer: DriverOffer, now=None) -> DriverOffer:
    """Expire a pending offer whose window has closed."""
    now = now or timezone.now()
    with transaction.atomic():
        locked = DriverOffer.objects.select_for_update(of=("self",)).filter(pk=offer.pk).first()
        if locked is None or locked.status != choices.DriverOfferStatus.PENDING:
            return locked or offer
        locked.status = choices.DriverOfferStatus.EXPIRED
        locked.responded_at = now
        locked.save(update_fields=["status", "responded_at", "updated_at"])
        _settle_attempt(locked, status=choices.DispatchAttemptStatus.EXPIRED)
        # Tell the driver the window closed. Without this their app keeps showing
        # an offer they can no longer accept, and only a REST poll would clear it.
        publish_driver_offer(locked, outcome="expired")
        return locked


def cancel_offers_for_trip(*, trip, status: str = choices.DriverOfferStatus.CANCELLED) -> int:
    """Close every open offer for a trip whose ride is no longer dispatchable."""
    now = timezone.now()
    cancelled: list[DriverOffer] = []
    with transaction.atomic():
        offers = list(
            DriverOffer.objects.select_for_update(of=("self",)).filter(
                trip=trip, status=choices.DriverOfferStatus.PENDING
            )
        )
        for offer in offers:
            offer.status = status
            offer.responded_at = now
            offer.save(update_fields=["status", "responded_at", "updated_at"])
            _settle_attempt(offer, status=choices.DispatchAttemptStatus.CANCELLED)
            cancelled.append(offer)
    # Announced after the block: each affected driver learns the ride is off the
    # table, which is what stops them driving to a pickup that no longer exists.
    for offer in cancelled:
        publish_driver_offer(offer, outcome="cancelled")
    return len(cancelled)


def _driver_still_eligible(driver: DriverProfile) -> tuple[bool, str | None]:
    eligible, reason = can_go_online(driver)
    if not eligible:
        return False, reason or "The driver is not eligible to take rides."
    if driver.availability_status != "ONLINE":
        return False, "The driver is no longer available."
    return True, None
