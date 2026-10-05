"""Offer lifecycle: acceptance, decline, expiry, and reassignment."""

from datetime import timedelta

import pytest
from django.db import IntegrityError, transaction
from django.test import override_settings
from django.utils import timezone

from apps.dispatch import choices
from apps.dispatch.models import DispatchAttempt, DriverOffer
from apps.dispatch.services import dispatch_service, offer_service
from apps.dispatch.services.errors import (
    OfferAlreadyRespondedError,
    OfferExpiredError,
    OfferNotFoundError,
    OfferNotOwnedError,
)
from apps.mobility.choices import TripStatus
from apps.mobility.models import Trip

from .dispatch_helpers import (
    dispatch_for,
    extra_driver,
    make_dispatchable_driver,
    pending_offer_for,
    ready_ride_request,
    trip_for,
)
from .mobility_helpers import client_for_driver

pytestmark = pytest.mark.django_db

SYNC_DISPATCH = override_settings(DISPATCH_ASYNC=False)


def _driver_profile(driver_user):
    return driver_user.driver_profile


def test_accepting_an_offer_assigns_the_driver(rider, driver, driver_client):
    rider_user, _ = rider
    driver_user, _ = driver
    profile, _vehicle = make_dispatchable_driver(driver_user, driver_client)
    request = ready_ride_request(rider_user)
    offer = dispatch_for(request).offer

    accepted = offer_service.accept_offer(driver=profile, offer_id=str(offer.pk))

    assert accepted.status == choices.DriverOfferStatus.ACCEPTED
    trip = trip_for(request)
    assert trip.status == TripStatus.DRIVER_ASSIGNED
    assert trip.driver_id == profile.pk
    # Accepting takes the driver off the idle clock.
    profile.refresh_from_db()
    assert profile.availability_status == "BUSY"
    assert profile.available_since is None
    attempt = DispatchAttempt.objects.get(pk=offer.dispatch_attempt_id)
    assert attempt.status == choices.DispatchAttemptStatus.ACCEPTED
    assert attempt.responded_at is not None


def test_accepting_twice_is_rejected(rider, driver, driver_client):
    rider_user, _ = rider
    driver_user, _ = driver
    profile, _vehicle = make_dispatchable_driver(driver_user, driver_client)
    request = ready_ride_request(rider_user)
    offer = dispatch_for(request).offer
    offer_service.accept_offer(driver=profile, offer_id=str(offer.pk))

    with pytest.raises(OfferAlreadyRespondedError):
        offer_service.accept_offer(driver=profile, offer_id=str(offer.pk))


def test_expired_offer_cannot_be_accepted(rider, driver, driver_client):
    rider_user, _ = rider
    driver_user, _ = driver
    profile, _vehicle = make_dispatchable_driver(driver_user, driver_client)
    request = ready_ride_request(rider_user)
    offer = dispatch_for(request).offer
    DriverOffer.objects.filter(pk=offer.pk).update(expires_at=timezone.now() - timedelta(seconds=1))

    with pytest.raises(OfferExpiredError):
        offer_service.accept_offer(driver=profile, offer_id=str(offer.pk))

    assert trip_for(request).status == TripStatus.SEARCHING


def test_driver_cannot_see_or_accept_another_drivers_offer(
    rider, driver_factory, driver, driver_client
):
    rider_user, _ = rider
    owner_user, _ = driver
    other_user = extra_driver(driver_factory, 11)
    owner, _v = make_dispatchable_driver(owner_user, driver_client)
    other, _v2 = make_dispatchable_driver(other_user, client_for_driver(other_user))
    request = ready_ride_request(rider_user)
    offer = dispatch_for(request).offer

    with pytest.raises(OfferNotFoundError):
        offer_service.get_offer(driver=other, offer_id=str(offer.pk))
    with pytest.raises(OfferNotOwnedError):
        offer_service.accept_offer(driver=other, offer_id=str(offer.pk))


def test_only_one_pending_offer_per_driver(rider, driver, driver_client):
    """A driver can never be considering two rides at once."""
    rider_user, _ = rider
    driver_user, _ = driver
    profile, _vehicle = make_dispatchable_driver(driver_user, driver_client)
    first_request = ready_ride_request(rider_user)
    dispatch_for(first_request)

    second_request = ready_ride_request(rider_user)
    outcome = dispatch_for(second_request)

    # The driver already holds a pending offer, so they are not a candidate.
    assert outcome.offer is None
    assert (
        DriverOffer.objects.filter(driver=profile, status=choices.DriverOfferStatus.PENDING).count()
        == 1
    )


@override_settings(DISPATCH_SEARCH_RADII_METERS=[30000])
def test_a_trip_can_never_hold_two_open_offers(rider, driver_factory):
    """A trip has one open offer at a time, and the database says so.

    ``uq_driver_offer_one_pending_per_trip`` is a partial unique index over
    ``trip`` where ``status = PENDING``. This pins that invariant from the
    dispatch side: two drivers cannot both be handed the same ride while it is
    being offered. If this test ever fails because someone wanted several drivers
    racing for one ride, then the constraint has been relaxed — at which point
    ``accept_offer``'s supersession loop stops being unreachable and the
    ``driver.offer.superseded`` frame becomes a real event to test end to end
    rather than only at the publication layer.
    """
    rider_user, _ = rider
    first_user = extra_driver(driver_factory, 31)
    second_user = extra_driver(driver_factory, 32)
    first, _ = make_dispatchable_driver(first_user, client_for_driver(first_user))
    second, _ = make_dispatchable_driver(second_user, client_for_driver(second_user))

    request = ready_ride_request(rider_user)
    dispatch_for(request)
    offer = pending_offer_for(request)
    trip = offer.trip
    assert trip.driver_id is None, "an offered trip is not yet assigned"

    # A second PENDING offer on the same trip is refused by the index, not merely
    # avoided by convention. `transaction.atomic()` contains the failure: without
    # a savepoint the aborted statement would poison the whole test transaction.
    with pytest.raises(IntegrityError), transaction.atomic():
        DriverOffer.objects.create(
            ride_request=request,
            trip=trip,
            driver=second,
            vehicle=second.active_vehicle,
            dispatch_attempt=offer.dispatch_attempt,
            status=choices.DriverOfferStatus.PENDING,
            distance_to_pickup_meters=40,
            search_radius_meters=30000,
            expires_at=offer.expires_at,
        )

    assert (
        DriverOffer.objects.filter(trip=trip, status=choices.DriverOfferStatus.PENDING).count() == 1
    )


@override_settings(DISPATCH_SEARCH_RADII_METERS=[30000])
def test_decline_reassigns_to_the_next_driver(rider, driver_factory):
    rider_user, _ = rider
    first_user = extra_driver(driver_factory, 21)
    second_user = extra_driver(driver_factory, 22)
    first, _ = make_dispatchable_driver(first_user, client_for_driver(first_user))
    second, _ = make_dispatchable_driver(second_user, client_for_driver(second_user))
    request = ready_ride_request(rider_user)

    with SYNC_DISPATCH:
        first_offer = dispatch_for(request).offer
        assert first_offer.driver_id == first.pk
        offer_service.decline_offer(driver=first, offer_id=str(first_offer.pk))
        dispatch_service.reassign_after_decline(offer=first_offer)

    second_offer = pending_offer_for(request)
    assert second_offer.driver_id == second.pk
    assert second_offer.id != first_offer.id


@override_settings(DISPATCH_SEARCH_RADII_METERS=[30000])
def test_declining_driver_is_never_offered_the_same_ride_again(rider, driver, driver_client):
    rider_user, _ = rider
    driver_user, _ = driver
    profile, _vehicle = make_dispatchable_driver(driver_user, driver_client)
    request = ready_ride_request(rider_user)

    offer = dispatch_for(request).offer
    offer_service.decline_offer(driver=profile, offer_id=str(offer.pk))
    outcome = dispatch_service.dispatch_ride(ride_request_id=str(request.pk))

    assert outcome.offer is None
    assert outcome.reason == choices.NoDriverReason.ALL_CANDIDATES_EXHAUSTED


@override_settings(DISPATCH_SEARCH_RADII_METERS=[30000])
def test_expired_offer_is_swept_and_reassigned(rider, driver_factory):
    rider_user, _ = rider
    first_user = extra_driver(driver_factory, 31)
    second_user = extra_driver(driver_factory, 32)
    first, _ = make_dispatchable_driver(first_user, client_for_driver(first_user))
    second, _ = make_dispatchable_driver(second_user, client_for_driver(second_user))
    request = ready_ride_request(rider_user)
    first_offer = dispatch_for(request).offer
    DriverOffer.objects.filter(pk=first_offer.pk).update(
        expires_at=timezone.now() - timedelta(seconds=1)
    )

    outcomes = dispatch_service.expire_offers_and_reassign()

    first_offer.refresh_from_db()
    assert first_offer.status == choices.DriverOfferStatus.EXPIRED
    assert len(outcomes) == 1
    assert pending_offer_for(request).driver_id == second.pk
    assert first.pk != second.pk


@override_settings(DISPATCH_SEARCH_RADII_METERS=[30000])
def test_sweep_is_a_noop_when_no_offer_has_lapsed(rider, driver, driver_client):
    rider_user, _ = rider
    driver_user, _ = driver
    make_dispatchable_driver(driver_user, driver_client)
    request = ready_ride_request(rider_user)
    dispatch_for(request)

    assert dispatch_service.expire_offers_and_reassign() == []


def test_pending_offers_list_excludes_answered_offers(rider, driver, driver_client):
    rider_user, _ = rider
    driver_user, _ = driver
    profile, _vehicle = make_dispatchable_driver(driver_user, driver_client)
    request = ready_ride_request(rider_user)
    dispatch_for(request)
    assert len(offer_service.list_pending_offers(driver=profile)) == 1

    offer_service.decline_offer(driver=profile, offer_id=str(pending_offer_for(request).pk))

    assert offer_service.list_pending_offers(driver=profile) == []


def test_driver_taken_offline_mid_offer_cannot_accept(rider, driver, driver_client):
    """Eligibility is re-checked at accept time, not only at offer time."""
    from apps.mobility.services.mobility_service import set_availability

    rider_user, _ = rider
    driver_user, _ = driver
    profile, _vehicle = make_dispatchable_driver(driver_user, driver_client)
    request = ready_ride_request(rider_user)
    offer = dispatch_for(request).offer
    set_availability(profile, "OFFLINE")

    from apps.dispatch.services.errors import DriverNotEligibleError

    with pytest.raises(DriverNotEligibleError):
        offer_service.accept_offer(driver=profile, offer_id=str(offer.pk))

    assert trip_for(request).driver_id is None


def test_dispatch_history_is_append_only(rider, driver_factory):
    rider_user, _ = rider
    first_user = extra_driver(driver_factory, 41)
    second_user = extra_driver(driver_factory, 42)
    first, _ = make_dispatchable_driver(first_user, client_for_driver(first_user))
    make_dispatchable_driver(second_user, client_for_driver(second_user))
    request = ready_ride_request(rider_user)

    with override_settings(DISPATCH_SEARCH_RADII_METERS=[30000]), SYNC_DISPATCH:
        offer = dispatch_for(request).offer
        offer_service.decline_offer(driver=first, offer_id=str(offer.pk))
        dispatch_service.reassign_after_decline(offer=offer)

    history = dispatch_service.dispatch_history(ride_request_id=str(request.pk))
    assert [attempt.attempt_number for attempt in history] == [1, 2]
    assert history[0].status == choices.DispatchAttemptStatus.DECLINED
    assert history[1].status == choices.DispatchAttemptStatus.OFFERED
    assert Trip.objects.filter(ride_request=request).count() == 1
