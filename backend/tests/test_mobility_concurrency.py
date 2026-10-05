"""Concurrency: exactly one actor wins a state transition under contention."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest
from django.db import connections

from apps.mobility.choices import TripStatus
from apps.mobility.services.mobility_service import (
    InvalidTripTransitionError,
    accept_trip,
    assign_driver_to_trip,
    cancel_trip,
    complete_trip,
    driver_arriving,
    driver_at_pickup,
    rider_onboard,
    start_trip,
)

from .mobility_helpers import eligible_driver, ready_ride_request

pytestmark = pytest.mark.django_db(transaction=True)


def _assigned(rider_user, driver_user, driver_client):
    request = ready_ride_request(rider_user)
    profile, vehicle = eligible_driver(driver_client, driver_user)
    trip = assign_driver_to_trip(ride_request_id=str(request.pk), driver=profile, vehicle=vehicle)
    return trip


@contextmanager
def _worker_session():
    """Close this thread's database connection when the worker finishes.

    Django connections are thread-local, so every ``ThreadPoolExecutor`` worker
    that touches the ORM opens a connection that nothing ever closes: the pool
    joins the threads and discards the connection along with them. The leaked
    sessions then block Django's teardown of the test database, which surfaces as
    ``database "test_rova" is being accessed by other users`` at the end of the
    run and ``already exists`` at the start of the next one.

    This has to run inside the worker — closing connections from the main thread
    cannot reach a connection owned by a thread that has already exited.
    """
    try:
        yield
    finally:
        connections.close_all()


def _race(attempts, workers=2):
    """Run the attempts concurrently; one outcome string per attempt."""
    outcomes: list[str] = []

    def run(entry):
        try:
            with _worker_session():
                entry()
            outcomes.append("ok")
        except InvalidTripTransitionError:
            outcomes.append("loser")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, fn) for fn in attempts]
        for f in futures:
            f.result()
    return outcomes


def test_two_concurrent_accepts_win_once(rider, driver, driver_client):
    rider_user, _ = rider
    driver_user, _ = driver
    trip = _assigned(rider_user, driver_user, driver_client)

    outcomes = _race([lambda: accept_trip(actor=driver_user, trip_id=str(trip.pk))] * 2)

    assert sorted(outcomes) == ["loser", "ok"]
    trip.refresh_from_db()
    assert trip.status == TripStatus.DRIVER_ACCEPTED


def test_two_concurrent_completes_win_once(rider, driver, driver_client):
    rider_user, _ = rider
    driver_user, _ = driver
    trip = _assigned(rider_user, driver_user, driver_client)
    trip = accept_trip(actor=driver_user, trip_id=str(trip.pk))
    trip = driver_arriving(actor=driver_user, trip_id=str(trip.pk))
    trip = driver_at_pickup(actor=driver_user, trip_id=str(trip.pk))
    trip = rider_onboard(actor=rider_user, trip_id=str(trip.pk))
    trip = start_trip(actor=driver_user, trip_id=str(trip.pk))

    outcomes = _race(
        [
            lambda: complete_trip(
                actor=driver_user,
                trip_id=str(trip.pk),
                distance_meters=2500,
                duration_seconds=400,
            )
        ]
        * 2
    )

    assert sorted(outcomes) == ["loser", "ok"]
    trip.refresh_from_db()
    assert trip.status == TripStatus.TRIP_COMPLETED
    assert trip.final_fare is not None
    from apps.pricing.models import Fare

    assert Fare.objects.filter(calculation_type="FINAL").count() == 1


def test_cancel_racing_accept(rider, driver, driver_client):
    rider_user, _ = rider
    driver_user, _ = driver
    trip = _assigned(rider_user, driver_user, driver_client)
    outcomes: list[str] = []

    def cancel():
        try:
            with _worker_session():
                cancel_trip(
                    actor=rider_user,
                    trip_id=str(trip.pk),
                    reason="RIDER_CHANGED_MIND",
                )
            outcomes.append("cancel")
        except InvalidTripTransitionError:
            outcomes.append("cancel-loser")

    def accept():
        try:
            with _worker_session():
                accept_trip(actor=driver_user, trip_id=str(trip.pk))
            outcomes.append("accept")
        except InvalidTripTransitionError:
            outcomes.append("accept-loser")

    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in [pool.submit(fn) for fn in (cancel, accept)]:
            f.result()

    # Rider cancellation is legal from both DRIVER_ASSIGNED and DRIVER_ACCEPTED,
    # so the cancellation always lands; the accept only wins if it ran first.
    assert outcomes.count("cancel") == 1
    trip.refresh_from_db()
    assert trip.status == TripStatus.CANCELLED_BY_RIDER
