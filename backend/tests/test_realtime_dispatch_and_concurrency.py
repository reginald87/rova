"""Dispatch over the wire, and behaviour under a lossy connection.

Two subjects the other realtime modules do not cover, both because they only appear
once Phase 9 and Phase 10 are wired together:

* **Dispatch on the driver's own socket.** An offer is the one event a driver app
  cannot afford to poll for, so ``driver.offer.created`` and the outcomes that follow
  it have to reach exactly the driver they were extended to — and nobody else.
* **Concurrency and duplicate delivery.** Mobile networks reorder, repeat and delay;
  a second socket from the same account multiplies the write rate; a client that was
  offline misses frames. None of that may corrupt a position or let a driver learn
  about a ride that is not theirs.
"""

import asyncio

import pytest
from django.test import override_settings
from django.utils import timezone

from apps.accounts.services.tokens import create_access_token
from apps.dispatch import choices as dispatch_choices
from apps.dispatch.models import DriverOffer
from apps.dispatch.services import offer_service
from apps.drivers.models import DriverProfile
from apps.drivers.services import driver_service
from apps.mobility.services import mobility_service
from apps.realtime.choices import RealtimeEventType
from apps.realtime.services.events import build_event, publish_driver_offer
from apps.realtime.services.location_store import resolve_location_store

from .dispatch_helpers import (
    dispatch_for,
    make_dispatchable_driver,
    pending_offer_for,
    trip_for,
)
from .mobility_helpers import client_for_driver, ready_ride_request
from .pricing_helpers import POINT_A, POINT_B
from .realtime_helpers import (
    WebsocketCommunicator,
    db_call,
    drain,
    realtime_application,
    realtime_test,
    store_call,
    wait_for_frame,
    wait_for_location,
)

pytestmark = pytest.mark.django_db(transaction=True)

DRIVER_PATH = "/ws/v1/driver/"

# Dispatch runs inline rather than through Celery: a socket assertion must not race
# a worker, and the publication under test is identical either way.
SYNC_DISPATCH = override_settings(DISPATCH_ASYNC=False)


def _socket(path: str, token: str) -> WebsocketCommunicator:
    return WebsocketCommunicator(realtime_application(), path, token=token)


@pytest.fixture
def offer_rider(rider_factory):
    return rider_factory(phone="+2348000000601")


@pytest.fixture
def offered_driver(driver_factory):
    """An online, placed driver — the one dispatch will hand a ride to.

    Re-read with ``active_vehicle`` selected: the eligibility helper round-trips
    through ``refresh_from_db()``, which clears cached relations, and the
    assignment path needs the vehicle.
    """
    user = driver_factory(phone="+2348000000602")
    make_dispatchable_driver(user, client_for_driver(user))
    profile = _reload_profile(user.pk)
    assert profile.availability_status == "ONLINE"
    return {"profile": profile, "user": user, "token": create_access_token(user)[0]}


@pytest.fixture
def bystander_driver(driver_factory):
    """A second online, placed driver who must hear nothing about that ride."""
    user = driver_factory(phone="+2348000000603")
    make_dispatchable_driver(user, client_for_driver(user))
    profile = _reload_profile(user.pk)
    return {"profile": profile, "user": user, "token": create_access_token(user)[0]}


@pytest.fixture
def ride_request(offer_rider):
    """A dispatchable ride request, with no offer raised yet.

    Deliberately un-dispatched: several tests need the socket connected *before* the
    offer exists, which is the whole shape of the production event.
    """
    return ready_ride_request(offer_rider)


@pytest.fixture
def offered_trip(ride_request, offered_driver):
    """The same request, already dispatched, for tests about later stages."""
    with SYNC_DISPATCH:
        dispatch_for(ride_request)
    return {
        "request": ride_request,
        "trip": trip_for(ride_request),
        "offer": pending_offer_for(ride_request),
    }


async def _open_driver_socket(token: str, *, drain_snapshot: bool = True) -> WebsocketCommunicator:
    """Connect a driver socket.

    ``drain_snapshot=True`` consumes the connect-time frames (snapshot, location
    policy) so a test can assert on what arrives *afterwards*. A test asserting on
    the snapshot itself passes ``False`` and reads the frames it is waiting for.
    """
    socket = _socket(DRIVER_PATH, token)
    assert (await socket.connect())["type"] == "websocket.accept"
    if drain_snapshot:
        await drain(socket)
    return socket


# --- dispatch reaches the driver it was extended to -------------------------


@realtime_test
async def test_an_offer_reaches_the_offered_driver_socket(ride_request, offered_driver):
    """The whole point of the driver socket: a driver holding it never polls.

    The frame carries what the driver needs to decide — where to go and how long
    they have — and nothing about the other candidates or the search radius, which
    are server-side decisions.
    """
    socket = await _open_driver_socket(offered_driver["token"])

    with SYNC_DISPATCH:
        await db_call(dispatch_for, ride_request)

    frame = await wait_for_frame(socket, str(RealtimeEventType.DRIVER_OFFER_CREATED))
    offer = await db_call(pending_offer_for, ride_request)

    assert frame["offer_id"] == str(offer.pk)
    assert frame["trip_id"] == str(offer.trip_id)
    assert frame["driver_id"] == str(offered_driver["profile"].pk)
    assert frame["status"] == str(dispatch_choices.DriverOfferStatus.PENDING)
    assert frame["origin_latitude"] == pytest.approx(POINT_A[0])
    assert frame["origin_longitude"] == pytest.approx(POINT_A[1])
    assert frame["distance_to_pickup_meters"] is not None
    assert frame["expires_at"]
    assert frame["estimated_fare_total"]
    assert "search_radius_meters" not in frame, "ranking internals stay server-side"

    await socket.disconnect()


@realtime_test
async def test_an_offer_is_not_broadcast_to_another_driver(
    ride_request, offered_driver, bystander_driver
):
    """Eligibility is not broadcast. A nearby driver who was offered nothing must
    see nothing, or their app shows a ride they cannot accept."""
    bystander = await _open_driver_socket(bystander_driver["token"])
    offered = await _open_driver_socket(offered_driver["token"])

    with SYNC_DISPATCH:
        await db_call(dispatch_for, ride_request)

    frame = await wait_for_frame(offered, str(RealtimeEventType.DRIVER_OFFER_CREATED))
    assert frame["driver_id"] == str(offered_driver["profile"].pk)
    assert await bystander.receive_nothing(timeout=0.5) is True, "not their ride"

    await bystander.disconnect()
    await offered.disconnect()


@realtime_test
async def test_a_declined_offer_is_announced_to_the_driver_who_declined(
    offered_trip, offered_driver
):
    """Their app must be able to clear the offer sheet without polling."""
    socket = await _open_driver_socket(offered_driver["token"])

    await db_call(
        offer_service.decline_offer,
        driver=offered_driver["profile"],
        offer_id=str(offered_trip["offer"].pk),
        reason="TOO_FAR",
    )

    frame = await wait_for_frame(socket, str(RealtimeEventType.DRIVER_OFFER_DECLINED))
    assert frame["offer_id"] == str(offered_trip["offer"].pk)
    assert frame["status"] == str(dispatch_choices.DriverOfferStatus.DECLINED)
    await socket.disconnect()


@realtime_test
async def test_an_expired_offer_is_announced_to_the_driver(offered_trip, offered_driver):
    """The Celery sweep closes the window; without the frame the driver keeps showing
    an offer only a REST call would reveal is dead."""
    socket = await _open_driver_socket(offered_driver["token"])

    await db_call(offer_service.expire_offer, offer=offered_trip["offer"])

    frame = await wait_for_frame(socket, str(RealtimeEventType.DRIVER_OFFER_EXPIRED))
    assert frame["offer_id"] == str(offered_trip["offer"].pk)
    assert frame["status"] == str(dispatch_choices.DriverOfferStatus.EXPIRED)
    await socket.disconnect()


@realtime_test
async def test_accepting_an_offer_tells_the_driver_they_have_a_trip(offered_trip, offered_driver):
    """``driver.offer.created`` is answered by the authoritative REST call, and the
    trip event is what the socket delivers back.

    This is the Phase 9 → Phase 10 handoff: the driver app must not have to
    re-authenticate or re-fetch to learn that the ride is now theirs.
    """
    socket = await _open_driver_socket(offered_driver["token"])

    await db_call(
        offer_service.accept_offer,
        driver=offered_driver["profile"],
        offer_id=str(offered_trip["offer"].pk),
    )

    frame = await wait_for_frame(socket, str(RealtimeEventType.TRIP_ASSIGNED))
    assert frame["trip_id"] == str(offered_trip["trip"].pk)
    assert frame["status"] == "DRIVER_ASSIGNED"
    assert frame["driver_id"] == str(offered_driver["profile"].pk)
    await socket.disconnect()


@realtime_test
async def test_a_superseded_offer_reaches_the_driver_who_held_it(offered_trip, offered_driver):
    """A driver holding an offer for a ride that has gone to somebody else must be
    told on the socket they already have open, or their app keeps driving them
    toward a pickup that is no longer theirs.

    Driven through the publication layer rather than ``accept_offer`` on purpose:
    ``uq_driver_offer_one_pending_per_trip`` is a partial unique index, so only one
    PENDING offer per trip can exist and the second driver cannot be given one. The
    supersession frame is therefore unreachable end-to-end, and what is under test
    here is the realtime contract — right driver group, right payload, right status.
    """
    socket = await _open_driver_socket(offered_driver["token"])
    offer = offered_trip["offer"]

    await asyncio.to_thread(_supersede_and_publish, offer.pk)

    frame = await wait_for_frame(socket, str(RealtimeEventType.DRIVER_OFFER_SUPERSEDED))
    assert frame["offer_id"] == str(offer.pk)
    assert frame["trip_id"] == str(offer.trip_id)
    assert frame["driver_id"] == str(offered_driver["profile"].pk)
    assert frame["status"] == str(dispatch_choices.DriverOfferStatus.SUPERSEDED)

    await socket.disconnect()


@realtime_test
async def test_a_trip_status_change_reaches_the_drivers_own_socket(offered_trip, offered_driver):
    """The driver learns of their own trip's progress on the socket they stream
    positions over — which is also what keeps their active-trip cache correct
    without a query per GPS fix."""
    trip = offered_trip["trip"]
    await db_call(
        mobility_service.assign_driver_to_trip,
        ride_request_id=str(offered_trip["request"].pk),
        driver=offered_driver["profile"],
        vehicle=offered_driver["profile"].active_vehicle,
        actor=offered_driver["user"],
    )
    socket = await _open_driver_socket(offered_driver["token"])

    await db_call(mobility_service.accept_trip, actor=offered_driver["user"], trip_id=str(trip.pk))

    frame = await wait_for_frame(socket, str(RealtimeEventType.TRIP_STATUS))
    assert frame["status"] == "DRIVER_ACCEPTED"
    assert frame["trip_id"] == str(trip.pk)
    await socket.disconnect()


@realtime_test
async def test_availability_is_announced_to_the_drivers_own_socket(offered_driver):
    """Phase 4 availability stays authoritative; realtime only mirrors it. The
    announcement happens because the driver went online through the business
    action, not because a socket opened or closed."""
    socket = await _open_driver_socket(offered_driver["token"])

    await db_call(_set_availability, user=offered_driver["user"], target="OFFLINE")

    frame = await wait_for_frame(socket, str(RealtimeEventType.DRIVER_AVAILABILITY))
    assert frame["driver_id"] == str(offered_driver["profile"].pk)
    assert frame["availability"] == "OFFLINE"
    await socket.disconnect()


@realtime_test
async def test_a_dropped_socket_does_not_make_a_driver_offline(offered_driver):
    """The distinction that matters on a bad network: a lost connection is a
    communication fact, not a business one.

    A driver's socket dying must not touch ``availability_status``; otherwise every
    tunnel in Lagos silently takes half the fleet offline and dispatch stops.
    """
    socket = await _open_driver_socket(offered_driver["token"])
    await socket.disconnect()

    profile = await db_call(DriverProfile.objects.get, pk=offered_driver["profile"].pk)

    assert profile.availability_status == "ONLINE", "a dropped socket is not a go-offline"
    assert profile.available_since is not None


# --- concurrency, reconnect and duplicate delivery ---------------------------


@realtime_test
async def test_a_burst_of_fixes_leaves_one_coherent_position(offered_driver):
    """A device faster than the configured cadence is coalesced, not queued.

    What matters after the burst is that the store holds a *complete, in-range*
    position from one of the frames actually sent — a partially applied write would
    show up here as a missing coordinate or a leftover from a previous driver.
    """
    socket = await _open_driver_socket(offered_driver["token"])
    await _clear_position(driver_id=offered_driver["profile"].pk)

    sent = []
    with override_settings(DRIVER_LOCATION_MIN_INTERVAL_MS=0):
        for index in range(25):
            frame = {
                "type": "location.update",
                "latitude": POINT_A[0] + index / 10_000,
                "longitude": POINT_A[1] + index / 10_000,
                "accuracy": 8.0,
                "heading": 120.0,
                "speed": 7.5,
            }
            sent.append(frame)
            await socket.send_json(frame)

    await wait_for_location(offered_driver["profile"].pk)
    latest = await _quiescent_location(driver_id=offered_driver["profile"].pk)

    assert any(
        latest.latitude == pytest.approx(frame["latitude"], abs=1e-9)
        and latest.longitude == pytest.approx(frame["longitude"], abs=1e-9)
        for frame in sent
    ), f"settled on a position that was never sent: {latest!r}"
    assert -90 <= latest.latitude <= 90
    assert -180 <= latest.longitude <= 180
    assert latest.accuracy == pytest.approx(8.0), "the whole frame is stored, not just the pair"
    assert latest.heading == pytest.approx(120.0)
    assert latest.speed == pytest.approx(7.5)
    assert latest.driver_id == str(offered_driver["profile"].pk)
    await socket.disconnect()


@realtime_test
async def test_a_second_socket_cannot_multiply_a_drivers_write_rate(offered_driver):
    """Coalescing is keyed on the driver, not the connection.

    Enforcing it in the store rather than the consumer is what makes this hold: a
    driver whose app reconnects, or who is signed in on a second device, cannot
    double ROVA's Redis and pub/sub cost by opening another socket.
    """
    with override_settings(DRIVER_LOCATION_MIN_INTERVAL_MS=60_000):
        first = await _open_driver_socket(offered_driver["token"])
        second = await _open_driver_socket(offered_driver["token"])
        await _clear_position(driver_id=offered_driver["profile"].pk)

        await first.send_json(
            {"type": "location.update", "latitude": POINT_A[0], "longitude": POINT_A[1]}
        )
        await wait_for_location(offered_driver["profile"].pk)

        # A different connection, same identity, immediately afterwards.
        await second.send_json(
            {"type": "location.update", "latitude": POINT_B[0], "longitude": POINT_B[1]}
        )

        assert await first.receive_nothing(timeout=0.5) is True, "coalesced, not an error"
        assert await second.receive_nothing(timeout=0.5) is True

        record = await store_call(_get_location, driver_id=offered_driver["profile"].pk)
        assert record.latitude == pytest.approx(POINT_A[0]), "the first fix still stands"

        await first.disconnect()
        await second.disconnect()


@realtime_test
async def test_a_reconnect_recovers_a_transition_the_driver_missed(offered_trip, offered_driver):
    """A socket that was down cannot replay what it missed, so the snapshot has to be
    authoritative.

    This is the case that separates "the socket is a stream" from "the snapshot is
    the state": the driver is disconnected across two transitions and comes back to
    the real current status rather than a stale one.
    """
    trip = offered_trip["trip"]
    socket = await _open_driver_socket(offered_driver["token"])
    await socket.disconnect()

    # The dispatcher's offer is what assigns the driver; transitions past DRIVER_ASSIGNED
    # are only legal once the trip has one.
    await db_call(
        offer_service.accept_offer,
        driver=offered_driver["profile"],
        offer_id=str(offered_trip["offer"].pk),
    )

    for transition in (
        mobility_service.accept_trip,
        mobility_service.driver_arriving,
        mobility_service.driver_at_pickup,
    ):
        await db_call(transition, actor=offered_driver["user"], trip_id=str(trip.pk))

    again = await _open_driver_socket(offered_driver["token"], drain_snapshot=False)
    snapshot = await wait_for_frame(again, str(RealtimeEventType.DRIVER_SNAPSHOT))
    assert snapshot["active_trip"] is not None
    assert snapshot["active_trip"]["trip_id"] == str(trip.pk)
    assert snapshot["active_trip"]["status"] == "DRIVER_AT_PICKUP"

    # And the stream resumes from there rather than replaying what was missed.
    await db_call(
        mobility_service.rider_onboard,
        actor=offered_driver["user"],
        trip_id=str(trip.pk),
    )
    frame = await wait_for_frame(again, str(RealtimeEventType.TRIP_STATUS))
    assert frame["status"] == "RIDER_ONBOARD"
    await again.disconnect()


@realtime_test
async def test_a_repeated_delivery_carries_a_fresh_identity():
    """Duplicate *delivery* is a client concern; duplicate *identity* is ours.

    A client deduplicates on ``event_id``, so two publishes of the same logical
    change must still be distinguishable — otherwise a redelivery after a reconnect
    would be dropped as a repeat of something the client never saw.
    """
    first = build_event("trip.status", trip_id="t", status="DRIVER_ACCEPTED")
    second = build_event("trip.status", trip_id="t", status="DRIVER_ACCEPTED")

    assert first["event_id"] != second["event_id"]
    assert first["type"] == second["type"]
    assert first["timestamp"] and second["timestamp"]


@realtime_test
async def test_the_driver_snapshot_reports_a_live_offer_after_a_reconnect(
    offered_trip, offered_driver
):
    """An offer created while the socket was down is exactly what a reconnecting
    driver would otherwise never learn about.

    The snapshot carries it, so the app can render an offer sheet on connect without
    a REST round trip.
    """
    socket = await _open_driver_socket(offered_driver["token"])
    await socket.disconnect()

    again = await _open_driver_socket(offered_driver["token"], drain_snapshot=False)
    snapshot = await wait_for_frame(again, str(RealtimeEventType.DRIVER_SNAPSHOT))

    open_offers = snapshot["open_offers"]
    assert len(open_offers) == 1
    assert open_offers[0]["offer_id"] == str(offered_trip["offer"].pk)
    assert open_offers[0]["trip_id"] == str(offered_trip["trip"].pk)
    assert open_offers[0]["expires_at"]
    assert snapshot["min_interval_ms"] > 0, "the client learns the cadence, not a guess"
    await again.disconnect()


@realtime_test
async def test_a_stored_position_survives_a_reconnect_as_stored_state(offered_driver):
    """Redis is ephemeral but not per-connection: a reconnect finds the position the
    driver already reported, marked with the server's own receipt time."""
    socket = await _open_driver_socket(offered_driver["token"])
    await _clear_position(driver_id=offered_driver["profile"].pk)
    await socket.send_json(
        {"type": "location.update", "latitude": POINT_A[0], "longitude": POINT_A[1]}
    )
    await wait_for_location(offered_driver["profile"].pk)
    await socket.disconnect()

    again = await _open_driver_socket(offered_driver["token"], drain_snapshot=False)
    snapshot = await wait_for_frame(again, str(RealtimeEventType.DRIVER_SNAPSHOT))
    assert snapshot["position"]["freshness"] == "FRESH"
    position = snapshot["position"]["position"]
    assert position["latitude"] == pytest.approx(POINT_A[0])
    assert position["longitude"] == pytest.approx(POINT_A[1])
    assert position["accuracy"] is None
    assert position["recorded_at"], "the server stamps its own receipt time"
    assert "client_timestamp" not in position, (
        "the device clock is diagnostics only and is not echoed as state"
    )
    await again.disconnect()


@realtime_test
async def test_concurrent_sockets_do_not_corrupt_one_drivers_position(offered_driver):
    """Two live sockets for one identity, interleaved fixes, one coherent record."""
    first = await _open_driver_socket(offered_driver["token"])
    second = await _open_driver_socket(offered_driver["token"])
    await _clear_position(driver_id=offered_driver["profile"].pk)

    sent: list[tuple[float, float]] = []
    with override_settings(DRIVER_LOCATION_MIN_INTERVAL_MS=0):
        for index in range(10):
            here = (POINT_A[0] + index / 10_000, POINT_A[1])
            there = (POINT_A[0], POINT_A[1] + index / 10_000)
            sent.extend((here, there))
            await asyncio.gather(
                first.send_json(
                    {
                        "type": "location.update",
                        "latitude": here[0],
                        "longitude": here[1],
                    }
                ),
                second.send_json(
                    {
                        "type": "location.update",
                        "latitude": there[0],
                        "longitude": there[1],
                    }
                ),
            )

    record = await _quiescent_location(driver_id=offered_driver["profile"].pk)

    assert record.driver_id == str(offered_driver["profile"].pk), "one identity, one record"
    assert any(
        record.latitude == pytest.approx(lat, abs=1e-9)
        and record.longitude == pytest.approx(lon, abs=1e-9)
        for lat, lon in sent
    ), f"record is a torn mix of two frames' coordinates: {record!r}"
    assert record.is_fresh(), "a coherent record, not a torn one"
    assert record.server_received_at is not None

    await first.disconnect()
    await second.disconnect()


# --- sync helpers (never touch the event loop) -------------------------------


async def _clear_position(*, driver_id):
    """Drop the eligibility fixture's position so a test owns the write budget.

    Without this the driver already has a stored fix with a fresh server receipt
    time, which the min-interval rule counts against — the test's own first update
    would be coalesced and it would appear to assert nothing.
    """
    await store_call(_delete_location, driver_id=driver_id)


async def _quiescent_location(*, driver_id, attempts=300):
    """Wait until the stored position stops changing.

    Frames from several connections are processed on independent worker threads, so
    under concurrency *which* frame lands last is a scheduling outcome, not a
    guarantee. A test asserting the final value of an interleaved burst would be
    asserting the GIL's mood. What is assertable is that the record converges — and
    then, on that settled record, that it is coherent and genuinely one of the
    positions sent.
    """
    previous = object()
    stable = 0
    record = None
    for _ in range(attempts):
        record = await store_call(_get_location, driver_id=driver_id)
        if record is not None and record == previous:
            stable += 1
            if stable >= 3:
                return record
        else:
            stable = 0
        previous = record
        await asyncio.sleep(0.05)
    raise AssertionError(f"Position for {driver_id} never stopped changing; last read {record!r}")


def _get_location(*, driver_id):
    return resolve_location_store().get_location(driver_id=driver_id)


def _delete_location(*, driver_id):
    return resolve_location_store().delete_location(driver_id=driver_id)


def _reload_profile(user_id):
    """Re-read a driver profile with the relations the tests read."""
    return DriverProfile.objects.select_related("user", "active_vehicle").get(user_id=user_id)


def _set_availability(*, user, target):
    return driver_service.set_availability(user=user, target=target)


def _supersede_and_publish(offer_id):
    """Mark an offer superseded, then announce it — in that order.

    The order is the contract, not an incidental detail: the frame's ``status`` is
    read off the offer, so publishing before the save would tell the driver the
    offer was superseded while the payload still claimed ``PENDING``.
    """
    offer = DriverOffer.objects.get(pk=offer_id)
    offer.status = dispatch_choices.DriverOfferStatus.SUPERSEDED
    offer.responded_at = timezone.now()
    offer.save(update_fields=["status", "responded_at", "updated_at"])
    publish_driver_offer(offer, outcome="superseded")
