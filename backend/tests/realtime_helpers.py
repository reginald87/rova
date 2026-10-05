"""Test helpers for the Phase 10 realtime surface.

Two things here are worth explaining.

**Why a hand-written communicator.** ``channels.testing.WebsocketCommunicator``
imports ``daphne.testing``, and Daphne is an ASGI *server* — a production
dependency that should not be dragged into a test suite only to exchange a handful
of ASGI messages. ``ApplicationCommunicator`` from ``asgiref`` is already a
test-only dependency that Channels itself requires, so the communicator below is
built on that instead: it speaks the WebSocket subset of the ASGI protocol
(``websocket.connect`` / ``websocket.receive`` / ``websocket.send`` /
``websocket.close``) and nothing more.

**Why the tests use ``InMemoryChannelLayer``.** A test that asserts "the driver
received ``driver.offer.created``" should not depend on a Redis pub/sub round trip
to observe it. The in-memory layer gives tests exact, ordered, inspectable sends,
and it is the same layer Channels' own test utilities use. The Redis layer is
exercised separately by the store tests.
"""

from __future__ import annotations

import asyncio
import functools
import json
import time
from collections.abc import Callable, Coroutine
from urllib.parse import parse_qs, urlencode

from asgiref.sync import sync_to_async
from asgiref.testing import ApplicationCommunicator
from channels.layers import get_channel_layer
from channels.security.websocket import AllowedHostsOriginValidator
from django.db import connections
from django.urls import path

from apps.realtime.middleware import JWTAuthMiddleware
from apps.realtime.routing import websocket_urlpatterns

# Origin/Host pair the handshake validator accepts in tests. ALLOWED_HOSTS
# contains "*" in every environment this suite runs in, but sending a real pair
# keeps the test honest about the shape of a browser handshake.
TEST_HOST = "testserver"
TEST_ORIGIN = f"http://{TEST_HOST}"


def realtime_test(fn: Callable[..., Coroutine]) -> Callable[..., None]:
    """Run an ``async def`` test body on a dedicated event loop.

    The project has no ``pytest-asyncio``, and adding one purely to drive the
    communicator would be a dependency bought for a test harness. Every ORM call
    inside the consumers already goes through ``database_sync_to_async``, so the
    bodies contain no direct database access and a plain ``asyncio.run`` is enough.

    ``functools.wraps`` keeps the wrapper's signature visible to pytest, so
    fixtures still resolve by name.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


def realtime_application():
    """The production WebSocket stack, minus the HTTP router.

    Built the same way ``config.asgi`` builds it, so a test exercises the real
    middleware order — origin validation, then authentication — rather than a
    consumer called directly.
    """
    return JWTAuthMiddleware(AllowedHostsOriginValidator(_url_router()))


def _url_router():
    from channels.routing import URLRouter

    return URLRouter(websocket_urlpatterns)


def _url_router_with_overrides(overrides):
    """A router with one route replaced, for the cross-tenant authorization tests."""
    from channels.routing import URLRouter

    from apps.realtime.consumers import DriverConsumer, OperationsConsumer, TripConsumer

    routes = [
        path("ws/v1/driver/", overrides.get("driver", DriverConsumer.as_asgi())),
        path(
            "ws/v1/trips/<str:trip_id>/",
            overrides.get("trip", TripConsumer.as_asgi()),
        ),
        path("ws/v1/operations/", overrides.get("operations", OperationsConsumer.as_asgi())),
    ]
    return URLRouter(routes)


class WebsocketCommunicator:
    """A minimal WebSocket client for ASGI applications."""

    def __init__(self, application, path: str, *, token: str | None = None, headers=None):
        query = urlencode({"token": token}) if token else ""
        self.scope = {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "scheme": "ws",
            "path": path,
            "raw_path": path.encode(),
            "query_string": query.encode(),
            "root_path": "",
            "headers": [
                (b"host", TEST_HOST.encode()),
                (b"origin", TEST_ORIGIN.encode()),
                *[(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()],
            ],
            "client": ("127.0.0.1", 5555),
            "server": (TEST_HOST, 80),
            "subprotocols": [],
        }
        self.application = application
        self.instance = ApplicationCommunicator(application, self.scope)

    async def connect(self, timeout: float = 5) -> dict:
        """Perform the handshake and return the accept or close message."""
        await self.instance.send_input({"type": "websocket.connect"})
        return await self.receive_output(timeout=timeout)

    async def send_json(self, payload: dict, timeout: float = 5) -> None:
        await self.instance.send_input({"type": "websocket.receive", "text": json.dumps(payload)})
        # Give the application a turn so a synchronous failure surfaces here
        # rather than as a mystery failure on the next read.
        await asyncio.sleep(0)

    async def send_text(self, text: str) -> None:
        await self.instance.send_input({"type": "websocket.receive", "text": text})

    async def receive_json(self, timeout: float = 5) -> dict:
        frame = await self.receive_json_or_none(timeout=timeout)
        if frame is None:
            raise AssertionError(f"No frame arrived within {timeout}s.")
        return frame

    async def receive_output(self, timeout: float = 5) -> dict:
        """Read one raw ASGI message.

        Deliberately not ``ApplicationCommunicator.receive_output``: that one
        *cancels the application task* when its timeout expires, so the first
        "nothing else arrived" probe would silently tear the socket down and every
        later assertion in the test would fail for the wrong reason.
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self.instance.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            if self.instance.future.done():
                self.instance.future.result()
            if time.monotonic() >= deadline:
                raise AssertionError(f"No ASGI message arrived within {timeout}s.")
            await asyncio.sleep(0.01)

    async def receive_json_or_none(self, timeout: float = 0.5) -> dict | None:
        """Read one frame, or return ``None`` if the socket stays quiet.

        Used for assertions about *absence* — "no frame arrived" is a real claim,
        and one that must be bounded in time rather than assumed. Polling the
        output queue is what makes it safe: nothing is cancelled, so the
        communicator is still usable afterwards.
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                message = self.instance.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                if time.monotonic() >= deadline:
                    return None
                await asyncio.sleep(0.01)
                continue
            if message["type"] == "websocket.send":
                return json.loads(message["text"])
            return None

    async def receive_nothing(self, timeout: float = 0.5) -> bool:
        """Whether the socket produced no frame for ``timeout`` seconds."""
        return await self.receive_json_or_none(timeout=timeout) is None

    async def disconnect(self, code: int = 1000) -> None:
        await self.instance.send_input({"type": "websocket.disconnect", "code": code})
        await self.instance.wait()

    async def wait(self) -> None:
        await self.instance.wait()


async def drain(communicator: WebsocketCommunicator, *, until: int = 50) -> list[dict]:
    """Read every frame the application has already produced.

    Frames arrive in order and the test suite asserts on that order, so this reads
    until the socket goes quiet rather than taking a fixed count.
    """
    frames: list[dict] = []
    for _ in range(until):
        frame = await communicator.receive_json_or_none(timeout=0.2)
        if frame is None:
            break
        frames.append(frame)
    return frames


def frames_of_type(frames: list[dict], event_type: str) -> list[dict]:
    return [frame for frame in frames if frame.get("type") == event_type]


def last_frame(frames: list[dict], event_type: str) -> dict:
    matches = frames_of_type(frames, event_type)
    assert matches, f"No {event_type} frame in {[f.get('type') for f in frames]}"
    return matches[-1]


def channel_layer():
    """The active channel layer, for direct group-send assertions."""
    return get_channel_layer()


def _in_worker(function, args, kwargs):
    """Run ``function``, then close the connections this thread opened.

    ``thread_sensitive=True`` routes to asgiref's single shared worker thread,
    which lives for the whole test session. Django connections are thread-local,
    so anything the ORM opened there is never reclaimed: the pool never retires
    that thread, so nothing else can close its connection either. The session
    outlives the tests and the leftover session then blocks Django's teardown of
    the test database.

    Closing inside the worker is the only place this can be done — a
    ``connections.close_all()`` on the main thread cannot reach a connection
    owned by a different, still-running thread.
    """
    try:
        return function(*args, **kwargs)
    finally:
        connections.close_all()


async def store_call(function, /, *args, **kwargs):
    """Call a blocking store function without stalling the event loop.

    The consumers do their Redis and database work on worker threads; a test that
    called the store directly from an ``async def`` body would block the very loop
    that has to deliver the frame it is waiting on, and the assertion would fail
    for a scheduling reason rather than a behavioural one.
    """
    return await sync_to_async(_in_worker, thread_sensitive=True)(function, args, kwargs)


async def db_call(function, /, *args, **kwargs):
    """Call a blocking, ORM-backed domain service off the event loop.

    Tests drive the same ``mobility_service``/``dispatch_service`` entry points
    production does, so a socket test can assert that realtime *observes* the state
    machine rather than inventing its own transitions.
    """
    return await sync_to_async(_in_worker, thread_sensitive=True)(function, args, kwargs)


async def wait_for_frame(communicator, event_type: str, *, timeout: float = 5) -> dict:
    """Wait for one specific frame type, tolerating anything that arrives first.

    ``drain`` answers "what has the socket already produced"; this answers "wait
    until X happens". A live-broadcast assertion needs the second one — the frame
    arrives some time after the test triggers its cause, and how long that takes is
    a property of the worker threads, not of the test.
    """
    deadline = time.monotonic() + timeout
    seen: list[str] = []
    while True:
        remaining = deadline - time.monotonic()
        frame = await communicator.receive_json_or_none(timeout=min(0.2, max(0.02, remaining)))
        if frame is not None:
            if frame.get("type") == event_type:
                return frame
            seen.append(str(frame.get("type")))
            continue
        if time.monotonic() >= deadline:
            raise AssertionError(f"No {event_type} frame within {timeout}s; saw {seen}.")


async def wait_for_location(driver_id, *, timeout: float = 5):
    """Wait until a driver has a stored position, and return it.

    An accepted ``location.update`` produces no acknowledgement frame: the driver
    is not told that its own fix was stored, because it is the source rather than
    an audience. The observable effect is the state itself, so a test that wants to
    assert ingestion waits for that effect within a bound rather than assuming the
    worker thread has already run.
    """
    from apps.realtime.services.location_store import resolve_location_store

    deadline = time.monotonic() + timeout
    while True:
        store = resolve_location_store()
        record = await store_call(store.get_location, driver_id=driver_id)
        if record is not None:
            return record
        if time.monotonic() >= deadline:
            raise AssertionError(f"No stored position for {driver_id} within {timeout}s.")
        await asyncio.sleep(0.02)


def denial(close_message: dict) -> dict:
    """Decode a refused handshake into ``{"code": ..., "reason": ...}``.

    The authentication middleware refuses *before* accepting, so a client learns
    why from the close frame's reason rather than from an error frame it never got
    to read. Keeping the two apart in the helpers stops a test from asserting the
    wrong channel for a refusal.
    """
    assert close_message["type"] == "websocket.close", close_message
    reason = close_message.get("reason") or ""
    parsed = parse_qs(reason)
    assert parsed, f"Refusal carried no reason: {close_message!r}"
    return {
        "code": (parsed.get("code") or [""])[0],
        "reason": (parsed.get("reason") or [""])[0],
        "close_code": close_message.get("code"),
    }


async def handshake_refusal(communicator: WebsocketCommunicator) -> dict:
    """Assert the handshake was refused, and return the machine-readable reason."""
    result = await communicator.connect()
    await communicator.wait()
    return denial(result)
