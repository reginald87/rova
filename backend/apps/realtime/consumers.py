"""WebSocket consumers — the three realtime endpoints.

One base class holds everything the three endpoints share (frame decoding, size
limits, error replies, the invalid-message counter, group cleanup) and each
consumer adds only what is specific to its authorization boundary:

* :class:`DriverConsumer`      — ``/ws/v1/driver/``: the driver's own offers,
  assignment and availability notifications. It is the only endpoint that accepts
  ``location.update``, because it is the only one whose identity is a driver.
* :class:`TripConsumer`        — ``/ws/v1/trips/{trip_id}/``: one ride, for its
  rider, its assigned driver and privileged staff.
* :class:`OperationsConsumer`  — ``/ws/v1/operations/``: the staff feed.

Two invariants hold across all three:

* **The client never chooses its audience.** Group membership is derived from the
  authenticated scope and authoritative records; a frame can influence which
  groups are *published to*, never which groups a socket has joined.
* **Receiving is idempotent, sending is not required to be.** A dropped frame is
  recovered by reconnecting and taking a fresh snapshot, so no endpoint keeps a
  buffer, an offset or an acknowledgement scheme.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, timedelta
from typing import Any

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from django.utils import timezone

from .choices import (
    WS_CLOSE_SERVER_ERROR,
    ClientMessageType,
    RealtimeEventType,
    TripViewerRole,
)
from .errors import (
    AuthorizationError,
    InvalidMessageError,
    MessageValidationError,
    PayloadTooLargeError,
    RealtimeError,
    UnknownMessageTypeError,
)
from .logging import log_event, log_warning_event
from .schemas import parse_location_update, parse_ping
from .services.auth import (
    coerce_trip_id,
    resolve_driver,
    trip_viewer_role,
)
from .services.events import EVENT_ENVELOPE_VERSION, server_now_isoformat
from .services.groups import (
    driver_group,
    operations_group,
    operations_role_for,
    trip_group,
)
from .services.location_ingest import (
    assert_trip_scope,
    ingest_driver_location,
    resolve_tracked_trip,
)
from .services.snapshot import (
    build_driver_snapshot,
    build_operations_snapshot,
    build_trip_snapshot,
    location_policy,
)

# How long a cached "which trip is this driver on" view is trusted before it is
# re-read from the database. The TTL is a backstop for events that were missed
# (the driver was offline when their trip was assigned); the normal path is that
# a trip event invalidates the cache immediately.
ACTIVE_TRIP_CACHE_TTL = timedelta(seconds=10)

# Frames that count against the per-connection invalid-message budget. A client
# stuck in a bad request loop is disconnected rather than served forever.
_COUNTED_ERRORS = (InvalidMessageError, MessageValidationError, PayloadTooLargeError)


class RealtimeConsumer(AsyncWebsocketConsumer):
    """Shared frame handling and group lifecycle."""

    async def connect(self) -> None:
        # Set before ``accept`` so a failure in a subclass still has the
        # attributes ``disconnect`` relies on.
        self.groups_joined: list[str] = []
        self._invalid_messages = 0
        self.user = self.scope.get("user")
        await self.accept()
        log_event(
            "realtime_ws_connected",
            consumer=type(self).__name__,
            channel_name=self.channel_name,
            user_id=str(getattr(self.user, "pk", "")),
        )

    async def disconnect(self, code) -> None:
        for group in getattr(self, "groups_joined", []):
            await self.channel_layer.group_discard(group, self.channel_name)
        log_event(
            "realtime_ws_disconnected",
            consumer=type(self).__name__,
            channel_name=self.channel_name,
            code=code,
        )

    # --- inbound -------------------------------------------------------------

    async def receive(self, text_data=None, bytes_data=None) -> None:
        """Decode, dispatch, and never let one bad frame kill the socket."""
        try:
            payload = self._decode(text_data, bytes_data)
            handler = self._handler_for(payload["type"])
            if handler is None:
                raise UnknownMessageTypeError(
                    f"'{payload['type']}' is not a supported message type."
                )
            await handler(payload)
        except RealtimeError as error:
            await self._reply_error(error)
        except Exception:
            # Unhandled here means an internal failure. It is logged with the
            # channel and reported to the client as an opaque server error rather
            # than as a traceback.
            log_warning_event(
                "realtime_ws_handler_failed",
                consumer=type(self).__name__,
                channel_name=self.channel_name,
            )
            await self.close(code=WS_CLOSE_SERVER_ERROR)

    def _decode(self, text_data, bytes_data) -> dict[str, Any]:
        """Turn a raw frame into a message dictionary, or raise."""
        if text_data is None:
            raise InvalidMessageError("Binary frames are not supported.")
        limit = int(getattr(settings, "REALTIME_MAX_MESSAGE_BYTES", 4096))
        if len(text_data.encode("utf-8")) > limit:
            raise PayloadTooLargeError(f"The frame exceeds the {limit} byte limit.")
        try:
            payload = json.loads(text_data)
        except json.JSONDecodeError as exc:
            raise InvalidMessageError("The frame is not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise InvalidMessageError("The frame must be a JSON object.")
        message_type = payload.get("type")
        if not isinstance(message_type, str) or not message_type:
            raise InvalidMessageError("The frame has no usable 'type'.")
        return payload

    def _handler_for(self, message_type: str):
        if message_type == ClientMessageType.PING:
            return self._handle_ping
        if message_type == ClientMessageType.LOCATION_UPDATE:
            return self._handle_location_update
        return None

    async def _handle_ping(self, payload: dict[str, Any]) -> None:
        message = parse_ping(payload)
        pong = {
            "event_id": str(uuid.uuid4()),
            "type": RealtimeEventType.PONG,
            "version": EVENT_ENVELOPE_VERSION,
            "timestamp": server_now_isoformat(),
        }
        if message.nonce is not None:
            pong["nonce"] = message.nonce
        await self._send(pong)

    async def _handle_location_update(self, payload: dict[str, Any]) -> None:
        """Implemented only by the driver endpoint.

        The base implementation parses the frame — so the client gets a precise
        error — and then refuses. A rider or operations socket sending telemetry
        is not a validation quirk: it means the client believes it may choose its
        own identity.
        """
        parse_location_update(payload)
        raise RealtimeError("This endpoint does not accept location updates.")

    # --- outbound ------------------------------------------------------------

    async def _send(self, frame: dict[str, Any]) -> None:
        await self.send(text_data=json.dumps(frame, default=str))

    async def _reply_error(self, error: RealtimeError) -> None:
        """Answer a bad frame, then close only if the error warrants it."""
        limit, _period = _invalid_limit()
        if isinstance(error, _COUNTED_ERRORS):
            self._invalid_messages += 1

        log_warning_event(
            "realtime_ws_message_rejected",
            consumer=type(self).__name__,
            code=str(error.code),
            invalid_messages=self._invalid_messages,
        )
        await self._send(error.as_payload())

        if self._invalid_messages >= limit or error.fatal:
            await self.close(code=error.close_code)

    # --- channel layer plumbing ---------------------------------------------

    async def realtime_frame(self, event) -> None:
        """Handler for every ``realtime.frame`` broadcast."""
        await self._send(event["frame"])

    # --- helpers -------------------------------------------------------------

    async def _join(self, group: str) -> None:
        await self.channel_layer.group_add(group, self.channel_name)
        self.groups_joined.append(group)

    async def _refuse(self, error: RealtimeError) -> None:
        """Send the error frame and close, for a failure during ``connect``.

        The handshake is already accepted by this point, so the client needs the
        reason as a frame rather than a close code it has to guess at.
        """
        await self._send(error.as_payload())
        await self.close(code=error.close_code)

    @database_sync_to_async
    def _active_trip_ids(self, *, limit: int = 200) -> list[str]:
        """Trip ids currently in flight, for the operations snapshot."""
        from apps.mobility.models import Trip

        from .services.snapshot import LIVE_TRIP_STATUSES

        return [
            str(trip_id)
            for trip_id in Trip.objects.filter(status__in=LIVE_TRIP_STATUSES).values_list(
                "id", flat=True
            )[:limit]
        ]


class DriverConsumer(RealtimeConsumer):
    """``/ws/v1/driver/`` — the driver's own realtime channel.

    Subscribes to its own driver group (offers, assignment, availability) and is
    the source of the telemetry stream. Group membership is derived from the
    authenticated user at connect time; no message can move a driver into another
    driver's group.
    """

    async def connect(self) -> None:
        await super().connect()
        try:
            driver = await _resolve_driver(self.scope["user"])
            active_trip = await _tracked_trip(driver.pk)
        except RealtimeError as error:
            await self._refuse(error)
            return

        self.driver = driver
        self.driver_id = driver.pk
        self._active_trip = active_trip
        self._active_trip_checked_at = timezone.now()

        await self._join(driver_group(self.driver_id))
        await self._send(await _driver_snapshot(driver, active_trip))
        await self._send(location_policy())

    async def _handle_location_update(self, payload: dict[str, Any]) -> None:
        message = parse_location_update(payload)
        trip = await self._current_trip()
        assert_trip_scope(message.trip_id, trip)
        await _ingest(
            driver_id=self.driver_id,
            message=message,
            trip=trip,
        )

    async def _current_trip(self):
        """The driver's active trip, from cache when it is still fresh.

        Resolving the active trip per fix would mean a database query every two
        seconds for every streaming driver. The cached value is also invalidated
        by trip events on this same socket, so the normal case costs nothing.

        A cached *absence* is honoured for the same TTL as a cached trip: "no
        active trip" is by far the common case (every driver between rides), and
        treating only a hit as cacheable would query the database on every single
        fix for the whole idle fleet.
        """
        if timezone.now() - self._active_trip_checked_at < ACTIVE_TRIP_CACHE_TTL:
            return self._active_trip
        self._active_trip = await _tracked_trip(self.driver_id)
        self._active_trip_checked_at = timezone.now()
        return self._active_trip

    async def realtime_frame(self, event) -> None:
        """Forward a broadcast, invalidating the active-trip cache when it moves.

        This is what makes the cache correct rather than merely fast: the driver
        hears about their own assignment and cancellation on the socket they are
        already streaming positions over.
        """
        frame = event.get("frame") or {}
        if frame.get("type") in _TRIP_LIFECYCLE_EVENTS:
            self._active_trip = None
            self._active_trip_checked_at = timezone.now()
        await self._send(frame)


class TripConsumer(RealtimeConsumer):
    """``/ws/v1/trips/{trip_id}/`` — one ride's live channel."""

    async def connect(self) -> None:
        await super().connect()
        try:
            trip_id = coerce_trip_id(
                (self.scope.get("url_route", {}).get("kwargs") or {}).get("trip_id")
            )
            trip, viewer_role, operations_role = await self._authorize(trip_id)
        except RealtimeError as error:
            # ``coerce_trip_id`` is inside the guard on purpose: a malformed path
            # segment is a client mistake, and an unhandled one would take the whole
            # application task down instead of answering with ``NOT_FOUND``.
            await self._refuse(error)
            return

        self.trip = trip
        self.viewer_role = viewer_role
        await self._join(trip_group(trip.pk))
        if operations_role:
            await self._join(operations_group(operations_role))
        await self._send(await _trip_snapshot(trip))

    @database_sync_to_async
    def _authorize(self, trip_id):
        """Resolve the trip and confirm the caller may watch it.

        ``trip_viewer_role`` returns ``NOT_FOUND`` for a trip the caller may not
        see, so an operations client, the rider and the assigned driver each get
        the same channel with the same events and nobody learns of a trip's
        existence by probing the URL.
        """
        from apps.mobility.models import Trip

        user = self.scope["user"]
        trip = Trip.objects.filter(pk=trip_id).first()
        role = trip_viewer_role(user, trip)
        operations_role = operations_role_for(user) if role == TripViewerRole.OPERATIONS else None
        return trip, role, operations_role


class OperationsConsumer(RealtimeConsumer):
    """``/ws/v1/operations/`` — the privileged staff feed.

    Role-gated at connect: support, safety, compliance and finance have no Phase 10
    stream, and the gate is an intersection with the configured allow-list rather
    than "not a rider".
    """

    async def connect(self) -> None:
        await super().connect()
        role = await _operations_role(self.scope["user"])
        if role is None:
            await self._refuse(AuthorizationError("Your role may not use this stream."))
            return

        self.role = role
        await self._join(operations_group(role))
        active_trips = await self._active_trip_ids()
        await self._send(build_operations_snapshot(role=role, active_trips=active_trips))


# --- module helpers --------------------------------------------------------

# Every entry point below touches the ORM, so each is wrapped rather than awaited
# directly: running a query on the event loop thread would stall every other
# socket this worker is serving.
_resolve_driver = database_sync_to_async(resolve_driver)
_tracked_trip = database_sync_to_async(resolve_tracked_trip)
_operations_role = database_sync_to_async(operations_role_for)


@database_sync_to_async
def _ingest(*, driver_id, message, trip):
    """Validate, store and fan out one accepted fix on a worker thread."""
    return ingest_driver_location(driver_id=driver_id, message=message, trip=trip)


# Trip events that mean "your active trip just changed".
_TRIP_LIFECYCLE_EVENTS = frozenset(
    {
        str(RealtimeEventType.TRIP_ASSIGNED),
        str(RealtimeEventType.TRIP_STATUS),
        str(RealtimeEventType.TRIP_CANCELLED),
        str(RealtimeEventType.TRIP_COMPLETED),
    }
)


@database_sync_to_async
def _trip_snapshot(trip) -> dict[str, Any]:
    """Build a trip's reconnect snapshot on a worker thread.

    The snapshot re-reads the trip with its relations (see
    :func:`apps.realtime.services.snapshot.build_trip_snapshot`), so it is a
    database query and must not run on the event loop — one connect would otherwise
    stall every other socket this worker is serving.
    """
    return build_trip_snapshot(trip)


@database_sync_to_async
def _driver_snapshot(driver, active_trip) -> dict[str, Any]:
    """Build a driver's reconnect snapshot on a worker thread.

    Includes the offers that are currently live for the driver: an offer created
    while the socket was down is exactly the event a reconnecting driver would
    otherwise never see.
    """
    from apps.dispatch.models import DriverOffer

    open_offers = [
        {
            "offer_id": str(offer.pk),
            "trip_id": str(offer.trip_id),
            "ride_request_id": str(offer.ride_request_id),
            "expires_at": _iso(offer.expires_at),
            "distance_to_pickup_meters": offer.distance_to_pickup_meters,
        }
        for offer in DriverOffer.objects.filter(
            driver_id=driver.pk,
            status="PENDING",
            expires_at__gt=timezone.now(),
        ).only("id", "trip_id", "ride_request_id", "expires_at", "distance_to_pickup_meters")
    ]
    return build_driver_snapshot(
        driver=driver,
        active_trip=active_trip,
        open_offers=open_offers,
    )


def _iso(value) -> str | None:
    """Render an aware datetime the way every frame's ``timestamp`` is rendered.

    ``datetime.UTC`` rather than ``django.utils.timezone.utc``: the latter was a
    re-export of ``datetime.timezone.utc`` that Django removed in 5.0, so reaching
    for it fails at the first driver who reconnects with an open offer — a path
    only taken when a live offer exists, which is why it can sit unnoticed behind
    tests that never hold one.
    """
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _invalid_limit() -> tuple[int, int]:
    limits = getattr(settings, "REALTIME_RATE_LIMITS", {})
    return limits.get("ws_invalid", (30, 300))


__all__ = ["DriverConsumer", "OperationsConsumer", "TripConsumer"]
