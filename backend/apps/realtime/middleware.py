"""WebSocket authentication middleware.

Browsers cannot set headers on a WebSocket handshake, so the credential arrives
in the query string. This middleware turns that query string into
``scope["user"]``, and it is deliberately strict:

* **No credential, no socket.** An unauthenticated connection is refused before a
  consumer ever runs, so there is no code path in which an anonymous scope exists.
* **The scope is stamped, not trusted.** Identity is resolved here and consumers
  read it from the scope; a consumer never re-parses a token.
* **Failures are typed.** Expired, revoked, wrong-audience and malformed tokens
  produce distinct errors, because "reconnect with a fresh token" and "your
  account is suspended" call for different client behaviour.
* **Refusal happens during the handshake**, not afterwards, so a token never
  reaches a consumer that would have to know how to discard it.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlencode

from channels.db import database_sync_to_async

from .errors import (
    AccountNotEligibleError,
    AuthenticationError,
    RateLimitedError,
    RealtimeError,
)
from .logging import log_warning_event
from .services.auth import authenticate_token

# Query parameters a client may use to present a credential. A client that sends
# more than one is ambiguous by definition, and ambiguity in authentication input
# is a vulnerability rather than a convenience.
TOKEN_QUERY_PARAMS = ("token", "access_token")


def _headers(scope) -> dict[str, str]:
    return {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in scope.get("headers") or []
    }


def _peer_label(scope) -> str:
    return _headers(scope).get("x-forwarded-for", (scope.get("client") or ("unknown",))[0])


def _credential(scope) -> str | None:
    """Extract the bearer credential from the handshake.

    Query string first, because that is the only option in a browser handshake;
    the ``Authorization`` header is accepted for native clients that can set one.
    """
    query = parse_qs((scope.get("query_string") or b"").decode("utf-8", "ignore"))
    candidates = [value for name in TOKEN_QUERY_PARAMS for value in query.get(name, []) if value]
    if len(candidates) > 1:
        raise AuthenticationError("Present exactly one access token.")
    if candidates:
        return candidates[0]

    header = _headers(scope).get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[len("bearer ") :].strip() or None
    return None


def _rate_limit_for_connect() -> tuple[int, int]:
    from django.conf import settings

    limits = getattr(settings, "REALTIME_RATE_LIMITS", {})
    return limits.get("ws_connect", (0, 0))


@database_sync_to_async
def _authenticate(token: str):
    """Authenticate on a thread that owns a database connection.

    ``database_sync_to_async`` is not optional: resolving a user means ORM
    queries, and running those on the event loop thread would stall every other
    socket on this worker.
    """
    return authenticate_token(token)


@database_sync_to_async
def _may_stream(user_id: object) -> bool:
    from apps.accounts.auth import account_can_authenticate
    from apps.accounts.models import User

    return account_can_authenticate(
        # `user_id` is deliberately `object`: every caller passes the authenticated
        # primary key. Django coerces it; the stubs declare only `UUID | str`.
        User.objects.get(pk=user_id)  # type: ignore[misc]
    )


@database_sync_to_async
def _check_connect_rate_limit(user_id: object, limit: int, period: int) -> None:
    """Bound how often one identity can open a socket.

    Below a day of driver shifts this limit is never the binding constraint; it
    exists so a client stuck in a reconnect loop cannot turn a flaky network into
    a connection storm.
    """
    from apps.accounts.services.ratelimit import RateLimitExceeded, check_rate_limit

    if limit <= 0:
        return
    try:
        check_rate_limit("ws:connect", str(user_id), limit=limit, period=period)
    except RateLimitExceeded as exc:
        log_warning_event("realtime_rate_limited", scope="ws_connect", user_id=str(user_id))
        raise RateLimitedError() from exc


class JWTAuthMiddleware:
    """ASGI middleware that authenticates a WebSocket scope in place."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "websocket":
            await self.app(scope, receive, send)
            return

        scope = dict(scope)
        try:
            token = _credential(scope)
            if not token:
                raise AuthenticationError()
            context = await _authenticate(token)
            # Re-checked per handshake rather than trusted from the token: a token
            # minted before a suspension is still cryptographically valid, and a
            # live socket must not outlive the account state that permits it.
            if not await _may_stream(context.user.pk):
                raise AccountNotEligibleError()
            await _check_connect_rate_limit(context.user.pk, *_rate_limit_for_connect())
        except RealtimeError as error:
            await _deny(scope, send, error)
            return
        except Exception:
            # An unexpected failure is still an authentication failure from the
            # client's point of view; the detail goes to the log, not the socket.
            log_warning_event(
                "realtime_ws_auth_failed",
                path=scope.get("path"),
                client=_peer_label(scope),
            )
            await _deny(scope, send, AuthenticationError())
            return

        scope["user"] = context.user
        scope["token_payload"] = context.claims
        await self.app(scope, receive, send)


async def _deny(scope, send, error: RealtimeError) -> None:
    """Close the handshake with the error's code and a machine-readable reason.

    The reason is a short public sentence plus the error code — never an internal
    message — so an operations client can log why a connection was refused.
    """
    log_warning_event(
        "realtime_ws_rejected",
        code=str(error.code),
        path=scope.get("path"),
        client=_peer_label(scope),
    )
    await send(
        {
            "type": "websocket.close",
            "code": error.close_code,
            "reason": urlencode({"code": str(error.code), "reason": error.message}),
        }
    )


__all__ = ["JWTAuthMiddleware", "TOKEN_QUERY_PARAMS"]
