"""Realtime domain errors.

Each error carries the ``RealtimeErrorCode`` a client should see. The mapping is
one-way on purpose: a stack trace, a database column name or an internal message
never reaches a WebSocket frame — only the code and a short human-readable
sentence do.
"""

from __future__ import annotations

from .choices import (
    WS_CLOSE_FORBIDDEN,
    WS_CLOSE_NOT_FOUND,
    WS_CLOSE_RATE_LIMITED,
    WS_CLOSE_SERVER_ERROR,
    WS_CLOSE_TOO_LARGE,
    WS_CLOSE_UNAUTHORIZED,
    RealtimeErrorCode,
)


class RealtimeError(Exception):
    """Base class for realtime failures that are safe to report to a client.

    ``fatal`` decides whether the socket survives. It is false by default because
    the common case — a malformed frame, a rejected coordinate, a message that
    arrived too quickly — is a message-level fault: the client is told, and the
    connection stays up. Only an authentication/authorization failure or an
    explicit policy decision closes it.
    """

    code: RealtimeErrorCode | str = RealtimeErrorCode.SERVER_ERROR
    default_message = "The realtime service could not handle this request."
    close_code = WS_CLOSE_SERVER_ERROR
    fatal = False

    def __init__(self, message: str | None = None, *, code: str | None = None) -> None:
        super().__init__(message or self.default_message)
        self.message = message or self.default_message
        if code is not None:
            self.code = code

    def as_payload(self) -> dict:
        """Client-facing ``error`` frame. Deliberately excludes tracebacks."""
        return {"type": "error", "code": str(self.code), "message": self.message}


class AuthenticationError(RealtimeError):
    """No usable credential was presented on the socket."""

    code = RealtimeErrorCode.UNAUTHORIZED
    default_message = "Authentication is required."
    close_code = WS_CLOSE_UNAUTHORIZED
    fatal = True


class AccountNotEligibleError(RealtimeError):
    """The account authenticated but may not use realtime features.

    A user suspended after the token was issued must not keep a live socket.
    """

    code = RealtimeErrorCode.ACCOUNT_NOT_ELIGIBLE
    default_message = "This account may not use realtime features."
    close_code = WS_CLOSE_FORBIDDEN
    fatal = True


class AuthorizationError(RealtimeError):
    """Authenticated, but not a participant of the requested channel."""

    code = RealtimeErrorCode.FORBIDDEN
    default_message = "You are not authorized to access this channel."
    close_code = WS_CLOSE_FORBIDDEN
    fatal = True


class DriverProfileNotFoundError(RealtimeError):
    """The account is a driver, but has no driver profile yet.

    Distinct from :class:`AuthorizationError` because the remedy differs: this
    account needs onboarding, and a client that is told "forbidden" cannot tell
    whether to retry, refresh, or send the user to the wrong screen.
    """

    code = RealtimeErrorCode.DRIVER_PROFILE_NOT_FOUND
    default_message = "This account has no driver profile yet."
    close_code = WS_CLOSE_FORBIDDEN
    fatal = True


class ResourceNotFoundError(RealtimeError):
    """The addressed trip/record does not exist.

    Raised instead of ``AuthorizationError`` when the caller cannot be told
    whether the resource exists: a missing trip and a trip owned by somebody else
    must be indistinguishable from outside.
    """

    code = RealtimeErrorCode.NOT_FOUND
    default_message = "Not found."
    close_code = WS_CLOSE_NOT_FOUND
    fatal = True


class InvalidMessageError(RealtimeError):
    """The frame is not JSON, is not an object, or has no usable ``type``."""

    code = RealtimeErrorCode.INVALID_MESSAGE
    default_message = "The message is not a valid ROVA message."
    # Not fatal on its own: the connection counter decides. A client that has sent
    # several bad frames in a row is disconnected by the consumer, not by the
    # first typo.
    fatal = False


class UnknownMessageTypeError(InvalidMessageError):
    """The ``type`` is syntactically fine but not part of the contract."""

    code = RealtimeErrorCode.UNKNOWN_MESSAGE_TYPE
    default_message = "Unknown message type."
    fatal = False


class MessageValidationError(RealtimeError):
    """A known message type failed schema validation."""

    code = RealtimeErrorCode.VALIDATION_ERROR
    default_message = "The message payload is invalid."
    fatal = False


class IdentityMismatchError(RealtimeError):
    """The message tried to speak for somebody else.

    A security event, not a validation quirk: it means the client believed it
    could choose its own identity.
    """

    code = RealtimeErrorCode.IDENTITY_MISMATCH
    default_message = "The message does not match the authenticated identity."
    close_code = WS_CLOSE_FORBIDDEN
    fatal = True


class TripScopeMismatchError(RealtimeError):
    """The message claimed a trip the driver is not currently driving.

    Not fatal: a driver app that still believes it is on a trip it has just
    finished is a normal race, and the fix is a fresh snapshot — not a dropped
    socket.
    """

    code = RealtimeErrorCode.TRIP_NOT_ASSIGNED
    default_message = "You are not driving this trip."
    fatal = False


class PayloadTooLargeError(RealtimeError):
    code = RealtimeErrorCode.PAYLOAD_TOO_LARGE
    default_message = "The message is too large."
    close_code = WS_CLOSE_TOO_LARGE
    fatal = False


class RateLimitedError(RealtimeError):
    """Too many messages. The connection may survive; the message does not."""

    code = RealtimeErrorCode.RATE_LIMITED
    default_message = "Too many messages. Slow down."
    close_code = WS_CLOSE_RATE_LIMITED
    fatal = False
