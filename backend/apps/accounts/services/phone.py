"""Phone number normalization and validation.

ROVA is a Nigerian-first platform. Phone numbers are normalized to
international (E.164) format so that equivalent local forms resolve to a
single canonical representation. The default region can be changed later via
settings to support other countries.
"""

import phonenumbers
from django.conf import settings


class PhoneNumberError(ValueError):
    """Raised when a phone number is missing, invalid, or unsupported."""


def _default_region() -> str:
    return getattr(settings, "PHONE_DEFAULT_REGION", "NG")


def normalize_phone(raw: str) -> str:
    """Validate and normalize a phone number to E.164, or raise PhoneNumberError."""
    if not raw or not raw.strip():
        raise PhoneNumberError("Phone number is required.")

    candidate = raw.strip()
    try:
        parsed = phonenumbers.parse(candidate, _default_region())
    except phonenumbers.NumberParseException as exc:
        raise PhoneNumberError("Phone number is invalid.") from exc

    if not phonenumbers.is_valid_number(parsed):
        raise PhoneNumberError("Phone number is invalid.")

    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
