"""Error text that can be logged or returned without leaking secret input.

Pydantic's ``str(ValidationError)`` embeds ``input_value=...``, which for a model
configuration includes API keys and service-account JSON. Use these helpers
instead of formatting a validation error directly.
"""

from __future__ import annotations

from typing import Iterable

from pydantic import ValidationError

REDACTED = "[REDACTED]"


def summarize_validation_error(exc: ValidationError) -> list[dict[str, str]]:
    """Location, type and message of each error; never the offending input."""
    return [
        {
            "loc": ".".join(str(part) for part in error.get("loc", ())),
            "type": str(error.get("type", "")),
            "message": str(error.get("msg", "")),
        }
        for error in exc.errors(
            include_input=False, include_url=False, include_context=False
        )
    ]


def validation_error_text(exc: ValidationError) -> str:
    return "; ".join(
        f"{item['loc'] or 'config'}: {item['message']}"
        for item in summarize_validation_error(exc)
    )


def redact_text(text: str, secrets: Iterable[str]) -> str:
    """Defence in depth: strip any known secret value from free text."""
    for secret in secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, REDACTED)
    return text


def safe_exception_detail(exc: Exception):
    """HTTP-safe detail for an exception raised while handling a model configuration.

    Pydantic errors are summarised without their input. The configuration
    validator raises ``ValueError(list_of_status_dicts)``; that list is returned
    as-is because it contains only provider names and fixed messages.
    """
    if isinstance(exc, ValidationError):
        return validation_error_text(exc)
    if exc.args and isinstance(exc.args[0], list):
        return exc.args[0]
    return str(exc)
