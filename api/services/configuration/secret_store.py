"""Fail-closed encryption for provider credentials at rest.

Keys come from ``PROVIDER_CREDENTIAL_ENCRYPTION_KEYS``: a comma-separated list of
Fernet keys. The first encrypts; every key can decrypt, so a key rotation is
"prepend new key, re-encrypt, drop old key". If no valid key is configured,
every operation raises ``SecretStoreUnavailable``; there is no plaintext fallback.

Generate a key with:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

Nothing here logs, and no exception message contains secret material.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

KEYS_ENV = "PROVIDER_CREDENTIAL_ENCRYPTION_KEYS"

CREDENTIAL_KINDS: dict[str, tuple[str, ...]] = {
    "api_key": ("api_key",),
    "service_account_json": ("credentials",),
    "aws_iam": ("aws_access_key", "aws_secret_key", "aws_session_token"),
}
_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "api_key": ("api_key",),
    "service_account_json": ("credentials",),
    "aws_iam": ("aws_access_key", "aws_secret_key"),
}


class SecretStoreUnavailable(RuntimeError):
    """No usable encryption key; credentials cannot be stored or read."""


class SecretStoreCorrupt(RuntimeError):
    """A stored credential failed decryption or its integrity binding."""


def _load_keys() -> list[Fernet]:
    raw = os.getenv(KEYS_ENV, "")
    keys = [part.strip() for part in raw.split(",") if part.strip()]
    if not keys:
        raise SecretStoreUnavailable(
            f"{KEYS_ENV} is not set; refusing to store or read provider credentials"
        )
    try:
        return [Fernet(key.encode()) for key in keys]
    except (ValueError, TypeError):
        # Do not echo the value: it may be a malformed but real key.
        raise SecretStoreUnavailable(
            f"{KEYS_ENV} contains an invalid Fernet key"
        ) from None


def key_id_for(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def primary_key_id() -> str:
    raw = os.getenv(KEYS_ENV, "")
    first = next((p.strip() for p in raw.split(",") if p.strip()), "")
    if not first:
        raise SecretStoreUnavailable(f"{KEYS_ENV} is not set")
    return key_id_for(first)


def validate_payload(kind: str, payload: dict[str, Any]) -> None:
    """Shape check only. Messages name fields, never values."""
    if kind not in CREDENTIAL_KINDS:
        raise ValueError(
            f"unknown credential kind; expected one of {sorted(CREDENTIAL_KINDS)}"
        )
    allowed = set(CREDENTIAL_KINDS[kind])
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unexpected fields for kind {kind}: {', '.join(unknown)}")
    for field in _REQUIRED_FIELDS[kind]:
        _check_secret_value(field, payload.get(field), kind)
    for field in allowed - set(_REQUIRED_FIELDS[kind]):
        if payload.get(field) is not None:
            _check_secret_value(field, payload[field], kind)


def _check_secret_value(field: str, value: Any, kind: str) -> None:
    values = value if isinstance(value, list) and field == "api_key" else [value]
    if not values:
        raise ValueError(f"{field} is required for kind {kind}")
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field} is required for kind {kind}")
        if "***" in item:
            raise ValueError(f"{field} looks like a masked placeholder")


def encrypt_credential(
    *, organization_id: int, credential_ref: str, version: int, payload: dict[str, Any]
) -> tuple[str, str]:
    """Return ``(ciphertext, key_id)``. Binds the token to org/ref/version."""
    fernet = MultiFernet(_load_keys())
    envelope = {"o": organization_id, "r": credential_ref, "v": version, "p": payload}
    token = fernet.encrypt(json.dumps(envelope, separators=(",", ":")).encode())
    return token.decode(), primary_key_id()


def decrypt_credential(
    *, organization_id: int, credential_ref: str, version: int, ciphertext: str
) -> dict[str, Any]:
    fernet = MultiFernet(_load_keys())
    try:
        envelope = json.loads(fernet.decrypt(ciphertext.encode()))
    except (InvalidToken, ValueError):
        raise SecretStoreCorrupt(
            "stored credential could not be decrypted with the configured keys"
        ) from None
    if (
        envelope.get("o") != organization_id
        or envelope.get("r") != credential_ref
        or envelope.get("v") != version
    ):
        raise SecretStoreCorrupt("stored credential does not match its reference")
    return envelope["p"]


def reencrypt(ciphertext: str) -> tuple[str, str]:
    """Re-encrypt under the primary key (for key rotation)."""
    fernet = MultiFernet(_load_keys())
    try:
        return fernet.rotate(ciphertext.encode()).decode(), primary_key_id()
    except InvalidToken:
        raise SecretStoreCorrupt(
            "stored credential could not be decrypted with the configured keys"
        ) from None
