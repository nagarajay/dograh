"""Authorization for platform-level endpoints: provisioning and the sample library.

Three kinds of credential can reach these endpoints, and they are deliberately
not interchangeable:

``X-API-Key``
    An organization API key. Scoped to exactly one tenant by construction --
    :func:`api.services.auth.depends._handle_api_key_auth` pins the caller's
    ``selected_organization_id`` to the key's organization -- so it can never
    be the credential that creates a *new* tenant, mints a key for another one
    or spends the platform's generation budget. It is rejected here outright,
    exactly as :func:`api.services.auth.depends.get_superuser` rejects it, and
    that check comes first, whatever else is presented.

``Authorization`` (interactive super-admin session)
    Held by a person, obtained through a browser sign-in. It is validated by
    :func:`api.services.auth.depends.get_superuser` and is what the super-admin
    console uses, so a signed-in super-admin may call every endpoint guarded by
    this dependency. :func:`get_superuser` itself is unchanged: this module
    adds a way in rather than relaxing it. If an ``Authorization`` header is
    present it alone decides the request: an invalid, expired or non-super-admin
    session is refused even when a valid ``X-Platform-Admin-Key`` accompanies
    it. There is no fallback from one credential to the other, so a proxy that
    injects a stale ``Authorization`` header into a server-to-server call gets
    a clear 401/403 instead of silently changing which credential applies.

``X-Platform-Admin-Key``
    A single shared secret held by the provisioning system (AVSIQ), checked
    with a constant-time comparison. It runs unattended and has no browser or
    session to refresh. It authenticates no user and resolves no ``UserModel``,
    so there is no tenant context for a bug to leak: every route guarded by
    this dependency names the organization it acts on explicitly. It grants
    the same set of endpoints as a super-admin session: tenant provisioning
    and API-key minting, bootstrap of a super-admin, and the platform-owned
    Gemini-TTS sample library. Treat it as a root credential.

Unset (or too short to be worth anything) means the key path is unavailable
rather than open. A deployment that never provisions from outside is the common
case and must not be the insecure one; the super-admin session path is
independent of the key and keeps working.
"""

import secrets
from typing import Annotated

from fastapi import Header, HTTPException, status

from api import constants

PLATFORM_ADMIN_HEADER = "X-Platform-Admin-Key"


def _configured_key() -> str | None:
    """Read the secret at call time, not at import time.

    ``api.constants`` caches environment variables when it is imported, which
    is fine for the process but makes the value impossible to substitute in a
    test without reaching into the module. Reading through the module object
    keeps that substitution honest: what the test patches is what the
    dependency reads.
    """
    key = constants.PLATFORM_ADMIN_API_KEY
    if not key or len(key) < constants.PLATFORM_ADMIN_API_KEY_MIN_LENGTH:
        return None
    return key


async def require_platform_admin(
    authorization: Annotated[str | None, Header()] = None,
    x_platform_admin_key: Annotated[
        str | None, Header(alias=PLATFORM_ADMIN_HEADER)
    ] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Authorize a platform provisioning call, or refuse it.

    Returns nothing. There is no principal to return: the credential belongs to
    the platform, not to a user or an organization, and every endpoint behind it
    takes its target organization from the request path or body.
    """
    # Checked before the shared secret so that presenting both headers cannot
    # launder an organization key into platform authority, and so the refusal
    # says which credential was wrong.
    if x_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Access denied. Organization API keys cannot be used for "
                "platform provisioning endpoints."
            ),
        )

    # Browser super-admins use their normal interactive session. The shared
    # platform key remains available for AVSIQ/server-to-server callers.
    if authorization:
        from api.services.auth.depends import get_superuser

        await get_superuser(authorization, None)
        return

    configured = _configured_key()
    if configured is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Platform provisioning is not configured on this deployment. "
                f"Set PLATFORM_ADMIN_API_KEY to at least "
                f"{constants.PLATFORM_ADMIN_API_KEY_MIN_LENGTH} characters."
            ),
        )

    presented = (x_platform_admin_key or "").strip()
    if not presented:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing {PLATFORM_ADMIN_HEADER} header.",
        )

    # compare_digest, not ==: a plain comparison leaks the length of the
    # matching prefix through timing, and this secret is guessed offline at
    # leisure by anyone who can reach the endpoint.
    if not secrets.compare_digest(presented, configured):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Invalid platform admin credential.",
        )
