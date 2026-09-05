"""The server-to-server credential that provisions client organizations.

Three credentials reach this API and they are deliberately not
interchangeable:

``X-API-Key``
    An organization API key. Scoped to exactly one tenant by construction --
    :func:`api.services.auth.depends._handle_api_key_auth` pins the caller's
    ``selected_organization_id`` to the key's organization -- so it can never
    be the credential that creates a *new* tenant or mints a key for another
    one. It is rejected here outright, exactly as
    :func:`api.services.auth.depends.get_superuser` rejects it.

``Authorization``
    An interactive super-admin session. Held by a person, obtained through a
    browser sign-in, and already the credential for the read-only super-admin
    console. Provisioning runs unattended from AVSIQ, which has no browser and
    no session to refresh, so making it depend on that flow would either mean
    storing a human's password or weakening ``get_superuser``. Neither happens:
    this module adds a credential rather than relaxing that one.

``X-Platform-Admin-Key``
    This one. A single shared secret held by the provisioning system, checked
    with a constant-time comparison, granting exactly two endpoints and nothing
    else. It authenticates no user and resolves no ``UserModel``, so there is
    no tenant context for a bug to leak: every route guarded by it names the
    organization it acts on explicitly.

Unset (or too short to be worth anything) means provisioning is unavailable
rather than open. A deployment that never provisions from outside is the common
case and must not be the insecure one.
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
