"""Create the platform's first super-admin, with no browser and no signup open.

``POST /auth/signup`` never sets ``is_superuser`` (see ``api/db/user_client.py``)
and no other route can raise it -- the only prior path was the operator running
``scripts/bootstrap_superadmin.py`` by hand against the database, which needs
shell access to the deployment. A production install with ``ENABLE_SIGNUP``
closed and no operator shell (a managed container, a fresh install with only
the platform credential) had no way to seat its first super-admin at all.

This closes that gap the same way ``services/superuser/provisioning.py``
closes the signup-is-closed gap for client tenants: a platform-credentialed
endpoint that performs, directly, the one write ordinary signup would never be
allowed to make.

Guarded to run exactly once: refused the moment any super-admin exists,
regardless of which email is named. A fresh row is created with no
organization and no membership -- a super-admin is not a client, and holds no
``selected_organization_id`` (see ``scripts/bootstrap_superadmin.py``'s
docstring for why). If the target email already belongs to an existing
(non-superuser) user, that account is promoted in place instead of creating a
second identity -- the same "existing user only" shape
``scripts/bootstrap_superadmin.py`` enforces, reached here over HTTP instead of
a database shell.

The "does one exist" check and the create-or-promote write are performed
atomically by ``db_client.bootstrap_first_superadmin`` (see its docstring in
``api/db/user_client.py``), so two simultaneous first-bootstrap requests
cannot both pass the check and each mint a super-admin.
"""

from dataclasses import dataclass

from api.db import db_client
from api.db.models import UserModel
from api.db.user_client import SuperadminAlreadyExists
from api.services.superuser.provisioning import ProvisioningConflict
from api.utils.auth import hash_password


@dataclass(frozen=True)
class BootstrappedSuperadmin:
    user: UserModel
    #: False when this call promoted an existing user instead of creating one.
    created: bool


async def bootstrap_platform_superadmin(
    *, email: str, password: str
) -> BootstrappedSuperadmin:
    """Create -- or promote -- the platform's one and only first super-admin.

    Raises :class:`ProvisioningConflict` if a super-admin already exists.
    """
    email = email.strip().lower()

    try:
        user, created = await db_client.bootstrap_first_superadmin(
            email=email, password_hash=hash_password(password)
        )
    except SuperadminAlreadyExists:
        raise ProvisioningConflict(
            "A super-admin already exists on this deployment. Bootstrap is a "
            "one-time operation; use the interactive super-admin console or "
            "scripts/bootstrap_superadmin.py for further changes."
        )

    return BootstrappedSuperadmin(user=user, created=created)
