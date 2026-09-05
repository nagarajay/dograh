"""Create a client tenant from outside, without opening public signup.

Before this, the only way to bring a Dograh organization into existence was
``POST /auth/signup`` -- a route that has to be *publicly reachable* to serve
its actual purpose, and which therefore cannot be the provisioning path for a
platform whose tenants are created by an operator. Deployments closed the hole
by setting ``ENABLE_SIGNUP=false``, which also closed provisioning. This module
is the replacement: the same four writes signup performs, reached through a
platform credential instead of an open door.

An organization here is one client tenant, and the identity that owns it is a
*service* identity, not a person:

1. a Dograh user holding the service email, with a generated password nobody
   outside this process ever sees,
2. the organization, labelled with the client's display name and the
   provisioning system's own reference,
3. the membership linking the two, and the user's selected organization,
4. ``ensure_organization_bootstrapped``, so the tenant has model configuration
   and SIP connectivity before anyone asks it to place a call.

Idempotency is keyed on ``external_reference`` because that is the only
identifier the caller controls and can reproduce across a retry: an internal id
is not known before the first attempt, and the provider id is derived from a
user that a retry would create a second copy of. The reference carries a
partial unique index (``uq_organizations_external_reference``), so the database
enforces the same rule this module checks.
"""

import secrets
from dataclasses import dataclass

from loguru import logger

from api.constants import PLATFORM_PROVISIONING_API_KEY_NAME
from api.db import db_client
from api.db.models import APIKeyModel, OrganizationModel, UserModel
from api.services.organization_bootstrap import ensure_organization_bootstrapped
from api.utils.auth import hash_password

# The service account authenticates nothing interactively -- AVSIQ holds an
# organization API key, not this password -- so the password exists only
# because the local auth layer stores a hash for every account. It is generated
# here, hashed, and discarded.
_SERVICE_PASSWORD_BYTES = 48


class ProvisioningConflict(Exception):
    """The request contradicts an organization that already exists.

    Distinct from an ordinary failure: nothing was created, and no retry of the
    same request will succeed. Callers turn this into 409.
    """


class OrganizationNotFound(Exception):
    """The named organization does not exist."""


@dataclass(frozen=True)
class ProvisionedOrganization:
    organization: OrganizationModel
    service_user: UserModel
    #: False when this call matched an existing organization by external
    #: reference. The response is otherwise identical, which is what makes a
    #: retry safe to issue blindly.
    created: bool
    #: Whether the tenant is fully provisioned *now*. Bootstrap is best effort
    #: by design (a provider outage must not fail the call), self-heals on the
    #: organization's later authenticated requests, and is re-entered on every
    #: retry of this endpoint -- so False means "not yet", not "broken".
    bootstrapped: bool


async def provision_client_organization(
    *,
    display_name: str,
    external_reference: str,
    service_email: str,
) -> ProvisionedOrganization:
    """Create -- or re-report -- the tenant identified by ``external_reference``."""
    display_name = display_name.strip()
    external_reference = external_reference.strip()
    service_email = service_email.strip().lower()

    existing = await db_client.get_organization_by_external_reference(
        external_reference
    )
    if existing is not None:
        return await _report_existing(existing, display_name, service_email)

    # A service email already in use means the caller is provisioning a second
    # tenant onto one identity. Refused rather than reused: the account's
    # ``selected_organization_id`` can only name one organization, so honouring
    # it would silently move the first tenant's owner into the second tenant.
    if await db_client.get_user_by_email(service_email) is not None:
        raise ProvisioningConflict(
            "The service identity email is already registered under a different "
            "external reference."
        )

    service_user = await db_client.create_user_with_email(
        email=service_email,
        password_hash=hash_password(secrets.token_urlsafe(_SERVICE_PASSWORD_BYTES)),
    )

    (
        organization,
        was_created,
    ) = await db_client.get_or_create_organization_for_external_reference(
        org_provider_id=f"org_{service_user.provider_id}",
        user_id=service_user.id,
        display_name=display_name,
        external_reference=external_reference,
    )

    if not was_created:
        # Two concurrent first-provisions of the same client. The unique index
        # on external_reference let exactly one through, and this call lost, so
        # it answers like an ordinary retry rather than raising. The service
        # user this call created is left behind: it owns nothing and belongs to
        # no organization, and deleting a user row here would be a far more
        # dangerous write than leaving an inert one.
        logger.warning(
            "Concurrent provisioning of external reference {}; reporting the "
            "organization that won",
            external_reference,
        )
        return await _report_existing(organization, display_name, service_email)

    await db_client.add_user_to_organization(service_user.id, organization.id)
    await db_client.update_user_selected_organization(
        service_user.id, organization.id
    )
    service_user.selected_organization_id = organization.id

    bootstrapped = await ensure_organization_bootstrapped(
        organization.id,
        created_by=service_user.provider_id,
    )

    return ProvisionedOrganization(
        organization=organization,
        service_user=service_user,
        created=True,
        bootstrapped=bootstrapped,
    )


async def _report_existing(
    organization: OrganizationModel,
    display_name: str,
    service_email: str,
) -> ProvisionedOrganization:
    """Answer a retry, or refuse a request that contradicts the live tenant.

    An exact retry returns the organization unchanged. Anything else is a
    conflict: the reference identifies one client, and quietly renaming or
    re-pointing a tenant that is already placing calls is worse than making the
    caller reconcile.
    """
    if (organization.display_name or "") != display_name:
        raise ProvisioningConflict(
            f"External reference {organization.external_reference!r} is already "
            f"registered to organization {organization.id} under a different "
            "display name."
        )

    members = await db_client.get_organization_users(organization.id)
    service_user = next(
        (member for member in members if (member.email or "").lower() == service_email),
        None,
    )
    if service_user is None:
        raise ProvisioningConflict(
            f"External reference {organization.external_reference!r} is already "
            f"registered to organization {organization.id} under a different "
            "service identity."
        )

    # Re-entered on every retry on purpose: a first attempt whose bootstrap was
    # incomplete is finished by the retry, which is what makes retrying useful
    # rather than merely safe.
    bootstrapped = await ensure_organization_bootstrapped(
        organization.id,
        created_by=service_user.provider_id,
    )

    return ProvisionedOrganization(
        organization=organization,
        service_user=service_user,
        created=False,
        bootstrapped=bootstrapped,
    )


async def mint_organization_api_key(
    *, organization_id: int
) -> tuple[APIKeyModel, str, list[int]]:
    """Issue -- or re-issue -- the organization's provisioning API key.

    Safely repeatable, which the previous append-a-new-key version was not. A
    caller whose response is lost has no way to learn the key it was given, so
    it must retry; appending would leave the organization holding a live
    credential nobody has, once per lost response, with nothing to distinguish
    the abandoned keys from the real one.

    So the key has a reserved, deterministic name and every call *replaces* it:
    the previous key is archived and the new one issued in one transaction, and
    a partial unique index (``uq_api_keys_active_platform_provisioning``) makes
    that hold under concurrency too. After any number of retries, successful or
    not, the organization has at most one active provisioning key, and it is the
    one the last successful response returned. Earlier keys stop authenticating
    immediately -- ``validate_api_key`` matches only active, unarchived rows.

    The raw key is still returned exactly once; only its hash is stored. A
    caller that loses it retries, which is now a safe thing to do.

    Returns ``(api_key, raw_key, archived_ids)``.

    This is an ordinary tenant credential: it authenticates as its organization
    and reaches nothing outside it, including the endpoint that minted it. Only
    the act of minting one for an organization the caller does not belong to
    needs platform authority, which is why this exists apart from
    ``POST /user/api-keys``.

    ``created_by`` is required by ``_handle_api_key_auth`` -- a key with no
    owner is refused at authentication time -- so it is set to the
    organization's earliest member, which for a provisioned tenant is the
    service identity created alongside it.
    """
    organization = await db_client.get_organization_by_id(organization_id)
    if organization is None:
        raise OrganizationNotFound(f"Organization {organization_id} not found.")

    members = await db_client.get_organization_users(organization_id)
    if not members:
        raise ProvisioningConflict(
            f"Organization {organization_id} has no members, so a key minted "
            "for it could not authenticate."
        )

    return await db_client.replace_api_key_by_name(
        organization_id=organization_id,
        name=PLATFORM_PROVISIONING_API_KEY_NAME,
        created_by=members[0].id,
    )
