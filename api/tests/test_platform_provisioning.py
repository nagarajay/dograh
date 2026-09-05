"""Provisioning a client tenant from outside, with signup closed.

The arrangement under test has three parts, and each of them is load-bearing:

- ``ENABLE_SIGNUP=false``. Public signup is the hole this replaces, so the
  tests start by pinning that it stays shut.
- ``X-Platform-Admin-Key``. A server-to-server secret that grants exactly the
  two provisioning endpoints. Not an organization API key -- those are
  tenant-scoped and are refused outright -- and not a super-admin session,
  which is interactive and is left exactly as strict as it was.
- Idempotency on ``external_reference``. The provisioning system retries; a
  retry must return the client it already has, and a request that contradicts
  that client must be refused rather than applied.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import constants
from api.routes import superuser as superuser_routes
from api.services.auth.platform_admin import (
    PLATFORM_ADMIN_HEADER,
    require_platform_admin,
)
from api.services.superuser import provisioning

# Long enough to clear the configured minimum; the point of the minimum is that
# a short secret protecting tenant creation is not a secret.
PLATFORM_KEY = "platform-admin-secret-key-with-enough-entropy"


@pytest.fixture
def platform_key(monkeypatch):
    """Configure the deployment's platform credential for one test."""
    monkeypatch.setattr(constants, "PLATFORM_ADMIN_API_KEY", PLATFORM_KEY)
    return PLATFORM_KEY


@pytest.fixture
def route_client(monkeypatch):
    """A client for the real routes, with the provisioning service stubbed.

    Route tests are about status codes, credentials and response shape; the
    service's own behaviour is exercised against a real database further down.
    """
    app = FastAPI()
    app.include_router(superuser_routes.router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def no_bootstrap(monkeypatch):
    """Stub managed provisioning: it reaches MPS and Cloudonix for real."""
    bootstrap = AsyncMock(return_value=True)
    monkeypatch.setattr(provisioning, "ensure_organization_bootstrapped", bootstrap)
    return bootstrap


def _payload(**overrides) -> dict:
    return {
        "display_name": "Northwind",
        "external_reference": "avsiq-client-0001",
        "service_email": "svc-northwind@example.com",
        **overrides,
    }


# ---------------------------------------------------------------------------
# 1. Public signup stays closed
# ---------------------------------------------------------------------------


def test_signup_is_refused_when_disabled(monkeypatch):
    """The endpoint this work replaces must not be the way in."""
    from api.routes import auth as auth_routes

    monkeypatch.setattr(auth_routes, "ENABLE_SIGNUP", False)
    created = AsyncMock()
    monkeypatch.setattr(auth_routes.db_client, "create_user_with_email", created)

    app = FastAPI()
    app.include_router(auth_routes.router)
    response = TestClient(app).post(
        "/auth/signup",
        json={"email": "someone@example.com", "password": "hunter2hunter2"},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Signup is disabled"}
    created.assert_not_awaited()


def test_provisioning_works_while_signup_is_disabled(
    monkeypatch, platform_key, route_client
):
    """Closing signup must not close provisioning -- that was the whole bug."""
    from api.routes import auth as auth_routes

    monkeypatch.setattr(auth_routes, "ENABLE_SIGNUP", False)
    monkeypatch.setattr(
        superuser_routes,
        "provision_client_organization",
        AsyncMock(
            return_value=provisioning.ProvisionedOrganization(
                organization=SimpleNamespace(
                    id=7,
                    provider_id="org_oss_1",
                    display_name="Northwind",
                    external_reference="avsiq-client-0001",
                ),
                service_user=SimpleNamespace(
                    id=3,
                    email="svc-northwind@example.com",
                    provider_id="oss_1",
                ),
                created=True,
                bootstrapped=True,
            )
        ),
    )

    response = route_client.post(
        "/superuser/organizations",
        json=_payload(),
        headers={PLATFORM_ADMIN_HEADER: PLATFORM_KEY},
    )

    assert response.status_code == 200
    assert response.json()["organization_id"] == 7
    assert response.json()["created"] is True


# ---------------------------------------------------------------------------
# 2. The platform credential
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_key_is_accepted(platform_key):
    assert (
        await require_platform_admin(
            x_platform_admin_key=PLATFORM_KEY, x_api_key=None
        )
        is None
    )


@pytest.mark.parametrize(
    "presented, expected_status",
    [
        (None, 401),
        ("", 401),
        ("   ", 401),
        ("wrong-key-of-a-perfectly-plausible-length", 403),
        # A prefix of the real key: rejected like any other wrong value.
        (PLATFORM_KEY[:-1], 403),
    ],
)
def test_bad_or_missing_credentials_are_refused(
    platform_key, route_client, presented, expected_status
):
    headers = {} if presented is None else {PLATFORM_ADMIN_HEADER: presented}
    response = route_client.post(
        "/superuser/organizations", json=_payload(), headers=headers
    )

    assert response.status_code == expected_status


def test_unconfigured_deployment_refuses_rather_than_opens(monkeypatch, route_client):
    """No configured secret must mean closed, never 'anything matches'."""
    monkeypatch.setattr(constants, "PLATFORM_ADMIN_API_KEY", None)

    response = route_client.post(
        "/superuser/organizations",
        json=_payload(),
        headers={PLATFORM_ADMIN_HEADER: "anything"},
    )

    assert response.status_code == 503


def test_a_short_key_counts_as_unconfigured(monkeypatch, route_client):
    """A secret too short to resist guessing is not a secret."""
    monkeypatch.setattr(constants, "PLATFORM_ADMIN_API_KEY", "short")

    response = route_client.post(
        "/superuser/organizations",
        json=_payload(),
        headers={PLATFORM_ADMIN_HEADER: "short"},
    )

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_interactive_superuser_auth_is_untouched(monkeypatch):
    """The pre-existing session gate keeps refusing API keys, as before."""
    from api.services.auth import depends as auth_depends

    get_user = AsyncMock(
        return_value=SimpleNamespace(id=1, is_superuser=True, provider_id="p")
    )
    monkeypatch.setattr(auth_depends, "get_user", get_user)

    with pytest.raises(Exception) as raised:
        await auth_depends.get_superuser(authorization=None, x_api_key="dg-key")

    assert raised.value.status_code == 403
    get_user.assert_not_awaited()


# ---------------------------------------------------------------------------
# 3. Organization API keys must never reach these endpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["/superuser/organizations", "/superuser/organizations/1/api-keys"],
)
def test_org_api_key_cannot_reach_provisioning(
    monkeypatch, platform_key, route_client, path
):
    """A tenant credential must not create tenants or mint other tenants' keys.

    Presented *alongside* the correct platform key, so this pins the ordering
    too: an organization key cannot be laundered by attaching a valid one.
    """
    provision = AsyncMock()
    mint = AsyncMock()
    monkeypatch.setattr(superuser_routes, "provision_client_organization", provision)
    monkeypatch.setattr(superuser_routes, "mint_organization_api_key", mint)

    response = route_client.post(
        path,
        json=_payload(name="avsiq-key"),
        headers={PLATFORM_ADMIN_HEADER: PLATFORM_KEY, "X-API-Key": "dg-tenant-key"},
    )

    assert response.status_code == 403
    assert "Organization API keys" in response.json()["detail"]
    provision.assert_not_awaited()
    mint.assert_not_awaited()


# ---------------------------------------------------------------------------
# 4. Creating a client, against a real database
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_provisioning_creates_the_whole_tenant(db_session, no_bootstrap):
    """Service identity, organization, membership, selection and bootstrap."""
    result = await provisioning.provision_client_organization(
        display_name="Northwind",
        external_reference="avsiq-client-0001",
        service_email="Svc-Northwind@Example.com",
    )

    assert result.created is True
    assert result.bootstrapped is True

    organization = result.organization
    assert organization.display_name == "Northwind"
    assert organization.external_reference == "avsiq-client-0001"

    service_user = result.service_user
    # Normalized on the way in: the caller's capitalization must not create a
    # second identity on the next retry.
    assert service_user.email == "svc-northwind@example.com"
    assert service_user.selected_organization_id == organization.id
    assert await db_session.is_user_member_of_organization(
        user_id=service_user.id, organization_id=organization.id
    )

    # Bootstrap ran for this organization, attributed to the service identity.
    no_bootstrap.assert_awaited_once_with(
        organization.id, created_by=service_user.provider_id
    )

    # Exactly one organization, and it is listed as a client.
    organizations, total = await db_session.list_organizations_for_superadmin(
        limit=100, offset=0
    )
    assert total == 1
    assert organizations[0]["id"] == organization.id
    assert organizations[0]["user_count"] == 1


@pytest.mark.asyncio
async def test_the_service_account_password_is_never_returned(
    db_session, no_bootstrap
):
    """AVSIQ holds no Dograh password, so there is none to vault or rotate."""
    result = await provisioning.provision_client_organization(
        display_name="Northwind",
        external_reference="avsiq-client-0001",
        service_email="svc@example.com",
    )

    stored = await db_session.get_user_by_id(result.service_user.id)
    assert stored.password_hash
    assert not any(
        "password" in field for field in provisioning.ProvisionedOrganization.__annotations__
    )


@pytest.mark.asyncio
async def test_exact_retry_returns_the_same_organization(db_session, no_bootstrap):
    """A retry is not a second tenant, and reports ``created=False``."""
    first = await provisioning.provision_client_organization(
        display_name="Northwind",
        external_reference="avsiq-client-0001",
        service_email="svc@example.com",
    )
    second = await provisioning.provision_client_organization(
        display_name="Northwind",
        external_reference="avsiq-client-0001",
        service_email="svc@example.com",
    )

    assert second.created is False
    assert second.organization.id == first.organization.id
    assert second.service_user.id == first.service_user.id

    _, total = await db_session.list_organizations_for_superadmin(limit=100, offset=0)
    assert total == 1


@pytest.mark.asyncio
async def test_retry_re_enters_bootstrap(db_session, monkeypatch):
    """A first attempt that could not finish provisioning is finished by a retry."""
    bootstrap = AsyncMock(side_effect=[False, True])
    monkeypatch.setattr(provisioning, "ensure_organization_bootstrapped", bootstrap)

    first = await provisioning.provision_client_organization(
        display_name="Northwind",
        external_reference="avsiq-client-0001",
        service_email="svc@example.com",
    )
    assert first.bootstrapped is False

    second = await provisioning.provision_client_organization(
        display_name="Northwind",
        external_reference="avsiq-client-0001",
        service_email="svc@example.com",
    )

    assert second.bootstrapped is True
    assert bootstrap.await_count == 2


@pytest.mark.asyncio
async def test_a_failed_bootstrap_still_returns_the_tenant(db_session, monkeypatch):
    """Provisioning is not rolled back over a downstream outage.

    ``ensure_organization_bootstrapped`` never raises; it reports. The
    organization exists either way, and the caller is told what is missing
    rather than being handed a 500 and no mapping.
    """
    monkeypatch.setattr(
        provisioning, "ensure_organization_bootstrapped", AsyncMock(return_value=False)
    )

    result = await provisioning.provision_client_organization(
        display_name="Northwind",
        external_reference="avsiq-client-0001",
        service_email="svc@example.com",
    )

    assert result.created is True
    assert result.bootstrapped is False
    assert result.organization.id is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "display_name, service_email",
    [
        ("Renamed Co", "svc@example.com"),
        ("Northwind", "someone-else@example.com"),
    ],
)
async def test_conflicting_retry_is_refused(
    db_session, no_bootstrap, display_name, service_email
):
    """One reference names one client. Contradicting it is a 409, not an edit."""
    original = await provisioning.provision_client_organization(
        display_name="Northwind",
        external_reference="avsiq-client-0001",
        service_email="svc@example.com",
    )

    with pytest.raises(provisioning.ProvisioningConflict):
        await provisioning.provision_client_organization(
            display_name=display_name,
            external_reference="avsiq-client-0001",
            service_email=service_email,
        )

    # Nothing was renamed or re-pointed by the rejected call.
    unchanged = await db_session.get_organization_by_external_reference(
        "avsiq-client-0001"
    )
    assert unchanged.id == original.organization.id
    assert unchanged.display_name == "Northwind"


@pytest.mark.asyncio
async def test_reusing_a_service_email_for_a_new_client_is_refused(
    db_session, no_bootstrap
):
    """One account can select one organization, so one identity is one tenant."""
    await provisioning.provision_client_organization(
        display_name="Northwind",
        external_reference="avsiq-client-0001",
        service_email="svc@example.com",
    )

    with pytest.raises(provisioning.ProvisioningConflict):
        await provisioning.provision_client_organization(
            display_name="Contoso",
            external_reference="avsiq-client-0002",
            service_email="svc@example.com",
        )

    _, total = await db_session.list_organizations_for_superadmin(limit=100, offset=0)
    assert total == 1


@pytest.mark.asyncio
async def test_conflict_surfaces_as_409(monkeypatch, platform_key, route_client):
    monkeypatch.setattr(
        superuser_routes,
        "provision_client_organization",
        AsyncMock(side_effect=provisioning.ProvisioningConflict("already registered")),
    )

    response = route_client.post(
        "/superuser/organizations",
        json=_payload(),
        headers={PLATFORM_ADMIN_HEADER: PLATFORM_KEY},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "already registered"


def test_incomplete_bootstrap_is_reported_as_its_own_field(
    monkeypatch, platform_key, route_client
):
    """``bootstrapped`` is a distinct boolean, not folded into the status code.

    The caller has to be able to tell "tenant recorded, not yet ready" from
    both success and failure: the organization exists and must be persisted, and
    the request is worth retrying. A 200 whose body said nothing about it, or a
    5xx that discarded the mapping, would each lose one half of that.
    """
    monkeypatch.setattr(
        superuser_routes,
        "provision_client_organization",
        AsyncMock(
            return_value=provisioning.ProvisionedOrganization(
                organization=SimpleNamespace(
                    id=7,
                    provider_id="org_oss_1",
                    display_name="Northwind",
                    external_reference="avsiq-client-0001",
                ),
                service_user=SimpleNamespace(
                    id=3, email="svc@example.com", provider_id="oss_1"
                ),
                created=True,
                bootstrapped=False,
            )
        ),
    )

    response = route_client.post(
        "/superuser/organizations",
        json=_payload(),
        headers={PLATFORM_ADMIN_HEADER: PLATFORM_KEY},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["bootstrapped"] is False
    assert body["created"] is True
    assert body["organization_id"] == 7


# ---------------------------------------------------------------------------
# 5. Minting a tenant's provisioning key, repeatably
#
# The credential AVSIQ authenticates with. A lost response is not an error the
# caller can inspect its way out of -- it never learned the key -- so the only
# recovery is to send the request again. These pin that doing so converges on
# one live key rather than accumulating them.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_mint_creates_the_reserved_key(db_session, no_bootstrap):
    """Two tenants exist; the key belongs to the one named, and only it."""
    first = await provisioning.provision_client_organization(
        display_name="Northwind",
        external_reference="avsiq-client-0001",
        service_email="svc-northwind@example.com",
    )
    second = await provisioning.provision_client_organization(
        display_name="Contoso",
        external_reference="avsiq-client-0002",
        service_email="svc-contoso@example.com",
    )

    api_key, raw_key, archived = await provisioning.mint_organization_api_key(
        organization_id=second.organization.id
    )

    assert archived == [], "a first mint replaces nothing"
    assert api_key.name == constants.PLATFORM_PROVISIONING_API_KEY_NAME
    assert api_key.organization_id == second.organization.id
    assert api_key.organization_id != first.organization.id
    # Owned by the tenant's own service identity: a key with no owner is
    # rejected at authentication time and would be dead on arrival.
    assert api_key.created_by == second.service_user.id

    # The raw key authenticates, and authenticates as the tenant it was minted
    # for -- not as the platform, and not as the other tenant.
    validated = await db_session.validate_api_key(raw_key)
    assert validated is not None
    assert validated.organization_id == second.organization.id


@pytest.mark.asyncio
async def test_a_retry_replaces_rather_than_appends(db_session, no_bootstrap):
    """The whole point: retrying a lost response must not add a second key."""
    tenant = await provisioning.provision_client_organization(
        display_name="Northwind",
        external_reference="avsiq-client-0001",
        service_email="svc@example.com",
    )

    original, original_raw, _ = await provisioning.mint_organization_api_key(
        organization_id=tenant.organization.id
    )
    replacement, replacement_raw, archived = (
        await provisioning.mint_organization_api_key(
            organization_id=tenant.organization.id
        )
    )

    assert replacement.id != original.id
    assert archived == [original.id], (
        "the retry must report which credential it invalidated"
    )
    assert replacement_raw != original_raw


@pytest.mark.asyncio
async def test_the_previous_key_stops_authenticating(db_session, no_bootstrap):
    """A replaced key is dead immediately, not merely superseded."""
    tenant = await provisioning.provision_client_organization(
        display_name="Northwind",
        external_reference="avsiq-client-0001",
        service_email="svc@example.com",
    )

    _, original_raw, _ = await provisioning.mint_organization_api_key(
        organization_id=tenant.organization.id
    )
    assert await db_session.validate_api_key(original_raw) is not None

    await provisioning.mint_organization_api_key(
        organization_id=tenant.organization.id
    )

    assert await db_session.validate_api_key(original_raw) is None


@pytest.mark.asyncio
async def test_the_replacement_key_works(db_session, no_bootstrap):
    """Rotation must leave the tenant with a working credential, not a gap."""
    tenant = await provisioning.provision_client_organization(
        display_name="Northwind",
        external_reference="avsiq-client-0001",
        service_email="svc@example.com",
    )

    await provisioning.mint_organization_api_key(
        organization_id=tenant.organization.id
    )
    replacement, replacement_raw, _ = await provisioning.mint_organization_api_key(
        organization_id=tenant.organization.id
    )

    validated = await db_session.validate_api_key(replacement_raw)
    assert validated is not None
    assert validated.id == replacement.id
    assert validated.organization_id == tenant.organization.id


@pytest.mark.asyncio
async def test_repeated_mints_leave_exactly_one_active_key(db_session, no_bootstrap):
    """However many attempts got through, the end state is one live key."""
    tenant = await provisioning.provision_client_organization(
        display_name="Northwind",
        external_reference="avsiq-client-0001",
        service_email="svc@example.com",
    )

    raws = []
    for _ in range(5):
        _, raw, _ = await provisioning.mint_organization_api_key(
            organization_id=tenant.organization.id
        )
        raws.append(raw)

    active = [
        key
        for key in await db_session.get_api_keys_by_organization(
            tenant.organization.id
        )
        if key.is_active
        and key.name == constants.PLATFORM_PROVISIONING_API_KEY_NAME
    ]
    assert len(active) == 1
    assert active[0].key_prefix == raws[-1][: len(active[0].key_prefix)]

    # Only the last one authenticates; every abandoned attempt is dead.
    assert await db_session.validate_api_key(raws[-1]) is not None
    for stale in raws[:-1]:
        assert await db_session.validate_api_key(stale) is None


@pytest.mark.asyncio
async def test_the_database_refuses_a_second_active_key(db_session, no_bootstrap):
    """The invariant is enforced by an index, not only by the code path above.

    Written as a direct insert because that is the shape a concurrent racer
    takes: it archived what it could see, then tried to add its own active row
    beside the winner's. The index is what turns that into a retryable error
    instead of two live credentials.
    """
    from sqlalchemy.exc import IntegrityError

    tenant = await provisioning.provision_client_organization(
        display_name="Northwind",
        external_reference="avsiq-client-0001",
        service_email="svc@example.com",
    )
    await provisioning.mint_organization_api_key(
        organization_id=tenant.organization.id
    )

    with pytest.raises(IntegrityError):
        await db_session.create_api_key(
            organization_id=tenant.organization.id,
            name=constants.PLATFORM_PROVISIONING_API_KEY_NAME,
            created_by=tenant.service_user.id,
        )


@pytest.mark.asyncio
async def test_a_contended_replacement_retries_instead_of_failing(
    db_session, no_bootstrap, monkeypatch
):
    """The loser of a race re-runs against the winner's state and converges.

    Simulated by failing the first attempt exactly as the unique index would,
    because two real transactions cannot interleave inside the test's single
    savepoint-scoped session.
    """
    from sqlalchemy.exc import IntegrityError

    tenant = await provisioning.provision_client_organization(
        display_name="Northwind",
        external_reference="avsiq-client-0001",
        service_email="svc@example.com",
    )

    real = db_session._replace_api_key_by_name
    calls = {"n": 0}

    async def flaky_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise IntegrityError("insert", {}, Exception("duplicate key"))
        return await real(*args, **kwargs)

    monkeypatch.setattr(db_session, "_replace_api_key_by_name", flaky_once)

    api_key, raw_key, _ = await provisioning.mint_organization_api_key(
        organization_id=tenant.organization.id
    )

    assert calls["n"] == 2
    validated = await db_session.validate_api_key(raw_key)
    assert validated is not None
    assert validated.id == api_key.id


@pytest.mark.asyncio
async def test_ordinary_keys_are_untouched_by_rotation(db_session, no_bootstrap):
    """Rotation is scoped to the reserved name, not to the organization.

    Every organization is created with a "Default API Key", and a client may
    hold keys of its own. Replacing the provisioning key must not revoke them.
    """
    tenant = await provisioning.provision_client_organization(
        display_name="Northwind",
        external_reference="avsiq-client-0001",
        service_email="svc@example.com",
    )
    _, client_raw = await db_session.create_api_key(
        organization_id=tenant.organization.id,
        name="the client's own key",
        created_by=tenant.service_user.id,
    )

    await provisioning.mint_organization_api_key(
        organization_id=tenant.organization.id
    )
    await provisioning.mint_organization_api_key(
        organization_id=tenant.organization.id
    )

    assert await db_session.validate_api_key(client_raw) is not None


@pytest.mark.asyncio
async def test_rotation_does_not_reach_another_tenant(db_session, no_bootstrap):
    """One tenant's rotation must not disturb another's credential."""
    first = await provisioning.provision_client_organization(
        display_name="Northwind",
        external_reference="avsiq-client-0001",
        service_email="svc-northwind@example.com",
    )
    second = await provisioning.provision_client_organization(
        display_name="Contoso",
        external_reference="avsiq-client-0002",
        service_email="svc-contoso@example.com",
    )

    _, first_raw, _ = await provisioning.mint_organization_api_key(
        organization_id=first.organization.id
    )
    await provisioning.mint_organization_api_key(
        organization_id=second.organization.id
    )
    await provisioning.mint_organization_api_key(
        organization_id=second.organization.id
    )

    still_valid = await db_session.validate_api_key(first_raw)
    assert still_valid is not None
    assert still_valid.organization_id == first.organization.id


@pytest.mark.asyncio
async def test_minting_for_an_unknown_organization_is_a_404(db_session):
    with pytest.raises(provisioning.OrganizationNotFound):
        await provisioning.mint_organization_api_key(organization_id=987654)


@pytest.mark.asyncio
async def test_minting_for_a_memberless_organization_is_refused(db_session):
    """A key with no owner cannot authenticate, so it is not minted."""
    owner, _ = await db_session.get_or_create_user_by_provider_id("orphan-owner")
    organization, _ = await db_session.get_or_create_organization_by_provider_id(
        org_provider_id="org_orphan", user_id=owner.id
    )

    with pytest.raises(provisioning.ProvisioningConflict):
        await provisioning.mint_organization_api_key(
            organization_id=organization.id
        )


def test_minted_key_is_returned_once_over_http(monkeypatch, platform_key, route_client):
    monkeypatch.setattr(
        superuser_routes,
        "mint_organization_api_key",
        AsyncMock(
            return_value=(
                SimpleNamespace(
                    id=5,
                    organization_id=9,
                    name=constants.PLATFORM_PROVISIONING_API_KEY_NAME,
                    key_prefix="dg_abcd",
                    created_at="2026-09-05T00:00:00Z",
                ),
                "dg_abcd_the_raw_secret",
                [],
            )
        ),
    )

    response = route_client.post(
        "/superuser/organizations/9/api-keys",
        headers={PLATFORM_ADMIN_HEADER: PLATFORM_KEY},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["api_key"] == "dg_abcd_the_raw_secret"
    assert body["organization_id"] == 9
    assert body["key_prefix"] == "dg_abcd"
    assert body["replaced_key_ids"] == []
    assert body["rotated"] is False


def test_a_retry_reports_what_it_invalidated(monkeypatch, platform_key, route_client):
    """The caller learns that the credential it may have lost is now dead."""
    monkeypatch.setattr(
        superuser_routes,
        "mint_organization_api_key",
        AsyncMock(
            return_value=(
                SimpleNamespace(
                    id=6,
                    organization_id=9,
                    name=constants.PLATFORM_PROVISIONING_API_KEY_NAME,
                    key_prefix="dg_efgh",
                    created_at="2026-09-05T00:00:00Z",
                ),
                "dg_efgh_the_raw_secret",
                [5],
            )
        ),
    )

    response = route_client.post(
        "/superuser/organizations/9/api-keys",
        headers={PLATFORM_ADMIN_HEADER: PLATFORM_KEY},
    )

    assert response.status_code == 200
    assert response.json()["replaced_key_ids"] == [5]
    assert response.json()["rotated"] is True


def test_unknown_organization_surfaces_as_404(monkeypatch, platform_key, route_client):
    monkeypatch.setattr(
        superuser_routes,
        "mint_organization_api_key",
        AsyncMock(side_effect=provisioning.OrganizationNotFound("not found")),
    )

    response = route_client.post(
        "/superuser/organizations/9/api-keys",
        headers={PLATFORM_ADMIN_HEADER: PLATFORM_KEY},
    )

    assert response.status_code == 404
