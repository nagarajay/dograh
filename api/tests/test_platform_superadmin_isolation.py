"""The platform super-admin owns no organization.

Organizations model client tenants -- one Dograh organization per AVSIQ client
-- so a database with no clients provisioned has no organization rows, and the
one account that exists is a platform super-admin with
``selected_organization_id = NULL`` and no ``organization_users`` membership.

Everything here pins one half of that arrangement:

- an org-less super-admin can still authenticate and reach the console
- an org-less *tenant* request fails closed instead of querying with ``None``
- cross-organization authority comes from the super-admin flag plus an explicit
  target organization, never from membership
- provisioning the first client creates exactly one organization
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.services.auth import depends as auth_depends
from api.services.auth.depends import (
    get_user,
    get_user_with_selected_organization,
)

ORG_LESS_SUPERADMIN = SimpleNamespace(
    id=1,
    email="platform@example.com",
    provider_id="oss-platform-admin",
    is_superuser=True,
    selected_organization_id=None,
)


# ---------------------------------------------------------------------------
# 1. Authenticating with no organization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oss_login_does_not_require_an_organization(monkeypatch):
    """``_handle_oss_auth`` must return an org-less user rather than refuse.

    This is the branch a platform super-admin signs in through. Bootstrap is
    skipped because there is no organization to bootstrap, not because the
    account is broken.
    """
    monkeypatch.setattr(
        auth_depends.db_client,
        "get_user_by_id",
        AsyncMock(return_value=ORG_LESS_SUPERADMIN),
    )
    monkeypatch.setattr(auth_depends, "decode_jwt_token", lambda token: {"sub": "1"})
    ensure_bootstrapped = AsyncMock()
    monkeypatch.setattr(
        auth_depends, "ensure_organization_bootstrapped", ensure_bootstrapped
    )

    user = await auth_depends._handle_oss_auth("Bearer token")

    assert user is ORG_LESS_SUPERADMIN
    assert user.selected_organization_id is None
    ensure_bootstrapped.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_superuser_accepts_a_user_with_no_organization(monkeypatch):
    """Platform authority is the flag alone; membership is not consulted."""
    monkeypatch.setattr(
        auth_depends, "get_user", AsyncMock(return_value=ORG_LESS_SUPERADMIN)
    )

    resolved = await auth_depends.get_superuser(
        authorization="Bearer token", x_api_key=None
    )

    assert resolved is ORG_LESS_SUPERADMIN
    assert resolved.selected_organization_id is None


def test_auth_me_reports_a_null_organization():
    """``/auth/me`` must answer for an org-less account, not refuse it."""
    from api.routes.auth import router as auth_router

    app = FastAPI()
    app.include_router(auth_router)
    app.dependency_overrides[get_user] = lambda: ORG_LESS_SUPERADMIN

    response = TestClient(app).get("/auth/me")

    assert response.status_code == 200
    assert response.json()["organization_id"] is None


def test_auth_user_exposes_the_null_organization_to_the_console():
    """The console decides where to send an org-less super-admin from this.

    ``/user/auth/user`` is deliberately not organization-scoped: it is the one
    authenticated endpoint that can answer "does this user have an organization
    at all", which every organization-scoped route now refuses to.
    """
    from api.routes.user import router as user_router

    app = FastAPI()
    app.include_router(user_router)
    app.dependency_overrides[get_user] = lambda: ORG_LESS_SUPERADMIN

    response = TestClient(app).get("/user/auth/user")

    assert response.status_code == 200
    assert response.json() == {
        "id": 1,
        "is_superuser": True,
        "selected_organization_id": None,
    }


# ---------------------------------------------------------------------------
# 3. The console with zero clients provisioned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_superuser_organizations_is_empty_with_no_clients(db_session):
    """Zero AVSIQ clients means zero rows -- not one 'unnamed' bootstrap row.

    The listing has no client/non-client predicate to get wrong because there is
    no non-client organization to exclude: a super-admin holds no organization
    of their own.
    """
    await db_session.get_or_create_user_by_provider_id("platform-superadmin")

    organizations, total_count = await db_session.list_organizations_for_superadmin(
        limit=100, offset=0
    )

    assert organizations == []
    assert total_count == 0


@pytest.mark.asyncio
async def test_a_superadmin_user_alone_creates_no_organization(db_session):
    """Creating the platform account must not mint a tenant as a side effect."""
    user, created = await db_session.get_or_create_user_by_provider_id("platform-only")

    assert created is True
    assert user.selected_organization_id is None
    _, total_count = await db_session.list_organizations_for_superadmin(
        limit=100, offset=0
    )
    assert total_count == 0


# ---------------------------------------------------------------------------
# 4. Tenant routes fail closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_org_scoped_dependency_refuses_a_null_organization():
    with pytest.raises(HTTPException) as raised:
        await get_user_with_selected_organization(ORG_LESS_SUPERADMIN)

    assert raised.value.status_code == 400
    assert raised.value.detail == "No organization selected"


@pytest.mark.parametrize(
    "module_path, method, path",
    [
        # Creating a workflow used to skip the organization check entirely in
        # OSS mode, writing a row with organization_id = NULL.
        ("api.routes.workflow", "post", "/workflow/create/definition"),
        ("api.routes.workflow", "get", "/workflow/count"),
        # This one built an S3 key literally containing the string "None".
        ("api.routes.knowledge_base", "post", "/knowledge-base/upload-url"),
        ("api.routes.knowledge_base", "get", "/knowledge-base/documents"),
        ("api.routes.folder", "get", "/folder/"),
        ("api.routes.credentials", "get", "/credentials/"),
    ],
)
def test_tenant_routes_refuse_an_org_less_caller(module_path, method, path):
    """No organization means 400, never a query or a write scoped to ``None``."""
    import importlib

    module = importlib.import_module(module_path)
    app = FastAPI()
    app.include_router(module.router)
    # Overriding the base dependency, not the org-scoped one: the point is that
    # the org-scoped wrapper is what each route now depends on, and that it
    # refuses this user.
    app.dependency_overrides[get_user] = lambda: ORG_LESS_SUPERADMIN

    client = TestClient(app)
    response = client.post(path, json={}) if method == "post" else client.get(path)

    assert response.status_code == 400
    assert response.json() == {"detail": "No organization selected"}


def test_workflow_create_is_unreachable_without_an_organization(monkeypatch):
    """Pin the specific regression: an org-less create must not reach the DB."""
    import api.routes.workflow as workflow_routes

    app = FastAPI()
    app.include_router(workflow_routes.router)
    app.dependency_overrides[get_user] = lambda: ORG_LESS_SUPERADMIN
    created = AsyncMock()
    # monkeypatch, not a bare assignment: db_client is a process-wide singleton,
    # and a stub left on it leaks into every later test in the session.
    monkeypatch.setattr(workflow_routes.db_client, "create_workflow", created)

    response = TestClient(app).post(
        "/workflow/create/definition",
        json={"name": "x", "workflow_definition": {}},
    )

    assert response.status_code == 400
    created.assert_not_awaited()


# ---------------------------------------------------------------------------
# 5. Cross-organization authority without membership
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_org_run_is_authorized_without_membership(db_session, monkeypatch):
    """A super-admin drives a client's agent while belonging to no organization.

    Before this, ``authorize_workflow_run_start`` answered "is the actor a
    member of this organization", which a platform super-admin never is, so the
    console's test run died with ``workflow_not_found`` after the socket had
    already authorized it.
    """
    from api.services import quota_service

    client_user, _ = await db_session.get_or_create_user_by_provider_id("client-owner")
    organization, _ = await db_session.get_or_create_organization_by_provider_id(
        org_provider_id="client-org", user_id=client_user.id
    )
    await db_session.add_user_to_organization(client_user.id, organization.id)
    workflow = await db_session.create_workflow(
        name="client-agent",
        workflow_definition={},
        user_id=client_user.id,
        organization_id=organization.id,
    )

    superadmin, _ = await db_session.get_or_create_user_by_provider_id("platform-admin")
    assert superadmin.selected_organization_id is None
    assert not await db_session.is_user_member_of_organization(
        user_id=superadmin.id, organization_id=organization.id
    )

    monkeypatch.setattr(quota_service, "db_client", db_session)

    denied = await quota_service.authorize_workflow_run_start(
        workflow_id=workflow.id,
        organization_id=organization.id,
        actor_user=superadmin,
    )
    assert denied.error_code == "workflow_not_found", (
        "without the platform-admin flag, a non-member must still be refused"
    )

    # With the flag, membership is not asked about at all. Asserting on the
    # question rather than on the final result keeps this test about
    # authorization: what the run does after this gate is the ordinary
    # MPS credit check, which has its own tests and its own failure modes.
    membership = AsyncMock(
        side_effect=AssertionError(
            "membership must not be consulted for a platform super-admin"
        )
    )
    monkeypatch.setattr(db_session, "is_user_member_of_organization", membership)

    allowed = await quota_service.authorize_workflow_run_start(
        workflow_id=workflow.id,
        organization_id=organization.id,
        actor_user=superadmin,
        actor_is_platform_admin=True,
    )
    membership.assert_not_awaited()
    assert allowed.error_code != "workflow_not_found"


@pytest.mark.asyncio
async def test_superadmin_test_run_authorization_still_needs_the_initiator(
    db_session, monkeypatch
):
    """The flag is trusted downstream, so the socket must earn it every time.

    ``_authorize_superadmin_test_run`` is what sets ``actor_is_platform_admin``.
    A superuser who did not create the run gets nothing, which is what keeps the
    quota exemption from becoming "any superuser may drive any run".
    """
    import api.routes.webrtc_signaling as signaling
    from api.services.superuser.test_runs import superadmin_test_run_extra

    owner, _ = await db_session.get_or_create_user_by_provider_id("run-owner")
    organization, _ = await db_session.get_or_create_organization_by_provider_id(
        org_provider_id="run-org", user_id=owner.id
    )
    workflow = await db_session.create_workflow(
        name="agent",
        workflow_definition={},
        user_id=owner.id,
        organization_id=organization.id,
    )
    from api.enums import WorkflowRunMode

    initiator, _ = await db_session.get_or_create_user_by_provider_id("initiator")
    run = await db_session.create_workflow_run(
        "superadmin-test",
        workflow.id,
        WorkflowRunMode.SMALLWEBRTC.value,
        initiator.id,
        organization_id=organization.id,
        extra=superadmin_test_run_extra(initiator.id),
    )

    monkeypatch.setattr(signaling, "db_client", db_session)

    other_superuser = SimpleNamespace(id=initiator.id + 999, is_superuser=True)
    assert (
        await signaling._authorize_superadmin_test_run(other_superuser, run.id) is None
    )

    initiator.is_superuser = True
    authorized = await signaling._authorize_superadmin_test_run(initiator, run.id)
    assert authorized is not None
    assert authorized.workflow.organization_id == organization.id


# ---------------------------------------------------------------------------
# 6. Provisioning the first client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_client_creates_exactly_one_organization(db_session):
    """One AVSIQ client is one Dograh organization -- and only one."""
    user, _ = await db_session.get_or_create_user_by_provider_id("first-client-owner")

    (
        organization,
        was_created,
    ) = await db_session.get_or_create_organization_by_provider_id(
        org_provider_id="org_first_client",
        user_id=user.id,
        display_name="Northwind",
        external_reference="avsiq-client-0001",
    )
    await db_session.add_user_to_organization(user.id, organization.id)

    assert was_created is True
    organizations, total_count = await db_session.list_organizations_for_superadmin(
        limit=100, offset=0
    )
    assert total_count == 1
    assert [org["id"] for org in organizations] == [organization.id]
    assert organizations[0]["display_name"] == "Northwind"
    assert organizations[0]["external_reference"] == "avsiq-client-0001"
    assert organizations[0]["user_count"] == 1


@pytest.mark.asyncio
async def test_reprovisioning_the_same_client_creates_no_second_organization(
    db_session,
):
    """Provisioning is idempotent per provider id, so a retry is not a tenant."""
    user, _ = await db_session.get_or_create_user_by_provider_id("retry-owner")
    for _ in range(3):
        await db_session.get_or_create_organization_by_provider_id(
            org_provider_id="org_retry_client",
            user_id=user.id,
            display_name="Retry Co",
            external_reference="avsiq-client-0002",
        )

    _, total_count = await db_session.list_organizations_for_superadmin(
        limit=100, offset=0
    )
    assert total_count == 1
