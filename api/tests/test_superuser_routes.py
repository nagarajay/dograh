"""Tests for the super-admin read APIs and their access boundary."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from api.enums import WorkflowRunMode
from api.services.auth import depends as auth_depends

# ---------------------------------------------------------------------------
# Access boundary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_key_cannot_reach_superuser_endpoints(monkeypatch):
    """An API key must never authenticate a superuser endpoint.

    ``_handle_api_key_auth`` returns the key owner's full UserModel, superuser
    flag included, so a key created by a superuser would otherwise become a
    permanent cross-tenant admin credential.
    """
    get_user = AsyncMock(
        return_value=SimpleNamespace(id=1, is_superuser=True, provider_id="p")
    )
    monkeypatch.setattr(auth_depends, "get_user", get_user)

    with pytest.raises(HTTPException) as excinfo:
        await auth_depends.get_superuser(authorization=None, x_api_key="dg-secret-key")

    assert excinfo.value.status_code == 403
    get_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_key_rejected_even_alongside_a_session_token(monkeypatch):
    """Presenting both headers must not let the session token launder the key."""
    monkeypatch.setattr(
        auth_depends,
        "get_user",
        AsyncMock(return_value=SimpleNamespace(id=1, is_superuser=True)),
    )

    with pytest.raises(HTTPException) as excinfo:
        await auth_depends.get_superuser(
            authorization="Bearer token", x_api_key="dg-secret-key"
        )

    assert excinfo.value.status_code == 403


@pytest.mark.asyncio
async def test_non_superuser_session_is_rejected(monkeypatch):
    monkeypatch.setattr(
        auth_depends,
        "get_user",
        AsyncMock(return_value=SimpleNamespace(id=2, is_superuser=False)),
    )

    with pytest.raises(HTTPException) as excinfo:
        await auth_depends.get_superuser(authorization="Bearer token", x_api_key=None)

    assert excinfo.value.status_code == 403


@pytest.mark.asyncio
async def test_superuser_session_is_accepted(monkeypatch):
    user = SimpleNamespace(id=3, is_superuser=True)
    monkeypatch.setattr(auth_depends, "get_user", AsyncMock(return_value=user))

    assert (
        await auth_depends.get_superuser(authorization="Bearer token", x_api_key=None)
        is user
    )


# ---------------------------------------------------------------------------
# Organization listing
# ---------------------------------------------------------------------------


async def _make_org(db_session, suffix: str):
    user, _ = await db_session.get_or_create_user_by_provider_id(f"su-user-{suffix}")
    organization, _ = await db_session.get_or_create_organization_by_provider_id(
        org_provider_id=f"su-org-{suffix}", user_id=user.id
    )
    await db_session.add_user_to_organization(user.id, organization.id)
    return user, organization


@pytest.mark.asyncio
async def test_list_organizations_includes_organizations_with_no_runs(db_session):
    """The whole point of the view: an empty organization must still appear."""
    _, empty_org = await _make_org(db_session, "empty")

    organizations, total_count = await db_session.list_organizations_for_superadmin(
        limit=100, offset=0
    )

    listed = {org["id"]: org for org in organizations}
    assert empty_org.id in listed
    assert listed[empty_org.id]["workflow_count"] == 0
    assert listed[empty_org.id]["run_count"] == 0
    assert listed[empty_org.id]["last_run_at"] is None
    assert total_count >= 1


@pytest.mark.asyncio
async def test_list_organizations_counts_are_not_multiplied_by_the_join(db_session):
    """Workflow and run counts must stay independent of one another."""
    user, organization = await _make_org(db_session, "counts")
    workflows = []
    for index in range(2):
        workflows.append(
            await db_session.create_workflow(
                name=f"agent-{index}",
                workflow_definition={},
                user_id=user.id,
                organization_id=organization.id,
            )
        )
    for index in range(3):
        await db_session.create_workflow_run(
            name=f"run-{index}",
            workflow_id=workflows[0].id,
            mode=WorkflowRunMode.SMALLWEBRTC.value,
            user_id=user.id,
            organization_id=organization.id,
        )

    organizations, _ = await db_session.list_organizations_for_superadmin(
        limit=100, offset=0, organization_id=organization.id
    )

    assert len(organizations) == 1
    assert organizations[0]["workflow_count"] == 2
    assert organizations[0]["run_count"] == 3
    assert organizations[0]["user_count"] == 1
    assert organizations[0]["last_run_at"] is not None


# ---------------------------------------------------------------------------
# Run attribution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runs_are_attributed_to_the_workflows_organization(db_session):
    """A run belongs to its workflow's organization, not the owner's current one.

    Regression test: attribution used to read the workflow owner's
    ``selected_organization_id``, so switching organizations silently
    re-attributed every historical run.
    """
    user, owning_org = await _make_org(db_session, "owner")
    _, other_org = await _make_org(db_session, "other")

    workflow = await db_session.create_workflow(
        name="attributed-agent",
        workflow_definition={},
        user_id=user.id,
        organization_id=owning_org.id,
    )
    run = await db_session.create_workflow_run(
        name="attributed-run",
        workflow_id=workflow.id,
        mode=WorkflowRunMode.SMALLWEBRTC.value,
        user_id=user.id,
        organization_id=owning_org.id,
    )

    # The owner moves to a different organization after the run happened.
    await db_session.add_user_to_organization(user.id, other_org.id)
    await db_session.update_user_selected_organization(user.id, other_org.id)

    runs, _ = await db_session.get_workflow_runs_for_superadmin(
        limit=100, offset=0, organization_id=owning_org.id
    )

    matching = [item for item in runs if item["id"] == run.id]
    assert len(matching) == 1
    assert matching[0]["organization_id"] == owning_org.id
    assert matching[0]["organization_name"] == owning_org.provider_id

    other_runs, _ = await db_session.get_workflow_runs_for_superadmin(
        limit=100, offset=0, organization_id=other_org.id
    )
    assert all(item["id"] != run.id for item in other_runs)


# ---------------------------------------------------------------------------
# Secret exposure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_operational_state_never_carries_stored_secrets(db_session, monkeypatch):
    """Configuration rows hold live credentials; the derived view must not.

    Asserted against the serialized response rather than field by field, so a
    future field that passes a stored value through fails here.
    """
    import json

    from api.enums import OrganizationConfigurationKey
    from api.services.superuser import org_health

    monkeypatch.setattr(org_health, "db_client", db_session)

    _, organization = await _make_org(db_session, "secrets")
    await db_session.upsert_configuration(
        organization.id,
        OrganizationConfigurationKey.LANGFUSE_CREDENTIALS.value,
        {"public_key": "pk-SECRET-VALUE", "secret_key": "sk-SECRET-VALUE"},
    )
    await db_session.upsert_configuration(
        organization.id,
        OrganizationConfigurationKey.MODEL_CONFIGURATION_V2.value,
        {"api_key": "model-SECRET-VALUE"},
    )
    await db_session.create_telephony_configuration(
        organization_id=organization.id,
        name="primary",
        provider="twilio",
        credentials={"auth_token": "telephony-SECRET-VALUE"},
    )

    state = await org_health.get_organization_operational_state(organization.id)
    serialized = json.dumps(state, default=str)

    assert "SECRET-VALUE" not in serialized
    assert state.langfuse_configured is True
    assert state.model_configuration_present is True
    assert [config.provider for config in state.telephony_configurations] == ["twilio"]
    assert state.telephony_configurations[0].phone_number_count == 0


@pytest.mark.asyncio
async def test_bootstrap_state_reports_never_started_without_a_lease(
    db_session, monkeypatch
):
    from api.services.superuser import org_health

    monkeypatch.setattr(org_health, "db_client", db_session)
    _, organization = await _make_org(db_session, "bootstrap")

    state = await org_health.get_organization_operational_state(organization.id)

    assert state.bootstrap_state == org_health.BOOTSTRAP_NEVER
    assert state.model_configuration_present is False


# ---------------------------------------------------------------------------
# Super-admin test runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_superadmin_test_runs_are_excluded_from_customer_usage(db_session):
    """A super admin's verification call is not the customer's traffic."""
    from api.services.superuser.test_runs import superadmin_test_run_extra

    user, organization = await _make_org(db_session, "usage-exclusion")
    workflow = await db_session.create_workflow(
        name="usage-agent",
        workflow_definition={},
        user_id=user.id,
        organization_id=organization.id,
    )

    customer_run = await db_session.create_workflow_run(
        name="customer-run",
        workflow_id=workflow.id,
        mode=WorkflowRunMode.SMALLWEBRTC.value,
        user_id=user.id,
        organization_id=organization.id,
    )
    test_run = await db_session.create_workflow_run(
        name="superadmin-test",
        workflow_id=workflow.id,
        mode=WorkflowRunMode.SMALLWEBRTC.value,
        user_id=user.id,
        organization_id=organization.id,
        extra=superadmin_test_run_extra(user.id),
    )
    for run in (customer_run, test_run):
        await db_session.update_workflow_run(
            run.id, usage_info={"call_duration_seconds": 30}
        )

    runs, total_count, _, _ = await db_session.get_usage_history(
        organization_id=organization.id, limit=100
    )

    listed_ids = {run["id"] for run in runs}
    assert customer_run.id in listed_ids
    assert test_run.id not in listed_ids
    assert total_count == 1


@pytest.mark.asyncio
async def test_superadmin_test_runs_still_appear_in_the_superadmin_run_list(db_session):
    """Excluded from the customer's view, but the operator must still see it."""
    from api.services.superuser.test_runs import superadmin_test_run_extra

    user, organization = await _make_org(db_session, "superadmin-visible")
    workflow = await db_session.create_workflow(
        name="visible-agent",
        workflow_definition={},
        user_id=user.id,
        organization_id=organization.id,
    )
    test_run = await db_session.create_workflow_run(
        name="superadmin-test",
        workflow_id=workflow.id,
        mode=WorkflowRunMode.SMALLWEBRTC.value,
        user_id=user.id,
        organization_id=organization.id,
        extra=superadmin_test_run_extra(user.id),
    )

    runs, _ = await db_session.get_workflow_runs_for_superadmin(
        limit=100, offset=0, organization_id=organization.id
    )

    assert test_run.id in {run["id"] for run in runs}


def test_untagged_runs_are_never_treated_as_test_runs():
    """Runs written before the marker existed must keep counting."""
    from api.services.superuser.test_runs import (
        is_superadmin_test_run,
        superadmin_test_run_initiator,
    )

    assert is_superadmin_test_run(None) is False
    assert is_superadmin_test_run({}) is False
    assert is_superadmin_test_run({"recordings": {}}) is False
    assert superadmin_test_run_initiator({"superadmin_test": True}) is None
    assert (
        superadmin_test_run_initiator(
            {"superadmin_test": True, "superadmin_initiated_by_user_id": 7}
        )
        == 7
    )


@pytest.mark.asyncio
async def test_only_the_initiating_superuser_may_drive_a_cross_org_test_run(db_session):
    """The WebRTC exception is per-run and per-superuser, not a role bypass."""
    from types import SimpleNamespace

    from api.routes import webrtc_signaling
    from api.services.superuser.test_runs import superadmin_test_run_extra

    owner, organization = await _make_org(db_session, "webrtc-gate")
    workflow = await db_session.create_workflow(
        name="gate-agent",
        workflow_definition={},
        user_id=owner.id,
        organization_id=organization.id,
    )
    initiator_id = 4242
    test_run = await db_session.create_workflow_run(
        name="superadmin-test",
        workflow_id=workflow.id,
        mode=WorkflowRunMode.SMALLWEBRTC.value,
        user_id=owner.id,
        organization_id=organization.id,
        extra=superadmin_test_run_extra(initiator_id),
    )
    customer_run = await db_session.create_workflow_run(
        name="customer-run",
        workflow_id=workflow.id,
        mode=WorkflowRunMode.SMALLWEBRTC.value,
        user_id=owner.id,
        organization_id=organization.id,
    )

    original_client = webrtc_signaling.db_client
    webrtc_signaling.db_client = db_session
    try:
        initiator = SimpleNamespace(id=initiator_id, is_superuser=True)
        other_superuser = SimpleNamespace(id=initiator_id + 1, is_superuser=True)
        non_superuser = SimpleNamespace(id=initiator_id, is_superuser=False)

        allowed = await webrtc_signaling._authorize_superadmin_test_run(
            initiator, test_run.id
        )
        assert allowed is not None
        assert allowed.workflow.organization_id == organization.id

        assert (
            await webrtc_signaling._authorize_superadmin_test_run(
                other_superuser, test_run.id
            )
            is None
        )
        assert (
            await webrtc_signaling._authorize_superadmin_test_run(
                non_superuser, test_run.id
            )
            is None
        )
        # A superuser gets no access to an ordinary run of another organization.
        assert (
            await webrtc_signaling._authorize_superadmin_test_run(
                initiator, customer_run.id
            )
            is None
        )
    finally:
        webrtc_signaling.db_client = original_client


# ---------------------------------------------------------------------------
# Customer identity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_organization_identity_is_stored_and_searchable(db_session):
    user, _ = await db_session.get_or_create_user_by_provider_id("identity-user")
    (
        organization,
        was_created,
    ) = await db_session.get_or_create_organization_by_provider_id(
        org_provider_id="su-org-identity",
        user_id=user.id,
        display_name="Northwind Clinics",
        external_reference="avsiq-client-8821",
    )
    assert was_created is True
    assert organization.display_name == "Northwind Clinics"
    assert organization.external_reference == "avsiq-client-8821"

    resolved = await db_session.get_organization_by_external_reference(
        "avsiq-client-8821"
    )
    assert resolved is not None and resolved.id == organization.id

    by_name, _ = await db_session.list_organizations_for_superadmin(
        limit=10, offset=0, search="northwind"
    )
    by_reference, _ = await db_session.list_organizations_for_superadmin(
        limit=10, offset=0, search="avsiq-client-8821"
    )
    assert [org["id"] for org in by_name] == [organization.id]
    assert [org["id"] for org in by_reference] == [organization.id]


@pytest.mark.asyncio
async def test_reprovisioning_never_renames_a_live_organization(db_session):
    """A repeated provisioning call must not re-point an existing customer."""
    user, _ = await db_session.get_or_create_user_by_provider_id("identity-stable")
    organization, _ = await db_session.get_or_create_organization_by_provider_id(
        org_provider_id="su-org-identity-stable",
        user_id=user.id,
        display_name="Original Name",
        external_reference="avsiq-client-original",
    )

    again, was_created = await db_session.get_or_create_organization_by_provider_id(
        org_provider_id="su-org-identity-stable",
        user_id=user.id,
        display_name="Renamed",
        external_reference="avsiq-client-different",
    )

    assert was_created is False
    assert again.id == organization.id
    assert again.display_name == "Original Name"
    assert again.external_reference == "avsiq-client-original"
