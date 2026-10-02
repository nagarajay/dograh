"""WORKFLOW_SLOT_TEMPLATE_ON_CREATE fails closed.

If copying the organization template into a new workflow's slots fails, the
workflow stays `pending`, refuses to run, never resolves to inherited
organization defaults, and can be retried.
"""

import pytest
from cryptography.fernet import Fernet

from api.db.models import OrganizationModel, UserModel
from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
from api.services.configuration import ai_model_configuration as aimc
from api.services.configuration import secret_store, slot_settings
from api.services.configuration.registry import OpenAILLMService

SECRET = "sk-TEMPLATESECRET-abcdef-123456"
GRAPH = {
    "nodes": [
        {"id": "1", "type": "startCall", "data": {"name": "Start", "prompt": "Hi"}},
        {"id": "2", "type": "endCall", "data": {"name": "End", "prompt": "Bye"}},
    ],
    "edges": [{"id": "e1", "source": "1", "target": "2", "data": {"label": "End"}}],
}
ORG_DEFAULT = EffectiveAIModelConfiguration(
    llm=OpenAILLMService(api_key=SECRET, model="gpt-4.1")
)


@pytest.fixture(autouse=True)
def flag_and_key(monkeypatch):
    monkeypatch.setenv("WORKFLOW_SLOT_TEMPLATE_ON_CREATE", "true")
    monkeypatch.setenv(secret_store.KEYS_ENV, Fernet.generate_key().decode())


@pytest.fixture
def org_template(monkeypatch):
    holder = {"effective": ORG_DEFAULT}

    class Resolved:
        @property
        def effective(self):
            return holder["effective"]

    async def resolved(**_):
        return Resolved()

    async def base(**_):
        return holder["effective"]

    monkeypatch.setattr(aimc, "get_resolved_ai_model_configuration", resolved)
    monkeypatch.setattr(aimc, "_base_effective_configuration", base)
    return holder


@pytest.fixture
async def user(async_session):
    org = OrganizationModel(provider_id="org-tpl")
    async_session.add(org)
    await async_session.flush()
    u = UserModel(provider_id="user-tpl", selected_organization_id=org.id)
    async_session.add(u)
    await async_session.flush()
    return u


async def _create(client, name="new"):
    return await client.post(
        "/api/v1/workflow/create/definition",
        json={"name": name, "workflow_definition": GRAPH},
    )


async def _only_workflow(db_session, org_id):
    workflows = await db_session.get_all_workflows(organization_id=org_id)
    return next(w for w in workflows if w.name == "new")


async def test_seed_failure_leaves_workflow_pending_and_unrunnable(
    test_client_factory, user, org_template, db_session, monkeypatch
):
    monkeypatch.delenv(secret_store.KEYS_ENV)
    async with test_client_factory(user) as client:
        resp = await _create(client)
        assert resp.status_code == 503
        detail = resp.json()["detail"]
        assert detail["code"] == "slot_template_incomplete"
        assert detail["cause"] == "secret_store_unavailable"
        assert "apply-template" in detail["retry"]
        assert SECRET not in resp.text
        workflow_id = detail["workflow_id"]
        readback = await client.get(f"/api/v1/workflow/{workflow_id}/model-slots")

    assert readback.json()["template_status"] == "pending"
    assert all(s["published_version"] is None for s in readback.json()["slots"])
    # The organization default exists, but the workflow must not use it.
    with pytest.raises(slot_settings.SlotResolutionError):
        await aimc.get_effective_ai_model_configuration_for_workflow(
            organization_id=user.selected_organization_id,
            workflow_configurations={},
            workflow_id=workflow_id,
        )


async def test_retry_completes_the_copy_and_clears_the_marker(
    test_client_factory, user, org_template, monkeypatch
):
    good_keys = __import__("os").environ[secret_store.KEYS_ENV]
    monkeypatch.delenv(secret_store.KEYS_ENV)
    async with test_client_factory(user) as client:
        failed = await _create(client)
        workflow_id = failed.json()["detail"]["workflow_id"]

        monkeypatch.setenv(secret_store.KEYS_ENV, good_keys)
        retry = await client.post(
            f"/api/v1/workflow/{workflow_id}/model-slots/apply-template"
        )
        assert retry.status_code == 200, retry.text
        assert retry.json()["seeded_slots"] == ["llm"]
        again = await client.post(
            f"/api/v1/workflow/{workflow_id}/model-slots/apply-template"
        )
        readback = await client.get(f"/api/v1/workflow/{workflow_id}/model-slots")

    assert (
        again.status_code == 409
        and again.json()["detail"]["code"] == "template_not_pending"
    )
    assert readback.json()["template_status"] is None
    llm = next(s for s in readback.json()["slots"] if s["slot"] == "llm")
    assert llm["source"] == "workflow_slot" and llm["published"]["origin"] == "template"
    effective = await aimc.get_effective_ai_model_configuration_for_workflow(
        organization_id=user.selected_organization_id,
        workflow_configurations={},
        workflow_id=workflow_id,
    )
    assert effective.llm.api_key == SECRET


async def test_unexpected_error_mid_copy_is_fail_closed_and_leaks_nothing(
    test_client_factory, user, org_template, db_session, monkeypatch
):
    async def boom(**_):
        raise RuntimeError(f"db exploded near {SECRET}")

    monkeypatch.setattr(db_session, "seed_published_slots", boom)
    async with test_client_factory(user) as client:
        resp = await _create(client)
        workflow_id = resp.json()["detail"]["workflow_id"]
        readback = await client.get(f"/api/v1/workflow/{workflow_id}/model-slots")
    assert resp.status_code == 503
    assert resp.json()["detail"]["cause"] == "template_seed_failed"
    assert SECRET not in resp.text and SECRET not in readback.text
    assert readback.json()["template_status"] == "pending"
    assert all(s["published_version"] is None for s in readback.json()["slots"])


async def test_empty_organization_template_stays_pending_instead_of_falling_back(
    test_client_factory, user, org_template
):
    org_template["effective"] = EffectiveAIModelConfiguration()
    async with test_client_factory(user) as client:
        resp = await _create(client)
        workflow_id = resp.json()["detail"]["workflow_id"]
        readback = await client.get(f"/api/v1/workflow/{workflow_id}/model-slots")
    assert resp.json()["detail"]["cause"] == "template_empty"
    assert readback.json()["template_status"] == "pending"
    with pytest.raises(slot_settings.SlotResolutionError):
        await aimc.get_effective_ai_model_configuration_for_workflow(
            organization_id=user.selected_organization_id,
            workflow_configurations={},
            workflow_id=workflow_id,
        )


async def test_success_path_creates_workflow_with_its_own_slots(
    test_client_factory, user, org_template
):
    async with test_client_factory(user) as client:
        resp = await _create(client)
        assert resp.status_code == 200, resp.text
        readback = await client.get(f"/api/v1/workflow/{resp.json()['id']}/model-slots")
    assert readback.json()["template_status"] is None
    llm = next(s for s in readback.json()["slots"] if s["slot"] == "llm")
    assert llm["source"] == "workflow_slot"
    # A later organization-default change does not reach the new workflow.
    org_template["effective"] = EffectiveAIModelConfiguration(
        llm=OpenAILLMService(api_key="other", model="gpt-5")
    )
    effective = await aimc.get_effective_ai_model_configuration_for_workflow(
        organization_id=user.selected_organization_id,
        workflow_configurations={},
        workflow_id=resp.json()["id"],
    )
    assert effective.llm.model == "gpt-4.1" and effective.llm.api_key == SECRET


async def test_duplicate_also_fails_closed(
    test_client_factory, user, org_template, monkeypatch
):
    async with test_client_factory(user) as client:
        ok = await _create(client)
        monkeypatch.delenv(secret_store.KEYS_ENV)
        dup = await client.post(f"/api/v1/workflow/{ok.json()['id']}/duplicate")
    assert dup.status_code == 503
    assert dup.json()["detail"]["code"] == "slot_template_incomplete"


async def test_flag_off_leaves_legacy_behaviour_untouched(
    test_client_factory, user, org_template, monkeypatch
):
    monkeypatch.setenv("WORKFLOW_SLOT_TEMPLATE_ON_CREATE", "false")
    monkeypatch.delenv(secret_store.KEYS_ENV)  # would fail if seeding were attempted
    async with test_client_factory(user) as client:
        resp = await _create(client)
        assert resp.status_code == 200
        readback = await client.get(f"/api/v1/workflow/{resp.json()['id']}/model-slots")
    assert readback.json()["template_status"] is None
    effective = await aimc.get_effective_ai_model_configuration_for_workflow(
        organization_id=user.selected_organization_id,
        workflow_configurations={},
        workflow_id=resp.json()["id"],
    )
    assert effective is ORG_DEFAULT  # legacy: inherits the organization default


async def test_other_organization_cannot_retry(
    test_client_factory, user, org_template, async_session, monkeypatch
):
    monkeypatch.delenv(secret_store.KEYS_ENV)
    async with test_client_factory(user) as client:
        failed = await _create(client)
    workflow_id = failed.json()["detail"]["workflow_id"]
    other_org = OrganizationModel(provider_id="org-tpl-2")
    async_session.add(other_org)
    await async_session.flush()
    other = UserModel(provider_id="user-tpl-2", selected_organization_id=other_org.id)
    async_session.add(other)
    await async_session.flush()
    async with test_client_factory(other) as client:
        resp = await client.post(
            f"/api/v1/workflow/{workflow_id}/model-slots/apply-template"
        )
    assert resp.status_code == 404
