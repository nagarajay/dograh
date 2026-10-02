"""Per-workflow model slot settings: isolation, versioning, secrets, embeddings.

DB-backed (transactional test session); no network and no provider calls. The
provider validator is replaced by a recording fake wherever a live check would run.
"""

import json

import pytest
from cryptography.fernet import Fernet
from loguru import logger

from api.db.knowledge_base_client import EmbeddingIndexMismatchError
from api.db.models import (
    KnowledgeBaseChunkModel,
    KnowledgeBaseDocumentModel,
    OrganizationModel,
    ProviderCredentialModel,
    UserModel,
    WorkflowDefinitionModel,
)
from api.routes import workflow_model_slots as slots_route
from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
from api.services.configuration import ai_model_configuration as aimc
from api.services.configuration import secret_store, slot_settings
from api.services.configuration.registry import (
    AzureLLMService,
    DeepgramSTTConfiguration,
    ElevenlabsTTSConfiguration,
    OpenAIEmbeddingsConfiguration,
    OpenAILLMService,
)
from api.services.configuration.safe_errors import safe_exception_detail

OPENAI_SECRET = "sk-TESTSECRET-abcdef-123456"
OPENAI_SECRET_2 = "sk-ROTATEDSECRET-zyxwvu-654321"
SA_PRIVATE = "SUPERSECRETPRIVATEKEYBODY"
GRAPH = {
    "nodes": [
        {"id": "1", "type": "startCall", "data": {"name": "Start", "prompt": "Hi"}},
        {"id": "2", "type": "endCall", "data": {"name": "End", "prompt": "Bye"}},
    ],
    "edges": [{"id": "e1", "source": "1", "target": "2", "data": {"label": "End"}}],
}
LLM_CFG = {"provider": "openai", "model": "gpt-4.1-mini"}


class FakeValidator:
    """Stands in for UserConfigurationValidator: no network."""

    def __init__(self, errors=None):
        self.errors = errors or []
        self.calls = []

    async def validate_single(self, service, slot, **_):
        self.calls.append((slot, service.provider))
        return list(self.errors)


@pytest.fixture(autouse=True)
def encryption_key(monkeypatch):
    monkeypatch.setenv(secret_store.KEYS_ENV, Fernet.generate_key().decode())


@pytest.fixture
def fake_validator(monkeypatch):
    fake = FakeValidator()
    monkeypatch.setattr(slots_route, "UserConfigurationValidator", lambda: fake)
    return fake


@pytest.fixture
def org_default(monkeypatch):
    """A complete organization default, so legacy workflows have something to inherit."""
    base = EffectiveAIModelConfiguration(
        llm=OpenAILLMService(api_key="org-default-key", model="gpt-4.1"),
        stt=DeepgramSTTConfiguration(api_key="org-default-key"),
        tts=ElevenlabsTTSConfiguration(api_key="org-default-key"),
        embeddings=OpenAIEmbeddingsConfiguration(api_key="org-default-key"),
    )

    async def fake_base(**_):
        return base

    monkeypatch.setattr(aimc, "_base_effective_configuration", fake_base)
    return base


async def _org(session, name):
    org = OrganizationModel(provider_id=f"org-{name}")
    session.add(org)
    await session.flush()
    user = UserModel(provider_id=f"user-{name}", selected_organization_id=org.id)
    session.add(user)
    await session.flush()
    return org, user


@pytest.fixture
async def world(async_session, db_session):
    org, user = await _org(async_session, "a")
    other_org, other_user = await _org(async_session, "b")
    wf1 = await db_session.create_workflow("wf1", GRAPH, user.id, org.id)
    wf2 = await db_session.create_workflow("wf2", GRAPH, user.id, org.id)
    foreign = await db_session.create_workflow(
        "foreign", GRAPH, other_user.id, other_org.id
    )
    return {
        "org": org,
        "user": user,
        "other_org": other_org,
        "other_user": other_user,
        "wf1": wf1,
        "wf2": wf2,
        "foreign": foreign,
    }


async def _cred(client, secret=OPENAI_SECRET, ref=None, kind="api_key", payload=None):
    body = {"kind": kind, "secret": payload or {"api_key": secret}, "label": "t"}
    if ref:
        body["credential_ref"] = ref
    resp = await client.post("/api/v1/model-credentials", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _publish(client, wf, slot, cfg, cred, revision):
    body = {
        "expected_revision": revision,
        "config": cfg,
        "credential": {
            "credential_ref": cred["credential_ref"],
            "version": cred["version"],
        }
        if cred
        else None,
    }
    draft = await client.put(
        f"/api/v1/workflow/{wf.id}/model-slots/{slot}/draft", json=body
    )
    assert draft.status_code == 200, draft.text
    version = draft.json()["version"]
    valid = await client.post(
        f"/api/v1/workflow/{wf.id}/model-slots/{slot}/draft/{version}/validate"
    )
    assert valid.json()["status"] == "valid", valid.text
    pub = await client.post(
        f"/api/v1/workflow/{wf.id}/model-slots/{slot}/publish",
        json={"version": version, "expected_revision": revision + 1},
    )
    assert pub.status_code == 200, pub.text
    return pub.json()


def _slot(readback, slot):
    return next(s for s in readback["slots"] if s["slot"] == slot)


async def _read(client, wf):
    resp = await client.get(f"/api/v1/workflow/{wf.id}/model-slots")
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------- isolation
async def test_publishing_one_workflow_leaves_another_untouched(
    test_client_factory, world, fake_validator, db_session, org_default
):
    async with test_client_factory(world["user"]) as client:
        cred = await _cred(client)
        await _publish(client, world["wf1"], "llm", LLM_CFG, cred, 0)
        one, two = await _read(client, world["wf1"]), await _read(client, world["wf2"])
    assert _slot(one, "llm")["source"] == "workflow_slot"
    assert _slot(two, "llm")["source"] == "organization_default_inherited"
    assert _slot(two, "llm")["published_version"] is None
    overlay2 = await slot_settings.load_published_overlay(
        repo=db_session, workflow_id=world["wf2"].id, organization_id=world["org"].id
    )
    overlay1 = await slot_settings.load_published_overlay(
        repo=db_session, workflow_id=world["wf1"].id, organization_id=world["org"].id
    )
    assert overlay2 == {}
    assert overlay1["llm"].service.model == "gpt-4.1-mini"


async def test_slots_version_independently(test_client_factory, world, fake_validator):
    async with test_client_factory(world["user"]) as client:
        cred = await _cred(client)
        await _publish(client, world["wf1"], "llm", LLM_CFG, cred, 0)
        stt_draft = await client.put(
            f"/api/v1/workflow/{world['wf1'].id}/model-slots/stt/draft",
            json={
                "expected_revision": 0,
                "config": {
                    "provider": "deepgram",
                    "model": "nova-3-general",
                    "language": "en",
                },
                "credential": {"credential_ref": cred["credential_ref"]},
            },
        )
        assert stt_draft.status_code == 200, stt_draft.text
        readback = await _read(client, world["wf1"])
    llm, stt = _slot(readback, "llm"), _slot(readback, "stt")
    assert llm["published_version"] == 1 and llm["draft_version"] is None
    assert stt["published_version"] is None and stt["draft_version"] == 1
    assert llm["revision"] == 2 and stt["revision"] == 1
    assert _slot(readback, "tts")["revision"] == 0


async def test_group_readback_is_all_or_nothing(test_client_factory, world):
    async with test_client_factory(world["user"]) as client:
        ok = await client.get(
            f"/api/v1/model-slots?workflow_ids={world['wf1'].id},{world['wf2'].id}"
        )
        bad = await client.get(
            f"/api/v1/model-slots?workflow_ids={world['wf1'].id},{world['foreign'].id}"
        )
    assert ok.status_code == 200 and len(ok.json()["workflows"]) == 2
    assert bad.status_code == 404


# ------------------------------------------------------------ authorization
async def test_other_organization_cannot_touch_workflow_or_credential(
    test_client_factory, world, fake_validator
):
    async with test_client_factory(world["user"]) as client:
        cred = await _cred(client)
    async with test_client_factory(world["other_user"]) as client:
        read = await client.get(f"/api/v1/workflow/{world['wf1'].id}/model-slots")
        write = await client.put(
            f"/api/v1/workflow/{world['wf1'].id}/model-slots/llm/draft",
            json={"expected_revision": 0, "config": LLM_CFG},
        )
        # Own workflow, but a credential owned by another organization.
        stolen = await client.put(
            f"/api/v1/workflow/{world['foreign'].id}/model-slots/llm/draft",
            json={
                "expected_revision": 0,
                "config": LLM_CFG,
                "credential": {"credential_ref": cred["credential_ref"], "version": 1},
            },
        )
        listing = await client.get("/api/v1/model-credentials")
        revoke = await client.delete(
            f"/api/v1/model-credentials/{cred['credential_ref']}/versions/1"
        )
    assert read.status_code == 404 and write.status_code == 404
    assert stolen.status_code == 404
    assert listing.json()["credentials"] == []
    assert revoke.status_code == 404


async def test_unauthenticated_request_is_rejected(world):
    from httpx import ASGITransport, AsyncClient

    from api.app import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        resp = await c.get(f"/api/v1/workflow/{world['wf1'].id}/model-slots")
    assert resp.status_code in (401, 403)


# --------------------------------------------------- concurrency / lifecycle
async def test_stale_edit_cannot_overwrite_newer_publish(
    test_client_factory, world, fake_validator
):
    async with test_client_factory(world["user"]) as client:
        cred = await _cred(client)
        await _publish(client, world["wf1"], "llm", LLM_CFG, cred, 0)  # revision -> 2
        stale = await client.put(
            f"/api/v1/workflow/{world['wf1'].id}/model-slots/llm/draft",
            json={
                "expected_revision": 0,
                "config": {**LLM_CFG, "model": "gpt-4.1"},
                "credential": {"credential_ref": cred["credential_ref"]},
            },
        )
        readback = await _read(client, world["wf1"])
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "stale_revision"
    assert stale.json()["detail"]["current_revision"] == 2
    assert _slot(readback, "llm")["published"]["config"]["model"] == "gpt-4.1-mini"


async def test_publish_needs_a_passing_validation(
    test_client_factory, world, fake_validator
):
    async with test_client_factory(world["user"]) as client:
        cred = await _cred(client)
        path = f"/api/v1/workflow/{world['wf1'].id}/model-slots/llm"
        draft = (
            await client.put(
                f"{path}/draft",
                json={
                    "expected_revision": 0,
                    "config": LLM_CFG,
                    "credential": {"credential_ref": cred["credential_ref"]},
                },
            )
        ).json()
        unvalidated = await client.post(
            f"{path}/publish",
            json={"version": draft["version"], "expected_revision": 1},
        )
        fake_validator.errors = ["Invalid openai API key."]
        failed = await client.post(f"{path}/draft/{draft['version']}/validate")
        blocked = await client.post(
            f"{path}/publish",
            json={"version": draft["version"], "expected_revision": 1},
        )
    assert unvalidated.status_code == 409
    assert failed.json()["status"] == "invalid"
    assert blocked.status_code == 409


async def test_rollback_restores_config_and_pinned_credential(
    test_client_factory, world, fake_validator, db_session
):
    wf, org = world["wf1"], world["org"]
    async with test_client_factory(world["user"]) as client:
        cred1 = await _cred(client, OPENAI_SECRET)
        await _publish(client, wf, "llm", LLM_CFG, cred1, 0)
        cred2 = await _cred(client, OPENAI_SECRET_2, ref=cred1["credential_ref"])
        assert (
            cred2["version"] == 2 and cred2["credential_ref"] == cred1["credential_ref"]
        )

        # Rotating alone must not move the live slot off the version it pinned.
        overlay = await slot_settings.load_published_overlay(
            repo=db_session, workflow_id=wf.id, organization_id=org.id
        )
        assert overlay["llm"].service.api_key == OPENAI_SECRET

        await _publish(client, wf, "llm", {**LLM_CFG, "model": "gpt-4.1"}, cred2, 2)
        overlay = await slot_settings.load_published_overlay(
            repo=db_session, workflow_id=wf.id, organization_id=org.id
        )
        assert overlay["llm"].service.model == "gpt-4.1"
        assert overlay["llm"].service.api_key == OPENAI_SECRET_2

        rolled = await client.post(
            f"/api/v1/workflow/{wf.id}/model-slots/llm/rollback",
            json={"to_version": 1, "expected_revision": 4},
        )
        assert rolled.status_code == 200, rolled.text
        history = (
            await client.get(f"/api/v1/workflow/{wf.id}/model-slots/llm/history")
        ).json()
    assert rolled.json()["version"] == 3 and rolled.json()["origin"] == "rollback"
    overlay = await slot_settings.load_published_overlay(
        repo=db_session, workflow_id=wf.id, organization_id=org.id
    )
    assert overlay["llm"].service.model == "gpt-4.1-mini"
    assert overlay["llm"].service.api_key == OPENAI_SECRET
    assert [v["state"] for v in history["versions"]] == [
        "published",
        "superseded",
        "superseded",
    ]


async def test_cannot_revoke_a_credential_that_is_live_and_rollback_to_revoked_fails(
    test_client_factory, world, fake_validator
):
    wf = world["wf1"]
    async with test_client_factory(world["user"]) as client:
        cred1 = await _cred(client)
        await _publish(client, wf, "llm", LLM_CFG, cred1, 0)
        cred2 = await _cred(client, OPENAI_SECRET_2, ref=cred1["credential_ref"])
        in_use = await client.delete(
            f"/api/v1/model-credentials/{cred1['credential_ref']}/versions/1"
        )
        await _publish(client, wf, "llm", {**LLM_CFG, "model": "gpt-4.1"}, cred2, 2)
        revoked = await client.delete(
            f"/api/v1/model-credentials/{cred1['credential_ref']}/versions/1"
        )
        rollback = await client.post(
            f"/api/v1/workflow/{wf.id}/model-slots/llm/rollback",
            json={"to_version": 1, "expected_revision": 4},
        )
    assert (
        in_use.status_code == 409
        and in_use.json()["detail"]["code"] == "credential_in_use"
    )
    assert revoked.status_code == 200 and revoked.json()["revoked_at"]
    assert (
        rollback.status_code == 422
        and rollback.json()["detail"]["code"] == "credential_revoked"
    )


async def test_runtime_fails_closed_when_pinned_credential_is_revoked(
    test_client_factory, world, fake_validator, db_session
):
    async with test_client_factory(world["user"]) as client:
        cred = await _cred(client)
        await _publish(client, world["wf1"], "llm", LLM_CFG, cred, 0)
    await db_session.revoke_provider_credential_version(
        world["org"].id, cred["credential_ref"], 1
    )
    with pytest.raises(slot_settings.SlotResolutionError):
        await slot_settings.load_published_overlay(
            repo=db_session,
            workflow_id=world["wf1"].id,
            organization_id=world["org"].id,
        )


# -------------------------------------------------------------- legacy path
async def test_workflow_without_slots_resolves_exactly_as_before(
    world, db_session, monkeypatch
):
    base = EffectiveAIModelConfiguration(
        llm=OpenAILLMService(api_key=OPENAI_SECRET, model="gpt-4.1")
    )

    async def fake_base(**_):
        return base

    monkeypatch.setattr(aimc, "_base_effective_configuration", fake_base)
    with_id = await aimc.get_effective_ai_model_configuration_for_workflow(
        organization_id=world["org"].id,
        workflow_configurations={},
        workflow_id=world["wf1"].id,
    )
    without_id = await aimc.get_effective_ai_model_configuration_for_workflow(
        organization_id=world["org"].id, workflow_configurations={}
    )
    assert with_id is base and without_id is base


async def test_published_slot_replaces_only_its_own_slot(
    test_client_factory, world, fake_validator, monkeypatch
):
    base = EffectiveAIModelConfiguration(
        llm=OpenAILLMService(api_key="org-default-key", model="gpt-4.1"),
        embeddings=OpenAIEmbeddingsConfiguration(api_key="org-default-key"),
    )

    async def fake_base(**_):
        return base

    monkeypatch.setattr(aimc, "_base_effective_configuration", fake_base)
    async with test_client_factory(world["user"]) as client:
        cred = await _cred(client)
        await _publish(client, world["wf1"], "llm", LLM_CFG, cred, 0)
    effective = await aimc.get_effective_ai_model_configuration_for_workflow(
        organization_id=world["org"].id,
        workflow_configurations={},
        workflow_id=world["wf1"].id,
    )
    assert (
        effective.llm.model == "gpt-4.1-mini" and effective.llm.api_key == OPENAI_SECRET
    )
    assert effective.embeddings.api_key == "org-default-key"
    assert base.llm.model == "gpt-4.1"  # base object untouched


async def test_legacy_override_is_reported_as_such(
    test_client_factory, world, async_session, org_default
):
    definition = await async_session.get(
        WorkflowDefinitionModel, world["wf1"].released_definition_id
    )
    definition.workflow_configurations = {"model_overrides": {"llm": {"model": "x"}}}
    await async_session.flush()
    async with test_client_factory(world["user"]) as client:
        readback = await _read(client, world["wf1"])
    assert _slot(readback, "llm")["source"] == "legacy_workflow_override"
    assert _slot(readback, "stt")["source"] == "organization_default_inherited"


async def test_org_template_is_copied_once_and_later_changes_do_not_reach_the_workflow(
    world, db_session, monkeypatch
):
    template = EffectiveAIModelConfiguration(
        llm=OpenAILLMService(api_key=OPENAI_SECRET, model="gpt-4.1-mini")
    )

    class Resolved:
        effective = template

    async def resolved(**_):
        return Resolved()

    monkeypatch.setattr(aimc, "get_resolved_ai_model_configuration", resolved)
    seeded = await slot_settings.apply_organization_template(
        repo=db_session,
        organization_id=world["org"].id,
        workflow_id=world["wf1"].id,
        created_by="t",
    )
    assert seeded == ["llm"]

    # Organization default changes afterwards.
    changed = EffectiveAIModelConfiguration(
        llm=OpenAILLMService(api_key="new-org-key", model="gpt-5")
    )

    async def fake_base(**_):
        return changed

    monkeypatch.setattr(aimc, "_base_effective_configuration", fake_base)
    effective = await aimc.get_effective_ai_model_configuration_for_workflow(
        organization_id=world["org"].id,
        workflow_configurations={},
        workflow_id=world["wf1"].id,
    )
    other = await aimc.get_effective_ai_model_configuration_for_workflow(
        organization_id=world["org"].id,
        workflow_configurations={},
        workflow_id=world["wf2"].id,
    )
    assert (
        effective.llm.model == "gpt-4.1-mini" and effective.llm.api_key == OPENAI_SECRET
    )
    assert (
        other.llm.model == "gpt-5"
    )  # a workflow without its own slot still inherits (legacy)
    # A second seed never overwrites.
    again = await slot_settings.apply_organization_template(
        repo=db_session,
        organization_id=world["org"].id,
        workflow_id=world["wf1"].id,
        created_by="t",
    )
    assert again == []


async def test_backfill_plan_is_secret_free_and_apply_keeps_effective_config(
    world, db_session
):
    effective = EffectiveAIModelConfiguration(
        llm=OpenAILLMService(api_key=OPENAI_SECRET, model="gpt-4.1"),
        embeddings=OpenAIEmbeddingsConfiguration(api_key=OPENAI_SECRET),
    )
    service = slot_settings.WorkflowSlotService(db_session)
    plan = await service.plan_snapshot(effective=effective)
    assert {p["slot"] for p in plan} == {"llm", "embeddings"}
    assert all(p["kind"] == "api_key" for p in plan)
    assert all(OPENAI_SECRET not in json.dumps(p["config"]) for p in plan)
    seeded = await service.seed_workflow_from_effective(
        organization_id=world["org"].id,
        workflow_id=world["wf1"].id,
        effective=effective,
        origin="backfill",
        created_by="t",
    )
    assert sorted(seeded) == ["embeddings", "llm"]
    overlay = await slot_settings.load_published_overlay(
        repo=db_session, workflow_id=world["wf1"].id, organization_id=world["org"].id
    )
    assert overlay["llm"].service.model_dump() == effective.llm.model_dump()


# ------------------------------------------------------------------ secrets
async def test_secrets_never_appear_in_responses_and_are_encrypted_at_rest(
    test_client_factory, world, fake_validator, async_session
):
    async with test_client_factory(world["user"]) as client:
        cred = await _cred(client)
        await _publish(client, world["wf1"], "llm", LLM_CFG, cred, 0)
        bodies = [
            json.dumps(cred),
            (await client.get("/api/v1/model-credentials")).text,
            (await client.get(f"/api/v1/workflow/{world['wf1'].id}/model-slots")).text,
            (
                await client.get(
                    f"/api/v1/workflow/{world['wf1'].id}/model-slots/llm/history"
                )
            ).text,
            (
                await client.get(
                    f"/api/v1/workflow/{world['wf1'].id}/effective-model-configuration"
                )
            ).text,
        ]
    for body in bodies:
        assert OPENAI_SECRET not in body
    from sqlalchemy import select

    row = (await async_session.execute(select(ProviderCredentialModel))).scalars().one()
    assert OPENAI_SECRET not in row.ciphertext
    assert row.ciphertext.startswith("gAAAA")
    assert row.key_id == secret_store.primary_key_id()


async def test_secret_in_slot_config_is_rejected_without_echo(
    test_client_factory, world, fake_validator
):
    async with test_client_factory(world["user"]) as client:
        resp = await client.put(
            f"/api/v1/workflow/{world['wf1'].id}/model-slots/llm/draft",
            json={
                "expected_revision": 0,
                "config": {**LLM_CFG, "api_key": OPENAI_SECRET},
            },
        )
    assert (
        resp.status_code == 422 and resp.json()["detail"]["code"] == "secret_in_config"
    )
    assert OPENAI_SECRET not in resp.text


async def test_request_validation_error_does_not_echo_secret(
    test_client_factory, world
):
    async with test_client_factory(world["user"]) as client:
        resp = await client.post(
            "/api/v1/model-credentials",
            json={"kind": "not-a-kind", "secret": {"api_key": OPENAI_SECRET}},
        )
    assert resp.status_code == 422
    assert OPENAI_SECRET not in resp.text and '"input"' not in resp.text


async def test_masked_placeholder_credential_is_rejected(test_client_factory, world):
    async with test_client_factory(world["user"]) as client:
        resp = await client.post(
            "/api/v1/model-credentials",
            json={"kind": "api_key", "secret": {"api_key": "************abcd***"}},
        )
    assert resp.status_code == 422


async def test_secret_store_fails_closed(test_client_factory, world, monkeypatch):
    monkeypatch.delenv(secret_store.KEYS_ENV)
    async with test_client_factory(world["user"]) as client:
        resp = await client.post(
            "/api/v1/model-credentials",
            json={"kind": "api_key", "secret": {"api_key": OPENAI_SECRET}},
        )
    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "secret_store_unavailable"
    monkeypatch.setenv(secret_store.KEYS_ENV, "not-a-fernet-key")
    with pytest.raises(secret_store.SecretStoreUnavailable):
        secret_store.encrypt_credential(
            organization_id=1, credential_ref="c", version=1, payload={"api_key": "x"}
        )


async def test_ciphertext_is_bound_to_its_reference(
    test_client_factory, world, fake_validator, async_session, db_session
):
    async with test_client_factory(world["user"]) as client:
        a = await _cred(client, OPENAI_SECRET)
        b = await _cred(client, OPENAI_SECRET_2)
    from sqlalchemy import select

    rows = {
        r.credential_ref: r
        for r in (
            await async_session.execute(select(ProviderCredentialModel))
        ).scalars()
    }
    rows[a["credential_ref"]].ciphertext = rows[b["credential_ref"]].ciphertext
    with pytest.raises(secret_store.SecretStoreCorrupt):
        await slot_settings.resolve_credential_secret(
            db_session, world["org"].id, a["credential_ref"], 1
        )


async def test_key_rotation_reads_old_and_new_ciphertext(monkeypatch):
    old, new = Fernet.generate_key().decode(), Fernet.generate_key().decode()
    monkeypatch.setenv(secret_store.KEYS_ENV, old)
    token, old_id = secret_store.encrypt_credential(
        organization_id=1, credential_ref="c", version=1, payload={"api_key": "k1"}
    )
    monkeypatch.setenv(secret_store.KEYS_ENV, f"{new},{old}")
    assert secret_store.decrypt_credential(
        organization_id=1, credential_ref="c", version=1, ciphertext=token
    ) == {"api_key": "k1"}
    rotated, new_id = secret_store.reencrypt(token)
    assert new_id != old_id
    monkeypatch.setenv(secret_store.KEYS_ENV, new)  # old key dropped
    assert secret_store.decrypt_credential(
        organization_id=1, credential_ref="c", version=1, ciphertext=rotated
    ) == {"api_key": "k1"}


def test_validation_error_text_carries_no_input():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as caught:
        AzureLLMService(api_key=OPENAI_SECRET, model="m")  # endpoint missing
    # Pydantic elides the middle of long values but still prints both ends.
    assert "sk-TESTSECRE" in str(caught.value) and "input_value" in str(caught.value)
    safe = str(safe_exception_detail(caught.value))
    assert "sk-TESTSEC" not in safe and "input_value" not in safe and "endpoint" in safe


def test_invalid_stored_org_configuration_is_logged_without_secret():
    from api.db.models import OrganizationConfigurationModel

    row = OrganizationConfigurationModel(
        organization_id=1,
        key="model_configuration_v2",
        value={
            "mode": "byok",
            "byok": {
                "mode": "pipeline",
                "pipeline": {
                    "llm": {"provider": "openai", "api_key": OPENAI_SECRET, "model": 5}
                },
            },
        },
    )
    captured: list[str] = []
    sink = logger.add(lambda m: captured.append(str(m)), level="DEBUG")
    try:
        assert aimc._parse_organization_ai_model_configuration_v2(row, 1) is None
    finally:
        logger.remove(sink)
    assert captured
    assert not any("sk-TESTSEC" in line or "input_value" in line for line in captured)


# ------------------------------------------------------------------- Vertex
VERTEX = {"provider": "google_vertex", "location": "global"}


@pytest.mark.parametrize(
    "slot,model",
    [
        # STT only accepts a key together with project_id (complete resource);
        # TTS accepts one through the Vertex API path. See the speech cut-over test.
        ("stt", "gemini-3.5-transcribe-live-preview"),
        ("embeddings", "gemini-embedding-001"),
    ],
)
async def test_vertex_api_key_is_refused_outside_the_llm_slot(
    test_client_factory, world, slot, model
):
    async with test_client_factory(world["user"]) as client:
        cred = await _cred(client, "AQ.vertexkey123456")
        resp = await client.put(
            f"/api/v1/workflow/{world['wf1'].id}/model-slots/{slot}/draft",
            json={
                "expected_revision": 0,
                "config": {**VERTEX, "model": model},
                "credential": {"credential_ref": cred["credential_ref"]},
            },
        )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "vertex_auth_not_supported"
    assert "AQ.vertexkey123456" not in resp.text


async def test_vertex_llm_accepts_api_key_or_service_account_or_adc(
    test_client_factory, world
):
    sa = json.dumps(
        {
            "type": "service_account",
            "project_id": "p",
            "client_email": "s@p.iam.gserviceaccount.com",
            "private_key": SA_PRIVATE,
        }
    )
    cfg = {**VERTEX, "model": "gemini-3.5-flash", "project_id": "p"}
    async with test_client_factory(world["user"]) as client:
        key = await _cred(client, "AQ.vertexkey123456")
        acct = await _cred(
            client, kind="service_account_json", payload={"credentials": sa}
        )
        results = []
        revision = 0
        for cred in (key, acct, None):
            resp = await client.put(
                f"/api/v1/workflow/{world['wf1'].id}/model-slots/llm/draft",
                json={
                    "expected_revision": revision,
                    "config": cfg,
                    "credential": {"credential_ref": cred["credential_ref"]}
                    if cred
                    else None,
                },
            )
            results.append(resp.status_code)
            revision += 1
        readback = await _read(client, world["wf1"])
    assert results == [200, 200, 200]
    assert SA_PRIVATE not in json.dumps(readback)


async def test_credential_kind_must_match_the_provider(test_client_factory, world):
    async with test_client_factory(world["user"]) as client:
        key = await _cred(client)
        resp = await client.put(
            f"/api/v1/workflow/{world['wf1'].id}/model-slots/tts/draft",
            json={
                "expected_revision": 0,
                "config": {"provider": "google", "model": "chirp_3_hd"},
                "credential": {"credential_ref": key["credential_ref"]},
            },
        )
        missing = await client.put(
            f"/api/v1/workflow/{world['wf1'].id}/model-slots/llm/draft",
            json={"expected_revision": 0, "config": LLM_CFG},
        )
    assert (
        resp.status_code == 422
        and resp.json()["detail"]["code"] == "credential_kind_mismatch"
    )
    assert (
        missing.status_code == 422
        and missing.json()["detail"]["code"] == "credential_required"
    )


async def test_unknown_provider_and_bad_field_are_rejected(test_client_factory, world):
    async with test_client_factory(world["user"]) as client:
        unknown = await client.put(
            f"/api/v1/workflow/{world['wf1'].id}/model-slots/llm/draft",
            json={"expected_revision": 0, "config": {"provider": "nope", "model": "m"}},
        )
        bad = await client.put(
            f"/api/v1/workflow/{world['wf1'].id}/model-slots/stt/draft",
            json={
                "expected_revision": 0,
                "config": {"provider": "deepgram", "model": 5},
            },
        )
    assert unknown.json()["detail"]["code"] == "unknown_provider"
    assert bad.json()["detail"]["code"] == "invalid_config"


# --------------------------------------------------------------- embeddings
async def _index(async_session, org, user, model, dimension=1536):
    doc = KnowledgeBaseDocumentModel(
        organization_id=org.id,
        created_by=user.id,
        filename="a.txt",
        file_size_bytes=1,
        file_hash="h",
        mime_type="text/plain",
        source_url=None,
        processing_status="completed",
        total_chunks=1,
        retrieval_mode="chunked",
    )
    async_session.add(doc)
    await async_session.flush()
    async_session.add(
        KnowledgeBaseChunkModel(
            document_id=doc.id,
            organization_id=org.id,
            chunk_text="x",
            chunk_index=0,
            chunk_metadata={},
            embedding_model=model,
            embedding_dimension=dimension,
            token_count=1,
            embedding=[0.1] * 1536,
        )
    )
    await async_session.flush()
    return doc


async def test_embedding_change_is_blocked_when_it_would_orphan_the_index(
    test_client_factory, world, fake_validator, async_session
):
    await _index(async_session, world["org"], world["user"], "text-embedding-3-small")
    async with test_client_factory(world["user"]) as client:
        key = await _cred(client)
        vertex_sa = await _cred(
            client,
            kind="service_account_json",
            payload={
                "credentials": json.dumps(
                    {
                        "type": "service_account",
                        "project_id": "p",
                        "client_email": "s@p.iam.gserviceaccount.com",
                        "private_key": SA_PRIVATE,
                    }
                )
            },
        )
        path = f"/api/v1/workflow/{world['wf1'].id}/model-slots/embeddings"
        draft = await client.put(
            f"{path}/draft",
            json={
                "expected_revision": 0,
                "config": {
                    **VERTEX,
                    "model": "gemini-embedding-001",
                    "project_id": "p",
                },
                "credential": {"credential_ref": vertex_sa["credential_ref"]},
            },
        )
        assert draft.status_code == 200, draft.text
        validation = await client.post(f"{path}/draft/1/validate")
        same = await client.put(
            f"{path}/draft",
            json={
                "expected_revision": 1,
                "config": {"provider": "openai", "model": "text-embedding-3-small"},
                "credential": {"credential_ref": key["credential_ref"]},
            },
        )
        ok = await client.post(f"{path}/draft/2/validate")
        published = await client.post(
            f"{path}/publish", json={"version": 2, "expected_revision": 2}
        )
    assert validation.json()["status"] == "invalid"
    assert (
        "already indexed" in validation.json()["errors"][0]
        or "indexed with" in validation.json()["errors"][0]
    )
    assert same.status_code == 200 and ok.json()["status"] == "valid"
    assert published.status_code == 200


async def test_publish_rechecks_embeddings_if_index_changed_after_validation(
    test_client_factory, world, fake_validator, async_session
):
    async with test_client_factory(world["user"]) as client:
        key = await _cred(client)
        path = f"/api/v1/workflow/{world['wf1'].id}/model-slots/embeddings"
        await client.put(
            f"{path}/draft",
            json={
                "expected_revision": 0,
                "config": {"provider": "openai", "model": "text-embedding-3-large"},
                "credential": {"credential_ref": key["credential_ref"]},
            },
        )
        assert (await client.post(f"{path}/draft/1/validate")).json()[
            "status"
        ] == "valid"
        await _index(
            async_session, world["org"], world["user"], "text-embedding-3-small"
        )
        blocked = await client.post(
            f"{path}/publish", json={"version": 1, "expected_revision": 1}
        )
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "embedding_index_incompatible"


async def test_zero_results_from_a_model_mismatch_is_an_error_not_an_empty_list(
    world, db_session, async_session
):
    await _index(async_session, world["org"], world["user"], "text-embedding-3-small")
    with pytest.raises(EmbeddingIndexMismatchError):
        await db_session.search_similar_chunks(
            query_embedding=[0.1] * 1536,
            organization_id=world["org"].id,
            embedding_model="gemini-embedding-001",
        )
    # Same model: a genuine "no match" stays a plain result.
    hits = await db_session.search_similar_chunks(
        query_embedding=[0.1] * 1536,
        organization_id=world["org"].id,
        embedding_model="text-embedding-3-small",
        document_uuids=["00000000-0000-0000-0000-000000000000"],
    )
    assert hits == []


async def test_embedding_signatures_are_scoped_to_the_organization(
    world, db_session, async_session
):
    await _index(async_session, world["org"], world["user"], "text-embedding-3-small")
    assert (
        await db_session.get_organization_embedding_signatures(world["other_org"].id)
        == []
    )


# ------------------------------------------------- speech cut-over and recovery
GEMINI_PROJECT = "project-943ab014-8a47-48e4-a92"
GEMINI_STT = {
    "provider": "google_vertex",
    "model": "gemini-3.5-transcribe-live-preview",
    "language": "en-US",
    "project_id": GEMINI_PROJECT,
    "location": "global",
}
GEMINI_TTS = {
    "provider": "google_vertex",
    "model": "gemini-2.5-flash-tts",
    "voice": "Kore",
    "language": "en-US",
    "project_id": GEMINI_PROJECT,
    "location": "global",
}
VERTEX_KEY = "AQ.TESTVERTEXKEY-0123456789"


async def test_gemini_speech_cutover_can_be_rolled_back_to_the_frozen_legacy_config(
    test_client_factory, world, fake_validator, org_default, db_session
):
    wf, org = world["wf1"], world["org"]
    service = slot_settings.WorkflowSlotService(db_session)
    # Step 0 for a workflow that already has an llm slot but no stt/tts history:
    # freeze what it runs today as v1, which the backfill does slot by slot.
    async with test_client_factory(world["user"]) as client:
        cred = await _cred(client, VERTEX_KEY)
        llm = await _publish(client, wf, "llm", LLM_CFG, await _cred(client), 0)
        seeded = await service.seed_workflow_from_effective(
            organization_id=org.id,
            workflow_id=wf.id,
            effective=org_default,
            origin="backfill",
            created_by="t",
        )
        assert sorted(seeded) == ["embeddings", "stt", "tts"]  # llm skipped

        async def revision(slot):
            return _slot(await _read(client, wf), slot)["revision"]

        stt_rev, tts_rev = await revision("stt"), await revision("tts")
        await _publish(client, wf, "stt", GEMINI_STT, cred, stt_rev)
        await _publish(client, wf, "tts", GEMINI_TTS, cred, tts_rev)
        overlay = await slot_settings.load_published_overlay(
            repo=db_session, workflow_id=wf.id, organization_id=org.id
        )
        assert overlay["stt"].service.provider == "google_vertex"
        assert overlay["stt"].service.project_id == GEMINI_PROJECT
        assert overlay["tts"].service.api_key == VERTEX_KEY
        assert overlay["llm"].service.model == "gpt-4.1-mini"

        for slot in ("stt", "tts"):
            rolled = await client.post(
                f"/api/v1/workflow/{wf.id}/model-slots/{slot}/rollback",
                json={"to_version": 1, "expected_revision": await revision(slot)},
            )
            assert rolled.status_code == 200, rolled.text

    overlay = await slot_settings.load_published_overlay(
        repo=db_session, workflow_id=wf.id, organization_id=org.id
    )
    assert overlay["stt"].service.model_dump() == org_default.stt.model_dump()
    assert overlay["tts"].service.model_dump() == org_default.tts.model_dump()
    assert overlay["llm"].service.model == "gpt-4.1-mini"  # never touched
    assert llm["version"] == 1


async def test_first_publish_without_a_frozen_v1_has_nothing_to_roll_back_to(
    test_client_factory, world, fake_validator
):
    wf = world["wf1"]
    async with test_client_factory(world["user"]) as client:
        cred = await _cred(client, VERTEX_KEY)
        await _publish(client, wf, "stt", GEMINI_STT, cred, 0)
        rolled = await client.post(
            f"/api/v1/workflow/{wf.id}/model-slots/stt/rollback",
            json={"to_version": 1, "expected_revision": 2},
        )
    assert rolled.status_code == 409  # v1 is the live Gemini config itself
