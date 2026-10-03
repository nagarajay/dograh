"""True concurrent tests for the credential revoke / slot publication invariant.

Invariant: a published slot never pins a revoked credential version, whichever of
publish, rollback, seed or revoke commits first.

The transactional fixtures used elsewhere cannot show this: they run everything
in one connection. These tests use a throwaway database (created, migrated and
dropped here) and real, separate connections, so row locks are actually
contended.
"""

import asyncio
import os
import uuid
from urllib.parse import urlparse, urlunparse

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from api.db.db_client import DBClient
from api.db.models import (
    OrganizationModel,
    ProviderCredentialModel,
    UserModel,
)
from api.db.workflow_slot_client import (
    CredentialInUseError,
    CredentialRevokedError,
    SlotRevisionConflict,
    SlotStateError,
)

pytestmark = pytest.mark.asyncio

GRAPH = {
    "nodes": [
        {"id": "1", "type": "startCall", "data": {"name": "Start", "prompt": "Hi"}},
        {"id": "2", "type": "endCall", "data": {"name": "End", "prompt": "Bye"}},
    ],
    "edges": [{"id": "e1", "source": "1", "target": "2", "data": {"label": "End"}}],
}
CFG = {"provider": "openai", "model": "gpt-4.1-mini"}
TIMEOUT = 20  # a deadlock would show up as a timeout, not a hang


def _url(dbname: str) -> str:
    parsed = urlparse(os.environ["DATABASE_URL"])
    return urlunparse(parsed._replace(path=f"/{dbname}"))


@pytest.fixture(scope="module")
async def client():
    from conftest import run_migrations

    name = f"race_{uuid.uuid4().hex[:10]}"
    admin = create_async_engine(
        _url("postgres"), poolclass=NullPool, isolation_level="AUTOCOMMIT"
    )
    async with admin.connect() as conn:
        await conn.execute(text(f'CREATE DATABASE "{name}" TEMPLATE template0'))
    await run_migrations(_url(name))
    engine = create_async_engine(_url(name), poolclass=NullPool)
    db = object.__new__(DBClient)
    db.engine = engine
    db.async_session = async_sessionmaker(engine, expire_on_commit=False)
    yield db
    await engine.dispose()
    async with admin.connect() as conn:
        await conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    await admin.dispose()


@pytest.fixture
async def world(client):
    tag = uuid.uuid4().hex[:8]
    async with client.async_session() as session:
        org = OrganizationModel(provider_id=f"org-{tag}")
        session.add(org)
        await session.flush()
        user = UserModel(provider_id=f"user-{tag}", selected_organization_id=org.id)
        session.add(user)
        await session.commit()
        org_id, user_id = org.id, user.id
    workflow = await client.create_workflow(f"wf-{tag}", GRAPH, user_id, org_id)
    return {"org": org_id, "user": user_id, "wf": workflow.id}


async def new_credential(client, org, ref=None):
    return await client.create_provider_credential_version(
        organization_id=org,
        credential_ref=ref,
        kind="api_key",
        label="t",
        source_ref=None,
        created_by="t",
        encrypt=lambda r, v: (f"ciphertext-{r}-{v}", "key0"),
    )


async def valid_draft(client, wf, slot, cred, revision):
    """Save and validate a draft pinned to ``cred``; returns (version, new_revision)."""
    row = await client.save_slot_draft(
        workflow_id=wf,
        slot=slot,
        config=CFG,
        credential_ref=cred.credential_ref,
        credential_version=cred.version,
        expected_revision=revision,
        created_by="t",
        change_note=None,
    )
    await client.mark_slot_validation(
        workflow_id=wf, slot=slot, version=row.version, status="valid"
    )
    return row.version, revision + 1


async def attempt(coro):
    try:
        await asyncio.wait_for(coro, TIMEOUT)
        return "ok"
    except CredentialRevokedError:
        return "revoked"
    except CredentialInUseError:
        return "in_use"
    except (SlotStateError, SlotRevisionConflict):
        return "state"


async def assert_invariant(client, org, ref, version):
    """No published slot of the org pins a revoked version of this credential."""
    async with client.async_session() as session:
        cred = (
            await session.execute(
                select(ProviderCredentialModel).where(
                    ProviderCredentialModel.organization_id == org,
                    ProviderCredentialModel.credential_ref == ref,
                    ProviderCredentialModel.version == version,
                )
            )
        ).scalar_one()
    in_use = await client.count_slot_usages_of_credential(
        organization_id=org, credential_ref=ref, version=version
    )
    assert not (cred.revoked_at is not None and in_use), (
        "a published slot pins a revoked credential"
    )
    return cred.revoked_at is not None, in_use


async def test_publish_and_revoke_never_both_win(client, world):
    outcomes = {"published": 0, "revoked": 0}
    for _ in range(25):
        cred = await new_credential(client, world["org"])
        version, revision = await valid_draft(
            client, world["wf"], "llm", cred, await _revision(client, world, "llm")
        )
        publish, revoke = await asyncio.gather(
            attempt(
                client.publish_slot(
                    workflow_id=world["wf"],
                    slot="llm",
                    version=version,
                    expected_revision=revision,
                )
            ),
            attempt(
                client.revoke_provider_credential_version(
                    world["org"], cred.credential_ref, cred.version
                )
            ),
        )
        revoked, in_use = await assert_invariant(
            client, world["org"], cred.credential_ref, cred.version
        )
        # Exactly one side wins.
        assert (publish, revoke) in {("ok", "in_use"), ("revoked", "ok")}, (
            publish,
            revoke,
        )
        outcomes["published" if publish == "ok" else "revoked"] += 1
        if publish == "ok":
            assert in_use == 1 and not revoked
            # Make the credential free again for the next trial's slot state.
    assert sum(outcomes.values()) == 25


async def _revision(client, world, slot):
    state = await client.get_slot_state(world["wf"], slot)
    return state.revision if state else 0


async def test_revoke_waits_for_an_in_flight_publication(client, world):
    """Hold a publisher's shared credential lock open: revoke must queue behind it."""
    cred = await new_credential(client, world["org"])
    version, revision = await valid_draft(client, world["wf"], "stt", cred, 0)
    session = client.async_session()
    await session.__aenter__()
    try:
        await client._lock_credentials_shared(
            session, world["org"], [(cred.credential_ref, cred.version)]
        )
        revoke = asyncio.ensure_future(
            attempt(
                client.revoke_provider_credential_version(
                    world["org"], cred.credential_ref, cred.version
                )
            )
        )
        await asyncio.sleep(0.5)
        assert not revoke.done(), "revoke did not wait for the publisher's lock"
        # The publisher finishes: the slot is now published on the credential.
        await session.rollback()
    finally:
        await session.__aexit__(None, None, None)
    # (the shared lock is gone and nothing was published, so revoke succeeds)
    assert await revoke == "ok"


async def test_publish_waits_for_an_in_flight_revocation_then_refuses(client, world):
    cred = await new_credential(client, world["org"])
    version, revision = await valid_draft(client, world["wf"], "tts", cred, 0)
    session = client.async_session()
    await session.__aenter__()
    try:
        row = (
            await session.execute(
                select(ProviderCredentialModel)
                .where(
                    ProviderCredentialModel.credential_ref == cred.credential_ref,
                    ProviderCredentialModel.version == cred.version,
                )
                .with_for_update()
            )
        ).scalar_one()
        publish = asyncio.ensure_future(
            attempt(
                client.publish_slot(
                    workflow_id=world["wf"],
                    slot="tts",
                    version=version,
                    expected_revision=revision,
                )
            )
        )
        await asyncio.sleep(0.5)
        assert not publish.done(), "publish did not wait for the revoker's lock"
        from datetime import UTC, datetime

        row.revoked_at = datetime.now(UTC)
        await session.commit()
    finally:
        await session.__aexit__(None, None, None)
    assert await publish == "revoked"
    state = await client.get_slot_state(world["wf"], "tts")
    assert state.published_version is None  # nothing went live
    await assert_invariant(client, world["org"], cred.credential_ref, cred.version)


async def test_rollback_and_revoke_never_both_win(client, world):
    for _ in range(15):
        old = await new_credential(client, world["org"])
        new = await new_credential(client, world["org"])
        revision = await _revision(client, world, "embeddings")
        v1, revision = await valid_draft(
            client, world["wf"], "embeddings", old, revision
        )
        await client.publish_slot(
            workflow_id=world["wf"],
            slot="embeddings",
            version=v1,
            expected_revision=revision,
        )
        v2, revision = await valid_draft(
            client, world["wf"], "embeddings", new, revision + 1
        )
        await client.publish_slot(
            workflow_id=world["wf"],
            slot="embeddings",
            version=v2,
            expected_revision=revision,
        )
        revision = await _revision(client, world, "embeddings")
        # ``old`` is no longer live, so it can be revoked: race that with a
        # rollback to the version that pins it.
        rollback, revoke = await asyncio.gather(
            attempt(
                client.rollback_slot(
                    workflow_id=world["wf"],
                    slot="embeddings",
                    to_version=v1,
                    expected_revision=revision,
                    created_by="t",
                    change_note=None,
                )
            ),
            attempt(
                client.revoke_provider_credential_version(
                    world["org"], old.credential_ref, old.version
                )
            ),
        )
        assert (rollback, revoke) in {("ok", "in_use"), ("revoked", "ok")}, (
            rollback,
            revoke,
        )
        await assert_invariant(client, world["org"], old.credential_ref, old.version)


async def test_seed_and_revoke_never_both_win(client, world):
    # A fresh workflow per trial: seeding only touches never-used slots.
    for index in range(10):
        async with client.async_session() as session:
            user = await session.get(UserModel, world["user"])
        wf = await client.create_workflow(f"seed-{index}", GRAPH, user.id, world["org"])
        cred = await new_credential(client, world["org"])
        seed, revoke = await asyncio.gather(
            attempt(
                client.seed_published_slots(
                    workflow_id=wf.id,
                    slots={
                        "llm": {
                            "config": CFG,
                            "credential_ref": cred.credential_ref,
                            "credential_version": cred.version,
                        }
                    },
                    origin="template",
                    created_by="t",
                )
            ),
            attempt(
                client.revoke_provider_credential_version(
                    world["org"], cred.credential_ref, cred.version
                )
            ),
        )
        assert (seed, revoke) in {("ok", "in_use"), ("revoked", "ok")}, (seed, revoke)
        await assert_invariant(client, world["org"], cred.credential_ref, cred.version)


async def test_many_concurrent_writers_do_not_deadlock(client, world):
    """Two workflows share one credential; publishes, rollbacks and revokes race."""
    async with client.async_session() as session:
        user = await session.get(UserModel, world["user"])
    wf2 = await client.create_workflow("second", GRAPH, user.id, world["org"])
    cred = await new_credential(client, world["org"])
    ready = []
    for wf in (world["wf"], wf2.id):
        for slot in ("llm", "stt"):
            revision = await _revision(client, {"wf": wf}, slot)
            version, revision = await valid_draft(client, wf, slot, cred, revision)
            ready.append((wf, slot, version, revision))
    results = await asyncio.gather(
        *(
            attempt(
                client.publish_slot(
                    workflow_id=wf,
                    slot=slot,
                    version=version,
                    expected_revision=revision,
                )
            )
            for wf, slot, version, revision in ready
        ),
        *(
            attempt(
                client.revoke_provider_credential_version(
                    world["org"], cred.credential_ref, cred.version
                )
            )
            for _ in range(3)
        ),
    )
    # Nothing timed out (no deadlock) and the invariant holds.
    assert all(r in {"ok", "revoked", "in_use", "state"} for r in results)
    await assert_invariant(client, world["org"], cred.credential_ref, cred.version)
