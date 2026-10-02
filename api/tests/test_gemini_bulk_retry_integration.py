"""Real Postgres constraints and ARQ Redis semantics; isolated disposable services only.

Run with GEMINI_BULK_INTEGRATION=1 and gemini-bulk-postgres/gemini-bulk-redis.
Provider and storage calls are substituted; live deployment verifies those boundaries.
"""

import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from arq import create_pool, func
from arq.connections import RedisSettings
from arq.worker import Worker
from sqlalchemy import text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from api.db.gemini_tts_sample_client import GeminiTTSSampleClient
from api.db.models import GeminiTTSSampleAssetModel as Asset
from api.db.models import GeminiTTSSamplePackModel as Pack
from api.routes.gemini_tts_samples import _pack_response
from api.services import gemini_tts_sample_packs as packs
from api.services.configuration.options.google_vertex_catalog import (
    gemini_tts_catalog_revision,
    gemini_tts_voices,
)
from api.services.gemini_tts_sample_library import pcm_to_wav
from api.tasks import gemini_tts_samples as tasks

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("GEMINI_BULK_INTEGRATION") != "1",
        reason="disposable integration services required",
    ),
]
MODEL = "gemini-3.1-flash-tts-preview"
VOICES = [{"voice_id": v.id, "gender": v.gender} for v in gemini_tts_voices(MODEL)]


@pytest.fixture
async def live(monkeypatch):
    # Never run destructive fixture setup against the development Supabase DB.
    assert "@gemini-bulk-postgres:" in os.environ["DATABASE_URL"]
    assert os.environ["REDIS_URL"] == "redis://gemini-bulk-redis:6379"
    engine = create_async_engine(os.environ["DATABASE_URL"], poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda c: Pack.metadata.create_all(
                c, tables=[Pack.__table__, Asset.__table__]
            )
        )
        await conn.execute(
            text(
                "TRUNCATE gemini_tts_sample_assets, gemini_tts_sample_packs RESTART IDENTITY CASCADE"
            )
        )
    client = object.__new__(GeminiTTSSampleClient)
    client.engine = engine
    client.async_session = async_sessionmaker(engine)
    redis = await create_pool(RedisSettings(host="gemini-bulk-redis"))
    await redis.flushdb()
    monkeypatch.setattr(packs, "db_client", client)
    monkeypatch.setattr(tasks, "db_client", client)
    monkeypatch.setattr(packs, "enqueue_job", redis.enqueue_job)
    monkeypatch.setenv("GEMINI_TTS_SAMPLE_CONCURRENCY", "1")
    yield client, redis
    await redis.aclose()
    await engine.dispose()


async def seed(client, assets):
    pack, _ = await client.create_or_get_pack(
        pack={
            "provider": "google_vertex",
            "model_id": MODEL,
            "catalog_revision": gemini_tts_catalog_revision(MODEL),
            "location": "global",
            "language": "en-IN",
            "style_text": "warm",
            "sample_text": "disposable test",
            "request_fingerprint": "test-only",
            "status": "partial",
        },
        assets=assets,
    )
    return pack.id


def asset(voice, version=1, status="failed", current=False, **extra):
    return dict(
        voice_id=voice,
        gender="Female",
        version=version,
        status=status,
        is_current=current,
        **extra,
    )


async def test_old_bulk_update_reproduces_actual_constraint_failure(live):
    client, _ = live
    pack_id = await seed(
        client,
        [
            asset("Achernar"),
            asset("Achernar", 2),
            asset(
                "Achernar",
                3,
                "completed",
                True,
                storage_key="old.wav",
                sha256="old-hash",
            ),
        ],
    )
    async with client.async_session() as session:
        with pytest.raises(IntegrityError, match="uq_gemini_tts_sample_asset_active"):
            await session.execute(
                update(Asset)
                .where(Asset.pack_id == pack_id, Asset.status == "failed")
                .values(status="queued")
            )
        await session.rollback()
    pack = await client.get_pack(pack_id)
    assert [a.status for a in pack.assets] == ["failed", "failed", "completed"]


async def test_selection_preserves_history_playable_and_active_voices(live):
    client, redis = live
    pack_id = await seed(
        client,
        [
            asset("Achernar"),
            asset(
                "Achernar",
                2,
                "completed",
                True,
                storage_key="old.wav",
                sha256="old-hash",
            ),
            asset("Achird"),
            asset("Achird", 2),
            asset("Algenib"),
            asset("Algenib", 2, "queued"),
            asset("Algieba"),
            asset("Algieba", 2, "running"),
        ],
    )
    row, summary = await packs.retry_failed_sample_pack(pack_id)
    assert summary["eligible_voices"] == summary["enqueued_jobs"] == 1
    assert summary["skipped_playable_voices"] == 1
    assert summary["skipped_active_voices"] == 2
    new = await client.get_asset(summary["asset_ids"][0])
    assert (new.voice_id, new.version, new.status) == ("Achird", 3, "queued")
    assert await client.claim_asset(new.id)  # scheduling transaction is committed
    assert not await client.claim_asset(new.id)
    assert await redis.zcard("arq:queue") == 1
    old = row.assets[1]
    assert (old.status, old.is_current, old.storage_key, old.sha256) == (
        "completed",
        True,
        "old.wav",
        "old-hash",
    )
    assert all(a.status == "failed" for a in row.assets if a.version == 1)


async def test_concurrent_bulk_requests_do_not_duplicate_jobs_or_synthesis(
    live, monkeypatch
):
    client, redis = live
    pack_id = await seed(
        client, [asset("Achernar"), asset("Achernar", 2), asset("Achird")]
    )
    results = await asyncio.gather(
        packs.retry_failed_sample_pack(pack_id), packs.retry_failed_sample_pack(pack_id)
    )
    assert sum(summary["enqueued_jobs"] for _, summary in results) == 2
    assert await redis.zcard("arq:queue") == 2
    synth, store = fake_provider(monkeypatch)
    await run_worker(redis)
    assert synth.await_count == store.await_count == 2
    assert len({call.kwargs["voice_id"] for call in synth.await_args_list}) == 2
    assert all(
        a.attempts == 1
        for a in (await client.get_pack(pack_id)).assets
        if a.version > (2 if a.voice_id == "Achernar" else 1)
    )


def fake_provider(monkeypatch):
    wav, duration = pcm_to_wav(b"\x01\x00" * 24000)
    synth = AsyncMock(
        return_value=(wav, duration, {"outbound_requests": 1, "pcm_bytes": 48000})
    )
    store = AsyncMock(return_value=True)
    monkeypatch.setattr(tasks, "synthesize_sample_wav", synth)
    monkeypatch.setattr(
        tasks, "storage_fs", SimpleNamespace(acreate_file_from_bytes=store)
    )
    monkeypatch.setattr(
        tasks,
        "platform_sample_generation_config",
        lambda: SimpleNamespace(
            api_key="test-only",
            project_id="test-only",
            location="global",
            credential_ref=None,
        ),
    )
    return synth, store


async def run_worker(redis):
    worker = Worker(
        functions=[func(tasks.generate_gemini_tts_sample, max_tries=1, timeout=10)],
        redis_pool=redis,
        burst=True,
        handle_signals=False,
        max_jobs=4,
    )
    await worker.async_run()
    assert worker.jobs_failed == 0


async def test_retained_arq_results_do_not_block_new_recovery_versions(
    live, monkeypatch
):
    client, redis = live
    pack_id = await seed(client, [asset("Achird")])
    synth, _ = fake_provider(monkeypatch)
    synth.side_effect = RuntimeError("test provider unavailable")
    _, first = await packs.retry_failed_sample_pack(pack_id)
    await run_worker(redis)
    assert (
        await redis.enqueue_job(
            "generate_gemini_tts_sample",
            first["asset_ids"][0],
            _job_id=first["job_ids"][0],
        )
        is None
    )  # actual ARQ result retention
    synth.side_effect = None
    _, second = await packs.retry_failed_sample_pack(pack_id)
    assert second["enqueued_jobs"] == 1
    assert second["job_ids"][0] != first["job_ids"][0]
    await run_worker(redis)
    pack = await client.get_pack(pack_id)
    assert [a.status for a in pack.assets] == ["failed", "failed", "completed"]
    assert pack.assets[-1].is_current


@pytest.mark.parametrize("failure", ["exception", "retained_result"])
async def test_enqueue_failure_compensates_transaction_and_does_not_hide_success(
    live, monkeypatch, failure
):
    client, redis = live
    pack_id = await seed(client, [asset("Achernar"), asset("Achird")])
    calls = 0

    async def enqueue(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            if failure == "exception":
                raise ConnectionError("disposable queue failure")
            await redis.set("arq:result:" + kwargs["_job_id"], b"retained", ex=60)
        return await redis.enqueue_job(*args, **kwargs)

    monkeypatch.setattr(packs, "enqueue_job", enqueue)
    with pytest.raises(packs.SampleEnqueueError) as error:
        await packs.retry_failed_sample_pack(pack_id)
    summary = error.value.summary
    assert summary["enqueued_jobs"] == 1
    assert summary["enqueue_failures"] + summary["enqueue_conflicts"] == 1
    failed = await client.get_asset(summary["asset_ids"][0])
    assert (failed.status, failed.attempts) == ("failed", 0)
    assert not await client.claim_asset(failed.id)
    good_id = summary["asset_ids"][1]
    assert await client.claim_asset(good_id)
    assert not await client.fail_sample_enqueue(
        good_id
    )  # don't reset an accepted/claimed job
    assert (await client.get_asset(good_id)).status == "running"


async def test_voice_progress_completed_despite_failed_history_and_audio_preserved(
    live, monkeypatch
):
    client, redis = live
    pack_id = await seed(client, [asset(v["voice_id"]) for v in VOICES])
    synth, _ = fake_provider(monkeypatch)
    _, _first = await packs.retry_failed_sample_pack(pack_id, limit=2)
    await run_worker(redis)
    pack = await client.get_pack(pack_id)
    assert pack.status == "partial"
    response = await _pack_response(pack, include_urls=False)
    assert (
        response.total_voices,
        response.playable_voices,
        response.eligible_voices,
    ) == (30, 2, 28)
    originals = [
        (a.id, a.storage_key, a.sha256, a.version) for a in pack.assets if a.is_current
    ]
    _, rest = await packs.retry_failed_sample_pack(pack_id)
    assert rest["skipped_playable_voices"] == 2 and rest["enqueued_jobs"] == 28
    await run_worker(redis)
    pack = await client.get_pack(pack_id)
    assert pack.status == "completed"
    assert originals == [
        (a.id, a.storage_key, a.sha256, a.version)
        for a in pack.assets
        if a.id in {r[0] for r in originals}
    ]
    response = await _pack_response(pack, include_urls=False)
    assert (
        response.playable_voices,
        response.failed_assets,
        response.eligible_voices,
    ) == (30, 30, 0)
    _, noop = await packs.retry_failed_sample_pack(pack_id)
    assert noop["enqueued_jobs"] == noop["selected_voices"] == 0
    assert noop["skipped_playable_voices"] == 30
    assert synth.await_count == 30
    assert all(
        a.generation_metadata["outbound_requests"] == 1
        for a in pack.assets
        if a.is_current
    )


async def test_distributed_provider_limit_across_processes(live):
    _, redis = live
    code = """
import asyncio, os
from redis.asyncio import Redis
from api.services.gemini_tts_sample_concurrency import sample_provider_slot
async def main():
 r = Redis.from_url(os.environ["REDIS_URL"])
 for _ in range(2):
  async with sample_provider_slot(r):
   await r.eval("local n=redis.call('incr',KEYS[1]); local m=tonumber(redis.call('get',KEYS[2]) or '0'); if n>m then redis.call('set',KEYS[2],n) end; return n",2,"test:active","test:max")
   await asyncio.sleep(.2)
   await r.decr("test:active")
 await r.aclose()
asyncio.run(main())
"""
    processes = [
        await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        for _ in range(2)
    ]
    for process in processes:
        _stdout, stderr = await process.communicate()
        assert process.returncode == 0, stderr.decode()
    assert int(await redis.get("test:max")) == 1
    assert int(await redis.get("test:active")) == 0


async def test_authenticated_bodyless_bulk_route_limit_and_playback(live, monkeypatch):
    import httpx
    from fastapi import FastAPI

    from api import constants
    from api.routes import gemini_tts_samples as routes

    client, redis = live
    pack_id = await seed(client, [asset("Achernar"), asset("Achird"), asset("Algenib")])
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/v1")
    monkeypatch.setattr(routes, "db_client", client)
    monkeypatch.setattr(constants, "PLATFORM_ADMIN_API_KEY", "disposable-platform-admin-key-long-enough")
    monkeypatch.setattr(routes, "storage_fs", SimpleNamespace(
        aget_signed_url=AsyncMock(return_value="https://disposable-storage.test/audio")))
    path = f"/api/v1/superuser/gemini-tts/sample-packs/{pack_id}"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        assert (await http.post(path + "/retry?limit=2")).status_code == 401
        http.headers["X-Platform-Admin-Key"] = "disposable-platform-admin-key-long-enough"
        assert (await http.post(path + "/retry?limit=0")).status_code == 422
        response = await http.post(path + "/retry?limit=2")
        assert response.status_code == 202
        summary = response.json()["retry_summary"]
        assert summary["selected_voices"] == summary["enqueued_jobs"] == 2
        assert response.json()["eligible_voices"] == 1
        fake_provider(monkeypatch)
        await run_worker(redis)
        for asset_id in summary["asset_ids"]:
            playback = await http.get(f"{path}/assets/{asset_id}/playback-url")
            assert playback.status_code == 200
            assert playback.json()["url"] == "https://disposable-storage.test/audio"


async def test_cancelled_bulk_request_compensates_unenqueued_reservations(live, monkeypatch):
    client, redis = live
    pack_id = await seed(client, [asset("Achernar"), asset("Achird"), asset("Algenib")])
    calls = 0

    async def enqueue(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise asyncio.CancelledError
        return await redis.enqueue_job(*args, **kwargs)

    monkeypatch.setattr(packs, "enqueue_job", enqueue)
    with pytest.raises(asyncio.CancelledError):
        await packs.retry_failed_sample_pack(pack_id)
    new = [a for a in (await client.get_pack(pack_id)).assets if a.version == 2]
    assert [a.status for a in new] == ["queued", "failed", "failed"]
    assert await redis.zcard("arq:queue") == 1
    assert not await client.claim_asset(new[1].id)
    assert not await client.claim_asset(new[2].id)
