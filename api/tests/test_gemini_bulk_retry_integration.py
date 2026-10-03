"""Real Postgres constraints and ARQ Redis semantics; isolated disposable services only.

Run with GEMINI_BULK_INTEGRATION=1 and a disposable Postgres and Redis reachable
as gemini-bulk-postgres / gemini-bulk-redis (override with
GEMINI_BULK_DATABASE_URL / GEMINI_BULK_REDIS_URL). These are separate from the
suite's own DATABASE_URL/REDIS_URL, so a single full run can include them;
scripts/test_api_in_docker.sh sets everything up.
Provider and storage calls are substituted; live deployment verifies those boundaries.
"""

import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from arq import create_pool
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
from api.services.gemini_tts_sample_jobs import QUEUED_GRACE_SECONDS, sample_job_id
from api.services.gemini_tts_sample_library import pcm_to_wav
from api.tasks import gemini_tts_samples as tasks

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("GEMINI_BULK_INTEGRATION") != "1",
        reason="disposable integration services required",
    ),
]
BULK_DATABASE_URL = os.getenv(
    "GEMINI_BULK_DATABASE_URL", "postgresql+asyncpg://t:t@gemini-bulk-postgres:5432/t"
)
BULK_REDIS_URL = os.getenv("GEMINI_BULK_REDIS_URL", "redis://gemini-bulk-redis:6379")
MODEL = "gemini-3.1-flash-tts-preview"
VOICES = [{"voice_id": v.id, "gender": v.gender} for v in gemini_tts_voices(MODEL)]


@pytest.fixture
async def live(monkeypatch):
    # Never run destructive fixture setup against the development Supabase DB.
    assert "@gemini-bulk-postgres:" in BULK_DATABASE_URL
    assert BULK_REDIS_URL == "redis://gemini-bulk-redis:6379"
    engine = create_async_engine(BULK_DATABASE_URL, poolclass=NullPool)
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
    monkeypatch.setattr(tasks, "SLOT_RETRY_DELAY_SECONDS", 0.05)
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
        functions=[tasks.GEMINI_SAMPLE_JOB],
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
from api.services.gemini_tts_sample_concurrency import ProviderSlotBusy, sample_provider_slot
async def main():
 r = Redis.from_url(os.environ["GEMINI_BULK_REDIS_URL"])
 done = 0
 while done < 2:
  try:
   async with sample_provider_slot(r):
    await r.eval("local n=redis.call('incr',KEYS[1]); local m=tonumber(redis.call('get',KEYS[2]) or '0'); if n>m then redis.call('set',KEYS[2],n) end; return n",2,"test:active","test:max")
    await asyncio.sleep(.2)
    await r.decr("test:active")
   done += 1
  except ProviderSlotBusy:
   await asyncio.sleep(.05)
 await r.aclose()
asyncio.run(main())
"""
    processes = [
        await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            env={**os.environ, "GEMINI_BULK_REDIS_URL": BULK_REDIS_URL},
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
    monkeypatch.setattr(
        constants, "PLATFORM_ADMIN_API_KEY", "disposable-platform-admin-key-long-enough"
    )
    monkeypatch.setattr(
        routes,
        "storage_fs",
        SimpleNamespace(
            aget_signed_url=AsyncMock(
                return_value="https://disposable-storage.test/audio"
            )
        ),
    )
    path = f"/api/v1/superuser/gemini-tts/sample-packs/{pack_id}"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http:
        assert (await http.post(path + "/retry?limit=2")).status_code == 401
        http.headers["X-Platform-Admin-Key"] = (
            "disposable-platform-admin-key-long-enough"
        )
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


async def test_cancelled_bulk_request_compensates_unenqueued_reservations(
    live, monkeypatch
):
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


# ----------------------------------------------- lease, recovery, scheduling
async def age(client, asset_id, *, lease=None, queued=None):
    """Move an asset's lease/queue timestamps into the past, as time would."""
    from datetime import UTC, datetime, timedelta

    values = {}
    if lease is not None:
        values["lease_expires_at"] = datetime.now(UTC) - timedelta(seconds=lease)
    if queued is not None:
        values["queued_at"] = datetime.now(UTC) - timedelta(seconds=queued)
    async with client.async_session() as session:
        await session.execute(
            update(Asset).where(Asset.id == asset_id).values(**values)
        )
        await session.commit()


async def sweep(redis):
    return await tasks.recover_stranded_gemini_tts_samples({"redis": redis})


async def test_exactly_one_of_many_concurrent_claims_wins(live):
    client, _ = live
    pack_id = await seed(client, [asset("Achernar", status="queued")])
    asset_id = (await client.get_pack(pack_id)).assets[0].id
    tokens = await asyncio.gather(*(client.claim_asset(asset_id) for _ in range(8)))
    assert len([t for t in tokens if t]) == 1
    row = await client.get_asset(asset_id)
    assert (row.status, row.attempts, row.claim_token) == (
        "running",
        1,
        next(t for t in tokens if t),
    )
    assert row.lease_expires_at is not None


async def test_dead_worker_is_recovered_and_cannot_finalize_later(live):
    client, redis = live
    pack_id = await seed(
        client,
        [
            asset(
                "Achernar",
                1,
                "completed",
                True,
                storage_key="keep.wav",
                sha256="keep-hash",
            ),
            asset("Achernar", 2, "queued"),
        ],
    )
    pack = await client.get_pack(pack_id)
    regen = pack.assets[1]
    old_worker = await client.claim_asset(regen.id)
    await age(client, regen.id, lease=5)  # the worker died; nobody renewed
    result = await sweep(redis)
    assert result["expired_running"] == [regen.id]
    row = await client.get_asset(regen.id)
    assert (row.status, row.claim_token, row.lease_expires_at) == ("failed", None, None)
    assert row.error_message.startswith("worker_lost")
    # The dead (or merely stalled) worker wakes up and tries to finish: refused.
    assert not await client.finish_gemini_tts_sample_asset(
        regen.id,
        status="completed",
        values={"storage_key": "late.wav", "sha256": "late"},
        claim_token=old_worker,
    )
    row = await client.get_asset(regen.id)
    assert (row.status, row.storage_key, row.is_current) == ("failed", None, False)
    # Current playable audio and history are untouched; the pack still plays.
    pack = await client.get_pack(pack_id)
    current = [a for a in pack.assets if a.is_current]
    assert [(a.id, a.storage_key, a.sha256) for a in current] == [
        (pack.assets[0].id, "keep.wav", "keep-hash")
    ]
    assert pack.status == "partial"  # one voice playable of thirty
    # Recovery is idempotent.
    assert (await sweep(redis))["expired_running"] == []
    # An explicit retry makes a new version instead of reusing the failed one.
    await client.queue_voice_generation(
        pack_id, voice_id="Achernar", gender="Female", regenerate=True
    )
    versions = sorted(a.version for a in (await client.get_pack(pack_id)).assets)
    assert versions == [1, 2, 3]


async def test_healthy_lease_is_never_reclaimed_and_renewal_extends_it(live):
    client, redis = live
    pack_id = await seed(client, [asset("Achernar", status="queued")])
    asset_id = (await client.get_pack(pack_id)).assets[0].id
    token = await client.claim_asset(asset_id, lease_seconds=60)
    assert (await sweep(redis))["expired_running"] == []
    assert (await client.get_asset(asset_id)).status == "running"
    await age(client, asset_id, lease=1)
    assert await client.renew_asset_lease(
        asset_id, token, lease_seconds=60
    )  # heartbeat arrived
    assert (await sweep(redis))["expired_running"] == []
    assert not await client.renew_asset_lease(asset_id, "someone-else")
    assert await client.finish_gemini_tts_sample_asset(
        asset_id, status="failed", values={"error_message": "x"}, claim_token=token
    )


async def test_concurrent_sweeps_recover_each_asset_once(live):
    client, redis = live
    pack_id = await seed(
        client,
        [
            asset("Achernar", status="running"),
            asset("Achird", status="queued"),
            asset("Algenib"),
        ],
    )
    assets = (await client.get_pack(pack_id)).assets
    await age(client, assets[0].id, lease=5)
    await age(client, assets[1].id, queued=QUEUED_GRACE_SECONDS + 5)
    results = await asyncio.gather(sweep(redis), sweep(redis), sweep(redis))
    assert sum(len(r["expired_running"]) for r in results) == 1
    assert sum(len(r["requeued"]) for r in results) == 1
    assert await redis.zcard("arq:queue") == 1
    queued = await client.get_asset(assets[1].id)
    assert queued.enqueue_epoch == 1
    # Run again: the re-enqueued job is alive, so nothing more happens.
    again = await sweep(redis)
    assert again == {"expired_running": [], "requeued": []}


async def test_lost_queue_job_is_re_enqueued_and_then_completes(live, monkeypatch):
    client, redis = live
    pack_id = await seed(client, [asset("Achernar", status="queued")])
    asset_id = (await client.get_pack(pack_id)).assets[0].id
    # Within the grace period (enqueue may still be in flight): left alone.
    assert (await sweep(redis))["requeued"] == []
    await age(client, asset_id, queued=QUEUED_GRACE_SECONDS + 5)
    assert await redis.zcard("arq:queue") == 0  # the job was lost
    assert (await sweep(redis))["requeued"] == [asset_id]
    assert await redis.exists("arq:job:" + sample_job_id(asset_id, 1))
    synth, _ = fake_provider(monkeypatch)
    await run_worker(redis)
    row = await client.get_asset(asset_id)
    assert (row.status, row.is_current, row.attempts) == ("completed", True, 1)
    assert synth.await_count == 1


async def test_finished_job_result_does_not_hide_a_still_queued_asset(live):
    client, redis = live
    pack_id = await seed(client, [asset("Achernar", status="queued")])
    asset_id = (await client.get_pack(pack_id)).assets[0].id
    await redis.set("arq:result:" + sample_job_id(asset_id), b"retained", ex=60)
    await age(client, asset_id, queued=QUEUED_GRACE_SECONDS + 5)
    assert (await sweep(redis))["requeued"] == [asset_id]


async def test_busy_provider_defers_without_occupying_workers(live, monkeypatch):
    """Jobs that find the provider busy must free their ARQ slot and finish later."""
    import time

    from arq import func
    from arq.worker import Worker

    client, redis = live
    monkeypatch.setattr(tasks, "SLOT_RETRY_DELAY_SECONDS", 0.3)
    pack_id = await seed(
        client, [asset(v, status="queued") for v in ("Achernar", "Achird", "Algenib")]
    )
    ids = [a.id for a in (await client.get_pack(pack_id)).assets]
    synth, _ = fake_provider(monkeypatch)
    await redis.set("gemini-tts-samples:provider-slot:0", "other-process", ex=3)
    for asset_id in ids:
        await redis.enqueue_job(
            "generate_gemini_tts_sample", asset_id, _job_id=sample_job_id(asset_id)
        )
    quick_done = {}

    async def unrelated(ctx):
        quick_done["at"] = time.monotonic()

    await redis.enqueue_job("unrelated")
    started = time.monotonic()
    # max_jobs equals the number of waiting sample jobs: with the old busy-wait
    # they would hold every slot and ``unrelated`` would only run after ~3 s.
    worker = Worker(
        functions=[tasks.GEMINI_SAMPLE_JOB, func(unrelated, name="unrelated")],
        redis_pool=redis,
        burst=True,
        handle_signals=False,
        max_jobs=3,
        poll_delay=0.05,
    )
    await worker.async_run()
    assert quick_done["at"] - started < 1.5
    assert worker.jobs_failed == 0
    pack = await client.get_pack(pack_id)
    assert [a.status for a in pack.assets] == ["completed"] * 3
    assert synth.await_count == 3
    assert {a.attempts for a in pack.assets} == {1}


async def test_lost_lease_cancels_provider_work(monkeypatch):
    from api.services import gemini_tts_sample_concurrency as concurrency

    class FakeRedis:
        async def set(self, *a, **k):
            return True

        async def eval(self, *a, **k):
            return 1

    monkeypatch.setattr(concurrency, "LEASE_SECONDS", 0.3)

    async def lost():
        return False

    cancelled = False
    try:
        async with concurrency.sample_provider_slot(FakeRedis()) as slot:
            slot.watch(lost)
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        cancelled = True
    assert cancelled


# ------------------------------------------- individual recovery of one voice
def _request(regenerate):
    from api.schemas.gemini_tts_samples import GeminiTTSSampleVoiceGenerationRequest

    return GeminiTTSSampleVoiceGenerationRequest(regenerate=regenerate)


async def _snapshot(client, pack_id):
    return {
        a.id: (a.voice_id, a.version, a.status, a.is_current, a.storage_key, a.sha256)
        for a in (await client.get_pack(pack_id)).assets
    }


async def test_retrying_one_failed_voice_leaves_every_other_voice_untouched(live):
    client, redis = live
    pack_id = await seed(
        client,
        [
            asset("Achernar", 1, "completed", True, storage_key="a.wav", sha256="a"),
            asset("Achird"),  # failed
            asset("Algenib"),  # failed
            asset("Algieba", 1, "completed", True, storage_key="g.wav", sha256="g"),
        ],
    )
    before = await _snapshot(client, pack_id)
    pack, new = await packs.generate_sample_voice(pack_id, "Achird", _request(True))
    after = await _snapshot(client, pack_id)
    # Exactly one new queued version, for the retried voice only.
    added = {k: v for k, v in after.items() if k not in before}
    assert list(added.values()) == [("Achird", 2, "queued", False, None, None)]
    assert new.id in added
    # Every pre-existing row, including the other failed voice and the playable
    # samples, is byte-for-byte unchanged.
    assert {k: after[k] for k in before} == before
    assert await redis.zcard("arq:queue") == 1


async def test_repeated_or_multi_tab_retry_creates_exactly_one_generation(live):
    client, redis = live
    pack_id = await seed(
        client,
        [asset("Achird"), asset("Achernar", 1, "completed", True, storage_key="a.wav")],
    )
    # Six tabs press Retry at the same moment.
    results = await asyncio.gather(
        *(
            packs.generate_sample_voice(pack_id, "Achird", _request(True))
            for _ in range(6)
        ),
        return_exceptions=True,
    )
    accepted = [r for r in results if not isinstance(r, BaseException)]
    refused = [r for r in results if isinstance(r, RuntimeError)]
    assert len(accepted) == 1 and len(refused) == 5
    assert all("already" in str(r) for r in refused)
    versions = [
        a.version
        for a in (await client.get_pack(pack_id)).assets
        if a.voice_id == "Achird"
    ]
    assert sorted(versions) == [1, 2]
    assert await redis.zcard("arq:queue") == 1
    # A later click while it is generating is also refused, not queued again.
    with pytest.raises(RuntimeError, match="already"):
        await packs.generate_sample_voice(pack_id, "Achird", _request(True))


async def test_a_failed_retry_can_be_retried_again_and_success_replaces_nothing(
    live, monkeypatch
):
    client, redis = live
    pack_id = await seed(
        client,
        [
            asset("Achird"),
            asset("Achernar", 1, "completed", True, storage_key="a.wav", sha256="a"),
        ],
    )
    synth, _ = fake_provider(monkeypatch)
    synth.side_effect = RuntimeError("test provider unavailable")
    await packs.generate_sample_voice(pack_id, "Achird", _request(True))
    await run_worker(redis)
    assert [
        a.status
        for a in (await client.get_pack(pack_id)).assets
        if a.voice_id == "Achird"
    ] == ["failed", "failed"]
    synth.side_effect = None
    await packs.generate_sample_voice(pack_id, "Achird", _request(True))
    await run_worker(redis)
    pack = await client.get_pack(pack_id)
    achird = [a for a in pack.assets if a.voice_id == "Achird"]
    assert [a.status for a in achird] == ["failed", "failed", "completed"] and achird[
        -1
    ].is_current
    ach = next(a for a in pack.assets if a.voice_id == "Achernar")
    assert (ach.status, ach.is_current, ach.storage_key, ach.sha256) == (
        "completed",
        True,
        "a.wav",
        "a",
    )
