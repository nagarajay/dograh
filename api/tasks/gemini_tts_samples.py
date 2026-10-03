"""ARQ worker for one durable Gemini-TTS sample asset, plus its recovery sweep."""

import asyncio
import hashlib
import random
from datetime import UTC, datetime

from arq import Retry, func
from arq.jobs import Job, JobStatus
from loguru import logger

from api.db import db_client
from api.services.configuration.options.google_vertex_catalog import (
    gemini_tts_audio_format,
)
from api.services.gemini_tts_sample_concurrency import (
    ProviderSlotBusy,
    sample_provider_slot,
)
from api.services.gemini_tts_sample_jobs import (
    JOB_TIMEOUT_SECONDS,
    MAX_JOB_TRIES,
    QUEUED_GRACE_SECONDS,
    SLOT_RETRY_DELAY_SECONDS,
    sample_job_id,
)
from api.services.gemini_tts_sample_library import (
    SampleGenerationProviderError,
    platform_sample_generation_config,
    safe_provider_diagnostic,
    synthesize_sample_wav,
)
from api.services.storage import storage_fs
from api.tasks.function_names import FunctionNames

SAMPLE_PROVIDER_TIMEOUT_SECONDS = 120


async def generate_gemini_tts_sample(ctx, asset_id: int) -> None:
    """Generate one sample.

    The provider slot is taken *before* the asset is claimed and without
    waiting: when every slot is busy the job reschedules itself with
    ``Retry`` and frees its worker slot, so a long batch cannot occupy the
    worker and delay unrelated jobs. The asset stays ``queued`` meanwhile.
    """
    asset = await db_client.get_gemini_tts_sample_asset(asset_id)
    if asset is None or asset.status != "queued":
        return
    redis = ctx.get("redis")
    if redis is None:
        from api.tasks.arq import get_arq_redis

        redis = await get_arq_redis()
    try:
        async with sample_provider_slot(redis) as slot:
            await _generate_with_slot(asset_id, slot)
    except ProviderSlotBusy:
        raise Retry(
            defer=SLOT_RETRY_DELAY_SECONDS + random.uniform(0, SLOT_RETRY_DELAY_SECONDS)
        ) from None


async def _generate_with_slot(asset_id: int, slot) -> None:
    config = None
    claim_token = await db_client.claim_gemini_tts_sample_asset(asset_id)
    asset = await db_client.get_gemini_tts_sample_asset(asset_id)
    operation_id = ((asset.generation_metadata if asset else None) or {}).get(
        "operation_id"
    )
    if not claim_token:
        logger.info(
            "Gemini sample claim_refused operation_id={} asset_id={}",
            operation_id,
            asset_id,
        )
        return

    async def lease_alive() -> bool:
        return await db_client.renew_asset_lease(asset_id, claim_token)

    slot.watch(lease_alive)
    logger.info(
        "Gemini sample job_started operation_id={} asset_id={}", operation_id, asset_id
    )
    pack = await db_client.get_gemini_tts_sample_pack(asset.pack_id)
    if pack is None:
        await db_client.finish_gemini_tts_sample_asset(
            asset_id,
            status="failed",
            values={"error_message": "sample pack no longer exists"},
            claim_token=claim_token,
        )
        return
    try:
        config = platform_sample_generation_config()
        audio_format = gemini_tts_audio_format(pack.model_id)
        if audio_format is None:
            raise RuntimeError("sample model has no supported audio framing")
        sample_rate_hz, channels = audio_format
        wav, duration, metadata = await asyncio.wait_for(
            synthesize_sample_wav(
                api_key=config.api_key,
                project_id=config.project_id,
                location=pack.location,
                model_id=pack.model_id,
                voice_id=asset.voice_id,
                language=pack.language,
                style_text=pack.style_text,
                sample_text=pack.sample_text,
                context_id=f"sample-pack-{pack.id}-asset-{asset.id}",
                sample_rate_hz=sample_rate_hz,
                channels=channels,
            ),
            timeout=SAMPLE_PROVIDER_TIMEOUT_SECONDS,
        )
        metadata = {**(asset.generation_metadata or {}), **metadata}
        storage_key = (
            f"gemini-tts-samples/{pack.id}/{asset.voice_id}/v{asset.version}.wav"
        )
        if not await storage_fs.acreate_file_from_bytes(storage_key, wav):
            raise RuntimeError("sample storage rejected the WAV asset")
        logger.info(
            "Gemini sample storage_uploaded asset_id={} bytes={}", asset_id, len(wav)
        )
        finalized = await db_client.finish_gemini_tts_sample_asset(
            asset_id,
            status="completed",
            values={
                "storage_key": storage_key,
                "playable_format": "wav",
                "mime_type": "audio/wav",
                "duration_seconds": duration,
                "sha256": hashlib.sha256(wav).hexdigest(),
                "generation_metadata": metadata,
                "generated_at": datetime.now(UTC),
            },
            claim_token=claim_token,
        )
        logger.info(
            "Gemini sample database_finalized operation_id={} asset_id={} "
            "outcome={} outbound_requests={}",
            operation_id,
            asset_id,
            "completed" if finalized else "claim_replaced",
            metadata.get("outbound_requests"),
        )
    except asyncio.CancelledError:
        # ARQ cancellation (including its job timeout) or a lost lease. Make the
        # row retryable when we still own it; a replaced claim changes nothing.
        await asyncio.shield(
            db_client.finish_gemini_tts_sample_asset(
                asset_id,
                status="failed",
                values={
                    "error_message": "Gemini-TTS generation was cancelled before provider completion."
                },
                claim_token=claim_token,
            )
        )
        raise
    except Exception as error:  # noqa: BLE001 - persist terminal state for every job failure
        if isinstance(error, asyncio.TimeoutError):
            diagnostic = {
                "status_code": None,
                "category": "provider_timeout",
                "detail": f"provider stream exceeded {SAMPLE_PROVIDER_TIMEOUT_SECONDS}s",
            }
        else:
            diagnostic = (
                {
                    "status_code": error.status_code,
                    "category": error.category,
                    "detail": error.detail,
                }
                if isinstance(error, SampleGenerationProviderError)
                else safe_provider_diagnostic(
                    error, api_key=getattr(config, "api_key", None)
                )
            )
        model_resource = (
            f"projects/{config.project_id}/locations/{pack.location}/publishers/google/models/{pack.model_id}"
            if config is not None
            else f"publishers/google/models/{pack.model_id}"
        )
        logger.error(
            "Gemini-TTS sample generation failed: "
            f"pack_id={pack.id}, asset_id={asset.id}, model={pack.model_id}, "
            f"model_resource={model_resource}, "
            f"endpoint=https://aiplatform.googleapis.com/v1/{model_resource}:streamGenerateContent, "
            f"location={pack.location}, language={pack.language}, "
            f"style_present={bool(pack.style_text.strip())}, "
            f"sample_present={bool(pack.sample_text.strip())}, "
            f"api_key_present={bool(getattr(config, 'api_key', None))}, "
            f"project_id_present={bool(getattr(config, 'project_id', None))}, "
            f"credential_ref_present={bool(getattr(config, 'credential_ref', None))}, "
            f"status_code={diagnostic['status_code']}, category={diagnostic['category']}, "
            f"detail={diagnostic['detail']}"
        )
        await db_client.finish_gemini_tts_sample_asset(
            asset_id,
            status="failed",
            claim_token=claim_token,
            values={
                "error_message": f"{diagnostic['category']}: {diagnostic['detail']}",
                "generation_metadata": {
                    **(asset.generation_metadata or {}),
                    **getattr(
                        error,
                        "metadata",
                        {
                            "exception_type": type(error).__name__,
                            "provider_timeout_seconds": SAMPLE_PROVIDER_TIMEOUT_SECONDS,
                        },
                    ),
                },
            },
        )
        logger.info(
            "Gemini sample recovery_outcome operation_id={} asset_id={} outcome=failed status_code={} category={}",
            operation_id,
            asset_id,
            diagnostic["status_code"],
            diagnostic["category"],
        )


async def recover_stranded_gemini_tts_samples(ctx) -> dict:
    """Cron sweep for abandoned sample work. Safe to run concurrently.

    * ``running`` with an expired lease: the worker died or stalled. The asset
      is failed (never silently re-run: the provider may have been billed) and
      its token cleared, so the old worker cannot finalize it later.
    * ``queued`` past the grace period with no live queue job: the job was lost
      (Redis flush, failed enqueue). The asset is re-enqueued under a new job
      id; no provider request has happened, so this is safe.
    """
    redis = ctx.get("redis")
    if redis is None:
        from api.tasks.arq import get_arq_redis

        redis = await get_arq_redis()
    expired = await db_client.fail_expired_running_assets()
    requeued: list[int] = []
    for asset_id, epoch in await db_client.list_stale_queued_assets(
        older_than_seconds=QUEUED_GRACE_SECONDS
    ):
        status = await Job(sample_job_id(asset_id, epoch), redis).status()
        if status in (JobStatus.queued, JobStatus.deferred, JobStatus.in_progress):
            continue  # healthy: a worker will pick it up
        new_epoch = await db_client.bump_enqueue_epoch(asset_id, epoch)
        if new_epoch is None:
            continue  # another sweeper won, or the asset moved on
        try:
            await redis.enqueue_job(
                FunctionNames.GENERATE_GEMINI_TTS_SAMPLE,
                asset_id,
                _job_id=sample_job_id(asset_id, new_epoch),
            )
        except Exception:  # noqa: BLE001 - the next sweep sees the missing job again
            logger.exception("Gemini sample re-enqueue failed asset_id={}", asset_id)
            continue
        requeued.append(asset_id)
    if expired or requeued:
        logger.warning(
            "Gemini sample recovery expired_running={} requeued={}", expired, requeued
        )
    return {"expired_running": expired, "requeued": requeued}


GEMINI_SAMPLE_JOB = func(
    generate_gemini_tts_sample,
    max_tries=MAX_JOB_TRIES,
    timeout=JOB_TIMEOUT_SECONDS,
)
