"""ARQ worker for one durable Gemini-TTS sample asset."""

import asyncio
import hashlib
from datetime import UTC, datetime

from loguru import logger

from api.db import db_client
from api.services.configuration.options.google_vertex_catalog import (
    gemini_tts_audio_format,
)
from api.services.gemini_tts_sample_concurrency import sample_provider_slot
from api.services.gemini_tts_sample_library import (
    SampleGenerationProviderError,
    platform_sample_generation_config,
    safe_provider_diagnostic,
    synthesize_sample_wav,
)
from api.services.storage import storage_fs

SAMPLE_PROVIDER_TIMEOUT_SECONDS = 120


async def generate_gemini_tts_sample(ctx, asset_id: int) -> None:
    config = None
    asset = await db_client.get_gemini_tts_sample_asset(asset_id)
    if asset is None or asset.status == "completed":
        return
    operation_id = (asset.generation_metadata or {}).get("operation_id")
    if not await db_client.claim_gemini_tts_sample_asset(asset_id):
        logger.info(
            "Gemini sample claim_refused operation_id={} asset_id={}",
            operation_id,
            asset_id,
        )
        return
    logger.info(
        "Gemini sample job_started operation_id={} asset_id={}", operation_id, asset_id
    )
    pack = await db_client.get_gemini_tts_sample_pack(asset.pack_id)
    if pack is None:
        return
    try:
        config = platform_sample_generation_config()
        audio_format = gemini_tts_audio_format(pack.model_id)
        if audio_format is None:
            raise RuntimeError("sample model has no supported audio framing")
        sample_rate_hz, channels = audio_format
        # Use the worker's Redis connection, shared by all deployed processes.
        # No in-process semaphore: it would multiply the limit by worker count.
        redis = ctx.get("redis")
        if redis is None:
            from api.tasks.arq import get_arq_redis

            redis = await get_arq_redis()
        async with sample_provider_slot(redis):
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
        await db_client.finish_gemini_tts_sample_asset(
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
        )
        logger.info(
            "Gemini sample database_finalized operation_id={} asset_id={} outcome=completed outbound_requests={}",
            operation_id,
            asset_id,
            metadata.get("outbound_requests"),
        )
    except asyncio.CancelledError:
        # ARQ cancellation (including its job timeout) otherwise strands the
        # claimed row in ``running`` forever. Preserve the cancellation after
        # making the durable state retryable.
        await asyncio.shield(
            db_client.finish_gemini_tts_sample_asset(
                asset_id,
                status="failed",
                values={
                    "error_message": "Gemini-TTS generation was cancelled before provider completion."
                },
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
