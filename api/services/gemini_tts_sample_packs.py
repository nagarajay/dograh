"""Pack lifecycle orchestration; reads never call Google."""

import asyncio
from uuid import uuid4

from loguru import logger

from api.db import db_client
from api.schemas.gemini_tts_samples import (
    GeminiTTSSamplePackCreateRequest,
    GeminiTTSSampleVoiceGenerationRequest,
)
from api.services.gemini_tts_sample_library import (
    platform_sample_generation_config,
    sample_pack_fingerprint,
    validate_pack_request,
)
from api.tasks.arq import enqueue_job
from api.tasks.function_names import FunctionNames


async def create_sample_pack(request: GeminiTTSSamplePackCreateRequest):
    revision, voices = validate_pack_request(
        model_id=request.model_id,
        location=request.location,
        catalog_revision=request.catalog_revision,
        language=request.language,
        style_text=request.style_text,
        sample_text=request.sample_text,
    )
    generation_config = platform_sample_generation_config()
    if generation_config.location != request.location:
        raise ValueError("sample pack location must match GEMINI_TTS_SAMPLE_LOCATION")
    selected_voice = None
    if request.voice_id is not None:
        selected_voice = next(
            (voice for voice in voices if voice.id == request.voice_id), None
        )
        if selected_voice is None:
            raise ValueError(
                f"voice {request.voice_id!r} is not in model {request.model_id!r} "
                f"catalog revision {revision!r}"
            )
    fingerprint = sample_pack_fingerprint(
        provider="google_vertex",
        model_id=request.model_id,
        catalog_revision=revision,
        location=request.location,
        language=request.language,
        style_text=request.style_text,
        sample_text=request.sample_text,
    )
    row, created = await db_client.create_or_get_pack(
        pack={
            "provider": "google_vertex",
            "model_id": request.model_id,
            "catalog_revision": revision,
            "location": request.location,
            "language": request.language,
            "style_text": request.style_text,
            "sample_text": request.sample_text,
            "request_fingerprint": fingerprint,
            "created_by": "platform-admin",
        },
        assets=[
            {
                "voice_id": voice.id,
                "gender": voice.gender,
                "status": "queued",
            }
            for voice in ([selected_voice] if selected_voice else voices)
        ],
    )
    if not request.voice_id:
        await db_client.ensure_gemini_tts_sample_pack_assets(
            row.id,
            voices=[{"voice_id": voice.id, "gender": voice.gender} for voice in voices],
        )
        row = await db_client.get_gemini_tts_sample_pack(row.id)
    for asset in row.assets:
        if created or asset.status == "queued":
            await enqueue_job(
                FunctionNames.GENERATE_GEMINI_TTS_SAMPLE,
                asset.id,
                _job_id=f"gemini-tts-sample-asset-{asset.id}",
            )
    return row, created


class SampleEnqueueError(RuntimeError):
    def __init__(self, summary):
        self.summary = summary
        super().__init__(
            "Some sample jobs could not be queued; inspect retry_summary and poll the pack."
        )


async def _enqueue_sample_asset(asset_id: int):
    """Fresh immutable versions mean fresh IDs, independent of ARQ result retention."""
    try:
        job = await enqueue_job(
            FunctionNames.GENERATE_GEMINI_TTS_SAMPLE,
            asset_id,
            _job_id=f"gemini-tts-sample-asset-{asset_id}",
        )
    except Exception:  # noqa: BLE001 - compensate all queue acceptance failures
        await db_client.fail_sample_enqueue(asset_id)
        return "failure", None
    if job is None:
        await db_client.fail_sample_enqueue(asset_id)
        return "conflict", None
    return "enqueued", job.job_id


async def retry_failed_sample_pack(pack_id: int, *, limit: int | None = None):
    # limit is used by the authenticated operator diagnostic query parameter.
    row = await db_client.get_gemini_tts_sample_pack(pack_id)
    if row is None:
        return None, {}
    _revision, voices = validate_pack_request(
        model_id=row.model_id,
        location=row.location,
        catalog_revision=row.catalog_revision,
        language=row.language,
        style_text=row.style_text,
        sample_text=row.sample_text,
    )
    asset_ids, summary = await db_client.queue_failed_voice_recovery(
        pack_id,
        voices=[{"voice_id": v.id, "gender": v.gender} for v in voices],
        operation_id=uuid4().hex,
        limit=limit,
    )
    logger.info("Gemini bulk retry selected pack_id={} summary={}", pack_id, summary)
    try:
        for index, asset_id in enumerate(asset_ids):
            outcome, job_id = await _enqueue_sample_asset(asset_id)
            summary[
                {"enqueued": "enqueued_jobs", "conflict": "enqueue_conflicts",
                 "failure": "enqueue_failures"}[outcome]
            ] += 1
            if job_id:
                summary["job_ids"].append(job_id)
            logger.info(
                "Gemini bulk retry enqueue operation_id={} asset_id={} outcome={} job_id={}",
                summary["operation_id"], asset_id, outcome, job_id,
            )
    except asyncio.CancelledError:
        # A cancelled request must not strand the not-yet-enqueued reservations.
        async def compensate():
            for pending_id in asset_ids[index:]:
                await db_client.fail_sample_enqueue(pending_id)
        await asyncio.shield(compensate())
        raise
    logger.info("Gemini bulk retry scheduled pack_id={} summary={}", pack_id, summary)
    if summary["enqueue_conflicts"] or summary["enqueue_failures"]:
        raise SampleEnqueueError(summary)
    return await db_client.get_gemini_tts_sample_pack(pack_id), summary


async def generate_sample_voice(
    pack_id: int, voice_id: str, request: GeminiTTSSampleVoiceGenerationRequest
):
    pack = await db_client.get_gemini_tts_sample_pack(pack_id)
    if pack is None:
        return None, None
    _revision, voices = validate_pack_request(
        model_id=pack.model_id,
        location=pack.location,
        catalog_revision=pack.catalog_revision,
        language=pack.language,
        style_text=pack.style_text,
        sample_text=pack.sample_text,
    )
    voice = next((item for item in voices if item.id == voice_id), None)
    if voice is None:
        raise ValueError(
            f"voice {voice_id!r} is not in model {pack.model_id!r} "
            f"catalog revision {pack.catalog_revision!r}"
        )
    asset, state = await db_client.queue_voice_generation(
        pack_id,
        voice_id=voice.id,
        gender=voice.gender,
        regenerate=request.regenerate,
    )
    if state == "queued" or state == "running":
        raise RuntimeError(f"voice {voice_id!r} is already {state}")
    if state == "regeneration_confirmation_required":
        raise PermissionError(
            "regeneration requires explicit confirmation: this sends a new Google "
            "TTS request for one voice and may incur usage"
        )
    if state == "failed_asset_use_retry":
        raise RuntimeError(
            "voice has a failed generation; use the pack retry action, which retries failed assets only"
        )
    if asset is not None:
        outcome, _job_id = await _enqueue_sample_asset(asset.id)
        if outcome != "enqueued":
            raise SampleEnqueueError(
                {"asset_ids": [asset.id], "enqueue_outcome": outcome}
            )
    return await db_client.get_gemini_tts_sample_pack(pack_id), asset
