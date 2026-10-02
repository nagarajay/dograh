"""Platform-admin lifecycle APIs for the reusable Gemini-TTS sample library."""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.db import db_client
from api.schemas.gemini_tts_samples import (
    GeminiTTSSampleAssetResponse,
    GeminiTTSSamplePackCreateRequest,
    GeminiTTSSamplePackResponse,
    GeminiTTSSampleVoiceGenerationRequest,
)
from api.services.auth.platform_admin import require_platform_admin
from api.services.configuration.options.google_vertex_catalog import gemini_tts_voices
from api.services.gemini_tts_sample_packs import (
    SampleEnqueueError,
    create_sample_pack,
    generate_sample_voice,
    retry_failed_sample_pack,
)
from api.services.storage import storage_fs

router = APIRouter(
    prefix="/superuser/gemini-tts/sample-packs",
    tags=["gemini-tts-samples"],
    dependencies=[Depends(require_platform_admin)],
)


async def _asset_response(asset, *, include_url: bool = True):
    sample_url = None
    if include_url and asset.status == "completed" and asset.storage_key:
        sample_url = await storage_fs.aget_signed_url(
            asset.storage_key, expiration=3600, force_inline=True
        )
    return GeminiTTSSampleAssetResponse(
        id=asset.id,
        voice_id=asset.voice_id,
        gender=asset.gender,
        version=getattr(asset, "version", 1),
        is_current=getattr(asset, "is_current", asset.status == "completed"),
        status=asset.status,
        storage_key=asset.storage_key,
        playable_format=asset.playable_format,
        mime_type=asset.mime_type,
        duration_seconds=asset.duration_seconds,
        sha256=asset.sha256,
        attempts=asset.attempts,
        error_message=asset.error_message,
        generated_at=asset.generated_at,
        sample_url=sample_url,
    )


async def _pack_response(pack, *, include_urls: bool = True, retry_summary=None):
    assets = [
        await _asset_response(asset, include_url=include_urls)
        for asset in pack.assets
    ]
    playable = {a.voice_id for a in assets if a.status == "completed" and a.is_current}
    active = {a.voice_id for a in assets if a.status in {"queued", "running"}}
    failed = {a.voice_id for a in assets if a.status == "failed"}
    return GeminiTTSSamplePackResponse(
        id=pack.id,
        provider=pack.provider,
        model_id=pack.model_id,
        catalog_revision=pack.catalog_revision,
        location=pack.location,
        language=pack.language,
        style_text=pack.style_text,
        sample_text=pack.sample_text,
        request_fingerprint=pack.request_fingerprint,
        status=pack.status,
        created_by=pack.created_by,
        created_at=pack.created_at,
        completed_at=pack.completed_at,
        total_assets=len(assets),
        completed_assets=sum(asset.status == "completed" for asset in assets),
        failed_assets=sum(asset.status == "failed" for asset in assets),
        total_voices=len(gemini_tts_voices(pack.model_id)),
        playable_voices=len(playable),
        eligible_voices=len(failed - playable - active),
        active_voices=len(active),
        retry_summary=retry_summary,
        assets=assets,
    )


@router.post("", response_model=GeminiTTSSamplePackResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_gemini_tts_sample_pack(request: GeminiTTSSamplePackCreateRequest):
    try:
        pack, _created = await create_sample_pack(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    return await _pack_response(pack)


@router.get("", response_model=list[GeminiTTSSamplePackResponse])
async def list_gemini_tts_sample_packs(
    completed_only: bool = Query(default=False),
):
    packs = await db_client.list_packs(completed_only=completed_only)
    return [await _pack_response(pack) for pack in packs]


@router.get("/{pack_id}", response_model=GeminiTTSSamplePackResponse)
async def get_gemini_tts_sample_pack(pack_id: int):
    pack = await db_client.get_gemini_tts_sample_pack(pack_id)
    if pack is None:
        raise HTTPException(status_code=404, detail="sample pack not found")
    return await _pack_response(pack)


@router.post("/{pack_id}/retry", response_model=GeminiTTSSamplePackResponse, status_code=202)
async def retry_gemini_tts_sample_pack(
    pack_id: int,
    limit: int | None = Query(default=None, ge=1, le=30,
        description="Platform-admin diagnostic cap; omit for one recovery pass over all eligible voices."),
):
    try:
        pack, summary = await retry_failed_sample_pack(pack_id, limit=limit)
    except SampleEnqueueError as exc:
        raise HTTPException(status_code=503, detail={"message": str(exc), "retry_summary": exc.summary}) from None
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    if pack is None:
        raise HTTPException(status_code=404, detail="sample pack not found")
    return await _pack_response(pack, retry_summary=summary)


@router.post(
    "/{pack_id}/voices/{voice_id}/generate",
    response_model=GeminiTTSSamplePackResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_gemini_tts_sample_voice(
    pack_id: int,
    voice_id: str,
    request: GeminiTTSSampleVoiceGenerationRequest,
):
    try:
        pack, _asset = await generate_sample_voice(pack_id, voice_id, request)
    except SampleEnqueueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if pack is None:
        raise HTTPException(status_code=404, detail="sample pack not found")
    return await _pack_response(pack)


@router.get("/{pack_id}/assets/{asset_id}/playback-url")
async def get_gemini_tts_sample_playback_url(pack_id: int, asset_id: int):
    asset = await db_client.get_gemini_tts_sample_asset(asset_id)
    if asset is None or asset.pack_id != pack_id:
        raise HTTPException(status_code=404, detail="sample asset not found")
    if asset.status != "completed" or not asset.storage_key:
        raise HTTPException(status_code=409, detail="sample asset is not completed")
    url = await storage_fs.aget_signed_url(
        asset.storage_key, expiration=3600, force_inline=True
    )
    if not url:
        raise HTTPException(status_code=503, detail="sample storage unavailable")
    return {"asset_id": asset.id, "mime_type": asset.mime_type, "url": url}
