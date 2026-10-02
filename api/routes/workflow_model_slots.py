"""Per-workflow model slot settings (LLM / STT / TTS / embeddings).

AVSIQ-facing contract; see docs/api-reference or docs/contribution notes.
Every route is organization-scoped through the caller's selected organization:
a workflow or credential from another organization is indistinguishable from a
missing one (404).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.db import db_client
from api.db.models import UserModel
from api.schemas.workflow_slot_settings import (
    ApplyTemplateResponse,
    CredentialCreateRequest,
    CredentialListResponse,
    CredentialMetadata,
    GeminiTTSCatalogResponse,
    Slot,
    SlotDraftRequest,
    SlotHistoryResponse,
    SlotPublishRequest,
    SlotRollbackRequest,
    SlotValidationResponse,
    SlotVersion,
    WorkflowSlotsGroupResponse,
    WorkflowSlotsResponse,
)
from api.services.auth.depends import get_user_with_selected_organization
from api.services.auth.platform_admin import require_platform_admin
from api.services.configuration.check_validity import UserConfigurationValidator
from api.services.configuration.effective_readback import (
    build_effective_model_configuration_readback,
)
from api.services.configuration.slot_settings import (
    SlotResolutionError,
    SlotSettingsError,
    WorkflowSlotService,
    credential_summary,
    version_view,
)
from api.services.configuration.options.google_vertex_catalog import (
    GEMINI_TTS_DOCUMENTATION_URL,
    gemini_tts_catalog_revision,
    gemini_tts_voices,
    get_vertex_model,
    vertex_models,
)
from api.services.storage import storage_fs

router = APIRouter(tags=["model-slots"])


async def _build_google_vertex_tts_catalog(
    model: str | None = Query(default=None),
    catalog_revision: str | None = Query(default=None),
    location: str | None = Query(default=None),
    language: str | None = Query(default=None),
    context: str | None = Query(default=None),
    sample_text: str | None = Query(default=None),
):
    """Read-only Gemini-TTS model/voice metadata; never contacts Google."""
    model_ids = (model,) if model else vertex_models("tts")
    if model and get_vertex_model("tts", model) is None:
        raise HTTPException(
            status_code=422, detail=f"unsupported Gemini-TTS model: {model}"
        )
    matching_samples: dict[tuple[str, str], list[dict]] = {}
    if language is not None and context is not None and sample_text is not None:
        for model_id in model_ids:
            for pack, asset in await db_client.list_matching_completed_assets(
                model_id=model_id,
                catalog_revision=catalog_revision,
                location=location,
                language=language,
                style_text=context,
                sample_text=sample_text,
            ):
                if not asset.storage_key:
                    continue
                url = await storage_fs.aget_signed_url(
                    asset.storage_key, expiration=3600, force_inline=True
                )
                if url:
                    matching_samples.setdefault((model_id, asset.voice_id), []).append(
                        {
                            "pack_id": pack.id,
                            "asset_id": asset.id,
                            "catalog_revision": pack.catalog_revision,
                            "location": pack.location,
                            "language": pack.language,
                            "context": pack.style_text,
                            "sample_text": pack.sample_text,
                            "duration_seconds": asset.duration_seconds,
                            "sha256": asset.sha256,
                            "sample_url": url,
                        }
                    )
    return {
        "provider": "google_vertex",
        "source_url": GEMINI_TTS_DOCUMENTATION_URL,
        "sample_policy": (
            "Google embeds demos in the official documentation, but Dograh has "
            "not verified stable direct media URLs. This endpoint only returns "
            "stored pack assets and never makes a Google synthesis request."
        ),
        "models": [
            {
                "model": entry.id,
                "catalog_revision": gemini_tts_catalog_revision(entry.id),
                "lifecycle": entry.lifecycle,
                "locations": list(entry.locations),
                "output_format": entry.output_format,
                "sample_rate_hz": entry.sample_rate_hz,
                "channels": entry.channels,
                "voices": [
                    {
                        "voice_id": voice.id,
                        "gender": voice.gender,
                        "preview": entry.lifecycle == "preview",
                        "demo_url": GEMINI_TTS_DOCUMENTATION_URL,
                        "sample_url": (
                            matching_samples.get((entry.id, voice.id), [])[0].get("sample_url")
                            if matching_samples.get((entry.id, voice.id))
                            else None
                        ),
                        "sample_reuse": "official_demo_page_only",
                        "samples": matching_samples.get((entry.id, voice.id), []),
                    }
                    for voice in gemini_tts_voices(entry.id)
                ],
            }
            for entry in (get_vertex_model("tts", model_id) for model_id in model_ids)
            if entry is not None
        ],
    }


@router.get(
    "/model-slots/google-vertex/tts/catalog",
    response_model=GeminiTTSCatalogResponse,
)
async def get_google_vertex_tts_catalog(
    model: str | None = Query(default=None),
    catalog_revision: str | None = Query(default=None),
    location: str | None = Query(default=None),
    language: str | None = Query(default=None),
    context: str | None = Query(default=None),
    sample_text: str | None = Query(default=None),
    user: UserModel = Depends(get_user_with_selected_organization),
):
    """Tenant-facing catalog; organization authentication is preserved."""
    del user
    return await _build_google_vertex_tts_catalog(
        model=model,
        catalog_revision=catalog_revision,
        location=location,
        language=language,
        context=context,
        sample_text=sample_text,
    )


@router.get(
    "/superuser/gemini-tts/catalog",
    response_model=GeminiTTSCatalogResponse,
    dependencies=[Depends(require_platform_admin)],
)
async def get_platform_gemini_tts_catalog(
    model: str | None = Query(default=None),
    catalog_revision: str | None = Query(default=None),
    location: str | None = Query(default=None),
    language: str | None = Query(default=None),
    context: str | None = Query(default=None),
    sample_text: str | None = Query(default=None),
):
    """Platform-admin catalog used by the Gemini-TTS sample-library UI."""
    return await _build_google_vertex_tts_catalog(
        model=model,
        catalog_revision=catalog_revision,
        location=location,
        language=language,
        context=context,
        sample_text=sample_text,
    )

MAX_GROUP_SIZE = 50


def _service() -> WorkflowSlotService:
    return WorkflowSlotService(db_client)


def _http(exc: SlotSettingsError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail())


async def _describe_effective(user: UserModel, workflow) -> dict:
    released = workflow.released_definition
    try:
        readback = await build_effective_model_configuration_readback(
            organization_id=user.selected_organization_id,
            workflow_configurations=(
                released.workflow_configurations if released else None
            ),
            workflow_id=workflow.id,
        )
    except SlotResolutionError as exc:
        # The live slot cannot be resolved; say so instead of guessing.
        return {"error": str(exc)}
    return {slot: readback.get(slot) for slot in ("llm", "stt", "tts", "embeddings")}


async def _read(user: UserModel, workflow_id: int) -> dict:
    async def describer(workflow):
        return await _describe_effective(user, workflow)

    return await _service().read_slots(
        organization_id=user.selected_organization_id,
        workflow_id=workflow_id,
        effective_describer=describer,
    )


@router.get("/workflow/{workflow_id}/model-slots", response_model=WorkflowSlotsResponse)
async def get_workflow_model_slots(
    workflow_id: int, user: UserModel = Depends(get_user_with_selected_organization)
):
    """Live, draft and history-pointer state of all four slots, without secrets."""
    try:
        return await _read(user, workflow_id)
    except SlotSettingsError as exc:
        raise _http(exc) from None


@router.post(
    "/workflow/{workflow_id}/model-slots/apply-template",
    response_model=ApplyTemplateResponse,
)
async def apply_slot_template(
    workflow_id: int, user: UserModel = Depends(get_user_with_selected_organization)
):
    """Retry the organization-default copy for a workflow left `pending`."""
    try:
        seeded = await _service().retry_template(
            organization_id=user.selected_organization_id,
            workflow_id=workflow_id,
            created_by=str(user.provider_id),
        )
    except SlotSettingsError as exc:
        raise _http(exc) from None
    return {"workflow_id": workflow_id, "seeded_slots": seeded}


@router.get("/model-slots", response_model=WorkflowSlotsGroupResponse)
async def get_model_slots_for_workflows(
    workflow_ids: str = Query(
        description="Comma-separated workflow ids, for example the workflows of one AVSIQ agent."
    ),
    user: UserModel = Depends(get_user_with_selected_organization),
):
    """Readback for a group of workflows. All-or-nothing: an unknown id is a 404."""
    try:
        ids = list(dict.fromkeys(int(p) for p in workflow_ids.split(",") if p.strip()))
    except ValueError:
        raise HTTPException(status_code=422, detail="workflow_ids must be integers")
    if not ids or len(ids) > MAX_GROUP_SIZE:
        raise HTTPException(
            status_code=422, detail=f"provide 1-{MAX_GROUP_SIZE} workflow ids"
        )
    try:
        return {"workflows": [await _read(user, wid) for wid in ids]}
    except SlotSettingsError as exc:
        raise _http(exc) from None


@router.put(
    "/workflow/{workflow_id}/model-slots/{slot}/draft", response_model=SlotVersion
)
async def save_slot_draft(
    workflow_id: int,
    slot: Slot,
    request: SlotDraftRequest,
    user: UserModel = Depends(get_user_with_selected_organization),
):
    """Save a draft for one slot. Other slots are untouched. Not live until published."""
    try:
        row = await _service().save_draft(
            organization_id=user.selected_organization_id,
            workflow_id=workflow_id,
            slot=slot,
            config=request.config,
            credential=(
                request.credential.model_dump() if request.credential else None
            ),
            expected_revision=request.expected_revision,
            change_note=request.change_note,
            created_by=str(user.provider_id),
        )
    except SlotSettingsError as exc:
        raise _http(exc) from None
    return version_view(row)


@router.post(
    "/workflow/{workflow_id}/model-slots/{slot}/draft/{version}/validate",
    response_model=SlotValidationResponse,
)
async def validate_slot_draft(
    workflow_id: int,
    slot: Slot,
    version: int,
    user: UserModel = Depends(get_user_with_selected_organization),
):
    """Validate a draft (auth rules, catalogue, embedding compatibility, provider check)."""
    try:
        status, errors = await _service().validate_draft(
            organization_id=user.selected_organization_id,
            workflow_id=workflow_id,
            slot=slot,
            version=version,
            validator=UserConfigurationValidator(),
            created_by=str(user.provider_id),
        )
    except SlotSettingsError as exc:
        raise _http(exc) from None
    return {"slot": slot, "version": version, "status": status, "errors": errors}


@router.post(
    "/workflow/{workflow_id}/model-slots/{slot}/publish", response_model=SlotVersion
)
async def publish_slot(
    workflow_id: int,
    slot: Slot,
    request: SlotPublishRequest,
    user: UserModel = Depends(get_user_with_selected_organization),
):
    """Make a validated draft live for this workflow's new calls."""
    try:
        row = await _service().publish(
            organization_id=user.selected_organization_id,
            workflow_id=workflow_id,
            slot=slot,
            version=request.version,
            expected_revision=request.expected_revision,
        )
    except SlotSettingsError as exc:
        raise _http(exc) from None
    return version_view(row)


@router.post(
    "/workflow/{workflow_id}/model-slots/{slot}/rollback", response_model=SlotVersion
)
async def rollback_slot(
    workflow_id: int,
    slot: Slot,
    request: SlotRollbackRequest,
    user: UserModel = Depends(get_user_with_selected_organization),
):
    """Re-publish an earlier published version (with its pinned credential version)."""
    try:
        row = await _service().rollback(
            organization_id=user.selected_organization_id,
            workflow_id=workflow_id,
            slot=slot,
            to_version=request.to_version,
            expected_revision=request.expected_revision,
            change_note=request.change_note,
            created_by=str(user.provider_id),
        )
    except SlotSettingsError as exc:
        raise _http(exc) from None
    return version_view(row)


@router.get(
    "/workflow/{workflow_id}/model-slots/{slot}/history",
    response_model=SlotHistoryResponse,
)
async def get_slot_history(
    workflow_id: int,
    slot: Slot,
    limit: int = Query(50, ge=1, le=200),
    user: UserModel = Depends(get_user_with_selected_organization),
):
    try:
        return await _service().history(
            organization_id=user.selected_organization_id,
            workflow_id=workflow_id,
            slot=slot,
            limit=limit,
        )
    except SlotSettingsError as exc:
        raise _http(exc) from None


@router.post("/model-credentials", response_model=CredentialMetadata, status_code=201)
async def create_model_credential(
    request: CredentialCreateRequest,
    user: UserModel = Depends(get_user_with_selected_organization),
):
    """Store a secret (write-only). Returns metadata only; there is no read-back of the value.

    Rotation: post again with the same ``credential_ref``; a new version is created and
    existing slots keep using the version they pinned until a new slot version is published.
    """
    try:
        row = await _service().create_credential(
            organization_id=user.selected_organization_id,
            kind=request.kind,
            secret=request.secret,
            credential_ref=request.credential_ref,
            label=request.label,
            source_ref=request.source_ref,
            created_by=str(user.provider_id),
        )
    except SlotSettingsError as exc:
        raise _http(exc) from None
    return credential_summary(row)


@router.get("/model-credentials", response_model=CredentialListResponse)
async def list_model_credentials(
    user: UserModel = Depends(get_user_with_selected_organization),
):
    rows = await db_client.list_provider_credentials(user.selected_organization_id)
    return {"credentials": [credential_summary(r) for r in rows]}


@router.delete(
    "/model-credentials/{credential_ref}/versions/{version}",
    response_model=CredentialMetadata,
)
async def revoke_model_credential_version(
    credential_ref: str,
    version: int,
    user: UserModel = Depends(get_user_with_selected_organization),
):
    """Revoke a version. Refused while a published slot still pins it."""
    try:
        row = await _service().revoke_credential(
            organization_id=user.selected_organization_id,
            credential_ref=credential_ref,
            version=version,
        )
    except SlotSettingsError as exc:
        raise _http(exc) from None
    return credential_summary(row)
