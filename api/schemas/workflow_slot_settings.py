"""Request/response contract for per-workflow model slot settings.

Nothing here carries a secret on the way out. Credentials are created through
their own write-only endpoint and referenced by ``credential_ref``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Slot = Literal["llm", "stt", "tts", "embeddings"]
CredentialKind = Literal["api_key", "service_account_json", "aws_iam"]


class CredentialRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    credential_ref: str = Field(min_length=1, max_length=64)
    version: Optional[int] = Field(
        default=None,
        ge=1,
        description="Exact credential version to pin. Omit to pin the latest active one.",
    )


class SlotDraftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(
        ge=0,
        description="Revision from the last readback of this slot. A stale value is rejected with 409.",
    )
    config: dict[str, Any] = Field(
        description="Provider settings without secrets: provider, model, and provider-specific fields."
    )
    credential: Optional[CredentialRef] = Field(
        default=None,
        description="Omit only for providers that need no secret (for example Vertex with ADC).",
    )
    change_note: Optional[str] = Field(default=None, max_length=255)


class SlotPublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1, description="Draft version to publish.")
    expected_revision: int = Field(ge=0)


class SlotRollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    to_version: int = Field(ge=1)
    expected_revision: int = Field(ge=0)
    change_note: Optional[str] = Field(default=None, max_length=255)


class SlotVersion(BaseModel):
    version: int
    state: Literal["draft", "published", "superseded", "discarded"]
    config: dict[str, Any]
    credential: Optional[dict[str, Any]] = None
    validation_status: Literal["unvalidated", "valid", "invalid"]
    validated_at: Optional[datetime] = None
    based_on_version: Optional[int] = None
    origin: str
    change_note: Optional[str] = None
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    published_at: Optional[datetime] = None


class SlotStatus(BaseModel):
    slot: Slot
    revision: int
    source: Literal[
        "workflow_slot",
        "legacy_workflow_override",
        "organization_default_inherited",
        "unconfigured",
    ] = Field(
        description=(
            "Where this workflow's live setting comes from. "
            "`organization_default_inherited` means later organization-default "
            "changes still change this workflow until it is migrated."
        )
    )
    published_version: Optional[int] = None
    draft_version: Optional[int] = None
    published: Optional[SlotVersion] = None
    draft: Optional[SlotVersion] = None
    effective: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "What a run resolves for this slot, without secrets. When the slot "
            "cannot be resolved (revoked credential, missing encryption key, "
            "pending template copy) it carries `error` (sanitized, actionable), "
            "`error_code`, and the `provider`/`model` that failed; `source` still "
            "says where the slot comes from. `null` only when nothing is configured."
        ),
    )


class WorkflowSlotsResponse(BaseModel):
    workflow_id: int
    workflow_uuid: str
    template_status: Optional[Literal["pending"]] = Field(
        default=None,
        description=(
            "`pending`: the organization-default copy has not completed and the "
            "workflow refuses to run. Retry with POST .../model-slots/apply-template."
        ),
    )
    slots: list[SlotStatus]


class WorkflowSlotsGroupResponse(BaseModel):
    workflows: list[WorkflowSlotsResponse]


class SlotValidationResponse(BaseModel):
    slot: Slot
    version: int
    status: Literal["valid", "invalid"]
    errors: list[str] = Field(default_factory=list)


class SlotHistoryResponse(BaseModel):
    workflow_id: int
    slot: Slot
    revision: int
    versions: list[SlotVersion]


class CredentialCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": []})

    credential_ref: Optional[str] = Field(
        default=None,
        description="Existing reference to add a new version to (rotation). Omit to create a new reference.",
        max_length=64,
    )
    kind: CredentialKind
    label: Optional[str] = Field(default=None, max_length=128)
    source_ref: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Non-secret pointer to the canonical copy, for example an AVSIQ Vault path and version.",
    )
    secret: dict[str, Any] = Field(
        description="Write-only. Keys depend on kind: api_key | credentials | aws_access_key, aws_secret_key, aws_session_token."
    )


class CredentialMetadata(BaseModel):
    credential_ref: str
    version: int
    kind: CredentialKind
    label: Optional[str] = None
    source_ref: Optional[str] = None
    key_id: str
    created_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None


class CredentialListResponse(BaseModel):
    credentials: list[CredentialMetadata]


class ApplyTemplateResponse(BaseModel):
    workflow_id: int
    seeded_slots: list[str]


class GeminiTTSSampleCatalogAsset(BaseModel):
    pack_id: int
    asset_id: int
    catalog_revision: str
    location: str
    language: str
    context: str
    sample_text: str
    duration_seconds: Optional[float] = None
    sha256: Optional[str] = None
    sample_url: Optional[str] = None


class GeminiTTSVoiceCatalogEntry(BaseModel):
    voice_id: str
    gender: Literal["Female", "Male"]
    preview: bool
    demo_url: str
    sample_url: Optional[str] = None
    sample_reuse: Literal["official_demo_page_only"]
    samples: list[GeminiTTSSampleCatalogAsset] = Field(default_factory=list)


class GeminiTTSModelCatalogEntry(BaseModel):
    model: str
    catalog_revision: str
    lifecycle: Literal["ga", "preview"]
    locations: list[str]
    output_format: Optional[str] = None
    sample_rate_hz: Optional[int] = None
    channels: Optional[int] = None
    voices: list[GeminiTTSVoiceCatalogEntry]


class GeminiTTSCatalogResponse(BaseModel):
    provider: Literal["google_vertex"]
    source_url: str
    sample_policy: str
    models: list[GeminiTTSModelCatalogEntry]
