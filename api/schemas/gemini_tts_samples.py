from datetime import datetime

from pydantic import BaseModel, Field


class GeminiTTSSamplePackCreateRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=128)
    voice_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description=(
            "Optional canonical Google voice ID. When supplied, only this voice "
            "is queued; omit it to generate the full catalog batch."
        ),
    )
    catalog_revision: str | None = Field(default=None, max_length=64)
    location: str = Field(default="global", min_length=1, max_length=128)
    language: str = Field(default="en-US", min_length=1, max_length=32)
    style_text: str = Field(min_length=1, max_length=4000)
    sample_text: str = Field(min_length=1, max_length=4000)


class GeminiTTSSampleVoiceGenerationRequest(BaseModel):
    regenerate: bool = Field(
        default=False,
        description=(
            "Set true only after explicitly confirming that a new Google TTS "
            "request will be sent for this one voice and may incur usage."
        ),
    )


class GeminiTTSSampleAssetResponse(BaseModel):
    id: int
    voice_id: str
    gender: str
    version: int = 1
    is_current: bool = False
    status: str
    storage_key: str | None = None
    playable_format: str | None = None
    mime_type: str | None = None
    duration_seconds: float | None = None
    sha256: str | None = None
    attempts: int
    error_message: str | None = None
    generated_at: datetime | None = None
    sample_url: str | None = None


class GeminiTTSSampleRetrySummary(BaseModel):
    operation_id: str
    eligible_voices: int
    skipped_playable_voices: int
    skipped_active_voices: int
    selected_voices: int
    enqueued_jobs: int
    enqueue_conflicts: int
    enqueue_failures: int
    asset_ids: list[int]
    job_ids: list[str]


class GeminiTTSSamplePackResponse(BaseModel):
    id: int
    provider: str
    model_id: str
    catalog_revision: str
    location: str
    language: str
    style_text: str
    sample_text: str
    request_fingerprint: str
    status: str
    created_by: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
    total_assets: int
    completed_assets: int
    failed_assets: int
    total_voices: int = 0
    playable_voices: int = 0
    eligible_voices: int = 0
    active_voices: int = 0
    retry_summary: GeminiTTSSampleRetrySummary | None = None
    assets: list[GeminiTTSSampleAssetResponse]
