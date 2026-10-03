"""Offline proof for the durable Gemini-TTS sample library."""

import io
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pipecat.frames.frames import TTSAudioRawFrame

from api import constants
from api.routes import gemini_tts_samples, workflow_model_slots
from api.routes.workflow_model_slots import get_google_vertex_tts_catalog
from api.schemas.gemini_tts_samples import GeminiTTSSamplePackCreateRequest
from api.services import gemini_tts_sample_packs as packs
from api.services.configuration.options.google_vertex_catalog import get_vertex_model
from api.services.gemini_tts_sample_library import (
    SampleGenerationNotConfigured,
    SampleGenerationProviderError,
    pcm_to_wav,
    platform_sample_generation_config,
    safe_provider_diagnostic,
    sample_pack_fingerprint,
    synthesize_sample_wav,
    validate_pack_request,
)
from api.tasks import gemini_tts_samples as sample_tasks


def test_provider_diagnostic_extracts_status_and_redacts_request_data():
    error = RuntimeError(
        "400 INVALID_ARGUMENT api_key=secret contents=private sample text"
    )
    diagnostic = safe_provider_diagnostic(error, api_key="secret")
    assert diagnostic["status_code"] == 400
    assert diagnostic["category"] == "invalid_argument"
    assert "secret" not in diagnostic["detail"]
    assert "private sample text" not in diagnostic["detail"]


def test_provider_error_has_actionable_safe_shape():
    error = SampleGenerationProviderError(
        status_code=400, category="invalid_argument", detail="unsupported language"
    )
    assert str(error) == "HTTP 400 invalid_argument: unsupported language"


def test_sample_provider_timeout_is_bounded():
    assert sample_tasks.SAMPLE_PROVIDER_TIMEOUT_SECONDS < 300


from api.enums import StorageBackend
from api.services import storage as storage_service
from api.services.auth.platform_admin import require_platform_admin


def test_pcm_is_browser_playable_wav_with_exact_duration():
    wav_bytes, duration = pcm_to_wav(b"\x00\x00" * 24000)
    assert duration == 1.0
    with wave.open(io.BytesIO(wav_bytes)) as wav:
        assert wav.getframerate() == 24000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 24000


def test_pack_fingerprint_changes_for_every_generation_input():
    base = dict(
        provider="google_vertex",
        model_id="gemini-3.1-flash-tts-preview",
        catalog_revision="rev-1",
        location="global",
        language="en-US",
        style_text="warm",
        sample_text="hello",
    )
    assert sample_pack_fingerprint(**base) != sample_pack_fingerprint(
        **{**base, "sample_text": "different"}
    )
    assert sample_pack_fingerprint(**base) == sample_pack_fingerprint(**base)


def test_three_one_requires_global_and_current_catalog_revision():
    revision, voices = validate_pack_request(
        model_id="gemini-3.1-flash-tts-preview",
        location="global",
        catalog_revision=None,
        language="en-US",
        style_text="warm",
        sample_text="hello",
    )
    assert revision and len(voices) == 30
    with pytest.raises(ValueError, match="requires one of"):
        validate_pack_request(
            model_id="gemini-3.1-flash-tts-preview",
            location="us",
            catalog_revision=None,
            language="en-US",
            style_text="warm",
            sample_text="hello",
        )


def test_sample_generation_follows_the_api_key_slot_location_contract():
    """The platform key is an API key: only the global endpoint may be requested,
    exactly as for an API-key TTS slot (a regional model location is documented
    but cannot be selected with a key)."""
    from api.services.configuration.options.google_vertex_catalog import (
        gemini_tts_voices,
        vertex_models,
    )

    regional = next(
        m
        for m in vertex_models("tts")
        if get_vertex_model("tts", m).locations
        and get_vertex_model("tts", m).locations != ("global",)
    )
    region = next(
        loc for loc in get_vertex_model("tts", regional).locations if loc != "global"
    )
    with pytest.raises(ValueError, match="cannot choose a location"):
        validate_pack_request(
            model_id=regional,
            location=region,
            catalog_revision=None,
            language="en-US",
            style_text="warm",
            sample_text="hello",
        )
    # The same model on the global endpoint, when the model offers it, is fine.
    if "global" in get_vertex_model("tts", regional).locations:
        revision, voices = validate_pack_request(
            model_id=regional,
            location="global",
            catalog_revision=None,
            language="en-US",
            style_text="warm",
            sample_text="hello",
        )
        assert revision and voices == gemini_tts_voices(regional)


def test_generation_requires_platform_credential_and_never_uses_tenant_key(monkeypatch):
    monkeypatch.setattr(constants, "GEMINI_TTS_SAMPLE_API_KEY", None)
    monkeypatch.setattr(constants, "GEMINI_TTS_SAMPLE_CREDENTIAL_REF", "vault://sample")
    with pytest.raises(SampleGenerationNotConfigured) as exc:
        platform_sample_generation_config()
    assert "tenant" not in str(exc.value).lower()
    assert "vault://sample" in str(exc.value)


@pytest.mark.asyncio
async def test_synthesis_frames_are_converted_without_a_google_call(monkeypatch):
    class FakeService:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._client = SimpleNamespace(
                aio=SimpleNamespace(aclose=AsyncMock()), close=lambda: None
            )

        async def _run_genai_tts(self, text, context_id):
            assert text == "hello" and context_id == "ctx"
            yield TTSAudioRawFrame(b"\x00\x00" * 24000, 24000, 1)

    monkeypatch.setattr(
        "api.services.gemini_tts_sample_library.DograhGeminiVertexApiTTSService",
        FakeService,
    )
    wav_bytes, duration, metadata = await synthesize_sample_wav(
        api_key="platform-only",
        project_id="project",
        location="global",
        model_id="gemini-3.1-flash-tts-preview",
        voice_id="Kore",
        language="en-US",
        style_text="warm",
        sample_text="hello",
        context_id="ctx",
        sample_rate_hz=24000,
        channels=1,
    )
    assert duration == 1.0 and wav_bytes.startswith(b"RIFF")
    assert metadata["native_sample_rate_hz"] == 24000


def test_sample_pack_routes_are_platform_admin_only():
    assert all(
        dependency.dependency == require_platform_admin
        for route in gemini_tts_samples.router.routes
        for dependency in route.dependencies
    )


def test_platform_catalog_requires_platform_admin_and_rejects_tenant_key(monkeypatch):
    monkeypatch.setattr(constants, "PLATFORM_ADMIN_API_KEY", "p" * 32)
    app = FastAPI()
    app.include_router(workflow_model_slots.router)
    client = TestClient(app)

    assert client.get("/superuser/gemini-tts/catalog").status_code == 401
    assert (
        client.get(
            "/superuser/gemini-tts/catalog", headers={"X-API-Key": "tenant-key"}
        ).status_code
        == 403
    )
    response = client.get(
        "/superuser/gemini-tts/catalog",
        headers={"X-Platform-Admin-Key": "p" * 32},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "google_vertex"
    assert payload["models"][0]["voices"]


def test_sample_pack_route_rejects_missing_platform_key(monkeypatch):
    monkeypatch.setattr(constants, "PLATFORM_ADMIN_API_KEY", "x" * 32)
    app = FastAPI()
    app.include_router(gemini_tts_samples.router)
    response = TestClient(app).get("/superuser/gemini-tts/sample-packs/9001")
    assert response.status_code == 401


def test_s3_backend_uses_configured_bucket_and_signing_options(monkeypatch):
    captured = {}

    class FakeS3FileSystem:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(storage_service, "S3FileSystem", FakeS3FileSystem)
    monkeypatch.setattr(storage_service, "S3_BUCKET", "avsiqvoiceagent")
    monkeypatch.setattr(
        storage_service, "S3_ENDPOINT_URL", "https://storage.example.invalid/s3"
    )
    monkeypatch.setattr(storage_service, "S3_SIGNATURE_VERSION", "s3v4")
    monkeypatch.setattr(storage_service, "S3_ADDRESSING_STYLE", "path")

    storage_service.get_storage_for_backend(StorageBackend.S3.value)

    assert captured == {
        "bucket_name": "avsiqvoiceagent",
        "region_name": storage_service.S3_REGION,
        "endpoint_url": "https://storage.example.invalid/s3",
        "signature_version": "s3v4",
        "addressing_style": "path",
    }


@pytest.mark.asyncio
async def test_playback_uses_one_hour_inline_signed_url(monkeypatch):
    asset = SimpleNamespace(
        id=202,
        pack_id=101,
        status="completed",
        storage_key="gemini-tts-samples/101/Kore.wav",
        mime_type="audio/wav",
    )
    get_asset = AsyncMock(return_value=asset)
    signed_url = AsyncMock(return_value="https://storage.example.invalid/signed")
    monkeypatch.setattr(
        gemini_tts_samples.db_client, "get_gemini_tts_sample_asset", get_asset
    )
    monkeypatch.setattr(gemini_tts_samples.storage_fs, "aget_signed_url", signed_url)

    response = await gemini_tts_samples.get_gemini_tts_sample_playback_url(101, 202)

    assert response["mime_type"] == "audio/wav"
    signed_url.assert_awaited_once_with(
        "gemini-tts-samples/101/Kore.wav", expiration=3600, force_inline=True
    )


@pytest.mark.asyncio
async def test_pack_creation_is_idempotent_and_does_not_requeue_existing(monkeypatch):
    monkeypatch.setattr(constants, "GEMINI_TTS_SAMPLE_API_KEY", "platform-key")
    monkeypatch.setattr(constants, "GEMINI_TTS_SAMPLE_PROJECT_ID", "project")
    monkeypatch.setattr(constants, "GEMINI_TTS_SAMPLE_LOCATION", "global")
    row = SimpleNamespace(
        id=7, assets=[SimpleNamespace(id=i, status="completed") for i in range(30)]
    )
    fake_db = SimpleNamespace(
        create_or_get_pack=AsyncMock(return_value=(row, False)),
        ensure_gemini_tts_sample_pack_assets=AsyncMock(),
        get_gemini_tts_sample_pack=AsyncMock(return_value=row),
    )
    monkeypatch.setattr(packs, "db_client", fake_db)
    enqueue = AsyncMock()
    monkeypatch.setattr(packs, "enqueue_job", enqueue)
    request = GeminiTTSSamplePackCreateRequest(
        model_id="gemini-3.1-flash-tts-preview",
        location="global",
        language="en-US",
        style_text="warm",
        sample_text="hello",
    )
    result, created = await packs.create_sample_pack(request)
    assert result is row and created is False
    enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_pack_creation_with_one_voice_queues_only_that_voice(monkeypatch):
    monkeypatch.setattr(constants, "GEMINI_TTS_SAMPLE_API_KEY", "platform-key")
    monkeypatch.setattr(constants, "GEMINI_TTS_SAMPLE_PROJECT_ID", "project")
    monkeypatch.setattr(constants, "GEMINI_TTS_SAMPLE_LOCATION", "global")
    row = SimpleNamespace(
        assets=[
            SimpleNamespace(id=16, status="queued", voice_id="Kore", enqueue_epoch=0)
        ]
    )
    fake_db = SimpleNamespace(
        create_or_get_pack=AsyncMock(return_value=(row, True)),
        get_gemini_tts_sample_pack=AsyncMock(return_value=row),
        ensure_gemini_tts_sample_pack_assets=AsyncMock(),
    )
    monkeypatch.setattr(packs, "db_client", fake_db)
    enqueue = AsyncMock()
    monkeypatch.setattr(packs, "enqueue_job", enqueue)

    result, created = await packs.create_sample_pack(
        GeminiTTSSamplePackCreateRequest(
            model_id="gemini-3.1-flash-tts-preview",
            voice_id="Kore",
            location="global",
            language="en-US",
            style_text="warm",
            sample_text="hello",
        )
    )

    assert result is row and created is True
    fake_db.ensure_gemini_tts_sample_pack_assets.assert_not_awaited()
    enqueue.assert_awaited_once()
    assert enqueue.await_args.args[1] == 16


@pytest.mark.asyncio
async def test_retry_reports_selected_and_enqueued_voice_operations(monkeypatch):
    row = SimpleNamespace(
        model_id="gemini-3.1-flash-tts-preview",
        location="global",
        catalog_revision=None,
        language="en-US",
        style_text="warm",
        sample_text="hello",
    )
    summary = dict(
        operation_id="op",
        eligible_voices=2,
        selected_voices=2,
        skipped_playable_voices=1,
        skipped_active_voices=1,
        enqueued_jobs=0,
        enqueue_conflicts=0,
        enqueue_failures=0,
        asset_ids=[11, 12],
        job_ids=[],
    )
    fake_db = SimpleNamespace(
        get_gemini_tts_sample_pack=AsyncMock(return_value=row),
        queue_failed_voice_recovery=AsyncMock(return_value=([11, 12], summary)),
    )
    monkeypatch.setattr(packs, "db_client", fake_db)
    enqueue = AsyncMock(
        side_effect=[SimpleNamespace(job_id="new-11"), SimpleNamespace(job_id="new-12")]
    )
    monkeypatch.setattr(packs, "enqueue_job", enqueue)
    retried, result = await packs.retry_failed_sample_pack(7, limit=2)
    assert retried is row and result["enqueued_jobs"] == 2
    assert result["job_ids"] == ["new-11", "new-12"]
    assert fake_db.queue_failed_voice_recovery.await_args.kwargs["limit"] == 2
    assert enqueue.await_args_list[0].kwargs["_job_id"] == "gemini-tts-sample-asset-11"


@pytest.mark.asyncio
async def test_individual_generation_enqueues_exactly_one_canonical_voice(monkeypatch):
    pack = SimpleNamespace(
        id=7,
        model_id="gemini-3.1-flash-tts-preview",
        catalog_revision=None,
        location="global",
        language="en-US",
        style_text="warm",
        sample_text="hello",
        assets=[],
    )
    asset = SimpleNamespace(id=41, voice_id="Kore", status="queued")
    fake_db = SimpleNamespace(
        get_gemini_tts_sample_pack=AsyncMock(side_effect=[pack, pack]),
        queue_voice_generation=AsyncMock(return_value=(asset, None)),
    )
    monkeypatch.setattr(packs, "db_client", fake_db)
    enqueue = AsyncMock()
    monkeypatch.setattr(packs, "enqueue_job", enqueue)

    await packs.generate_sample_voice(
        7, "Kore", packs.GeminiTTSSampleVoiceGenerationRequest()
    )

    fake_db.queue_voice_generation.assert_awaited_once_with(
        7, voice_id="Kore", gender="Female", regenerate=False
    )
    enqueue.assert_awaited_once()
    assert enqueue.await_args.args[1] == 41


@pytest.mark.asyncio
async def test_completed_voice_requires_confirmation_and_regeneration_is_one_voice(
    monkeypatch,
):
    pack = SimpleNamespace(
        id=7,
        model_id="gemini-3.1-flash-tts-preview",
        catalog_revision=None,
        location="global",
        language="en-US",
        style_text="warm",
        sample_text="hello",
        assets=[],
    )
    old = SimpleNamespace(
        id=41, voice_id="Kore", status="completed", version=1, is_current=True
    )
    new = SimpleNamespace(
        id=42, voice_id="Kore", status="queued", version=2, is_current=False
    )
    fake_db = SimpleNamespace(
        get_gemini_tts_sample_pack=AsyncMock(side_effect=[pack, pack, pack]),
        queue_voice_generation=AsyncMock(
            side_effect=[(old, "regeneration_confirmation_required"), (new, None)]
        ),
    )
    monkeypatch.setattr(packs, "db_client", fake_db)
    enqueue = AsyncMock()
    monkeypatch.setattr(packs, "enqueue_job", enqueue)

    with pytest.raises(PermissionError, match="new Google TTS request"):
        await packs.generate_sample_voice(
            7, "Kore", packs.GeminiTTSSampleVoiceGenerationRequest()
        )
    await packs.generate_sample_voice(
        7, "Kore", packs.GeminiTTSSampleVoiceGenerationRequest(regenerate=True)
    )
    assert enqueue.await_args.args[1] == new.id
    assert old.id != new.id  # the old immutable row is never replaced


@pytest.mark.asyncio
async def test_individual_generation_reports_existing_queue_without_enqueuing(
    monkeypatch,
):
    pack = SimpleNamespace(
        id=7,
        model_id="gemini-3.1-flash-tts-preview",
        catalog_revision=None,
        location="global",
        language="en-US",
        style_text="warm",
        sample_text="hello",
        assets=[],
    )
    fake_db = SimpleNamespace(
        get_gemini_tts_sample_pack=AsyncMock(return_value=pack),
        queue_voice_generation=AsyncMock(
            return_value=(SimpleNamespace(id=41, status="running"), "running")
        ),
    )
    monkeypatch.setattr(packs, "db_client", fake_db)
    enqueue = AsyncMock()
    monkeypatch.setattr(packs, "enqueue_job", enqueue)

    with pytest.raises(RuntimeError, match="already running"):
        await packs.generate_sample_voice(
            7, "Kore", packs.GeminiTTSSampleVoiceGenerationRequest()
        )
    enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_catalog_read_does_not_invoke_google(monkeypatch):
    called = False

    async def fail_google(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("catalog must not synthesize")

    monkeypatch.setattr(
        "api.services.gemini_tts_sample_library.synthesize_sample_wav", fail_google
    )
    response = await get_google_vertex_tts_catalog(
        model="gemini-3.1-flash-tts-preview",
        catalog_revision=None,
        location=None,
        language=None,
        context=None,
        sample_text=None,
        user=SimpleNamespace(),
    )
    assert len(response["models"][0]["voices"]) == 30
    assert called is False


@pytest.mark.asyncio
async def test_platform_catalog_preserves_sample_filters(monkeypatch):
    lookup = AsyncMock(return_value=[])
    monkeypatch.setattr(
        workflow_model_slots.db_client, "list_matching_completed_assets", lookup
    )

    await workflow_model_slots.get_platform_gemini_tts_catalog(
        model="gemini-3.1-flash-tts-preview",
        catalog_revision="be1fc646b9e6f5bc",
        location="global",
        language="en-US",
        context="warm",
        sample_text="hello",
    )

    lookup.assert_awaited_once_with(
        model_id="gemini-3.1-flash-tts-preview",
        catalog_revision="be1fc646b9e6f5bc",
        location="global",
        language="en-US",
        style_text="warm",
        sample_text="hello",
    )


@pytest.mark.asyncio
async def test_catalog_sample_lookup_matches_revision_and_location(monkeypatch):
    pack = SimpleNamespace(
        id=101,
        catalog_revision="be1fc646b9e6f5bc",
        location="global",
        language="en-US",
        style_text="warm",
        sample_text="hello",
    )
    asset = SimpleNamespace(
        id=202,
        voice_id="Kore",
        storage_key="gemini-tts-samples/101/Kore.wav",
        duration_seconds=1.0,
        sha256="fake-sha256",
    )
    lookup = AsyncMock(return_value=[(pack, asset)])
    monkeypatch.setattr(
        workflow_model_slots.db_client, "list_matching_completed_assets", lookup
    )
    monkeypatch.setattr(
        workflow_model_slots.storage_fs,
        "aget_signed_url",
        AsyncMock(return_value="https://storage.invalid/sample.wav"),
    )

    response = await get_google_vertex_tts_catalog(
        model="gemini-3.1-flash-tts-preview",
        catalog_revision="be1fc646b9e6f5bc",
        location="global",
        language="en-US",
        context="warm",
        sample_text="hello",
        user=SimpleNamespace(),
    )

    lookup.assert_awaited_once_with(
        model_id="gemini-3.1-flash-tts-preview",
        catalog_revision="be1fc646b9e6f5bc",
        location="global",
        language="en-US",
        style_text="warm",
        sample_text="hello",
    )
    assert response["models"][0]["catalog_revision"] == "be1fc646b9e6f5bc"
    assert response["models"][0]["voices"][15]["samples"][0]["location"] == "global"
    assert (
        response["models"][0]["voices"][15]["sample_url"]
        == "https://storage.invalid/sample.wav"
    )
