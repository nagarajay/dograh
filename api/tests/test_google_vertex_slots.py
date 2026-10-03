"""Google Vertex across the LLM / STT / TTS / embeddings slots."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
from api.services.configuration import effective_readback
from api.services.configuration.check_validity import UserConfigurationValidator
from api.services.configuration.options.google_vertex_catalog import (
    check_vertex_config,
    get_vertex_model,
    vertex_models,
)
from api.services.configuration.registry import (
    REGISTRY,
    GoogleVertexEmbeddingsConfiguration,
    GoogleVertexLLMConfiguration,
    GoogleVertexSTTConfiguration,
    GoogleVertexTTSConfiguration,
    ServiceProviders,
    ServiceType,
)
from api.services.gen_ai.embedding.google_vertex_service import (
    GoogleVertexEmbeddingService,
    VertexEmbeddingConfigError,
)
from api.services.pipecat import service_factory as sf

SECRET_KEY = "AQ.SECRETVERTEXKEY123"
SA = json.dumps(
    {
        "type": "service_account",
        "project_id": "proj-1",
        "private_key": "-----BEGIN PRIVATE KEY-----\nSECRETBODY\n-----END PRIVATE KEY-----\n",
        "client_email": "svc@proj-1.iam.gserviceaccount.com",
    }
)


def _errors(service, name):
    return UserConfigurationValidator()._validate_service(service, name)


# --------------------------------------------------------------- registry
def test_vertex_registered_in_every_slot():
    for slot in (
        ServiceType.LLM,
        ServiceType.STT,
        ServiceType.TTS,
        ServiceType.EMBEDDINGS,
    ):
        assert ServiceProviders.GOOGLE_VERTEX in REGISTRY[slot]


def test_defaults_are_catalogued():
    assert GoogleVertexSTTConfiguration().model in vertex_models("stt")
    assert GoogleVertexTTSConfiguration().model in vertex_models("tts")
    assert GoogleVertexEmbeddingsConfiguration().model in vertex_models("embeddings")


# ---------------------------------------------------------------- catalog
def test_catalog_metadata_complete():
    for slot in ("stt", "tts", "embeddings"):
        for model_id in vertex_models(slot):
            entry = get_vertex_model(slot, model_id)
            assert entry.api and entry.lifecycle in ("ga", "preview")
            assert entry.auth["service_account"] == "documented"
    assert get_vertex_model("stt", "gemini-3.5-transcribe-live-preview").streaming
    assert get_vertex_model("embeddings", "gemini-embedding-001").dimensions == 1536


def test_unknown_model_rejected_outside_llm():
    msg = check_vertex_config(
        "tts", model="nope", location="global", has_api_key=False, has_credentials=True
    )
    assert "not in Dograh's catalogue" in msg
    assert (
        check_vertex_config(
            "llm",
            model="gemini-9-future",
            location="global",
            has_api_key=True,
            has_credentials=False,
        )
        is None
    )


def test_location_errors():
    msg = check_vertex_config(
        "tts",
        model="gemini-3.1-flash-tts-preview",
        location="us",
        has_api_key=False,
        has_credentials=True,
    )
    assert "not available in location 'us'" in msg
    assert (
        check_vertex_config(
            "tts",
            model="gemini-2.5-flash-tts",
            location="europe-west4",
            has_api_key=False,
            has_credentials=True,
        )
        is None
    )
    assert check_vertex_config(
        "stt",
        model="gemini-3.5-transcribe-live-preview",
        location="us-central1",
        has_api_key=False,
        has_credentials=True,
    )


# -------------------------------------------------------------- validation
def test_llm_accepts_api_key_without_project():
    llm = GoogleVertexLLMConfiguration(api_key=SECRET_KEY)
    assert _errors(llm, "llm") == []


def test_llm_rejects_key_and_service_account_together():
    llm = GoogleVertexLLMConfiguration(
        project_id="p", api_key=SECRET_KEY, credentials=SA
    )
    errors = _errors(llm, "llm")
    assert errors and "not both" in errors[0]["message"]


def test_llm_without_key_needs_project():
    assert _errors(GoogleVertexLLMConfiguration(), "llm")


@pytest.mark.parametrize(
    "cfg,name",
    [
        (lambda **k: GoogleVertexEmbeddingsConfiguration(**k), "embeddings"),
    ],
)
def test_unverified_api_key_rejected_without_echo(cfg, name):
    kwargs = {"api_key": SECRET_KEY, "project_id": "p"}
    errors = _errors(cfg(**kwargs), name)
    assert errors
    assert SECRET_KEY not in errors[0]["message"]
    assert "Application Default" in errors[0]["message"]


@pytest.mark.parametrize(
    "cfg,name",
    [
        (GoogleVertexSTTConfiguration(project_id="p", credentials=SA), "stt"),
        (GoogleVertexSTTConfiguration(project_id="p"), "stt"),  # ADC
        (GoogleVertexTTSConfiguration(credentials=SA), "tts"),
        (GoogleVertexTTSConfiguration(), "tts"),
        (
            GoogleVertexEmbeddingsConfiguration(project_id="p", credentials=SA),
            "embeddings",
        ),
    ],
)
def test_service_account_and_adc_accepted(cfg, name):
    assert _errors(cfg, name) == []


def test_stt_needs_project_for_service_account_paths():
    assert _errors(GoogleVertexSTTConfiguration(credentials=SA), "stt")


def test_invalid_credentials_json_not_echoed():
    errors = _errors(
        GoogleVertexTTSConfiguration(credentials="not-json-SECRETTEXT"), "tts"
    )
    assert errors and "SECRETTEXT" not in errors[0]["message"]


# ---------------------------------------------------------------- factory
def _audio():
    return SimpleNamespace(transport_in_sample_rate=16000)


def test_llm_api_key_builds_express_client():
    with patch.object(sf, "GenaiClient") as client:
        sf.create_llm_service_from_provider(
            provider=ServiceProviders.GOOGLE_VERTEX.value,
            model="gemini-3.5-flash",
            api_key=SECRET_KEY,
        )
    kwargs = client.call_args.kwargs
    assert kwargs["vertexai"] is True and kwargs["api_key"] == SECRET_KEY
    assert "project" not in kwargs and "credentials" not in kwargs


def test_llm_service_account_path_unchanged():
    with (
        patch.object(
            sf.GoogleVertexLLMService, "_get_credentials", return_value="creds"
        ),
        patch("pipecat.services.google.vertex.llm.Client") as client,
    ):
        sf.create_llm_service_from_provider(
            provider=ServiceProviders.GOOGLE_VERTEX.value,
            model="gemini-3.5-flash",
            api_key=None,
            project_id="proj-1",
            location="global",
            credentials=SA,
        )
    kwargs = client.call_args.kwargs
    assert kwargs["project"] == "proj-1" and kwargs["credentials"] == "creds"
    assert "api_key" not in kwargs


def test_stt_uses_gemini_live_on_vertex():
    cfg = GoogleVertexSTTConfiguration(
        project_id="proj-1", credentials=SA, language="en-US"
    )
    with (
        patch.object(sf, "GenaiClient") as client,
        patch.object(
            sf.GoogleVertexLLMService, "_get_credentials", return_value="creds"
        ),
    ):
        service = sf.create_stt_service(SimpleNamespace(stt=cfg), _audio())
    assert isinstance(service, sf.DograhGeminiVertexSTTService)
    assert service._settings.model == "gemini-3.5-transcribe-live-preview"
    kwargs = client.call_args.kwargs
    assert kwargs["vertexai"] is True
    assert kwargs["project"] == "proj-1" and kwargs["location"] == "global"
    assert kwargs["credentials"] == "creds"


def test_stt_without_language_auto_detects():
    cfg = GoogleVertexSTTConfiguration(project_id="proj-1")
    with (
        patch.object(sf, "GenaiClient"),
        patch.object(
            sf.GoogleVertexLLMService, "_get_credentials", return_value="creds"
        ),
    ):
        service = sf.create_stt_service(SimpleNamespace(stt=cfg), _audio())
    assert service._get_language_codes() == []


def test_tts_uses_cloud_gemini_tts_not_genai():
    cfg = GoogleVertexTTSConfiguration(
        model="gemini-2.5-flash-tts", voice="Puck", location="global", prompt="warm"
    )
    with patch.object(sf, "GeminiTTSService") as svc:
        sf.create_tts_service(SimpleNamespace(tts=cfg), _audio())
    kwargs = svc.call_args.kwargs
    assert kwargs["use_genai"] is False
    assert kwargs["location"] is None  # global host has no prefix
    assert "api_key" not in kwargs
    assert kwargs["settings"].model == "gemini-2.5-flash-tts"
    assert kwargs["settings"].voice == "Puck"
    assert kwargs["settings"].prompt == "warm"


def test_tts_regional_location_passed_through():
    cfg = GoogleVertexTTSConfiguration(location="eu")
    with patch.object(sf, "GeminiTTSService") as svc:
        sf.create_tts_service(SimpleNamespace(tts=cfg), _audio())
    assert svc.call_args.kwargs["location"] == "eu"


# ------------------------------------------------------------- embeddings
class _FakeModels:
    def __init__(self, dim=1536):
        self.calls = []
        self.dim = dim

    async def embed_content(self, *, model, contents, config):
        self.calls.append((model, list(contents), config))
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1] * self.dim) for _ in contents]
        )


def _service(dim=1536):
    models = _FakeModels(dim)
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    return GoogleVertexEmbeddingService(db_client=MagicMock(), client=client), models


@pytest.mark.asyncio
async def test_embeddings_request_shape_and_batching():
    service, models = _service()
    vectors = await service.embed_texts([f"t{i}" for i in range(20)])
    assert len(vectors) == 20 and len(vectors[0]) == 1536
    assert [len(c[1]) for c in models.calls] == [16, 4]
    model, _, config = models.calls[0]
    assert model == "gemini-embedding-001"
    assert config.output_dimensionality == 1536
    assert config.task_type == "RETRIEVAL_DOCUMENT"
    await service.embed_query("q")
    assert models.calls[-1][2].task_type == "RETRIEVAL_QUERY"
    assert service.get_embedding_dimension() == 1536


@pytest.mark.asyncio
async def test_embeddings_reject_wrong_dimension():
    service, _ = _service(dim=768)
    with pytest.raises(ValueError, match="768-dimensional"):
        await service.embed_texts(["x"])


def test_embeddings_reject_uncatalogued_model():
    with pytest.raises(VertexEmbeddingConfigError):
        GoogleVertexEmbeddingService(
            db_client=MagicMock(), model_id="text-embedding-005"
        )


@pytest.mark.asyncio
async def test_embeddings_search_scopes_by_org_and_model():
    service, _ = _service()
    service.db.search_similar_chunks = AsyncMock(return_value=[])
    await service.search_similar_chunks("q", organization_id=7, limit=3)
    kwargs = service.db.search_similar_chunks.call_args.kwargs
    assert kwargs["organization_id"] == 7
    assert kwargs["embedding_model"] == "gemini-embedding-001"


@pytest.mark.asyncio
async def test_factory_builds_vertex_embeddings_from_one_effective_config(monkeypatch):
    """Every Vertex field comes from the single config the caller resolved."""
    from api.services.gen_ai.embedding import factory

    slot_cfg = GoogleVertexEmbeddingsConfiguration(
        project_id="slot-project",
        location="us-central1",
        credentials=SA,
        api_key=None,
    )

    async def must_not_be_called(**_):  # the old second lookup of the org config
        raise AssertionError("factory must not re-resolve the organization config")

    monkeypatch.setattr(
        "api.services.configuration.ai_model_configuration."
        "get_resolved_ai_model_configuration",
        must_not_be_called,
    )
    service = await factory.build_embedding_service(
        db_client=MagicMock(),
        provider="google_vertex",
        api_key=None,
        model="gemini-embedding-001",
        embeddings_config=slot_cfg,
    )
    assert isinstance(service, GoogleVertexEmbeddingService)
    assert service._location == "us-central1" and service._project_id == "slot-project"
    assert service._credentials == SA and service.get_model_id() == slot_cfg.model


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,config,model",
    [
        ("google_vertex", None, "gemini-embedding-001"),  # no coherent config
        ("google_vertex", SimpleNamespace(provider="openai", model="x"), None),
        (
            "google_vertex",
            GoogleVertexEmbeddingsConfiguration(project_id="p", credentials=SA),
            "some-other-model",  # model from one place, config from another
        ),
    ],
)
async def test_factory_refuses_to_assemble_vertex_embeddings_from_mixed_sources(
    provider, config, model
):
    from api.services.gen_ai.embedding import factory

    with pytest.raises(ValueError):
        await factory.build_embedding_service(
            db_client=MagicMock(),
            provider=provider,
            api_key="k",
            model=model,
            embeddings_config=config,
        )


@pytest.mark.asyncio
async def test_runtime_retrieval_uses_the_slot_config_not_the_organization_project(
    monkeypatch,
):
    """A Vertex slot on a workflow whose organization is on another provider/project."""
    from api.services.workflow.tools import knowledge_base as tool

    slot_cfg = GoogleVertexEmbeddingsConfiguration(
        project_id="slot-project", location="us-central1", credentials=SA
    )
    seen = {}

    async def fake_search(self, **kwargs):
        seen.update(
            project=self._project_id, location=self._location, key=self._api_key
        )
        return []

    monkeypatch.setattr(
        GoogleVertexEmbeddingService, "search_similar_chunks", fake_search
    )
    result = await tool._perform_retrieval(
        "q",
        7,
        None,
        3,
        None,
        slot_cfg.model,
        None,
        "google_vertex",
        None,
        None,
        None,
        embeddings_config=slot_cfg,
    )
    assert result["total_results"] == 0
    assert seen == {"project": "slot-project", "location": "us-central1", "key": None}


# ---------------------------------------------------------------- readback
@pytest.mark.asyncio
async def test_readback_describes_vertex_slots_without_secrets(monkeypatch):
    effective = EffectiveAIModelConfiguration(
        llm=GoogleVertexLLMConfiguration(api_key=SECRET_KEY),
        stt=GoogleVertexSTTConfiguration(project_id="proj-1", credentials=SA),
        tts=GoogleVertexTTSConfiguration(),
        embeddings=GoogleVertexEmbeddingsConfiguration(project_id="proj-1"),
    )

    async def fake(**_):
        return effective

    async def resolved(**kwargs):
        return await fake(**kwargs), {}

    monkeypatch.setattr(
        effective_readback,
        "resolve_effective_ai_model_configuration_for_workflow",
        resolved,
    )
    out = await effective_readback.build_effective_model_configuration_readback(
        organization_id=1, workflow_configurations={}
    )
    dumped = json.dumps(out, default=str)
    assert SECRET_KEY not in dumped and "SECRETBODY" not in dumped
    assert out["llm"]["credential"] == {"kind": "api_key", "configured": True}
    assert out["stt"]["credential"]["kind"] == "service_account_json"
    assert out["tts"]["credential"]["kind"] == "application_default_credentials"
    assert out["stt"]["catalog"]["streaming"] is True
    assert out["stt"]["catalog"]["lifecycle"] == "preview"
    assert out["embeddings"]["catalog"]["auth"]["api_key"] == "unverified"
