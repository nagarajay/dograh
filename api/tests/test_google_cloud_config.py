import json

import pytest

from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
from api.services.configuration import effective_readback
from api.services.configuration.check_validity import UserConfigurationValidator
from api.services.configuration.registry import (
    GoogleSTTConfiguration,
    GoogleTTSConfiguration,
    GoogleVertexLLMConfiguration,
)

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


def test_stt_tts_accept_service_account_and_adc():
    assert _errors(GoogleSTTConfiguration(credentials=SA), "stt") == []
    assert _errors(GoogleTTSConfiguration(credentials=SA), "tts") == []
    assert _errors(GoogleTTSConfiguration(), "tts") == []  # ADC


def test_api_key_string_rejected_without_echo():
    errors = _errors(GoogleSTTConfiguration(credentials="AIzaSyFAKEKEY123"), "stt")
    assert errors and "AIzaSyFAKEKEY123" not in errors[0]["message"]
    assert "API key" in errors[0]["message"]


def test_incomplete_service_account_rejected():
    bad = json.dumps({"type": "service_account", "project_id": "p"})
    errors = _errors(GoogleTTSConfiguration(credentials=bad), "tts")
    assert "private_key" in errors[0]["message"]


def test_chirp3_requires_us_or_eu():
    assert _errors(GoogleSTTConfiguration(model="chirp_3", credentials=SA), "stt")
    ok = GoogleSTTConfiguration(model="chirp_3", location="us", credentials=SA)
    assert _errors(ok, "stt") == []


def test_vertex_checks_credentials_shape():
    llm = GoogleVertexLLMConfiguration(project_id="proj-1", credentials="notjson")
    assert _errors(llm, "llm")
    llm = GoogleVertexLLMConfiguration(project_id="proj-1", credentials=SA)
    assert _errors(llm, "llm") == []


@pytest.mark.asyncio
async def test_readback_never_contains_secrets(monkeypatch):
    effective = EffectiveAIModelConfiguration(
        llm=GoogleVertexLLMConfiguration(project_id="proj-1", credentials=SA),
        stt=GoogleSTTConfiguration(credentials=SA),
        tts=GoogleTTSConfiguration(),
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
    assert "SECRETBODY" not in dumped and "PRIVATE KEY" not in dumped
    assert out["llm"]["credential"] == {
        "kind": "service_account_json",
        "configured": True,
        "service_account_email": "svc@proj-1.iam.gserviceaccount.com",
        "credential_project_id": "proj-1",
    }
    assert out["tts"]["credential"]["kind"] == "application_default_credentials"
    assert out["llm"]["model"] and out["llm"]["location"] == "global"
