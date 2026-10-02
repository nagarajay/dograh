"""Secret-free readback of the model configuration a workflow run will use.

The model configuration comes from the same resolver the runtime calls, so
defaults and workflow overrides are already applied. Secrets never leave this
module: each service reports only whether a credential is set and what kind.
"""

import json
from typing import Any

from api.services.configuration.ai_model_configuration import (
    get_effective_ai_model_configuration_for_workflow,
)
from api.services.configuration.masking import SERVICE_SECRET_FIELDS
from api.services.configuration.options.google_vertex_catalog import (
    GEMINI_TTS_DOCUMENTATION_URL,
    gemini_tts_catalog_revision,
    gemini_tts_voices,
    get_vertex_model,
)

_SERVICE_SECTIONS = ("llm", "stt", "tts", "realtime", "embeddings")


def _credential_status(data: dict[str, Any]) -> dict[str, Any]:
    raw_credentials = data.get("credentials")
    if raw_credentials:
        status: dict[str, Any] = {"kind": "service_account_json", "configured": True}
        try:
            info = json.loads(raw_credentials)
        except (TypeError, ValueError):
            info = None
        if isinstance(info, dict):
            # Identifiers, not secrets: they tell an operator which account is in use.
            status["service_account_email"] = info.get("client_email")
            status["credential_project_id"] = info.get("project_id")
        else:
            status["configured"] = False
            status["problem"] = "credentials are not valid service-account JSON"
        return status
    if data.get("api_key") or data.get("aws_access_key"):
        return {"kind": "api_key", "configured": True}
    if data.get("provider") in ("google_vertex", "google_vertex_realtime") or (
        data.get("provider") == "google" and "credentials" in data
    ):
        return {"kind": "application_default_credentials", "configured": None}
    return {"kind": "none", "configured": False}


def _vertex_catalog_entry(section: str, data: dict[str, Any]) -> dict[str, Any] | None:
    if data.get("provider") != "google_vertex" or section not in (
        "llm",
        "stt",
        "tts",
        "embeddings",
    ):
        return None
    entry = get_vertex_model(section, data.get("model", ""))
    if entry is None:
        return None
    described = {
        "api": entry.api,
        "lifecycle": entry.lifecycle,
        "streaming": entry.streaming,
        "auth": dict(entry.auth),
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
                "sample_url": None,
                "sample_reuse": "official_demo_page_only",
            }
            for voice in gemini_tts_voices(entry.id)
        ],
    }
    if section == "tts":
        described["sample_library"] = {
            "catalog_revision": gemini_tts_catalog_revision(entry.id),
            "lookup": (
                "GET /api/v1/model-slots/google-vertex/tts/catalog?model="
                f"{entry.id}&language=<language>&context=<style_text>&"
                "sample_text=<spoken_text>"
            ),
            "google_requests_on_read": False,
        }
    return described


def _describe_service(service: Any, section: str = "") -> dict[str, Any] | None:
    if service is None:
        return None
    data = service.model_dump()
    described = {
        key: value for key, value in data.items() if key not in SERVICE_SECRET_FIELDS
    }
    described["credential"] = _credential_status(data)
    catalog = _vertex_catalog_entry(section, data)
    if catalog is not None:
        described["catalog"] = catalog
    return described


async def build_effective_model_configuration_readback(
    *,
    organization_id: int,
    workflow_configurations: dict | None,
    workflow_id: int | None = None,
) -> dict[str, Any]:
    effective = await get_effective_ai_model_configuration_for_workflow(
        organization_id=organization_id,
        workflow_configurations=workflow_configurations,
        workflow_id=workflow_id,
    )
    readback: dict[str, Any] = {"is_realtime": bool(effective.is_realtime)}
    for section in _SERVICE_SECTIONS:
        readback[section] = _describe_service(
            getattr(effective, section, None), section
        )
    return readback
