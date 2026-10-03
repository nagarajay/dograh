"""Catalogue of Google Vertex models Dograh can use, per provider slot.

One entry per model. Adapters, validators and the configuration schema read
from here, so supporting a new Google model means adding an entry, not new
provider logic. Entries are pinned: nothing here auto-upgrades a stored
workflow configuration to a newer model.

Sources (checked 2026-09-30):
- Express mode / API key: docs.cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview
- Gemini 3.5 Transcribe:  docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-5-transcribe
- Gemini-TTS:             docs.cloud.google.com/text-to-speech/docs/gemini-tts
- API keys by method:     docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/authentication
- Text embeddings:        docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings

Auth statuses:
- ``documented``   Google's documentation shows this credential working.
- ``method_documented``  Google documents API keys for the API *method* the adapter
                   calls (``generateContent`` / ``streamGenerateContent``) but not
                   per model. Accepted; run the live probe before relying on it.
- ``probe_verified``     Not documented, but a live probe succeeded with a complete
                   ``projects/{p}/locations/{l}/publishers/google/models/{m}``
                   resource (2026-10-01). Accepted; requires ``project_id``.
- ``unverified``   Not documented for this API. Rejected by validation until a
                   live probe proves it and the entry is flipped.
- ``unsupported``  Google documents a different credential type only.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal, Mapping

Slot = Literal["llm", "stt", "tts", "embeddings"]
AuthMethod = Literal["api_key", "service_account", "adc"]
AuthStatus = Literal[
    "documented", "method_documented", "probe_verified", "unverified", "unsupported"
]
Lifecycle = Literal["ga", "preview"]

_SA_ADC: Mapping[str, str] = {
    "service_account": "documented",
    "adc": "documented",
}


@dataclass(frozen=True)
class VertexModel:
    id: str
    slot: Slot
    # Transport the adapter uses, for readback and operators.
    api: str
    lifecycle: Lifecycle
    auth: Mapping[str, str]
    streaming: bool
    # Locations Google documents for this model. Empty means "not restricted here".
    locations: tuple[str, ...] = ()
    # True when Google documents further single regions beyond ``locations``
    # whose per-model availability this catalogue does not track.
    other_regions_allowed: bool = False
    # Fixed output shape the adapter converts to, where it matters.
    output_format: str | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    # Vector size Dograh stores (embeddings only).
    dimensions: int | None = None
    notes: str = ""
    extra: Mapping[str, object] = field(default_factory=dict)

    def auth_status(self, method: str) -> str:
        return self.auth.get(method, "unsupported")


GEMINI_TTS_DOCUMENTATION_URL = (
    "https://docs.cloud.google.com/text-to-speech/docs/gemini-tts"
)


@dataclass(frozen=True)
class GeminiTTSVoice:
    id: str
    gender: Literal["Female", "Male"]


# Google documents these as the Gemini-TTS prebuilt voice names.  They are
# model-aware in the public response, even though the documented list is shared
# by the supported Gemini-TTS models.
GEMINI_TTS_VOICES: tuple[GeminiTTSVoice, ...] = tuple(
    GeminiTTSVoice(name, gender)
    for name, gender in (
        ("Achernar", "Female"),
        ("Achird", "Male"),
        ("Algenib", "Male"),
        ("Algieba", "Male"),
        ("Alnilam", "Male"),
        ("Aoede", "Female"),
        ("Autonoe", "Female"),
        ("Callirrhoe", "Female"),
        ("Charon", "Male"),
        ("Despina", "Female"),
        ("Enceladus", "Male"),
        ("Erinome", "Female"),
        ("Fenrir", "Male"),
        ("Gacrux", "Female"),
        ("Iapetus", "Male"),
        ("Kore", "Female"),
        ("Laomedeia", "Female"),
        ("Leda", "Female"),
        ("Orus", "Male"),
        ("Pulcherrima", "Female"),
        ("Puck", "Male"),
        ("Rasalgethi", "Male"),
        ("Sadachbia", "Male"),
        ("Sadaltager", "Male"),
        ("Schedar", "Male"),
        ("Sulafat", "Female"),
        ("Umbriel", "Male"),
        ("Vindemiatrix", "Female"),
        ("Zephyr", "Female"),
        ("Zubenelgenubi", "Male"),
    )
)


def gemini_tts_voices(model_id: str) -> tuple[GeminiTTSVoice, ...]:
    """Return Google's documented prebuilt voices for a supported TTS model."""
    if get_vertex_model("tts", model_id) is None:
        return ()
    return GEMINI_TTS_VOICES


def gemini_tts_catalog_revision(model_id: str) -> str | None:
    """Stable revision for the exact model/voice metadata exposed to clients."""
    entry = get_vertex_model("tts", model_id)
    if entry is None:
        return None
    payload = {
        "model": entry.id,
        "lifecycle": entry.lifecycle,
        "locations": entry.locations,
        "output_format": entry.output_format,
        "sample_rate_hz": entry.sample_rate_hz,
        "channels": entry.channels,
        "voices": [(voice.id, voice.gender) for voice in gemini_tts_voices(model_id)],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


_LLM_AUTH: Mapping[str, str] = {
    # Express mode documents `genai.Client(vertexai=True, api_key=...)`.
    "api_key": "documented",
    "service_account": "documented",
    "adc": "documented",
}

_TTS_REGIONS_25 = ("global", "us", "eu")

# Gemini-TTS is served by two APIs. Service account / ADC use Cloud
# Text-to-Speech StreamingSynthesize. An API key can only go through the Vertex
# API (streamGenerateContent), the one TTS method Google lists as key-capable;
# Dograh picks the API from the credential.
_TTS_AUTH: Mapping[str, str] = {**_SA_ADC, "api_key": "method_documented"}
_TTS_NOTES = (
    "Service account/ADC: Cloud Text-to-Speech StreamingSynthesize. API key: "
    "Vertex API streamGenerateContent (PCM 16-bit 24 kHz). Without project_id "
    "Google resolves project and location from the key's account (probe: "
    "asia-southeast1, 404 for gemini-2.5-flash-tts). Dograh requires project_id "
    "for API-key TTS so it can send a complete resource at the configured "
    "location: the key + complete "
    "resource combination is live-proven (2026-10-01) but NOT documented by "
    "Google, which documents API keys only per method."
)

_MODELS: tuple[VertexModel, ...] = (
    # ------------------------------------------------------------------ LLM
    *(
        VertexModel(
            id=model_id,
            slot="llm",
            api="generateContent (streamGenerateContent)",
            lifecycle="ga",
            auth=_LLM_AUTH,
            streaming=True,
            notes=(
                "Gemini gen-3 models are served from the 'global' location. "
                "Express-mode API keys only cover the models Google lists for "
                "express mode; other models need a billed project."
            ),
        )
        for model_id in (
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        )
    ),
    # ------------------------------------------------------------------ STT
    VertexModel(
        id="gemini-3.5-transcribe-live-preview",
        slot="stt",
        api="Live API (BidiGenerateContent) input_audio_transcription",
        lifecycle="preview",
        auth={**_SA_ADC, "api_key": "probe_verified"},
        streaming=True,
        locations=("global",),
        output_format="text (interim + final transcripts)",
        notes=(
            "Input audio/pcm 16 kHz mono; up to 10 minutes per session; 85+ languages. "
            "Google documents API keys only for generateContent/streamGenerateContent, "
            "not for the Live (BidiGenerateContent) method this model streams over; "
            "a live probe succeeded with an API key and a complete project/location/"
            "model resource. The partial resource the SDK builds for a key alone is "
            "rejected (websocket 1007), so a key needs project_id."
        ),
    ),
    # ------------------------------------------------------------------ TTS
    VertexModel(
        id="gemini-3.1-flash-tts-preview",
        slot="tts",
        api="Cloud Text-to-Speech StreamingSynthesize, or Vertex streamGenerateContent with an API key",
        lifecycle="preview",
        auth=_TTS_AUTH,
        streaming=True,
        locations=("global",),
        output_format="PCM 24 kHz mono",
        sample_rate_hz=24000,
        channels=1,
        notes=_TTS_NOTES,
    ),
    *(
        VertexModel(
            id=model_id,
            slot="tts",
            api="Cloud Text-to-Speech StreamingSynthesize, or Vertex streamGenerateContent with an API key",
            lifecycle=lifecycle,
            auth=_TTS_AUTH,
            streaming=True,
            locations=_TTS_REGIONS_25,
            other_regions_allowed=True,
            output_format="PCM 24 kHz mono",
            sample_rate_hz=24000,
            channels=1,
            notes=_TTS_NOTES,
        )
        for model_id, lifecycle in (
            ("gemini-2.5-flash-tts", "ga"),
            ("gemini-2.5-pro-tts", "ga"),
            ("gemini-2.5-flash-lite-preview-tts", "preview"),
        )
    ),
    # ----------------------------------------------------------- EMBEDDINGS
    VertexModel(
        id="gemini-embedding-001",
        slot="embeddings",
        api="embedContent (google-genai Client(vertexai=True).models.embed_content)",
        lifecycle="ga",
        auth={**_SA_ADC, "api_key": "unverified"},
        streaming=False,
        output_format="float vector",
        # Native size is 3072; output_dimensionality shrinks it. Dograh's
        # knowledge-base column is vector(1536), so that is the size requested.
        dimensions=1536,
        notes="Native 3072 dims, reduced via output_dimensionality. Max 250 inputs / 20,000 tokens per request.",
    ),
)

CATALOG: Mapping[tuple[str, str], VertexModel] = {(m.slot, m.id): m for m in _MODELS}

DEFAULT_LOCATION = "global"


def vertex_models(slot: Slot) -> tuple[str, ...]:
    return tuple(m.id for m in _MODELS if m.slot == slot)


def get_vertex_model(slot: Slot, model_id: str) -> VertexModel | None:
    return CATALOG.get((slot, model_id))


def gemini_tts_audio_format(model_id: str) -> tuple[int, int] | None:
    entry = get_vertex_model("tts", model_id)
    if entry is None or entry.sample_rate_hz is None or entry.channels is None:
        return None
    return entry.sample_rate_hz, entry.channels


def _is_region_like(location: str) -> bool:
    # europe-west4, us-central1, northamerica-northeast1 ...
    parts = location.split("-")
    return len(parts) >= 2 and all(p.isalnum() for p in parts)


def check_vertex_config(
    slot: Slot,
    *,
    model: str,
    location: str | None,
    has_api_key: bool,
    has_credentials: bool,
    project_id: str | None = None,
    voice: str | None = None,
) -> str | None:
    """Return a user-safe error message, or None when the config is coherent.

    Never includes credential values; callers pass booleans only.
    """
    entry = get_vertex_model(slot, model)
    if entry is None:
        if slot == "llm":
            # LLM model IDs stay free text so new Gemini models work without a
            # release; the catalogue only carries the ones we have checked.
            return None
        known = ", ".join(vertex_models(slot))
        return (
            f"Google Vertex {slot} model '{model}' is not in Dograh's catalogue "
            f"(known: {known}). Add it to google_vertex_catalog.py once its API "
            f"contract is verified."
        )

    if has_api_key and has_credentials:
        return (
            "Set either an API key or service-account credentials for Google "
            "Vertex, not both."
        )
    if has_api_key:
        status = entry.auth_status("api_key")
        if status == "unsupported":
            return (
                f"Google does not document API-key authentication for {entry.id}. "
                f"Use service-account JSON or Application Default Credentials."
            )

        if status == "unverified":
            return (
                f"API-key authentication is not yet verified for {entry.id} "
                f"({entry.api}). Use service-account JSON or Application Default "
                f"Credentials."
            )

    if (
        slot == "tts"
        and voice
        and voice not in {item.id for item in gemini_tts_voices(model)}
    ):
        return (
            f"Google Vertex TTS voice '{voice}' is not in the documented voice "
            f"catalogue for {model}."
        )

    if has_api_key and slot in ("stt", "tts") and not (project_id or "").strip():
        return (
            f"{entry.id} over the Vertex API needs project_id with an API key: "
            f"the API requires a complete projects/.../models/... resource."
        )
    loc = (location or DEFAULT_LOCATION).strip()
    if has_api_key and slot != "llm" and loc != DEFAULT_LOCATION:
        return (
            f"API-key requests for {entry.id} cannot choose a location (Google "
            f"resolves it from the key's account); set location to "
            f"'{DEFAULT_LOCATION}' or use a service account."
        )
    if entry.locations and loc not in entry.locations:
        if not (entry.other_regions_allowed and _is_region_like(loc)):
            return (
                f"{entry.id} is not available in location '{loc}'. Documented "
                f"locations: {', '.join(entry.locations)}."
            )
    return None
