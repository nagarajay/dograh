"""Platform-owned, replayable Gemini-TTS sample generation."""

from __future__ import annotations

import asyncio
import hashlib
import io
import re
import time
import wave
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from google.genai import types as genai_types
from loguru import logger
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame
from pipecat.services.google.tts import GeminiTTSSettings

from api import constants
from api.services.configuration.options.google_vertex_catalog import (
    check_vertex_config,
    gemini_tts_audio_format,
    gemini_tts_catalog_revision,
    gemini_tts_voices,
    get_vertex_model,
)
from api.services.pipecat.service_factory import DograhGeminiVertexApiTTSService


class SampleGenerationNotConfigured(RuntimeError):
    """The deployment has not supplied a platform-owned generation credential."""


class SampleGenerationProviderError(RuntimeError):
    """A provider failure with only safe, non-request diagnostic fields."""

    def __init__(
        self,
        *,
        status_code: int | None,
        category: str,
        detail: str,
        metadata: dict | None = None,
    ):
        self.status_code = status_code
        self.category = category
        self.detail = detail
        self.metadata = metadata or {}
        status = f"HTTP {status_code} " if status_code else ""
        super().__init__(f"{status}{category}: {detail}")


SAMPLE_PROVIDER_TIMEOUT_SECONDS = 120


_STATUS_RE = re.compile(r"\b([45]\d\d)\b")
_SENSITIVE_RE = re.compile(
    r"(?i)(?P<key>api[_ -]?key|authorization|bearer|access[_ -]?token|secret|"
    r"private[_ -]?key|contents?|prompt|sample[_ -]?text|text)\s*[:=]\s*[^,;}\]]+"
)


def safe_provider_diagnostic(
    error: BaseException,
    *,
    api_key: str | None = None,
    private_values: tuple[str, ...] = (),
) -> dict:
    """Extract bounded provider diagnostics without persisting request data."""
    status_code = None
    current: BaseException | None = error
    chain: list[BaseException] = []
    while current is not None and current not in chain:
        chain.append(current)
        for candidate in (
            getattr(current, "status_code", None),
            getattr(current, "code", None),
            getattr(getattr(current, "response", None), "status_code", None),
        ):
            if isinstance(candidate, int) and 400 <= candidate <= 599:
                status_code = candidate
                break
        current = current.__cause__ or current.__context__
    if status_code is None:
        match = _STATUS_RE.search(" ".join(str(item) for item in chain))
        status_code = int(match.group(1)) if match else None
    category = {
        400: "invalid_argument",
        401: "authentication",
        403: "permission",
        404: "not_found",
        408: "timeout",
        409: "conflict",
        429: "quota",
    }.get(
        status_code,
        "provider_unavailable"
        if status_code and status_code >= 500
        else "provider_error",
    )
    detail = str(error).replace(api_key, "[REDACTED]") if api_key else str(error)
    for value in private_values:
        if value:
            detail = detail.replace(value, "[REDACTED]")
    detail = _SENSITIVE_RE.sub(lambda match: f"{match.group('key')}=[REDACTED]", detail)
    detail = (
        re.sub(r"\s+", " ", detail).strip()[:240] or "provider returned no diagnostic"
    )
    return {"status_code": status_code, "category": category, "detail": detail}


@dataclass(frozen=True)
class PlatformSampleGenerationConfig:
    api_key: str
    project_id: str
    location: str
    credential_ref: str | None


def platform_sample_generation_config() -> PlatformSampleGenerationConfig:
    """Load platform-only generation settings without exposing the secret."""
    if not constants.GEMINI_TTS_SAMPLE_API_KEY:
        ref = constants.GEMINI_TTS_SAMPLE_CREDENTIAL_REF
        suffix = f" for credential reference {ref!r}" if ref else ""
        raise SampleGenerationNotConfigured(
            "Gemini-TTS sample generation is not configured: set the platform-"
            f"owned GEMINI_TTS_SAMPLE_API_KEY{suffix}."
        )
    if not constants.GEMINI_TTS_SAMPLE_PROJECT_ID:
        raise SampleGenerationNotConfigured(
            "Gemini-TTS sample generation is not configured: set "
            "GEMINI_TTS_SAMPLE_PROJECT_ID."
        )
    return PlatformSampleGenerationConfig(
        api_key=constants.GEMINI_TTS_SAMPLE_API_KEY,
        project_id=constants.GEMINI_TTS_SAMPLE_PROJECT_ID,
        location=constants.GEMINI_TTS_SAMPLE_LOCATION,
        credential_ref=constants.GEMINI_TTS_SAMPLE_CREDENTIAL_REF,
    )


def sample_pack_fingerprint(
    *,
    provider: str,
    model_id: str,
    catalog_revision: str,
    location: str,
    language: str,
    style_text: str,
    sample_text: str,
) -> str:
    payload = "\x1f".join(
        (
            provider,
            model_id,
            catalog_revision,
            location,
            language,
            style_text,
            sample_text,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pcm_to_wav(
    pcm: bytes, *, sample_rate: int = 24000, channels: int = 1
) -> tuple[bytes, float]:
    """Frame signed 16-bit PCM as a browser-playable WAV and return duration."""
    if not pcm or len(pcm) % 2:
        raise ValueError("Gemini-TTS returned empty or unaligned PCM")
    duration = len(pcm) / (sample_rate * channels * 2)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue(), duration


async def synthesize_sample_wav(
    *,
    api_key: str,
    project_id: str,
    location: str,
    model_id: str,
    voice_id: str,
    language: str,
    style_text: str,
    sample_text: str,
    context_id: str,
    sample_rate_hz: int,
    channels: int,
) -> tuple[bytes, float, dict]:
    """Use the same Dograh Vertex Gemini adapter used by live TTS calls."""
    started = time.monotonic()
    stages = {}
    outcomes = {}
    requests = 0

    def trace(stage, **fields):
        if stage in stages:
            return
        stages[stage] = round(time.monotonic() - started, 3)
        if fields:
            outcomes[stage] = fields
        logger.info(
            "Gemini sample stage={} context={} elapsed={} fields={}",
            stage,
            context_id,
            stages[stage],
            fields,
        )

    async def request_hook(request):
        nonlocal requests
        requests += 1
        if requests > 1:
            raise RuntimeError("Sample synthesis permits only one outbound request")
        trace("http_request", outbound_requests=requests)

        async def transport_trace(event, info):
            # httpcore includes raw headers in info: intentionally never log it.
            trace("transport_" + event)

        request.extensions["trace"] = transport_trace

    async def response_hook(response):
        trace(
            "response_headers",
            status=response.status_code,
            request_id=response.headers.get("x-request-id", "")[:100],
        )

    # Explicit httpx transport avoids the SDK's implicit aiohttp reconnect retry.
    transport = httpx.AsyncHTTPTransport(retries=0)
    service = DograhGeminiVertexApiTTSService(
        vertex_api_key=api_key,
        project_id=project_id,
        location=location,
        settings=GeminiTTSSettings(
            model=model_id,
            voice=voice_id,
            language=language,
            prompt=style_text or None,
        ),
        sample_rate=sample_rate_hz,
        http_options=genai_types.HttpOptions(
            timeout=SAMPLE_PROVIDER_TIMEOUT_SECONDS * 1000,
            retry_options=genai_types.HttpRetryOptions(attempts=1),
            async_client_args={
                "transport": transport,
                "event_hooks": {
                    "request": [request_hook],
                    "response": [response_hook],
                },
            },
        ),
    )
    service._sample_trace = trace
    trace("client_created")
    pcm_parts: list[bytes] = []
    stream = service._run_genai_tts(sample_text, context_id)
    try:
        async for frame in stream:
            if isinstance(frame, ErrorFrame):
                diagnostic = safe_provider_diagnostic(
                    frame.exception or RuntimeError(frame.error),
                    api_key=api_key,
                    private_values=(sample_text, style_text),
                )
                raise SampleGenerationProviderError(
                    **diagnostic,
                    metadata={
                        "failure_stage": next(reversed(stages)),
                        "exception_type": type(frame.exception).__name__,
                        "provider_timeout_seconds": SAMPLE_PROVIDER_TIMEOUT_SECONDS,
                        "outbound_requests": requests,
                        "stage_seconds": stages,
                        "stage_outcomes": outcomes,
                    },
                ) from frame.exception
            if not isinstance(frame, TTSAudioRawFrame):
                continue
            if frame.sample_rate != sample_rate_hz or frame.num_channels != channels:
                raise RuntimeError("Gemini-TTS returned an unexpected audio frame")
            if not frame.audio:
                raise RuntimeError("Gemini-TTS adapter yielded an empty audio frame")
            pcm_parts.append(frame.audio)
    except asyncio.CancelledError:
        trace("cancelled")
        raise
    finally:
        await stream.aclose()
        await service._client.aio.aclose()
        service._client.close()
        await transport.aclose()
    wav, duration = pcm_to_wav(
        b"".join(pcm_parts), sample_rate=sample_rate_hz, channels=channels
    )
    trace("pcm_validated", audio_bytes=sum(map(len, pcm_parts)))
    trace("wav_constructed")
    return (
        wav,
        duration,
        {
            "stage_seconds": stages,
            "stage_outcomes": outcomes,
            "outbound_requests": requests,
            "native_sample_rate_hz": sample_rate_hz,
            "channels": channels,
            "encoding": "signed_16bit_pcm_framed_as_wav",
            "adapter": "DograhGeminiVertexApiTTSService",
            "generated_at": datetime.now(UTC).isoformat(),
        },
    )


def validate_pack_request(
    *,
    model_id: str,
    location: str,
    catalog_revision: str | None,
    language: str,
    style_text: str,
    sample_text: str,
) -> tuple[str, tuple]:
    entry = get_vertex_model("tts", model_id)
    revision = gemini_tts_catalog_revision(model_id)
    if entry is None or revision is None:
        raise ValueError(f"unsupported Gemini-TTS model: {model_id}")
    if gemini_tts_audio_format(model_id) is None:
        raise ValueError(f"{model_id} has no supported audio framing metadata")
    if catalog_revision and catalog_revision != revision:
        raise ValueError(
            f"catalog revision {catalog_revision!r} is stale; current revision is {revision!r}"
        )
    if entry.locations and location not in entry.locations:
        raise ValueError(f"{model_id} requires one of: {', '.join(entry.locations)}")
    # Sample generation always authenticates with the platform API key, so it
    # follows the same rules as an API-key TTS slot (global endpoint only: the
    # key's account, not the request, resolves any other location). The project
    # is supplied by the platform configuration, checked separately.
    error = check_vertex_config(
        "tts",
        model=model_id,
        location=location,
        has_api_key=True,
        has_credentials=False,
        project_id=constants.GEMINI_TTS_SAMPLE_PROJECT_ID or "platform-configured",
        voice=None,
    )
    if error:
        raise ValueError(error)
    if not language.strip() or not style_text.strip() or not sample_text.strip():
        raise ValueError("language, style_text, and sample_text are required")
    return revision, gemini_tts_voices(model_id)
