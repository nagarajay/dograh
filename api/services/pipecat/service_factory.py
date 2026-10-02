import json
import time
from functools import wraps
from typing import TYPE_CHECKING
from urllib.parse import urlencode, urlparse, urlunparse

import aiohttp
from fastapi import HTTPException
from google.genai import Client as GenaiClient
from google.genai import types as genai_types
from loguru import logger

from api.constants import MPS_API_URL
from api.errors.failure import (
    ErrorSource,
    annotate_failure_metadata,
    classify_exception,
    log_failure,
)
from api.services.configuration.options import (
    DEEPGRAM_DEFAULT_BASE_URL,
    DEEPGRAM_FLUX_MODELS,
    GOOGLE_VERTEX_DEFAULT_LOCATION,
)
from api.services.configuration.registry import ServiceProviders
from api.services.configuration.safe_errors import redact_text
from api.services.pipecat.gemini_json_schema_adapter import (
    DograhGeminiJSONSchemaAdapter,
)
from api.services.pipecat.minimax_tts import MiniMaxOwnedSessionTTSService
from api.utils.url_security import validate_user_configured_service_url
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame
from pipecat.services.assemblyai.stt import AssemblyAISTTService, AssemblyAISTTSettings
from pipecat.services.aws.llm import AWSBedrockLLMService, AWSBedrockLLMSettings
from pipecat.services.azure.llm import AzureLLMService, AzureLLMSettings
from pipecat.services.azure.stt import AzureSTTService, AzureSTTSettings
from pipecat.services.azure.tts import AzureTTSService, AzureTTSSettings
from pipecat.services.cartesia.stt import CartesiaSTTService, CartesiaSTTSettings
from pipecat.services.cartesia.tts import (
    CartesiaTTSService,
    CartesiaTTSSettings,
    GenerationConfig,
)
from pipecat.services.cartesia.turns.stt import CartesiaTurnsSTTService
from pipecat.services.deepgram.flux.stt import (
    DeepgramFluxSTTService,
    DeepgramFluxSTTSettings,
)
from pipecat.services.deepgram.stt import DeepgramSTTService, DeepgramSTTSettings
from pipecat.services.deepgram.tts import DeepgramTTSService, DeepgramTTSSettings
from pipecat.services.dograh.flux.stt import DograhFluxSTTService
from pipecat.services.dograh.llm import DograhLLMService
from pipecat.services.dograh.stt import DograhSTTService, DograhSTTSettings
from pipecat.services.dograh.tts import DograhTTSService, DograhTTSSettings
from pipecat.services.elevenlabs.stt import (
    CommitStrategy,
    ElevenLabsRealtimeSTTService,
    ElevenLabsRealtimeSTTSettings,
)
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService, ElevenLabsTTSSettings
from pipecat.services.gladia.stt import GladiaSTTService, GladiaSTTSettings
from pipecat.services.google.gemini_live.stt import GeminiSTTService, GeminiSTTSettings
from pipecat.services.google.llm import GoogleLLMService, GoogleLLMSettings
from pipecat.services.google.stt import GoogleSTTService, GoogleSTTSettings
from pipecat.services.google.tts import (
    GeminiTTSService,
    GeminiTTSSettings,
    GoogleTTSService,
    GoogleTTSSettings,
)
from pipecat.services.google.vertex.llm import (
    GoogleVertexLLMService,
    GoogleVertexLLMSettings,
)
from pipecat.services.groq.llm import GroqLLMService, GroqLLMSettings
from pipecat.services.huggingface.llm import (
    HuggingFaceLLMService,
    HuggingFaceLLMSettings,
)
from pipecat.services.huggingface.stt import (
    HuggingFaceSTTService,
    HuggingFaceSTTSettings,
)
from pipecat.services.inworld.tts import InworldTTSService, InworldTTSSettings
from pipecat.services.minimax.llm import MiniMaxLLMService
from pipecat.services.minimax.tts import MiniMaxTTSSettings
from pipecat.services.openai._constants import OPENAI_SAMPLE_RATE
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import (
    OpenAISTTService,
    OpenAISTTSettings,
)
from pipecat.services.openai.tts import OpenAITTSService, OpenAITTSSettings
from pipecat.services.openrouter.llm import OpenRouterLLMService, OpenRouterLLMSettings
from pipecat.services.rime.tts import RimeTTSService, RimeTTSSettings
from pipecat.services.sarvam.llm import SarvamLLMService, SarvamLLMSettings
from pipecat.services.sarvam.stt import SarvamSTTService, SarvamSTTSettings
from pipecat.services.sarvam.tts import SarvamTTSService, SarvamTTSSettings
from pipecat.services.smallest.stt import SmallestSTTService, SmallestSTTSettings
from pipecat.services.smallest.tts import SmallestTTSService, SmallestTTSSettings
from pipecat.services.speaches.llm import SpeachesLLMService, SpeachesLLMSettings
from pipecat.services.speaches.stt import SpeachesSTTService, SpeachesSTTSettings
from pipecat.services.speaches.tts import SpeachesTTSService, SpeachesTTSSettings
from pipecat.services.speechmatics.stt import (
    SpeechmaticsSTTService,
    SpeechmaticsSTTSettings,
)
from pipecat.services.xai.tts import XAITTSService, XAIWebsocketTTSSettings
from pipecat.transcriptions.language import Language
from pipecat.utils.text.xml_function_tag_filter import XMLFunctionTagFilter
from pipecat.utils.types import assert_given, is_given

if TYPE_CHECKING:
    from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
    from api.services.pipecat.audio_config import AudioConfig


_GOOGLE_PROVIDERS = {
    ServiceProviders.GOOGLE.value,
    ServiceProviders.GOOGLE_VERTEX.value,
}


def _google_secret_values(config) -> list[str]:
    """Every string that must never reach a log line or an ErrorFrame."""
    values: list[str] = []
    for field in ("api_key", "credentials"):
        raw = getattr(config, field, None)
        for item in raw if isinstance(raw, list) else [raw]:
            if not isinstance(item, str) or not item:
                continue
            values.append(item)
            try:
                info = json.loads(item)
            except ValueError:
                continue
            if isinstance(info, dict):
                values.extend(
                    str(info[k])
                    for k in ("private_key", "private_key_id")
                    if info.get(k)
                )
                # Errors quote fragments, and JSON escapes the newlines, so the
                # whole value alone would miss a partial or re-encoded key.
                key = str(info.get("private_key") or "")
                values.append(json.dumps(key)[1:-1])
                values.extend(
                    line.strip()
                    for line in key.splitlines()
                    if len(line.strip()) >= 16 and not line.startswith("-----")
                )
    return values


def _install_error_redaction(service, secrets: list[str]) -> None:
    """Strip credential material from errors a Google service reports.

    Google SDK exceptions and the adapters' own ``f"...{e}"`` messages are
    forwarded to the pipeline as ErrorFrames and end up in run logs, so the
    redaction happens on the service rather than at each call site.
    """
    if not secrets:
        return
    push_error = service.push_error
    push_frame = service.push_frame

    def scrub(exc):
        if exc is not None and redact_text(str(exc), secrets) != str(exc):
            exc.args = (redact_text(str(exc), secrets),)
            if hasattr(exc, "message"):
                try:
                    exc.message = redact_text(str(exc.message), secrets)
                except AttributeError:
                    pass

    async def safe_push_error(error_msg, exception=None, *args, **kwargs):
        scrub(exception)
        return await push_error(
            redact_text(error_msg, secrets), exception, *args, **kwargs
        )

    async def safe_push_frame(frame, *args, **kwargs):
        if isinstance(frame, ErrorFrame):
            frame.error = redact_text(frame.error, secrets)
            scrub(frame.exception)
        return await push_frame(frame, *args, **kwargs)

    service.push_error = safe_push_error
    service.push_frame = safe_push_frame


def _report_service_factory_failures(
    source: ErrorSource,
    *,
    config_section: str | None = None,
    provider_argument: int | None = None,
):
    """Classify constructor failures and tag successful services for ErrorFrames."""

    def decorator(factory):
        @wraps(factory)
        def wrapped(*args, **kwargs):
            provider = None
            if config_section:
                user_config = args[0] if args else kwargs.get("user_config")
                config = getattr(user_config, config_section, None)
                provider = getattr(config, "provider", None)
            elif provider_argument is not None:
                if len(args) > provider_argument:
                    provider = args[provider_argument]
                else:
                    provider = kwargs.get("provider")

            provider_value = getattr(provider, "value", provider)
            error_owner = (
                "operator" if str(provider_value).lower() == "dograh" else "user"
            )
            try:
                service = factory(*args, **kwargs)
            except Exception as exc:
                log_failure(
                    classify_exception(
                        exc,
                        source=source,
                        provider=provider,
                        error_owner=error_owner,
                    )
                )
                raise

            if config_section and str(provider_value).lower() in _GOOGLE_PROVIDERS:
                _install_error_redaction(service, _google_secret_values(config))
            return annotate_failure_metadata(
                service,
                source=source,
                provider=provider,
                error_owner=error_owner,
            )

        return wrapped

    return decorator


DEEPGRAM_FLUX_LANGUAGE_HINTS = {
    "de": Language.DE,
    "en": Language.EN,
    "es": Language.ES,
    "fr": Language.FR,
    "hi": Language.HI,
    "it": Language.IT,
    "ja": Language.JA,
    "nl": Language.NL,
    "pt": Language.PT,
    "ru": Language.RU,
}


def _resolve_deepgram_flux_language_hint(language: str | None) -> Language | None:
    """Resolve a supported BCP-47 language or locale to its Flux base language."""
    base_language = (language or "").split("-", 1)[0].lower()
    return DEEPGRAM_FLUX_LANGUAGE_HINTS.get(base_language)


def dograh_stt_uses_flux_language(language: str | None) -> bool:
    if not language or language.lower() == "multi":
        return True
    return _resolve_deepgram_flux_language_hint(language) is not None


def _resolve_elevenlabs_stt_language(
    language_code: str | None,
) -> Language | str | None:
    if not language_code or language_code == "auto":
        return None
    try:
        return Language(language_code)
    except ValueError:
        return language_code


def _elevenlabs_websocket_url(base_url: str) -> str:
    """Normalize an ElevenLabs API base URL for WebSocket clients."""
    base_url = base_url.strip()
    parsed = urlparse(base_url)
    if not parsed.netloc:
        return base_url.rstrip("/")

    websocket_scheme = {
        "http": "ws",
        "https": "wss",
    }.get(parsed.scheme, parsed.scheme)
    return urlunparse(
        parsed._replace(
            scheme=websocket_scheme,
            path=parsed.path.rstrip("/"),
        )
    )


def _elevenlabs_realtime_stt_host(base_url: str) -> str:
    """Return the host/path prefix Pipecat's ElevenLabs realtime STT expects.

    Pipecat's realtime STT service builds
    ``wss://{host}/v1/speech-to-text/realtime`` internally, so remove the scheme
    from the same normalized WebSocket URL used by ElevenLabs TTS. Preserve
    netloc (including optional ports) and any path prefix used by BYOK proxies.
    """
    websocket_url = _elevenlabs_websocket_url(base_url)
    parsed = urlparse(websocket_url)
    if parsed.netloc:
        path = parsed.path
        return f"{parsed.netloc}{path}" if path else parsed.netloc
    return websocket_url


def stt_uses_external_turns(user_config) -> bool:
    if user_config.stt.provider == ServiceProviders.DEEPGRAM.value:
        return user_config.stt.model in DEEPGRAM_FLUX_MODELS
    if user_config.stt.provider == ServiceProviders.DOGRAH.value:
        return dograh_stt_uses_flux_language(getattr(user_config.stt, "language", None))
    if user_config.stt.provider == ServiceProviders.CARTESIA.value:
        return user_config.stt.model == "ink-2"
    return False


class DograhGoogleLLMService(GoogleLLMService):
    adapter_class = DograhGeminiJSONSchemaAdapter


class DograhGoogleVertexLLMService(GoogleVertexLLMService):
    """Vertex LLM that also accepts a Vertex API key (express mode).

    With an API key the SDK talks to the express-mode endpoint, which has no
    project or location, so neither is passed. Service-account / ADC behaviour
    is unchanged.
    """

    adapter_class = DograhGeminiJSONSchemaAdapter

    def __init__(self, *, vertex_api_key: str | None = None, **kwargs):
        # Read by _get_credentials/create_client, which the parent constructor
        # calls, so it has to be set first.
        self._vertex_api_key = vertex_api_key
        super().__init__(**kwargs)

    def _get_credentials(self, credentials, credentials_path):  # type: ignore[override]
        if self._vertex_api_key:
            return None
        return super()._get_credentials(credentials, credentials_path)

    def create_client(self):
        if not self._vertex_api_key:
            return super().create_client()
        self._client = GenaiClient(
            vertexai=True,
            api_key=self._vertex_api_key,
            http_options=self._http_options,
        )


def _vertex_client(
    *,
    api_key: str | None,
    credentials: str | None,
    project_id: str | None,
    location: str,
    http_options=None,
):
    """Build the google-genai Vertex client for the chosen authentication."""
    if api_key:
        return GenaiClient(vertexai=True, api_key=api_key, http_options=http_options)
    return GenaiClient(
        vertexai=True,
        credentials=GoogleVertexLLMService._get_credentials(credentials, None),
        project=project_id,
        location=location,
        http_options=http_options,
    )


def _vertex_model_resource(model: str, project_id: str | None, location: str) -> str:
    """Complete ``projects/{p}/locations/{l}/publishers/google/models/{m}`` name.

    The SDK builds only ``publishers/google/models/{m}`` for an API-key client,
    which the Live API rejects (websocket 1007) and which makes the Vertex API
    resolve the location from the key's account. Passing the full resource
    works with a key (live probes, 2026-10-01) and keeps the location explicit.
    """
    if not project_id or model.startswith("projects/"):
        return model
    return (
        f"projects/{project_id}/locations/{location}/publishers/google/models/{model}"
    )


class DograhGeminiVertexApiTTSService(GeminiTTSService):
    """Gemini-TTS over the Vertex API (``streamGenerateContent``) with an API key.

    The Cloud Text-to-Speech API takes OAuth credentials only, so a key has to
    use the Vertex API path instead. The parent's GenAI branch would build an
    AI Studio client; this subclass always builds a Vertex one and never reads
    ``GOOGLE_API_KEY``.
    """

    def __init__(
        self,
        *,
        vertex_api_key: str,
        project_id: str | None = None,
        location: str = "global",
        **kwargs,
    ):
        self._vertex_api_key = vertex_api_key
        self._vertex_project_id = project_id
        self._vertex_location = location
        # Vertex Gemini TTS returns signed 16-bit mono PCM at 24 kHz. Keep the
        # native rate on the frames so BaseOutputTransport resamples it to the
        # configured WebRTC rate instead of treating 24 kHz bytes as 16 kHz.
        kwargs.setdefault("sample_rate", self.GOOGLE_SAMPLE_RATE)
        super().__init__(use_genai=True, api_key=vertex_api_key, **kwargs)

    def _create_client(self, credentials, credentials_path):
        options = self._http_options
        if self._vertex_project_id:
            # Google's REST reference for the complete resource uses /v1/.
            options = (options or genai_types.HttpOptions()).model_copy(
                update={"api_version": "v1"}
            )
        return _vertex_client(
            api_key=self._vertex_api_key,
            credentials=None,
            project_id=None,
            location=None,
            http_options=options,
        )

    def _warn_unsupported_genai_settings(self, *, multi_speaker, prompt) -> None:
        # The Vertex API takes the style prompt in the request text.
        if multi_speaker:
            logger.warning(
                f"{self}: multi-speaker is not supported here; using one speaker."
            )

    async def _run_genai_tts(self, text: str, context_id: str):
        prompt = assert_given(self._settings.prompt)
        contents = f"{prompt}: {text}" if prompt else text
        # Sample-library generation invokes this method directly, before the
        # normal TTS setup frame initializes ``self.sample_rate``. Gemini-TTS
        # always returns native 24 kHz mono PCM, so use that rate for direct
        # calls as well as normal pipeline calls.
        output_sample_rate = self.sample_rate or self.GOOGLE_SAMPLE_RATE
        # Direct sample generation has no pipeline setup; self.chunk_size is
        # then zero. Always consume bytes using the resolved native rate.
        chunk_size = int(output_sample_rate * 0.5 * 2)
        stream = None
        trace = getattr(self, "_sample_trace", lambda *args, **kwargs: None)
        try:
            config = genai_types.GenerateContentConfig(
                # Gemini-TTS rejects requests that do not explicitly ask for
                # audio, even though speech_config is present.
                response_modalities=["AUDIO"],
                speech_config=genai_types.SpeechConfig(
                    language_code=assert_given(self._settings.language),
                    voice_config=genai_types.VoiceConfig(
                        prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                            voice_name=assert_given(self._settings.voice)
                        )
                    ),
                ),
            )
            await self.start_tts_usage_metrics(text)
            trace("request_dispatched")
            stream = await self._client.aio.models.generate_content_stream(
                model=_vertex_model_resource(
                    assert_given(self._settings.model),
                    self._vertex_project_id,
                    self._vertex_location,
                ),
                contents=contents,
                config=config,
            )
            buffer, first = b"", False
            async for chunk in stream:
                trace("response_chunk")
                feedback = getattr(chunk, "prompt_feedback", None)
                if feedback and feedback.block_reason:
                    raise RuntimeError(f"Google safety block: {feedback.block_reason.name}")
                for candidate in (chunk.candidates or [])[:1]:
                    finish = getattr(candidate, "finish_reason", None)
                    if finish and finish.name not in {"STOP", "FINISH_REASON_UNSPECIFIED"}:
                        raise RuntimeError(f"Google finish reason: {finish.name}")
                    for part in (
                        candidate.content.parts if candidate.content else None
                    ) or []:
                        data = part.inline_data.data if part.inline_data else None
                        if not data:
                            continue
                        if not first:
                            trace("first_audio", audio_bytes=len(data), mime_type=getattr(part.inline_data, "mime_type", None))
                            await self.stop_ttfb_metrics()
                            first = True
                        buffer += data
                        while len(buffer) >= chunk_size:
                            piece, buffer = (
                                buffer[:chunk_size],
                                buffer[chunk_size:],
                            )
                            yield TTSAudioRawFrame(
                                piece, output_sample_rate, 1, context_id=context_id
                            )
            if buffer:
                yield TTSAudioRawFrame(
                    buffer, output_sample_rate, 1, context_id=context_id
                )
            trace("stream_completed")
            if not first:
                raise RuntimeError("Google completed without audio")
        except Exception as e:
            detail = str(e)
            for private_value in (self._vertex_api_key, contents, text, prompt):
                if private_value:
                    detail = detail.replace(private_value, "[REDACTED]")
            yield ErrorFrame(error=f"Gemini Vertex TTS generation error: {detail}", exception=e)
        finally:
            # The google-genai stream owns an HTTP response body. Explicitly
            # close it when the provider stalls or the ARQ job is cancelled so
            # the next generation does not inherit a leaked connection.
            close_stream = getattr(stream, "aclose", None)
            if close_stream is not None:
                try:
                    await close_stream()
                except Exception:
                    logger.debug("Gemini Vertex TTS stream close failed", exc_info=True)


class DograhGeminiVertexSTTService(GeminiSTTService):
    """Gemini Live transcription on Vertex, with service-account/ADC auth."""

    def __init__(
        self,
        *,
        project_id: str | None,
        location: str,
        credentials: str | None,
        api_key: str | None = None,
        **kwargs,
    ):
        self._vertex_project_id = project_id
        self._vertex_location = location
        self._vertex_credentials = credentials
        self._vertex_api_key = api_key
        # The parent requires an api_key argument that this subclass never uses.
        super().__init__(api_key="unused-vertex", **kwargs)

    def _create_client(self):
        self._client = _vertex_client(
            api_key=self._vertex_api_key,
            credentials=self._vertex_credentials,
            project_id=self._vertex_project_id,
            location=self._vertex_location,
            http_options=self._http_options,
        )

    async def _open_session(self):
        """Open the Live session; with an API key, name the model completely."""
        if not (self._vertex_api_key and self._vertex_project_id):
            return await super()._open_session()
        model = _vertex_model_resource(
            assert_given(self._settings.model),
            self._vertex_project_id,
            self._vertex_location,
        )
        self._session_ctx = self._client.aio.live.connect(
            model=model, config=self._build_live_config()
        )
        self._session = await self._session_ctx.__aenter__()
        self._connection_start_time = time.time()
        await self._call_event_handler("on_connected")
        return self._session

    def _build_live_config(self):
        """Live config in the form Gemini 3.5 Transcribe documents.

        The parent sends ``language_hints`` / ``language_auto`` /
        ``adaptation_phrases``, which the SDK now marks deprecated in favour of
        top-level ``language_codes`` / ``custom_vocabulary``. Omitting
        ``language_codes`` already means automatic detection.
        """
        kwargs: dict = {}
        codes = self._get_language_codes()
        if codes:
            kwargs["language_codes"] = codes
        phrases = self._settings.adaptation_phrases
        if is_given(phrases) and phrases:
            kwargs["custom_vocabulary"] = list(phrases)
        return genai_types.LiveConnectConfig(
            response_modalities=[genai_types.Modality.TEXT],
            input_audio_transcription=genai_types.AudioTranscriptionConfig(**kwargs),
        )


def _validate_runtime_service_url(url: str, field_name: str) -> None:
    try:
        validate_user_configured_service_url(
            url,
            field_name=field_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _deepgram_base_url(service_config) -> str:
    """Resolve the Deepgram endpoint for an STT or TTS config section.

    Deepgram's regional hosts are the only thing that decides which
    jurisdiction processes the audio, so this is the single place the value is
    normalised and checked. Everything downstream builds on the result.
    """
    base_url = (getattr(service_config, "base_url", None) or "").strip()
    if not base_url:
        return DEEPGRAM_DEFAULT_BASE_URL
    # Deepgram documents the regional switch as "replace api.deepgram.com with
    # api.eu.deepgram.com", so operators reasonably type a bare host. The URL
    # validator - and the SaaS SSRF checks behind it - need a scheme, so assume
    # TLS rather than reject a value that is obviously well intentioned.
    if "://" not in base_url:
        base_url = f"https://{base_url}"
    _validate_runtime_service_url(base_url, "base_url")
    return base_url.rstrip("/")


def _deepgram_websocket_url(base_url: str, path: str = "") -> str:
    """Rewrite a Deepgram base URL as the WebSocket URL a service expects.

    Each Deepgram service in pipecat wants a different shape of the same host:
    ``DeepgramSTTService`` takes the base URL and derives both schemes itself,
    ``DeepgramFluxSTTService`` wants a full ``wss://host/v2/listen``, and
    ``DeepgramTTSService`` wants ``wss://host`` and appends its own path.
    Deriving all three from one configured value keeps a single source of truth
    for where audio goes. Expects the output of :func:`_deepgram_base_url`,
    which guarantees a scheme.
    """
    parsed = urlparse(base_url)
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
    return urlunparse(
        parsed._replace(scheme=scheme, path=parsed.path.rstrip("/") + path)
    )


def _google_vertex_location(location: str | None, service: str) -> str:
    """Resolve the Vertex location a service will connect to.

    Vertex derives its endpoint from the location, and only the regional and
    multi-region endpoints keep processing inside a geography - global routes
    anywhere. Falling back to anything other than the configured default would
    mean an operator who left the field alone gets a region nobody chose and
    cannot see, which is the wrong answer for anyone with residency
    obligations.
    """
    resolved = (location or "").strip()
    if resolved:
        # The resolved value is reported by the caller, on the same line as the
        # service being created - one decision, one line.
        return resolved

    # Unset is the one case where nobody picked the endpoint, and the default
    # carries no residency commitment, so it gets a line of its own rather than
    # riding along with the successful cases.
    logger.warning(
        f"Google Vertex {service}: no location configured, falling back to "
        f"{GOOGLE_VERTEX_DEFAULT_LOCATION!r}, which carries no data residency "
        f"guarantee"
    )
    return GOOGLE_VERTEX_DEFAULT_LOCATION


@_report_service_factory_failures(ErrorSource.STT, config_section="stt")
def create_stt_service(
    user_config,
    audio_config: "AudioConfig",
    keyterms: list[str] | None = None,
    correlation_id: str | None = None,
):
    """Create and return appropriate STT service based on user configuration

    Args:
        user_config: User configuration containing STT settings
        keyterms: Optional list of keyterms for speech recognition boosting (Deepgram only)
    """
    # Resolved before the line below so that where the audio is processed is
    # recorded alongside what processes it. The endpoint decides which
    # jurisdiction transcribes the call, and that is the only record an operator
    # has that the configured region was the one actually used - but it is one
    # fact about one service, so it belongs on one line.
    is_deepgram = user_config.stt.provider == ServiceProviders.DEEPGRAM.value
    deepgram_base_url = _deepgram_base_url(user_config.stt) if is_deepgram else None

    logger.info(
        f"Creating STT service: provider={user_config.stt.provider}, "
        f"model={user_config.stt.model}"
        + (f", endpoint={deepgram_base_url}" if deepgram_base_url else "")
    )

    if is_deepgram:
        if user_config.stt.model in DEEPGRAM_FLUX_MODELS:
            settings_kwargs = {
                "model": user_config.stt.model,
                "eot_timeout_ms": 3000,
                "eot_threshold": 0.7,
                "eager_eot_threshold": 0.5,
                "keyterm": keyterms or [],
            }
            if user_config.stt.model == "flux-general-multi":
                language = getattr(user_config.stt, "language", None)
                language_hint = _resolve_deepgram_flux_language_hint(language)
                if language_hint:
                    settings_kwargs["language_hints"] = [language_hint]

            return DeepgramFluxSTTService(
                api_key=user_config.stt.api_key,
                # Flux takes a fully-qualified socket URL, not a host.
                url=_deepgram_websocket_url(deepgram_base_url, "/v2/listen"),
                settings=DeepgramFluxSTTSettings(**settings_kwargs),
                should_interrupt=False,  # Let UserAggregator take care of sending InterruptionFrame
                sample_rate=audio_config.transport_in_sample_rate,
            )

        # Other models than flux
        # Use language from user config, defaulting to "multi" for multilingual support
        language = getattr(user_config.stt, "language", None) or "multi"
        return DeepgramSTTService(
            api_key=user_config.stt.api_key,
            # Takes the host and derives the wss and https URLs itself.
            base_url=deepgram_base_url,
            settings=DeepgramSTTSettings(
                language=language,
                profanity_filter=False,
                endpointing=100,
                model=user_config.stt.model,
                keyterm=keyterms or [],
            ),
            should_interrupt=False,  # Let UserAggregator take care of sending InterruptionFrame
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.OPENAI.value:
        kwargs = {}
        base_url = getattr(user_config.stt, "base_url", None)
        if base_url:
            _validate_runtime_service_url(base_url, "base_url")
            kwargs["base_url"] = base_url
        return OpenAISTTService(
            api_key=user_config.stt.api_key,
            settings=OpenAISTTSettings(model=user_config.stt.model),
            should_interrupt=False,  # Let UserAggregator own interruption confirmation.
            **kwargs,
        )
    elif user_config.stt.provider == ServiceProviders.GOOGLE.value:
        language = getattr(user_config.stt, "language", None) or "en-US"
        location = getattr(user_config.stt, "location", None) or "global"
        credentials = getattr(user_config.stt, "credentials", None)

        settings_kwargs = {"model": user_config.stt.model}
        try:
            settings_kwargs["languages"] = [Language(language)]
        except ValueError:
            settings_kwargs["language_codes"] = [language]

        return GoogleSTTService(
            credentials=credentials,
            location=location,
            settings=GoogleSTTSettings(**settings_kwargs),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.GOOGLE_VERTEX.value:
        stt = user_config.stt
        language = (getattr(stt, "language", None) or "").strip()
        settings_kwargs = {"model": stt.model}
        if language:
            try:
                settings_kwargs["languages"] = [Language(language)]
            except ValueError:
                settings_kwargs["language"] = language
        api_key = stt.api_key if isinstance(stt.api_key, str) else None
        logger.info(
            f"Vertex STT: model={stt.model}, location={stt.location}, "
            f"auth={'api_key' if api_key else 'service_account' if stt.credentials else 'adc'}"
        )
        return DograhGeminiVertexSTTService(
            project_id=stt.project_id,
            location=_google_vertex_location(stt.location, "STT"),
            credentials=stt.credentials,
            api_key=api_key,
            settings=GeminiSTTSettings(**settings_kwargs),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.CARTESIA.value:
        if user_config.stt.model == "ink-2":
            return CartesiaTurnsSTTService(
                api_key=user_config.stt.api_key,
                should_interrupt=False,  # Let UserAggregator emit interruption frames.
                sample_rate=audio_config.transport_in_sample_rate,
            )

        language = getattr(user_config.stt, "language", None) or "en"
        return CartesiaSTTService(
            api_key=user_config.stt.api_key,
            settings=CartesiaSTTSettings(
                model=user_config.stt.model,
                language=language,
            ),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.DOGRAH.value:
        base_url = MPS_API_URL.replace("http://", "ws://").replace("https://", "wss://")
        language = getattr(user_config.stt, "language", None) or "multi"

        if dograh_stt_uses_flux_language(language):
            # Dograh's Flux proxy only supports multilingual auto-detect and the
            # same language hint subset as Deepgram Flux multilingual.
            settings_kwargs = {
                "model": "flux-general-multi",
                "eot_timeout_ms": 3000,
                "eot_threshold": 0.7,
                "eager_eot_threshold": 0.5,
                "keyterm": keyterms or [],
            }
            language_hint = _resolve_deepgram_flux_language_hint(language)
            if language_hint:
                settings_kwargs["language_hints"] = [language_hint]
            return DograhFluxSTTService(
                base_url=base_url,
                api_key=user_config.stt.api_key,
                correlation_id=correlation_id,
                settings=DeepgramFluxSTTSettings(**settings_kwargs),
                should_interrupt=False,  # external turn strategies own interruption
                sample_rate=audio_config.transport_in_sample_rate,
            )

        return DograhSTTService(
            base_url=base_url,
            api_key=user_config.stt.api_key,
            correlation_id=correlation_id,
            settings=DograhSTTSettings(
                model=user_config.stt.model,
                language=language,
            ),
            keyterms=keyterms,
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.SARVAM.value:
        language = getattr(user_config.stt, "language", None)
        language_mapping = {
            "bn-IN": Language.BN_IN,
            "gu-IN": Language.GU_IN,
            "hi-IN": Language.HI_IN,
            "kn-IN": Language.KN_IN,
            "ml-IN": Language.ML_IN,
            "mr-IN": Language.MR_IN,
            "ta-IN": Language.TA_IN,
            "te-IN": Language.TE_IN,
            "pa-IN": Language.PA_IN,
            "od-IN": Language.OR_IN,
            "en-IN": Language.EN_IN,
            "as-IN": Language.AS_IN,
            "ur-IN": Language.UR_IN,
            "kok-IN": Language.KOK_IN,
            "mai-IN": Language.MAI_IN,
            "sd-IN": Language.SD_IN,
        }
        if not language or language == "unknown":
            pipecat_language = None
        elif language in language_mapping:
            pipecat_language = language_mapping[language]
        else:
            # Unmapped BCP-47 codes pass through; Sarvam accepts them per https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe
            pipecat_language = language
        return SarvamSTTService(
            api_key=user_config.stt.api_key,
            settings=SarvamSTTSettings(
                model=user_config.stt.model,
                language=pipecat_language,
            ),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.SPEACHES.value:
        language = getattr(user_config.stt, "language", None)
        _validate_runtime_service_url(user_config.stt.base_url, "base_url")
        return SpeachesSTTService(
            base_url=user_config.stt.base_url,
            api_key=user_config.stt.api_key or "none",
            settings=SpeachesSTTSettings(
                model=user_config.stt.model,
                language=language,
            ),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.HUGGINGFACE.value:
        base_url = (
            getattr(user_config.stt, "base_url", None)
            or "https://router.huggingface.co/hf-inference"
        )
        _validate_runtime_service_url(base_url, "base_url")
        return HuggingFaceSTTService(
            api_key=user_config.stt.api_key,
            base_url=base_url,
            bill_to=getattr(user_config.stt, "bill_to", None),
            settings=HuggingFaceSTTSettings(
                model=user_config.stt.model,
                return_timestamps=getattr(user_config.stt, "return_timestamps", False),
            ),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.ASSEMBLYAI.value:
        language = getattr(user_config.stt, "language", None)
        settings_kwargs = {"model": user_config.stt.model, "language": language}
        if keyterms:
            settings_kwargs["keyterms_prompt"] = keyterms
        return AssemblyAISTTService(
            api_key=user_config.stt.api_key,
            settings=AssemblyAISTTSettings(**settings_kwargs),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.GLADIA.value:
        from pipecat.services.gladia.config import LanguageConfig

        language = getattr(user_config.stt, "language", None) or "en"
        settings_kwargs = {
            "model": user_config.stt.model,
            "language_config": LanguageConfig(
                languages=[language], code_switching=False
            ),
        }
        return GladiaSTTService(
            api_key=user_config.stt.api_key,
            settings=GladiaSTTSettings(**settings_kwargs),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.SPEECHMATICS.value:
        from pipecat.services.speechmatics.stt import (
            AdditionalVocabEntry,
            Model,
            TurnDetectionMode,
        )

        language = getattr(user_config.stt, "language", None) or "en"
        # Saved configurations may still use the legacy operating-point names.
        model = user_config.stt.model
        if model in ("standard", "enhanced"):
            model = Model.LINDEN_1.value
        # Convert keyterms to AdditionalVocabEntry objects for Speechmatics
        additional_vocab = []
        if keyterms:
            additional_vocab = [AdditionalVocabEntry(content=term) for term in keyterms]
        return SpeechmaticsSTTService(
            api_key=user_config.stt.api_key,
            settings=SpeechmaticsSTTSettings(
                language=language,
                model=model,
                turn_detection_mode=TurnDetectionMode.EXTERNAL,
                additional_vocab=additional_vocab,
            ),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.AZURE_SPEECH.value:
        from pipecat.transcriptions.language import Language as PipecatLanguage

        language_code = getattr(user_config.stt, "language", None) or "en-US"
        region = getattr(user_config.stt, "region", None) or "eastus"
        try:
            pipecat_language = PipecatLanguage(language_code)
        except ValueError:
            pipecat_language = language_code
        return AzureSTTService(
            api_key=user_config.stt.api_key,
            region=region,
            settings=AzureSTTSettings(language=pipecat_language),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.SMALLEST.value:
        language_code = getattr(user_config.stt, "language", None) or "en"
        try:
            pipecat_language = Language(language_code)
        except ValueError:
            pipecat_language = Language.EN
        return SmallestSTTService(
            api_key=user_config.stt.api_key,
            settings=SmallestSTTSettings(
                model=user_config.stt.model,
                language=pipecat_language,
            ),
            sample_rate=audio_config.transport_in_sample_rate,
        )
    elif user_config.stt.provider == ServiceProviders.ELEVENLABS.value:
        language_code = getattr(user_config.stt, "language", None)
        pipecat_language = _resolve_elevenlabs_stt_language(language_code)

        _validate_runtime_service_url(user_config.stt.base_url, "base_url")
        elevenlabs_host = _elevenlabs_realtime_stt_host(user_config.stt.base_url)

        return ElevenLabsRealtimeSTTService(
            api_key=user_config.stt.api_key,
            base_url=elevenlabs_host,
            commit_strategy=CommitStrategy.VAD,
            settings=ElevenLabsRealtimeSTTSettings(
                model=user_config.stt.model,
                language=pipecat_language,
            ),
            should_interrupt=False,
            sample_rate=audio_config.transport_in_sample_rate,
        )
    else:
        raise HTTPException(
            status_code=400, detail=f"Invalid STT provider {user_config.stt.provider}"
        )


@_report_service_factory_failures(ErrorSource.TTS, config_section="tts")
def create_tts_service(
    user_config, audio_config: "AudioConfig", correlation_id: str | None = None
):
    """Create and return appropriate TTS service based on user configuration

    Args:
        user_config: User configuration containing TTS settings
        transport_type: Type of transport (e.g., 'twilio', 'webrtc')
    """
    # Synthesis carries the same residency question as transcription - the text
    # sent for speaking is drawn from the conversation - so the endpoint is
    # resolved up front and reported on the creation line.
    is_deepgram = user_config.tts.provider == ServiceProviders.DEEPGRAM.value
    deepgram_base_url = _deepgram_base_url(user_config.tts) if is_deepgram else None

    logger.info(
        f"Creating TTS service: provider={user_config.tts.provider}, "
        f"model={user_config.tts.model}"
        + (f", endpoint={deepgram_base_url}" if deepgram_base_url else "")
    )

    # Create function call filter to prevent TTS from speaking function call tags
    xml_function_tag_filter = XMLFunctionTagFilter()
    if is_deepgram:
        return DeepgramTTSService(
            api_key=user_config.tts.api_key,
            # Wants wss://host with no path; it appends /v1/speak itself.
            base_url=_deepgram_websocket_url(deepgram_base_url),
            settings=DeepgramTTSSettings(voice=user_config.tts.voice),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=1.0,
        )
    elif user_config.tts.provider == ServiceProviders.OPENAI.value:
        kwargs = {}
        base_url = getattr(user_config.tts, "base_url", None)
        if base_url:
            _validate_runtime_service_url(base_url, "base_url")
            kwargs["base_url"] = base_url
        return OpenAITTSService(
            api_key=user_config.tts.api_key,
            sample_rate=OPENAI_SAMPLE_RATE,
            settings=OpenAITTSSettings(model=user_config.tts.model),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=1.0,
            **kwargs,
        )
    elif user_config.tts.provider == ServiceProviders.GOOGLE.value:
        model = getattr(user_config.tts, "model", None) or "chirp_3_hd"
        language = getattr(user_config.tts, "language", None) or "en-US"
        voice = getattr(user_config.tts, "voice", None) or "en-US-Chirp3-HD-Charon"
        speed = getattr(user_config.tts, "speed", None)
        # The default Text-to-Speech endpoint is the global one; only regional
        # and multi-region locations ("us", "eu", ...) are prefixed onto the host.
        location = getattr(user_config.tts, "location", None) or None
        if location and location.strip().lower() == "global":
            location = None
        credentials = getattr(user_config.tts, "credentials", None)

        settings_kwargs = {
            "model": model,
            "voice": voice,
            "language": language,
        }
        if speed is not None and speed != 1.0:
            settings_kwargs["speaking_rate"] = speed

        return GoogleTTSService(
            credentials=credentials,
            location=location,
            settings=GoogleTTSSettings(**settings_kwargs),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=1.0,
        )
    elif user_config.tts.provider == ServiceProviders.GOOGLE_VERTEX.value:
        tts = user_config.tts
        # Gemini-TTS runs on Cloud Text-to-Speech: "global" is the unprefixed host.
        location = (tts.location or "").strip()
        if not location or location.lower() == "global":
            location = None
        logger.info(
            f"Vertex TTS: model={tts.model}, voice={tts.voice}, "
            f"location={location or 'global'}, "
            f"auth={'service_account' if tts.credentials else 'adc'}"
        )
        if isinstance(tts.api_key, str) and tts.api_key:
            logger.info(
                f"Vertex TTS: model={tts.model}, voice={tts.voice}, "
                "api=vertex_stream_generate_content, auth=api_key"
            )
            return DograhGeminiVertexApiTTSService(
                vertex_api_key=tts.api_key,
                project_id=getattr(tts, "project_id", None),
                location=(tts.location or "global").strip() or "global",
                settings=GeminiTTSSettings(
                    model=tts.model,
                    voice=tts.voice,
                    language=tts.language,
                    prompt=tts.prompt or None,
                ),
                text_filters=[xml_function_tag_filter],
                skip_aggregator_types=["recording_router", "recording"],
                silence_time_s=1.0,
            )
        return GeminiTTSService(
            credentials=tts.credentials,
            location=location,
            use_genai=False,
            settings=GeminiTTSSettings(
                model=tts.model,
                voice=tts.voice,
                language=tts.language,
                prompt=tts.prompt or None,
            ),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=1.0,
        )
    elif user_config.tts.provider == ServiceProviders.ELEVENLABS.value:
        # Backward compatible with older configuration "Name - voice_id"
        try:
            voice_id = user_config.tts.voice.split(" - ")[1]
        except IndexError:
            voice_id = user_config.tts.voice
        # ElevenLabs TTS consumes the full normalized WebSocket URL. Realtime
        # STT uses the same normalization before adapting it to Pipecat's
        # scheme-less base_url contract.
        _validate_runtime_service_url(user_config.tts.base_url, "base_url")
        elevenlabs_url = _elevenlabs_websocket_url(user_config.tts.base_url)
        return ElevenLabsTTSService(
            reconnect_on_error=False,
            api_key=user_config.tts.api_key,
            url=elevenlabs_url,
            settings=ElevenLabsTTSSettings(
                voice=voice_id,
                model=user_config.tts.model,
                stability=0.8,
                speed=user_config.tts.speed,
                similarity_boost=0.75,
            ),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=1.0,
        )
    elif user_config.tts.provider == ServiceProviders.CARTESIA.value:
        speed = getattr(user_config.tts, "speed", None)
        volume = getattr(user_config.tts, "volume", None)
        gen_config_kwargs = {}
        if speed and speed != 1.0:
            gen_config_kwargs["speed"] = speed
        if volume and volume != 1.0:
            gen_config_kwargs["volume"] = volume
        generation_config = (
            GenerationConfig(**gen_config_kwargs) if gen_config_kwargs else None
        )
        language = getattr(user_config.tts, "language", None) or "en"
        return CartesiaTTSService(
            api_key=user_config.tts.api_key,
            settings=CartesiaTTSSettings(
                voice=user_config.tts.voice,
                model=user_config.tts.model,
                language=language,
                **(
                    {"generation_config": generation_config}
                    if generation_config
                    else {}
                ),
            ),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=1.0,
        )
    elif user_config.tts.provider == ServiceProviders.INWORLD.value:
        voice = getattr(user_config.tts, "voice", None) or "Ashley"
        model = getattr(user_config.tts, "model", None) or "inworld-tts-2"
        speed = getattr(user_config.tts, "speed", None)
        language = getattr(user_config.tts, "language", None) or "en-US"
        delivery_mode = getattr(user_config.tts, "delivery_mode", None) or "BALANCED"
        return InworldTTSService(
            api_key=user_config.tts.api_key,
            settings=InworldTTSSettings(
                voice=voice,
                model=model,
                language=language,
                speaking_rate=speed,
                delivery_mode=delivery_mode,
            ),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=1.0,
        )
    elif user_config.tts.provider == ServiceProviders.DOGRAH.value:
        # Convert HTTP URL to WebSocket URL for TTS
        base_url = MPS_API_URL.replace("http://", "ws://").replace("https://", "wss://")
        return DograhTTSService(
            base_url=base_url,
            api_key=user_config.tts.api_key,
            correlation_id=correlation_id,
            settings=DograhTTSSettings(
                model=user_config.tts.model,
                voice=user_config.tts.voice,
                speed=user_config.tts.speed,
            ),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=1.0,
        )
    elif user_config.tts.provider == ServiceProviders.CAMB.value:
        from pipecat.services.camb.tts import CambTTSService

        voice_id = int(getattr(user_config.tts, "voice", None) or "147320")
        language = getattr(user_config.tts, "language", None) or "en-us"
        tts = CambTTSService(
            api_key=user_config.tts.api_key,
            voice_id=voice_id,
            model=user_config.tts.model,
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
        )
        # Set language directly as BCP-47 code (bypasses Language enum conversion)
        tts._settings.language = language
        return tts
    elif user_config.tts.provider == ServiceProviders.SPEACHES.value:
        _validate_runtime_service_url(user_config.tts.base_url, "base_url")
        return SpeachesTTSService(
            base_url=user_config.tts.base_url,
            api_key=user_config.tts.api_key or "none",
            settings=SpeachesTTSSettings(
                model=user_config.tts.model,
                voice=user_config.tts.voice,
                speed=user_config.tts.speed,
            ),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=1.0,
        )
    elif user_config.tts.provider == ServiceProviders.RIME.value:
        speed = getattr(user_config.tts, "speed", None)
        language_code = getattr(user_config.tts, "language", None) or "en"
        rime_language_mapping = {
            "en": Language.EN,
            "de": Language.DE,
            "fr": Language.FR,
            "es": Language.ES,
            "hi": Language.HI,
        }
        pipecat_language = rime_language_mapping.get(language_code, Language.EN)
        settings_kwargs = {
            "voice": user_config.tts.voice,
            "model": user_config.tts.model,
            "language": pipecat_language,
        }
        if speed and speed != 1.0:
            settings_kwargs["speedAlpha"] = speed
        return RimeTTSService(
            api_key=user_config.tts.api_key,
            settings=RimeTTSSettings(**settings_kwargs),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=1.0,
        )
    elif user_config.tts.provider == ServiceProviders.SARVAM.value:
        # Map Sarvam language code to pipecat Language enum for TTS
        language_mapping = {
            "bn-IN": Language.BN,
            "en-IN": Language.EN,
            "gu-IN": Language.GU,
            "hi-IN": Language.HI,
            "kn-IN": Language.KN,
            "ml-IN": Language.ML,
            "mr-IN": Language.MR,
            "od-IN": Language.OR,
            "pa-IN": Language.PA,
            "ta-IN": Language.TA,
            "te-IN": Language.TE,
        }
        language = getattr(user_config.tts, "language", None)
        pipecat_language = language_mapping.get(language, Language.HI)

        voice = (
            getattr(user_config.tts, "voice", None) or ""
        ).strip().lower() or "anushka"
        speed = getattr(user_config.tts, "speed", None)
        settings_kwargs = {
            "model": user_config.tts.model,
            "voice": voice,
            "language": pipecat_language,
        }
        if speed and speed != 1.0:
            settings_kwargs["pace"] = speed
        return SarvamTTSService(
            api_key=user_config.tts.api_key,
            settings=SarvamTTSSettings(**settings_kwargs),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=1.0,
        )
    elif user_config.tts.provider == ServiceProviders.MINIMAX.value:
        group_id = getattr(user_config.tts, "group_id", None)
        if not group_id:
            raise HTTPException(
                status_code=400,
                detail="MiniMax TTS requires a group_id. Configure it in your TTS settings.",
            )
        voice = getattr(user_config.tts, "voice", None) or "English_Graceful_Lady"
        speed = getattr(user_config.tts, "speed", None) or 1.0

        # Pipecat appends "?GroupId=..." to base_url as-is, so /t2a_v2 must
        # already be in the path.
        base_url = (
            getattr(user_config.tts, "base_url", None)
            or "https://api.minimax.io/v1/t2a_v2"
        ).rstrip("/")
        if not base_url.endswith("/t2a_v2"):
            base_url = f"{base_url}/t2a_v2"
        _validate_runtime_service_url(base_url, "base_url")

        session = aiohttp.ClientSession()
        return MiniMaxOwnedSessionTTSService(
            api_key=user_config.tts.api_key,
            group_id=group_id,
            base_url=base_url,
            aiohttp_session=session,
            settings=MiniMaxTTSSettings(
                model=user_config.tts.model,
                voice=voice,
                speed=speed,
            ),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=1.0,
        )
    elif user_config.tts.provider == ServiceProviders.AZURE_SPEECH.value:
        region = getattr(user_config.tts, "region", None) or "eastus"
        voice = getattr(user_config.tts, "voice", None) or "en-US-AriaNeural"
        language = getattr(user_config.tts, "language", None) or "en-US"
        speed = getattr(user_config.tts, "speed", None) or 1.0
        # Map speed multiplier (0.5–2.0) to Azure SSML rate string (e.g. "1.25")
        rate = str(speed) if speed != 1.0 else None
        settings_kwargs: dict = {
            "voice": voice,
            "language": language,
        }
        if rate:
            settings_kwargs["rate"] = rate
        return AzureTTSService(
            api_key=user_config.tts.api_key,
            region=region,
            settings=AzureTTSSettings(**settings_kwargs),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=1.0,
        )
    elif user_config.tts.provider == ServiceProviders.SMALLEST.value:
        language_code = getattr(user_config.tts, "language", None) or "en"
        try:
            pipecat_language = Language(language_code)
        except ValueError:
            pipecat_language = Language.EN
        speed = getattr(user_config.tts, "speed", None)
        model = user_config.tts.model.replace("lightning-v", "lightning_v")
        settings_kwargs = SmallestTTSSettings(
            model=model,
            voice=user_config.tts.voice,
            language=pipecat_language,
        )
        if speed and speed != 1.0:
            settings_kwargs.speed = speed
        return SmallestTTSService(
            api_key=user_config.tts.api_key,
            settings=settings_kwargs,
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=1.0,
        )
    elif user_config.tts.provider == ServiceProviders.XAI.value:
        voice = getattr(user_config.tts, "voice", None) or "eve"
        language_code = getattr(user_config.tts, "language", None) or "en"
        if language_code.lower() == "auto":
            pipecat_language = "auto"
        else:
            try:
                pipecat_language = Language(language_code)
            except ValueError:
                pipecat_language = Language.EN
        return XAITTSService(
            api_key=user_config.tts.api_key,
            settings=XAIWebsocketTTSSettings(
                voice=voice,
                language=pipecat_language,
            ),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=1.0,
        )
    elif user_config.tts.provider == ServiceProviders.LMNT.value:
        raise ValueError(
            "LMNT is no longer available. Please select another TTS provider."
        )
    elif user_config.tts.provider == ServiceProviders.SPEECHIFY.value:
        # SpeechifyHttpTTSService ships in upstream pipecat; imported lazily so
        # this module keeps loading on pipecat checkouts that predate it.
        try:
            from api.services.pipecat.speechify_tts import (
                SpeechifyOwnedSessionTTSService,
            )
            from pipecat.services.speechify.tts import SpeechifyTTSSettings
        except ModuleNotFoundError as e:
            missing = e.name or ""
            if missing != "pipecat.services.speechify" and not missing.startswith(
                "pipecat.services.speechify."
            ):
                raise
            raise HTTPException(
                status_code=400,
                detail=(
                    "Speechify TTS requires a pipecat build that includes "
                    "pipecat.services.speechify; the installed pipecat does not."
                ),
            ) from e

        voice = getattr(user_config.tts, "voice", None) or "beatrice_32"
        model = getattr(user_config.tts, "model", None) or "simba-3.2"
        language_code = getattr(user_config.tts, "language", None) or "en"
        language: Language | str
        try:
            language = Language(language_code)
        except ValueError:
            # The config allows custom language codes; codes the pipecat enum
            # doesn't model (e.g. "en-ZA") are sent to Speechify verbatim
            # rather than silently replaced with English.
            language = language_code
        session = aiohttp.ClientSession()
        return SpeechifyOwnedSessionTTSService(
            api_key=user_config.tts.api_key,
            aiohttp_session=session,
            sample_rate=audio_config.transport_out_sample_rate,
            settings=SpeechifyTTSSettings(
                voice=voice,
                model=model,
                language=language,
            ),
            text_filters=[xml_function_tag_filter],
            skip_aggregator_types=["recording_router", "recording"],
            silence_time_s=1.0,
        )
    else:
        raise HTTPException(
            status_code=400, detail=f"Invalid TTS provider {user_config.tts.provider}"
        )


# Groq exposes a `reasoning_format` request parameter only for its reasoning
# models. Sending it to a non-reasoning model is rejected, so the switch is
# keyed off the model family rather than applied to every Groq model.
_GROQ_REASONING_MODEL_MARKERS = ("gpt-oss", "deepseek-r1", "qwen3")


# Models observed to reject every reasoning_effort but "none" on Chat
# Completions when the request carries function tools (which every workflow
# turn does): "Function tools with reasoning_effort are not supported for
# gpt-5.6-luna in /v1/chat/completions ... set reasoning_effort to 'none'".
# Only models confirmed this way belong here; other GPT-5 models keep "minimal".
_OPENAI_TOOLS_REQUIRE_NO_REASONING = frozenset({"gpt-5.6-luna"})


def _openai_gpt5_reasoning_effort(model: str) -> str:
    return "none" if model in _OPENAI_TOOLS_REQUIRE_NO_REASONING else "minimal"


def _is_groq_reasoning_model(model: str) -> bool:
    """Whether a Groq model emits reasoning that has to be kept out of content."""
    lowered = (model or "").lower()
    return any(marker in lowered for marker in _GROQ_REASONING_MODEL_MARKERS)


def _migrate_deprecated_google_model(model: str) -> str:
    """Google removed the ``gemini-2.0-flash*`` models. Transparently upgrade
    any stored config that still references them to the 2.5 equivalent so old
    user configurations keep working instead of failing at runtime."""
    if model and model.startswith("gemini-2.0-flash"):
        migrated = model.replace("gemini-2.0-", "gemini-2.5-", 1)
        logger.warning(
            f"Google model '{model}' is no longer supported; using '{migrated}' instead"
        )
        return migrated
    return model


@_report_service_factory_failures(ErrorSource.LLM, provider_argument=0)
def create_llm_service_from_provider(
    provider: str,
    model: str,
    api_key: str | None,
    *,
    correlation_id: str | None = None,
    base_url: str | None = None,
    endpoint: str | None = None,
    aws_access_key: str | None = None,
    aws_secret_key: str | None = None,
    aws_region: str | None = None,
    project_id: str | None = None,
    location: str | None = None,
    credentials: str | None = None,
    temperature: float | None = None,
    bill_to: str | None = None,
    usage_context: str | None = None,
):
    """Create an LLM service from explicit provider/model/api_key.

    Also used by create_llm_service which extracts these from user_config.

    Args:
        usage_context: Optional tag describing what the LLM instance is used for
            (e.g. "voicemail_detection"). Sent as request metadata by the Dograh
            provider; ignored by other providers.
    """
    # Vertex builds its endpoint from the location, so it is part of what this
    # service is, not a separate event. Resolved here to keep it on one line.
    vertex_location = (
        _google_vertex_location(location, "LLM")
        if provider == ServiceProviders.GOOGLE_VERTEX.value
        else None
    )

    logger.info(
        f"Creating LLM service: provider={provider}, model={model}"
        + (f", location={vertex_location}" if vertex_location else "")
    )

    if provider in (
        ServiceProviders.OPENAI.value,
        ServiceProviders.ATLASCLOUD.value,
    ):
        kwargs = {}
        if base_url:
            _validate_runtime_service_url(base_url, "base_url")
            kwargs["base_url"] = base_url
        if "gpt-5" in model:
            return OpenAILLMService(
                api_key=api_key,
                settings=OpenAILLMSettings(
                    model=model,
                    extra={
                        "reasoning_effort": _openai_gpt5_reasoning_effort(model),
                        "verbosity": "low",
                    },
                ),
                **kwargs,
            )
        return OpenAILLMService(
            api_key=api_key,
            settings=OpenAILLMSettings(model=model, temperature=0.1),
            **kwargs,
        )
    elif provider == ServiceProviders.GROQ.value:
        groq_extra: dict[str, object] = {}
        if _is_groq_reasoning_model(model):
            # Groq's reasoning models return their chain of thought in the
            # message content by default, which the pipeline then hands to TTS
            # -- the bot speaks its own reasoning. "hidden" keeps the reasoning
            # out of the content entirely.
            #
            # It has to travel in extra_body: `extra` is merged into the
            # top-level kwargs of AsyncCompletions.create(), whose signature is
            # typed, so a Groq-only field passed there raises TypeError before
            # any HTTP request is made. extra_body forwards it verbatim in the
            # request body instead.
            groq_extra["extra_body"] = {"reasoning_format": "hidden"}
        return GroqLLMService(
            api_key=api_key,
            settings=GroqLLMSettings(model=model, temperature=0.1, extra=groq_extra),
        )
    elif provider == ServiceProviders.OPENROUTER.value:
        kwargs = {}
        if base_url:
            _validate_runtime_service_url(base_url, "base_url")
            kwargs["base_url"] = base_url
        return OpenRouterLLMService(
            api_key=api_key,
            settings=OpenRouterLLMSettings(model=model, temperature=0.1),
            **kwargs,
        )
    elif provider == ServiceProviders.GOOGLE.value:
        model = _migrate_deprecated_google_model(model)
        return DograhGoogleLLMService(
            api_key=api_key,
            settings=GoogleLLMSettings(
                model=model,
                temperature=0.1,
                # Pipecat executes tools; the SDK should return their calls.
                extra={"automatic_function_calling": {"disable": True}},
            ),
        )
    elif provider == ServiceProviders.GOOGLE_VERTEX.value:
        vertex_api_key = api_key if isinstance(api_key, str) and api_key else None
        return DograhGoogleVertexLLMService(
            vertex_api_key=vertex_api_key,
            credentials=credentials,
            project_id=project_id or "",
            location=vertex_location,
            settings=GoogleVertexLLMSettings(
                model=model,
                temperature=0.1,
                extra={"automatic_function_calling": {"disable": True}},
            ),
        )
    elif provider == ServiceProviders.AZURE.value:
        if endpoint:
            _validate_runtime_service_url(endpoint, "endpoint")
        return AzureLLMService(
            api_key=api_key,
            endpoint=endpoint,
            settings=AzureLLMSettings(model=model, temperature=0.1),
        )
    elif provider == ServiceProviders.DOGRAH.value:
        return DograhLLMService(
            base_url=f"{MPS_API_URL}/api/v1/llm",
            api_key=api_key,
            correlation_id=correlation_id,
            usage_context=usage_context,
            settings=OpenAILLMSettings(model=model),
        )
    elif provider == ServiceProviders.AWS_BEDROCK.value:
        return AWSBedrockLLMService(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region,
            settings=AWSBedrockLLMSettings(model=model),
        )
    elif provider == ServiceProviders.SPEACHES.value:
        base_url = base_url or "http://localhost:11434/v1"
        _validate_runtime_service_url(base_url, "base_url")
        return SpeachesLLMService(
            base_url=base_url,
            api_key=api_key or "none",
            settings=SpeachesLLMSettings(model=model),
        )
    elif provider == ServiceProviders.HUGGINGFACE.value:
        base_url = base_url or "https://router.huggingface.co/v1"
        _validate_runtime_service_url(base_url, "base_url")
        return HuggingFaceLLMService(
            api_key=api_key,
            base_url=base_url,
            bill_to=bill_to,
            settings=HuggingFaceLLMSettings(model=model, temperature=0.1),
        )
    elif provider == ServiceProviders.MINIMAX.value:
        base_url = base_url or "https://api.minimax.io/v1"
        _validate_runtime_service_url(base_url, "base_url")
        return MiniMaxLLMService(
            api_key=api_key,
            base_url=base_url,
            settings=MiniMaxLLMService.Settings(
                model=model,
                temperature=temperature if temperature is not None else 1.0,
            ),
        )
    elif provider == ServiceProviders.SARVAM.value:
        return SarvamLLMService(
            api_key=api_key,
            settings=SarvamLLMSettings(
                model=model,
                temperature=temperature if temperature is not None else 0.5,
            ),
        )
    else:
        raise HTTPException(status_code=400, detail=f"Invalid LLM provider {provider}")


@_report_service_factory_failures(ErrorSource.LLM, config_section="realtime")
def create_realtime_llm_service(user_config, audio_config: "AudioConfig"):
    """Create a realtime (speech-to-speech) LLM service that handles STT+LLM+TTS.

    These services bypass separate STT/TTS and handle audio directly via
    a bidirectional WebSocket connection. Reads from user_config.realtime.
    """
    realtime_config = user_config.realtime
    provider = realtime_config.provider
    model = realtime_config.model
    api_key = realtime_config.api_key
    voice = getattr(realtime_config, "voice", None)
    language = getattr(realtime_config, "language", None)

    vertex_location = (
        _google_vertex_location(getattr(realtime_config, "location", None), "Realtime")
        if provider == ServiceProviders.GOOGLE_VERTEX_REALTIME.value
        else None
    )

    logger.info(
        f"Creating realtime LLM service: provider={provider}, model={model}, "
        f"voice={voice}, language={language}"
        + (f", location={vertex_location}" if vertex_location else "")
    )

    if provider == ServiceProviders.OPENAI_REALTIME.value and model == "gpt-live-1":
        from api.services.pipecat.realtime.openai_live import DograhOpenAILiveLLMService

        return DograhOpenAILiveLLMService(
            api_key=api_key,
            backend_model=realtime_config.backend_model,
            settings=DograhOpenAILiveLLMService.Settings(
                model=model,
                voice=voice or "marin",
            ),
        )
    elif provider == ServiceProviders.OPENAI_REALTIME.value:
        from api.services.pipecat.realtime.openai_realtime import (
            DograhOpenAIRealtimeLLMService,
        )
        from pipecat.services.openai.realtime.events import (
            AudioConfiguration,
            AudioInput,
            AudioOutput,
            InputAudioTranscription,
            SessionProperties,
        )

        # Pin the transcription language when configured. Without it the model
        # auto-detects per utterance, which misfires on short/noisy telephony
        # audio (e.g. Portuguese transcribed as English or Chinese).
        transcription_kwargs = {}
        if language:
            transcription_kwargs["language"] = language

        return DograhOpenAIRealtimeLLMService(
            api_key=api_key,
            settings=DograhOpenAIRealtimeLLMService.Settings(
                model=model,
                session_properties=SessionProperties(
                    audio=AudioConfiguration(
                        input=AudioInput(
                            transcription=InputAudioTranscription(
                                **transcription_kwargs
                            ),
                        ),
                        output=AudioOutput(
                            voice=voice or "alloy",
                        ),
                    ),
                ),
            ),
        )
    elif provider == ServiceProviders.GROK_REALTIME.value:
        from api.services.pipecat.realtime.grok_realtime import (
            DograhGrokRealtimeLLMService,
        )
        from pipecat.services.xai.realtime.events import (
            AudioConfiguration,
            AudioInput,
            InputAudioTranscription,
            SessionProperties,
        )

        grok_voice = voice or "ara"
        if grok_voice.lower() in {"ara", "rex", "sal", "eve", "leo"}:
            grok_voice = grok_voice.lower()

        return DograhGrokRealtimeLLMService(
            api_key=api_key,
            settings=DograhGrokRealtimeLLMService.Settings(
                model=model,
                session_properties=SessionProperties(
                    voice=grok_voice,
                    audio=AudioConfiguration(
                        input=AudioInput(
                            transcription=InputAudioTranscription(),
                        ),
                    ),
                ),
            ),
        )
    elif provider == ServiceProviders.ULTRAVOX_REALTIME.value:
        from api.services.pipecat.realtime.ultravox_realtime import (
            DograhUltravoxOneShotInputParams,
            DograhUltravoxRealtimeLLMService,
        )

        return DograhUltravoxRealtimeLLMService(
            params=DograhUltravoxOneShotInputParams(
                api_key=api_key,
                model=model,
                voice=voice,
                output_medium="voice",
            ),
            settings=DograhUltravoxRealtimeLLMService.Settings(
                model=model,
                output_medium="voice",
            ),
        )
    elif provider == ServiceProviders.AWS_NOVA_SONIC.value:
        from api.services.pipecat.realtime.aws_nova_sonic import (
            DograhAWSNovaSonicLLMService,
        )
        from pipecat.services.aws.nova_sonic.llm import AudioConfig as NovaAudioConfig

        return DograhAWSNovaSonicLLMService(
            secret_access_key=realtime_config.aws_secret_key,
            access_key_id=realtime_config.aws_access_key,
            session_token=realtime_config.aws_session_token or None,
            region=realtime_config.aws_region,
            audio_config=NovaAudioConfig(
                input_sample_rate=audio_config.transport_in_sample_rate,
                output_sample_rate=audio_config.transport_out_sample_rate,
            ),
            settings=DograhAWSNovaSonicLLMService.Settings(
                model=model,
                voice=voice or "matthew",
                endpointing_sensitivity=realtime_config.endpointing_sensitivity,
                temperature=realtime_config.temperature,
                max_tokens=realtime_config.max_tokens,
                top_p=realtime_config.top_p,
            ),
        )
    elif provider == ServiceProviders.GOOGLE_REALTIME.value:
        from api.services.pipecat.realtime.gemini_live import (
            DograhGeminiLiveLLMService,
        )

        # Gemini Live enables input/output audio transcription by default
        # in its _connect() method — no need to configure it explicitly.
        settings_kwargs = {
            "model": model,
            "voice": voice or "Puck",
        }
        if language:
            settings_kwargs["language"] = language
        return DograhGeminiLiveLLMService(
            api_key=api_key,
            settings=DograhGeminiLiveLLMService.Settings(**settings_kwargs),
        )
    elif provider == ServiceProviders.GOOGLE_VERTEX_REALTIME.value:
        from api.services.pipecat.realtime.gemini_live_vertex import (
            DograhGeminiLiveVertexLLMService,
        )

        project_id = getattr(realtime_config, "project_id", None)
        credentials = getattr(realtime_config, "credentials", None)

        settings_kwargs = {
            "model": model,
            "voice": voice or "Charon",
        }
        if language:
            settings_kwargs["language"] = language
        return DograhGeminiLiveVertexLLMService(
            credentials=credentials,
            project_id=project_id,
            location=vertex_location,
            settings=DograhGeminiLiveVertexLLMService.Settings(**settings_kwargs),
        )
    elif provider == ServiceProviders.AZURE_REALTIME.value:
        from api.services.pipecat.realtime.azure_realtime import (
            DograhAzureRealtimeLLMService,
        )
        from pipecat.services.openai.realtime.events import (
            AudioConfiguration,
            AudioInput,
            AudioOutput,
            InputAudioTranscription,
            SessionProperties,
        )

        endpoint = getattr(realtime_config, "endpoint", None) or ""
        if not endpoint:
            raise HTTPException(
                status_code=400,
                detail="Azure Realtime requires an endpoint.",
            )
        _validate_runtime_service_url(endpoint, "endpoint")
        api_version = getattr(realtime_config, "api_version", None) or "v1"
        parsed_endpoint = urlparse(endpoint)
        if api_version == "v1":
            # Azure's GA Realtime API uses the deployment name as `model` and
            # deliberately has no date-based api-version query parameter.
            path = "/openai/v1/realtime"
            query = urlencode({"model": model})
        else:
            # Preserve explicitly configured preview deployments while users
            # migrate. Microsoft deprecated this protocol on April 30, 2026.
            path = "/openai/realtime"
            query = urlencode({"api-version": api_version, "deployment": model})
        wss_url = urlunparse(
            (
                "wss",
                parsed_endpoint.netloc,
                path,
                "",
                query,
                "",
            )
        )
        return DograhAzureRealtimeLLMService(
            api_key=api_key,
            base_url=wss_url,
            settings=DograhAzureRealtimeLLMService.Settings(
                model=model,
                session_properties=SessionProperties(
                    audio=AudioConfiguration(
                        input=AudioInput(
                            transcription=InputAudioTranscription(),
                        ),
                        output=AudioOutput(
                            voice=voice or "alloy",
                        ),
                    ),
                ),
            ),
        )
    else:
        raise HTTPException(
            status_code=400, detail=f"Invalid realtime LLM provider {provider}"
        )


def create_llm_service(
    user_config,
    correlation_id: str | None = None,
    usage_context: str | None = None,
):
    """Create and return appropriate LLM service based on user configuration."""
    provider = user_config.llm.provider
    model = user_config.llm.model
    api_key = user_config.llm.api_key

    kwargs = {}
    if provider in (
        ServiceProviders.OPENAI.value,
        ServiceProviders.ATLASCLOUD.value,
    ):
        kwargs["base_url"] = user_config.llm.base_url
    elif provider == ServiceProviders.OPENROUTER.value:
        kwargs["base_url"] = user_config.llm.base_url
    elif provider == ServiceProviders.AZURE.value:
        kwargs["endpoint"] = user_config.llm.endpoint
    elif provider == ServiceProviders.SPEACHES.value:
        kwargs["base_url"] = user_config.llm.base_url
    elif provider == ServiceProviders.HUGGINGFACE.value:
        kwargs["base_url"] = user_config.llm.base_url
        kwargs["bill_to"] = user_config.llm.bill_to
    elif provider == ServiceProviders.AWS_BEDROCK.value:
        kwargs["aws_access_key"] = user_config.llm.aws_access_key
        kwargs["aws_secret_key"] = user_config.llm.aws_secret_key
        kwargs["aws_region"] = user_config.llm.aws_region
    elif provider == ServiceProviders.GOOGLE_VERTEX.value:
        kwargs["project_id"] = user_config.llm.project_id
        kwargs["location"] = user_config.llm.location
        kwargs["credentials"] = user_config.llm.credentials
    elif provider == ServiceProviders.MINIMAX.value:
        kwargs["base_url"] = user_config.llm.base_url
        kwargs["temperature"] = user_config.llm.temperature
    elif provider == ServiceProviders.SARVAM.value:
        kwargs["temperature"] = user_config.llm.temperature

    return create_llm_service_from_provider(
        provider,
        model,
        api_key,
        correlation_id=correlation_id,
        usage_context=usage_context,
        **kwargs,
    )


def create_llm_service_with_model_override(
    user_config: "EffectiveAIModelConfiguration",
    model_override: str | None,
    correlation_id: str | None = None,
    usage_context: str | None = None,
):
    """Create an LLM service with an optional model override.

    The copied configuration is delegated to ``create_llm_service`` so provider-
    specific settings continue to be extracted in one place.
    """
    if model_override is None:
        return create_llm_service(
            user_config,
            correlation_id=correlation_id,
            usage_context=usage_context,
        )

    if user_config.llm is None:
        raise ValueError("Cannot override the model without an LLM configuration")

    llm_config = user_config.llm.model_copy(update={"model": model_override})
    overridden_config = user_config.model_copy(update={"llm": llm_config})
    return create_llm_service(
        overridden_config,
        correlation_id=correlation_id,
        usage_context=usage_context,
    )
