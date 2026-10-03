"""Gemini speech on Vertex: which credential reaches which Google API."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame

from api.services.configuration.check_validity import UserConfigurationValidator
from api.services.configuration.options.google_vertex_catalog import (
    check_vertex_config,
    gemini_tts_voices,
    vertex_models,
)
from api.services.configuration.registry import (
    GoogleVertexSTTConfiguration,
    GoogleVertexTTSConfiguration,
)
from api.services.pipecat import service_factory as sf

KEY = "AQ.SECRETVERTEXKEY123"


def _errors(service, name):
    return UserConfigurationValidator()._validate_service(service, name)


def _audio(rate=16000):
    return SimpleNamespace(transport_in_sample_rate=rate)


@pytest.mark.parametrize("model", vertex_models("tts"))
def test_tts_api_key_is_accepted_for_the_vertex_api_path(model):
    cfg = GoogleVertexTTSConfiguration(model=model, api_key=KEY, project_id="p")
    assert _errors(cfg, "tts") == []


def test_gemini_voice_catalog_is_exact_and_validates_voice_ids():
    voices = gemini_tts_voices("gemini-3.1-flash-tts-preview")
    assert len(voices) == 30
    assert {voice.id for voice in voices} >= {"Kore", "Puck", "Charon"}
    assert (
        check_vertex_config(
            "tts",
            model="gemini-3.1-flash-tts-preview",
            location="global",
            has_api_key=True,
            has_credentials=False,
            project_id="p",
            voice="Kore",
        )
        is None
    )
    assert "not in the documented voice" in check_vertex_config(
        "tts",
        model="gemini-3.1-flash-tts-preview",
        location="global",
        has_api_key=True,
        has_credentials=False,
        project_id="p",
        voice="Unknown",
    )


def test_tts_api_key_needs_the_global_endpoint_and_no_service_account():
    errors = _errors(
        GoogleVertexTTSConfiguration(api_key=KEY, project_id="p", location="us"),
        "tts",
    )
    assert errors and "cannot choose a location" in errors[0]["message"]
    assert KEY not in errors[0]["message"]
    errors = _errors(GoogleVertexTTSConfiguration(api_key=KEY), "tts")
    assert errors and "needs project_id" in errors[0]["message"]
    both = GoogleVertexTTSConfiguration(api_key=KEY, credentials="{}")
    assert _errors(both, "tts")


def test_stt_live_api_key_needs_project_for_a_complete_resource():
    kwargs = dict(
        model="gemini-3.5-transcribe-live-preview",
        location="global",
        has_api_key=True,
        has_credentials=False,
    )
    assert "needs project_id" in check_vertex_config("stt", **kwargs)
    assert check_vertex_config("stt", project_id="proj-1", **kwargs) is None
    cfg = GoogleVertexSTTConfiguration(api_key=KEY)
    errors = _errors(cfg, "stt")
    assert errors and KEY not in errors[0]["message"]
    assert (
        _errors(GoogleVertexSTTConfiguration(api_key=KEY, project_id="p"), "stt") == []
    )


def test_model_resource_is_complete_only_when_a_project_is_known():
    assert sf._vertex_model_resource("m", None, "global") == "m"
    assert (
        sf._vertex_model_resource("m", "p", "global")
        == "projects/p/locations/global/publishers/google/models/m"
    )
    full = "projects/x/locations/us/publishers/google/models/m"
    assert sf._vertex_model_resource(full, "p", "global") == full


@pytest.mark.asyncio
async def test_stt_api_key_session_uses_the_complete_model_resource():
    service = _stt(api_key=KEY, language="en-US")
    connect = MagicMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value="session")
    connect.return_value = ctx
    service._client = SimpleNamespace(
        aio=SimpleNamespace(live=SimpleNamespace(connect=connect))
    )
    service._call_event_handler = AsyncMock()
    await service._open_session()
    assert connect.call_args.kwargs["model"] == (
        "projects/p/locations/global/publishers/google/models/"
        "gemini-3.5-transcribe-live-preview"
    )


def test_api_key_tts_uses_a_vertex_client_never_ai_studio():
    cfg = GoogleVertexTTSConfiguration(api_key=KEY, prompt="warm")
    with patch.object(sf, "GenaiClient") as client:
        service = sf.create_tts_service(SimpleNamespace(tts=cfg), _audio())
    assert isinstance(service, sf.DograhGeminiVertexApiTTSService)
    client.assert_called_once()
    kwargs = client.call_args.kwargs
    assert kwargs["vertexai"] is True and kwargs["api_key"] == KEY
    assert "project" not in kwargs or kwargs["project"] is None


def test_api_key_tts_keeps_project_location_in_complete_model_resource():
    cfg = GoogleVertexTTSConfiguration(
        api_key=KEY,
        project_id="proj-1",
        location="global",
        model="gemini-3.1-flash-tts-preview",
        voice="Kore",
    )
    with patch.object(sf, "GenaiClient"):
        service = sf.create_tts_service(SimpleNamespace(tts=cfg), _audio())
    assert service._vertex_project_id == "proj-1"
    assert service._vertex_location == "global"


def test_service_account_tts_still_uses_cloud_text_to_speech():
    cfg = GoogleVertexTTSConfiguration(location="us")
    with patch.object(sf, "GeminiTTSService") as svc:
        sf.create_tts_service(SimpleNamespace(tts=cfg), _audio())
    assert svc.call_args.kwargs["use_genai"] is False


class _Chunk:
    def __init__(self, data, mime_type=None):
        part = SimpleNamespace(
            inline_data=SimpleNamespace(data=data, mime_type=mime_type)
        )
        self.candidates = [SimpleNamespace(content=SimpleNamespace(parts=[part]))]


def _api_key_service(**cfg):
    config = GoogleVertexTTSConfiguration(api_key=KEY, **cfg)
    with patch.object(sf, "GenaiClient"):
        service = sf.create_tts_service(SimpleNamespace(tts=config), _audio())
    service._sample_rate = 24000
    service.start_tts_usage_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    return service


def test_api_key_tts_pins_gemini_native_output_rate():
    service = _api_key_service()

    assert service._init_sample_rate == 24000


@pytest.mark.asyncio
async def test_api_key_tts_streams_pcm_with_voice_language_and_prompt():
    service = _api_key_service(
        model="gemini-3.1-flash-tts-preview",
        voice="Puck",
        language="en-US",
        prompt="calm",
        project_id="proj-1",
        location="global",
    )
    audio = b"\x01\x02" * 40000

    async def stream():
        yield _Chunk(audio[:50000])
        yield _Chunk(audio[50000:])

    models = SimpleNamespace(generate_content_stream=AsyncMock(return_value=stream()))
    service._client = SimpleNamespace(aio=SimpleNamespace(models=models))

    frames = [f async for f in service.run_tts("Hello there.", "ctx")]

    out = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    assert out and not [f for f in frames if isinstance(f, ErrorFrame)]
    assert all(f.sample_rate == 24000 and f.num_channels == 1 for f in out)
    assert b"".join(f.audio for f in out) == audio
    call = models.generate_content_stream.call_args.kwargs
    assert (
        call["model"]
        == "projects/proj-1/locations/global/publishers/google/models/gemini-3.1-flash-tts-preview"
    )
    assert call["contents"] == "calm: Hello there."
    speech = call["config"].speech_config
    assert call["config"].response_modalities == ["AUDIO"]
    assert speech.language_code == "en-US"
    assert speech.voice_config.prebuilt_voice_config.voice_name == "Puck"


@pytest.mark.asyncio
async def test_api_key_tts_error_frame_is_redacted_by_the_factory_hook():
    service = _api_key_service()
    models = SimpleNamespace(
        generate_content_stream=AsyncMock(side_effect=RuntimeError(f"403 key={KEY}"))
    )
    service._client = SimpleNamespace(aio=SimpleNamespace(models=models))
    seen = []

    async def sink(frame, *a, **k):
        seen.append(frame)

    service.push_frame = sink
    sf._install_error_redaction(service, [KEY])
    for frame in [f async for f in service.run_tts("Hi.", "ctx")]:
        await service.push_frame(frame)
    errors = [f for f in seen if isinstance(f, ErrorFrame)]
    assert errors and KEY not in errors[0].error and "[REDACTED]" in errors[0].error


def _stt(**cfg):
    config = GoogleVertexSTTConfiguration(project_id="p", **cfg)
    with (
        patch.object(sf, "GenaiClient"),
        patch.object(sf.GoogleVertexLLMService, "_get_credentials", return_value="c"),
    ):
        return sf.create_stt_service(SimpleNamespace(stt=config), _audio())


def test_stt_live_config_uses_documented_language_codes():
    live = _stt(language="en-US")._build_live_config()
    tx = live.input_audio_transcription
    assert tx.language_codes == ["en-US"]
    assert tx.language_hints is None and tx.language_auto is None
    assert [m.value for m in live.response_modalities] == ["TEXT"]


def test_stt_live_config_without_language_leaves_auto_detection_to_the_api():
    tx = _stt()._build_live_config().input_audio_transcription
    assert tx.language_codes is None and tx.language_auto is None


def test_stt_live_config_sends_custom_vocabulary_not_deprecated_phrases():
    service = _stt(language="en-US")
    service._settings.adaptation_phrases = ["WaHPACT"]
    tx = service._build_live_config().input_audio_transcription
    assert tx.custom_vocabulary == ["WaHPACT"] and tx.adaptation_phrases is None


# ---- adapter against a local server that speaks Vertex's streaming wire format
def _serve(audio: bytes):
    import base64
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = {}
    event = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/L16;codec=pcm;rate=24000",
                                "data": base64.b64encode(audio).decode(),
                            }
                        }
                    ],
                }
            }
        ]
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path.split("?")[0]
            seen["has_key_header"] = "x-goog-api-key" in {
                k.lower() for k in self.headers
            }
            seen["body"] = json.loads(
                self.rfile.read(int(self.headers["content-length"]))
            )
            body = f"data: {json.dumps(event)}\r\n\r\n".encode()
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


@pytest.mark.asyncio
async def test_api_key_tts_request_construction_and_audio_decoding_end_to_end():
    from google.genai.types import HttpOptions

    audio = b"\x01\x02" * 40000
    server, seen = _serve(audio)
    try:
        service = sf.DograhGeminiVertexApiTTSService(
            vertex_api_key=KEY,
            project_id="proj-1",
            settings=sf.GeminiTTSSettings(
                model="gemini-2.5-flash-tts", voice="Kore", language="en-US"
            ),
            http_options=HttpOptions(
                base_url=f"http://127.0.0.1:{server.server_port}/"
            ),
        )
        service._sample_rate = 24000
        service.start_tts_usage_metrics = AsyncMock()
        service.stop_ttfb_metrics = AsyncMock()
        frames = [f async for f in service.run_tts("Hello there.", "ctx")]
        await service._client.aio.aclose()
    finally:
        server.shutdown()

    assert not [f for f in frames if isinstance(f, ErrorFrame)]
    assert b"".join(f.audio for f in frames if isinstance(f, TTSAudioRawFrame)) == audio
    # Express-style resource: no project or location, key sent as a header.
    assert seen["path"] == (
        "/v1/projects/proj-1/locations/global/publishers/google/models/"
        "gemini-2.5-flash-tts:streamGenerateContent"
    )
    assert seen["has_key_header"] is True
    speech = seen["body"]["generationConfig"]["speechConfig"]
    assert speech["languageCode"] == "en-US"
    voice = speech["voiceConfig"]["prebuiltVoiceConfig"]
    # The SDK serializes this one field in snake_case; proto3 JSON accepts both.
    assert (voice.get("voiceName") or voice.get("voice_name")) == "Kore"


@pytest.mark.parametrize(
    "mime",
    [
        None,
        "audio/L16;codec=pcm;rate=24000",
        "audio/pcm;rate=24000",
        "audio/l16; rate=24000; channels=1",
    ],
)
def test_pcm_audio_metadata_that_matches_is_accepted(mime):
    sf._check_pcm_audio_mime(mime, expected_rate=24000)


@pytest.mark.parametrize(
    "mime",
    [
        "audio/L16;codec=pcm;rate=16000",  # another sample rate
        "audio/wav",  # a container, not raw PCM
        "audio/mpeg",
        "audio/L16;codec=pcm;rate=24000;channels=2",
        "audio/pcm;rate=fast",
    ],
)
def test_incompatible_audio_metadata_is_rejected_clearly(mime):
    with pytest.raises(RuntimeError, match="unsupported audio format.*24000 Hz"):
        sf._check_pcm_audio_mime(mime, expected_rate=24000)


@pytest.mark.asyncio
async def test_api_key_tts_fails_with_an_error_frame_on_incompatible_audio_format():
    service = _api_key_service()

    async def stream():
        yield _Chunk(b"\x01\x02" * 40000, "audio/L16;codec=pcm;rate=16000")

    models = SimpleNamespace(generate_content_stream=AsyncMock(return_value=stream()))
    service._client = SimpleNamespace(aio=SimpleNamespace(models=models))

    frames = [f async for f in service.run_tts("Hello there.", "ctx")]

    assert not [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    errors = [f for f in frames if isinstance(f, ErrorFrame)]
    assert len(errors) == 1 and "16000" in errors[0].error


@pytest.mark.asyncio
async def test_api_key_tts_accepts_the_documented_gemini_audio_format():
    service = _api_key_service()
    audio = b"\x01\x02" * 40000

    async def stream():
        yield _Chunk(audio, "audio/L16;codec=pcm;rate=24000")

    models = SimpleNamespace(generate_content_stream=AsyncMock(return_value=stream()))
    service._client = SimpleNamespace(aio=SimpleNamespace(models=models))

    frames = [f async for f in service.run_tts("Hello there.", "ctx")]

    assert not [f for f in frames if isinstance(f, ErrorFrame)]
    assert b"".join(f.audio for f in frames if isinstance(f, TTSAudioRawFrame)) == audio
