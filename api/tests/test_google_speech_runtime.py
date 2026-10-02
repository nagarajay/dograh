"""Google Cloud Speech-to-Text v2 / Text-to-Speech runtime, with mocked Google.

Nothing here touches the network: services are built through the real factory
with a synthetic service account, then their Google client is replaced by a fake.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from google.cloud.speech_v2.types import cloud_speech
from pipecat.frames.frames import (
    ErrorFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.services.google.tts import GeminiTTSSettings

from api.services.configuration.registry import (
    GoogleSTTConfiguration,
    GoogleTTSConfiguration,
    GoogleVertexTTSConfiguration,
)
from api.services.pipecat import service_factory as sf


def _service_account() -> tuple[str, str]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    info = {
        "type": "service_account",
        "project_id": "proj-1",
        "private_key_id": "keyid1234567890",
        "private_key": pem,
        "client_email": "svc@proj-1.iam.gserviceaccount.com",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    return json.dumps(info), pem


SA, PEM = _service_account()
PEM_BODY = PEM.splitlines()[1]


def _audio(rate=16000):
    return SimpleNamespace(transport_in_sample_rate=rate)


def _stt(**cfg):
    config = GoogleSTTConfiguration(credentials=SA, **cfg)
    return sf.create_stt_service(SimpleNamespace(stt=config), _audio())


def _tts(config):
    return sf.create_tts_service(SimpleNamespace(tts=config), _audio())


def _pcm(service, rate=24000):
    service._sample_rate = rate
    return service


async def _agen(items):
    for item in items:
        yield item


def _collect_frames(service):
    frames = []

    async def push_frame(frame, *a, **k):
        frames.append(frame)

    service.push_frame = push_frame
    return frames


# ------------------------------------------------------------------ STT
def test_cloud_stt_uses_regional_endpoint_and_project_from_credentials():
    service = _stt(model="chirp_3", location="us", language="en-US")
    assert service._project_id == "proj-1"
    assert service._client.transport._host.startswith("us-speech.googleapis.com")
    assert service._settings.model == "chirp_3"


@pytest.mark.asyncio
async def test_cloud_stt_streaming_config_is_linear16_at_transport_rate():
    service = _stt(model="chirp_3", location="us", language="en-US")
    service.create_task = lambda coro, *a, **k: (coro.close(), SimpleNamespace())[1]
    service._call_event_handler = AsyncMock()
    await service._connect()

    config = service._config
    decoding = config.config.explicit_decoding_config
    assert (
        decoding.encoding == cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16
    )
    assert decoding.sample_rate_hertz == service.sample_rate or service.sample_rate in (
        0,
        None,
    )
    assert decoding.audio_channel_count == 1
    assert config.config.model == "chirp_3"
    assert list(config.config.language_codes) == ["en-US"]
    assert config.streaming_features.interim_results is True

    first = await anext(service._request_generator())
    assert first.recognizer == "projects/proj-1/locations/us/recognizers/_"


@pytest.mark.asyncio
async def test_cloud_stt_emits_interim_then_final_transcription():
    service = _stt(model="chirp_3", location="us")
    frames = _collect_frames(service)
    service.emit_stt_usage_metrics = AsyncMock()
    service._handle_transcription = AsyncMock()
    service._stream_start_time = 10**15  # far future: never hits the 5 min limit

    def result(text, final):
        return SimpleNamespace(
            is_final=final, alternatives=[SimpleNamespace(transcript=text)]
        )

    responses = [
        SimpleNamespace(results=[]),
        SimpleNamespace(results=[result("hello th", False)]),
        SimpleNamespace(results=[result("", True)]),
        SimpleNamespace(results=[result("hello there", True)]),
    ]
    await service._process_responses(_agen(responses))

    assert [type(f) for f in frames] == [InterimTranscriptionFrame, TranscriptionFrame]
    assert frames[0].text == "hello th"
    assert frames[1].text == "hello there" and frames[1].finalized is True


@pytest.mark.asyncio
async def test_cloud_stt_cancel_stops_the_stream_task():
    service = _stt()
    service._call_event_handler = AsyncMock()
    task = SimpleNamespace()
    service._streaming_task = task
    service.cancel_task = AsyncMock()
    await service._disconnect()
    service.cancel_task.assert_awaited_once_with(task)
    assert service._streaming_task is None


@pytest.mark.asyncio
async def test_cloud_stt_audio_is_dropped_not_queued_without_a_stream():
    service = _stt()
    service._streaming_task = None
    service._request_queue = asyncio.Queue()
    assert [f async for f in service.run_stt(b"\x00\x01")] == [None]
    assert service._request_queue.empty()


# ------------------------------------------------------------------ TTS
class _FakeTTSClient:
    def __init__(self, chunks, boom=None, hang=False):
        self.chunks = chunks
        self.hang = hang
        self.boom = boom
        self.requests = []
        self.closed = False

    async def streaming_synthesize(self, request_generator):
        async for request in request_generator:
            self.requests.append(request)
        if self.boom:
            raise self.boom

        async def responses():
            try:
                for chunk in self.chunks:
                    yield SimpleNamespace(audio_content=chunk)
                if self.hang:
                    await asyncio.sleep(3600)
            finally:
                self.closed = True

        return responses()


def _chirp_service():
    config = GoogleTTSConfiguration(credentials=SA, location="us")
    return _pcm(_tts(config))


@pytest.mark.asyncio
async def test_cloud_tts_requests_pcm_at_service_rate_and_yields_mono_frames():
    service = _chirp_service()
    audio = b"\x01\x02" * 4000
    service._client = _FakeTTSClient([audio[:3000], audio[3000:]])
    service.start_tts_usage_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()

    frames = [f async for f in service.run_tts("Hello there.", "ctx-1")]

    audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    assert audio_frames and not [f for f in frames if isinstance(f, ErrorFrame)]
    assert all(f.sample_rate == 24000 and f.num_channels == 1 for f in audio_frames)
    assert b"".join(f.audio for f in audio_frames) == audio

    config, text = service._client.requests
    streaming = config.streaming_config
    assert streaming.streaming_audio_config.audio_encoding.name == "PCM"
    assert streaming.streaming_audio_config.sample_rate_hertz == 24000
    assert streaming.voice.name == "en-US-Chirp3-HD-Charon"
    assert streaming.voice.language_code == "en-US"
    assert text.input.text == "Hello there."


@pytest.mark.asyncio
async def test_cloud_tts_interruption_cancels_cleanly_and_closes_the_stream():
    service = _chirp_service()
    client = _FakeTTSClient([b"\x00\x00" * 100000], hang=True)
    service._client = client
    service.start_tts_usage_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    got_first = asyncio.Event()

    async def consume():
        async for frame in service.run_tts("A long answer.", "ctx-2"):
            if isinstance(frame, TTSAudioRawFrame):
                got_first.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(got_first.wait(), 5)
    task.cancel()  # what a barge-in does to the in-flight synthesis
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.closed is True


@pytest.mark.asyncio
async def test_gemini_tts_requests_model_voice_and_prompt():
    config = GoogleVertexTTSConfiguration(
        model="gemini-2.5-flash-tts",
        voice="Kore",
        location="us",
        credentials=SA,
        prompt="calm",
    )
    service = _pcm(_tts(config))
    client = _FakeTTSClient([b"\x00\x00" * 3000])
    service._client = client
    service.start_tts_usage_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()

    frames = [f async for f in service.run_tts("Hi.", "ctx-3")]

    assert [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    config_req, text_req = client.requests
    voice = config_req.streaming_config.voice
    assert (voice.name, voice.model_name) == ("Kore", "gemini-2.5-flash-tts")
    assert config_req.streaming_config.streaming_audio_config.sample_rate_hertz == 24000
    assert text_req.input.prompt == "calm"


@pytest.mark.asyncio
async def test_vertex_gemini_tts_direct_generation_uses_native_rate_before_setup():
    service = sf.DograhGeminiVertexApiTTSService(
        vertex_api_key="platform-key",
        project_id="proj-1",
        location="global",
        settings=GeminiTTSSettings(
            model="gemini-3.1-flash-tts-preview",
            voice="Achernar",
            language="en-IN",
            prompt="warm",
        ),
        sample_rate=24000,
    )
    # The sample worker calls _run_genai_tts directly, before TTS setup stores
    # the requested rate in the runtime property.
    service._sample_rate = 0
    class _FakeGenAIModels:
        def __init__(self):
            self.kwargs = None

        async def generate_content_stream(self, **kwargs):
            self.kwargs = kwargs
            chunk = SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(
                            parts=[
                                SimpleNamespace(
                                    inline_data=SimpleNamespace(
                                        data=b"\x00\x00" * 3000
                                    )
                                )
                            ]
                        )
                    )
                ]
            )
            async def stream():
                yield chunk
            return stream()

    models = _FakeGenAIModels()
    service._client = SimpleNamespace(aio=SimpleNamespace(models=models))
    service.start_tts_usage_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()

    frames = [f async for f in service._run_genai_tts("Hi.", "ctx-sample")]

    audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    assert audio_frames
    assert all(f.sample_rate == 24000 and f.num_channels == 1 for f in audio_frames)
    request = models.kwargs
    assert request["config"].response_modalities == ["AUDIO"]
    assert request["model"] == (
        "projects/proj-1/locations/global/publishers/google/models/"
        "gemini-3.1-flash-tts-preview"
    )


# ------------------------------------------------------- safe errors
@pytest.mark.asyncio
async def test_tts_error_frame_never_carries_the_credential():
    service = _chirp_service()
    service.start_tts_usage_metrics = AsyncMock()
    leaked = f"403 denied for {SA} key {PEM_BODY} id keyid1234567890"
    service._client = _FakeTTSClient([], boom=RuntimeError(leaked))
    pushed = []

    async def upstream_push_error(error_msg, exception=None, *a, **k):
        pushed.append((error_msg, str(exception)))

    service.push_error = upstream_push_error
    # Re-install redaction on top of the stand-in the pipeline would provide.
    sf._install_error_redaction(
        service, sf._google_secret_values(SimpleNamespace(credentials=SA, api_key=None))
    )
    frames = [f async for f in service.run_tts("Hi.", "ctx-4")]
    for frame in frames:
        if isinstance(frame, ErrorFrame):
            await service.push_frame(frame)

    assert pushed, "the adapter should report the failure"
    for message, exception_text in pushed:
        for secret in (PEM_BODY, "keyid1234567890", "private_key"):
            assert secret not in message and secret not in exception_text
    assert "[REDACTED]" in pushed[0][0]


@pytest.mark.asyncio
async def test_error_frames_pushed_directly_are_redacted_too():
    service = _chirp_service()
    seen = []

    async def sink(frame, *a, **k):
        seen.append(frame)

    service.push_frame = sink
    sf._install_error_redaction(service, [PEM_BODY])
    await service.push_frame(ErrorFrame(error=f"bad {PEM_BODY}"))
    assert PEM_BODY not in seen[0].error


def test_secret_values_cover_inline_key_material_only():
    secrets = sf._google_secret_values(
        SimpleNamespace(credentials=SA, api_key=["AQ.key1", None])
    )
    assert SA in secrets and PEM in secrets and "keyid1234567890" in secrets
    assert "AQ.key1" in secrets
    assert "svc@proj-1.iam.gserviceaccount.com" not in secrets
