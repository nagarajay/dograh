"""Minimal live probes for Gemini speech on Google Cloud / Vertex.

Answers, per credential type, whether Google accepts it for each Gemini speech
API. Every probe is one request with no retries. Output is model / API / auth /
status / latency / audio format only: never a credential, request or transcript
(beyond a word-match verdict).

Credentials come from the environment, never arguments:

    VERTEX_API_KEY                   Vertex API key (express-mode key, or a Google
                                     Cloud API key from a billed project)
    GOOGLE_APPLICATION_CREDENTIALS   path to a service-account JSON file
    VERTEX_PROJECT_ID                needed with a service account (default: from file)
    VERTEX_LOCATION                  default: global
    GEMINI_TTS_MODEL                 default: gemini-2.5-flash-tts
    GEMINI_STT_MODEL                 default: gemini-3.5-transcribe-live-preview
    GEMINI_TTS_VOICE                 default: Kore

Probes (default: all that the supplied credentials can attempt):

    tts-key        Vertex API streamGenerateContent with the API key
    tts-vertex-sa  Vertex API streamGenerateContent with the service account
    tts-cloud-sa   Cloud Text-to-Speech StreamingSynthesize with the service account
    stt-key        Live API (BidiGenerateContent) with the API key
    stt-sa         Live API with the service account

STT probes feed audio produced by the first TTS probe that succeeds, so the
chain costs one short synthesis plus one short transcription per credential.

    docker run --rm --env VERTEX_API_KEY ... --entrypoint python \
      -v "$PWD/scripts":/app/scripts:ro avsiq-local/dograh-api:latest \
      -m scripts.probe_gemini_speech [probe ...]
"""

import asyncio
import os
import sys
import time
import traceback

import numpy as np

LOCATION = os.environ.get("VERTEX_LOCATION", "global")
PROJECT = os.environ.get("VERTEX_PROJECT_ID")
# With a project set, the key probes send the complete resource
# projects/{project}/locations/{location}/publishers/google/models/{model}
# instead of the SDK's partial express-mode resource.
FULL_RESOURCE = bool(PROJECT) and os.environ.get("VERTEX_FULL_RESOURCE", "0") == "1"
API_KEY = os.environ.get("VERTEX_API_KEY")
KEY_FILE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-tts")
STT_MODEL = os.environ.get("GEMINI_STT_MODEL", "gemini-3.5-transcribe-live-preview")
VOICE = os.environ.get("GEMINI_TTS_VOICE", "Kore")
TEXT = "Hello, this is a short test of the voice pipeline."
EXPECTED = {"hello", "test", "voice", "pipeline"}
TTS_RATE = 24000
STT_RATE = 16000


def _model(model_id: str) -> str:
    if FULL_RESOURCE:
        return f"projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/{model_id}"
    return model_id


def _sa_credentials():
    from google.oauth2 import service_account

    return service_account.Credentials.from_service_account_file(
        KEY_FILE, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )


def _client(auth: str, for_tts: bool = False):
    from google import genai

    if auth == "key":
        from google.genai import types

        # Documented REST for generateContent uses /v1/; keep the SDK default
        # (v1beta1) for the Live websocket, which is what Dograh runs.
        options = types.HttpOptions(api_version="v1") if for_tts else None
        return genai.Client(vertexai=True, api_key=API_KEY, http_options=options)
    creds = _sa_credentials()
    return genai.Client(
        vertexai=True,
        credentials=creds,
        project=os.environ.get("VERTEX_PROJECT_ID") or creds.project_id,
        location=LOCATION,
    )


def _safe(exc: Exception) -> str:
    """Status and a short reason, with any credential value removed."""
    text = str(getattr(exc, "message", None) or exc)[:160].replace("\n", " ")
    for secret in (API_KEY,):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    code = getattr(exc, "code", "")
    frames = " < ".join(
        f"{os.path.basename(f.filename)}:{f.lineno}:{f.name}"
        for f in reversed(traceback.extract_tb(exc.__traceback__)[-3:])
    )
    return f"{type(exc).__name__} code={code} reason={text!r} at={frames}"


def _report(probe, model, api, auth, started, status, **extra):
    ms = int((time.perf_counter() - started) * 1000)
    detail = " ".join(f"{k}={v}" for k, v in extra.items())
    print(
        f"probe {probe} model={model} api={api} auth={auth} location={LOCATION} "
        f"status={status} latency_ms={ms} {detail}"
    )


def _resample(pcm: bytes, src: int, dst: int) -> bytes:
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    idx = np.arange(0, len(samples), src / dst)
    out = np.interp(idx, np.arange(len(samples)), samples)
    return out.astype(np.int16).tobytes()


async def _tts_vertex(probe: str, auth: str) -> bytes | None:
    from google.genai import types

    started = time.perf_counter()
    audio, first_ms = b"", None
    try:
        # Keep a reference: the SDK closes its HTTP session when the client is
        # garbage-collected, and the stream sends its request lazily.
        client = _client(auth, for_tts=True)
        stream = await client.aio.models.generate_content_stream(
            model=_model(TTS_MODEL),
            contents=TEXT,
            config=types.GenerateContentConfig(
                speech_config=types.SpeechConfig(
                    language_code="en-US",
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=VOICE
                        )
                    ),
                )
            ),
        )
        async for chunk in stream:
            for part in (
                (chunk.candidates[0].content.parts or []) if chunk.candidates else []
            ):
                if part.inline_data and part.inline_data.data:
                    if first_ms is None:
                        first_ms = int((time.perf_counter() - started) * 1000)
                    audio += part.inline_data.data
    except Exception as exc:  # noqa: BLE001 - report status, never the request
        _report(
            probe,
            TTS_MODEL,
            "vertex_streamGenerateContent",
            auth,
            started,
            "error",
            why=_safe(exc),
        )
        return None
    _report(
        probe,
        TTS_MODEL,
        "vertex_streamGenerateContent",
        auth,
        started,
        "ok" if audio else "empty",
        format="PCM16-mono",
        sample_rate=TTS_RATE,
        bytes=len(audio),
        audio_seconds=f"{len(audio) / 2 / TTS_RATE:.2f}",
        first_audio_ms=first_ms,
    )
    return audio or None


async def _tts_cloud(probe: str) -> bytes | None:
    from google.api_core.client_options import ClientOptions
    from google.cloud import texttospeech_v1 as tts

    options = (
        ClientOptions(api_endpoint=f"{LOCATION}-texttospeech.googleapis.com")
        if LOCATION != "global"
        else None
    )
    client = tts.TextToSpeechAsyncClient(
        credentials=_sa_credentials(), client_options=options
    )
    config = tts.StreamingSynthesizeConfig(
        voice=tts.VoiceSelectionParams(
            language_code="en-US", name=VOICE, model_name=TTS_MODEL
        ),
        streaming_audio_config=tts.StreamingAudioConfig(
            audio_encoding=tts.AudioEncoding.PCM, sample_rate_hertz=TTS_RATE
        ),
    )

    async def requests():
        yield tts.StreamingSynthesizeRequest(streaming_config=config)
        yield tts.StreamingSynthesizeRequest(
            input=tts.StreamingSynthesisInput(text=TEXT)
        )

    started = time.perf_counter()
    audio, first_ms = b"", None
    try:
        async for response in await client.streaming_synthesize(requests()):
            if response.audio_content and first_ms is None:
                first_ms = int((time.perf_counter() - started) * 1000)
            audio += response.audio_content
    except Exception as exc:  # noqa: BLE001
        _report(
            probe,
            TTS_MODEL,
            "cloud_tts_StreamingSynthesize",
            "service_account",
            started,
            "error",
            why=_safe(exc),
        )
        return None
    _report(
        probe,
        TTS_MODEL,
        "cloud_tts_StreamingSynthesize",
        "service_account",
        started,
        "ok" if audio else "empty",
        format="PCM16-mono",
        sample_rate=TTS_RATE,
        bytes=len(audio),
        audio_seconds=f"{len(audio) / 2 / TTS_RATE:.2f}",
        first_audio_ms=first_ms,
    )
    return audio or None


async def _stt(probe: str, auth: str, pcm24k: bytes | None):
    from google.genai import types

    synthetic = not pcm24k
    if synthetic:
        # No synthesized speech to feed: a 2 s tone still proves whether the
        # credential is accepted and the stream opens, but not transcription.
        t = np.arange(STT_RATE * 2) / STT_RATE
        pcm = (3000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16).tobytes()
    else:
        pcm = _resample(pcm24k, TTS_RATE, STT_RATE)
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.TEXT],
        input_audio_transcription=types.AudioTranscriptionConfig(
            language_codes=["en-US"]
        ),
    )
    started = time.perf_counter()
    interim, text, first_ms = 0, "", None
    try:
        async with _client(auth).aio.live.connect(
            model=_model(STT_MODEL), config=config
        ) as session:
            step = STT_RATE // 10 * 2  # 100 ms chunks, like the live pipeline
            for i in range(0, len(pcm), step):
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=pcm[i : i + step], mime_type=f"audio/pcm;rate={STT_RATE}"
                    )
                )
                await asyncio.sleep(0.05)
            await session.send_realtime_input(audio_stream_end=True)
            try:
                async with asyncio.timeout(10 if synthetic else 20):
                    async for msg in session.receive():
                        sc = msg.server_content
                        if not sc:
                            continue
                        if (
                            sc.interim_input_transcription
                            and sc.interim_input_transcription.text
                        ):
                            interim += 1
                            first_ms = first_ms or int(
                                (time.perf_counter() - started) * 1000
                            )
                        if sc.input_transcription and sc.input_transcription.text:
                            text += " " + sc.input_transcription.text
                            first_ms = first_ms or int(
                                (time.perf_counter() - started) * 1000
                            )
                        if sc.turn_complete or sc.generation_complete:
                            break
            except TimeoutError:
                pass  # stream was accepted but produced no final result in time
    except Exception as exc:  # noqa: BLE001
        _report(
            probe,
            STT_MODEL,
            "live_BidiGenerateContent",
            auth,
            started,
            "error",
            why=_safe(exc),
        )
        return
    words = {w.strip(".,!?").lower() for w in text.split()}
    matched = len(EXPECTED & words)
    _report(
        probe,
        STT_MODEL,
        "live_BidiGenerateContent",
        auth,
        started,
        "accepted_tone_only" if synthetic else "ok" if matched >= 3 else "mismatch",
        input_format=f"PCM16-mono-{STT_RATE}",
        input="synthetic_tone" if synthetic else "synthesized_speech",
        interim_results=interim,
        expected_words_matched=f"{matched}/{len(EXPECTED)}",
        first_result_ms=first_ms,
    )


async def main(which: list[str]):
    if not (API_KEY or KEY_FILE):
        sys.exit("set VERTEX_API_KEY and/or GOOGLE_APPLICATION_CREDENTIALS")
    default = []
    if API_KEY:
        default += ["tts-key", "stt-key"]
    if KEY_FILE:
        default += ["tts-vertex-sa", "tts-cloud-sa", "stt-sa"]
    which = which or default
    audio = None
    for probe in [p for p in which if p.startswith("tts")]:
        result = (
            await _tts_vertex(probe, "key")
            if probe == "tts-key"
            else await _tts_vertex(probe, "service_account")
            if probe == "tts-vertex-sa"
            else await _tts_cloud(probe)
        )
        audio = audio or result
    for probe in [p for p in which if p.startswith("stt")]:
        await _stt(probe, "key" if probe == "stt-key" else "service_account", audio)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
