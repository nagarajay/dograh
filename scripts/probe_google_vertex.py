"""Minimal live probes for Google Vertex: one LLM call, one TTS clip, one STT
utterance, one embedding. Each probe is a single request (no retries) and logs
model / location / status / latency only - never a credential or a payload.

Credentials are read from the environment, never from arguments:

    VERTEX_API_KEY                   Vertex API key (LLM; others only if catalogued)
    GOOGLE_APPLICATION_CREDENTIALS   path to a service-account JSON file
    VERTEX_PROJECT_ID                required with service account / ADC
    VERTEX_LOCATION                  default: global

Run inside the API image so the pinned SDK versions match runtime, e.g.

    docker run --rm --env-file <(printf 'VERTEX_API_KEY=%s\n' "$(pbpaste)") ...

or export the variable in a shell that is not recorded to history
(``read -rs VERTEX_API_KEY; export VERTEX_API_KEY``).

    python -m scripts.probe_google_vertex [llm] [tts] [stt] [embeddings]
"""

import asyncio
import os
import sys
import time

LOCATION = os.environ.get("VERTEX_LOCATION", "global")
PROJECT = os.environ.get("VERTEX_PROJECT_ID")
API_KEY = os.environ.get("VERTEX_API_KEY")


def _client(*, allow_api_key: bool):
    from google import genai
    from google.oauth2 import service_account

    if allow_api_key and API_KEY:
        return genai.Client(vertexai=True, api_key=API_KEY), "api_key"
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    creds = None
    if path:
        creds = service_account.Credentials.from_service_account_file(
            path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    return (
        genai.Client(
            vertexai=True, credentials=creds, project=PROJECT, location=LOCATION
        ),
        "service_account" if creds else "adc",
    )


def _report(slot, model, auth, started, status):
    ms = int((time.perf_counter() - started) * 1000)
    print(
        f"probe slot={slot} model={model} location={LOCATION} auth={auth} "
        f"status={status} latency_ms={ms}"
    )


async def probe_llm():
    model = os.environ.get("VERTEX_LLM_MODEL", "gemini-3.5-flash-lite")
    client, auth = _client(allow_api_key=True)
    t = time.perf_counter()
    try:
        r = await client.aio.models.generate_content(
            model=model, contents="Reply with the single word: ok"
        )
        _report("llm", model, auth, t, f"ok chars={len(r.text or '')}")
    except Exception as e:
        _report("llm", model, auth, t, f"error={type(e).__name__}")


def _synth(model: str):
    from google.cloud import texttospeech_v1 as tts
    from google.oauth2 import service_account

    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    creds = (
        service_account.Credentials.from_service_account_file(
            path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        if path
        else None
    )
    endpoint = (
        None if LOCATION == "global" else f"{LOCATION}-texttospeech.googleapis.com"
    )
    client = tts.TextToSpeechClient(
        credentials=creds,
        client_options={"api_endpoint": endpoint} if endpoint else None,
    )
    return client.synthesize_speech(
        input=tts.SynthesisInput(text="Hello, this is a short test."),
        voice=tts.VoiceSelectionParams(
            language_code="en-US", name="Kore", model_name=model
        ),
        audio_config=tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16, sample_rate_hertz=24000
        ),
    ).audio_content


async def probe_tts():
    model = os.environ.get("VERTEX_TTS_MODEL", "gemini-2.5-flash-tts")
    t = time.perf_counter()
    try:
        audio = await asyncio.to_thread(_synth, model)
        _report("tts", model, "service_account/adc", t, f"ok bytes={len(audio)}")
        return audio
    except Exception as e:
        _report("tts", model, "service_account/adc", t, f"error={type(e).__name__}")


async def probe_stt(pcm24k: bytes | None):
    from google.genai import types

    model = "gemini-3.5-transcribe-live-preview"
    if not pcm24k:
        print(f"probe slot=stt model={model} status=skipped (no TTS audio to feed)")
        return
    import numpy as np

    samples = np.frombuffer(pcm24k[44:], dtype=np.int16).astype(np.float32)
    idx = np.arange(0, len(samples), 1.5)
    pcm16k = np.interp(idx, np.arange(len(samples)), samples).astype(np.int16).tobytes()

    client, auth = _client(allow_api_key=False)
    config = types.LiveConnectConfig(
        response_modalities=["TEXT"],
        input_audio_transcription=types.AudioTranscriptionConfig(),
    )
    t = time.perf_counter()
    try:
        final = ""
        async with client.aio.live.connect(model=model, config=config) as session:
            for i in range(0, len(pcm16k), 3200):
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=pcm16k[i : i + 3200], mime_type="audio/pcm;rate=16000"
                    )
                )
            await session.send_realtime_input(audio_stream_end=True)
            async for msg in session.receive():
                sc = msg.server_content
                if sc and sc.input_transcription and sc.input_transcription.text:
                    final += sc.input_transcription.text
                    break
        _report("stt", model, auth, t, f"ok chars={len(final)}")
    except Exception as e:
        _report("stt", model, auth, t, f"error={type(e).__name__}")


async def probe_embeddings():
    from google.genai.types import EmbedContentConfig

    model = "gemini-embedding-001"
    client, auth = _client(allow_api_key=True)
    t = time.perf_counter()
    try:
        r = await client.aio.models.embed_content(
            model=model,
            contents=["short probe"],
            config=EmbedContentConfig(output_dimensionality=1536),
        )
        _report("embeddings", model, auth, t, f"ok dims={len(r.embeddings[0].values)}")
    except Exception as e:
        _report("embeddings", model, auth, t, f"error={type(e).__name__}")


async def main(which: list[str]):
    if not (API_KEY or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or PROJECT):
        print("no Vertex credentials in the environment; nothing probed")
        return
    which = which or ["llm", "tts", "stt", "embeddings"]
    audio = None
    if "llm" in which:
        await probe_llm()
    if "tts" in which or "stt" in which:
        audio = await probe_tts()
    if "stt" in which:
        await probe_stt(audio)
    if "embeddings" in which:
        await probe_embeddings()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
