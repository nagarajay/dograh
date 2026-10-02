"""Minimal live probe for Google Cloud Speech-to-Text v2 and Text-to-Speech.

Two paid requests at most, no retries: one streaming synthesis (a few words) and
one streaming recognition of that same audio. Output is model / location /
status / format / latency only - never a credential, a payload or a transcript
beyond a word-match verdict.

Credentials come from the environment, never from arguments:

    GOOGLE_APPLICATION_CREDENTIALS   path to a service-account JSON file
                                     (Cloud Speech has no API-key path)
    GOOGLE_SPEECH_LOCATION           default: us   (chirp_3 needs us or eu)
    GOOGLE_STT_MODEL                 default: chirp_3
    GOOGLE_TTS_VOICE                 default: en-US-Chirp3-HD-Charon
    GOOGLE_TTS_MODEL                 default: chirp_3_hd   (a Gemini-TTS model id
                                     such as gemini-2.5-flash-tts switches the
                                     request to a Gemini voice, e.g. Kore)

Run inside the API image so SDK versions match runtime:

    docker run --rm -v "$KEYFILE":/key.json:ro -e GOOGLE_APPLICATION_CREDENTIALS=/key.json \
      -v "$PWD/scripts":/app/scripts:ro --entrypoint python avsiq-local/dograh-api:latest \
      -m scripts.probe_google_speech [tts] [stt]

``stt`` alone synthesizes its own input first (one extra TTS request).
"""

import asyncio
import os
import sys
import time

LOCATION = os.environ.get("GOOGLE_SPEECH_LOCATION", "us")
STT_MODEL = os.environ.get("GOOGLE_STT_MODEL", "chirp_3")
TTS_MODEL = os.environ.get("GOOGLE_TTS_MODEL", "chirp_3_hd")
TTS_VOICE = os.environ.get("GOOGLE_TTS_VOICE", "en-US-Chirp3-HD-Charon")
LANGUAGE = os.environ.get("GOOGLE_LANGUAGE", "en-US")
SAMPLE_RATE = 16000
TEXT = "Hello, this is a short test of the voice pipeline."
EXPECTED = {"hello", "test", "voice", "pipeline"}


def _credentials():
    from google.oauth2 import service_account

    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not path:
        sys.exit(
            "GOOGLE_APPLICATION_CREDENTIALS must point at a service-account JSON file"
        )
    return service_account.Credentials.from_service_account_file(
        path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )


def _report(kind, model, started, status, **extra):
    ms = int((time.perf_counter() - started) * 1000)
    detail = " ".join(f"{k}={v}" for k, v in extra.items())
    print(
        f"probe {kind} model={model} location={LOCATION} status={status} latency_ms={ms} {detail}"
    )


def _is_gemini(model: str) -> bool:
    return model.startswith("gemini-")


async def probe_tts() -> bytes:
    from google.api_core.client_options import ClientOptions
    from google.cloud import texttospeech_v1 as tts

    creds = _credentials()
    options = (
        ClientOptions(api_endpoint=f"{LOCATION}-texttospeech.googleapis.com")
        if LOCATION != "global"
        else None
    )
    client = tts.TextToSpeechAsyncClient(credentials=creds, client_options=options)
    voice_kwargs = {"language_code": LANGUAGE, "name": TTS_VOICE}
    if _is_gemini(TTS_MODEL):
        voice_kwargs["model_name"] = TTS_MODEL
    config = tts.StreamingSynthesizeConfig(
        voice=tts.VoiceSelectionParams(**voice_kwargs),
        streaming_audio_config=tts.StreamingAudioConfig(
            audio_encoding=tts.AudioEncoding.PCM, sample_rate_hertz=SAMPLE_RATE
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
    except Exception as exc:  # noqa: BLE001  # status only: never the message, it may quote the request
        _report("tts", TTS_MODEL, started, f"error:{type(exc).__name__}")
        raise SystemExit(1) from None
    seconds = len(audio) / 2 / SAMPLE_RATE
    _report(
        "tts",
        TTS_MODEL,
        started,
        "ok" if audio else "empty",
        voice=TTS_VOICE,
        format="PCM16-mono",
        sample_rate=SAMPLE_RATE,
        bytes=len(audio),
        audio_seconds=f"{seconds:.2f}",
        first_audio_ms=first_ms,
    )
    return audio


async def probe_stt(audio: bytes):
    from google.api_core.client_options import ClientOptions
    from google.cloud import speech_v2
    from google.cloud.speech_v2.types import cloud_speech

    creds = _credentials()
    project = creds.project_id
    options = (
        ClientOptions(api_endpoint=f"{LOCATION}-speech.googleapis.com")
        if LOCATION != "global"
        else None
    )
    client = speech_v2.SpeechAsyncClient(credentials=creds, client_options=options)
    config = cloud_speech.StreamingRecognitionConfig(
        config=cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE,
                audio_channel_count=1,
            ),
            language_codes=[LANGUAGE],
            model=STT_MODEL,
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=True
            ),
        ),
        streaming_features=cloud_speech.StreamingRecognitionFeatures(
            interim_results=True
        ),
    )
    recognizer = f"projects/{project}/locations/{LOCATION}/recognizers/_"

    async def requests():
        yield cloud_speech.StreamingRecognizeRequest(
            recognizer=recognizer, streaming_config=config
        )
        step = SAMPLE_RATE // 10 * 2  # 100 ms chunks, like the live pipeline
        for i in range(0, len(audio), step):
            yield cloud_speech.StreamingRecognizeRequest(audio=audio[i : i + step])
            await asyncio.sleep(0.05)

    started = time.perf_counter()
    interim = final = 0
    first_ms = None
    text = ""
    try:
        async for response in await client.streaming_recognize(requests=requests()):
            for result in response.results:
                if not result.alternatives or not result.alternatives[0].transcript:
                    continue
                if first_ms is None:
                    first_ms = int((time.perf_counter() - started) * 1000)
                if result.is_final:
                    final += 1
                    text += " " + result.alternatives[0].transcript
                else:
                    interim += 1
    except Exception as exc:  # noqa: BLE001
        _report("stt", STT_MODEL, started, f"error:{type(exc).__name__}")
        raise SystemExit(1) from None
    words = {w.strip(".,!?").lower() for w in text.split()}
    matched = len(EXPECTED & words)
    _report(
        "stt",
        STT_MODEL,
        started,
        "ok" if matched >= 3 else "mismatch",
        interim_results=interim,
        final_results=final,
        expected_words_matched=f"{matched}/{len(EXPECTED)}",
        first_result_ms=first_ms,
    )


async def main(which: list[str]):
    which = which or ["tts", "stt"]
    audio = b""
    if "tts" in which or "stt" in which:
        audio = await probe_tts()
    if "stt" in which:
        await probe_stt(audio)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
