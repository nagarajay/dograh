"""Exercise the actual adapter before pipeline setup, without Google traffic."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai import types

from api.services.gemini_tts_sample_library import (
    SampleGenerationProviderError,
    synthesize_sample_wav,
)
from api.services.pipecat.service_factory import DograhGeminiVertexApiTTSService


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["audio", "empty", "text", "error", "cancel", "timeout"]
)
async def test_uninitialized_adapter_boundary(monkeypatch, outcome):
    closed = []
    client = SimpleNamespace(
        aio=SimpleNamespace(aclose=AsyncMock()), close=lambda: closed.append("client")
    )

    async def chunks():
        try:
            if outcome == "error":
                raise RuntimeError("synthetic stream error")
            if outcome == "cancel":
                raise asyncio.CancelledError()
            if outcome == "timeout":
                await asyncio.sleep(60)
            if outcome == "audio":
                yield types.GenerateContentResponse(
                    candidates=[
                        types.Candidate(
                            content=types.Content(
                                parts=[
                                    types.Part(
                                        inline_data=types.Blob(
                                            data=b"\x01\x00" * 24001,
                                            mime_type="audio/pcm;rate=24000",
                                        )
                                    )
                                ]
                            )
                        )
                    ]
                )
            elif outcome == "text":
                yield types.GenerateContentResponse(
                    candidates=[
                        types.Candidate(
                            content=types.Content(parts=[types.Part(text="non-audio")])
                        )
                    ]
                )
        finally:
            closed.append("stream")

    client.aio.models = SimpleNamespace(
        generate_content_stream=AsyncMock(return_value=chunks())
    )
    monkeypatch.setattr(
        DograhGeminiVertexApiTTSService, "_create_client", lambda *a: client
    )
    call = synthesize_sample_wav(
        api_key="offline",
        project_id="offline",
        location="global",
        model_id="gemini-3.1-flash-tts-preview",
        voice_id="Achernar",
        language="en-IN",
        style_text="warm",
        sample_text="hello",
        context_id="offline",
        sample_rate_hz=24000,
        channels=1,
    )
    if outcome == "audio":
        wav, duration, metadata = await call
        assert len(wav) == 44 + 48002
        assert duration == 24001 / 24000
    else:
        expected = {"cancel": asyncio.CancelledError, "timeout": TimeoutError}.get(
            outcome, SampleGenerationProviderError
        )
        with pytest.raises(expected):
            await asyncio.wait_for(call, timeout=0.05 if outcome == "timeout" else 3)
    assert closed == ["stream", "client"]
    client.aio.aclose.assert_awaited_once()
    config = client.aio.models.generate_content_stream.call_args.kwargs
    assert config["config"].response_modalities == ["AUDIO"]
    assert config["config"].speech_config.language_code == "en-IN"
    assert (
        config["model"]
        == "projects/offline/locations/global/publishers/google/models/gemini-3.1-flash-tts-preview"
    )
