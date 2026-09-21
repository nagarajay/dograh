"""Tests for the recording router's marker-expectation streaming path.

Nodes whose prompt never mentions a recording cannot produce a ``▸``/``●``
marker. Holding every reply for marker detection therefore only delays TTS, so
the router is told (via ``markers_expected``) when to stream straight through.
Without that callback it behaves exactly as before.
"""

import time

import pytest
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    MetricsFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame

from api.services.pipecat.recording_audio_cache import RecordingAudio
from api.services.pipecat.recording_router_processor import RecordingRouterProcessor
from api.services.workflow.pipecat_engine_context_composer import (
    RECORDING_MARKER,
    TTS_MARKER,
)
from pipecat.tests import run_test

HOLD_SECS = 0.3


async def _fetch(recording_id: str):
    return RecordingAudio(audio=b"\x00\x01" * 800)


class _Timeline(FrameProcessor):
    """Records when each frame reaches it."""

    def __init__(self):
        super().__init__()
        self.arrivals: list[tuple[float, Frame]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        self.arrivals.append((time.monotonic(), frame))
        await self.push_frame(frame, direction)

    def first(self, kind):
        return next(t for t, f in self.arrivals if isinstance(f, kind))

    def spoken(self) -> str:
        """Text that reaches TTS, whitespace-normalised (the marker path keeps
        the space that followed a marker, which TTS ignores)."""
        return " ".join(
            "".join(
                f.text
                for _, f in self.arrivals
                if isinstance(f, LLMTextFrame) and not f.skip_tts
            ).split()
        )


async def _run(tokens: list[str], **router_kwargs) -> _Timeline:
    router = RecordingRouterProcessor(
        audio_sample_rate=16_000, fetch_recording_audio=_fetch, **router_kwargs
    )
    timeline = _Timeline()
    await run_test(
        Pipeline([router, timeline]),
        frames_to_send=[
            *[LLMTextFrame(text=t) for t in tokens],
            SleepFrame(sleep=HOLD_SECS),
            LLMFullResponseEndFrame(),
        ],
        expected_down_frames=None,
    )
    return timeline


class TestStreamingWhenNoMarkerCanArrive:
    @pytest.mark.asyncio
    async def test_text_reaches_tts_long_before_the_reply_ends(self):
        timeline = await _run(
            ["We offer", " JEE Main, NEET", " and CLAT."],
            markers_expected=lambda: False,
        )

        ahead = timeline.first(LLMFullResponseEndFrame) - timeline.first(LLMTextFrame)
        assert ahead >= HOLD_SECS * 0.8, (
            f"text should stream while the reply is still arriving, got {ahead:.3f}s"
        )
        assert timeline.spoken() == "We offer JEE Main, NEET and CLAT."

    @pytest.mark.asyncio
    async def test_no_hold_is_reported_because_nothing_was_held(self):
        timeline = await _run(["Hello."], markers_expected=lambda: False)

        assert not any(isinstance(f, MetricsFrame) for _, f in timeline.arrivals)

    @pytest.mark.asyncio
    async def test_marker_glyphs_in_a_streamed_reply_are_left_alone(self):
        # Streaming is chosen on the reply's first text; later glyphs are the
        # model's own words and must not be interpreted.
        timeline = await _run(
            ["Plain start ", f"{RECORDING_MARKER} kept"], markers_expected=lambda: False
        )

        assert timeline.spoken() == f"Plain start {RECORDING_MARKER} kept"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("marker", [TTS_MARKER, RECORDING_MARKER])
    async def test_a_reply_that_opens_with_a_marker_is_still_parsed(self, marker):
        timeline = await _run([marker, " Hello there."], markers_expected=lambda: False)

        if marker == TTS_MARKER:
            assert timeline.spoken() == "Hello there."
        else:
            assert timeline.spoken() == ""


class TestDefaultBehaviourUnchanged:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("markers_expected", [None, lambda: True])
    async def test_reply_without_a_marker_is_held_until_it_ends(self, markers_expected):
        timeline = await _run(
            ["We offer", " JEE Main."], markers_expected=markers_expected
        )

        behind = timeline.first(LLMFullResponseEndFrame) - timeline.first(LLMTextFrame)
        assert behind < HOLD_SECS * 0.5, (
            "without a marker the text is only released with the reply's end"
        )
        assert timeline.spoken() == "We offer JEE Main."

    @pytest.mark.asyncio
    async def test_the_hold_is_reported_for_the_latency_breakdown(self):
        timeline = await _run(["We offer", " JEE Main."])

        holds = [
            d
            for _, f in timeline.arrivals
            if isinstance(f, MetricsFrame)
            for d in f.data
            if isinstance(d, TTFBMetricsData)
        ]
        assert len(holds) == 1
        assert holds[0].processor.startswith("RecordingRouterProcessor")
        assert holds[0].value >= HOLD_SECS * 0.8

    @pytest.mark.asyncio
    @pytest.mark.parametrize("markers_expected", [None, lambda: True])
    async def test_marker_is_stripped_and_text_flows(self, markers_expected):
        timeline = await _run(
            [TTS_MARKER, " Hello, how are you today?"],
            markers_expected=markers_expected,
        )

        assert timeline.spoken() == "Hello, how are you today?"


class TestDecisionIsPerReply:
    @pytest.mark.asyncio
    async def test_expectation_is_read_again_for_each_reply(self):
        answers = iter([False, True])
        router = RecordingRouterProcessor(
            audio_sample_rate=16_000,
            fetch_recording_audio=_fetch,
            markers_expected=lambda: next(answers),
        )
        timeline = _Timeline()

        await run_test(
            Pipeline([router, timeline]),
            frames_to_send=[
                LLMTextFrame(text="Streamed reply."),
                LLMFullResponseEndFrame(),
                LLMTextFrame(text=TTS_MARKER),
                LLMTextFrame(text=" Marked reply."),
                LLMFullResponseEndFrame(),
            ],
            expected_down_frames=None,
        )

        assert timeline.spoken() == "Streamed reply. Marked reply."


class TestFirstSentenceReachesTtsEarlier:
    """LLM -> router -> TTS, with the real sentence aggregator in between."""

    REPLY = "We offer JEE Main. The entry requirement is class tenth. Apply online."

    @staticmethod
    async def _seconds_until_tts_starts(markers_expected) -> float:
        from pipecat.frames.frames import LLMContextFrame
        from pipecat.processors.aggregators.llm_context import LLMContext

        from pipecat.tests import MockLLMService, MockTTSService

        class TimedTTS(MockTTSService):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.first_text_at: float | None = None

            async def run_tts(self, text, context_id):
                if self.first_text_at is None and text.strip():
                    self.first_text_at = time.monotonic()
                async for frame in super().run_tts(text, context_id):
                    yield frame

        llm = MockLLMService(
            mock_steps=[
                MockLLMService.create_text_chunks(
                    TestFirstSentenceReachesTtsEarlier.REPLY, chunk_size=6
                )
            ],
            chunk_delay=0.04,
        )
        router = RecordingRouterProcessor(
            audio_sample_rate=16_000,
            fetch_recording_audio=_fetch,
            markers_expected=markers_expected,
        )
        tts = TimedTTS(
            mock_audio_duration_ms=40, frame_delay=0, pause_frame_processing=False
        )

        started = time.monotonic()
        await run_test(
            Pipeline([llm, router, tts]),
            frames_to_send=[
                LLMContextFrame(
                    LLMContext([{"role": "user", "content": "What do you offer?"}])
                )
            ],
            expected_down_frames=None,
        )
        assert tts.first_text_at is not None
        return tts.first_text_at - started

    @pytest.mark.asyncio
    async def test_streaming_starts_tts_well_before_a_held_reply_does(self):
        held = await self._seconds_until_tts_starts(None)
        streamed = await self._seconds_until_tts_starts(lambda: False)

        # Three sentences arrive over ~0.5 s; the first is complete after ~0.2 s
        # but a held reply is only released once all of it has arrived.
        assert held - streamed >= 0.15, (held, streamed)
