"""Turn-stop strategy selection and commit timing for non-realtime calls.

The default strategy commits a caller's turn a fixed time after they stop; the
opt-in ``turn_analyzer`` strategy asks a model whether the utterance is complete
and, if so, commits as soon as the transcript is final. These tests pin:

* the configuration keys that select and tune them, with the long-standing
  behaviour as the default;
* that the application adds no wait of its own on top of the configured one;
* that a pause shorter than the timeout does not commit a turn (hesitation).

They use short timeouts and the strategies' own state machines, never a live STT
or LLM, so they are deterministic.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest
from pipecat.audio.turn.base_turn_analyzer import (
    BaseTurnAnalyzer,
    BaseTurnParams,
    EndOfTurnState,
)
from pipecat.frames.frames import (
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.user_stop import (
    ExternalUserTurnStopStrategy,
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

from api.schemas.workflow_configurations import DEFAULT_USER_SPEECH_TIMEOUT_SECS
from api.services.pipecat.run_pipeline import (
    _create_non_realtime_user_turn_stop_strategies,
    _resolve_user_speech_timeout,
)

TIMEOUT = 0.15
SLACK = 0.12


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestUserSpeechTimeoutConfiguration:
    def test_default_is_the_long_standing_600_ms(self):
        assert DEFAULT_USER_SPEECH_TIMEOUT_SECS == 0.6
        assert _resolve_user_speech_timeout({}) == 0.6

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(0.8, 0.8), ("0.9", 0.9), (1, 1.0), (0.2, 0.2), (3.0, 3.0)],
    )
    def test_valid_values_are_used(self, raw, expected):
        assert (
            _resolve_user_speech_timeout({"user_speech_timeout_secs": raw}) == expected
        )

    @pytest.mark.parametrize(
        "raw", [None, True, "soon", [], 0.05, 0.19, 3.01, 60, float("nan")]
    )
    def test_unset_or_unusable_values_keep_the_default(self, raw):
        assert (
            _resolve_user_speech_timeout({"user_speech_timeout_secs": raw})
            == DEFAULT_USER_SPEECH_TIMEOUT_SECS
        )

    def test_default_strategy_is_speech_timeout_at_600_ms(self):
        (strategy,) = _create_non_realtime_user_turn_stop_strategies(
            {}, uses_external_turns=False
        )

        assert type(strategy) is SpeechTimeoutUserTurnStopStrategy
        assert strategy._user_speech_timeout == 0.6

    def test_configured_timeout_reaches_the_strategy(self):
        (strategy,) = _create_non_realtime_user_turn_stop_strategies(
            {"user_speech_timeout_secs": 0.9}, uses_external_turns=False
        )

        assert strategy._user_speech_timeout == 0.9

    def test_turn_analyzer_is_only_used_when_asked_for(self):
        (strategy,) = _create_non_realtime_user_turn_stop_strategies(
            {"turn_stop_strategy": "turn_analyzer", "smart_turn_stop_secs": 1.5},
            uses_external_turns=False,
        )

        assert isinstance(strategy, TurnAnalyzerUserTurnStopStrategy)
        assert strategy._turn_analyzer.params.stop_secs == 1.5

    def test_external_turns_ignore_both_settings(self):
        (strategy,) = _create_non_realtime_user_turn_stop_strategies(
            {"turn_stop_strategy": "turn_analyzer", "user_speech_timeout_secs": 0.9},
            uses_external_turns=True,
        )

        assert isinstance(strategy, ExternalUserTurnStopStrategy)


# ---------------------------------------------------------------------------
# Commit timing
# ---------------------------------------------------------------------------


@pytest.fixture
async def task_manager():
    manager = TaskManager()
    manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    return manager


def _transcript(text="Hello there.", finalized=False):
    frame = TranscriptionFrame(text=text, user_id="caller", timestamp="")
    frame.finalized = finalized
    return frame


class _Commits:
    def __init__(self, strategy):
        self.times: list[float] = []

        @strategy.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(_strategy, _params):
            self.times.append(time.monotonic())


async def _speech_timeout_strategy(task_manager):
    strategy = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=TIMEOUT)
    await strategy.setup(
        SimpleNamespace(task_manager=task_manager, audio_in_sample_rate=16000)
    )
    # A zero STT budget leaves the speech timeout as the only wait, so any extra
    # delay would be the application's.
    await strategy.process_frame(
        STTMetadataFrame(service_name="test", ttfs_p99_latency=0.0)
    )
    return strategy


class TestSpeechTimeoutCommitTiming:
    @pytest.mark.asyncio
    async def test_commits_after_the_configured_silence_and_no_later(
        self, task_manager
    ):
        strategy = await _speech_timeout_strategy(task_manager)
        commits = _Commits(strategy)

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(_transcript())
        stopped_at = time.monotonic()
        await strategy.process_frame(VADUserStoppedSpeakingFrame())

        await asyncio.sleep(TIMEOUT + SLACK)

        assert len(commits.times) == 1
        waited = commits.times[0] - stopped_at
        assert TIMEOUT * 0.9 <= waited <= TIMEOUT + SLACK

    @pytest.mark.asyncio
    async def test_a_pause_shorter_than_the_timeout_does_not_commit(self, task_manager):
        strategy = await _speech_timeout_strategy(task_manager)
        commits = _Commits(strategy)

        # "I would like to know... [pause] ...about JEE Main."
        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(_transcript("I would like to know"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(TIMEOUT * 0.5)
        await strategy.process_frame(VADUserStartedSpeakingFrame())

        await asyncio.sleep(TIMEOUT + SLACK)
        assert commits.times == [], "the caller resumed; nothing should have committed"

        await strategy.process_frame(_transcript("about JEE Main."))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(TIMEOUT + SLACK)

        assert len(commits.times) == 1

    @pytest.mark.asyncio
    async def test_a_pause_longer_than_the_timeout_does_commit(self, task_manager):
        """The known limit of a silence timeout, stated so nobody expects more."""
        strategy = await _speech_timeout_strategy(task_manager)
        commits = _Commits(strategy)

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(_transcript("Okay. How about I we go with"))
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(TIMEOUT + SLACK)

        assert len(commits.times) == 1


class _FakeAnalyzer(BaseTurnAnalyzer):
    """An analyzer whose verdict is scripted."""

    def __init__(self, verdict: EndOfTurnState):
        super().__init__()
        self._verdict = verdict
        self.analysed = 0

    @property
    def speech_triggered(self) -> bool:
        return False

    @property
    def params(self) -> BaseTurnParams:
        return BaseTurnParams()

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        return EndOfTurnState.INCOMPLETE

    async def analyze_end_of_turn(self):
        self.analysed += 1
        return self._verdict, None

    def clear(self):
        pass


async def _analyzer_strategy(task_manager, verdict):
    analyzer = _FakeAnalyzer(verdict)
    strategy = TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer)
    await strategy.setup(
        SimpleNamespace(task_manager=task_manager, audio_in_sample_rate=16000)
    )
    return strategy, analyzer


class TestTurnAnalyzerCommitTiming:
    @pytest.mark.asyncio
    async def test_complete_utterance_commits_as_soon_as_the_transcript_is_final(
        self, task_manager
    ):
        strategy, analyzer = await _analyzer_strategy(
            task_manager, EndOfTurnState.COMPLETE
        )
        commits = _Commits(strategy)

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(_transcript("Thank you.", finalized=True))
        stopped_at = time.monotonic()
        await strategy.process_frame(VADUserStoppedSpeakingFrame())

        # No silence timer at all: the commit is on the stop frame itself.
        assert len(commits.times) == 1
        assert commits.times[0] - stopped_at < 0.05
        assert analyzer.analysed == 1

    @pytest.mark.asyncio
    async def test_incomplete_utterance_is_not_committed_by_a_final_transcript(
        self, task_manager
    ):
        strategy, _ = await _analyzer_strategy(task_manager, EndOfTurnState.INCOMPLETE)
        commits = _Commits(strategy)

        await strategy.process_frame(VADUserStartedSpeakingFrame())
        await strategy.process_frame(
            _transcript("Okay. How about I we go with", finalized=True)
        )
        await strategy.process_frame(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(TIMEOUT + SLACK)

        assert commits.times == []
