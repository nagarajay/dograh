"""Engine behaviour added for low-latency voice workflows.

Three independent, opt-in runtime capabilities, each defaulting to the previous
behaviour:

* ``allow_interrupt_after_greeting`` (Start node): only the opening greeting is
  protected from interruption; later replies from that node can be interrupted.
* ``generate_closing_turn=False`` (End node): reaching the End node runs no
  further LLM completion, and the call ends after the goodbye already said.
* ``recording_markers_expected``: whether the current node's prompt can produce a
  response-mode marker, which lets the recording router stream instead of hold.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.mock_transport import MockTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_mute import (
    CallbackUserMuteStrategy,
    MuteUntilFirstBotCompleteUserMuteStrategy,
)
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.enums import EndTaskReason
from pipecat.utils.time import time_now_iso8601

from api.schemas.workflow_configurations import CallDispositionOption
from api.services.pipecat.agent_generation_processor import (
    AgentGenerationProcessor,
)
from api.services.workflow.disposition_extraction import DispositionExtractionService
from api.services.workflow.dto import (
    EdgeDataDTO,
    EndCallNodeData,
    Position,
    ReactFlowDTO,
    RFEdgeDTO,
    RFNodeDTO,
    StartCallNodeData,
)
from api.services.workflow.pipecat_engine import PipecatEngine
from api.services.workflow.pipecat_engine_variable_extractor import (
    VariableExtractionManager,
)
from api.services.workflow.workflow_graph import WorkflowGraph
from api.tests.pipecat_test_utils import run_engine_test_pipeline
from pipecat.tests import MockLLMService, MockTTSService

GREETING = "Hello, thanks for calling."
EDGE = "caller_wants_to_end"


def _workflow(*, start=None, end=None, greeting=GREETING) -> WorkflowGraph:
    start_data = dict(
        name="Start Call",
        prompt="Start prompt",
        is_start=True,
        allow_interrupt=False,
        add_global_prompt=False,
        greeting=greeting,
        greeting_type="text" if greeting else None,
        extraction_enabled=False,
    )
    start_data.update(start or {})
    end_data = dict(
        name="End Call",
        prompt="End prompt",
        is_end=True,
        allow_interrupt=False,
        add_global_prompt=False,
        extraction_enabled=False,
    )
    end_data.update(end or {})
    return WorkflowGraph(
        ReactFlowDTO(
            nodes=[
                RFNodeDTO(
                    id="start",
                    type="startCall",
                    position=Position(x=0, y=0),
                    data=StartCallNodeData(**start_data),
                ),
                RFNodeDTO(
                    id="end",
                    type="endCall",
                    position=Position(x=0, y=200),
                    data=EndCallNodeData(**end_data),
                ),
            ],
            edges=[
                RFEdgeDTO(
                    id="start-end",
                    source="start",
                    target="end",
                    data=EdgeDataDTO(label=EDGE, condition="Caller is done"),
                )
            ],
        )
    )


def _unit_engine(workflow: WorkflowGraph, **kwargs):
    llm = Mock()
    llm.queue_frame = AsyncMock()
    llm._update_settings = AsyncMock()
    task = Mock()
    task.queue_frame = AsyncMock()
    engine = PipecatEngine(
        llm=llm,
        context=LLMContext(),
        workflow=workflow,
        call_context_vars={},
        workflow_run_id=1,
        **kwargs,
    )
    engine.call_worker = task
    return engine


async def _open_start(engine: PipecatEngine, workflow: WorkflowGraph) -> str:
    return await engine.queue_node_opening(
        node_id=workflow.start_node_id,
        previous_node_id=None,
        generate_if_no_greeting=True,
        generate_after_greeting=False,
    )


async def _muted(engine: PipecatEngine, frame: Frame) -> bool:
    return await engine.should_mute_user(frame)


# ---------------------------------------------------------------------------
# allow_interrupt_after_greeting — mute decisions
# ---------------------------------------------------------------------------


class TestGreetingOnlyProtection:
    @pytest.mark.asyncio
    async def test_default_start_node_protects_every_reply(self):
        workflow = _workflow()
        engine = _unit_engine(workflow)
        await engine.set_node(workflow.start_node_id)
        await _open_start(engine, workflow)

        assert await _muted(engine, BotStartedSpeakingFrame()) is True
        await _muted(engine, BotStoppedSpeakingFrame())
        # The node's own reply: still not interruptible, exactly as before.
        assert await _muted(engine, BotStartedSpeakingFrame()) is True

    @pytest.mark.asyncio
    async def test_flag_protects_the_greeting_then_frees_later_replies(self):
        workflow = _workflow(start={"allow_interrupt_after_greeting": True})
        engine = _unit_engine(workflow)
        await engine.set_node(workflow.start_node_id)
        assert await _open_start(engine, workflow) == "greeting"

        assert await _muted(engine, BotStartedSpeakingFrame()) is True
        await _muted(engine, BotStoppedSpeakingFrame())

        assert await _muted(engine, BotStartedSpeakingFrame()) is False

    @pytest.mark.asyncio
    async def test_flag_keeps_the_greeting_protected_until_it_finishes(self):
        workflow = _workflow(start={"allow_interrupt_after_greeting": True})
        engine = _unit_engine(workflow)
        await engine.set_node(workflow.start_node_id)
        await _open_start(engine, workflow)

        # Nothing has played yet, and once it has begun it is still the greeting.
        assert engine._opening_speech_active is True
        assert await _muted(engine, BotStartedSpeakingFrame()) is True
        assert engine._opening_speech_active is True

    @pytest.mark.asyncio
    async def test_flag_without_a_greeting_protects_nothing(self):
        workflow = _workflow(
            start={"allow_interrupt_after_greeting": True}, greeting=None
        )
        engine = _unit_engine(workflow)
        await engine.set_node(workflow.start_node_id)
        assert await _open_start(engine, workflow) == "llm"

        assert await _muted(engine, BotStartedSpeakingFrame()) is False

    @pytest.mark.asyncio
    async def test_flag_does_not_change_a_node_that_already_allows_interruption(self):
        workflow = _workflow(
            start={"allow_interrupt": True, "allow_interrupt_after_greeting": True}
        )
        engine = _unit_engine(workflow)
        await engine.set_node(workflow.start_node_id)
        await _open_start(engine, workflow)

        assert await _muted(engine, BotStartedSpeakingFrame()) is False

    @pytest.mark.asyncio
    async def test_shutdown_mute_still_wins(self):
        workflow = _workflow(start={"allow_interrupt_after_greeting": True})
        engine = _unit_engine(workflow)
        await engine.set_node(workflow.start_node_id)
        await _open_start(engine, workflow)
        await _muted(engine, BotStartedSpeakingFrame())
        await _muted(engine, BotStoppedSpeakingFrame())
        engine.set_mute_pipeline(True)

        assert await _muted(engine, BotStartedSpeakingFrame()) is True


# ---------------------------------------------------------------------------
# allow_interrupt_after_greeting — through a real pipeline
# ---------------------------------------------------------------------------


class _Interruptions(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            self.count += 1
        await self.push_frame(frame, direction)


async def _until(condition, timeout: float = 4.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.005)


def _caller_turn(text: str) -> list[Frame]:
    return [
        UserStartedSpeakingFrame(),
        TranscriptionFrame(text, "caller", time_now_iso8601()),
        UserStoppedSpeakingFrame(),
    ]


async def _run_conversation(allow_after_greeting: bool) -> dict:
    workflow = _workflow(
        start={
            "allow_interrupt_after_greeting": allow_after_greeting,
            "wait_for_user_after_greeting": True,
        }
    )
    llm = MockLLMService(
        mock_steps=[
            MockLLMService.create_text_chunks("First reply"),
            MockLLMService.create_text_chunks("Second reply"),
        ],
        chunk_delay=0.001,
    )
    tts = MockTTSService(mock_audio_duration_ms=700, frame_delay=0)
    transport = MockTransport(
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            audio_out_end_silence_secs=0,
        ),
    )
    context = LLMContext()
    engine = PipecatEngine(
        llm=llm,
        context=context,
        workflow=workflow,
        call_context_vars={},
        workflow_run_id=1,
    )
    aggregators = LLMContextAggregatorPair(
        context,
        assistant_params=LLMAssistantAggregatorParams(),
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[TranscriptionUserTurnStartStrategy()],
                stop=[ExternalUserTurnStopStrategy()],
            ),
            user_mute_strategies=[
                MuteUntilFirstBotCompleteUserMuteStrategy(),
                CallbackUserMuteStrategy(should_mute_callback=engine.should_mute_user),
            ],
        ),
    )
    interruptions = _Interruptions()
    task = PipelineWorker(
        Pipeline(
            [
                transport.input(),
                aggregators.user(),
                llm,
                tts,
                interruptions,
                transport.output(),
                aggregators.assistant(),
            ]
        ),
        params=PipelineParams(),
        enable_rtvi=False,
    )
    engine.call_worker = task
    seen: dict = {}

    async def on_ready() -> None:
        await engine.set_node(workflow.start_node_id)
        await _open_start(engine, workflow)

        # The caller talks over the greeting.
        await asyncio.sleep(0.25)
        await task.queue_frames(_caller_turn("Who is this?"))
        seen["interruptions_over_greeting"] = interruptions.count
        await asyncio.sleep(1.0)
        seen["llm_steps_after_greeting"] = llm.get_current_step()

        # A normal turn, then the caller talks over the agent's reply.
        await task.queue_frames(_caller_turn("What do you offer?"))
        await _until(lambda: engine._bot_is_speaking)
        seen["interruptions_before_barge_in"] = interruptions.count
        await task.queue_frames(_caller_turn("Sorry, one more thing"))
        await asyncio.sleep(0.4)
        seen["interruptions_after_barge_in"] = interruptions.count
        await task.cancel()

    await run_engine_test_pipeline(task, engine, transport, on_ready=on_ready)
    return seen


class TestGreetingOnlyProtectionInAPipeline:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("allow_after_greeting", [False, True])
    async def test_greeting_is_protected_either_way(self, allow_after_greeting):
        seen = await _run_conversation(allow_after_greeting)

        assert seen["interruptions_over_greeting"] == 0
        # The caller's speech over the greeting produced no turn at all.
        assert seen["llm_steps_after_greeting"] == 0

    @pytest.mark.asyncio
    async def test_default_reply_cannot_be_interrupted(self):
        seen = await _run_conversation(allow_after_greeting=False)

        assert (
            seen["interruptions_after_barge_in"]
            == seen["interruptions_before_barge_in"]
        )

    @pytest.mark.asyncio
    async def test_flagged_reply_can_be_interrupted_after_the_greeting(self):
        seen = await _run_conversation(allow_after_greeting=True)

        assert (
            seen["interruptions_after_barge_in"] > seen["interruptions_before_barge_in"]
        )


# ---------------------------------------------------------------------------
# recording_markers_expected
# ---------------------------------------------------------------------------


class TestRecordingMarkersExpected:
    @pytest.mark.asyncio
    async def test_prompt_without_a_recording_expects_no_marker(self):
        workflow = _workflow()
        engine = _unit_engine(workflow, has_recordings=True)
        await engine.set_node(workflow.start_node_id)

        assert engine.recording_markers_expected() is False

    @pytest.mark.asyncio
    async def test_prompt_naming_a_recording_expects_a_marker(self):
        workflow = _workflow(start={"prompt": "Greet. RECORDING_ID: hello-01 plays"})
        engine = _unit_engine(workflow, has_recordings=True)
        await engine.set_node(workflow.start_node_id)

        assert engine.recording_markers_expected() is True

    @pytest.mark.asyncio
    async def test_workflow_without_recordings_expects_none(self):
        workflow = _workflow(start={"prompt": "Greet. RECORDING_ID: hello-01 plays"})
        engine = _unit_engine(workflow, has_recordings=False)
        await engine.set_node(workflow.start_node_id)

        assert engine.recording_markers_expected() is False

    def test_before_any_node_the_router_keeps_detecting(self):
        engine = _unit_engine(_workflow(), has_recordings=True)

        assert engine.recording_markers_expected() is True


# ---------------------------------------------------------------------------
# generate_closing_turn=False — ending on the goodbye already said
# ---------------------------------------------------------------------------


def _steps(*, text: str | None, second: str | None = "SECOND COMPLETION"):
    first = (
        MockLLMService.create_mixed_chunks(
            text=text, function_name=EDGE, arguments={}, tool_call_id="call_end_1"
        )
        if text
        else MockLLMService.create_function_call_chunks(
            function_name=EDGE, arguments={}, tool_call_id="call_end_1"
        )
    )
    steps = [first]
    if second:
        steps.append(MockLLMService.create_text_chunks(second))
    return steps


async def _run_end(workflow: WorkflowGraph, steps, *, audio_ms: int = 300) -> dict:
    llm = MockLLMService(mock_steps=steps, chunk_delay=0.001)
    tts = MockTTSService(mock_audio_duration_ms=audio_ms, frame_delay=0)
    transport = MockTransport(
        generate_audio=True,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            audio_out_end_silence_secs=0,
        ),
    )
    context = LLMContext()
    engine = PipecatEngine(
        llm=llm,
        context=context,
        workflow=workflow,
        call_context_vars={},
        workflow_run_id=1,
        call_dispositions=[
            CallDispositionOption(
                code="caller_done", description="The caller finished the call."
            )
        ],
    )
    aggregators = LLMContextAggregatorPair(
        context,
        assistant_params=LLMAssistantAggregatorParams(),
        user_params=LLMUserAggregatorParams(
            user_mute_strategies=[
                MuteUntilFirstBotCompleteUserMuteStrategy(),
                CallbackUserMuteStrategy(should_mute_callback=engine.should_mute_user),
            ],
        ),
    )
    # The same tap production uses to tell the engine what a generation said.
    callbacks = AgentGenerationProcessor(
        generation_started_callback=engine.create_generation_started_callback(),
        llm_text_frame_callback=engine.handle_llm_text_frame,
    )
    task = PipelineWorker(
        Pipeline(
            [
                transport.input(),
                aggregators.user(),
                llm,
                callbacks,
                tts,
                transport.output(),
                aggregators.assistant(),
            ]
        ),
        params=PipelineParams(),
        enable_rtvi=False,
    )
    engine.call_worker = task
    end_reasons: list[str] = []

    at_end: dict = {}
    original = engine.end_call_with_reason

    async def observing_end_call(call_status: str, abort_immediately: bool = False):
        end_reasons.append(call_status)
        at_end["bot_speaking"] = engine._bot_is_speaking
        at_end["playback_finished"] = engine._speech_playback_finished.is_set()
        at_end["spoken_so_far"] = list(tts.received_texts)
        at_end["closing_in_progress"] = engine.closing_in_progress
        await original(call_status, abort_immediately)

    engine.end_call_with_reason = observing_end_call

    with patch(
        "api.db:db_client.get_organization_id_by_workflow_run_id",
        new_callable=AsyncMock,
        return_value=1,
    ):
        with patch.object(
            VariableExtractionManager,
            "_perform_extraction",
            new_callable=AsyncMock,
            return_value={},
        ):
            with patch.object(
                DispositionExtractionService,
                "extract",
                new_callable=AsyncMock,
                return_value="caller_done",
            ) as extract_disposition:
                await run_engine_test_pipeline(task, engine, transport)

    return {
        "completions": llm.get_current_step(),
        "spoken": [t.strip() for t in tts.received_texts if t.strip()],
        "end_reasons": end_reasons,
        "at_end": at_end,
        "extract_disposition": extract_disposition,
        "gathered": await engine.get_gathered_context(),
    }


class TestEndWithoutClosingTurn:
    @pytest.mark.asyncio
    async def test_default_still_runs_a_second_completion(self):
        result = await _run_end(
            _workflow(), _steps(text="Goodbye!", second="Take care.")
        )

        assert result["completions"] == 2
        assert result["spoken"] == ["Goodbye!", "Take care."]

    @pytest.mark.asyncio
    async def test_opt_out_runs_a_single_completion_and_one_goodbye(self):
        workflow = _workflow(end={"generate_closing_turn": False})

        result = await _run_end(workflow, _steps(text="Goodbye!"))

        assert result["completions"] == 1
        assert result["spoken"] == ["Goodbye!"]
        assert result["end_reasons"] == [EndTaskReason.END_CALL.value]

    @pytest.mark.asyncio
    async def test_goodbye_finishes_playing_before_the_call_is_ended(self):
        workflow = _workflow(end={"generate_closing_turn": False})

        # Long enough that the goodbye is still playing when the tool call lands.
        result = await _run_end(workflow, _steps(text="Goodbye!"), audio_ms=900)

        at_end = result["at_end"]
        assert at_end["spoken_so_far"] and "Goodbye!" in at_end["spoken_so_far"][0]
        assert at_end["bot_speaking"] is False
        assert at_end["playback_finished"] is True

    @pytest.mark.asyncio
    async def test_disposition_and_extraction_still_run(self):
        workflow = _workflow(end={"generate_closing_turn": False})

        result = await _run_end(workflow, _steps(text="Goodbye!"))

        result["extract_disposition"].assert_awaited_once()
        assert result["gathered"]["call_status"] == EndTaskReason.END_CALL.value
        assert result["gathered"]["call_disposition"] == "caller_done"

    @pytest.mark.asyncio
    async def test_silent_turn_gets_the_fallback_message_once(self):
        workflow = _workflow(
            end={
                "generate_closing_turn": False,
                "closing_fallback_message": "Thank you, goodbye.",
            }
        )

        result = await _run_end(workflow, _steps(text=None))

        assert result["completions"] == 1
        assert result["spoken"] == ["Thank you, goodbye."]
        assert result["at_end"]["playback_finished"] is True

    @pytest.mark.asyncio
    async def test_fallback_is_not_added_on_top_of_a_goodbye_already_said(self):
        workflow = _workflow(
            end={
                "generate_closing_turn": False,
                "closing_fallback_message": "Thank you, goodbye.",
            }
        )

        result = await _run_end(workflow, _steps(text="Goodbye!"))

        assert result["spoken"] == ["Goodbye!"]

    @pytest.mark.asyncio
    async def test_silent_turn_without_a_fallback_still_ends_the_call(self):
        workflow = _workflow(end={"generate_closing_turn": False})

        result = await _run_end(workflow, _steps(text=None))

        assert result["completions"] == 1
        assert result["spoken"] == []
        assert result["end_reasons"] == [EndTaskReason.END_CALL.value]

    @pytest.mark.asyncio
    async def test_tool_only_turn_speaks_the_fallback_once_and_marks_the_closing(self):
        # What the live model does when told to call the end tool without speaking.
        workflow = _workflow(
            end={
                "generate_closing_turn": False,
                "closing_fallback_message": "Thank you, goodbye.",
            }
        )

        result = await _run_end(workflow, _steps(text=None))

        assert result["completions"] == 1
        assert result["spoken"] == ["Thank you, goodbye."]
        assert result["at_end"]["closing_in_progress"] is True
        assert result["at_end"]["playback_finished"] is True

    @pytest.mark.asyncio
    async def test_default_end_node_does_not_mark_a_silent_closing(self):
        result = await _run_end(
            _workflow(), _steps(text="Goodbye!", second="Take care.")
        )

        assert result["at_end"]["closing_in_progress"] is False


class TestUserIdleDuringClosing:
    @pytest.mark.asyncio
    async def test_no_still_there_prompt_once_the_closing_has_begun(self):
        from api.services.workflow.pipecat_engine_callbacks import UserIdleHandler

        engine = Mock(closing_in_progress=True)
        engine.is_call_disposed.return_value = False
        aggregator = Mock(push_frame=AsyncMock())

        await UserIdleHandler(engine).handle_idle(aggregator)

        aggregator.push_frame.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_prompt_once_the_call_is_disposed(self):
        from api.services.workflow.pipecat_engine_callbacks import UserIdleHandler

        engine = Mock(closing_in_progress=False)
        engine.is_call_disposed.return_value = True
        aggregator = Mock(push_frame=AsyncMock())

        await UserIdleHandler(engine).handle_idle(aggregator)

        aggregator.push_frame.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_live_call_is_still_asked_if_anyone_is_there(self):
        from api.services.workflow.pipecat_engine_callbacks import UserIdleHandler

        engine = Mock(
            closing_in_progress=False,
            answer_supervisor=None,
            transfer_in_progress=False,
        )
        engine.is_call_disposed.return_value = False
        aggregator = Mock(push_frame=AsyncMock())

        await UserIdleHandler(engine).handle_idle(aggregator)

        aggregator.push_frame.assert_awaited_once()
