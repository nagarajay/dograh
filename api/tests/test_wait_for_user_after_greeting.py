"""Tests for the start node's opt-in ``wait_for_user_after_greeting`` flag.

By default the engine queues an LLM generation once the start greeting has
played, so the bot's first turn is not just the greeting. With the flag set the
greeting still plays, but nothing is generated after it: the bot stays silent
until the caller speaks, and that speech takes the normal user-turn path.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
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
from pipecat.tests import MockLLMService, MockTTSService
from pipecat.tests.mock_transport import MockTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_mute import (
    CallbackUserMuteStrategy,
    MuteUntilFirstBotCompleteUserMuteStrategy,
)
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.time import time_now_iso8601

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
from api.services.workflow.workflow_graph import WorkflowGraph
from api.tests.pipecat_test_utils import run_engine_test_pipeline

GREETING = "Hello, thanks for calling."


def _workflow(**start_kwargs) -> WorkflowGraph:
    return WorkflowGraph(
        ReactFlowDTO(
            nodes=[
                RFNodeDTO(
                    id="start",
                    type="startCall",
                    position=Position(x=0, y=0),
                    data=StartCallNodeData(
                        name="Start Call",
                        prompt="Start prompt",
                        is_start=True,
                        allow_interrupt=False,
                        add_global_prompt=False,
                        greeting=GREETING,
                        greeting_type="text",
                        extraction_enabled=False,
                        **start_kwargs,
                    ),
                ),
                RFNodeDTO(
                    id="end",
                    type="endCall",
                    position=Position(x=0, y=200),
                    data=EndCallNodeData(
                        name="End Call",
                        prompt="End prompt",
                        is_end=True,
                        allow_interrupt=False,
                        add_global_prompt=False,
                        extraction_enabled=False,
                    ),
                ),
            ],
            edges=[
                RFEdgeDTO(
                    id="start-end",
                    source="start",
                    target="end",
                    data=EdgeDataDTO(label="End Call", condition="Caller is done"),
                )
            ],
        )
    )


def _engine(workflow: WorkflowGraph):
    llm = Mock()
    llm.queue_frame = AsyncMock()
    task = Mock()
    task.queue_frame = AsyncMock()
    engine = PipecatEngine(
        llm=llm,
        context=LLMContext(),
        workflow=workflow,
        call_context_vars={},
        workflow_run_id=1,
    )
    engine.set_task(task)
    return engine, llm, task


async def _open_start(engine: PipecatEngine, workflow: WorkflowGraph) -> str:
    return await engine.queue_node_opening(
        node_id=workflow.start_node_id,
        previous_node_id=None,
        generate_if_no_greeting=True,
        generate_after_greeting=True,
    )


async def _finish_greeting(engine: PipecatEngine) -> None:
    await engine.should_mute_user(BotStartedSpeakingFrame())
    await engine.should_mute_user(BotStoppedSpeakingFrame())


@pytest.mark.parametrize("start_kwargs", [{}, {"wait_for_user_after_greeting": False}])
@pytest.mark.asyncio
async def test_flag_absent_or_false_still_generates_after_greeting(start_kwargs):
    workflow = _workflow(**start_kwargs)
    engine, llm, task = _engine(workflow)

    assert await _open_start(engine, workflow) == "greeting"

    assert task.queue_frame.await_args.args[0].text == GREETING
    assert engine._post_greeting_generation_task is not None
    await _finish_greeting(engine)
    await engine._post_greeting_generation_task
    llm.queue_frame.assert_awaited_once()


@pytest.mark.asyncio
async def test_flag_true_plays_greeting_but_never_generates():
    workflow = _workflow(wait_for_user_after_greeting=True)
    engine, llm, task = _engine(workflow)

    assert await _open_start(engine, workflow) == "greeting"

    frame = task.queue_frame.await_args.args[0]
    assert isinstance(frame, TTSSpeakFrame)
    assert frame.text == GREETING
    assert frame.append_to_context is True
    assert engine._post_greeting_generation_task is None

    await _finish_greeting(engine)
    await asyncio.sleep(0)
    llm.queue_frame.assert_not_awaited()


@pytest.mark.asyncio
async def test_flag_true_leaves_caller_speech_during_greeting_untouched():
    """Barge-in is the aggregator's job; the flag adds no generation of its own."""
    workflow = _workflow(wait_for_user_after_greeting=True)
    engine, llm, _ = _engine(workflow)

    await _open_start(engine, workflow)
    engine.context.add_message({"role": "user", "content": "Who is this?"})
    await _finish_greeting(engine)
    await asyncio.sleep(0)

    llm.queue_frame.assert_not_awaited()


@pytest.mark.asyncio
async def test_flag_true_does_not_change_node_reached_by_transition():
    workflow = _workflow(wait_for_user_after_greeting=True)
    engine, llm, task = _engine(workflow)

    result = await engine.queue_node_opening(
        node_id=workflow.start_node_id,
        previous_node_id="some-other-node",
        generate_if_no_greeting=True,
    )

    assert result == "greeting"
    assert engine._post_greeting_generation_task is None
    llm.queue_frame.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("wait", "llm_calls_before_caller"), [(False, 1), (True, 0)]
)
async def test_pipeline_caller_speech_after_greeting_runs_normal_llm_turn(
    wait: bool, llm_calls_before_caller: int
):
    workflow = _workflow(wait_for_user_after_greeting=wait)
    llm = MockLLMService(
        mock_steps=[
            MockLLMService.create_text_chunks("First reply"),
            MockLLMService.create_text_chunks("Second reply"),
        ],
        chunk_delay=0.001,
    )
    tts = MockTTSService(mock_audio_duration_ms=40, frame_delay=0)
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
    task = PipelineWorker(
        Pipeline(
            [
                transport.input(),
                aggregators.user(),
                llm,
                tts,
                transport.output(),
                aggregators.assistant(),
            ]
        ),
        params=PipelineParams(),
        enable_rtvi=False,
    )
    engine.set_task(task)
    seen: dict[str, int] = {}

    async def on_ready() -> None:
        await engine.set_node(workflow.start_node_id)
        await _open_start(engine, workflow)
        # Greeting playback plus, when not waiting, the post-greeting generation.
        await asyncio.sleep(1.0)
        seen["before_caller"] = llm.get_current_step()
        await task.queue_frames(
            [
                UserStartedSpeakingFrame(),
                TranscriptionFrame("What do you offer?", "caller", time_now_iso8601()),
                UserStoppedSpeakingFrame(),
            ]
        )
        await asyncio.sleep(1.0)
        seen["after_caller"] = llm.get_current_step()
        await task.cancel()

    await run_engine_test_pipeline(task, engine, transport, on_ready=on_ready)

    assert seen["before_caller"] == llm_calls_before_caller
    assert seen["after_caller"] == llm_calls_before_caller + 1
    assert any(
        m.get("role") == "assistant" and GREETING in str(m.get("content"))
        for m in context.get_messages()
    )
