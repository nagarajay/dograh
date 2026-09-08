"""Tests for the tool set published to the LLM on node changes.

The engine has to publish the destination node's tools on every node change,
including when the destination has none: a terminal node that inherits the
previous node's transition tools can still be asked to call a transition that
no longer applies.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from pipecat.processors.aggregators.llm_context import LLMContext, is_given

from api.services.workflow.dto import (
    AgentNodeData,
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


@pytest.fixture
def workflow() -> WorkflowGraph:
    """start --Continue--> agent --Finish--> end."""
    dto = ReactFlowDTO(
        nodes=[
            RFNodeDTO(
                id="start",
                type="startCall",
                position=Position(x=0, y=0),
                data=StartCallNodeData(
                    name="Start",
                    prompt="Start prompt",
                    is_start=True,
                    add_global_prompt=False,
                    extraction_enabled=False,
                ),
            ),
            RFNodeDTO(
                id="agent",
                type="agentNode",
                position=Position(x=0, y=100),
                data=AgentNodeData(
                    name="Agent",
                    prompt="Agent prompt",
                    add_global_prompt=False,
                    extraction_enabled=False,
                ),
            ),
            RFNodeDTO(
                id="end",
                type="endCall",
                position=Position(x=0, y=200),
                data=EndCallNodeData(
                    name="End",
                    prompt="End prompt",
                    is_end=True,
                    add_global_prompt=False,
                    extraction_enabled=False,
                ),
            ),
        ],
        edges=[
            RFEdgeDTO(
                id="e1",
                source="start",
                target="agent",
                data=EdgeDataDTO(label="Continue", condition="Continue"),
            ),
            RFEdgeDTO(
                id="e2",
                source="agent",
                target="end",
                data=EdgeDataDTO(label="Finish", condition="Finish"),
            ),
        ],
    )
    return WorkflowGraph(dto)


def _engine(workflow: WorkflowGraph, context: LLMContext) -> PipecatEngine:
    llm = Mock()
    llm._update_settings = AsyncMock()
    llm.register_function = Mock()
    return PipecatEngine(
        llm=llm,
        context=context,
        workflow=workflow,
        call_context_vars={},
        workflow_run_id=1,
    )


@pytest.mark.asyncio
async def test_moving_to_end_node_clears_transition_tools(workflow: WorkflowGraph):
    context = LLMContext()
    engine = _engine(workflow, context)

    await engine.set_node("agent", emit_transition_event=False)

    assert is_given(context.tools)
    tool_names = [tool.name for tool in context.tools.standard_tools]
    assert tool_names, "Agent node should advertise its outgoing transition"

    await engine.set_node("end", emit_transition_event=False)

    assert not is_given(context.tools), (
        "End node has no outgoing functions, so the previous node's transition "
        "tools must no longer be advertised"
    )


@pytest.mark.asyncio
async def test_node_change_replaces_previous_tool_set(workflow: WorkflowGraph):
    context = LLMContext()
    engine = _engine(workflow, context)

    await engine.set_node("start", emit_transition_event=False)
    start_tools = [tool.name for tool in context.tools.standard_tools]

    await engine.set_node("agent", emit_transition_event=False)
    agent_tools = [tool.name for tool in context.tools.standard_tools]

    assert start_tools != agent_tools
