"""Technical picture of one agent, assembled for super-admin verification.

The question this answers is "what will this agent actually do when it runs",
which is why it reads the definition a run would bind to rather than whatever
the editor last drew: the draft when one exists (the binding a test run gets
from ``use_draft=True``), the published version otherwise.

Secret handling: node payloads go through ``mask_workflow_definition`` and the
model configuration through ``mask_user_config``, and beyond that only named
fields are copied out. Tool ``definition`` documents and document contents are
never read — a tool's HTTP definition can carry authorization headers, so only
its catalog entry is reported.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from api.db import db_client
from api.services.configuration.ai_model_configuration import (
    get_effective_ai_model_configuration_for_workflow,
)
from api.services.configuration.masking import (
    mask_user_config,
    mask_workflow_configurations,
    mask_workflow_definition,
)
from api.services.workflow.dto import NodeType

# Node fields that describe behaviour. Anything not listed here — including any
# field a future node type adds — is left out rather than passed through.
_NODE_NARRATIVE_FIELDS = (
    "prompt",
    "greeting",
    "greeting_type",
    "extraction_enabled",
    "extraction_prompt",
    "extraction_variables",
    "allow_interrupt",
    "add_global_prompt",
    "delayed_start",
    "delayed_start_duration",
    "pre_call_fetch_mode",
    "pre_call_fetch_url",
    "is_start",
    "is_end",
)


@dataclass
class InspectedNode:
    id: str
    type: Optional[str]
    name: Optional[str]
    narrative: dict[str, Any]
    tool_uuids: list[str]
    document_uuids: list[str]
    mcp_tool_filters: Optional[dict[str, Any]]


@dataclass
class InspectedEdge:
    source: Optional[str]
    target: Optional[str]
    label: Optional[str]
    condition: Optional[str]


@dataclass
class InspectedTool:
    tool_uuid: str
    name: Optional[str]
    description: Optional[str]
    category: Optional[str]
    status: Optional[str]
    resolved: bool


@dataclass
class InspectedDocument:
    document_uuid: str
    filename: Optional[str]
    retrieval_mode: Optional[str]
    processing_status: Optional[str]
    total_chunks: Optional[int]
    resolved: bool


@dataclass
class AgentInspection:
    inspected_source: str
    inspected_definition_id: Optional[int]
    inspected_version_number: Optional[int]
    published_definition_id: Optional[int]
    published_version_number: Optional[int]
    published_at: Optional[Any]
    has_unpublished_draft: bool
    global_prompt: Optional[str]
    nodes: list[InspectedNode] = field(default_factory=list)
    edges: list[InspectedEdge] = field(default_factory=list)
    tools: list[InspectedTool] = field(default_factory=list)
    documents: list[InspectedDocument] = field(default_factory=list)
    model_configuration: dict[str, Any] = field(default_factory=dict)
    workflow_configurations: dict[str, Any] = field(default_factory=dict)
    template_context_variables: dict[str, Any] = field(default_factory=dict)


def _narrative(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: data[key] for key in _NODE_NARRATIVE_FIELDS if data.get(key) is not None
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


async def get_agent_inspection(*, workflow, organization_id: int) -> AgentInspection:
    """Assemble the secret-free technical state of one agent."""
    draft = await db_client.get_draft_version(workflow.id)
    published = workflow.released_definition

    inspected = draft or published
    inspected_source = "draft" if draft is not None else "published"

    definition_json = (
        getattr(inspected, "workflow_json", None)
        if inspected is not None
        else workflow.workflow_definition
    )
    if inspected is None:
        inspected_source = "legacy_workflow_definition"

    masked_definition = mask_workflow_definition(definition_json) or {}
    workflow_configurations = (
        getattr(inspected, "workflow_configurations", None)
        if inspected is not None
        else workflow.workflow_configurations
    ) or {}

    nodes: list[InspectedNode] = []
    tool_uuids: list[str] = []
    document_uuids: list[str] = []
    global_prompt: Optional[str] = None

    for raw_node in masked_definition.get("nodes", []) or []:
        data = raw_node.get("data") or {}
        node_tool_uuids = _string_list(data.get("tool_uuids"))
        node_document_uuids = _string_list(data.get("document_uuids"))
        tool_uuids.extend(node_tool_uuids)
        document_uuids.extend(node_document_uuids)

        node_type = raw_node.get("type")
        if node_type == NodeType.globalNode.value and data.get("prompt"):
            global_prompt = data.get("prompt")

        nodes.append(
            InspectedNode(
                id=str(raw_node.get("id")),
                type=node_type,
                name=data.get("name"),
                narrative=_narrative(data),
                tool_uuids=node_tool_uuids,
                document_uuids=node_document_uuids,
                mcp_tool_filters=data.get("mcp_tool_filters")
                if isinstance(data.get("mcp_tool_filters"), dict)
                else None,
            )
        )

    edges = [
        InspectedEdge(
            source=edge.get("source"),
            target=edge.get("target"),
            label=(edge.get("data") or {}).get("label") or edge.get("label"),
            condition=(edge.get("data") or {}).get("condition"),
        )
        for edge in masked_definition.get("edges", []) or []
    ]

    tools: list[InspectedTool] = []
    unique_tool_uuids = list(dict.fromkeys(tool_uuids))
    if unique_tool_uuids:
        # Resolved through the organization-scoped lookup, so a definition that
        # references a tool from another tenant reports as unresolved rather
        # than leaking that tool's name.
        resolved_tools = await db_client.get_tools_by_uuids(
            unique_tool_uuids, organization_id
        )
        by_uuid = {tool.tool_uuid: tool for tool in resolved_tools}
        for uuid_value in unique_tool_uuids:
            tool = by_uuid.get(uuid_value)
            tools.append(
                InspectedTool(
                    tool_uuid=uuid_value,
                    name=tool.name if tool else None,
                    description=tool.description if tool else None,
                    category=tool.category if tool else None,
                    status=tool.status if tool else None,
                    resolved=tool is not None,
                )
            )

    documents: list[InspectedDocument] = []
    for uuid_value in list(dict.fromkeys(document_uuids)):
        document = await db_client.get_document_by_uuid(uuid_value, organization_id)
        documents.append(
            InspectedDocument(
                document_uuid=uuid_value,
                filename=document.filename if document else None,
                retrieval_mode=document.retrieval_mode if document else None,
                processing_status=document.processing_status if document else None,
                total_chunks=document.total_chunks if document else None,
                resolved=document is not None,
            )
        )

    effective_model_configuration = (
        await get_effective_ai_model_configuration_for_workflow(
            organization_id=organization_id,
            workflow_configurations=workflow_configurations,
        )
    )

    return AgentInspection(
        inspected_source=inspected_source,
        inspected_definition_id=getattr(inspected, "id", None),
        inspected_version_number=getattr(inspected, "version_number", None),
        published_definition_id=getattr(published, "id", None),
        published_version_number=getattr(published, "version_number", None),
        published_at=getattr(published, "published_at", None),
        has_unpublished_draft=draft is not None,
        global_prompt=global_prompt,
        nodes=nodes,
        edges=edges,
        tools=tools,
        documents=documents,
        model_configuration=mask_user_config(effective_model_configuration),
        workflow_configurations=mask_workflow_configurations(workflow_configurations)
        or {},
        template_context_variables=(
            getattr(inspected, "template_context_variables", None)
            if inspected is not None
            else workflow.template_context_variables
        )
        or {},
    )
