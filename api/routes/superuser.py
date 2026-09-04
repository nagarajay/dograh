import json
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from api.db import db_client
from api.db.models import UserModel
from api.enums import CallType, WorkflowRunMode
from api.services.auth.depends import get_superuser
from api.services.auth.stack_auth import (
    StackAuthSessionError,
    StackAuthUserSearchError,
    stackauth,
)
from api.services.superuser.agent_inspection import get_agent_inspection
from api.services.superuser.org_health import (
    get_organization_operational_state,
)
from api.services.superuser.test_runs import superadmin_test_run_extra
from api.services.workflow.run_creation import prepare_workflow_run_inputs

router = APIRouter(prefix="/superuser", tags=["superuser"])


class ImpersonateRequest(BaseModel):
    """Request payload for superadmin impersonation.

    ``provider_user_id``, ``user_id``, or ``email`` may be supplied. If more
    than one is provided, ``provider_user_id`` takes precedence, followed by
    ``user_id`` and then ``email``.
    """

    provider_user_id: str | None = None
    user_id: int | None = None
    email: str | None = None


class ImpersonateResponse(BaseModel):
    refresh_token: str
    access_token: str


class SuperuserWorkflowRunResponse(BaseModel):
    id: int
    name: str
    workflow_id: int
    workflow_name: Optional[str]
    user_id: Optional[int]
    organization_id: Optional[int]
    organization_name: Optional[str]
    mode: str
    is_completed: bool
    recording_url: Optional[str]
    transcript_url: Optional[str]
    usage_info: Optional[dict]
    cost_info: Optional[dict]
    initial_context: Optional[dict]
    gathered_context: Optional[dict]
    created_at: datetime
    is_superadmin_test: bool = False
    superadmin_initiated_by_user_id: Optional[int] = None


class SuperuserWorkflowRunsListResponse(BaseModel):
    workflow_runs: List[SuperuserWorkflowRunResponse]
    total_count: int
    page: int
    limit: int
    total_pages: int


class SuperuserOrganizationSummary(BaseModel):
    """One organization as it appears in the super-admin organization list.

    Identity is the Dograh organization id plus the auth provider's id. No
    display name is resolved: names live in the auth provider, not in Dograh,
    and fetching them would mean an outbound call per row.
    """

    id: int
    provider_id: str
    display_name: Optional[str]
    external_reference: Optional[str]
    created_at: Optional[datetime]
    workflow_count: int
    run_count: int
    user_count: int
    last_run_at: Optional[datetime]


class SuperuserOrganizationsListResponse(BaseModel):
    organizations: List[SuperuserOrganizationSummary]
    total_count: int
    page: int
    limit: int
    total_pages: int


class SuperuserOrganizationUser(BaseModel):
    id: int
    email: Optional[str]
    provider_id: Optional[str]
    is_superuser: bool


class SuperuserTelephonyConfigurationState(BaseModel):
    id: int
    name: str
    provider: str
    is_default_outbound: bool
    inactive: bool
    inactive_since: Optional[datetime]
    inactive_reason: Optional[str]
    phone_number_count: int
    created_at: Optional[datetime]


class SuperuserOrganizationOperationalState(BaseModel):
    """Derived provisioning state. Never carries a stored configuration value."""

    bootstrap_state: str
    bootstrap_updated_at: Optional[datetime]
    model_configuration_present: bool
    model_configuration_updated_at: Optional[datetime]
    model_configuration_last_validated_at: Optional[datetime]
    langfuse_configured: bool
    active_api_key_count: int
    telephony_configurations: List[SuperuserTelephonyConfigurationState]


class SuperuserOrganizationDetailResponse(BaseModel):
    organization: SuperuserOrganizationSummary
    users: List[SuperuserOrganizationUser]
    operational_state: SuperuserOrganizationOperationalState


class SuperuserWorkflowSummary(BaseModel):
    id: int
    workflow_uuid: str
    name: str
    status: str
    folder_id: Optional[int]
    is_published: bool
    total_runs: int
    created_at: Optional[datetime]


class SuperuserWorkflowsListResponse(BaseModel):
    organization_id: int
    workflows: List[SuperuserWorkflowSummary]


class SuperuserWorkflowDetailResponse(BaseModel):
    """Agent-level operational state. Carries no workflow definition body.

    The definition can embed provider configuration and prompt-level secrets,
    so the detail view reports only whether a published version exists and how
    the agent is reachable.
    """

    workflow: SuperuserWorkflowSummary
    organization_id: int
    organization_provider_id: str
    organization_display_name: Optional[str]
    owner_user_id: Optional[int]
    released_definition_id: Optional[int]
    current_definition_id: Optional[int]
    last_run_at: Optional[datetime]
    attached_phone_numbers: List[str]


@router.post("/impersonate")
async def impersonate(
    request: ImpersonateRequest, user: UserModel = Depends(get_superuser)
) -> ImpersonateResponse:
    """Impersonate a user as a super-admin.
    Internally, Stack Auth requires the **provider user ID** (a UUID-ish string)
    to create an impersonation session.
    """

    provider_user_id = (
        request.provider_user_id.strip() if request.provider_user_id else None
    ) or None
    email = request.email.strip().lower() if request.email else None

    # ------------------------------------------------------------------
    # Fallback: resolve provider_user_id from internal ``user_id`` or email.
    # ------------------------------------------------------------------
    if provider_user_id is None:
        if request.user_id is not None:
            db_user = await db_client.get_user_by_id(request.user_id)

            if db_user is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User with ID {request.user_id} not found.",
                )

            provider_user_id = db_user.provider_id
        elif email:
            db_user = await db_client.get_user_by_email(email)

            if db_user is not None:
                provider_user_id = db_user.provider_id
            else:
                try:
                    stack_users = await stackauth.find_users_by_email(email)
                except StackAuthUserSearchError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail="Failed to search Stack Auth users.",
                    ) from exc

                if len(stack_users) == 1 and isinstance(stack_users[0].get("id"), str):
                    provider_user_id = stack_users[0]["id"]
                elif len(stack_users) > 1:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Multiple Stack Auth users matched that email.",
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"User with email {email} not found.",
                    )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "One of 'provider_user_id', 'user_id', or 'email' must be provided."
                ),
            )

    # ------------------------------------------------------------------
    # Call Stack Auth to create the impersonation session
    # ------------------------------------------------------------------
    try:
        session = await stackauth.impersonate(provider_user_id)
    except StackAuthSessionError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to create Stack Auth impersonation session.",
        ) from exc

    if (
        not isinstance(session, dict)
        or "refresh_token" not in session
        or "access_token" not in session
    ):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to create Stack Auth impersonation session.",
        )

    return ImpersonateResponse(
        refresh_token=session["refresh_token"],
        access_token=session["access_token"],
    )


class SuperuserInspectedNode(BaseModel):
    id: str
    type: Optional[str]
    name: Optional[str]
    narrative: dict
    tool_uuids: List[str]
    document_uuids: List[str]
    mcp_tool_filters: Optional[dict]


class SuperuserInspectedEdge(BaseModel):
    source: Optional[str]
    target: Optional[str]
    label: Optional[str]
    condition: Optional[str]


class SuperuserInspectedTool(BaseModel):
    tool_uuid: str
    name: Optional[str]
    description: Optional[str]
    category: Optional[str]
    status: Optional[str]
    resolved: bool


class SuperuserInspectedDocument(BaseModel):
    document_uuid: str
    filename: Optional[str]
    retrieval_mode: Optional[str]
    processing_status: Optional[str]
    total_chunks: Optional[int]
    resolved: bool


class SuperuserAgentInspectionResponse(BaseModel):
    """What the agent will actually do, with every secret masked or omitted."""

    workflow: SuperuserWorkflowSummary
    organization_id: int
    inspected_source: str
    inspected_definition_id: Optional[int]
    inspected_version_number: Optional[int]
    published_definition_id: Optional[int]
    published_version_number: Optional[int]
    published_at: Optional[datetime]
    has_unpublished_draft: bool
    global_prompt: Optional[str]
    nodes: List[SuperuserInspectedNode]
    edges: List[SuperuserInspectedEdge]
    tools: List[SuperuserInspectedTool]
    documents: List[SuperuserInspectedDocument]
    model_configuration: dict
    workflow_configurations: dict
    template_context_variables: dict
    attached_phone_numbers: List[str]


class SuperuserTestRunRequest(BaseModel):
    name: str | None = None


class SuperuserTestRunResponse(BaseModel):
    """A browser test run a super admin may connect to.

    ``organization_id`` is the customer's, not the super admin's: the run
    genuinely executes inside the customer's organization, which is what makes
    it evidence that the agent works there.
    """

    id: int
    workflow_id: int
    organization_id: int
    name: str
    mode: str
    definition_id: Optional[int]
    is_superadmin_test: bool
    superadmin_initiated_by_user_id: Optional[int]
    created_at: datetime


def _workflow_summary(workflow, run_count: int) -> SuperuserWorkflowSummary:
    return SuperuserWorkflowSummary(
        id=workflow.id,
        workflow_uuid=workflow.workflow_uuid,
        name=workflow.name,
        status=workflow.status,
        folder_id=workflow.folder_id,
        is_published=workflow.released_definition_id is not None,
        total_runs=run_count,
        created_at=workflow.created_at,
    )


@router.get("/organizations")
async def list_organizations(
    page: int = Query(1, ge=1, description="Page number (starts from 1)"),
    limit: int = Query(50, ge=1, le=100, description="Number of items per page"),
    search: Optional[str] = Query(
        None, description="Case-insensitive match against the provider organization id"
    ),
    user: UserModel = Depends(get_superuser),
) -> SuperuserOrganizationsListResponse:
    """List every organization, including ones that have never had a run.

    This is the view that answers "was the organization actually created at the
    Dograh layer" — so an organization with zero workflows and zero runs must
    still be listed.
    """
    offset = (page - 1) * limit
    organizations, total_count = await db_client.list_organizations_for_superadmin(
        limit=limit,
        offset=offset,
        search=search.strip() if search else None,
    )
    total_pages = (total_count + limit - 1) // limit

    return SuperuserOrganizationsListResponse(
        organizations=[SuperuserOrganizationSummary(**org) for org in organizations],
        total_count=total_count,
        page=page,
        limit=limit,
        total_pages=total_pages,
    )


async def _load_organization_summary(
    organization_id: int,
) -> SuperuserOrganizationSummary:
    organizations, _ = await db_client.list_organizations_for_superadmin(
        limit=1, offset=0, organization_id=organization_id
    )
    if not organizations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Organization {organization_id} not found.",
        )
    return SuperuserOrganizationSummary(**organizations[0])


@router.get("/organizations/{organization_id}")
async def get_organization(
    organization_id: int,
    user: UserModel = Depends(get_superuser),
) -> SuperuserOrganizationDetailResponse:
    """Operational detail for one organization. Exposes no stored secrets."""
    summary = await _load_organization_summary(organization_id)
    members = await db_client.get_organization_users(organization_id)
    state = await get_organization_operational_state(organization_id)

    return SuperuserOrganizationDetailResponse(
        organization=summary,
        users=[
            SuperuserOrganizationUser(
                id=member.id,
                email=member.email,
                provider_id=member.provider_id,
                is_superuser=bool(member.is_superuser),
            )
            for member in members
        ],
        operational_state=SuperuserOrganizationOperationalState(
            bootstrap_state=state.bootstrap_state,
            bootstrap_updated_at=state.bootstrap_updated_at,
            model_configuration_present=state.model_configuration_present,
            model_configuration_updated_at=state.model_configuration_updated_at,
            model_configuration_last_validated_at=(
                state.model_configuration_last_validated_at
            ),
            langfuse_configured=state.langfuse_configured,
            active_api_key_count=state.active_api_key_count,
            telephony_configurations=[
                SuperuserTelephonyConfigurationState(
                    id=configuration.id,
                    name=configuration.name,
                    provider=configuration.provider,
                    is_default_outbound=configuration.is_default_outbound,
                    inactive=configuration.inactive,
                    inactive_since=configuration.inactive_since,
                    inactive_reason=configuration.inactive_reason,
                    phone_number_count=configuration.phone_number_count,
                    created_at=configuration.created_at,
                )
                for configuration in state.telephony_configurations
            ],
        ),
    )


@router.get("/organizations/{organization_id}/workflows")
async def list_organization_workflows(
    organization_id: int,
    user: UserModel = Depends(get_superuser),
) -> SuperuserWorkflowsListResponse:
    """List an organization's agents, whether or not they have ever run."""
    await _load_organization_summary(organization_id)

    workflows = await db_client.get_all_workflows(organization_id=organization_id)
    run_counts = await db_client.get_workflow_run_counts(
        [workflow.id for workflow in workflows]
    )

    return SuperuserWorkflowsListResponse(
        organization_id=organization_id,
        workflows=[
            _workflow_summary(workflow, run_counts.get(workflow.id, 0))
            for workflow in sorted(workflows, key=lambda item: item.id, reverse=True)
        ],
    )


@router.get("/organizations/{organization_id}/workflows/{workflow_id}")
async def get_organization_workflow(
    organization_id: int,
    workflow_id: int,
    user: UserModel = Depends(get_superuser),
) -> SuperuserWorkflowDetailResponse:
    """Provisioning state of one agent, with no definition body and no secrets."""
    summary = await _load_organization_summary(organization_id)

    workflow = await db_client.get_workflow(
        workflow_id, organization_id=organization_id
    )
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found in organization {organization_id}.",
        )

    run_counts = await db_client.get_workflow_run_counts([workflow.id])
    last_run_at = await db_client.get_last_run_at_for_workflow(workflow.id)
    phone_numbers = await db_client.list_inbound_addresses_for_workflow(
        organization_id=organization_id, workflow_id=workflow.id
    )

    return SuperuserWorkflowDetailResponse(
        workflow=_workflow_summary(workflow, run_counts.get(workflow.id, 0)),
        organization_id=organization_id,
        organization_provider_id=summary.provider_id,
        organization_display_name=summary.display_name,
        owner_user_id=workflow.user_id,
        released_definition_id=workflow.released_definition_id,
        current_definition_id=workflow.current_definition_id,
        last_run_at=last_run_at,
        attached_phone_numbers=phone_numbers,
    )


@router.get("/organizations/{organization_id}/workflows/{workflow_id}/inspection")
async def inspect_organization_workflow(
    organization_id: int,
    workflow_id: int,
    user: UserModel = Depends(get_superuser),
) -> SuperuserAgentInspectionResponse:
    """The agent's technical truth: instructions, knowledge, tools, models, channels.

    Reads the definition a run would bind to — the draft when one exists, since
    that is what the test flow uses — and reports the published version
    alongside it so draft-versus-live drift is visible.
    """
    await _load_organization_summary(organization_id)

    workflow = await db_client.get_workflow(
        workflow_id, organization_id=organization_id
    )
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found in organization {organization_id}.",
        )

    inspection = await get_agent_inspection(
        workflow=workflow, organization_id=organization_id
    )
    run_counts = await db_client.get_workflow_run_counts([workflow.id])
    phone_numbers = await db_client.list_inbound_addresses_for_workflow(
        organization_id=organization_id, workflow_id=workflow.id
    )

    return SuperuserAgentInspectionResponse(
        workflow=_workflow_summary(workflow, run_counts.get(workflow.id, 0)),
        organization_id=organization_id,
        inspected_source=inspection.inspected_source,
        inspected_definition_id=inspection.inspected_definition_id,
        inspected_version_number=inspection.inspected_version_number,
        published_definition_id=inspection.published_definition_id,
        published_version_number=inspection.published_version_number,
        published_at=inspection.published_at,
        has_unpublished_draft=inspection.has_unpublished_draft,
        global_prompt=inspection.global_prompt,
        nodes=[SuperuserInspectedNode(**vars(node)) for node in inspection.nodes],
        edges=[SuperuserInspectedEdge(**vars(edge)) for edge in inspection.edges],
        tools=[SuperuserInspectedTool(**vars(tool)) for tool in inspection.tools],
        documents=[
            SuperuserInspectedDocument(**vars(document))
            for document in inspection.documents
        ],
        model_configuration=inspection.model_configuration,
        workflow_configurations=inspection.workflow_configurations,
        template_context_variables=inspection.template_context_variables,
        attached_phone_numbers=phone_numbers,
    )


@router.post("/organizations/{organization_id}/workflows/{workflow_id}/test-run")
async def create_superadmin_test_run(
    organization_id: int,
    workflow_id: int,
    request: SuperuserTestRunRequest,
    user: UserModel = Depends(get_superuser),
) -> SuperuserTestRunResponse:
    """Start a browser test of another organization's agent.

    Reuses the same path the workflow editor's own test button takes —
    ``use_draft=True`` — so the run binds to the draft definition and nothing
    is published, released, or otherwise written back to the agent.

    The run is stamped as a super-admin test so it can be excluded from the
    customer's usage and reports, and so the WebRTC signalling socket can let
    this one superuser — and only this one — drive a run outside their own
    organization. Model calls still bill against the customer's managed-service
    key while the test runs; Dograh has no non-billable credential path.
    """
    await _load_organization_summary(organization_id)

    workflow = await db_client.get_workflow(
        workflow_id, organization_id=organization_id
    )
    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found in organization {organization_id}.",
        )

    run_inputs = await prepare_workflow_run_inputs(
        db_client,
        workflow,
        use_draft=True,
        include_template_context=True,
    )

    initial_context = dict(run_inputs.initial_context or {})
    configured_direction = initial_context.get("direction")
    normalized_direction = (
        configured_direction.strip().lower()
        if isinstance(configured_direction, str)
        else None
    )
    call_type = (
        CallType.OUTBOUND
        if normalized_direction == CallType.OUTBOUND.value
        else CallType.INBOUND
    )
    initial_context["direction"] = call_type.value

    run = await db_client.create_workflow_run(
        request.name or f"superadmin-test-{workflow.id}",
        workflow_id,
        WorkflowRunMode.SMALLWEBRTC.value,
        user.id,
        call_type=call_type,
        organization_id=organization_id,
        definition_id=run_inputs.definition_id,
        initial_context=initial_context,
        extra=superadmin_test_run_extra(user.id),
    )

    return SuperuserTestRunResponse(
        id=run.id,
        workflow_id=run.workflow_id,
        organization_id=organization_id,
        name=run.name,
        mode=run.mode,
        definition_id=run.definition_id,
        is_superadmin_test=True,
        superadmin_initiated_by_user_id=user.id,
        created_at=run.created_at,
    )


@router.get("/workflow-runs")
async def get_workflow_runs(
    page: int = Query(1, ge=1, description="Page number (starts from 1)"),
    limit: int = Query(50, ge=1, le=100, description="Number of items per page"),
    filters: Optional[str] = Query(None, description="JSON-encoded filter criteria"),
    sort_by: Optional[str] = Query(
        None, description="Field to sort by (e.g., 'duration', 'created_at')"
    ),
    sort_order: Optional[str] = Query(
        "desc", description="Sort order ('asc' or 'desc')"
    ),
    organization_id: Optional[int] = Query(
        None, description="Restrict to runs owned by this organization"
    ),
    user: UserModel = Depends(get_superuser),
) -> SuperuserWorkflowRunsListResponse:
    """
    Get paginated list of all workflow runs with organization information.
    Requires superuser privileges.

    Filters should be provided as a JSON-encoded array of filter criteria.
    Example: [{"field": "id", "type": "number", "value": {"value": 680}}]
    """
    offset = (page - 1) * limit

    # Parse filters if provided
    filter_criteria = None
    if filters:
        try:
            filter_criteria = json.loads(filters)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid filter format")

    # Validate sort_order
    if sort_order not in ("asc", "desc"):
        sort_order = "desc"

    workflow_runs, total_count = await db_client.get_workflow_runs_for_superadmin(
        limit=limit,
        offset=offset,
        filters=filter_criteria,
        sort_by=sort_by,
        sort_order=sort_order,
        organization_id=organization_id,
    )

    total_pages = (total_count + limit - 1) // limit  # Ceiling division

    return SuperuserWorkflowRunsListResponse(
        workflow_runs=[SuperuserWorkflowRunResponse(**run) for run in workflow_runs],
        total_count=total_count,
        page=page,
        limit=limit,
        total_pages=total_pages,
    )
