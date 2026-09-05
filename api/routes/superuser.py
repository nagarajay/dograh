import json
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field

from api.db import db_client
from api.db.models import UserModel
from api.enums import CallType, WorkflowRunMode
from api.services.auth.depends import get_superuser
from api.services.auth.platform_admin import require_platform_admin
from api.services.auth.stack_auth import (
    StackAuthSessionError,
    StackAuthUserSearchError,
    stackauth,
)
from api.services.superuser.agent_inspection import get_agent_inspection
from api.services.superuser.org_health import (
    get_organization_operational_state,
)
from api.services.superuser.provisioning import (
    OrganizationNotFound,
    ProvisioningConflict,
    mint_organization_api_key,
    provision_client_organization,
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

    Every row is a client tenant: one Dograh organization per AVSIQ client, and
    no platform-internal organizations exist to filter out. A database with no
    clients provisioned therefore lists nothing at all.

    ``display_name`` and ``external_reference`` are Dograh's own columns,
    recorded when the client is provisioned. Both are nullable and are returned
    verbatim, including as ``None``: an organization whose identity was never
    recorded is unnamed, and presenting it as a client called something would
    hide the omission rather than surface it.
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

    # The run's owning user is deliberately the super-admin, not the agent's
    # owner: a super-admin belongs to no client organization, and stamping a
    # customer's user id here would record a call that customer never placed.
    # Organization scope still comes from the workflow (see
    # api/db/workflow_run_client.py), so quota, concurrency, configuration and
    # billing remain the customer's; only the actor is the platform's.
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


# ---------------------------------------------------------------------------
# Platform provisioning
#
# These two endpoints share the /superuser prefix with the console's read APIs
# above, and share nothing else. They are guarded by ``require_platform_admin``
# -- a server-to-server shared secret -- rather than by ``get_superuser``,
# which is an interactive human session and stays exactly as strict as it was.
# Neither endpoint resolves a UserModel, so neither has a caller organization
# that could leak into the tenant it acts on: the target is named explicitly.
# ---------------------------------------------------------------------------


class PlatformOrganizationRequest(BaseModel):
    """Everything the platform must supply to create one client tenant.

    Notably absent: a password. The service account's credential is generated
    inside Dograh, hashed, and discarded, so the provisioning system never
    holds a password it would have to vault, rotate, or leak. What it holds
    afterwards is the organization API key minted by the endpoint below.
    """

    display_name: str = Field(min_length=1, max_length=128)
    external_reference: str = Field(
        min_length=1,
        max_length=128,
        description=(
            "The provisioning system's own identifier for this client. Unique "
            "across organizations, and the key this endpoint is idempotent on."
        ),
    )
    service_email: EmailStr = Field(
        description="Address of the service identity that will own the tenant."
    )


class PlatformOrganizationResponse(BaseModel):
    organization_id: int
    organization_provider_id: str
    display_name: Optional[str]
    external_reference: Optional[str]
    service_user_id: int
    service_user_email: Optional[str]
    service_user_provider_id: str
    #: False when the request matched an existing organization. The rest of the
    #: response is the same either way, so a caller that lost its bookkeeping
    #: can re-send and recover the mapping.
    created: bool
    #: Whether managed model configuration and SIP connectivity are in place.
    #: False is recoverable -- retry this endpoint, or let the tenant's own
    #: authenticated traffic re-enter bootstrap.
    bootstrapped: bool


class PlatformAPIKeyResponse(BaseModel):
    """The raw key appears here and nowhere else, ever.

    Only the hash is stored. A caller that loses this response can safely mint
    again: the endpoint replaces rather than appends, so retrying costs the
    previous key its validity and nothing else.

    There is no request body. The key's name is reserved and deterministic --
    that is precisely what a retry needs in order to find and replace the
    previous key rather than add another one beside it.
    """

    id: int
    organization_id: int
    name: str
    key_prefix: str
    api_key: str
    created_at: datetime
    #: Keys this call invalidated. Empty on a first mint, one id on a retry.
    #: The caller's evidence that the credential it previously held (and may
    #: have lost) is now dead rather than still live and unaccounted for.
    replaced_key_ids: List[int]
    #: True when this call rotated an existing key rather than issuing a first
    #: one. Purely informational: both outcomes are success, and a caller that
    #: cannot tell which of its attempts got through does not have to care.
    rotated: bool


@router.post(
    "/organizations",
    dependencies=[Depends(require_platform_admin)],
    status_code=status.HTTP_200_OK,
)
async def provision_organization(
    request: PlatformOrganizationRequest,
) -> PlatformOrganizationResponse:
    """Provision one client tenant: service identity, organization, bootstrap.

    Idempotent on ``external_reference``. An exact retry returns the existing
    organization with ``created=false``; a retry whose display name or service
    email contradicts the live tenant is refused with 409 rather than applied,
    because the reference names a client that may already be placing calls.

    Deliberately 200 rather than 201: the same request produces the same
    response whether or not this call was the one that created the row, and a
    status code that flips between retries is a status code callers branch on.
    """
    try:
        provisioned = await provision_client_organization(
            display_name=request.display_name,
            external_reference=request.external_reference,
            service_email=request.service_email,
        )
    except ProvisioningConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    organization = provisioned.organization
    service_user = provisioned.service_user

    return PlatformOrganizationResponse(
        organization_id=organization.id,
        organization_provider_id=organization.provider_id,
        display_name=organization.display_name,
        external_reference=organization.external_reference,
        service_user_id=service_user.id,
        service_user_email=service_user.email,
        service_user_provider_id=service_user.provider_id,
        created=provisioned.created,
        bootstrapped=provisioned.bootstrapped,
    )


@router.post(
    "/organizations/{organization_id}/api-keys",
    dependencies=[Depends(require_platform_admin)],
    status_code=status.HTTP_200_OK,
)
async def mint_platform_api_key(
    organization_id: int,
) -> PlatformAPIKeyResponse:
    """Issue -- or re-issue -- a tenant's provisioning API key, as the platform.

    Safely repeatable. The key carries a reserved name and every call replaces
    the one holding it, atomically, so a caller that retries after a lost
    response ends up with exactly one live credential rather than one live
    credential per attempt. Two concurrent retries collapse to one too: the
    partial unique index behind this refuses a second active row, and the loser
    re-runs against the winner's state.

    The key that comes back is an ordinary tenant credential and carries no
    platform authority whatsoever -- it cannot reach this endpoint, or any other
    endpoint under this prefix.

    200 rather than 201: a retry is the same request as the original and must
    not report a different status depending on which attempt won.
    """
    try:
        api_key, raw_key, archived_ids = await mint_organization_api_key(
            organization_id=organization_id
        )
    except OrganizationNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ProvisioningConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    return PlatformAPIKeyResponse(
        id=api_key.id,
        organization_id=api_key.organization_id,
        name=api_key.name,
        key_prefix=api_key.key_prefix,
        api_key=raw_key,
        created_at=api_key.created_at,
        replaced_key_ids=archived_ids,
        rotated=bool(archived_ids),
    )
