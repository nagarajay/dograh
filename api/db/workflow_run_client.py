import uuid
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload, selectinload

from api.db.base_client import BaseDBClient
from api.db.filters import (
    apply_workflow_run_filters,
    exclude_superadmin_test_runs,
    get_workflow_run_order_clause,
)
from api.db.models import (
    OrganizationModel,
    UserModel,
    WorkflowDefinitionModel,
    WorkflowModel,
    WorkflowRunModel,
)
from api.enums import CallType, StorageBackend
from api.schemas.workflow import WorkflowRunResponseSchema
from api.services.superuser.test_runs import (
    is_superadmin_test_run,
    superadmin_test_run_initiator,
)
from api.services.workflow.run_usage_response import format_public_cost_info
from api.utils.recording_artifacts import get_recording_storage_key


class WorkflowRunClient(BaseDBClient):
    async def create_workflow_run(
        self,
        name: str,
        workflow_id: int,
        mode: str,
        user_id: int,
        call_type: CallType = CallType.OUTBOUND,
        initial_context: dict = None,
        gathered_context: dict = None,
        logs: dict = None,
        campaign_id: int = None,
        queued_run_id: int = None,
        organization_id: int | None = None,
        definition_id: int | None = None,
        extra: dict | None = None,
    ) -> WorkflowRunModel:
        async with self.async_session() as session:
            workflow_query = (
                select(WorkflowModel)
                .options(joinedload(WorkflowModel.user))
                .where(WorkflowModel.id == workflow_id)
            )
            if organization_id is not None:
                workflow_query = workflow_query.where(
                    WorkflowModel.organization_id == organization_id
                )
            elif user_id is not None:
                workflow_query = workflow_query.where(WorkflowModel.user_id == user_id)

            workflow = await session.execute(workflow_query)
            workflow = workflow.scalars().first()
            if not workflow:
                raise ValueError(f"Workflow with ID {workflow_id} not found")

            if definition_id is not None:
                definition_result = await session.execute(
                    select(WorkflowDefinitionModel.id).where(
                        WorkflowDefinitionModel.id == definition_id,
                        WorkflowDefinitionModel.workflow_id == workflow.id,
                    )
                )
                if definition_result.scalar_one_or_none() is None:
                    raise ValueError(
                        f"Workflow definition {definition_id} does not belong to "
                        f"workflow {workflow.id}"
                    )

            # Get the current storage backend based on ENABLE_AWS_S3 flag
            current_backend = StorageBackend.get_current_backend()

            new_run = WorkflowRunModel(
                name=name,
                workflow=workflow,
                mode=mode,
                definition_id=definition_id,
                initial_context=initial_context or {},
                gathered_context=gathered_context or {},
                logs=logs or {},
                campaign_id=campaign_id,
                queued_run_id=queued_run_id,
                storage_backend=current_backend.value,
                call_type=call_type.value,
                extra=extra or {},
            )
            session.add(new_run)
            try:
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise e
            await session.refresh(new_run)
        return new_run

    async def get_all_workflow_runs(self) -> list[WorkflowRunModel]:
        async with self.async_session() as session:
            result = await session.execute(select(WorkflowRunModel))
            return result.scalars().all()

    async def get_workflow_run_with_workflow(
        self, run_id: int
    ) -> WorkflowRunModel | None:
        """Fetch a run and its workflow with no tenant scoping.

        Only for callers that establish authorization from the run itself
        rather than from a caller-supplied organization — the super-admin test
        run check does exactly that. Ordinary lookups must keep using the
        scoped :meth:`get_workflow_run`.
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(WorkflowRunModel)
                .options(joinedload(WorkflowRunModel.workflow))
                .where(WorkflowRunModel.id == run_id)
            )
            return result.scalars().first()

    async def get_last_run_at_for_workflow(self, workflow_id: int):
        """Timestamp of a workflow's most recent run, or None if it never ran."""
        async with self.async_session() as session:
            result = await session.execute(
                select(func.max(WorkflowRunModel.created_at)).where(
                    WorkflowRunModel.workflow_id == workflow_id
                )
            )
            return result.scalar()

    async def get_workflow_runs_for_superadmin(
        self,
        limit: int = 50,
        offset: int = 0,
        filters: Optional[List[Dict[str, Any]]] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "desc",
        organization_id: Optional[int] = None,
    ) -> tuple[list[dict], int]:
        """
        Get paginated workflow runs for superadmin with organization information.
        Returns tuple of (workflow_runs, total_count).

        Args:
            sort_by: Field to sort by ('duration', 'created_at', etc.)
            sort_order: 'asc' or 'desc'
        """
        async with self.async_session() as session:
            # Build base query with joins
            # A run belongs to the organization that owns its workflow. Deriving
            # it from the workflow owner's *currently selected* organization
            # instead mis-attributes every historical run of a user who has
            # since switched organizations, and every run of a user who belongs
            # to more than one. WorkflowModel.organization_id is what the
            # organization-scoped queries elsewhere in this client filter on, so
            # this keeps the super-admin view consistent with them.
            base_query = (
                select(WorkflowRunModel)
                .join(WorkflowModel, WorkflowRunModel.workflow_id == WorkflowModel.id)
                .outerjoin(UserModel, WorkflowModel.user_id == UserModel.id)
                .outerjoin(
                    OrganizationModel,
                    WorkflowModel.organization_id == OrganizationModel.id,
                )
            )

            if organization_id is not None:
                base_query = base_query.where(
                    WorkflowModel.organization_id == organization_id
                )

            # Apply filters
            base_query = apply_workflow_run_filters(base_query, filters)

            # Count total with filters
            count_query = base_query.with_only_columns(func.count(WorkflowRunModel.id))
            count_result = await session.execute(count_query)
            total_count = count_result.scalar()

            # Get paginated results with filters and sorting
            order_clause = get_workflow_run_order_clause(sort_by, sort_order)
            result = await session.execute(
                base_query.options(
                    joinedload(WorkflowRunModel.workflow).joinedload(
                        WorkflowModel.user
                    ),
                    joinedload(WorkflowRunModel.workflow).joinedload(
                        WorkflowModel.organization
                    ),
                )
                .order_by(order_clause)
                .limit(limit)
                .offset(offset)
            )
            workflow_runs = result.scalars().all()

            # Format the response
            formatted_runs = []
            for run in workflow_runs:
                organization = run.workflow.organization if run.workflow else None
                formatted_runs.append(
                    {
                        "id": run.id,
                        "name": run.name,
                        "workflow_id": run.workflow_id,
                        "workflow_name": run.workflow.name if run.workflow else None,
                        "user_id": run.workflow.user_id if run.workflow else None,
                        "organization_id": organization.id if organization else None,
                        "organization_name": (
                            organization.provider_id if organization else None
                        ),
                        "mode": run.mode,
                        "is_completed": run.is_completed,
                        "recording_url": run.recording_url,
                        "transcript_url": run.transcript_url,
                        "user_recording_url": get_recording_storage_key(
                            run.extra, "user"
                        ),
                        "bot_recording_url": get_recording_storage_key(
                            run.extra, "bot"
                        ),
                        "usage_info": run.usage_info,
                        "cost_info": run.cost_info,
                        "initial_context": run.initial_context,
                        "gathered_context": run.gathered_context,
                        "created_at": run.created_at,
                        "is_superadmin_test": is_superadmin_test_run(run.extra),
                        "superadmin_initiated_by_user_id": superadmin_test_run_initiator(
                            run.extra
                        ),
                    }
                )

            return formatted_runs, total_count

    async def get_workflow_run(
        self, run_id: int, user_id: int = None, organization_id: int = None
    ) -> WorkflowRunModel | None:
        async with self.async_session() as session:
            query = (
                select(WorkflowRunModel)
                .options(selectinload(WorkflowRunModel.definition))
                .join(WorkflowRunModel.workflow)
            )

            if organization_id:
                # Filter by organization_id when provided
                query = query.where(
                    WorkflowRunModel.id == run_id,
                    WorkflowModel.organization_id == organization_id,
                )
            elif user_id:
                # Fallback to user_id for backwards compatibility
                query = query.where(
                    WorkflowRunModel.id == run_id,
                    WorkflowModel.user_id == user_id,
                )
            else:
                query = query.where(WorkflowRunModel.id == run_id)

            result = await session.execute(query)
            return result.scalars().first()

    async def get_workflow_run_by_id(self, run_id: int) -> WorkflowRunModel | None:
        """Get workflow run by ID without user filtering - for background tasks"""
        async with self.async_session() as session:
            result = await session.execute(
                select(WorkflowRunModel)
                .options(
                    joinedload(WorkflowRunModel.workflow).joinedload(WorkflowModel.user)
                )
                .where(WorkflowRunModel.id == run_id)
            )
            return result.scalars().first()

    async def get_workflow_run_configurations(
        self, run_id: int, organization_id: int
    ) -> dict:
        """Load the immutable workflow configuration snapshot for one run."""

        async with self.async_session() as session:
            result = await session.execute(
                select(WorkflowDefinitionModel.workflow_configurations)
                .join(
                    WorkflowRunModel,
                    WorkflowRunModel.definition_id == WorkflowDefinitionModel.id,
                )
                .join(WorkflowModel, WorkflowRunModel.workflow_id == WorkflowModel.id)
                .where(
                    WorkflowRunModel.id == run_id,
                    WorkflowModel.organization_id == organization_id,
                )
            )
            return result.scalar_one_or_none() or {}

    async def get_organization_id_by_workflow_run_id(
        self, run_id: int | None
    ) -> int | None:
        """Resolve organization_id from a workflow run via workflow.user."""
        if not run_id:
            return None
        async with self.async_session() as session:
            result = await session.execute(
                select(WorkflowModel.organization_id)
                .join(
                    WorkflowRunModel, WorkflowRunModel.workflow_id == WorkflowModel.id
                )
                .where(WorkflowRunModel.id == run_id)
            )
            return result.scalar_one_or_none()

    async def get_workflow_runs_by_workflow_id(
        self,
        workflow_id: int,
        user_id: int = None,
        organization_id: int = None,
        limit: int = 50,
        offset: int = 0,
        filters: Optional[List[Dict[str, Any]]] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = "desc",
    ) -> tuple[list[WorkflowRunResponseSchema], int]:
        async with self.async_session() as session:
            # Build base query
            base_query = (
                select(WorkflowRunModel)
                .join(WorkflowModel, WorkflowRunModel.workflow_id == WorkflowModel.id)
                .where(WorkflowRunModel.workflow_id == workflow_id)
            )

            if organization_id:
                # Filter by organization_id when provided
                base_query = base_query.where(
                    WorkflowModel.organization_id == organization_id
                )
            elif user_id:
                # Fallback to user_id for backwards compatibility
                base_query = base_query.where(WorkflowModel.user_id == user_id)

            # Apply filters
            base_query = apply_workflow_run_filters(base_query, filters)

            # A super admin verifying this agent drives a real run inside the
            # customer's organization. The customer did not place that call, so
            # it must not appear in their own run history — same predicate the
            # usage and report queries already use.
            base_query = exclude_superadmin_test_runs(base_query)

            # Count total with filters
            count_query = base_query.with_only_columns(func.count(WorkflowRunModel.id))
            count_result = await session.execute(count_query)
            total_count = count_result.scalar()

            # Get paginated results with filters and sorting
            order_clause = get_workflow_run_order_clause(sort_by, sort_order)
            result = await session.execute(
                base_query.order_by(order_clause).limit(limit).offset(offset)
            )
            runs = [
                WorkflowRunResponseSchema.model_validate(
                    {
                        "id": run.id,
                        "workflow_id": run.workflow_id,
                        "name": run.name,
                        "mode": run.mode,
                        "created_at": run.created_at,
                        "is_completed": run.is_completed,
                        "recording_url": run.recording_url,
                        "transcript_url": run.transcript_url,
                        "user_recording_url": get_recording_storage_key(
                            run.extra, "user"
                        ),
                        "bot_recording_url": get_recording_storage_key(
                            run.extra, "bot"
                        ),
                        "cost_info": format_public_cost_info(
                            run.cost_info, run.usage_info
                        ),
                        "definition_id": run.definition_id,
                        "initial_context": run.initial_context,
                        "gathered_context": run.gathered_context,
                        "call_type": run.call_type,
                    }
                )
                for run in result.scalars().all()
            ]
            return runs, total_count

    async def update_workflow_run(
        self,
        run_id: int,
        is_completed: bool = False,
        recording_url: str | None = None,
        transcript_url: str | None = None,
        storage_backend: str | None = None,
        usage_info: dict | None = None,
        cost_info: dict | None = None,
        initial_context: dict | None = None,
        gathered_context: dict | None = None,
        logs: dict | None = None,
        state: str | None = None,
        annotations: dict | None = None,
        extra: dict | None = None,
    ) -> WorkflowRunModel:
        async with self.async_session() as session:
            # Use SELECT FOR UPDATE to lock the row during the update
            result = await session.execute(
                select(WorkflowRunModel)
                .where(WorkflowRunModel.id == run_id)
                .with_for_update()
            )
            run = result.scalars().first()
            if not run:
                raise ValueError(f"Workflow run with ID {run_id} not found")
            if recording_url:
                run.recording_url = recording_url
            if transcript_url:
                run.transcript_url = transcript_url
            if storage_backend:
                run.storage_backend = storage_backend
            if usage_info:
                run.usage_info = usage_info
            if cost_info:
                run.cost_info = cost_info
            if initial_context:
                # Merge initial context patches so independent call-start/runtime
                # writers do not erase keys stored earlier in the run lifecycle.
                run.initial_context = {
                    **(run.initial_context or {}),
                    **initial_context,
                }
            if gathered_context:
                # Lets merge the incoming gathered context keys with the existing ones
                run.gathered_context = {
                    **run.gathered_context,
                    **gathered_context,
                }
            if logs:
                # Lets merge the incoming logs key with existing ones
                run.logs = {**run.logs, **logs}
            if annotations:
                run.annotations = {**run.annotations, **annotations}
            if extra:
                run.extra = {**run.extra, **extra}
            if is_completed:
                run.is_completed = is_completed
            if state:
                run.state = state
            try:
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise e
            await session.refresh(run)
        return run

    async def get_workflow_run_with_context(
        self, workflow_run_id: int
    ) -> Tuple[Optional[WorkflowRunModel], Optional[int]]:
        """
        Get workflow run with all related data and return organization_id.

        Returns:
            Tuple of (workflow_run, organization_id) or (None, None) if not found
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(WorkflowRunModel)
                .options(
                    selectinload(WorkflowRunModel.definition),
                    selectinload(WorkflowRunModel.workflow).options(
                        selectinload(WorkflowModel.user),
                        selectinload(WorkflowModel.current_definition),
                    ),
                )
                .where(WorkflowRunModel.id == workflow_run_id)
            )
            workflow_run = result.scalars().first()

            if not workflow_run:
                return None, None

            if not workflow_run.workflow:
                return workflow_run, None

            organization_id = workflow_run.workflow.organization_id
            return workflow_run, organization_id

    async def ensure_public_access_token(self, workflow_run_id: int) -> Optional[str]:
        """Generate a public access token if not exists, return existing if present (idempotent).

        Args:
            workflow_run_id: The ID of the workflow run

        Returns:
            The public access token string, or None if workflow run not found
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(WorkflowRunModel).where(WorkflowRunModel.id == workflow_run_id)
            )
            run = result.scalars().first()
            if not run:
                return None

            # Return existing token if present
            if run.public_access_token:
                return run.public_access_token

            # Generate and persist new token
            token = str(uuid.uuid4())
            run.public_access_token = token

            try:
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise e
            await session.refresh(run)

            return run.public_access_token

    async def get_workflow_run_by_public_token(
        self, token: str
    ) -> Optional[WorkflowRunModel]:
        """Lookup workflow run by public access token.

        Args:
            token: The public access token

        Returns:
            The WorkflowRunModel if found, None otherwise
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(WorkflowRunModel).where(
                    WorkflowRunModel.public_access_token == token
                )
            )
            return result.scalars().first()

    async def get_workflow_run_by_call_id(
        self, call_id: str
    ) -> Optional[WorkflowRunModel]:
        """Find workflow run by call_id stored in gathered_context.

        Args:
            call_id: The telephony call ID to search for

        Returns:
            The WorkflowRunModel if found, None otherwise
        """
        async with self.async_session() as session:
            # Use JSON text extraction to find matching call_id
            # This leverages the idx_workflow_runs_call_id index
            result = await session.execute(
                select(WorkflowRunModel)
                .options(
                    joinedload(WorkflowRunModel.workflow).joinedload(WorkflowModel.user)
                )
                .where(
                    WorkflowRunModel.gathered_context.op("->>")("call_id") == call_id
                )
                .order_by(WorkflowRunModel.created_at.desc())
                .limit(1)
            )
            return result.scalars().first()
