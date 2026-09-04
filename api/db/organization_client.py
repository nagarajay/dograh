from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import exists, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.future import select

from api.db.base_client import BaseDBClient
from api.db.models import (
    APIKeyModel,
    OrganizationModel,
    UserModel,
    WorkflowModel,
    WorkflowRunModel,
    organization_users_association,
)
from api.utils.api_key import generate_api_key


class OrganizationClient(BaseDBClient):
    async def get_organization_by_id(
        self, organization_id: int
    ) -> Optional[OrganizationModel]:
        """Get an organization by its ID."""
        async with self.async_session() as session:
            result = await session.execute(
                select(OrganizationModel).where(OrganizationModel.id == organization_id)
            )
            return result.scalars().first()

    async def list_organizations_for_superadmin(
        self,
        limit: int = 50,
        offset: int = 0,
        search: Optional[str] = None,
        organization_id: Optional[int] = None,
    ) -> tuple[list[dict], int]:
        """List every organization with activity counts, for super-admin review.

        Deliberately built on outer joins so an organization that has never had
        a workflow or a run still appears — verifying that an organization was
        created at all is the point of the view, and an inner join would hide
        exactly the case worth checking.

        Counts come from correlated scalar subqueries rather than joined
        aggregates: joining workflows and runs in one query multiplies the rows
        and would inflate both counts.
        """
        workflow_count = (
            select(func.count(WorkflowModel.id))
            .where(WorkflowModel.organization_id == OrganizationModel.id)
            .correlate(OrganizationModel)
            .scalar_subquery()
        )
        run_count = (
            select(func.count(WorkflowRunModel.id))
            .select_from(WorkflowRunModel)
            .join(WorkflowModel, WorkflowRunModel.workflow_id == WorkflowModel.id)
            .where(WorkflowModel.organization_id == OrganizationModel.id)
            .correlate(OrganizationModel)
            .scalar_subquery()
        )
        last_run_at = (
            select(func.max(WorkflowRunModel.created_at))
            .select_from(WorkflowRunModel)
            .join(WorkflowModel, WorkflowRunModel.workflow_id == WorkflowModel.id)
            .where(WorkflowModel.organization_id == OrganizationModel.id)
            .correlate(OrganizationModel)
            .scalar_subquery()
        )
        user_count = (
            select(func.count(organization_users_association.c.user_id))
            .where(
                organization_users_association.c.organization_id == OrganizationModel.id
            )
            .correlate(OrganizationModel)
            .scalar_subquery()
        )

        async with self.async_session() as session:
            base = select(OrganizationModel)
            if organization_id is not None:
                base = base.where(OrganizationModel.id == organization_id)
            if search:
                # Super admins search by whichever identifier they have to hand:
                # the customer name they know, the reference the provisioning
                # system uses, or the raw provider id.
                pattern = f"%{search}%"
                base = base.where(
                    OrganizationModel.provider_id.ilike(pattern)
                    | OrganizationModel.display_name.ilike(pattern)
                    | OrganizationModel.external_reference.ilike(pattern)
                )

            total_count = (
                await session.execute(select(func.count()).select_from(base.subquery()))
            ).scalar() or 0

            result = await session.execute(
                base.add_columns(
                    workflow_count.label("workflow_count"),
                    run_count.label("run_count"),
                    last_run_at.label("last_run_at"),
                    user_count.label("user_count"),
                )
                .order_by(OrganizationModel.id.desc())
                .limit(limit)
                .offset(offset)
            )

            organizations = [
                {
                    "id": org.id,
                    "provider_id": org.provider_id,
                    "display_name": org.display_name,
                    "external_reference": org.external_reference,
                    "created_at": org.created_at,
                    "workflow_count": workflows or 0,
                    "run_count": runs or 0,
                    "last_run_at": last_run,
                    "user_count": users or 0,
                }
                for org, workflows, runs, last_run, users in result.all()
            ]
            return organizations, total_count

    async def get_organization_users(self, organization_id: int) -> list[UserModel]:
        """Get all users linked to an organization (many-to-many)."""
        async with self.async_session() as session:
            result = await session.execute(
                select(UserModel)
                .join(
                    organization_users_association,
                    organization_users_association.c.user_id == UserModel.id,
                )
                .where(
                    organization_users_association.c.organization_id == organization_id
                )
                .order_by(UserModel.id)
            )
            return list(result.scalars().all())

    async def get_or_create_organization_by_provider_id(
        self,
        org_provider_id: str,
        user_id: int,
        display_name: str | None = None,
        external_reference: str | None = None,
    ) -> tuple[OrganizationModel, bool]:
        """Get an existing organization by provider_id or create a new one.

        ``display_name`` and ``external_reference`` are only written when this
        call creates the organization. An existing organization keeps whatever
        identity it already has, so a repeated provisioning call cannot rename
        or re-point a live customer.

        Returns:
            A tuple of (organization, was_created) where was_created is True if the organization
            was created in this call, False if it already existed.
        """
        async with self.async_session() as session:
            # First try to get existing organization
            result = await session.execute(
                select(OrganizationModel).where(
                    OrganizationModel.provider_id == org_provider_id
                )
            )
            organization = result.scalars().first()

            if organization is None:
                # Use PostgreSQL's INSERT ... ON CONFLICT DO NOTHING
                # This is atomic and handles race conditions at the database level

                stmt = insert(OrganizationModel.__table__).values(
                    provider_id=org_provider_id,
                    created_at=datetime.now(timezone.utc),
                    display_name=display_name,
                    external_reference=external_reference,
                )
                # ON CONFLICT DO NOTHING - if another request already inserted, this becomes a no-op
                stmt = stmt.on_conflict_do_nothing(index_elements=["provider_id"])

                result = await session.execute(stmt)
                await session.commit()

                # Check if we actually inserted (rowcount > 0) or if there was a conflict (rowcount == 0)
                was_created = result.rowcount > 0

                # Now fetch the organization (either the one we just created or the one that existed)
                result = await session.execute(
                    select(OrganizationModel).where(
                        OrganizationModel.provider_id == org_provider_id
                    )
                )
                organization = result.scalars().first()

                if organization is None:
                    # This should never happen, but handle it just in case
                    error_msg = f"Failed to create or fetch organization with provider_id {org_provider_id}"
                    raise ValueError(error_msg)

                # Only create API key if we actually created the organization
                if was_created:
                    # Create a default API key for the new organization
                    _, key_hash, key_prefix = generate_api_key()

                    api_key = APIKeyModel(
                        organization_id=organization.id,
                        name="Default API Key",
                        key_hash=key_hash,
                        key_prefix=key_prefix,
                        is_active=True,
                        created_by=user_id,
                    )
                    session.add(api_key)
                    await session.commit()

                await session.refresh(organization)
                return organization, was_created
            return organization, False

    async def get_organization_by_external_reference(
        self, external_reference: str
    ) -> Optional[OrganizationModel]:
        """Resolve the organization a provisioning system knows by its own id."""
        async with self.async_session() as session:
            result = await session.execute(
                select(OrganizationModel).where(
                    OrganizationModel.external_reference == external_reference
                )
            )
            return result.scalars().first()

    async def set_organization_identity(
        self,
        organization_id: int,
        *,
        display_name: str | None,
        external_reference: str | None,
    ) -> OrganizationModel:
        """Set the customer identity of one organization.

        Exists so an organization provisioned before this data was recorded can
        be labelled without a super-admin write path: the caller must already
        be authenticated into the organization it is naming.
        """
        async with self.async_session() as session:
            organization = await session.get(OrganizationModel, organization_id)
            if organization is None:
                raise ValueError(f"Organization {organization_id} not found")

            if display_name is not None:
                organization.display_name = display_name
            if external_reference is not None:
                organization.external_reference = external_reference

            await session.commit()
            await session.refresh(organization)
            return organization

    async def is_user_member_of_organization(
        self, user_id: int, organization_id: int
    ) -> bool:
        """Return True if the user belongs to the given organization."""
        async with self.async_session() as session:
            result = await session.execute(
                select(
                    exists().where(
                        (organization_users_association.c.user_id == user_id)
                        & (
                            organization_users_association.c.organization_id
                            == organization_id
                        )
                    )
                )
            )
            return bool(result.scalar())

    async def add_user_to_organization(
        self, user_id: int, organization_id: int
    ) -> None:
        """Ensure that a user is linked to an organization (many-to-many).

        The association is created only if it does not already exist.
        Uses INSERT ... ON CONFLICT DO NOTHING to handle race conditions.
        """
        async with self.async_session() as session:
            # Use PostgreSQL's INSERT ... ON CONFLICT DO NOTHING
            # This handles race conditions at the database level

            stmt = insert(organization_users_association).values(
                user_id=user_id, organization_id=organization_id
            )
            # ON CONFLICT DO NOTHING - if another request already inserted, this becomes a no-op
            # The primary key constraint on (user_id, organization_id) will trigger the conflict
            stmt = stmt.on_conflict_do_nothing()

            await session.execute(stmt)
            await session.commit()
