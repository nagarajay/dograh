"""Persistence for per-workflow model slot settings and provider credentials.

All writes to a (workflow, slot) pair happen under a row lock on its
``workflow_slot_state`` row and are guarded by ``expected_revision`` so a stale
edit cannot overwrite a newer publish.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Callable, Optional

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError

from api.db.base_client import BaseDBClient
from api.db.models import (
    ProviderCredentialModel,
    WorkflowModel,
    WorkflowSlotSettingModel,
    WorkflowSlotStateModel,
)


class SlotRevisionConflict(Exception):
    def __init__(self, current_revision: int):
        super().__init__(f"stale revision; current revision is {current_revision}")
        self.current_revision = current_revision


class SlotStateError(Exception):
    """The requested transition is not valid from the slot's current state."""


class WorkflowSlotClient(BaseDBClient):
    # ---- slot reads -------------------------------------------------------

    async def get_slot_state(
        self, workflow_id: int, slot: str
    ) -> Optional[WorkflowSlotStateModel]:
        async with self.async_session() as session:
            result = await session.execute(
                select(WorkflowSlotStateModel).where(
                    WorkflowSlotStateModel.workflow_id == workflow_id,
                    WorkflowSlotStateModel.slot == slot,
                )
            )
            return result.scalars().first()

    async def list_slot_states(self, workflow_id: int) -> list[WorkflowSlotStateModel]:
        async with self.async_session() as session:
            result = await session.execute(
                select(WorkflowSlotStateModel).where(
                    WorkflowSlotStateModel.workflow_id == workflow_id
                )
            )
            return list(result.scalars().all())

    async def get_slot_setting(
        self, workflow_id: int, slot: str, version: int
    ) -> Optional[WorkflowSlotSettingModel]:
        async with self.async_session() as session:
            result = await session.execute(
                select(WorkflowSlotSettingModel).where(
                    WorkflowSlotSettingModel.workflow_id == workflow_id,
                    WorkflowSlotSettingModel.slot == slot,
                    WorkflowSlotSettingModel.version == version,
                )
            )
            return result.scalars().first()

    async def list_slot_settings(
        self, workflow_id: int, slot: str, limit: int = 50
    ) -> list[WorkflowSlotSettingModel]:
        async with self.async_session() as session:
            result = await session.execute(
                select(WorkflowSlotSettingModel)
                .where(
                    WorkflowSlotSettingModel.workflow_id == workflow_id,
                    WorkflowSlotSettingModel.slot == slot,
                )
                .order_by(WorkflowSlotSettingModel.version.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def list_published_slot_settings(
        self, workflow_id: int
    ) -> list[WorkflowSlotSettingModel]:
        async with self.async_session() as session:
            result = await session.execute(
                select(WorkflowSlotSettingModel)
                .join(
                    WorkflowSlotStateModel,
                    (
                        WorkflowSlotStateModel.workflow_id
                        == WorkflowSlotSettingModel.workflow_id
                    )
                    & (WorkflowSlotStateModel.slot == WorkflowSlotSettingModel.slot)
                    & (
                        WorkflowSlotStateModel.published_version
                        == WorkflowSlotSettingModel.version
                    ),
                )
                .where(WorkflowSlotSettingModel.workflow_id == workflow_id)
            )
            return list(result.scalars().all())

    # ---- slot writes ------------------------------------------------------

    @staticmethod
    async def _lock_state(
        session, workflow_id: int, slot: str
    ) -> WorkflowSlotStateModel:
        await session.execute(
            insert(WorkflowSlotStateModel)
            .values(workflow_id=workflow_id, slot=slot, last_version=0, revision=0)
            .on_conflict_do_nothing(index_elements=["workflow_id", "slot"])
        )
        result = await session.execute(
            select(WorkflowSlotStateModel)
            .where(
                WorkflowSlotStateModel.workflow_id == workflow_id,
                WorkflowSlotStateModel.slot == slot,
            )
            .with_for_update()
        )
        return result.scalars().one()

    @staticmethod
    def _check_revision(state: WorkflowSlotStateModel, expected: int) -> None:
        if state.revision != expected:
            raise SlotRevisionConflict(state.revision)

    async def save_slot_draft(
        self,
        *,
        workflow_id: int,
        slot: str,
        config: dict[str, Any],
        credential_ref: Optional[str],
        credential_version: Optional[int],
        expected_revision: int,
        created_by: Optional[str],
        change_note: Optional[str],
    ) -> WorkflowSlotSettingModel:
        async with self.async_session() as session:
            state = await self._lock_state(session, workflow_id, slot)
            self._check_revision(state, expected_revision)
            if state.draft_version is not None:
                old = await session.execute(
                    select(WorkflowSlotSettingModel).where(
                        WorkflowSlotSettingModel.workflow_id == workflow_id,
                        WorkflowSlotSettingModel.slot == slot,
                        WorkflowSlotSettingModel.version == state.draft_version,
                    )
                )
                previous = old.scalars().first()
                if previous is not None and previous.state == "draft":
                    previous.state = "discarded"
            version = state.last_version + 1
            row = WorkflowSlotSettingModel(
                workflow_id=workflow_id,
                slot=slot,
                version=version,
                state="draft",
                config=config,
                credential_ref=credential_ref,
                credential_version=credential_version,
                validation_status="unvalidated",
                based_on_version=state.published_version,
                origin="edit",
                change_note=change_note,
                created_by=created_by,
            )
            session.add(row)
            state.draft_version = version
            state.last_version = version
            state.revision += 1
            state.updated_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(row)
            return row

    async def mark_slot_validation(
        self, *, workflow_id: int, slot: str, version: int, status: str
    ) -> Optional[WorkflowSlotSettingModel]:
        """Record a validation result on a draft. No-op if the draft moved on."""
        async with self.async_session() as session:
            state = await self._lock_state(session, workflow_id, slot)
            if state.draft_version != version:
                return None
            result = await session.execute(
                select(WorkflowSlotSettingModel).where(
                    WorkflowSlotSettingModel.workflow_id == workflow_id,
                    WorkflowSlotSettingModel.slot == slot,
                    WorkflowSlotSettingModel.version == version,
                )
            )
            row = result.scalars().one()
            row.validation_status = status
            row.validated_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(row)
            return row

    async def publish_slot(
        self, *, workflow_id: int, slot: str, version: int, expected_revision: int
    ) -> WorkflowSlotSettingModel:
        async with self.async_session() as session:
            state = await self._lock_state(session, workflow_id, slot)
            self._check_revision(state, expected_revision)
            if state.draft_version != version:
                raise SlotStateError(f"version {version} is not the current draft")
            result = await session.execute(
                select(WorkflowSlotSettingModel).where(
                    WorkflowSlotSettingModel.workflow_id == workflow_id,
                    WorkflowSlotSettingModel.slot == slot,
                    WorkflowSlotSettingModel.version.in_(
                        [v for v in (version, state.published_version) if v is not None]
                    ),
                )
            )
            rows = {r.version: r for r in result.scalars().all()}
            draft = rows[version]
            if draft.state != "draft":
                raise SlotStateError(f"version {version} is not a draft")
            if draft.validation_status != "valid":
                raise SlotStateError("draft has not passed validation")
            if state.published_version in rows:
                rows[state.published_version].state = "superseded"
            draft.state = "published"
            draft.published_at = datetime.now(UTC)
            state.published_version = version
            state.draft_version = None
            state.revision += 1
            state.updated_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(draft)
            return draft

    async def rollback_slot(
        self,
        *,
        workflow_id: int,
        slot: str,
        to_version: int,
        expected_revision: int,
        created_by: Optional[str],
        change_note: Optional[str],
    ) -> WorkflowSlotSettingModel:
        """Republish an earlier published version as a new version (linear history)."""
        async with self.async_session() as session:
            state = await self._lock_state(session, workflow_id, slot)
            self._check_revision(state, expected_revision)
            if state.published_version == to_version:
                raise SlotStateError(f"version {to_version} is already live")
            result = await session.execute(
                select(WorkflowSlotSettingModel).where(
                    WorkflowSlotSettingModel.workflow_id == workflow_id,
                    WorkflowSlotSettingModel.slot == slot,
                    WorkflowSlotSettingModel.version.in_(
                        [
                            v
                            for v in (to_version, state.published_version)
                            if v is not None
                        ]
                    ),
                )
            )
            rows = {r.version: r for r in result.scalars().all()}
            target = rows.get(to_version)
            if target is None or target.published_at is None:
                raise SlotStateError(
                    f"version {to_version} was never published; cannot roll back to it"
                )
            if state.published_version in rows:
                rows[state.published_version].state = "superseded"
            version = state.last_version + 1
            row = WorkflowSlotSettingModel(
                workflow_id=workflow_id,
                slot=slot,
                version=version,
                state="published",
                config=target.config,
                credential_ref=target.credential_ref,
                credential_version=target.credential_version,
                validation_status=target.validation_status,
                validated_at=target.validated_at,
                based_on_version=to_version,
                origin="rollback",
                change_note=change_note,
                created_by=created_by,
                published_at=datetime.now(UTC),
            )
            session.add(row)
            state.published_version = version
            state.last_version = version
            state.revision += 1
            state.updated_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(row)
            return row

    async def seed_published_slots(
        self,
        *,
        workflow_id: int,
        slots: dict[str, dict[str, Any]],
        origin: str,
        created_by: Optional[str],
    ) -> list[str]:
        """Publish version 1 of each given slot that has never been touched.

        ``slots`` maps slot -> {config, credential_ref, credential_version}.
        A slot that already has any history is skipped, never overwritten.
        Returns the slots that were seeded.
        """
        seeded: list[str] = []
        async with self.async_session() as session:
            for slot, data in slots.items():
                state = await self._lock_state(session, workflow_id, slot)
                if state.last_version != 0:
                    continue
                session.add(
                    WorkflowSlotSettingModel(
                        workflow_id=workflow_id,
                        slot=slot,
                        version=1,
                        state="published",
                        config=data["config"],
                        credential_ref=data.get("credential_ref"),
                        credential_version=data.get("credential_version"),
                        validation_status="valid",
                        validated_at=datetime.now(UTC),
                        origin=origin,
                        change_note=f"seeded by {origin}",
                        created_by=created_by,
                        published_at=datetime.now(UTC),
                    )
                )
                state.published_version = 1
                state.last_version = 1
                state.revision += 1
                state.updated_at = datetime.now(UTC)
                seeded.append(slot)
            await session.commit()
        return seeded

    # ---- template status --------------------------------------------------

    async def get_workflow_slot_template_status(
        self, workflow_id: int
    ) -> Optional[str]:
        async with self.async_session() as session:
            result = await session.execute(
                select(WorkflowModel.slot_template_status).where(
                    WorkflowModel.id == workflow_id
                )
            )
            return result.scalar()

    async def set_workflow_slot_template_status(
        self, workflow_id: int, status: Optional[str]
    ) -> None:
        async with self.async_session() as session:
            result = await session.execute(
                select(WorkflowModel)
                .where(WorkflowModel.id == workflow_id)
                .with_for_update()
            )
            workflow = result.scalars().one()
            workflow.slot_template_status = status
            await session.commit()

    # ---- credentials ------------------------------------------------------

    async def create_provider_credential_version(
        self,
        *,
        organization_id: int,
        credential_ref: Optional[str],
        kind: str,
        label: Optional[str],
        source_ref: Optional[str],
        created_by: Optional[str],
        encrypt: Callable[[str, int], tuple[str, str]],
    ) -> ProviderCredentialModel:
        """Create a new version (or a new reference when ``credential_ref`` is None).

        ``encrypt(ref, version) -> (ciphertext, key_id)`` runs inside the write so
        ciphertext is bound to the version it is stored under.
        """
        ref = credential_ref or f"cred_{uuid.uuid4().hex[:20]}"
        for attempt in range(2):
            async with self.async_session() as session:
                current = await session.execute(
                    select(func.max(ProviderCredentialModel.version)).where(
                        ProviderCredentialModel.organization_id == organization_id,
                        ProviderCredentialModel.credential_ref == ref,
                    )
                )
                latest = current.scalar()
                if credential_ref and latest is None:
                    raise SlotStateError("credential reference not found")
                version = (latest or 0) + 1
                ciphertext, key_id = encrypt(ref, version)
                row = ProviderCredentialModel(
                    organization_id=organization_id,
                    credential_ref=ref,
                    version=version,
                    label=label,
                    kind=kind,
                    ciphertext=ciphertext,
                    key_id=key_id,
                    source_ref=source_ref,
                    created_by=created_by,
                )
                session.add(row)
                try:
                    await session.commit()
                except IntegrityError:
                    await session.rollback()
                    if attempt == 1:
                        raise
                    continue
                await session.refresh(row)
                return row
        raise RuntimeError("unreachable")

    async def get_provider_credential(
        self, organization_id: int, credential_ref: str, version: int
    ) -> Optional[ProviderCredentialModel]:
        async with self.async_session() as session:
            result = await session.execute(
                select(ProviderCredentialModel).where(
                    ProviderCredentialModel.organization_id == organization_id,
                    ProviderCredentialModel.credential_ref == credential_ref,
                    ProviderCredentialModel.version == version,
                )
            )
            return result.scalars().first()

    async def get_latest_provider_credential(
        self, organization_id: int, credential_ref: str
    ) -> Optional[ProviderCredentialModel]:
        async with self.async_session() as session:
            result = await session.execute(
                select(ProviderCredentialModel)
                .where(
                    ProviderCredentialModel.organization_id == organization_id,
                    ProviderCredentialModel.credential_ref == credential_ref,
                    ProviderCredentialModel.revoked_at.is_(None),
                )
                .order_by(ProviderCredentialModel.version.desc())
                .limit(1)
            )
            return result.scalars().first()

    async def list_provider_credentials(
        self, organization_id: int
    ) -> list[ProviderCredentialModel]:
        async with self.async_session() as session:
            result = await session.execute(
                select(ProviderCredentialModel)
                .where(ProviderCredentialModel.organization_id == organization_id)
                .order_by(
                    ProviderCredentialModel.credential_ref,
                    ProviderCredentialModel.version,
                )
            )
            return list(result.scalars().all())

    async def revoke_provider_credential_version(
        self, organization_id: int, credential_ref: str, version: int
    ) -> Optional[ProviderCredentialModel]:
        async with self.async_session() as session:
            result = await session.execute(
                select(ProviderCredentialModel)
                .where(
                    ProviderCredentialModel.organization_id == organization_id,
                    ProviderCredentialModel.credential_ref == credential_ref,
                    ProviderCredentialModel.version == version,
                )
                .with_for_update()
            )
            row = result.scalars().first()
            if row is None:
                return None
            if row.revoked_at is None:
                row.revoked_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(row)
            return row

    async def count_slot_usages_of_credential(
        self, *, organization_id: int, credential_ref: str, version: int
    ) -> int:
        """Published slots (of this org's workflows) pinned to a credential version."""
        async with self.async_session() as session:
            result = await session.execute(
                select(func.count())
                .select_from(WorkflowSlotSettingModel)
                .join(
                    WorkflowSlotStateModel,
                    (
                        WorkflowSlotStateModel.workflow_id
                        == WorkflowSlotSettingModel.workflow_id
                    )
                    & (WorkflowSlotStateModel.slot == WorkflowSlotSettingModel.slot)
                    & (
                        WorkflowSlotStateModel.published_version
                        == WorkflowSlotSettingModel.version
                    ),
                )
                .join(
                    WorkflowModel,
                    WorkflowModel.id == WorkflowSlotSettingModel.workflow_id,
                )
                .where(
                    WorkflowModel.organization_id == organization_id,
                    WorkflowSlotSettingModel.credential_ref == credential_ref,
                    WorkflowSlotSettingModel.credential_version == version,
                )
            )
            return int(result.scalar() or 0)
