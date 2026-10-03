"""Persistence operations for platform-owned Gemini-TTS sample packs."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from api.db.base_client import BaseDBClient
from api.db.models import GeminiTTSSampleAssetModel, GeminiTTSSamplePackModel
from api.services.configuration.options.google_vertex_catalog import gemini_tts_voices
from api.services.gemini_tts_sample_jobs import (
    CLAIM_LEASE_SECONDS,
    EXPIRED_LEASE_MESSAGE,
)


class GeminiTTSSampleClient(BaseDBClient):
    @staticmethod
    def _new_voice_version(session, pack, voice_id, gender, assets, metadata=None):
        row = GeminiTTSSampleAssetModel(
            pack_id=pack.id,
            voice_id=voice_id,
            gender=gender,
            version=max((a.version for a in assets), default=0) + 1,
            is_current=False,
            status="queued",
            generation_metadata=metadata or {},
        )
        session.add(row)
        pack.status = "queued"
        pack.completed_at = None
        return row

    @staticmethod
    async def _refresh_progress(session, pack):
        # Serialize scheduling and finalization on the same parent row.
        await session.flush()
        rows = (
            await session.execute(
                select(
                    GeminiTTSSampleAssetModel.voice_id,
                    GeminiTTSSampleAssetModel.status,
                    GeminiTTSSampleAssetModel.is_current,
                ).where(GeminiTTSSampleAssetModel.pack_id == pack.id)
            )
        ).all()
        expected = {v.id for v in gemini_tts_voices(pack.model_id)}
        playable = {v for v, s, current in rows if s == "completed" and current}
        required = [(v, s) for v, s, _ in rows if v in expected - playable]
        if expected and expected <= playable:
            pack.status = "completed"
            pack.completed_at = pack.completed_at or datetime.now(UTC)
        else:
            pack.completed_at = None
            if any(s == "running" for _, s in required):
                pack.status = "running"
            elif any(s == "queued" for _, s in required):
                pack.status = "queued"
            else:
                pack.status = "partial" if playable else "failed"

    @staticmethod
    async def _pack_in_session(session, pack_id: int):
        return await session.scalar(
            select(GeminiTTSSamplePackModel)
            .options(selectinload(GeminiTTSSamplePackModel.assets))
            .where(GeminiTTSSamplePackModel.id == pack_id)
        )

    async def create_or_get_pack(
        self, *, pack: dict, assets: list[dict]
    ) -> tuple[GeminiTTSSamplePackModel, bool]:
        """Create a pack exactly once by fingerprint; return (row, created)."""
        async with self.async_session() as session:
            existing = await session.scalar(
                select(GeminiTTSSamplePackModel).where(
                    GeminiTTSSamplePackModel.request_fingerprint
                    == pack["request_fingerprint"]
                )
            )
            if existing:
                return await self._pack_in_session(session, existing.id), False
            row = GeminiTTSSamplePackModel(**pack)
            row.assets = [GeminiTTSSampleAssetModel(**asset) for asset in assets]
            session.add(row)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                existing = await session.scalar(
                    select(GeminiTTSSamplePackModel)
                    .options(selectinload(GeminiTTSSamplePackModel.assets))
                    .where(
                        GeminiTTSSamplePackModel.request_fingerprint
                        == pack["request_fingerprint"]
                    )
                )
                if existing is None:
                    raise
                return await self._pack_in_session(session, existing.id), False
            await session.refresh(row)
            return await self._pack_in_session(session, row.id), True

    async def get_pack(self, pack_id: int) -> GeminiTTSSamplePackModel | None:
        async with self.async_session() as session:
            return await self._pack_in_session(session, pack_id)

    async def list_packs(self, *, completed_only: bool = False) -> list:
        async with self.async_session() as session:
            query = (
                select(GeminiTTSSamplePackModel)
                .options(selectinload(GeminiTTSSamplePackModel.assets))
                .order_by(GeminiTTSSamplePackModel.created_at.desc())
            )
            if completed_only:
                query = query.where(GeminiTTSSamplePackModel.status == "completed")
            result = await session.execute(query)
            return list(result.scalars().unique().all())

    async def list_matching_completed_assets(
        self,
        *,
        model_id: str,
        language: str,
        style_text: str,
        sample_text: str,
        catalog_revision: str | None = None,
        location: str | None = None,
    ) -> list[tuple[GeminiTTSSamplePackModel, GeminiTTSSampleAssetModel]]:
        conditions = [
            GeminiTTSSamplePackModel.model_id == model_id,
            GeminiTTSSamplePackModel.language == language,
            GeminiTTSSamplePackModel.style_text == style_text,
            GeminiTTSSamplePackModel.sample_text == sample_text,
            GeminiTTSSampleAssetModel.status == "completed",
            GeminiTTSSampleAssetModel.is_current.is_(True),
        ]
        if catalog_revision is not None:
            conditions.append(
                GeminiTTSSamplePackModel.catalog_revision == catalog_revision
            )
        if location is not None:
            conditions.append(GeminiTTSSamplePackModel.location == location)
        async with self.async_session() as session:
            result = await session.execute(
                select(GeminiTTSSamplePackModel, GeminiTTSSampleAssetModel)
                .join(
                    GeminiTTSSampleAssetModel,
                    GeminiTTSSampleAssetModel.pack_id == GeminiTTSSamplePackModel.id,
                )
                .where(*conditions)
                .order_by(GeminiTTSSamplePackModel.created_at.desc())
            )
            return list(result.all())

    async def get_asset(self, asset_id: int) -> GeminiTTSSampleAssetModel | None:
        async with self.async_session() as session:
            return await session.scalar(
                select(GeminiTTSSampleAssetModel).where(
                    GeminiTTSSampleAssetModel.id == asset_id
                )
            )

    async def ensure_gemini_tts_sample_pack_assets(
        self, pack_id: int, *, voices: list[dict]
    ) -> None:
        """Add missing catalog voices without touching existing asset rows."""
        async with self.async_session() as session:
            pack = await session.scalar(
                select(GeminiTTSSamplePackModel)
                .where(GeminiTTSSamplePackModel.id == pack_id)
                .with_for_update()
            )
            if pack is None:
                return
            existing = set(
                (
                    await session.scalars(
                        select(GeminiTTSSampleAssetModel.voice_id).where(
                            GeminiTTSSampleAssetModel.pack_id == pack_id
                        )
                    )
                ).all()
            )
            for voice in voices:
                if voice["voice_id"] in existing:
                    continue
                session.add(
                    GeminiTTSSampleAssetModel(
                        pack_id=pack_id,
                        voice_id=voice["voice_id"],
                        gender=voice["gender"],
                        version=1,
                        is_current=False,
                        status="queued",
                    )
                )
            await session.commit()

    async def queue_voice_generation(
        self, pack_id: int, *, voice_id: str, gender: str, regenerate: bool
    ) -> tuple[GeminiTTSSampleAssetModel | None, str | None]:
        """Atomically create at most one queued generation for this pack/voice."""
        async with self.async_session() as session:
            pack = await session.scalar(
                select(GeminiTTSSamplePackModel)
                .where(GeminiTTSSamplePackModel.id == pack_id)
                .with_for_update()
            )
            if pack is None:
                return None, "not_found"
            assets = list(
                (
                    await session.scalars(
                        select(GeminiTTSSampleAssetModel)
                        .where(
                            GeminiTTSSampleAssetModel.pack_id == pack_id,
                            GeminiTTSSampleAssetModel.voice_id == voice_id,
                        )
                        .order_by(GeminiTTSSampleAssetModel.version.desc())
                    )
                ).all()
            )
            active = next(
                (a for a in assets if a.status in {"queued", "running"}), None
            )
            if active is not None:
                return active, active.status
            current = next((a for a in assets if a.is_current), None)
            if current is not None and current.status == "completed" and not regenerate:
                return current, "regeneration_confirmation_required"
            latest = assets[0] if assets else None
            if (
                latest is not None
                and latest.status == "failed"
                and current is None
                and not regenerate
            ):
                return latest, "failed_asset_use_retry"
            row = self._new_voice_version(session, pack, voice_id, gender, assets)
            await self._refresh_progress(session, pack)
            await session.commit()
            await session.refresh(row)
            return row, None

    async def get_gemini_tts_sample_pack(self, pack_id: int):
        return await self.get_pack(pack_id)

    async def get_gemini_tts_sample_asset(self, asset_id: int):
        return await self.get_asset(asset_id)

    async def claim_gemini_tts_sample_asset(self, asset_id: int) -> str | None:
        return await self.claim_asset(asset_id)

    async def finish_gemini_tts_sample_asset(
        self, asset_id: int, *, status: str, values: dict, claim_token: str
    ) -> bool:
        return await self.finish_asset(
            asset_id, status=status, values=values, claim_token=claim_token
        )

    @staticmethod
    async def _lock_pack_of(session, asset_id: int):
        return await session.scalar(
            select(GeminiTTSSamplePackModel)
            .join(
                GeminiTTSSampleAssetModel,
                GeminiTTSSampleAssetModel.pack_id == GeminiTTSSamplePackModel.id,
            )
            .where(GeminiTTSSampleAssetModel.id == asset_id)
            .with_for_update(of=GeminiTTSSamplePackModel)
        )

    async def claim_asset(
        self, asset_id: int, *, lease_seconds: int = CLAIM_LEASE_SECONDS
    ) -> str | None:
        """Move one ``queued`` asset to ``running`` and return its claim token.

        Returns None when the asset is not queued (already claimed, finished or
        compensated), so exactly one worker ever owns an asset version.
        """
        token = uuid4().hex
        async with self.async_session() as session:
            pack = await self._lock_pack_of(session, asset_id)
            result = await session.execute(
                update(GeminiTTSSampleAssetModel)
                .where(
                    GeminiTTSSampleAssetModel.id == asset_id,
                    GeminiTTSSampleAssetModel.status == "queued",
                )
                .values(
                    status="running",
                    attempts=GeminiTTSSampleAssetModel.attempts + 1,
                    error_message=None,
                    claim_token=token,
                    lease_expires_at=datetime.now(UTC)
                    + timedelta(seconds=lease_seconds),
                )
            )
            if pack is not None:
                await self._refresh_progress(session, pack)
            await session.commit()
            return token if result.rowcount == 1 else None

    async def renew_asset_lease(
        self,
        asset_id: int,
        claim_token: str,
        *,
        lease_seconds: int = CLAIM_LEASE_SECONDS,
    ) -> bool:
        """Extend the lease; False means the claim was replaced or ended."""
        async with self.async_session() as session:
            result = await session.execute(
                update(GeminiTTSSampleAssetModel)
                .where(
                    GeminiTTSSampleAssetModel.id == asset_id,
                    GeminiTTSSampleAssetModel.status == "running",
                    GeminiTTSSampleAssetModel.claim_token == claim_token,
                )
                .values(
                    lease_expires_at=datetime.now(UTC)
                    + timedelta(seconds=lease_seconds)
                )
            )
            await session.commit()
            return result.rowcount == 1

    async def finish_asset(
        self, asset_id: int, *, status: str, values: dict, claim_token: str
    ) -> bool:
        """Finalize a claimed asset. A stale worker (claim replaced) changes nothing.

        Returns False, leaving the row untouched, unless the asset is still
        ``running`` under ``claim_token``.
        """
        async with self.async_session() as session:
            pack = await self._lock_pack_of(session, asset_id)
            owned = await session.execute(
                update(GeminiTTSSampleAssetModel)
                .where(
                    GeminiTTSSampleAssetModel.id == asset_id,
                    GeminiTTSSampleAssetModel.status == "running",
                    GeminiTTSSampleAssetModel.claim_token == claim_token,
                )
                .values(
                    status=status,
                    claim_token=None,
                    lease_expires_at=None,
                    **values,
                )
            )
            if owned.rowcount != 1:
                await session.rollback()
                return False
            asset = await session.scalar(
                select(GeminiTTSSampleAssetModel).where(
                    GeminiTTSSampleAssetModel.id == asset_id
                )
            )
            if status == "completed":
                await session.execute(
                    update(GeminiTTSSampleAssetModel)
                    .where(
                        GeminiTTSSampleAssetModel.pack_id == asset.pack_id,
                        GeminiTTSSampleAssetModel.voice_id == asset.voice_id,
                        GeminiTTSSampleAssetModel.id != asset.id,
                    )
                    .values(is_current=False)
                )
                await session.execute(
                    update(GeminiTTSSampleAssetModel)
                    .where(GeminiTTSSampleAssetModel.id == asset.id)
                    .values(is_current=True)
                )
            await self._refresh_progress(session, pack)
            await session.commit()
            return True

    async def fail_expired_running_assets(self) -> list[int]:
        """Fail ``running`` assets whose lease expired (worker died or stalled).

        The provider may already have been called, so the asset is not silently
        re-run: it becomes ``failed`` and an explicit retry creates a new
        version. Clearing the token means the old worker can no longer finalize.
        Idempotent; a healthy (renewed) lease is never touched.
        """
        async with self.async_session() as session:
            ids = list(
                (
                    await session.scalars(
                        select(GeminiTTSSampleAssetModel.id).where(
                            GeminiTTSSampleAssetModel.status == "running",
                            GeminiTTSSampleAssetModel.lease_expires_at
                            < datetime.now(UTC),
                        )
                    )
                ).all()
            )
        recovered: list[int] = []
        for asset_id in ids:
            async with self.async_session() as session:
                pack = await self._lock_pack_of(session, asset_id)
                # Re-check under the pack lock: the lease may have been renewed
                # or the worker may have finished since the scan.
                result = await session.execute(
                    update(GeminiTTSSampleAssetModel)
                    .where(
                        GeminiTTSSampleAssetModel.id == asset_id,
                        GeminiTTSSampleAssetModel.status == "running",
                        GeminiTTSSampleAssetModel.lease_expires_at < datetime.now(UTC),
                    )
                    .values(
                        status="failed",
                        claim_token=None,
                        lease_expires_at=None,
                        error_message=EXPIRED_LEASE_MESSAGE,
                    )
                )
                if result.rowcount == 1 and pack is not None:
                    await self._refresh_progress(session, pack)
                    recovered.append(asset_id)
                await session.commit()
        return recovered

    async def list_stale_queued_assets(
        self, *, older_than_seconds: int
    ) -> list[tuple[int, int]]:
        """``(asset_id, enqueue_epoch)`` of assets queued longer than the grace."""
        async with self.async_session() as session:
            rows = await session.execute(
                select(
                    GeminiTTSSampleAssetModel.id,
                    GeminiTTSSampleAssetModel.enqueue_epoch,
                ).where(
                    GeminiTTSSampleAssetModel.status == "queued",
                    GeminiTTSSampleAssetModel.queued_at
                    < datetime.now(UTC) - timedelta(seconds=older_than_seconds),
                )
            )
            return [(row[0], row[1]) for row in rows.all()]

    async def bump_enqueue_epoch(self, asset_id: int, seen_epoch: int) -> int | None:
        """Claim the right to re-enqueue a lost job; only one caller wins."""
        async with self.async_session() as session:
            result = await session.execute(
                update(GeminiTTSSampleAssetModel)
                .where(
                    GeminiTTSSampleAssetModel.id == asset_id,
                    GeminiTTSSampleAssetModel.status == "queued",
                    GeminiTTSSampleAssetModel.enqueue_epoch == seen_epoch,
                )
                .values(
                    enqueue_epoch=seen_epoch + 1,
                    queued_at=datetime.now(UTC),
                )
            )
            await session.commit()
            return seen_epoch + 1 if result.rowcount == 1 else None

    async def queue_failed_voice_recovery(
        self,
        pack_id: int,
        *,
        voices: list[dict],
        operation_id: str,
        limit: int | None = None,
    ) -> tuple[list[int], dict]:
        """Reserve one NEW version per unplayable failed voice under the pack lock.

        Commit before enqueue so the worker can claim. Failed history is immutable.
        The same lock and version constructor serve individual generation.
        """
        async with self.async_session() as session:
            pack = await session.scalar(
                select(GeminiTTSSamplePackModel)
                .where(GeminiTTSSamplePackModel.id == pack_id)
                .with_for_update()
            )
            if pack is None:
                return [], {}
            assets = list(
                (
                    await session.scalars(
                        select(GeminiTTSSampleAssetModel).where(
                            GeminiTTSSampleAssetModel.pack_id == pack_id
                        )
                    )
                ).all()
            )
            grouped = {}
            for asset in assets:
                grouped.setdefault(asset.voice_id, []).append(asset)
            summary = {
                "operation_id": operation_id,
                "eligible_voices": 0,
                "skipped_playable_voices": 0,
                "skipped_active_voices": 0,
                "selected_voices": 0,
                "enqueued_jobs": 0,
                "enqueue_conflicts": 0,
                "enqueue_failures": 0,
                "asset_ids": [],
                "job_ids": [],
            }
            selected = []
            for voice in voices:
                history = grouped.get(voice["voice_id"], [])
                if any(a.is_current and a.status == "completed" for a in history):
                    summary["skipped_playable_voices"] += 1
                elif any(a.status in {"queued", "running"} for a in history):
                    summary["skipped_active_voices"] += 1
                elif any(a.status == "failed" for a in history):
                    summary["eligible_voices"] += 1
                    if limit is None or len(selected) < limit:
                        selected.append(
                            self._new_voice_version(
                                session,
                                pack,
                                voice["voice_id"],
                                voice["gender"],
                                history,
                                {"operation_id": operation_id},
                            )
                        )
            await self._refresh_progress(session, pack)
            await session.commit()
            ids = []
            for row in selected:
                await session.refresh(row)
                ids.append(row.id)
            summary["selected_voices"] = len(ids)
            summary["asset_ids"] = ids
            return ids, summary

    async def fail_sample_enqueue(self, asset_id: int) -> bool:
        """Compensate failed/uncertain enqueue only if the worker has not claimed.

        A late Redis job then fails its conditional claim, never synthesizing.
        A worker already running keeps ownership and is never reset.
        """
        async with self.async_session() as session:
            pack = await session.scalar(
                select(GeminiTTSSamplePackModel)
                .join(
                    GeminiTTSSampleAssetModel,
                    GeminiTTSSampleAssetModel.pack_id == GeminiTTSSamplePackModel.id,
                )
                .where(GeminiTTSSampleAssetModel.id == asset_id)
                .with_for_update(of=GeminiTTSSamplePackModel)
            )
            result = await session.execute(
                update(GeminiTTSSampleAssetModel)
                .where(
                    GeminiTTSSampleAssetModel.id == asset_id,
                    GeminiTTSSampleAssetModel.status == "queued",
                )
                .values(
                    status="failed",
                    error_message="Queue acceptance failed; no provider request was claimed.",
                )
            )
            if pack is not None:
                await self._refresh_progress(session, pack)
            await session.commit()
            return result.rowcount == 1
