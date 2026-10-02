"""Persistence operations for platform-owned Gemini-TTS sample packs."""

from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from api.db.base_client import BaseDBClient
from api.db.models import GeminiTTSSampleAssetModel, GeminiTTSSamplePackModel
from api.services.configuration.options.google_vertex_catalog import gemini_tts_voices


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

    async def claim_gemini_tts_sample_asset(self, asset_id: int) -> bool:
        return await self.claim_asset(asset_id)

    async def finish_gemini_tts_sample_asset(
        self, asset_id: int, *, status: str, values: dict
    ) -> None:
        await self.finish_asset(asset_id, status=status, values=values)

    async def claim_asset(self, asset_id: int) -> bool:
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
                    status="running",
                    attempts=GeminiTTSSampleAssetModel.attempts + 1,
                    error_message=None,
                )
            )
            if pack is not None:
                await self._refresh_progress(session, pack)
            await session.commit()
            return result.rowcount == 1

    async def finish_asset(self, asset_id: int, *, status: str, values: dict) -> None:
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
            await session.execute(
                update(GeminiTTSSampleAssetModel)
                .where(GeminiTTSSampleAssetModel.id == asset_id)
                .values(status=status, **values)
            )
            asset = await session.scalar(
                select(GeminiTTSSampleAssetModel).where(
                    GeminiTTSSampleAssetModel.id == asset_id
                )
            )
            if asset:
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
