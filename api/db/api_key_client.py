from datetime import datetime, timezone
from typing import List, Optional

from loguru import logger
from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.future import select

from api.db.base_client import BaseDBClient
from api.db.models import APIKeyModel
from api.utils.api_key import generate_api_key, hash_api_key

# How many times a losing racer re-runs archive-then-insert before giving up.
# Each attempt observes the winner's committed state, so one retry is enough in
# practice; the bound exists so a pathological loop cannot spin forever.
_REPLACE_ATTEMPTS = 3


class APIKeyClient(BaseDBClient):
    async def create_api_key(
        self, organization_id: int, name: str, created_by: Optional[int] = None
    ) -> tuple[APIKeyModel, str]:
        """Create a new API key for an organization.

        Returns:
            Tuple of (APIKeyModel, raw_api_key)
        """
        # Generate a secure random API key
        raw_api_key, key_hash, key_prefix = generate_api_key()

        async with self.async_session() as session:
            api_key = APIKeyModel(
                organization_id=organization_id,
                name=name,
                key_hash=key_hash,
                key_prefix=key_prefix,
                created_by=created_by,
                is_active=True,
            )
            session.add(api_key)
            await session.commit()
            await session.refresh(api_key)

            return api_key, raw_api_key

    async def replace_api_key_by_name(
        self, organization_id: int, name: str, created_by: Optional[int] = None
    ) -> tuple[APIKeyModel, str, list[int]]:
        """Rotate the organization's one key with this name, atomically.

        Archives every active key the organization holds under ``name`` and
        issues a replacement in the same transaction, so the organization is
        never briefly without a key and never briefly holding two.

        This is what makes minting retry-safe. ``create_api_key`` appends: a
        caller that retries because it lost the response leaves behind a live
        credential nobody holds, and no later call can tell which of the two the
        caller actually has. Replacing means the answer after any number of
        retries is the same -- one active key, and it is the one the last
        response returned.

        Returns ``(api_key, raw_key, archived_ids)``. ``archived_ids`` is the
        keys this call invalidated, which is the caller's evidence that the
        rotation happened rather than a no-op.
        """
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                return await self._replace_api_key_by_name(
                    organization_id, name, created_by
                )
            except IntegrityError:
                # A concurrent replacement committed between this call's
                # archive statement and its insert, and the partial unique
                # index refused the second active row -- which is the whole
                # point of that index. Re-running now observes the winner's key
                # and archives it, so the retry converges instead of failing.
                if attempt == _REPLACE_ATTEMPTS - 1:
                    raise
                logger.warning(
                    "Contended replacement of api key {!r} for organization "
                    "{}; retrying",
                    name,
                    organization_id,
                )
        raise AssertionError("unreachable")

    async def _replace_api_key_by_name(
        self, organization_id: int, name: str, created_by: Optional[int]
    ) -> tuple[APIKeyModel, str, list[int]]:
        raw_api_key, key_hash, key_prefix = generate_api_key()
        now = datetime.now(timezone.utc)

        async with self.async_session() as session:
            existing = (
                (
                    await session.execute(
                        select(APIKeyModel).where(
                            and_(
                                APIKeyModel.organization_id == organization_id,
                                APIKeyModel.name == name,
                                APIKeyModel.is_active == True,  # noqa: E712
                                APIKeyModel.archived_at.is_(None),
                            )
                        )
                    )
                )
                .scalars()
                .all()
            )

            archived_ids = []
            for api_key in existing:
                api_key.is_active = False
                api_key.archived_at = now
                archived_ids.append(api_key.id)

            replacement = APIKeyModel(
                organization_id=organization_id,
                name=name,
                key_hash=key_hash,
                key_prefix=key_prefix,
                created_by=created_by,
                is_active=True,
            )
            session.add(replacement)

            # One commit for both halves: a crash between them would otherwise
            # either revoke the caller's only key or leave two live.
            await session.commit()
            await session.refresh(replacement)

            return replacement, raw_api_key, archived_ids

    async def get_api_keys_by_organization(
        self, organization_id: int, include_archived: bool = False
    ) -> List[APIKeyModel]:
        """Get all API keys for an organization."""
        async with self.async_session() as session:
            query = select(APIKeyModel).where(
                APIKeyModel.organization_id == organization_id
            )

            if not include_archived:
                query = query.where(APIKeyModel.archived_at.is_(None))

            result = await session.execute(query)
            return result.scalars().all()

    async def get_api_key_by_hash(self, key_hash: str) -> Optional[APIKeyModel]:
        """Get an API key by its hash."""
        async with self.async_session() as session:
            result = await session.execute(
                select(APIKeyModel).where(
                    and_(
                        APIKeyModel.key_hash == key_hash,
                        APIKeyModel.is_active == True,
                        APIKeyModel.archived_at.is_(None),
                    )
                )
            )
            return result.scalars().first()

    async def validate_api_key(self, raw_api_key: str) -> Optional[APIKeyModel]:
        """Validate an API key and return the associated model if valid."""
        key_hash = hash_api_key(raw_api_key)
        api_key = await self.get_api_key_by_hash(key_hash)

        if api_key:
            # Update last_used_at
            from datetime import datetime, timezone

            async with self.async_session() as session:
                await session.execute(
                    APIKeyModel.__table__.update()
                    .where(APIKeyModel.id == api_key.id)
                    .values(last_used_at=datetime.now(timezone.utc))
                )
                await session.commit()

        return api_key

    async def archive_api_key(self, api_key_id: int) -> bool:
        """Archive an API key (soft delete)."""
        from datetime import datetime, timezone

        async with self.async_session() as session:
            result = await session.execute(
                APIKeyModel.__table__.update()
                .where(APIKeyModel.id == api_key_id)
                .values(is_active=False, archived_at=datetime.now(timezone.utc))
            )
            await session.commit()
            return result.rowcount > 0

    async def reactivate_api_key(self, api_key_id: int) -> bool:
        """Reactivate an archived API key."""
        async with self.async_session() as session:
            result = await session.execute(
                APIKeyModel.__table__.update()
                .where(APIKeyModel.id == api_key_id)
                .values(is_active=True, archived_at=None)
            )
            await session.commit()
            return result.rowcount > 0
