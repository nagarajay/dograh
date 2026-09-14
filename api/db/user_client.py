import uuid
from datetime import datetime, timezone

from loguru import logger
from pydantic import ValidationError
from sqlalchemy import func, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.future import select

from api.db.base_client import BaseDBClient
from api.db.models import UserConfigurationModel, UserModel
from api.enums import UserConfigurationKey
from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration

# Arbitrary fixed key for the platform-bootstrap advisory lock (see
# ``bootstrap_first_superadmin``). Any 64-bit constant works as long as it is
# not reused by another advisory-lock caller in this codebase; there is none
# today.
_BOOTSTRAP_SUPERADMIN_LOCK_KEY = 0x646F677261685F31  # "dograh_1" in hex


class SuperadminAlreadyExists(Exception):
    """A platform super-admin already exists; bootstrap is one-time only."""


class UserClient(BaseDBClient):
    async def get_or_create_user_by_provider_id(
        self, provider_id: str
    ) -> tuple[UserModel, bool]:
        """Return (user, was_created) tuple."""
        async with self.async_session() as session:
            # First try to get existing user
            result = await session.execute(
                select(UserModel).where(UserModel.provider_id == provider_id)
            )
            user = result.scalars().first()

            if user is not None:
                return user, False

            # Use PostgreSQL's INSERT ... ON CONFLICT DO NOTHING
            # This is atomic and handles race conditions at the database level
            stmt = insert(UserModel.__table__).values(
                provider_id=provider_id,
                created_at=datetime.now(timezone.utc),
                selected_organization_id=None,  # Will be set later
                is_superuser=False,  # Default value
            )
            # ON CONFLICT DO NOTHING - if another request already inserted, this becomes a no-op
            stmt = stmt.on_conflict_do_nothing(index_elements=["provider_id"])

            result = await session.execute(stmt)
            await session.commit()
            was_created = result.rowcount > 0

            # Now fetch the user (either the one we just created or the one that existed)
            result = await session.execute(
                select(UserModel).where(UserModel.provider_id == provider_id)
            )
            user = result.scalars().first()

            if user is None:
                # This should never happen, but handle it just in case
                error_msg = (
                    f"Failed to create or fetch user with provider_id {provider_id}"
                )
                raise ValueError(error_msg)
        return user, was_created

    async def get_user_by_id(self, user_id: int) -> UserModel | None:
        """Fetch a user by their internal ID."""
        async with self.async_session() as session:
            result = await session.execute(
                select(UserModel).where(UserModel.id == user_id)
            )
            return result.scalars().first()

    async def _get_user_configuration_row(
        self, session, user_id: int, key: str
    ) -> UserConfigurationModel | None:
        result = await session.execute(
            select(UserConfigurationModel).where(
                UserConfigurationModel.user_id == user_id,
                UserConfigurationModel.key == key,
            )
        )
        return result.scalars().first()

    async def get_user_configuration_value(self, user_id: int, key: str) -> dict | None:
        """Get the JSON value stored for a user under `key`, or None."""
        async with self.async_session() as session:
            row = await self._get_user_configuration_row(session, user_id, key)
            return row.configuration if row else None

    async def upsert_user_configuration_value(
        self, user_id: int, key: str, value: dict
    ) -> dict:
        """Create or update the JSON value stored for a user under `key`."""
        async with self.async_session() as session:
            stmt = insert(UserConfigurationModel.__table__).values(
                user_id=user_id,
                key=key,
                configuration=value,
            )
            stmt = stmt.on_conflict_do_update(
                constraint="_user_configuration_key_uc",
                set_={"configuration": stmt.excluded.configuration},
            ).returning(UserConfigurationModel.configuration)
            try:
                result = await session.execute(stmt)
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise e
            return result.scalar_one()

    async def get_user_configurations(
        self, user_id: int
    ) -> EffectiveAIModelConfiguration:
        async with self.async_session() as session:
            configuration_obj = await self._get_user_configuration_row(
                session, user_id, UserConfigurationKey.MODEL_CONFIGURATION.value
            )
            if not configuration_obj:
                return EffectiveAIModelConfiguration()

            try:
                return EffectiveAIModelConfiguration.model_validate(
                    {
                        **configuration_obj.configuration,
                        "last_validated_at": configuration_obj.last_validated_at,
                    }
                )
            except ValidationError as e:
                # If configuration contains an unsupported provider,
                # return a default configuration without failing
                logger.warning(
                    f"Failed to validate user configuration for user {user_id}: {e}. "
                    "Returning default configuration."
                )
                return EffectiveAIModelConfiguration()

    async def update_user_configuration(
        self, user_id: int, configuration: EffectiveAIModelConfiguration
    ) -> EffectiveAIModelConfiguration:
        value = await self.upsert_user_configuration_value(
            user_id,
            UserConfigurationKey.MODEL_CONFIGURATION.value,
            configuration.model_dump(),
        )
        return EffectiveAIModelConfiguration.model_validate(value)

    async def update_user_configuration_last_validated_at(self, user_id: int) -> None:
        async with self.async_session() as session:
            configuration_obj = await self._get_user_configuration_row(
                session, user_id, UserConfigurationKey.MODEL_CONFIGURATION.value
            )
            if not configuration_obj:
                raise ValueError(f"User configuration with ID {user_id} not found")
            configuration_obj.last_validated_at = datetime.now()
            try:
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise e
            await session.refresh(configuration_obj)

    async def update_user_selected_organization(
        self, user_id: int, organization_id: int
    ) -> None:
        """Update the user's selected organization ID."""
        async with self.async_session() as session:
            from sqlalchemy import update

            # Use a direct UPDATE statement to avoid race conditions
            # This is atomic at the database level
            stmt = (
                update(UserModel)
                .where(UserModel.id == user_id)
                .values(selected_organization_id=organization_id)
            )

            result = await session.execute(stmt)

            if result.rowcount == 0:
                raise ValueError(f"User with ID {user_id} not found")

            await session.commit()

    async def update_user_email(self, user_id: int, email: str) -> None:
        """Update the user's email address."""
        async with self.async_session() as session:
            from sqlalchemy import update

            stmt = (
                update(UserModel)
                .where(UserModel.id == user_id)
                .values(email=email.lower())
            )
            await session.execute(stmt)
            await session.commit()

    async def get_user_by_email(self, email: str) -> UserModel | None:
        """Fetch a user by their email address (case-insensitive).

        Email addresses are case-insensitive in practice, so a user who
        signed up as "User@example.com" must still be found when they later
        log in as "user@example.com". Compare on lower(email) so lookups are
        robust to capitalization differences across sign-in flows.
        """
        normalized_email = email.lower()
        async with self.async_session() as session:
            result = await session.execute(
                select(UserModel).where(func.lower(UserModel.email) == normalized_email)
            )
            return result.scalars().first()

    async def create_user_with_email(
        self,
        email: str,
        password_hash: str,
        name: str | None = None,
        is_superuser: bool = False,
    ) -> UserModel:
        """Create a new user with email and password hash.

        ``is_superuser`` defaults to False, matching ordinary signup. The
        platform bootstrap path is the only caller that passes True, and it
        does so directly at creation time rather than creating then
        promoting, so there is never a moment where the row exists without
        the authority it was created for.
        """
        async with self.async_session() as session:
            user = UserModel(
                provider_id=f"oss_{int(datetime.now(timezone.utc).timestamp())}_{uuid.uuid4()}",
                email=email.lower(),
                password_hash=password_hash,
                is_superuser=is_superuser,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    async def bootstrap_first_superadmin(
        self, *, email: str, password_hash: str
    ) -> tuple[UserModel, bool]:
        """Atomically check-and-create/promote the platform's first super-admin.

        The "no super-admin exists yet" check and the create-or-promote write
        must happen as one indivisible step, or two concurrent first-bootstrap
        requests can both pass the check and each mint a super-admin. A
        Postgres transaction-scoped advisory lock (``pg_advisory_xact_lock``)
        is the simplest tool for that here: it needs no new table or column,
        it is held only for the lifetime of this transaction, and it is
        released automatically on commit or rollback -- including on a crash
        mid-transaction, unlike a session-scoped lock that would need explicit
        unlocking. The second concurrent caller simply blocks until the first
        commits, then re-runs its own count check against the now-updated
        table and correctly refuses.

        Returns ``(user, created)``, where ``created`` is False when an
        existing user matching ``email`` was promoted instead of a new row
        being inserted. Raises :class:`SuperadminAlreadyExists` if a
        super-admin already exists anywhere -- this module raises no HTTP
        exception itself, leaving that translation to the caller.
        """
        normalized_email = email.lower()
        async with self.async_session() as session:
            # No explicit ``session.begin()``: this session's transaction may
            # already be open (the test suite's savepoint-isolation fixture
            # reuses one shared session across calls), so the lock, the check
            # and the write all run against whatever transaction is already
            # current -- exactly like every other method in this class.
            await session.execute(
                text("SELECT pg_advisory_xact_lock(:key)"),
                {"key": _BOOTSTRAP_SUPERADMIN_LOCK_KEY},
            )

            count_result = await session.execute(
                select(func.count()).where(UserModel.is_superuser.is_(True))
            )
            if count_result.scalar_one() > 0:
                raise SuperadminAlreadyExists()

            existing_result = await session.execute(
                select(UserModel).where(func.lower(UserModel.email) == normalized_email)
            )
            existing = existing_result.scalars().first()

            if existing is not None:
                existing.is_superuser = True
                await session.commit()
                await session.refresh(existing)
                return existing, False

            user = UserModel(
                provider_id=(
                    f"oss_{int(datetime.now(timezone.utc).timestamp())}_"
                    f"{uuid.uuid4()}"
                ),
                email=normalized_email,
                password_hash=password_hash,
                is_superuser=True,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user, True
