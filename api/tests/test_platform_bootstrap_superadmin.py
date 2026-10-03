"""Seating a deployment's first super-admin with signup closed and no shell.

Before this endpoint, a fresh production database with ``ENABLE_SIGNUP=false``
and no operator shell access had no way to create *any* super-admin: ordinary
signup never sets ``is_superuser`` (see ``api/db/user_client.py``), and the
only prior path -- ``scripts/bootstrap_superadmin.py`` -- needs a database
connection an operator running only the platform credential does not have.

What is under test:

- guarded by ``X-Platform-Admin-Key``, exactly like the other two provisioning
  endpoints, and organization API keys are refused the same way
- a fresh database with zero users gets a first super-admin with no
  organization and no membership
- naming an email that already exists promotes that user instead of creating
  a second identity
- once any super-admin exists, the endpoint refuses with 409, regardless of
  which email is named -- true one-time bootstrap, not "first call wins"
- two simultaneous first-bootstrap requests cannot both succeed
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select

from api import constants
from api.db.models import UserModel
from api.routes import superuser as superuser_routes
from api.services.auth.platform_admin import PLATFORM_ADMIN_HEADER
from api.services.superuser import bootstrap as bootstrap_service
from api.services.superuser.provisioning import ProvisioningConflict
from api.utils.auth import verify_password

PLATFORM_KEY = "platform-admin-secret-key-with-enough-entropy"


@pytest.fixture
def platform_key(monkeypatch):
    monkeypatch.setattr(constants, "PLATFORM_ADMIN_API_KEY", PLATFORM_KEY)
    return PLATFORM_KEY


@pytest.fixture
def route_client():
    app = FastAPI()
    app.include_router(superuser_routes.router)
    return TestClient(app, raise_server_exceptions=False)


def _payload(**overrides) -> dict:
    return {"email": "owner@example.com", "password": "hunter2hunter2", **overrides}


# ---------------------------------------------------------------------------
# 1. Credential gate -- identical to the other platform-provisioning endpoints
# ---------------------------------------------------------------------------


def test_missing_credential_is_refused(route_client, platform_key):
    response = route_client.post("/superuser/bootstrap-superadmin", json=_payload())
    assert response.status_code == 401


def test_wrong_credential_is_refused(route_client, platform_key):
    response = route_client.post(
        "/superuser/bootstrap-superadmin",
        json=_payload(),
        headers={PLATFORM_ADMIN_HEADER: "wrong-key-of-plausible-length"},
    )
    assert response.status_code == 403


def test_unconfigured_deployment_refuses_rather_than_opens(monkeypatch, route_client):
    monkeypatch.setattr(constants, "PLATFORM_ADMIN_API_KEY", None)
    response = route_client.post(
        "/superuser/bootstrap-superadmin",
        json=_payload(),
        headers={PLATFORM_ADMIN_HEADER: "anything"},
    )
    assert response.status_code == 503


def test_org_api_key_cannot_reach_bootstrap(route_client, platform_key):
    """A tenant credential must not be able to seat a super-admin, even paired
    with the correct platform key."""
    response = route_client.post(
        "/superuser/bootstrap-superadmin",
        json=_payload(),
        headers={PLATFORM_ADMIN_HEADER: PLATFORM_KEY, "X-API-Key": "dg-tenant-key"},
    )
    assert response.status_code == 403
    assert "Organization API keys" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 2. Route shape, with the service stubbed
# ---------------------------------------------------------------------------


def test_conflict_surfaces_as_409(monkeypatch, route_client, platform_key):
    monkeypatch.setattr(
        superuser_routes,
        "bootstrap_platform_superadmin",
        AsyncMock(side_effect=ProvisioningConflict("a super-admin already exists")),
    )
    response = route_client.post(
        "/superuser/bootstrap-superadmin",
        json=_payload(),
        headers={PLATFORM_ADMIN_HEADER: PLATFORM_KEY},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "a super-admin already exists"


def test_short_password_is_rejected(route_client, platform_key):
    response = route_client.post(
        "/superuser/bootstrap-superadmin",
        json=_payload(password="short"),
        headers={PLATFORM_ADMIN_HEADER: PLATFORM_KEY},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 3. The service, against a real database
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fresh_database_gets_a_first_superadmin_with_no_organization(
    db_session,
):
    result = await bootstrap_service.bootstrap_platform_superadmin(
        email="Owner@Example.com", password="hunter2hunter2"
    )

    assert result.created is True
    user = result.user
    assert user.email == "owner@example.com"
    assert user.is_superuser is True
    assert user.selected_organization_id is None
    assert verify_password("hunter2hunter2", user.password_hash)


@pytest.mark.asyncio
async def test_existing_user_is_promoted_not_duplicated(db_session):
    existing = await db_session.create_user_with_email(
        email="owner@example.com", password_hash="irrelevant-hash"
    )
    assert existing.is_superuser is False

    result = await bootstrap_service.bootstrap_platform_superadmin(
        email="owner@example.com", password="hunter2hunter2"
    )

    assert result.created is False
    assert result.user.id == existing.id
    assert result.user.is_superuser is True

    # No second row was created for the email.
    fetched = await db_session.get_user_by_email("owner@example.com")
    assert fetched.id == existing.id


@pytest.mark.asyncio
async def test_promotion_does_not_touch_the_password(db_session):
    """Promoting an existing user must not reset a credential it did not set."""
    existing = await db_session.create_user_with_email(
        email="owner@example.com", password_hash="original-hash-untouched"
    )

    result = await bootstrap_service.bootstrap_platform_superadmin(
        email="owner@example.com", password="a-different-password"
    )

    assert result.user.password_hash == "original-hash-untouched"


@pytest.mark.asyncio
async def test_once_a_superadmin_exists_bootstrap_refuses_any_email(db_session):
    await bootstrap_service.bootstrap_platform_superadmin(
        email="first@example.com", password="hunter2hunter2"
    )

    with pytest.raises(ProvisioningConflict):
        await bootstrap_service.bootstrap_platform_superadmin(
            email="second@example.com", password="hunter2hunter2"
        )

    # The second email was never created at all.
    assert await db_session.get_user_by_email("second@example.com") is None


@pytest.mark.asyncio
async def test_a_pre_existing_superadmin_blocks_bootstrap_too(db_session):
    """The guard is "does one exist", not "did this call create one"."""
    seeded = await db_session.create_user_with_email(
        email="already-admin@example.com",
        password_hash="irrelevant-hash",
        is_superuser=True,
    )
    assert seeded.is_superuser is True

    with pytest.raises(ProvisioningConflict):
        await bootstrap_service.bootstrap_platform_superadmin(
            email="new-owner@example.com", password="hunter2hunter2"
        )


# ---------------------------------------------------------------------------
# 4. Concurrent first-bootstrap requests
#
# Deliberately does not use the ``db_session`` fixture: that fixture pins every
# call to one shared connection/savepoint, which cannot exhibit a real
# cross-connection race. This test goes through the unpatched ``db_client``
# instead, so the two calls open genuinely separate Postgres connections and
# ``pg_advisory_xact_lock`` (api/db/user_client.py:bootstrap_first_superadmin)
# is exercised for real. Because it commits directly against the real test
# database rather than through the rollback-on-exit fixture, it cleans up
# after itself in a ``finally`` so it leaves no super-admin behind for tests
# that run after it in the same session.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_first_bootstrap_produces_exactly_one_superadmin(
    setup_test_database,
):
    from api.db import db_client

    email_a = "racer-a@example.com"
    email_b = "racer-b@example.com"

    try:
        results = await asyncio.gather(
            bootstrap_service.bootstrap_platform_superadmin(
                email=email_a, password="hunter2hunter2"
            ),
            bootstrap_service.bootstrap_platform_superadmin(
                email=email_b, password="hunter2hunter2"
            ),
            return_exceptions=True,
        )

        successes = [
            r
            for r in results
            if isinstance(r, bootstrap_service.BootstrappedSuperadmin)
        ]
        conflicts = [r for r in results if isinstance(r, ProvisioningConflict)]
        others = [
            r
            for r in results
            if not isinstance(
                r, (bootstrap_service.BootstrappedSuperadmin, ProvisioningConflict)
            )
        ]

        assert others == [], f"unexpected outcome: {others}"
        assert len(successes) == 1, "exactly one racer must win"
        assert len(conflicts) == 1, "the loser must see the conflict, not a second win"

        async with db_client.async_session() as session:
            count = await session.execute(
                select(func.count()).where(UserModel.is_superuser.is_(True))
            )
            assert count.scalar_one() == 1, (
                "the database itself must end up with exactly one super-admin, "
                "not just the Python-level result count"
            )

            winner_email = successes[0].user.email
            assert winner_email in (email_a, email_b)
    finally:
        async with db_client.async_session() as session:
            async with session.begin():
                await session.execute(
                    delete(UserModel).where(UserModel.email.in_([email_a, email_b]))
                )
