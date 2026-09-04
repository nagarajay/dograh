from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.routes.auth as auth_routes
from api.routes.auth import router
from api.services.auth import depends as auth_depends
from api.services.auth.depends import get_user


def _make_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_stack_mode_hides_email_password_auth_routes(monkeypatch):
    monkeypatch.setattr(auth_depends, "AUTH_PROVIDER", "stack")
    client = TestClient(_make_test_app())

    signup_response = client.post(
        "/auth/signup",
        json={
            "email": "user@example.com",
            "password": "password123",
            "name": "User",
        },
    )
    login_response = client.post(
        "/auth/login",
        json={
            "email": "user@example.com",
            "password": "password123",
        },
    )

    assert signup_response.status_code == 404
    assert signup_response.json() == {"detail": "Not found"}
    assert login_response.status_code == 404
    assert login_response.json() == {"detail": "Not found"}


def test_signup_disabled_returns_403(monkeypatch):
    monkeypatch.setattr(auth_routes, "ENABLE_SIGNUP", False)
    client = TestClient(_make_test_app())

    response = client.post(
        "/auth/signup",
        json={
            "email": "user@example.com",
            "password": "password123",
            "name": "User",
        },
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Signup is disabled"}


def test_stack_mode_keeps_current_user_route_available(monkeypatch):
    monkeypatch.setattr(auth_depends, "AUTH_PROVIDER", "stack")
    app = _make_test_app()
    app.dependency_overrides[get_user] = lambda: SimpleNamespace(
        id=7,
        email="user@example.com",
        selected_organization_id=42,
        provider_id="stack-user-1",
    )
    client = TestClient(app)

    response = client.get("/auth/me")

    assert response.status_code == 200
    assert response.json() == {
        "id": 7,
        "email": "user@example.com",
        "name": None,
        "organization_id": 42,
        "provider_id": "stack-user-1",
    }


def test_signup_rejects_a_duplicate_external_reference(monkeypatch):
    """Provisioning twice with one reference must not create a second customer.

    ``ENABLE_SIGNUP`` is patched on the imported module for this test only, so
    the route is reachable here without the deployment enabling public signup.
    """
    from unittest.mock import AsyncMock

    monkeypatch.setattr(auth_routes, "ENABLE_SIGNUP", True)
    monkeypatch.setattr(auth_depends, "AUTH_PROVIDER", "local")

    created_user = SimpleNamespace(
        id=11, email="dupe@example.com", provider_id="oss-dupe"
    )
    get_or_create_organization = AsyncMock()
    monkeypatch.setattr(
        auth_routes.db_client, "get_user_by_email", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        auth_routes.db_client,
        "create_user_with_email",
        AsyncMock(return_value=created_user),
    )
    monkeypatch.setattr(
        auth_routes.db_client,
        "get_organization_by_external_reference",
        AsyncMock(return_value=SimpleNamespace(id=5)),
    )
    monkeypatch.setattr(
        auth_routes.db_client,
        "get_or_create_organization_by_provider_id",
        get_or_create_organization,
    )

    client = TestClient(_make_test_app())
    response = client.post(
        "/auth/signup",
        json={
            "email": "dupe@example.com",
            "password": "password123",
            "name": "Dupe",
            "organization_display_name": "Second Customer",
            "organization_external_reference": "avsiq-client-8821",
        },
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Organization external reference already registered"
    }
    # The conflict is detected before any organization is created.
    get_or_create_organization.assert_not_awaited()
