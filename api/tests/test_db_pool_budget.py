"""Pool sizing is configurable and the compose defaults fit Supabase's limit."""

import importlib
import importlib.util
import re
from pathlib import Path

SUPABASE_SESSION_LIMIT = 15
COMPOSE = Path(__file__).resolve().parents[2] / "docker-compose.yaml"


def _compose_default(name: str) -> int:
    match = re.search(rf'{name}: "\$\{{{name}:-(\d+)\}}"', COMPOSE.read_text())
    assert match, f"{name} default missing from docker-compose.yaml"
    return int(match.group(1))


def test_engine_uses_configured_pool(monkeypatch):
    monkeypatch.setenv("DB_POOL_SIZE", "2")
    monkeypatch.setenv("DB_MAX_OVERFLOW", "1")
    monkeypatch.setenv("DB_POOL_TIMEOUT", "7")
    monkeypatch.setenv("DB_POOL_RECYCLE", "99")
    import api.constants as constants

    importlib.reload(constants)
    # Load by path: importing api.db.base_client would import every DB client.
    spec = importlib.util.spec_from_file_location(
        "_base_client_under_test",
        Path(__file__).resolve().parents[1] / "db" / "base_client.py",
    )
    base_client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_client)
    try:
        pool = base_client.BaseDBClient().engine.pool
        assert pool.size() == 2
        assert pool._max_overflow == 1
        assert pool._timeout == 7
        assert pool._recycle == 99
    finally:
        monkeypatch.undo()
        importlib.reload(constants)


def test_compose_default_budget_stays_under_limit():
    per_process = _compose_default("DB_POOL_SIZE") + _compose_default("DB_MAX_OVERFLOW")
    # ari_manager + campaign_orchestrator + 1 uvicorn + 1 arq
    default_processes = 4
    assert default_processes * per_process < SUPABASE_SESSION_LIMIT
