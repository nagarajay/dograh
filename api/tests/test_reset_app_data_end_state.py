"""What a reset leaves behind: one super-admin, zero organizations.

The guard tests in ``test_reset_app_data_guards.py`` cover everything that runs
*before* a connection is opened. These cover the other end -- the statements
that actually delete -- against a real database, because the property that
matters here ("no organization survives") is a property of the SQL, not of the
argument parser.

Each test opens its own asyncpg connection and rolls back, so nothing is
committed and the tables the rest of the suite uses are left alone.
"""

import importlib.util
import os
import sys
from pathlib import Path

import asyncpg
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "reset_app_data_end_state_under_test",
    REPO_ROOT / "scripts" / "reset_app_data.py",
)
assert _spec is not None and _spec.loader is not None
reset_app_data = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = reset_app_data
_spec.loader.exec_module(reset_app_data)


@pytest.fixture
async def raw_conn(setup_test_database):
    """A rolled-back asyncpg connection to the test database.

    asyncpg rather than the shared SQLAlchemy session because the script under
    test speaks asyncpg, and the point is to exercise its statements verbatim.
    ``setup_test_database`` is depended on so migrations have run.
    """
    dsn = reset_app_data.normalize_dsn(os.environ["DATABASE_URL"])
    conn = await asyncpg.connect(dsn, statement_cache_size=0)
    transaction = conn.transaction()
    await transaction.start()
    try:
        yield conn
    finally:
        await transaction.rollback()
        await conn.close()


async def _seed(conn, *, owner_is_superuser: bool = True) -> dict:
    """A database as it looks before a reset: an owner, plus a client tenant."""
    await conn.execute("DELETE FROM public.organization_users")
    await conn.execute("UPDATE public.users SET selected_organization_id = NULL")
    await conn.execute("DELETE FROM public.users")
    await conn.execute("DELETE FROM public.organizations")

    owner_org = await conn.fetchval(
        "INSERT INTO public.organizations (provider_id, created_at) "
        "VALUES ('org_bootstrap_owner', now()) RETURNING id"
    )
    client_org = await conn.fetchval(
        "INSERT INTO public.organizations "
        "(provider_id, created_at, display_name, external_reference) "
        "VALUES ('org_client', now(), 'Northwind', 'avsiq-client-0001') "
        "RETURNING id"
    )
    owner_id = await conn.fetchval(
        "INSERT INTO public.users "
        "(provider_id, email, is_superuser, selected_organization_id, created_at) "
        "VALUES ('oss-owner', 'owner@example.com', $1, $2, now()) RETURNING id",
        owner_is_superuser,
        owner_org,
    )
    client_user_id = await conn.fetchval(
        "INSERT INTO public.users "
        "(provider_id, email, is_superuser, selected_organization_id, created_at) "
        "VALUES ('oss-client', 'client@example.com', false, $1, now()) RETURNING id",
        client_org,
    )
    await conn.executemany(
        "INSERT INTO public.organization_users (user_id, organization_id) "
        "VALUES ($1, $2)",
        [(owner_id, owner_org), (client_user_id, client_org)],
    )
    return {
        "owner_id": owner_id,
        "owner_org": owner_org,
        "client_org": client_org,
        "client_user_id": client_user_id,
    }


@pytest.mark.asyncio
async def test_reset_leaves_zero_organizations_and_one_user(raw_conn):
    """The headline clean state: super-admin present, no client organizations."""
    seeded = await _seed(raw_conn)
    assert await raw_conn.fetchval("SELECT count(*) FROM public.organizations") == 2

    await reset_app_data.apply_reset(
        raw_conn, owner_id=seeded["owner_id"], clear_tables=[]
    )

    assert await raw_conn.fetchval("SELECT count(*) FROM public.organizations") == 0
    assert (
        await raw_conn.fetchval("SELECT count(*) FROM public.organization_users") == 0
    )
    assert await raw_conn.fetchval("SELECT count(*) FROM public.users") == 1
    survivor = await raw_conn.fetchrow(
        "SELECT id, email, is_superuser, selected_organization_id FROM public.users"
    )
    assert survivor["id"] == seeded["owner_id"]
    assert survivor["is_superuser"] is True
    assert survivor["selected_organization_id"] is None


@pytest.mark.asyncio
async def test_the_owners_bootstrap_organization_is_deleted_too(raw_conn):
    """The organization signup minted for the super-admin is not a client.

    This is the whole point of the change: the reset used to preserve exactly
    that row, so a clean database showed one organization and the console read
    it as a client.
    """
    seeded = await _seed(raw_conn)

    await reset_app_data.apply_reset(
        raw_conn, owner_id=seeded["owner_id"], clear_tables=[]
    )

    assert not await raw_conn.fetchval(
        "SELECT 1 FROM public.organizations WHERE id = $1", seeded["owner_org"]
    )


@pytest.mark.asyncio
async def test_the_next_organization_starts_from_one(raw_conn):
    """A clean database really is clean: the first client is organization 1."""
    seeded = await _seed(raw_conn)

    await reset_app_data.apply_reset(
        raw_conn, owner_id=seeded["owner_id"], clear_tables=[]
    )

    first_client = await raw_conn.fetchval(
        "INSERT INTO public.organizations (provider_id, created_at) "
        "VALUES ('org_first_client', now()) RETURNING id"
    )
    assert first_client == 1


@pytest.mark.asyncio
async def test_preserved_user_configuration_keys_survive(raw_conn):
    seeded = await _seed(raw_conn)
    await raw_conn.executemany(
        "INSERT INTO public.user_configurations (user_id, key, configuration) "
        "VALUES ($1, $2, $3::json)",
        [
            (seeded["owner_id"], "ONBOARDING", "{}"),
            (seeded["owner_id"], "MODEL_CONFIGURATION", '{"secret": "byok"}'),
            (seeded["client_user_id"], "ONBOARDING", "{}"),
        ],
    )

    await reset_app_data.apply_reset(
        raw_conn, owner_id=seeded["owner_id"], clear_tables=[]
    )

    rows = await raw_conn.fetch("SELECT user_id, key FROM public.user_configurations")
    assert [(r["user_id"], r["key"]) for r in rows] == [
        (seeded["owner_id"], "ONBOARDING")
    ]


@pytest.mark.asyncio
async def test_resolve_owner_refuses_a_user_who_is_not_a_super_admin(raw_conn):
    """A reset ends with no organization, which only a super-admin can use."""
    await _seed(raw_conn, owner_is_superuser=False)

    with pytest.raises(SystemExit, match="not a super-admin"):
        await reset_app_data.resolve_owner(raw_conn, "owner@example.com")


@pytest.mark.asyncio
async def test_resolve_owner_accepts_a_super_admin_with_no_organization(raw_conn):
    """Running the reset twice must not fail on the state the first one left.

    The old resolver required the owner to have an organization, so a second
    run refused -- exactly when the database was already in the intended state.
    """
    seeded = await _seed(raw_conn)
    await reset_app_data.apply_reset(
        raw_conn, owner_id=seeded["owner_id"], clear_tables=[]
    )

    owner = await reset_app_data.resolve_owner(raw_conn, "owner@example.com")

    assert owner["id"] == seeded["owner_id"]
    assert owner["selected_organization_id"] is None


@pytest.mark.asyncio
async def test_resolve_owner_refuses_an_unknown_email(raw_conn):
    await _seed(raw_conn)

    with pytest.raises(SystemExit, match="expected exactly 1 user"):
        await reset_app_data.resolve_owner(raw_conn, "nobody@example.com")


class TestManagedServiceKeyExtraction:
    """Which configuration rows name a key the reset is about to orphan."""

    def test_reads_the_key_from_a_dograh_managed_configuration(self):
        assert (
            reset_app_data.extract_dograh_service_key(
                {"version": 2, "mode": "dograh", "dograh": {"api_key": "oss_sk_abc"}}
            )
            == "oss_sk_abc"
        )

    def test_accepts_a_json_encoded_row(self):
        assert (
            reset_app_data.extract_dograh_service_key(
                '{"mode": "dograh", "dograh": {"api_key": "oss_sk_abc"}}'
            )
            == "oss_sk_abc"
        )

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "not json",
            {},
            {"mode": "byok"},
            {"dograh": {}},
            {"dograh": {"api_key": ""}},
            {"dograh": "not a mapping"},
        ],
    )
    def test_anything_without_a_minted_key_yields_none(self, value):
        # A BYOK organization has no key to archive, and a malformed row is not
        # a reason to abort a reset.
        assert reset_app_data.extract_dograh_service_key(value) is None


@pytest.mark.asyncio
async def test_resolve_managed_service_keys_reads_only_managed_rows(raw_conn):
    seeded = await _seed(raw_conn)
    await raw_conn.executemany(
        "INSERT INTO public.organization_configurations "
        "(organization_id, key, value, created_at, updated_at) "
        "VALUES ($1, $2, $3::json, now(), now())",
        [
            (
                seeded["client_org"],
                "MODEL_CONFIGURATION_V2",
                '{"mode": "dograh", "dograh": {"api_key": "oss_sk_live"}}',
            ),
            (seeded["client_org"], "ORGANIZATION_BOOTSTRAP", '{"status": "completed"}'),
            (
                seeded["owner_org"],
                "MODEL_CONFIGURATION_V2",
                '{"mode": "byok", "openai": {"api_key": "sk-user-owned"}}',
            ),
        ],
    )

    keys = await reset_app_data.resolve_managed_service_keys(raw_conn)

    # Only the Dograh-minted key: the sentinel carries none, and a BYOK key is
    # the customer's own credential, not one this reset minted.
    assert keys == ["oss_sk_live"]


class TestArchiveManagedServiceKeys:
    """Archival is irreversible, so it must be exact and it must fail closed."""

    @staticmethod
    def _client(monkeypatch, *, existing, archive_result=True, archive_error=None):
        from api.services import mps_service_key_client as module

        archived: list[int] = []

        async def get_service_keys(organization_id=None, created_by=None, **_):
            return existing

        async def archive_service_key(key_id, organization_id=None, created_by=None):
            if archive_error is not None:
                raise archive_error
            archived.append(key_id)
            return archive_result

        monkeypatch.setattr(
            module.mps_service_key_client, "get_service_keys", get_service_keys
        )
        monkeypatch.setattr(
            module.mps_service_key_client, "archive_service_key", archive_service_key
        )
        return archived

    @pytest.mark.asyncio
    async def test_archives_only_the_referenced_key(self, monkeypatch):
        archived = self._client(
            monkeypatch,
            existing=[
                {"id": 1, "key_prefix": "oss_sk_AAA"},
                {"id": 2, "key_prefix": "oss_sk_BBB"},
                {"id": 3, "key_prefix": "oss_sk_CCC"},
            ],
        )

        await reset_app_data.archive_managed_service_keys(
            ["oss_sk_BBBrestofkey"], created_by="oss-owner"
        )

        assert archived == [2]

    @pytest.mark.asyncio
    async def test_no_keys_means_no_calls(self, monkeypatch):
        archived = self._client(monkeypatch, existing=[{"id": 1, "key_prefix": "x"}])

        await reset_app_data.archive_managed_service_keys([], created_by="oss-owner")

        assert archived == []

    @pytest.mark.asyncio
    async def test_aborts_when_the_key_matches_nothing(self, monkeypatch):
        self._client(monkeypatch, existing=[{"id": 1, "key_prefix": "oss_sk_AAA"}])

        with pytest.raises(SystemExit, match="found 0"):
            await reset_app_data.archive_managed_service_keys(
                ["oss_sk_ZZZ"], created_by="oss-owner"
            )

    @pytest.mark.asyncio
    async def test_aborts_when_the_match_is_ambiguous(self, monkeypatch):
        self._client(
            monkeypatch,
            existing=[
                {"id": 1, "key_prefix": "oss_sk_A"},
                {"id": 2, "key_prefix": "oss_sk_AA"},
            ],
        )

        with pytest.raises(SystemExit, match="found 2"):
            await reset_app_data.archive_managed_service_keys(
                ["oss_sk_AAA"], created_by="oss-owner"
            )

    @pytest.mark.asyncio
    async def test_aborts_when_mps_refuses_the_archive(self, monkeypatch):
        self._client(
            monkeypatch,
            existing=[{"id": 7, "key_prefix": "oss_sk_AAA"}],
            archive_result=False,
        )

        with pytest.raises(SystemExit, match="refused to archive"):
            await reset_app_data.archive_managed_service_keys(
                ["oss_sk_AAAkey"], created_by="oss-owner"
            )

    @pytest.mark.asyncio
    async def test_aborts_when_mps_is_unreachable(self, monkeypatch):
        self._client(
            monkeypatch,
            existing=[{"id": 7, "key_prefix": "oss_sk_AAA"}],
            archive_error=RuntimeError("connection refused"),
        )

        with pytest.raises(SystemExit, match="Nothing was deleted"):
            await reset_app_data.archive_managed_service_keys(
                ["oss_sk_AAAkey"], created_by="oss-owner"
            )

    @pytest.mark.asyncio
    async def test_aborts_when_the_key_list_cannot_be_read(self, monkeypatch):
        from api.services import mps_service_key_client as module

        async def boom(**_):
            raise RuntimeError("MPS down")

        monkeypatch.setattr(module.mps_service_key_client, "get_service_keys", boom)

        with pytest.raises(SystemExit, match="could not list MPS service keys"):
            await reset_app_data.archive_managed_service_keys(
                ["oss_sk_AAAkey"], created_by="oss-owner"
            )
