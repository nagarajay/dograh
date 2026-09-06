"""Guards that decide whether scripts/reset_app_data.py may touch a database.

These are the checks that run before asyncpg is asked for a connection, so
every case here is exercised without a database and without deleting anything:
what is under test is precisely the code that refuses.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Loaded by path rather than imported as `scripts.reset_app_data`: `scripts` is
# not a package on the test path, and the module is also run as a plain file
# inside the api container (see scripts/reset-db.sh), so loading it this way is
# the arrangement the wrapper actually uses.
_spec = importlib.util.spec_from_file_location(
    "reset_app_data_under_test", REPO_ROOT / "scripts" / "reset_app_data.py"
)
assert _spec is not None and _spec.loader is not None
reset_app_data = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = reset_app_data
_spec.loader.exec_module(reset_app_data)

PROJECT_REF = "tyrthfbmldwxcryhtkqp"
POOLED = (
    f"postgresql+asyncpg://postgres.{PROJECT_REF}:pw"
    "@aws-0-ap-south-1.pooler.supabase.com:5432/postgres"
)
DIRECT = f"postgresql://postgres:pw@db.{PROJECT_REF}.supabase.co:5432/postgres"
OTHER_PROJECT = (
    "postgresql+asyncpg://postgres.ormecixneglrqlqnseou:pw"
    "@aws-0-ap-south-1.pooler.supabase.com:5432/postgres"
)
CONTAINER_LOCAL = "postgresql+asyncpg://postgres:postgres@postgres:5432/postgres"


def env(**overrides: str) -> dict[str, str]:
    """A valid development environment, with the given keys replaced."""
    base = {
        "DOGRAH_ENV": "development",
        "DOGRAH_SUPABASE_PROJECT_REF": PROJECT_REF,
        "DATABASE_URL": POOLED,
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


class TestDeriveProjectRef:
    def test_reads_the_reference_from_a_pooled_username(self):
        assert reset_app_data.derive_project_ref(POOLED) == PROJECT_REF

    def test_reads_the_reference_from_a_direct_host(self):
        assert reset_app_data.derive_project_ref(DIRECT) == PROJECT_REF

    def test_a_container_local_database_names_no_project(self):
        assert reset_app_data.derive_project_ref(CONTAINER_LOCAL) is None

    def test_a_regional_pooler_host_alone_names_no_project(self):
        # Every project in the region answers on this host, so it identifies
        # nothing on its own.
        assert (
            reset_app_data.derive_project_ref(
                "postgresql://postgres:pw@aws-0-ap-south-1.pooler.supabase.com:5432/postgres"
            )
            is None
        )

    def test_an_unparseable_url_names_no_project(self):
        assert reset_app_data.derive_project_ref("not a url at all") is None


class TestResolveTargetDsn:
    def test_accepts_a_development_pool_pointing_at_the_configured_project(self):
        assert reset_app_data.resolve_target_dsn(env()) == POOLED.replace(
            "postgresql+asyncpg://", "postgresql://"
        )

    def test_accepts_a_direct_connection_to_the_configured_project(self):
        assert reset_app_data.resolve_target_dsn(env(DATABASE_URL=DIRECT)) == DIRECT

    @pytest.mark.parametrize("value", ["production", "staging", "Development", " "])
    def test_refuses_any_environment_that_is_not_development(self, value: str):
        with pytest.raises(SystemExit, match="DOGRAH_ENV"):
            reset_app_data.resolve_target_dsn(env(DOGRAH_ENV=value))

    def test_refuses_a_missing_environment(self):
        environ = env()
        del environ["DOGRAH_ENV"]
        with pytest.raises(SystemExit, match="DOGRAH_ENV is not set"):
            reset_app_data.resolve_target_dsn(environ)

    def test_refuses_a_missing_database_url(self):
        environ = env()
        del environ["DATABASE_URL"]
        with pytest.raises(SystemExit, match="DATABASE_URL is not set"):
            reset_app_data.resolve_target_dsn(environ)

    def test_refuses_a_missing_project_reference(self):
        environ = env()
        del environ["DOGRAH_SUPABASE_PROJECT_REF"]
        with pytest.raises(SystemExit, match="DOGRAH_SUPABASE_PROJECT_REF is not set"):
            reset_app_data.resolve_target_dsn(environ)

    def test_refuses_a_different_supabase_project(self):
        with pytest.raises(SystemExit, match="points at Supabase project"):
            reset_app_data.resolve_target_dsn(env(DATABASE_URL=OTHER_PROJECT))

    def test_refuses_a_database_url_with_no_derivable_reference(self):
        with pytest.raises(
            SystemExit, match="no Supabase project reference could be derived"
        ):
            reset_app_data.resolve_target_dsn(env(DATABASE_URL=CONTAINER_LOCAL))

    def test_refuses_an_unparseable_database_url(self):
        with pytest.raises(SystemExit):
            reset_app_data.resolve_target_dsn(env(DATABASE_URL="not a url at all"))

    def test_the_environment_is_checked_before_the_target(self):
        # A production deployment fails on DOGRAH_ENV even when everything else
        # about the connection is fine, so a mismatch there is never the first
        # thing an operator sees.
        with pytest.raises(SystemExit, match="not 'development'"):
            reset_app_data.resolve_target_dsn(
                env(DOGRAH_ENV="production", DATABASE_URL=OTHER_PROJECT)
            )

    def test_every_refusal_says_nothing_was_touched(self):
        for environ in (
            env(DOGRAH_ENV="production"),
            env(DATABASE_URL=OTHER_PROJECT),
            env(DATABASE_URL=CONTAINER_LOCAL),
        ):
            with pytest.raises(SystemExit) as raised:
                reset_app_data.resolve_target_dsn(environ)
            assert "Nothing was read or deleted" in str(raised.value)


class TestCommandLine:
    """The confirmations the reset asks for, and the ones it deliberately does not."""

    def test_owner_email_is_required(self):
        parser = reset_app_data.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_dry_run_is_the_default(self):
        parser = reset_app_data.build_parser()
        assert parser.parse_args(["--owner-email", "a@b.c"]).yes is False

    def test_yes_is_what_asks_for_deletion(self):
        parser = reset_app_data.build_parser()
        assert parser.parse_args(["--owner-email", "a@b.c", "--yes"]).yes is True

    def test_there_is_no_hostname_confirmation(self):
        # A regional pooler hostname answers for many projects, so typing it
        # back confirmed nothing the project reference check does not already
        # confirm -- and it made the reset unrunnable without a terminal.
        parser = reset_app_data.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["--owner-email", "a@b.c", "--yes", "--confirm-host", "anything"]
            )

    def test_nothing_reads_from_stdin(self):
        source = (REPO_ROOT / "scripts" / "reset_app_data.py").read_text()
        assert "input(" not in source


class TestNormalizeDsn:
    @pytest.mark.parametrize(
        "raw",
        [
            "postgresql+asyncpg://u:p@h:5432/d",
            "postgresql+psycopg://u:p@h:5432/d",
            "postgres://u:p@h:5432/d",
        ],
    )
    def test_strips_driver_and_alias_prefixes(self, raw: str):
        assert reset_app_data.normalize_dsn(raw) == "postgresql://u:p@h:5432/d"

    def test_leaves_a_plain_url_alone(self):
        assert (
            reset_app_data.normalize_dsn("postgresql://u:p@h:5432/d")
            == "postgresql://u:p@h:5432/d"
        )


class TestTableClassification:
    def test_an_unclassified_table_aborts_the_run(self):
        with pytest.raises(SystemExit, match="brand_new_table"):
            reset_app_data.assert_tables_classified(
                sorted(reset_app_data.CLASSIFIED_TABLES) + ["brand_new_table"]
            )

    def test_the_reviewed_classification_passes(self):
        reset_app_data.assert_tables_classified(
            sorted(reset_app_data.CLASSIFIED_TABLES)
        )

    def test_clear_and_preserve_lists_do_not_overlap(self):
        # A table in both would be truncated and reported as kept.
        assert not (
            frozenset(reset_app_data.CLEAR_TABLES) & reset_app_data.PRESERVED_TABLES
        )
        assert not (
            frozenset(reset_app_data.CLEAR_TABLES) & reset_app_data.KEY_FILTERED_TABLES
        )

    def test_organization_configurations_is_cleared_outright(self):
        # Every row belongs to an organization, and no organization survives a
        # reset, so there is nothing left for a key filter to preserve.
        assert "organization_configurations" in reset_app_data.CLEAR_TABLES
        assert "organization_configurations" not in reset_app_data.KEY_FILTERED_TABLES

    def test_organization_tables_are_emptied_by_scoped_delete_not_truncate(self):
        # users.selected_organization_id points into organizations, so the FK
        # has to be cleared before the rows go. TRUNCATE cannot express that
        # ordering, which is why these two are not in CLEAR_TABLES.
        assert "organizations" in reset_app_data.PRESERVED_TABLES
        assert "organization_users" in reset_app_data.PRESERVED_TABLES
