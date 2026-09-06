"""Reset Dograh application data to zero client organizations.

Organizations model client tenants, one Dograh organization per AVSIQ client.
A platform super-admin is not a client, so a clean database has exactly one
user -- the super-admin -- and *no* organizations at all. The reset therefore
preserves the super-admin identity and drops every organization and every
membership, including the bootstrap organization that signup used to create
for the super-admin themselves.

The owner is required to be a super-admin (``users.is_superuser``). That is a
safety property, not a formality: a reset leaves the surviving user with
``selected_organization_id = NULL``, which only a platform super-admin can use.
Preserving an ordinary user would leave an account that can authenticate but
cannot reach a single organization-scoped route.


Target database comes from ``DATABASE_URL`` only. There is no fallback and no
hard-coded host, so the script always follows whatever database the running
deployment is configured against.

Two environment variables decide whether that database may be reset at all, and
both are checked before the connection is opened:

- ``DOGRAH_ENV`` must be ``development``. A production or staging deployment
  fails here, with no read and no write, rather than relying on whoever ran the
  command having checked first.
- ``DOGRAH_SUPABASE_PROJECT_REF`` must equal the Supabase project reference
  derived from ``DATABASE_URL``. The hostname does not identify a project --
  pooled connections share a regional host, so several projects answer on
  ``aws-0-ap-south-1.pooler.supabase.com`` -- but the reference does, and it
  appears in the pooler username (``postgres.<ref>``) or in a direct host
  (``db.<ref>.supabase.co``).

Missing, unparseable or mismatched values all abort. Both halves of the
comparison are written down in advance (``.env`` at the repository root), so
nothing here can be satisfied by a value copied out of this script's own
output.

What survives a reset:

- the ``users`` row for ``--owner-email``, which is required and has no
  default: the identity to keep must be named deliberately on every run. Its
  ``selected_organization_id`` is set to NULL, because the organization it
  pointed at is deleted with the rest.
- ``alembic_version`` and ``workflow_templates`` (schema/migration state and
  shipped templates, not per-tenant data)
- that user's ``user_configurations`` rows for the keys in
  ``PRESERVED_USER_CONFIGURATION_KEYS`` -- account/default state only

What never survives:

- every row in ``organizations`` and ``organization_users``
- every row in ``organization_configurations``. With no organization left,
  ``PRESERVED_ORGANIZATION_CONFIGURATION_KEYS`` has nothing to attach to; the
  first client provisioned after a reset gets its own configuration from
  ``ensure_organization_bootstrapped()``.

Every table in ``CLEAR_TABLES`` is emptied, including API keys, telephony and
provider configuration rows that belong to the preserved owner. The intent is
"keep the identity, drop the operational state".

Both the clear list and the preserve list are explicit. Table discovery at
runtime is used only to validate them: if the ``public`` schema contains a
table classified as neither, the script aborts and names it, so a future
migration cannot cause new state to be wiped without review.

Non-``public`` schemas are never touched.

Usage (dry run is the default -- it only reports what it would do):

    python -m scripts.reset_app_data --owner-email owner@example.com

    # inside the running api container, which already has DATABASE_URL
    docker compose exec api python -m scripts.reset_app_data \
        --owner-email owner@example.com

Deleting requires ``--yes``. There is no second, typed confirmation: the
hostname does not identify a database -- a regional pooler host answers for
many projects -- so typing it back proved nothing that the project reference
check above does not already prove, and asking for it made the reset
non-scriptable for no safety gained.

    python -m scripts.reset_app_data --owner-email owner@example.com --yes

``scripts/reset-db.sh`` wraps all of this for the usual local case.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from collections.abc import Mapping
from urllib.parse import urlparse

import asyncpg

# The only value of DOGRAH_ENV this script will run against.
REQUIRED_ENVIRONMENT = "development"

# The organization_configurations key holding the Dograh-minted MPS service key.
MANAGED_MODEL_CONFIGURATION_KEY = "MODEL_CONFIGURATION_V2"

# Where a Supabase project reference can be read from a connection string. The
# pooled form is what a Supabase deployment normally uses; the direct form is
# what psql sessions and some migration tooling use.
_PROJECT_REF_IN_USER = re.compile(r"^postgres\.([a-z0-9]{16,})$", re.IGNORECASE)
_PROJECT_REF_IN_HOST = re.compile(
    r"^db\.([a-z0-9]{16,})\.supabase\.(co|com)$", re.IGNORECASE
)

# Tables that are not per-tenant operational data. Rows here are left alone
# except for the scoped identity deletes handled separately.
#
# organizations and organization_users are listed here rather than in
# CLEAR_TABLES because they are emptied by a scoped DELETE inside the same
# transaction as the identity deletes, not by TRUNCATE: users.selected_
# organization_id references organizations, so the FK has to be cleared first
# and a TRUNCATE ... RESTART IDENTITY would not respect that ordering.
PRESERVED_TABLES = frozenset(
    {
        "alembic_version",
        "workflow_templates",
        "users",
        "organizations",
        "organization_users",
    }
)

# The two keyed JSON config stores are neither wholly identity nor wholly
# operational, so they are filtered by key rather than truncated. Anything not
# listed here is operational state and goes, including for the preserved owner.
#
# user_configurations: ONBOARDING is the server-side post-signup gate and
# one-time tooltip state. MODEL_CONFIGURATION (legacy per-user v1) holds BYOK
# provider credentials, so it is not preserved.
PRESERVED_USER_CONFIGURATION_KEYS = ("ONBOARDING",)

KEY_FILTERED_TABLES = frozenset({"user_configurations"})

# Operational tables emptied outright. This is an explicit allowlist, reviewed
# table by table, and not a "everything not preserved" fallback: a migration
# that adds a table must be classified deliberately here or in PRESERVED_TABLES
# before this script will run, so new state is never silently wiped.
CLEAR_TABLES = (
    "agent_triggers",
    "api_keys",
    "campaigns",
    "embed_sessions",
    "embed_tokens",
    "external_credentials",
    "folders",
    "integrations",
    "knowledge_base_chunks",
    "knowledge_base_documents",
    # Every row here belongs to an organization, and no organization survives a
    # reset. ORGANIZATION_PREFERENCES used to be preserved by key; with zero
    # organizations there is nothing for it to hang off. The first client
    # provisioned after a reset is configured by
    # ensure_organization_bootstrapped(), which re-mints MODEL_CONFIGURATION_V2
    # and re-writes the ORGANIZATION_BOOTSTRAP sentinel.
    "organization_configurations",
    "organization_usage_cycles",
    "queued_runs",
    "telephony_configurations",
    "telephony_phone_numbers",
    "telephony_trunks",
    "tools",
    "webhook_deliveries",
    "workflow_definitions",
    "workflow_recordings",
    "workflow_run_text_sessions",
    "workflow_runs",
    "workflows",
)

CLASSIFIED_TABLES = frozenset(CLEAR_TABLES) | PRESERVED_TABLES | KEY_FILTERED_TABLES


def assert_tables_classified(tables: list[str]) -> None:
    """Refuse to run when the schema has drifted from the reviewed classification.

    Runtime discovery is a validation check only. An unclassified table is
    almost always a migration landing after this script was last reviewed, and
    guessing its disposition either wipes data nobody agreed to wipe or leaves
    tenant state behind in a reset that claimed to remove it.
    """
    unknown = sorted(set(tables) - CLASSIFIED_TABLES)
    if unknown:
        raise SystemExit(
            "refusing to reset: public tables not classified as CLEAR or PRESERVE:\n"
            + "\n".join(f"  {t}" for t in unknown)
            + "\nAdd each to CLEAR_TABLES or PRESERVED_TABLES after deciding "
            "whether it holds identity or operational state."
        )


def normalize_dsn(raw: str) -> str:
    """Strip the SQLAlchemy driver suffix so asyncpg accepts the URL."""
    for prefix, replacement in (
        ("postgresql+asyncpg://", "postgresql://"),
        ("postgresql+psycopg://", "postgresql://"),
        ("postgres://", "postgresql://"),
    ):
        if raw.startswith(prefix):
            return replacement + raw[len(prefix) :]
    return raw


def derive_project_ref(dsn: str) -> str | None:
    """Return the Supabase project reference a connection string names.

    ``None`` means the string names no project -- a container-local Postgres,
    say -- which is not an error here but is refused by the caller, because a
    target that cannot be identified cannot be confirmed either.
    """
    try:
        parsed = urlparse(dsn)
        username = parsed.username or ""
        hostname = parsed.hostname or ""
    except ValueError:
        return None

    match = _PROJECT_REF_IN_USER.match(username) or _PROJECT_REF_IN_HOST.match(hostname)
    return match.group(1) if match else None


def resolve_target_dsn(env: Mapping[str, str]) -> str:
    """Return the DSN to reset, or refuse before anything is opened.

    Every check that can be made without the database is made here: the
    deployment's environment, the presence and shape of the connection string,
    and whether it points at the one Supabase project this checkout is
    configured to reset. A wrong ``DATABASE_URL`` therefore never becomes a
    query against the wrong database, not even the dry run's reads.
    """
    environment = (env.get("DOGRAH_ENV") or "").strip()
    if not environment:
        raise SystemExit(
            "refusing to reset: DOGRAH_ENV is not set. This script runs only "
            f"against a {REQUIRED_ENVIRONMENT!r} deployment, and an unset value "
            "is not evidence that this is one. Set it in .env at the repository "
            "root."
        )
    if environment != REQUIRED_ENVIRONMENT:
        raise SystemExit(
            f"refusing to reset: DOGRAH_ENV is {environment!r}, not "
            f"{REQUIRED_ENVIRONMENT!r}. Nothing was read or deleted."
        )

    raw_dsn = (env.get("DATABASE_URL") or "").strip()
    if not raw_dsn:
        raise SystemExit("DATABASE_URL is not set; nothing to target.")
    dsn = normalize_dsn(raw_dsn)

    expected_ref = (env.get("DOGRAH_SUPABASE_PROJECT_REF") or "").strip()
    if not expected_ref:
        raise SystemExit(
            "refusing to reset: DOGRAH_SUPABASE_PROJECT_REF is not set. It is "
            "the Supabase project reference of the Dograh database, and the "
            "only value that tells one project apart from another sharing the "
            "same regional pooler hostname. Set it in .env at the repository "
            "root. Nothing was read or deleted."
        )

    actual_ref = derive_project_ref(dsn)
    if actual_ref is None:
        raise SystemExit(
            "refusing to reset: no Supabase project reference could be derived "
            f"from DATABASE_URL, so it cannot be checked against "
            f"DOGRAH_SUPABASE_PROJECT_REF ({expected_ref}). Expected either a "
            "pooled connection string whose user is `postgres.<project ref>`, "
            "or a direct host `db.<project ref>.supabase.co`. Nothing was read "
            "or deleted."
        )
    if actual_ref != expected_ref:
        raise SystemExit(
            f"refusing to reset: DATABASE_URL points at Supabase project "
            f"{actual_ref!r}, but DOGRAH_SUPABASE_PROJECT_REF is "
            f"{expected_ref!r}. Fix whichever is wrong. Nothing was read or "
            "deleted."
        )

    return dsn


def describe_target(dsn: str) -> str:
    parsed = urlparse(dsn)
    return (
        f"host={parsed.hostname} port={parsed.port or 5432} "
        f"database={(parsed.path or '/').lstrip('/')} user={parsed.username}"
    )


async def list_public_tables(conn: asyncpg.Connection) -> list[str]:
    rows = await conn.fetch(
        """
        SELECT c.relname
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind = 'r'
        ORDER BY c.relname
        """
    )
    return [r["relname"] for r in rows]


async def counts_for(conn: asyncpg.Connection, tables: list[str]) -> dict[str, int]:
    return {
        t: await conn.fetchval(f'SELECT count(*) FROM public."{t}"') for t in tables
    }


async def resolve_owner(conn: asyncpg.Connection, email: str) -> asyncpg.Record:
    """Find the identity to preserve, or refuse to continue.

    A missing owner almost always means DATABASE_URL points at a database this
    script was never meant to touch, so that is a hard stop rather than a
    reset that would empty the whole database.

    The owner must be a super-admin. A reset ends with zero organizations, so
    the surviving user necessarily has ``selected_organization_id = NULL``;
    that state is usable by a platform super-admin and by nobody else. An
    ordinary user preserved here could sign in and reach nothing.

    Whether the owner currently *has* an organization is not checked: the
    bootstrap organization signup created for them is exactly what this reset
    is meant to remove.
    """
    rows = await conn.fetch(
        "SELECT id, email, provider_id, is_superuser, selected_organization_id "
        "FROM public.users WHERE email = $1",
        email,
    )
    if len(rows) != 1:
        raise SystemExit(
            f"refusing to reset: expected exactly 1 user with email {email!r}, "
            f"found {len(rows)}. Check DATABASE_URL points at the intended database."
        )
    owner = rows[0]
    if not owner["is_superuser"]:
        raise SystemExit(
            f"refusing to reset: user {email!r} is not a super-admin. A reset "
            "leaves the preserved user with no organization, which only a "
            "platform super-admin can use. Promote the account first:\n"
            f"  python -m scripts.bootstrap_superadmin --email {email}"
        )
    return owner


def extract_dograh_service_key(value) -> str | None:
    """Return the MPS service key a MODEL_CONFIGURATION_V2 row carries.

    Anything that is not a Dograh-managed v2 configuration yields None: a BYOK
    organization has no minted key to archive, and a malformed row is not a
    reason to abort a reset.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, Mapping):
        return None
    dograh = value.get("dograh")
    if not isinstance(dograh, Mapping):
        return None
    api_key = dograh.get("api_key")
    return api_key if isinstance(api_key, str) and api_key else None


async def resolve_managed_service_keys(conn: asyncpg.Connection) -> list[str]:
    """The Dograh-minted service keys this reset is about to drop the record of.

    ``create_service_key`` is not idempotent, and the reset deletes the only
    copy of the key it minted. Without archiving, every reset leaves one more
    live, non-expiring credential behind at MPS with nothing in the database
    pointing at it.
    """
    rows = await conn.fetch(
        "SELECT value FROM public.organization_configurations WHERE key = $1",
        MANAGED_MODEL_CONFIGURATION_KEY,
    )
    keys = [extract_dograh_service_key(row["value"]) for row in rows]
    return [key for key in keys if key]


async def archive_managed_service_keys(
    service_keys: list[str], *, created_by: str
) -> None:
    """Archive exactly the keys this reset is about to orphan, or abort.

    Keys are matched to MPS ids by prefix, and only keys the database actually
    references are archived: an unrelated key belonging to the same creator is
    somebody else's business, and archiving is irreversible.

    A failure here aborts before anything is deleted. The alternative -- delete
    the row anyway -- is what produced the orphans this exists to stop.
    """
    if not service_keys:
        return

    # Imported here, not at module scope: this script is also loaded by path
    # outside the api container (see the tests), where the api package and its
    # dependencies are not importable.
    from api.services.mps_service_key_client import mps_service_key_client

    try:
        existing = await mps_service_key_client.get_service_keys(
            organization_id=None, created_by=created_by
        )
    except Exception as exc:
        raise SystemExit(
            f"refusing to reset: could not list MPS service keys ({exc}). "
            "Deleting the model configuration now would orphan a live key."
        ) from exc

    for service_key in service_keys:
        matches = [
            candidate
            for candidate in existing
            if candidate.get("key_prefix")
            and service_key.startswith(candidate["key_prefix"])
        ]
        if len(matches) != 1:
            raise SystemExit(
                "refusing to reset: expected exactly 1 MPS service key matching "
                f"the configured key, found {len(matches)}. Archive it by hand "
                "before resetting, so no live credential is orphaned."
            )
        key_id = matches[0]["id"]
        try:
            archived = await mps_service_key_client.archive_service_key(
                key_id, organization_id=None, created_by=created_by
            )
        except Exception as exc:
            raise SystemExit(
                f"refusing to reset: archiving MPS service key {key_id} failed "
                f"({exc}). Nothing was deleted."
            ) from exc
        if not archived:
            raise SystemExit(
                f"refusing to reset: MPS refused to archive service key {key_id}. "
                "Nothing was deleted."
            )
        print(f"archived MPS service key {key_id}")


async def apply_reset(
    conn: asyncpg.Connection, *, owner_id: int, clear_tables: list[str]
) -> None:
    """Empty the database down to one super-admin and no organizations.

    Every statement is here rather than inline in ``run()`` so the end state can
    be asserted against a real database without going through the environment
    guards, the argument parser and a live connection to a Supabase project.
    The caller owns the transaction: either all of this lands or none of it does.
    """
    if clear_tables:
        quoted = ", ".join(f'public."{t}"' for t in clear_tables)
        await conn.execute(f"TRUNCATE {quoted} RESTART IDENTITY")

    # Before the users delete, not after: user_configurations has a foreign key
    # to users, so a non-owner who holds any configuration row would otherwise
    # make the whole reset fail on a constraint violation.
    await conn.execute(
        "DELETE FROM public.user_configurations "
        "WHERE user_id IS DISTINCT FROM $1 OR key <> ALL($2::text[])",
        owner_id,
        list(PRESERVED_USER_CONFIGURATION_KEYS),
    )
    await conn.execute("DELETE FROM public.organization_users")
    await conn.execute("DELETE FROM public.users WHERE id <> $1", owner_id)
    # Must precede the organizations delete: users.selected_organization_id is a
    # foreign key into the table about to be emptied.
    await conn.execute(
        "UPDATE public.users SET selected_organization_id = NULL "
        "WHERE selected_organization_id IS NOT NULL"
    )
    await conn.execute("DELETE FROM public.organizations")
    # DELETE leaves the sequence where it was, so the first client provisioned
    # after a reset would be organization 7 or 41. The CLEAR_TABLES truncate
    # above restarts its own identities; do the same here so a clean database
    # really does start from 1.
    await conn.execute(
        "SELECT setval(pg_get_serial_sequence('public.organizations', 'id'), 1, false)"
    )


async def run(args: argparse.Namespace) -> int:
    # Environment, connection string and project reference are all settled
    # before asyncpg is asked for a connection.
    dsn = resolve_target_dsn(os.environ)

    # Supabase's session pooler rejects the prepared statements asyncpg caches.
    conn = await asyncpg.connect(dsn, statement_cache_size=0)
    try:
        tables = await list_public_tables(conn)
        assert_tables_classified(tables)
        # Intersect rather than use CLEAR_TABLES directly: a table listed here
        # but dropped by a later migration should not fail the TRUNCATE.
        clear_tables = [t for t in tables if t in frozenset(CLEAR_TABLES)]
        missing = sorted(set(CLEAR_TABLES) - set(tables))
        before = await counts_for(conn, tables)
        owner = await resolve_owner(conn, args.owner_email)

        print(f"environment: {REQUIRED_ENVIRONMENT}")
        print(f"supabase project: {derive_project_ref(dsn)}")
        print(f"target: {describe_target(dsn)}")
        print(f"connected database: {await conn.fetchval('SELECT current_database()')}")
        print(
            f"preserving super-admin id={owner['id']} email={owner['email']} "
            f"(selected_organization_id will be set to NULL)"
        )
        print("\ncurrent row counts (public schema):")
        for table in tables:
            if table in ("organizations", "organization_users"):
                marker = "clear (no client organizations survive)"
            elif table in PRESERVED_TABLES:
                marker = "keep"
            elif table in KEY_FILTERED_TABLES:
                marker = "keep owner rows for preserved keys"
            else:
                marker = "clear"
            print(f"  {table:35} {before[table]:>8}  [{marker}]")
        if missing:
            print(
                "\nnote: CLEAR_TABLES entries no longer present in the schema: "
                + ", ".join(missing)
            )

        other_users = await conn.fetchval(
            "SELECT count(*) FROM public.users WHERE id <> $1", owner["id"]
        )
        all_orgs = await conn.fetchval("SELECT count(*) FROM public.organizations")
        all_members = await conn.fetchval(
            "SELECT count(*) FROM public.organization_users"
        )
        stale_user_config = await conn.fetchval(
            "SELECT count(*) FROM public.user_configurations "
            "WHERE user_id IS DISTINCT FROM $1 OR key <> ALL($2::text[])",
            owner["id"],
            list(PRESERVED_USER_CONFIGURATION_KEYS),
        )
        print(
            f"\nidentity deletes: users={other_users} "
            f"organizations={all_orgs} organization_users={all_members}"
        )
        print(f"scoped config deletes: user_configurations={stale_user_config}")

        managed_service_keys = await resolve_managed_service_keys(conn)
        if managed_service_keys:
            print(
                f"\nMPS service keys to archive: {len(managed_service_keys)} "
                f"(referenced by {MANAGED_MODEL_CONFIGURATION_KEY}; archived "
                "before any row is deleted, and archival failure aborts the reset)"
            )
        else:
            print("\nMPS service keys to archive: none")

        if not args.yes:
            print("\ndry run: no rows were deleted. Re-run with --yes to apply.")
            return 0

        # Outside the transaction and before it: archiving is an outbound call
        # that cannot be rolled back, so it must either succeed or stop the
        # reset while the database still holds the key it refers to.
        await archive_managed_service_keys(
            managed_service_keys, created_by=owner["provider_id"]
        )

        async with conn.transaction():
            await apply_reset(conn, owner_id=owner["id"], clear_tables=clear_tables)

        after = await counts_for(conn, tables)
        print("\nrow counts after reset:")
        for table in tables:
            print(f"  {table:35} {before[table]:>8} -> {after[table]:>8}")
        return 0
    finally:
        await conn.close()


def build_parser() -> argparse.ArgumentParser:
    """The command line, separated from main() so it can be tested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--owner-email",
        required=True,
        help=(
            "identity to preserve; required, so the account that survives is "
            "always named deliberately rather than inherited from a default."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "actually delete; without it the script only reports. The target "
            "was already settled by DOGRAH_ENV and "
            "DOGRAH_SUPABASE_PROJECT_REF before the connection was opened, so "
            "this is the only confirmation asked for."
        ),
    )
    return parser


def main() -> int:
    return asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    sys.exit(main())
