"""Reset Dograh application data, preserving one owner identity.

Target database comes from ``DATABASE_URL`` only. There is no fallback and no
hard-coded host, so the script always follows whatever database the running
deployment is configured against.

What survives a reset:

- the ``users`` row for ``--owner-email`` (default: nagarajay@gmail.com)
- the ``organizations`` row that user's ``selected_organization_id`` points at
- the ``organization_users`` membership joining those two
- ``alembic_version`` and ``workflow_templates`` (schema/migration state and
  shipped templates, not per-tenant data)
- that user's ``user_configurations`` rows for the keys in
  ``PRESERVED_USER_CONFIGURATION_KEYS``, and its organization's
  ``organization_configurations`` rows for the keys in
  ``PRESERVED_ORGANIZATION_CONFIGURATION_KEYS`` -- account/default state only

Every table in ``CLEAR_TABLES`` is emptied, including API keys, telephony and
provider configuration rows that belong to the preserved owner. The intent is
"keep the identity, drop the operational state".

Both the clear list and the preserve list are explicit. Table discovery at
runtime is used only to validate them: if the ``public`` schema contains a
table classified as neither, the script aborts and names it, so a future
migration cannot cause new state to be wiped without review.

Non-``public`` schemas are never touched.

Usage (dry run is the default -- it only reports what it would do):

    python -m scripts.reset_app_data

    # inside the running api container, which already has DATABASE_URL
    docker compose exec api python -m scripts.reset_app_data

To actually delete, both confirmations are required: the ``--yes`` flag and
typing the target host back when prompted (or passing --confirm-host).

    python -m scripts.reset_app_data --yes
"""

import argparse
import asyncio
import os
import sys
from urllib.parse import urlparse

import asyncpg

DEFAULT_OWNER_EMAIL = "nagarajay@gmail.com"

# Tables that are not per-tenant operational data. Rows here are left alone
# except for the scoped identity deletes handled separately.
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

# organization_configurations: ORGANIZATION_PREFERENCES is account-level
# defaults (timezone, test call number). Everything else is provisioning or
# provider state -- MODEL_CONFIGURATION_V2 carries the minted MPS service key,
# LANGFUSE_CREDENTIALS and TELEPHONY_CONFIGURATION carry provider credentials,
# CONCURRENT_CALL_LIMIT is a capacity override with a code default.
#
# ORGANIZATION_BOOTSTRAP must be cleared alongside them. It is a completion
# sentinel, and ensure_organization_bootstrapped() short-circuits on the
# sentinel alone (api/services/organization_bootstrap.py:118), so keeping it
# while dropping the model configuration would strand the organization
# unprovisioned forever instead of re-provisioning on the owner's next sign-in.
PRESERVED_ORGANIZATION_CONFIGURATION_KEYS = ("ORGANIZATION_PREFERENCES",)

KEY_FILTERED_TABLES = frozenset(
    {"user_configurations", "organization_configurations"}
)

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
    """
    rows = await conn.fetch(
        "SELECT id, email, selected_organization_id FROM public.users WHERE email = $1",
        email,
    )
    if len(rows) != 1:
        raise SystemExit(
            f"refusing to reset: expected exactly 1 user with email {email!r}, "
            f"found {len(rows)}. Check DATABASE_URL points at the intended database."
        )
    owner = rows[0]
    if owner["selected_organization_id"] is None:
        raise SystemExit(
            f"refusing to reset: user {email!r} has no selected_organization_id, "
            "so there is no organization to preserve."
        )
    org_exists = await conn.fetchval(
        "SELECT 1 FROM public.organizations WHERE id = $1",
        owner["selected_organization_id"],
    )
    if not org_exists:
        raise SystemExit(
            f"refusing to reset: organization {owner['selected_organization_id']} "
            f"referenced by {email!r} does not exist."
        )
    return owner


async def run(args: argparse.Namespace) -> int:
    raw_dsn = os.environ.get("DATABASE_URL")
    if not raw_dsn:
        raise SystemExit("DATABASE_URL is not set; nothing to target.")
    dsn = normalize_dsn(raw_dsn)

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
        org_id = owner["selected_organization_id"]

        print(f"target: {describe_target(dsn)}")
        print(f"connected database: {await conn.fetchval('SELECT current_database()')}")
        print(
            f"preserving user id={owner['id']} email={owner['email']} "
            f"organization_id={org_id}"
        )
        print("\ncurrent row counts (public schema):")
        for table in tables:
            if table in PRESERVED_TABLES:
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
        other_orgs = await conn.fetchval(
            "SELECT count(*) FROM public.organizations WHERE id <> $1", org_id
        )
        other_members = await conn.fetchval(
            "SELECT count(*) FROM public.organization_users "
            "WHERE user_id <> $1 OR organization_id <> $2",
            owner["id"],
            org_id,
        )
        stale_user_config = await conn.fetchval(
            "SELECT count(*) FROM public.user_configurations "
            "WHERE user_id IS DISTINCT FROM $1 OR key <> ALL($2::text[])",
            owner["id"],
            list(PRESERVED_USER_CONFIGURATION_KEYS),
        )
        stale_org_config = await conn.fetchval(
            "SELECT count(*) FROM public.organization_configurations "
            "WHERE organization_id <> $1 OR key <> ALL($2::text[])",
            org_id,
            list(PRESERVED_ORGANIZATION_CONFIGURATION_KEYS),
        )
        print(
            f"\nscoped identity deletes: users={other_users} "
            f"organizations={other_orgs} organization_users={other_members}"
        )
        print(
            f"scoped config deletes: user_configurations={stale_user_config} "
            f"organization_configurations={stale_org_config}"
        )

        if not args.yes:
            print("\ndry run: no rows were deleted. Re-run with --yes to apply.")
            return 0

        confirm = args.confirm_host
        if confirm is None:
            host = urlparse(dsn).hostname or ""
            confirm = input(f"\nType the target host ({host}) to confirm: ").strip()
        if confirm != (urlparse(dsn).hostname or ""):
            raise SystemExit("host confirmation did not match; aborting.")

        async with conn.transaction():
            if clear_tables:
                quoted = ", ".join(f'public."{t}"' for t in clear_tables)
                await conn.execute(f"TRUNCATE {quoted} RESTART IDENTITY")
            await conn.execute(
                "DELETE FROM public.organization_users "
                "WHERE user_id <> $1 OR organization_id <> $2",
                owner["id"],
                org_id,
            )
            await conn.execute(
                "DELETE FROM public.users WHERE id <> $1", owner["id"]
            )
            await conn.execute(
                "DELETE FROM public.organizations WHERE id <> $1", org_id
            )
            await conn.execute(
                "DELETE FROM public.user_configurations "
                "WHERE user_id IS DISTINCT FROM $1 OR key <> ALL($2::text[])",
                owner["id"],
                list(PRESERVED_USER_CONFIGURATION_KEYS),
            )
            await conn.execute(
                "DELETE FROM public.organization_configurations "
                "WHERE organization_id <> $1 OR key <> ALL($2::text[])",
                org_id,
                list(PRESERVED_ORGANIZATION_CONFIGURATION_KEYS),
            )

        after = await counts_for(conn, tables)
        print("\nrow counts after reset:")
        for table in tables:
            print(f"  {table:35} {before[table]:>8} -> {after[table]:>8}")
        return 0
    finally:
        await conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--owner-email",
        default=DEFAULT_OWNER_EMAIL,
        help=f"identity to preserve (default: {DEFAULT_OWNER_EMAIL})",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="actually delete; without it the script only reports.",
    )
    parser.add_argument(
        "--confirm-host",
        help="skip the interactive host prompt by passing the expected host.",
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
