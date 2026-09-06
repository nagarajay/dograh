"""Promote a Dograh user to platform super-admin.

``users.is_superuser`` is the only marker of platform authority, and nothing in
the application ever sets it: signup writes ``False`` (api/db/user_client.py)
and no route can raise it. Before this script the flag was a manual ``UPDATE``
typed into a psql session, which is exactly the kind of ad-hoc write that ends
up applied to the wrong row or the wrong database.

A platform super-admin is not a client. Organizations model client tenants, one
per AVSIQ client, so a super-admin holds no ``selected_organization_id`` and no
``organization_users`` membership. Their authority comes from the flag plus an
explicit target organization id on every super-admin route, never from
membership. ``--detach-organization`` performs that separation for an account
that was created through ordinary signup and therefore still carries the
bootstrap organization signup mints for every new user.

Target database comes from ``DATABASE_URL``, the same as the running deployment.
Unlike ``scripts/reset_app_data.py`` this script deletes nothing and is safe to
run against any environment, so it carries no ``DOGRAH_ENV`` gate -- promoting
an account in staging or production is a legitimate operation. It still refuses
to guess: the email must match exactly one user.

Usage:

    python -m scripts.bootstrap_superadmin --email owner@example.com

    # also drop the bootstrap organization membership signup created
    python -m scripts.bootstrap_superadmin --email owner@example.com \
        --detach-organization

    # inside the running api container, which already has DATABASE_URL
    docker compose exec api python -m scripts.bootstrap_superadmin \
        --email owner@example.com

``--detach-organization`` clears ``selected_organization_id`` and removes the
user's ``organization_users`` rows. It never deletes an organization: an
organization the super-admin was the only member of becomes a memberless row,
which the super-admin console still lists, and deciding whether that row is a
real client is a judgement this script will not make on its own.
"""

import argparse
import asyncio
import os
import sys

import asyncpg


def normalize_dsn(raw: str) -> str:
    """Strip the SQLAlchemy driver suffix so asyncpg accepts the URL.

    Duplicated from ``scripts/reset_app_data.py`` rather than imported: both
    scripts are run by absolute path inside a container as often as they are run
    as modules from the repository root, and an import between them would work
    in one of those two cases only.
    """
    for prefix, replacement in (
        ("postgresql+asyncpg://", "postgresql://"),
        ("postgresql+psycopg://", "postgresql://"),
        ("postgres://", "postgresql://"),
    ):
        if raw.startswith(prefix):
            return replacement + raw[len(prefix) :]
    return raw


async def resolve_user(conn: asyncpg.Connection, email: str) -> asyncpg.Record:
    """Return the single user with this email, or refuse.

    Zero matches usually means DATABASE_URL points somewhere unexpected; more
    than one means the email is not the identity this script assumed it was.
    Either way, guessing which row to promote is worse than stopping.
    """
    rows = await conn.fetch(
        "SELECT id, email, is_superuser, selected_organization_id "
        "FROM public.users WHERE lower(email) = lower($1)",
        email,
    )
    if len(rows) != 1:
        raise SystemExit(
            f"refusing to promote: expected exactly 1 user with email {email!r}, "
            f"found {len(rows)}. Check DATABASE_URL points at the intended "
            "database, and that the account has signed up."
        )
    return rows[0]


async def run(args: argparse.Namespace) -> int:
    raw_dsn = (os.environ.get("DATABASE_URL") or "").strip()
    if not raw_dsn:
        raise SystemExit("DATABASE_URL is not set; nothing to target.")

    # Supabase's session pooler rejects the prepared statements asyncpg caches.
    conn = await asyncpg.connect(normalize_dsn(raw_dsn), statement_cache_size=0)
    try:
        user = await resolve_user(conn, args.email)
        memberships = await conn.fetchval(
            "SELECT count(*) FROM public.organization_users WHERE user_id = $1",
            user["id"],
        )

        print(f"connected database: {await conn.fetchval('SELECT current_database()')}")
        print(
            f"user id={user['id']} email={user['email']} "
            f"is_superuser={user['is_superuser']} "
            f"selected_organization_id={user['selected_organization_id']} "
            f"organization_memberships={memberships}"
        )

        will_promote = not user["is_superuser"]
        will_detach = args.detach_organization and (
            user["selected_organization_id"] is not None or memberships
        )
        if not will_promote and not will_detach:
            print("\nnothing to do: already a super-admin with nothing to detach.")
            return 0

        print("\nplanned changes:")
        if will_promote:
            print("  set is_superuser = true")
        if will_detach:
            print("  set selected_organization_id = NULL")
            print(f"  delete {memberships} organization_users row(s)")

        if not args.yes:
            print("\ndry run: nothing was written. Re-run with --yes to apply.")
            return 0

        async with conn.transaction():
            if will_promote:
                await conn.execute(
                    "UPDATE public.users SET is_superuser = true WHERE id = $1",
                    user["id"],
                )
            if will_detach:
                await conn.execute(
                    "UPDATE public.users SET selected_organization_id = NULL "
                    "WHERE id = $1",
                    user["id"],
                )
                await conn.execute(
                    "DELETE FROM public.organization_users WHERE user_id = $1",
                    user["id"],
                )

        after = await conn.fetchrow(
            "SELECT is_superuser, selected_organization_id FROM public.users "
            "WHERE id = $1",
            user["id"],
        )
        print(
            f"\napplied: is_superuser={after['is_superuser']} "
            f"selected_organization_id={after['selected_organization_id']}"
        )
        return 0
    finally:
        await conn.close()


def build_parser() -> argparse.ArgumentParser:
    """The command line, separated from main() so it can be tested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--email",
        required=True,
        help="the account to promote; matched case-insensitively, exactly once.",
    )
    parser.add_argument(
        "--detach-organization",
        action="store_true",
        help=(
            "also clear selected_organization_id and remove the user's "
            "organization memberships, so the account is a platform identity "
            "rather than a member of a client tenant."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="actually write; without it the script only reports what it would do.",
    )
    return parser


def main() -> int:
    return asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    sys.exit(main())
