#!/usr/bin/env bash
#
# Reset the local Dograh development database to zero client organizations.
#
# What survives is the platform super-admin account and nothing else: no
# organizations, no memberships, no agents. Organizations model client tenants,
# so a database with no clients provisioned has no organization rows at all.
#
# The owner is fixed here rather than taken from the command line: this wrapper
# exists for one machine's development database, and an owner that can be typed
# is an owner that can be mistyped, which would preserve nobody and empty the
# users table. reset_app_data.py additionally refuses an owner that is not a
# super-admin, since the account it preserves ends up with no organization.
#
#   ./scripts/reset-db.sh          dry run — reports what would be deleted
#   ./scripts/reset-db.sh --yes    performs the reset
#
# The reset itself runs inside the running `api` container, so it targets
# exactly the database that deployment is configured against. scripts/
# reset_app_data.py is copied in at run time instead of being baked into the
# image: the script that runs is then the one in this checkout, and no rebuild
# stands between editing a guard and that guard taking effect.
#
# Every safety check lives in reset_app_data.py — the development-environment
# check, the Supabase project reference match, the table classification and the
# owner resolution. This script adds none of its own and weakens none of them.

set -euo pipefail

OWNER_EMAIL="nagarajay@gmail.com"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESET_SCRIPT="${REPO_ROOT}/scripts/reset_app_data.py"
CONTAINER_PATH="/tmp/reset_app_data.py"
SERVICE="api"

die() {
    echo "reset-db: $*" >&2
    exit 1
}

apply=false
for arg in "$@"; do
    case "$arg" in
        --yes)
            apply=true
            ;;
        -h | --help)
            sed -n '3,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            die "unknown argument: ${arg}. Usage: ./scripts/reset-db.sh [--yes]"
            ;;
    esac
done

[[ -f "$RESET_SCRIPT" ]] || die "${RESET_SCRIPT} not found."

# The guards are configuration, so they are read from the repository's .env
# rather than from whatever happens to be exported in this shell, and passed
# into the container explicitly. Passing them means the reset works against a
# container created before these variables were added to docker-compose.yaml,
# without a recreate; reading only these two keys means nothing else in .env is
# evaluated by this shell.
env_value() {
    local key="$1" line
    [[ -f "${REPO_ROOT}/.env" ]] || return 0
    line="$(grep -E "^[[:space:]]*${key}=" "${REPO_ROOT}/.env" | tail -n 1)" || return 0
    line="${line#*=}"
    # Strip one layer of surrounding quotes, as Compose's .env parser does.
    line="${line%\"}"
    line="${line#\"}"
    line="${line%\'}"
    line="${line#\'}"
    printf '%s' "$line"
}

DOGRAH_ENV="${DOGRAH_ENV:-$(env_value DOGRAH_ENV)}"
DOGRAH_SUPABASE_PROJECT_REF="${DOGRAH_SUPABASE_PROJECT_REF:-$(env_value DOGRAH_SUPABASE_PROJECT_REF)}"

[[ -n "$DOGRAH_ENV" ]] ||
    die "DOGRAH_ENV is not set in the environment or in ${REPO_ROOT}/.env. The reset refuses to run without it."
[[ -n "$DOGRAH_SUPABASE_PROJECT_REF" ]] ||
    die "DOGRAH_SUPABASE_PROJECT_REF is not set in the environment or in ${REPO_ROOT}/.env. The reset refuses to run without it."

command -v docker >/dev/null 2>&1 ||
    die "docker is not installed or not on PATH. Start Docker and try again."

docker info >/dev/null 2>&1 ||
    die "cannot talk to the Docker daemon. Is Docker Desktop running?"

docker compose version >/dev/null 2>&1 ||
    die "'docker compose' is unavailable. This needs Docker Compose v2."

cd "$REPO_ROOT"

# `ps --status running` rather than `ps -q`: a created-but-exited api container
# still has an id, and `exec` against it fails with a message that says nothing
# about why.
running="$(docker compose ps --status running --services 2>/dev/null || true)"
if ! grep -qx "$SERVICE" <<<"$running"; then
    die "the '${SERVICE}' service is not running. Start it with 'docker compose up -d ${SERVICE}' and try again."
fi

# Copied in on every run, and removed afterwards whichever way this exits.
docker compose cp "$RESET_SCRIPT" "${SERVICE}:${CONTAINER_PATH}" >/dev/null ||
    die "could not copy ${RESET_SCRIPT} into the ${SERVICE} container."

# `docker compose cp` writes as root, so the removal has to be root too — the
# container's own user cannot unlink a root-owned file in /tmp. A failure here
# is not worth aborting over: what is left behind is a copy of this repository's
# script, which refuses to do anything on its own.
cleanup() {
    docker compose exec -T -u root "$SERVICE" rm -f "$CONTAINER_PATH" >/dev/null 2>&1 || true
}
trap cleanup EXIT

args=(--owner-email "$OWNER_EMAIL")
if [[ "$apply" == true ]]; then
    args+=(--yes)
    echo "reset-db: about to reset the development database to zero organizations, preserving super-admin ${OWNER_EMAIL}."
else
    echo "reset-db: dry run — nothing will be deleted. Re-run with --yes to apply."
fi
echo

# -T: nothing here reads from stdin. Which database may be reset was settled by
# DOGRAH_ENV and DOGRAH_SUPABASE_PROJECT_REF before the connection was opened,
# and --yes is the only confirmation, so the run works unchanged from a script
# or a CI shell with no terminal attached.
docker compose exec -T \
    -e "DOGRAH_ENV=${DOGRAH_ENV}" \
    -e "DOGRAH_SUPABASE_PROJECT_REF=${DOGRAH_SUPABASE_PROJECT_REF}" \
    "$SERVICE" python "$CONTAINER_PATH" "${args[@]}"
