#!/usr/bin/env bash
#
# Promote an existing Dograh user to platform super-admin, against the running
# Docker Compose deployment's database.
#
# This is the single supported dev/recovery path for this operation. It is a
# thin wrapper around scripts/bootstrap_superadmin.py -- every safety check
# (existing user only, exact single match, no password touched, no
# organization created) lives there. This script adds none of its own and
# weakens none of them; it only runs that script inside the `api` container so
# it targets exactly the database that deployment is configured against, using
# DATABASE_URL already set on the container.
#
#   ./scripts/bootstrap-superadmin.sh owner@example.com               dry run
#   ./scripts/bootstrap-superadmin.sh owner@example.com --yes          apply
#   ./scripts/bootstrap-superadmin.sh owner@example.com --detach-organization --yes
#
# The script itself is copied into the container at run time instead of
# relying on whatever was baked into the image at build time, so the version
# that runs is the one in this checkout -- same reasoning as reset-db.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_SCRIPT="${REPO_ROOT}/scripts/bootstrap_superadmin.py"
CONTAINER_PATH="/tmp/bootstrap_superadmin.py"
SERVICE="api"

die() {
    echo "bootstrap-superadmin: $*" >&2
    exit 1
}

[[ $# -ge 1 ]] || die "usage: ./scripts/bootstrap-superadmin.sh <email> [--detach-organization] [--yes]"

email="$1"
shift
case "$email" in
    --*) die "first argument must be the account's email, not a flag. Usage: ./scripts/bootstrap-superadmin.sh <email> [--detach-organization] [--yes]" ;;
esac

extra_args=()
for arg in "$@"; do
    case "$arg" in
        --detach-organization | --yes)
            extra_args+=("$arg")
            ;;
        -h | --help)
            sed -n '3,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            die "unknown argument: ${arg}. Usage: ./scripts/bootstrap-superadmin.sh <email> [--detach-organization] [--yes]"
            ;;
    esac
done

[[ -f "$SOURCE_SCRIPT" ]] || die "${SOURCE_SCRIPT} not found."

command -v docker >/dev/null 2>&1 ||
    die "docker is not installed or not on PATH. Start Docker and try again."

docker info >/dev/null 2>&1 ||
    die "cannot talk to the Docker daemon. Is Docker Desktop running?"

docker compose version >/dev/null 2>&1 ||
    die "'docker compose' is unavailable. This needs Docker Compose v2."

cd "$REPO_ROOT"

running="$(docker compose ps --status running --services 2>/dev/null || true)"
if ! grep -qx "$SERVICE" <<<"$running"; then
    die "the '${SERVICE}' service is not running. Start it with 'docker compose up -d ${SERVICE}' and try again."
fi

docker compose cp "$SOURCE_SCRIPT" "${SERVICE}:${CONTAINER_PATH}" >/dev/null ||
    die "could not copy ${SOURCE_SCRIPT} into the ${SERVICE} container."

cleanup() {
    docker compose exec -T -u root "$SERVICE" rm -f "$CONTAINER_PATH" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# -T: nothing here reads from stdin. DATABASE_URL is already set on the
# container by docker-compose.yaml, so it is not passed explicitly here.
docker compose exec -T "$SERVICE" python "$CONTAINER_PATH" --email "$email" "${extra_args[@]+"${extra_args[@]}"}"
