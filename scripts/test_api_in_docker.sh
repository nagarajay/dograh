#!/usr/bin/env bash
# Run the API test suite in a disposable Linux container with its own Postgres
# (pgvector) and Redis, without a host virtualenv. Useful on machines where the
# pipecat extras cannot be installed (for example Intel macOS).
#
#   scripts/test_api_in_docker.sh                         # whole suite
#   scripts/test_api_in_docker.sh tests/test_ts_bridge.py # chosen tests (paths relative to api/)
#
# The staged copy keeps the repository layout (api/, docs/, scripts/, sdk/ and
# docker-compose.yaml side by side), because some tests read repo-level files such
# as docker-compose.yaml relative to api/tests. The real working tree is never
# modified or mounted writable. GEMINI_BULK_INTEGRATION=1 is set so the real
# Postgres/Redis bulk-retry tests run (in a full run they use their own
# gemini-bulk-postgres / gemini-bulk-redis containers) instead of skipping.
#
# Requires docker, rsync, and node/npm on the host (the MCP TypeScript validator
# tests need `npm ci` of api/mcp_server/ts_validator). IMAGE must already contain
# the API's Python dependencies (default: the locally built API image).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-avsiq-local/dograh-api:latest}"
RUN_ID="apitest-$$"
STAGE="$(mktemp -d)"
NET="$RUN_ID-net"

cleanup() {
  docker rm -f "$RUN_ID-pg" "$RUN_ID-redis" >/dev/null 2>&1 || true
  docker network rm "$NET" >/dev/null 2>&1 || true
  rm -rf "$STAGE"
}
trap cleanup EXIT

# Stage the repository layout (read-only inputs only).
mkdir -p "$STAGE/repo"
rsync -a --exclude node_modules --exclude __pycache__ "$ROOT/api" "$ROOT/docs" "$ROOT/scripts" "$ROOT/sdk" "$STAGE/repo/"
cp "$ROOT/docker-compose.yaml" "$STAGE/repo/"
if [ ! -d "$STAGE/repo/api/mcp_server/ts_validator/node_modules" ]; then
  (cd "$STAGE/repo/api/mcp_server/ts_validator" && npm ci --silent)
fi

# Test credentials come from api/.env.test (quotes stripped for docker --env-file).
sed 's/"//g' "$ROOT/api/.env.test" > "$STAGE/env"
PGUSER_="$(python3 -c "import re,sys;print(re.search(r'//([^:]+):', open(sys.argv[1]).read().split('DATABASE_URL=')[1]).group(1))" "$STAGE/env")"
PGPASS_="$(python3 -c "import re,sys;print(re.search(r'//[^:]+:([^@]+)@', open(sys.argv[1]).read().split('DATABASE_URL=')[1]).group(1))" "$STAGE/env")"
REDISPASS_="$(python3 -c "import re,sys;m=re.search(r'REDIS_URL=redis://:?([^@]*)@', open(sys.argv[1]).read());print(m.group(1) if m else '')" "$STAGE/env")"

docker network create "$NET" >/dev/null
# Aliases match the host names in api/.env.test ("postgres", "redis").
docker run -d --name "$RUN_ID-pg" --network "$NET" --network-alias postgres \
  -e POSTGRES_USER="$PGUSER_" -e POSTGRES_PASSWORD="$PGPASS_" -e POSTGRES_DB=test_db \
  pgvector/pgvector:pg17 >/dev/null
docker run -d --name "$RUN_ID-redis" --network "$NET" --network-alias redis redis:7 \
  redis-server ${REDISPASS_:+--requirepass "$REDISPASS_"} >/dev/null
for _ in $(seq 1 30); do
  docker exec "$RUN_ID-pg" pg_isready -q && break || sleep 1
done

# Dedicated database and queue for the bulk-retry integration tests (they
# truncate tables and flush Redis, so they refuse to touch anything else).
docker run -d --name "$RUN_ID-bulk-pg" --network "$NET" --network-alias gemini-bulk-postgres \
  -e POSTGRES_USER=t -e POSTGRES_PASSWORD=t -e POSTGRES_DB=t pgvector/pgvector:pg17 >/dev/null
docker run -d --name "$RUN_ID-bulk-redis" --network "$NET" --network-alias gemini-bulk-redis redis:7 >/dev/null
trap 'docker rm -f "$RUN_ID-bulk-pg" "$RUN_ID-bulk-redis" >/dev/null 2>&1 || true; cleanup' EXIT
for _ in $(seq 1 30); do
  docker exec "$RUN_ID-bulk-pg" pg_isready -q -U t && break || sleep 1
done

ARGS="${*:-tests}"
docker run --rm --network "$NET" --entrypoint sh \
  -v "$STAGE/repo:/repo" -v "$ROOT/pipecat:/pipecat:ro" \
  --env-file "$STAGE/env" \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/pipecat/src:/repo:/tmp/pt \
  -e GEMINI_BULK_INTEGRATION=1 \
  -w /repo/api "$IMAGE" -c "
    python -m pip install -q --target /tmp/pt pytest pytest-asyncio pytest-mock >/dev/null 2>&1
    python -m pytest -p no:cacheprovider -q -rs $ARGS"
