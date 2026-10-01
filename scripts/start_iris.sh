#!/usr/bin/env bash
# Start the full IRIS stack (Governor + Evaluator + IRIS API + Channel Gateway +
# Phoenix observability + Web UI) with .env loaded, in dependency order, then
# drop into the chat REPL. Single entry point for bringing IRIS up locally.
#
# Phoenix runs as its OWN process (`phoenix serve`, managed like every other
# service) and iris_api exports spans to it over OTLP — so the Phoenix server
# and its growing trace store stay OUT of the iris_api process (the single
# biggest driver of the harness's RAM footprint). Its trace UI is served at
# http://127.0.0.1:6006 and traces persist under $IRIS_PHOENIX_WORKING_DIR.
# Enabling it also turns on /observability/llm-metrics so the Web UI Overview
# is populated.
#
# Phoenix is the optional `phoenix` extra (`poetry install -E phoenix`; it is
# Elastic-2.0, so the core install leaves it out). Without it the script does not
# launch it: spans still export over OTLP when IRIS_PHOENIX_ENDPOINT names a
# backend of your own, and tracing is off otherwise.
#
# Usage:
#   scripts/start_iris.sh                 # all services + Phoenix + Web UI + REPL
#   scripts/start_iris.sh --services-only # services + Phoenix + Web UI, no REPL
#   scripts/start_iris.sh --no-ui         # skip the Web UI (combine with the above)
#   scripts/start_iris.sh --no-phoenix    # skip Phoenix + metrics
#   scripts/start_iris.sh --enable-writes # set IRIS_WEBUI_ALLOW_WRITES=1 for this run
#   scripts/start_iris.sh --env K=V        # override/add env var(s) for this run
#   scripts/start_iris.sh --status        # list running services (incl. Phoenix)
#   scripts/start_iris.sh --stop          # stop all services
#   scripts/start_iris.sh --restart=NAME  # stop + start ONE service (e.g. governor);
#                                         # the health watch's repair (ADR-0116)
#
# Per-service port overrides (defaults shown):
#   IRIS_GOVERNOR_PORT=8080
#   IRIS_EVALUATOR_PORT=8090
#   IRIS_API_PORT=8003
#   IRIS_CHANNEL_GATEWAY_PORT=8006
#   IRIS_PHOENIX_PORT=6006
#   IRIS_WEBUI_PORT=5181
#
# Other overrides:
#   IRIS_API_HOST=127.0.0.1           # binds all services to this host
#   IRIS_WEBUI_ENABLED=0              # same as --no-ui
#   IRIS_PHOENIX_ENABLED=0           # same as --no-phoenix
#   IRIS_WEBUI_ALLOW_WRITES=1         # same as --enable-writes
#   IRIS_LOG_DIR=~/.iris/logs
#   IRIS_ENV_FILE=.env                # path to env file (relative to repo root)
#   IRIS_API_FORCE_RESTART=1          # force restart even when PID is healthy

set -euo pipefail

SCRIPT_START_TS="$(date +%s)"

print_elapsed() {
  local now elapsed
  now="$(date +%s)"
  elapsed=$((now - SCRIPT_START_TS))
  printf 'elapsed: %02dm %02ds\n' "$((elapsed / 60))" "$((elapsed % 60))"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Load .env so every service inherits secrets + config ────────────
ENV_FILE="${IRIS_ENV_FILE:-.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  echo "loaded env from $ENV_FILE"
else
  echo "warn: $ENV_FILE not found — services will rely on shell env only" >&2
fi

require_auth_secret() {
  if [[ -z "${IRIS_AUTH_SECRET:-}" || -z "${IRIS_AUTH_SECRET//[[:space:]]/}" ]]; then
    cat >&2 <<EOF
error: IRIS_AUTH_SECRET is not set.

The IRIS HTTP services fail closed and will return HTTP 503 for every
non-health request when the shared auth secret is missing.

Set IRIS_AUTH_SECRET in $ENV_FILE or export it in your shell before running
scripts/start_iris.sh.
EOF
    exit 1
  fi
}

HOST="${IRIS_API_HOST:-127.0.0.1}"
LOG_DIR="${IRIS_LOG_DIR:-$HOME/.iris/logs}"
mkdir -p "$LOG_DIR"

# ── Argument parsing (mode + flags, order-independent) ───────────────
MODE="default"            # default | services | status | stop | restart
WEBUI_ENABLED="${IRIS_WEBUI_ENABLED:-1}"
PHOENIX_ENABLED="${IRIS_PHOENIX_ENABLED:-1}"   # own `phoenix serve` process by default
PHOENIX_PORT="${IRIS_PHOENIX_PORT:-6006}"
RUNTIME_ENV_OVERRIDES=()
RESTART_NAME=""
for arg in "$@"; do
  case "$arg" in
    --services-only|--api-only) MODE="services" ;;
    --status) MODE="status" ;;
    --stop)   MODE="stop" ;;
    --restart=*)
      MODE="restart"
      RESTART_NAME="${arg#--restart=}"
      [[ "$RESTART_NAME" =~ ^[a-z_]+$ ]] || { echo "invalid service name in $arg" >&2; exit 2; }
      ;;
    --no-ui)  WEBUI_ENABLED=0 ;;
    --no-phoenix) PHOENIX_ENABLED=0 ;;
    --enable-writes) RUNTIME_ENV_OVERRIDES+=("IRIS_WEBUI_ALLOW_WRITES=1") ;;
    --disable-writes) RUNTIME_ENV_OVERRIDES+=("IRIS_WEBUI_ALLOW_WRITES=0") ;;
    --env=*)
      kv="${arg#--env=}"
      [[ "$kv" == *=* ]] || { echo "invalid --env value: $arg (expected --env=KEY=VALUE)" >&2; exit 2; }
      key="${kv%%=*}"
      [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || { echo "invalid env key in $arg" >&2; exit 2; }
      RUNTIME_ENV_OVERRIDES+=("$kv")
      ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# Apply runtime env overrides after .env/default parsing so CLI flags win
# without requiring edits to checked-in/local env files.
if [[ "${#RUNTIME_ENV_OVERRIDES[@]}" -gt 0 ]]; then
  for kv in "${RUNTIME_ENV_OVERRIDES[@]}"; do
    export "$kv"
    echo "runtime override: $kv"
  done
fi

# ── Phoenix is the optional `phoenix` extra: launch it only when it is installed.
# PHOENIX_ENABLED = spans are exported; PHOENIX_LAUNCH = this script runs the Phoenix
# server they go to. Without the extra, an IRIS_PHOENIX_ENDPOINT the operator set
# (any OTLP backend) keeps tracing on; with neither, tracing is off. Checked only
# when starting something: --stop/--status must still see a Phoenix left running.
PHOENIX_LAUNCH="$PHOENIX_ENABLED"
PHOENIX_EXTRA_HINT="install iris-harness[phoenix] (poetry install -E phoenix)"
if [[ "$PHOENIX_ENABLED" == "1" && "$MODE" != "stop" && "$MODE" != "status" ]]; then
  if ! poetry run python -c 'import importlib.metadata as m; m.version("arize-phoenix")' \
      >/dev/null 2>&1; then
    PHOENIX_LAUNCH=0
    if [[ -n "${IRIS_PHOENIX_ENDPOINT:-}" ]]; then
      echo "Phoenix is not installed, so it is not launched; spans export over OTLP to" \
        "$IRIS_PHOENIX_ENDPOINT. For the local trace UI, $PHOENIX_EXTRA_HINT."
    else
      PHOENIX_ENABLED=0
      echo "Phoenix is not installed: tracing is off. For the local trace UI," \
        "$PHOENIX_EXTRA_HINT; or set IRIS_PHOENIX_ENDPOINT to any OTLP backend."
    fi
  fi
fi

# ── Observability: Phoenix runs as its own `phoenix serve` process (added to
# the service registry below) and iris_api exports spans to it over OTLP in
# EXTERNAL mode — keeping the Phoenix server out of iris_api. Turning it on also
# enables the /observability/llm-metrics endpoint the Web UI reads.
PHOENIX_WORKING_DIR="${IRIS_PHOENIX_WORKING_DIR:-$HOME/.iris/phoenix}"
if [[ "$PHOENIX_ENABLED" == "1" ]]; then
  export IRIS_PHOENIX_ENABLED=1
  export IRIS_PHOENIX_MODE=external
  export IRIS_PHOENIX_PORT="$PHOENIX_PORT"
  export IRIS_PHOENIX_ENDPOINT="${IRIS_PHOENIX_ENDPOINT:-http://$HOST:$PHOENIX_PORT}"
  export IRIS_OBSERVABILITY_METRICS_ENABLED="${IRIS_OBSERVABILITY_METRICS_ENABLED:-1}"
  if [[ "$PHOENIX_LAUNCH" == "1" ]]; then
    export IRIS_PHOENIX_URL="${IRIS_PHOENIX_URL:-http://$HOST:$PHOENIX_PORT}"
    mkdir -p "$PHOENIX_WORKING_DIR"
  fi
else
  export IRIS_PHOENIX_ENABLED=0
fi

# ── Service registry: NAME|PORT|COMMAND, in startup order (earlier =
# dependency of later). The command runs under `bash -c` with $HOST and
# $PORT exported, so it stays declarative. The Web UI (Vite) is a
# first-class service started last — it proxies /api to iris_api.
SERVICES=()
# Phoenix first: iris_api exports spans to it at startup, so it must be up
# before iris_api. `phoenix serve` reads PHOENIX_HOST/PORT/WORKING_DIR from env
# (no host/port flags), so we pass them explicitly rather than via $HOST/$PORT.
if [[ "$PHOENIX_LAUNCH" == "1" ]]; then
  SERVICES+=("phoenix|$PHOENIX_PORT|PHOENIX_HOST=\"\$HOST\" PHOENIX_PORT=\"\$PORT\" PHOENIX_WORKING_DIR=\"$PHOENIX_WORKING_DIR\" exec poetry run phoenix serve")
fi
SERVICES+=(
  "governor|${IRIS_GOVERNOR_PORT:-8080}|poetry run python -m uvicorn iris_harness.server.governor.main:app --host \"\$HOST\" --port \"\$PORT\""
  "evaluator|${IRIS_EVALUATOR_PORT:-8090}|poetry run python -m uvicorn iris_harness.server.evaluator.main:app --host \"\$HOST\" --port \"\$PORT\""
  "iris_api|${IRIS_API_PORT:-8003}|poetry run python -m uvicorn iris_harness.server.iris_api.main:app --host \"\$HOST\" --port \"\$PORT\""
  "channel_gateway|${IRIS_CHANNEL_GATEWAY_PORT:-8006}|poetry run python -m uvicorn iris_harness.server.channel_gateway.main:app --host \"\$HOST\" --port \"\$PORT\""
)
if [[ "$WEBUI_ENABLED" == "1" ]]; then
  if command -v npm >/dev/null 2>&1; then
    SERVICES+=("webui|${IRIS_WEBUI_PORT:-5181}|cd \"$REPO_ROOT/webui\" && exec ./node_modules/.bin/vite --host \"\$HOST\" --port \"\$PORT\" --strictPort")
  else
    echo "warn: npm not found — skipping Web UI (install Node.js to enable it)" >&2
  fi
fi

pid_file_for()  { echo "$LOG_DIR/$1.pid"; }
log_file_for()  { echo "$LOG_DIR/$1.log"; }

service_running() {
  local pf; pf="$(pid_file_for "$1")"
  [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null
}

service_stale() {
  [[ "${IRIS_API_FORCE_RESTART:-}" == "1" ]] && return 0
  # The Web UI is JS (Vite has its own HMR); never restart it on Python edits.
  [[ "$1" == "webui" ]] && return 1
  local pf; pf="$(pid_file_for "$1")"
  [[ ! -f "$pf" ]] && return 0
  find src services config pyproject.toml -type f \
    \( -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o -name '*.toml' \) \
    -newer "$pf" -print -quit 2>/dev/null | grep -q .
}

ensure_webui_deps() {
  local dir="$REPO_ROOT/webui"
  if [[ ! -x "$dir/node_modules/.bin/vite" ]]; then
    echo "webui: installing npm deps (first run)…"
    ( cd "$dir" && npm install ) || { echo "webui: npm install failed" >&2; return 1; }
  fi
}

# Per-service readiness budget (in 0.5s ticks). iris_api is heavy — runtime
# build + eager memory embed + model warmup legitimately take ~60-80s cold; a
# short wait would wrongly flag it failed and skip everything after it (the Web
# UI). Phoenix serve also has a slow cold import (its web app + DB migrations).
tries_for() {
  case "$1" in
    iris_api) echo 160 ;;  # ~80s
    phoenix)  echo 80 ;;   # ~40s — cold import + SQLite migrations
    webui)    echo 40 ;;   # vite is fast
    *)        echo 60 ;;   # ~30s
  esac
}

# Non-fatal: a service that isn't ready in time is NOT treated as a hard failure
# — its process keeps starting in the background while we move on to the rest.
wait_for_port() {
  local name="$1" port="$2" log_file="$3" tries="${4:-60}"
  echo -n "  waiting for $name on http://$HOST:$port "
  for ((i = 0; i < tries; i++)); do
    if curl -sS -o /dev/null --max-time 1 "http://$HOST:$port/" 2>/dev/null; then
      echo "[ok]"
      return 0
    fi
    echo -n "."
    sleep 0.5
  done
  echo "[still starting]"
  echo "  $name not ready after ~$((tries / 2))s — continuing; it may finish in the"
  echo "  background (check: $0 --status). last log lines:"
  tail -n 8 "$log_file" 2>/dev/null | sed 's/^/    /' || true
  return 1
}

stop_service() {
  local name="$1" pf; pf="$(pid_file_for "$name")"
  if service_running "$name"; then
    local pid; pid="$(cat "$pf")"
    echo "stopping $name (pid $pid)..."
    kill "$pid" 2>/dev/null || true
    for _ in {1..20}; do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pf"
}

stop_all() {
  # Reverse order — tear down dependents (Web UI) first.
  for ((i=${#SERVICES[@]}-1; i>=0; i--)); do
    stop_service "${SERVICES[i]%%|*}"
  done
}

start_one() {
  local entry="$1" name port cmd
  IFS='|' read -r name port cmd <<<"$entry"
  local log_file; log_file="$(log_file_for "$name")"

  if service_running "$name"; then
    if service_stale "$name"; then
      echo "$name: source changed since pid $(cat "$(pid_file_for "$name")") started — restarting."
      stop_service "$name"
    else
      echo "$name: already running (pid $(cat "$(pid_file_for "$name")")) — reusing."
      return 0
    fi
  fi
  [[ "$name" == "webui" ]] && { ensure_webui_deps || return 1; }
  echo "starting $name on $HOST:$port (logs: $log_file)"
  HOST="$HOST" PORT="$port" nohup bash -c "$cmd" >>"$log_file" 2>&1 &
  echo $! >"$(pid_file_for "$name")"
  wait_for_port "$name" "$port" "$log_file" "$(tries_for "$name")" || true
}

start_all() {
  # Never abort the whole stack because one service is slow or unhealthy —
  # otherwise a slow iris_api would skip the Web UI and its URL would never
  # print. Each service is launched best-effort; --status shows what's up.
  for entry in "${SERVICES[@]}"; do start_one "$entry" || true; done
}

status() {
  printf "%-18s %-7s %-7s %s\n" "service" "port" "status" "pid"
  for entry in "${SERVICES[@]}"; do
    local name port; IFS='|' read -r name port _ <<<"$entry"
    if service_running "$name"; then
      printf "%-18s %-7s %-7s %s\n" "$name" "$port" "running" "$(cat "$(pid_file_for "$name")")"
    else
      printf "%-18s %-7s %-7s %s\n" "$name" "$port" "stopped" "-"
    fi
  done
  # Phoenix is a managed service now (its own pid), so the loop above already
  # lists it when enabled — no special-case probe needed.
}

case "$MODE" in
  stop)
    stop_all
    print_elapsed
    exit 0
    ;;
  status)
    status
    print_elapsed
    exit 0
    ;;
  restart)
    require_auth_secret
    for entry in "${SERVICES[@]}"; do
      if [[ "${entry%%|*}" == "$RESTART_NAME" ]]; then
        stop_service "$RESTART_NAME"
        start_one "$entry" || true
        print_elapsed
        exit 0
      fi
    done
    echo "unknown service: $RESTART_NAME (see $0 --status)" >&2
    exit 2
    ;;
  services)
    require_auth_secret
    start_all
    print_elapsed
    echo "all services up. stop with: $0 --stop"
    [[ "$PHOENIX_LAUNCH" == "1" ]] && echo "Phoenix  → http://$HOST:$PHOENIX_PORT  (traces, own process)"
    [[ "$WEBUI_ENABLED" == "1" ]] && echo "Web UI   → http://$HOST:${IRIS_WEBUI_PORT:-5181}"
    exit 0
    ;;
esac

# Default: services + REPL.
require_auth_secret
start_all
print_elapsed

# Surface common service URLs so the REPL inherits them.
export IRIS_API_URL="${IRIS_API_URL:-http://$HOST:${IRIS_API_PORT:-8003}}"
export IRIS_GOVERNOR_URL="${IRIS_GOVERNOR_URL:-http://$HOST:${IRIS_GOVERNOR_PORT:-8080}}"
export IRIS_EVALUATOR_URL="${IRIS_EVALUATOR_URL:-http://$HOST:${IRIS_EVALUATOR_PORT:-8090}}"
export IRIS_CHANNEL_GATEWAY_URL="${IRIS_CHANNEL_GATEWAY_URL:-http://$HOST:${IRIS_CHANNEL_GATEWAY_PORT:-8006}}"

[[ "$PHOENIX_LAUNCH" == "1" ]] && echo "Phoenix → http://$HOST:$PHOENIX_PORT  (trace UI, own process)"
[[ "$WEBUI_ENABLED" == "1" ]] && echo "Web UI → http://$HOST:${IRIS_WEBUI_PORT:-5181}  (Chat, Call Trace, Knowledge, …)"
echo "→ launching iris chat (services stay running in background; stop with: $0 --stop)"
poetry run python -m iris_harness.main
