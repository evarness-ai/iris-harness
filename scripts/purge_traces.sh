#!/usr/bin/env bash
# Purge IRIS *trace / observability* artifacts so fresh test runs produce clean
# call traces. Deletes ONLY: session JSONL logs, the governance + governor audit
# DBs, and run logs. It NEVER touches IRIS state (email / calendar / finance /
# memory / knowledge DBs under data/).
#
# Everything is BACKED UP first to a timestamped folder under the archive dir
# (default ~/.iris/archive; override with IRIS_TRACE_ARCHIVE_DIR in the env or .env).
#
# Usage:
#   scripts/purge_traces.sh            # show plan, prompt before deleting
#   scripts/purge_traces.sh --dry-run  # show plan only, delete nothing
#   scripts/purge_traces.sh --yes      # delete without prompting
#   scripts/purge_traces.sh --no-backup  # skip the backup step
#
# Stop IRIS services first (API / governor / gateway) so audit DBs aren't open.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IRIS_LOGS="${HOME}/.iris/logs"
GOV_AUDIT_DB="${IRIS_GOVERNANCE_AUDIT_DB_PATH:-${HOME}/.local/share/iris/audit.db}"
GOVERNOR_AUDIT_DB="${REPO_ROOT}/data/audit.db"

DRY_RUN=0
ASSUME_YES=0
DO_BACKUP=1
for arg in "$@"; do
  case "$arg" in
    -n|--dry-run) DRY_RUN=1 ;;
    -y|--yes) ASSUME_YES=1 ;;
    --no-backup) DO_BACKUP=0 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# --- archive dir: env var > .env > default ~/.iris/archive ---------------------
read_env() { # echo value of KEY=... from repo .env (last wins), quotes stripped
  local key="$1" line
  [ -f "${REPO_ROOT}/.env" ] || return 0
  line="$(grep -E "^[[:space:]]*${key}=" "${REPO_ROOT}/.env" | tail -1 || true)"
  [ -n "$line" ] || return 0
  line="${line#*=}"
  line="${line%\"}"; line="${line#\"}"
  line="${line%\'}"; line="${line#\'}"
  printf '%s' "$line"
}
ARCHIVE_DIR="${IRIS_TRACE_ARCHIVE_DIR:-$(read_env IRIS_TRACE_ARCHIVE_DIR)}"
[ -n "$ARCHIVE_DIR" ] || ARCHIVE_DIR="${HOME}/.iris/archive"
ARCHIVE_DIR="${ARCHIVE_DIR/#\~/$HOME}"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_DEST="${ARCHIVE_DIR}/traces-${STAMP}"

# --- collect targets, grouped by source (groups -> backup subdirs) ------------
# Bash 3.2 (macOS default) has no `mapfile`; read newline-delimited paths into
# arrays with a portable loop. Splitting on newline (not spaces) keeps paths that
# contain spaces — e.g. the repo lives under ".../My Workspace/..." — intact.
shopt -s nullglob
collect() { local p; for p in "$@"; do [ -e "$p" ] && printf '%s\n' "$p"; done; return 0; }

SESSIONS=(); while IFS= read -r l; do SESSIONS+=("$l"); done < <(collect "${IRIS_LOGS}"/session-*.jsonl)
OTHERLOGS=(); while IFS= read -r l; do OTHERLOGS+=("$l"); done < <(collect \
  "${IRIS_LOGS}/code_exec.jsonl" "${IRIS_LOGS}/evaluator.log" "${IRIS_LOGS}/channel_gateway.log" \
  "${IRIS_LOGS}/governor.log" "${IRIS_LOGS}/iris_api.log")
GOVAUDIT=(); while IFS= read -r l; do GOVAUDIT+=("$l"); done < <(collect \
  "${GOV_AUDIT_DB}" "${GOV_AUDIT_DB}-wal" "${GOV_AUDIT_DB}-shm")
GOVERNOR=(); while IFS= read -r l; do GOVERNOR+=("$l"); done < <(collect \
  "${GOVERNOR_AUDIT_DB}" "${GOVERNOR_AUDIT_DB}-wal" "${GOVERNOR_AUDIT_DB}-shm")
RUNLOGS=(); while IFS= read -r l; do RUNLOGS+=("$l"); done < <(collect \
  "${REPO_ROOT}/iris.log" "${REPO_ROOT}/iris_api.log" "${REPO_ROOT}/iris_api.run.log" \
  "${REPO_ROOT}/governor.run.log" "${REPO_ROOT}/channel_gateway.log")
shopt -u nullglob

TOTAL=$(( ${#SESSIONS[@]} + ${#OTHERLOGS[@]} + ${#GOVAUDIT[@]} + ${#GOVERNOR[@]} + ${#RUNLOGS[@]} ))
if [ "${TOTAL}" -eq 0 ]; then
  echo "Nothing to purge — traces already clean."
  exit 0
fi

# --- show the plan ------------------------------------------------------------
echo "IRIS trace/observability purge — will DELETE:"
if [ ${#SESSIONS[@]} -gt 0 ]; then
  printf '  %d session-*.jsonl trace files in %s\n' "${#SESSIONS[@]}" "${IRIS_LOGS}"
fi
for p in ${OTHERLOGS[@]+"${OTHERLOGS[@]}"} ${GOVAUDIT[@]+"${GOVAUDIT[@]}"} \
         ${GOVERNOR[@]+"${GOVERNOR[@]}"} ${RUNLOGS[@]+"${RUNLOGS[@]}"}; do
  printf '  %s\n' "$p"
done
echo
if [ "${DO_BACKUP}" -eq 1 ]; then
  echo "Backup -> ${BACKUP_DEST}"
else
  echo "Backup -> DISABLED (--no-backup)"
fi
echo "PRESERVED (state, never touched): data/{email,calendar,finance,memory,iris,learning,"
echo "  routines,tasks,missions,coding_tasks,filemanager_audit}.db, data/chroma, data/wiki, ~/.iris config"

if [ "${DRY_RUN}" -eq 1 ]; then
  echo
  echo "(dry-run) nothing backed up or deleted."
  exit 0
fi

if [ "${ASSUME_YES}" -ne 1 ]; then
  printf '\nProceed? [y/N] '
  read -r ans
  case "${ans}" in y|Y|yes|YES) ;; *) echo "aborted."; exit 1 ;; esac
fi

# --- backup (copy preserving source layout; abort the whole run if it fails) --
backup_group() { # <subdir> <files...>
  local subdir="$1"; shift
  [ "$#" -gt 0 ] || return 0
  mkdir -p "${BACKUP_DEST}/${subdir}"
  cp -p "$@" "${BACKUP_DEST}/${subdir}/"
}
if [ "${DO_BACKUP}" -eq 1 ]; then
  backup_group "iris-logs"  ${SESSIONS[@]+"${SESSIONS[@]}"} ${OTHERLOGS[@]+"${OTHERLOGS[@]}"}
  backup_group "share-iris" ${GOVAUDIT[@]+"${GOVAUDIT[@]}"}
  backup_group "repo-data"  ${GOVERNOR[@]+"${GOVERNOR[@]}"}
  backup_group "repo-root"  ${RUNLOGS[@]+"${RUNLOGS[@]}"}
  echo "Backed up ${TOTAL} item(s) to ${BACKUP_DEST}"
fi

# --- delete -------------------------------------------------------------------
if [ ${#SESSIONS[@]} -gt 0 ]; then rm -f "${SESSIONS[@]}"; fi
for p in ${OTHERLOGS[@]+"${OTHERLOGS[@]}"} ${GOVAUDIT[@]+"${GOVAUDIT[@]}"} \
         ${GOVERNOR[@]+"${GOVERNOR[@]}"} ${RUNLOGS[@]+"${RUNLOGS[@]}"}; do
  rm -f "$p"
done

echo "Purged ${TOTAL} item(s). Next IRIS run starts with clean traces."
