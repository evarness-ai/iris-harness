#!/usr/bin/env bash
# The `quickstart` check (OSS plan R11, R6, R19): the README's hero path on a fresh
# machine, as a new user runs it, against the scripted fake model -- no model server, no
# credentials, no network after the install.
#
#   1. `uv tool install "<tree>[email]"` into a throwaway tool dir with a COLD package
#      cache, timed: R19's target is under 60 s. Over the budget fails the check (the owner's
#      call: the target is a promise to users), and the number lands in the job summary.
#   2. `iris doctor --json`: it must run from the install and give a verdict that is not
#      "not ready" (the model server is a closed port and there is no OS keyring, so a
#      CI runner is "demo only"; the vault key is a throwaway, as a user's first
#      `iris doctor --fix` would create).
#   3. `iris email demo` in its own home: exit 0, the "What IRIS just did" summary, no
#      network connection attempted, and the labels step done.
#   4. The getting-started pages of the docs site (R7), block by block, in the same shell
#      (scripts/getting_started.py): every block a page does not mark install or skip
#      must exit with a code the page allows.
#
# Usage:  scripts/ci_quickstart.sh [<tree>]          (default: the repository root)
# Env:    IRIS_QUICKSTART_PYTHON      Python for the tool env (default 3.12)
#         IRIS_QUICKSTART_BUDGET_S    the install budget in seconds (default 60)
#         IRIS_QUICKSTART_WARM_CACHE=1  reuse uv's cache (local iteration only: the
#                                      number is then not the cold-install number)
# Requires: uv, python3. Bash 3.2 compatible.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TREE="$(cd "${1:-$ROOT}" && pwd)"
PY_VERSION="${IRIS_QUICKSTART_PYTHON:-3.12}"
BUDGET="${IRIS_QUICKSTART_BUDGET_S:-60}"
command -v uv >/dev/null || { echo "ci_quickstart: uv is required" >&2; exit 2; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/iris-quickstart.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home" "$WORK/tools" "$WORK/bin"
summary() { # a line for the GitHub job summary, when there is one
  [[ -n "${GITHUB_STEP_SUMMARY:-}" ]] && printf '%s\n' "$*" >> "$GITHUB_STEP_SUMMARY"
  return 0
}
now() { python3 -c 'import time; print(f"{time.time():.3f}")'; }

# 1. The install, timed.
CACHE_ARGS=()
[[ "${IRIS_QUICKSTART_WARM_CACHE:-0}" == "1" ]] || CACHE_ARGS=(--cache-dir "$WORK/uv-cache")
echo "quickstart: uv tool install \"<tree>[email]\" (Python $PY_VERSION, cache: ${CACHE_ARGS[*]:-warm})"
START=$(now)
UV_TOOL_DIR="$WORK/tools" UV_TOOL_BIN_DIR="$WORK/bin" \
  uv tool install -q ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"} --python "$PY_VERSION" "${TREE}[email]"
SECS=$(python3 -c "print(f'{$(now) - $START:.1f}')")
echo "quickstart: [email] install took ${SECS}s (target < ${BUDGET}s)"
summary "### quickstart"
summary "- \`uv tool install \"iris-harness[email]\"\`, cold cache: **${SECS} s** (target < ${BUDGET} s)"
if ! python3 -c "import sys; sys.exit(0 if $SECS < $BUDGET else 1)"; then
  echo "::error title=quickstart::[email] install took ${SECS}s, over the ${BUDGET}s target (R19)"
  exit 1
fi

# The user's shell, minus everything of this machine's: a throwaway HOME, no keyring, a
# model server that is a closed port on this host (the probe fails fast, nothing leaves).
user() {
  env -i \
    HOME="$WORK/home" \
    PATH="$WORK/bin:/usr/bin:/bin" \
    LANG="${LANG:-C.UTF-8}" \
    COLUMNS=200 \
    TMPDIR="$WORK" \
    IRIS_HOME="$WORK/home/.iris" \
    OLLAMA_BASE_URL="http://127.0.0.1:9" \
    PYTHON_KEYRING_BACKEND="keyring.backends.fail.Keyring" \
    IRIS_VAULT_MASTER_KEY="$KEY" \
    "$@"
}
KEY="$("$WORK/tools/iris-harness/bin/python" -c \
  'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"

# 2. iris doctor.
set +e
user iris doctor --json > "$WORK/doctor.json" 2> "$WORK/doctor.err"
DOCTOR_RC=$?
set -e
python3 - "$WORK/doctor.json" "$DOCTOR_RC" <<'PY'
import json
import sys

path, rc = sys.argv[1], int(sys.argv[2])
try:
    report = json.load(open(path, encoding="utf-8"))
except ValueError:
    sys.exit(f"quickstart: iris doctor printed no JSON report (exit {rc})")
for check in report["checks"]:
    print(f"  {check['status']:<4} {check['name']:<16} {check['detail']}")
verdict = report["verdict"]
print(f"quickstart: iris doctor verdict {verdict!r} (exit {rc})")
if rc != report["exit_code"]:
    sys.exit(f"quickstart: exit code {rc} does not match the report's {report['exit_code']}")
if verdict == "not_ready":
    sys.exit("quickstart: the doctor says this install cannot even run the demo")
PY
summary "- \`iris doctor\`: exit ${DOCTOR_RC} ($(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['verdict'])" "$WORK/doctor.json"))"

# 3. iris email demo, offline on the scripted model.
set +e
user iris email demo --home "$WORK/demo" > "$WORK/demo.out" 2>&1
DEMO_RC=$?
set -e
cat "$WORK/demo.out"
[[ $DEMO_RC -eq 0 ]] || { echo "quickstart: iris email demo exited $DEMO_RC" >&2; exit 1; }
for expected in "What IRIS just did" "Network connections attempted: 0" \
  "Labels (setup step 6): labelled"; do
  grep -qF "$expected" "$WORK/demo.out" ||
    { echo "quickstart: the demo's output lacks: $expected" >&2; exit 1; }
done
summary "- \`iris email demo\`: OK (offline, scripted model, 0 network connections)"

# 4. The getting-started pages (R7: every one is exercised by CI), as the docs site shows
#    them: every fenced shell block, in page order, in the same fresh user's shell. The
#    install block is the one step above; blocks that need a real mailbox, a browser or a
#    model download say so on the page (<!-- ci: skip ... -->), and the unit suite parses
#    their commands against the CLI instead (tests/unit/test_docs/test_getting_started.py).
mkdir -p "$WORK/getting-started"
user python3 "$TREE/scripts/getting_started.py" run \
  --docs "$TREE/docs/getting-started" --workdir "$WORK/getting-started" ||
  { echo "quickstart: a getting-started page failed" >&2; exit 1; }
summary "- getting-started pages: every runnable block OK"
echo "quickstart: OK (install ${SECS}s, doctor exit ${DOCTOR_RC}, demo OK, getting-started OK)"
