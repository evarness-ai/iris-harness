#!/usr/bin/env bash
# Local stand-in for .github/workflows/ci-linux.yml.
#
# The hosted workflow is disabled (see "Why this exists" below), so this script IS
# the gate. It runs the same steps, in the same order, with the same environment as
# the two CI jobs:
#
#   job quality      ruff -> black -> mypy -> lint-imports -> pytest (full) -> playground smoke
#   job identity-scan  gitleaks + PII patterns over tracked files
#   job webui        Playwright specs — not in ci-linux.yml; the public CI's webui job; see below
#
# Usage:
#   scripts/ci_local.sh              # both jobs, exactly what CI ran
#   scripts/ci_local.sh --fast       # changed-scope tests instead of the full suite
#   scripts/ci_local.sh --no-ml      # also hide the `ml` + `phoenix` extras from pytest (below)
#   scripts/ci_local.sh --quality    # skip the identity scan
#   scripts/ci_local.sh --scan       # only the identity scan
#   scripts/ci_local.sh --webui      # force the webui viewport smoke (auto when webui/ changed)
#   scripts/ci_local.sh --no-webui   # skip it even when webui/ changed
#   scripts/ci_local.sh --smoke      # ONLY the smoke tests: pytest -m smoke + playground run
#   IRIS_PYTEST_WORKERS=4 scripts/ci_local.sh   # cap xdist workers (default auto = one per core)
#
# Smoke tests (`@pytest.mark.smoke`: end-to-end playground runs against a real
# in-process runtime) are deselected by pyproject's addopts, so plain `pytest`
# and --fast skip them. The full-suite path runs them as their own step, so the
# no-flag gate still covers everything it covered before the marker existed.
#
# Why this exists: waiting ~8 minutes for a hosted runner before every merge was the
# slow part of the loop, and the same checks run here in ~3. Re-enable the hosted
# workflow with `gh workflow enable ci-linux.yml` (and disable with `gh workflow
# disable ci-linux.yml`); the YAML is unchanged and still correct.
#
# WHERE THIS IS NOT CI, honestly:
#   1. This machine is macOS, the runner was ubuntu-latest. pyobjc is darwin-scoped
#      and lazily imported, so macOS has MORE available, never less.
#   2. CI installed from poetry.lock with NO extras; this venv has `-E ml`
#      (sentence-transformers -> torch). So a module-level import of an ml-only
#      package passes here and would have failed there. `--no-ml` closes that gap by
#      making those packages unimportable for the pytest step. The `phoenix` extra
#      (arize-phoenix) is hidden with them, since CI installs without it too.
#   3. CI installed from a fresh clone, so it proved the lock resolves at all. The
#      closest local check is `poetry check --lock`, which this script runs.
# Bash 3.2 compatible (macOS /bin/bash): no associative arrays, no mapfile.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Exactly the workflow's `env:` block. IRIS_DISABLE_ARBITER is deliberately NOT set:
# CI does not set it, and the point of this script is to match CI.
export IRIS_AUTH_SECRET=ci-only-secret
export IRIS_DISABLE_WARMUP=1
export PYTHONDONTWRITEBYTECODE=1

# pytest-xdist worker count: IRIS_PYTEST_WORKERS=<n>|auto|logical, default auto (one per
# core). Set it lower when several gates share the machine; 0 runs without xdist.
WORKERS="${IRIS_PYTEST_WORKERS:-auto}"
if [[ ! "$WORKERS" =~ ^(auto|logical|[0-9]+)$ ]]; then
  echo "ci_local: IRIS_PYTEST_WORKERS must be a number, auto or logical (got '$WORKERS')" >&2
  exit 2
fi

GITLEAKS_PIN="8.30.1"   # keep in step with .github/workflows/ci-linux.yml and brew

RUN_QUALITY=1
RUN_SCAN=1
FAST=0
NO_ML=0
RUN_WEBUI=auto   # auto | 1 | 0
SMOKE_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --fast) FAST=1 ;;
    --no-ml) NO_ML=1 ;;
    --quality) RUN_SCAN=0 ;;
    --scan) RUN_QUALITY=0 ;;
    --webui) RUN_WEBUI=1 ;;
    --no-webui) RUN_WEBUI=0 ;;
    --smoke) SMOKE_ONLY=1 ;;
    -h|--help) sed -n '2,/^# Bash 3.2/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

RESULTS=""        # "status<TAB>seconds<TAB>name" per line
FAILED=0

step() {
  local name="$1"; shift
  printf '\033[1m==> %s\033[0m\n' "$name"
  local start; start=$(date +%s)
  if "$@"; then
    local secs=$(( $(date +%s) - start ))
    RESULTS="${RESULTS}pass	${secs}	${name}
"
    printf '    ok (%ss)\n\n' "$secs"
    return 0
  fi
  local secs=$(( $(date +%s) - start ))
  RESULTS="${RESULTS}FAIL	${secs}	${name}
"
  FAILED=1
  printf '\033[31m    FAILED (%ss)\033[0m\n\n' "$secs"
  return 1
}

skip() {
  RESULTS="${RESULTS}skip	0	$1
"
}

# `-m smoke` on the command line replaces addopts' `-m`, so exactly the smoke
# tests run (real_llm / real_embeddings stay out: none is marked smoke).
run_smoke_pytest() {
  poetry run pytest --no-cov -n "$WORKERS" -q -m smoke
}

# ------------------------------------------------------------ mode: --smoke
if [[ $SMOKE_ONLY -eq 1 ]]; then
  RUN_QUALITY=0
  RUN_SCAN=0
  RUN_WEBUI=0
  step "Pytest (smoke marker only)" run_smoke_pytest &&
  step "Playground smoke (deterministic suite)" \
      poetry run iris playground run core-deterministic
fi

# --------------------------------------------------------------- job: quality
if [[ $RUN_QUALITY -eq 1 ]]; then
  # Pre-flight, not a CI step: CI proved the lock resolves by installing from it on a
  # fresh clone. A mismatch fails the gate but does not stop the run, because it tells
  # you nothing about whether the code is correct.
  step "poetry check --lock (stands in for CI's fresh install)" \
      poetry check --lock || true

  # From here, stop at the first failure, the way a CI job's steps do.
  step "Ruff (lint)"          poetry run ruff check src/ tests/ examples/ &&
  step "Black (format check)" poetry run black --check src/ tests/ examples/ &&
  step "Mypy (types)"         poetry run mypy src/iris_harness/ src/memris/ $( [[ -d src/iris_personal ]] && echo src/iris_personal/ ) $( [[ -d src/iris_code ]] && echo src/iris_code/ ) &&
  # The examples type-check against the stable tier as a plugin author's would. MYPYPATH:
  # in a worktree sharing the main checkout's venv, the editable install points there.
  step "Mypy (examples)"      env MYPYPATH="$ROOT/src" poetry run mypy examples/ &&
  step "Import contracts"     poetry run lint-imports &&
  {
    if [[ $FAST -eq 1 ]]; then
      step "Pytest (CHANGED SCOPE — not what CI ran)" bash scripts/changed_tests.sh
    elif [[ $NO_ML -eq 1 ]]; then
      # Exported, not prefixed onto the `step` call: a var assignment in front of a
      # shell function leaks into the rest of the shell in bash, and xdist workers
      # need it in the environment anyway.
      export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
      step "Pytest (full suite, ml extra hidden)" \
        poetry run pytest --no-cov -n "$WORKERS" -q -p ci_no_ml
    else
      step "Pytest (full suite, no models)" poetry run pytest --no-cov -n "$WORKERS" -q
    fi
  } &&
  {
    # The full suite above deselects the smoke marker; run it here so the full
    # gate still covers it. --fast is the changed scope and leaves it out.
    if [[ $FAST -eq 1 ]]; then
      skip "Pytest smoke marker (--fast)"
    else
      step "Pytest (smoke marker)" run_smoke_pytest
    fi
  } &&
  step "Playground smoke (deterministic suite)" \
      poetry run iris playground run core-deterministic
else
  skip "job: quality"
fi

# ---------------------------------------------------- job: identity-scan
# A separate job in CI, so it runs even when quality failed.
if [[ $RUN_SCAN -eq 1 ]]; then
  if ! command -v gitleaks >/dev/null 2>&1; then
    echo "gitleaks not installed — brew install gitleaks (CI pins v${GITLEAKS_PIN})" >&2
    RESULTS="${RESULTS}FAIL	0	PII + secret scan (gitleaks missing)
"
    FAILED=1
  else
    have=$(gitleaks version 2>/dev/null | tr -d 'v[:space:]')
    [[ "$have" == "$GITLEAKS_PIN" ]] || \
      echo "note: gitleaks ${have} here, CI pins ${GITLEAKS_PIN} — rule sets may differ" >&2
    step "PII + secret scan (block tier)" bash scripts/oss_pii_scan.sh
  fi
else
  skip "job: identity-scan"
fi

# ------------------------------------------------------------ job: webui
# The phone viewport smoke (mobile + cloud UI plan, decisions 22 and 33): every
# nav route (webui/tests/routes.ts) at 390x844 in WebKit, against canned fixtures, asserting the
# screen did not crash, does not scroll sideways, and logs no console errors.
#
# Not in ci-linux.yml (no Node step, and disabled anyway). The public CI's `webui` job
# (.github/public/workflows/ci.yml) runs the same `npx playwright test` on every PR.
# Here it runs by default ONLY when webui/ changed —
# building the bundle and driving a browser costs ~40s, which is not worth paying
# on a Python-only change.
#
# "Changed" here is the merge-base with origin/main plus the working tree, which
# is broader than scripts/changed_tests.sh's upstream-aware window. That is the
# deliberate direction to err in for a gate: running the smoke when it was not
# needed wastes 40 seconds, skipping it when it was needed ships the regression.
webui_changed() {
  local base changed
  base=$(git merge-base HEAD origin/main 2>/dev/null || git rev-parse HEAD~1 2>/dev/null || true)
  changed=$(
    {
      [[ -n "$base" ]] && git diff --name-only "$base" HEAD
      git diff --name-only HEAD
      git diff --name-only --cached
      git ls-files --others --exclude-standard
    } 2>/dev/null
  )
  # Matched with a bash pattern, NOT `| grep -q`. Under `set -o pipefail` grep -q
  # exits on the first match, the git command upstream takes SIGPIPE (141), and
  # the pipeline reports failure even though the match succeeded — so the smoke
  # silently skipped on a branch that had changed webui/. A gate that lies about
  # what it skipped is worse than no gate.
  [[ $'\n'"$changed" == *$'\n'webui/* ]]
}

run_webui_smoke() {
  ( cd "$ROOT/webui" || return 1
    # npm ci on every run would add ~20s; only install when the tree is absent or
    # older than the lockfile (a dependency bump since the last smoke).
    if [[ ! -d node_modules || package-lock.json -nt node_modules ]]; then
      echo "    installing webui dependencies…"
      npm ci --silent || return 1
    fi
    npx playwright test )
}

if [[ "$RUN_WEBUI" == "1" ]] || { [[ "$RUN_WEBUI" == "auto" ]] && webui_changed; }; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "npm not installed — the webui viewport smoke needs Node (brew install node)" >&2
    RESULTS="${RESULTS}FAIL	0	Webui viewport smoke (npm missing)
"
    FAILED=1
  elif [[ ! -d "$HOME/Library/Caches/ms-playwright" ]]; then
    # A missing browser is a setup gap, not a code failure, but it must not read
    # as a pass: the screens went unchecked either way.
    # Two engines, on purpose: WebKit is Safari, the target; Chromium is the
    # only one Playwright runs service workers in (webui/tests/serviceworker.spec.ts).
    echo "Playwright browsers missing — run: cd webui && npx playwright install webkit chromium" >&2
    RESULTS="${RESULTS}FAIL	0	Webui viewport smoke (browser not installed)
"
    FAILED=1
  else
    step "Webui viewport smoke (390x844, WebKit)" run_webui_smoke
  fi
elif [[ $SMOKE_ONLY -eq 1 ]]; then
  :   # --smoke ran only the smoke steps; no job lines to add
elif [[ "$RUN_WEBUI" == "0" ]]; then
  skip "job: webui (--no-webui)"
else
  skip "job: webui (no webui/ changes)"
fi

# ------------------------------------------------------------------ summary
echo "────────────────────────────────────────────────────────"
printf '%s' "$RESULTS" | while IFS=$'\t' read -r status secs name; do
  [[ -z "${name:-}" ]] && continue
  case "$status" in
    pass) printf '  \033[32mok  \033[0m %-52s %4ss\n' "$name" "$secs" ;;
    FAIL) printf '  \033[31mFAIL\033[0m %-52s %4ss\n' "$name" "$secs" ;;
    *)    printf '  skip %-52s\n' "$name" ;;
  esac
done
echo "────────────────────────────────────────────────────────"

if [[ $FAILED -eq 1 ]]; then
  echo "ci_local: FAILED — do not merge"
  exit 1
fi
if [[ $SMOKE_ONLY -eq 1 ]]; then
  echo "ci_local: smoke passed — ONLY the smoke tests ran, NOT the gate."
  echo "          Run scripts/ci_local.sh with no flags before merging."
  exit 0
fi
if [[ $RUN_QUALITY -eq 0 || $RUN_SCAN -eq 0 ]]; then
  echo "ci_local: passed, but one job was skipped — NOT the full gate."
  echo "          Run scripts/ci_local.sh with no flags before merging."
  exit 0
fi
if [[ $FAST -eq 1 ]]; then
  echo "ci_local: passed, but --fast ran the changed scope, NOT the full suite."
  echo "          Run without --fast before merging."
  exit 0
fi
echo "ci_local: passed — this is what CI (Linux) would have run"
