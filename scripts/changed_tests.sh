#!/usr/bin/env bash
# Proportionate test gate: run only the tests that the changed files can reach.
#
# Maps every changed path (vs the merge-base with origin/main, plus the working
# tree) to a test target. The tests tree mirrors the source tree (OSS plan M6,
# decision 9), so the rule is one line: replace `src/` with `tests/unit/` and walk
# up until a directory exists —
#     src/<root>/<layer>/<pkg>/x.py  ->  tests/unit/<root>/<layer>/<pkg>/  (or a parent)
#     tests/...                      ->  that test file / directory
#     src/iris_harness/main.py       ->  tests/unit/iris_harness/test_cli/  (the CLI entry point)
#     src/iris_harness/cli/x.py      ->  tests/unit/iris_harness/test_cli/  (its command modules)
# with the pre-M6 flat convention kept as a fallback until every package has moved:
#     src/iris_harness/<pkg>/...     ->  tests/unit/test_<pkg>/
#     services/<svc>/...             ->  tests/unit/test_<svc>/
# A source change that maps to NO test directory escalates to the full suite and
# says so. It never reports "nothing to run" for code — a green line the script did
# not earn is exactly the failure mode a tree move produces.
# The full suite also runs when a change touches something every turn depends on
# (scripts/changed_tests_full_triggers.txt: the composition root, the turn pipeline,
# the plugin layer, shared fixtures, dependencies, or runtime config).
#
# Usage:
#   scripts/changed_tests.sh              # run the selected tests
#   scripts/changed_tests.sh --print      # only print what would run
#   FULL=1 scripts/changed_tests.sh       # force the full suite (what CI runs)
#   IRIS_PYTEST_WORKERS=4 scripts/changed_tests.sh   # cap xdist workers (default auto)
#   CHANGED_OVERRIDE=$'a.py\nb.py' ...     # bypass git: map these paths (tests use this)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PRINT=0
[[ "${1:-}" == "--print" ]] && PRINT=1

# pytest-xdist worker count: IRIS_PYTEST_WORKERS=<n>|auto|logical, default auto (one per
# core). Set it lower when several gates share the machine; 0 runs without xdist.
WORKERS="${IRIS_PYTEST_WORKERS:-auto}"
if [[ ! "$WORKERS" =~ ^(auto|logical|[0-9]+)$ ]]; then
  echo "changed_tests: IRIS_PYTEST_WORKERS must be a number, auto or logical (got '$WORKERS')" >&2
  exit 2
fi

if [[ "${FULL:-0}" == "1" ]]; then
  echo "changed_tests: FULL=1 → full suite"
  [[ $PRINT -eq 1 ]] || exec poetry run pytest --no-cov -n "$WORKERS" -q
  exit 0
fi

# "What am I about to push": commits since the upstream tracking branch when there is
# one (a follow-up on a long-lived branch stays small). On a branch's FIRST push there is
# no upstream, and falling straight back to origin/main would re-select every commit of
# the branch this one is stacked on — which is the normal state here, since milestone
# branches stack on an open PR. So ask git for the commits that exist on no remote branch
# and start just before the oldest of them. BASE=<ref> overrides both.
# CHANGED_OVERRIDE names the paths itself, so then nothing here asks git: the script (and
# its tests) also run in a tree that is not a git checkout, such as the public export.
BASE="${BASE:-}"
if [[ -z "${CHANGED_OVERRIDE:-}" ]]; then
  if [[ -z "$BASE" ]]; then
    BASE=$(git merge-base HEAD '@{upstream}' 2>/dev/null || true)
  fi
  if [[ -z "$BASE" ]] && git rev-parse --verify -q HEAD >/dev/null; then
    UNIQUE=$(git rev-list HEAD --not --remotes 2>/dev/null || true)
    if [[ -z "$UNIQUE" ]]; then
      # Every commit is already on a remote branch: only the working tree is new.
      BASE=$(git rev-parse HEAD)
    else
      BASE=$(git rev-parse "$(printf '%s\n' "$UNIQUE" | tail -1)^" 2>/dev/null || true)
    fi
  fi
  if [[ -z "$BASE" ]]; then
    BASE=$(git merge-base HEAD origin/main 2>/dev/null || git rev-parse HEAD~1)
  fi
fi
if [[ -n "${CHANGED_OVERRIDE:-}" ]]; then
  CHANGED=$(printf '%s\n' "$CHANGED_OVERRIDE" | sort -u)
else
  CHANGED=$( (git diff --name-only "$BASE" HEAD; git diff --name-only HEAD; git diff --name-only --cached) | sort -u )
fi

if [[ -z "$CHANGED" ]]; then
  echo "changed_tests: no changes vs $BASE → nothing to run"
  exit 0
fi

# Paths whose change invalidates every test's assumptions: one regex per line in the
# triggers file (a unit test keeps every line pointing at a real file).
FULL_TRIGGERS=""
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  FULL_TRIGGERS="${FULL_TRIGGERS:+${FULL_TRIGGERS}|}${line}"
done < "$ROOT/scripts/changed_tests_full_triggers.txt"
if printf '%s\n' "$CHANGED" | grep -qE "$FULL_TRIGGERS"; then
  echo "changed_tests: a core file changed → full suite:"
  printf '%s\n' "$CHANGED" | grep -E "$FULL_TRIGGERS" | sed 's/^/    /'
  [[ $PRINT -eq 1 ]] || exec poetry run pytest --no-cov -n "$WORKERS" -q
  exit 0
fi

# macOS ships bash 3.2 (no associative arrays): accumulate lines, dedupe with sort -u.
TARGETS=""
MISSES=""
add_target() { TARGETS="${TARGETS}${1}
"; }
# Mirror rule: tests/unit/<path under src/>, walking up to (not including) tests/unit.
# Prints the first existing directory, or nothing.
#
# A match at the ROOT level (tests/unit/<root>, e.g. tests/unit/iris_harness) counts
# only when that directory holds test files of its own — as tests/unit/iris_code does.
# When it is merely a container of per-layer directories, landing there means the path
# mapped to nothing in particular, and the caller must escalate to the full suite. Until
# M6.2 finishes, most of the core has no mirrored home, so without this check the
# existence of tests/unit/iris_harness/ would quietly answer for all of it: a change to
# an untested package would "run tests" that cannot reach it. That is the green line the
# script did not earn, which this whole mechanism exists to prevent (M6.0).
mirror_target() {
  local dir root
  dir="tests/unit/${1#src/}"
  root="tests/unit/$(echo "${1#src/}" | cut -d/ -f1)"
  dir=$(dirname "$dir")
  while [[ "$dir" != "tests/unit" && "$dir" != "." ]]; do
    if [[ -d "$dir" ]]; then
      if [[ "$dir" == "$root" ]] && ! compgen -G "$dir/test_*.py" >/dev/null 2>&1; then
        return 1
      fi
      printf '%s' "$dir"; return 0
    fi
    dir=$(dirname "$dir")
  done
  return 1
}
while IFS= read -r path; do
  case "$path" in
    tests/*.py)
      add_target "$path" ;;
    tests/*)
      ;;  # non-python test assets: covered by their directory's tests below
    src/*)
      hit=""
      if t=$(mirror_target "$path"); then
        hit=1; add_target "$t"
      fi
      # Pre-M6 flat convention, kept until every package has moved (M6.2).
      case "$path" in
        src/iris_harness/main.py | src/iris_harness/cli/*)
          # The CLI: the entry point (Typer wiring + cmd_* handlers) and its command
          # modules. main.py mirrors to the container root, which answers for nothing,
          # and cli/ mirrors to a `cli/` test dir that does not exist: the CLI tests
          # live in `test_cli/`. Name that home, or every subcommand escalates.
          [[ -d "tests/unit/iris_harness/test_cli" ]] && { hit=1; add_target "tests/unit/iris_harness/test_cli"; } ;;
        src/iris_harness/plugins_builtin/*/*)
          # A built-in plugin: tests live under its own name, not the container's.
          plug=$(echo "$path" | cut -d/ -f4)
          [[ -d "tests/unit/test_${plug}" ]] && { hit=1; add_target "tests/unit/test_${plug}"; }
          [[ -d "tests/unit/test_plugins" ]] && { hit=1; add_target "tests/unit/test_plugins"; } ;;
        src/iris_harness/*/*)
          pkg=$(echo "$path" | cut -d/ -f3)
          [[ -d "tests/unit/test_${pkg}" ]] && { hit=1; add_target "tests/unit/test_${pkg}"; } ;;
        src/iris_harness/*.py)
          mod=$(basename "$path" .py)
          [[ -d "tests/unit/test_${mod}" ]] && { hit=1; add_target "tests/unit/test_${mod}"; }
          [[ -f "tests/unit/test_${mod}.py" ]] && { hit=1; add_target "tests/unit/test_${mod}.py"; } ;;
      esac
      [[ -n "$hit" ]] || MISSES="${MISSES}${path}
" ;;
    services/*/*)
      svc=$(echo "$path" | cut -d/ -f2)
      if [[ -d "tests/unit/test_${svc}" ]]; then add_target "tests/unit/test_${svc}"; else MISSES="${MISSES}${path}
"; fi ;;
    scripts/*)
      [[ -d "tests/unit/test_scripts" ]] && add_target "tests/unit/test_scripts" ;;
    examples/*/*)
      # An example carries its own test, and the stable-tier contract holds every one.
      add_target "$(echo "$path" | cut -d/ -f1-2)"
      add_target "tests/unit/test_stable_tier.py" ;;
    examples/*)
      add_target "tests/unit/test_stable_tier.py" ;;
    config/skills/*)
      add_target "tests/unit/test_skills" ;;
    config/*)
      add_target "tests/unit/test_playground" ;;
    # Docs are not untested any more: tests/unit/test_docs asserts that every source
    # path a current-state doc names actually exists. A doc edit that points at a moved
    # file is a real failure, so it must reach a test rather than "nothing to run".
    docs/*.md|docs/*/*.md|docs/*/*/*.md|README.md|CONTRIBUTING.md|CLAUDE.md|*/CLAUDE.md)
      add_target "tests/unit/test_docs" ;;
    # The image and the local stack; the console's wiring; the owner's server kit
    # (private: the public tree has neither deploy/ nor its tests).
    Dockerfile|docker-compose*.yml|.dockerignore)
      add_target "tests/unit/test_container" ;;
    webui/*)
      add_target "tests/unit/test_webui" ;;
    deploy/*)
      [[ -d "tests/unit/test_deploy" ]] && add_target "tests/unit/test_deploy" ;;
  esac
done <<< "$CHANGED"

# A source change with no test home is not "nothing to run" — it is "we cannot tell".
if [[ -n "$MISSES" ]]; then
  echo "changed_tests: no test directory maps to these source paths → full suite:"
  printf '%s' "$MISSES" | grep -v '^$' | sort -u | sed 's/^/    /'
  [[ $PRINT -eq 1 ]] || exec poetry run pytest --no-cov -n "$WORKERS" -q
  exit 0
fi

# Test directories whose files were touched (so a changed fixture re-runs its siblings).
while IFS= read -r path; do
  case "$path" in
    tests/*/conftest.py|tests/*/*/conftest.py)
      add_target "$(dirname "$path")" ;;
  esac
done <<< "$CHANGED"

# Drop targets that no longer exist: a change that DELETES a test file lists that path
# as changed, and pytest exits 4 ("file or directory not found") on a stale argument.
SELECTED=""
while IFS= read -r target; do
  [[ -z "$target" ]] && continue
  [[ -e "$target" ]] && SELECTED="${SELECTED}${target}
"
done <<< "$(printf '%b' "$TARGETS" | grep -v '^$' | sort -u || true)"
SELECTED=$(printf '%s' "$SELECTED" | grep -v '^$' || true)

# Drop a target that sits inside another selected DIRECTORY: the directory already runs
# it. This is not tidiness — pytest (8.4) given `dir` plus `dir/.../test_x.py` collects
# only the file, so tests/unit/iris_harness/memory (477 tests) + one file inside it ran
# 29. The narrowing was silent: the summary line still said "passed".
NESTED=""
while IFS= read -r target; do
  [[ -z "$target" ]] && continue
  while IFS= read -r outer; do
    if [[ -n "$outer" && "$outer" != "$target" && -d "$outer" && "$target" == "${outer%/}/"* ]]; then
      NESTED="${NESTED}${target}
"
      break
    fi
  done <<< "$SELECTED"
done <<< "$SELECTED"
if [[ -n "$NESTED" ]]; then
  SELECTED=$(printf '%s\n' "$SELECTED" | grep -vxF -f <(printf '%s' "$NESTED" | grep -v '^$') || true)
fi

if [[ -z "$SELECTED" ]]; then
  echo "changed_tests: changes touch no tested code (docs only) → nothing to run"
  exit 0
fi

echo "changed_tests: $(printf '%s\n' "$CHANGED" | wc -l | tr -d ' ') changed path(s) → running:"
printf '%s\n' "$SELECTED" | sed 's/^/    /'
[[ $PRINT -eq 1 ]] && exit 0
# shellcheck disable=SC2086
exec poetry run pytest --no-cov -n "$WORKERS" -q $SELECTED
