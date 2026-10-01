#!/usr/bin/env bash
# PII / identity scan for the open-source export (OSS plan decision 7, release gate 6).
#
# Two tiers:
#   BLOCK — patterns that must never reach a public commit: personal email addresses,
#           home paths and the machine username, private data (phone, tailnet, amounts
#           from real statements), and any gitleaks finding.
#   WARN  — patterns that are acceptable in the private repo but must be resolved before
#           the clean export: the old GitHub handle (repo URLs), the maintainer's name
#           used as fixture data, private network ranges, fixture data from real life.
#
# The owner's name and GitHub account are NOT PII where they credit the owner (OSS plan
# R13): authorship, README credit, CODEOWNERS. A pattern's optional third column lists
# the path globs (comma-separated, bash `[[ == ]]` syntax) where a match is allowed;
# everywhere else it counts as before. The token `@private` in that column allows every
# file the export never ships (scripts/oss_public_tests.py never-ships): real institutions the
# private finance domain must name, and no shipped fixture may.
#
# Usage:
#   scripts/oss_pii_scan.sh            # scan tracked files; exit 1 only on BLOCK hits
#   scripts/oss_pii_scan.sh --strict   # WARN hits also fail (used by the export script)
#   scripts/oss_pii_scan.sh --staged   # scan only staged files (pre-commit)
#
# Patterns live in scripts/oss_pii_patterns.txt so the list is reviewable and the script
# stays free of the strings it hunts. Format: one `TIER<TAB>regex[<TAB>allowed globs]`
# per line.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATTERNS="$ROOT/scripts/oss_pii_patterns.txt"
STRICT=0
STAGED=0
for arg in "$@"; do
  case "$arg" in
    --strict) STRICT=1 ;;
    --staged) STAGED=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ ! -f "$PATTERNS" ]]; then
  echo "pattern file missing: $PATTERNS" >&2
  exit 2
fi

# Files to scan: tracked (or staged) text files, minus lockfiles and vendored trees.
EXCLUDES=(':!poetry.lock' ':!webui/package-lock.json' ':!**/node_modules/**' ':!scripts/oss_pii_patterns.txt')
if [[ $STAGED -eq 1 ]]; then
  FILES=$(cd "$ROOT" && git diff --cached --name-only --diff-filter=ACMR -- . "${EXCLUDES[@]}")
else
  FILES=$(cd "$ROOT" && git ls-files -- . "${EXCLUDES[@]}")
fi
if [[ -z "$FILES" ]]; then
  echo "oss_pii_scan: nothing to scan"
  exit 0
fi

block_hits=0
warn_hits=0
# NUL-delimited so paths with spaces survive xargs; `|| true` because xargs splits the
# list across several grep invocations and any one with no match exits non-zero.
scan() { # $1 = grep flags, $2 = regex, $3 = newline-separated files
  (cd "$ROOT" && printf '%s\n' "$3" | tr '\n' '\0' | xargs -0 grep "$1" -- "$2" 2>/dev/null) || true
}
# `@private` in a pattern's allowed column: the files the export never ships, as the
# export's own rules say (scripts/oss_public_tests.py never-ships). Resolved once. A tree
# without that script -- the exported one, where nothing is private -- allows nothing.
PRIVATE_LIST=""
private_list() {
  if [[ -z "$PRIVATE_LIST" ]]; then
    PRIVATE_LIST="$(mktemp "${TMPDIR:-/tmp}/iris-pii-private.XXXXXX")"
    trap 'rm -f "$PRIVATE_LIST"' EXIT
    if [[ -f "$ROOT/scripts/oss_public_tests.py" ]]; then
      local py="$ROOT/.venv/bin/python"
      [[ -x "$py" ]] || py="python3"
      if ! printf '%s\n' "$FILES" |
          "$py" "$ROOT/scripts/oss_public_tests.py" never-ships "$ROOT" > "$PRIVATE_LIST"; then
        echo "oss_pii_scan: could not resolve @private (oss_public_tests.py never-ships)" >&2
        exit 2
      fi
    fi
  fi
}
# The files outside every glob in $1 (comma-separated); all of them when $1 is empty.
not_allowed() {
  local f glob skip private=0
  local -a globs
  if [[ -z "$1" ]]; then
    printf '%s\n' "$FILES"
    return
  fi
  IFS=, read -r -a globs <<< "$1"
  for glob in "${globs[@]}"; do
    [[ "$glob" == "@private" ]] && private=1
  done
  printf '%s\n' "$FILES" | while IFS= read -r f; do
    skip=0
    for glob in "${globs[@]}"; do
      [[ "$glob" == "@private" ]] && continue
      # shellcheck disable=SC2053 # the glob is the point
      [[ "$f" == $glob ]] && { skip=1; break; }
    done
    [[ $skip -eq 1 ]] || printf '%s\n' "$f"
  done | if [[ $private -eq 1 && -s "$PRIVATE_LIST" ]]; then
    grep -vxF -f "$PRIVATE_LIST" || true
  else
    cat
  fi
}
while IFS=$'\t' read -r tier regex allowed; do
  [[ -z "$tier" || "$tier" == \#* ]] && continue
  # Resolved here, in this shell: not_allowed runs in a command substitution.
  [[ ",${allowed:-}," == *,@private,* ]] && private_list
  files=$(not_allowed "${allowed:-}")
  [[ -z "$files" ]] && continue
  matched=$(scan -lIiE "$regex" "$files")
  [[ -z "$matched" ]] && continue
  count=$(printf '%s\n' "$matched" | grep -c . || true)
  echo "[$tier] /$regex/ — $count file(s)${allowed:+ (allowed in: $allowed)}"
  scan -nIiE "$regex" "$files" | head -20 | cut -c1-160 | sed 's/^/    /'
  if [[ "$tier" == "BLOCK" ]]; then
    block_hits=$((block_hits + count))
  else
    warn_hits=$((warn_hits + count))
  fi
done < "$PATTERNS"

# Secrets: gitleaks over the SAME files the patterns scanned, copied into a scratch tree
# at their repo-relative paths (so .gitleaks.toml's anchored allowlist still applies).
# Directory mode over the repo itself also read untracked trees — agent worktrees under
# .claude/worktrees/ carried copies of the allowlisted fixtures at un-anchored paths and
# failed every scan. Requires gitleaks >= 8.19.
if command -v gitleaks >/dev/null 2>&1; then
  SECRETS_TREE="$(mktemp -d "${TMPDIR:-/tmp}/iris-pii-scan.XXXXXX")"
  trap 'rm -rf "$SECRETS_TREE" ${PRIVATE_LIST:+"$PRIVATE_LIST"}' EXIT
  # A tracked file deleted in the working tree is skipped: there is nothing to scan.
  (cd "$ROOT" && printf '%s\n' "$FILES" | while IFS= read -r f; do
    [[ -f "$f" ]] && printf '%s\n' "$f"
  done | rsync -a --files-from=- . "$SECRETS_TREE/")
  # --verbose prints each (redacted) finding with rule id, file, and line so a CI log
  # says WHAT tripped, not just that something did.
  if ! (cd "$SECRETS_TREE" && gitleaks dir . --redact --no-banner --verbose \
        --config "$ROOT/.gitleaks.toml" 2>&1 | grep -E "RuleID|File:|Line:|leaks found|no leaks" | tail -40); then
    echo "[BLOCK] gitleaks found secrets"
    block_hits=$((block_hits + 1))
  fi
else
  echo "[WARN] gitleaks not installed (brew install gitleaks) — secret scan skipped"
  warn_hits=$((warn_hits + 1))
fi

echo
echo "oss_pii_scan: BLOCK=$block_hits WARN=$warn_hits strict=$STRICT"
if [[ $block_hits -gt 0 ]]; then
  echo "oss_pii_scan: FAIL — blocking identity/secret patterns present"
  exit 1
fi
if [[ $STRICT -eq 1 && $warn_hits -gt 0 ]]; then
  echo "oss_pii_scan: FAIL (strict) — warn-tier patterns must be resolved before export"
  exit 1
fi
echo "oss_pii_scan: OK"
