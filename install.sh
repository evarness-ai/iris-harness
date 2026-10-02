#!/usr/bin/env bash
# Bootstrap a clone of this repo: check Python, run `poetry install`, then hand off
# to `iris setup` for everything after that (home dir, secret, services, Telegram,
# email). `iris setup` itself needs Poetry's venv to already exist, so this one
# step can't live inside it -- that's the whole reason this script exists.
#
# Usage: ./install.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "checking Python..."
if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 not found. Install Python 3.12 first (pyenv, or your OS package manager)." >&2
  exit 1
fi
PY_VERSION="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
if [[ "$PY_VERSION" != "3.12" ]]; then
  echo "error: Python 3.12.x required, found $PY_VERSION (see .python-version)." >&2
  exit 1
fi
echo "  ok ($PY_VERSION)"

echo "checking Poetry..."
if ! command -v poetry >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: Poetry not found.

Install it yourself first (this script won't pipe a remote installer for you):
  curl -sSL https://install.python-poetry.org | python3 -

Then re-run ./install.sh.
EOF
  exit 1
fi
echo "  ok ($(poetry --version))"

echo "installing dependencies (poetry install)..."
poetry install

echo "handing off to the setup wizard..."
echo
exec poetry run iris setup
