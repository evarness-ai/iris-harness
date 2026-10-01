"""Resolve the harness's data/config locations, with test isolation.

Foundation's, not the kernel's, as of M6.2: the kernel writes the governance stores and
the observability layer READS one of them (the trace builder renders the audit ledger),
so the path cannot belong to either without one importing the other. It belongs under
both.

Originally ``governance/paths.py``.

Production keeps the user's real ``~/.local/share/iris`` (data) and ``~/.config/iris``
(config). The test suite relocates ``IRIS_HOME`` to a throwaway temp dir *before any
import* (see ``tests/conftest.py``); basing these paths on it ensures a bare store
constructor never writes into the developer's real governance stores — the audit
ledger, checkpoints, cost ledger, approvals, side-effects ledger, and vault. This is
the same leak class fixed for the session log (#295) and the audit DB (#296): a module
that resolves ``Path.home()`` directly ignores ``IRIS_HOME`` and leaks test writes into
the real profile.

Each store's explicit ``db_path``/env override (if any) still takes precedence; these
helpers only supply the default.
"""

from __future__ import annotations

import os
from functools import lru_cache
from importlib.resources import files
from pathlib import Path


def governance_data_dir() -> Path:
    """Directory for governance *data* stores (SQLite DBs, archive).

    ``IRIS_HOME`` (test isolation) → ``$IRIS_HOME/governance``; otherwise the
    production XDG data location ``~/.local/share/iris``.
    """
    home = os.environ.get("IRIS_HOME")
    if home:
        return Path(home) / "governance"
    return Path.home() / ".local" / "share" / "iris"


def governance_config_dir() -> Path:
    """Directory for governance *config* stores (e.g. the secret vault).

    ``IRIS_HOME`` (test isolation) → ``$IRIS_HOME/governance``; otherwise the
    production XDG config location ``~/.config/iris``.
    """
    home = os.environ.get("IRIS_HOME")
    if home:
        return Path(home) / "governance"
    return Path.home() / ".config" / "iris"


__all__ = [
    "DATA_DIR_ENV",
    "config_dir",
    "config_path",
    "config_root",
    "data_dir",
    "default_config_dir",
    "governance_config_dir",
    "governance_data_dir",
    "iris_home",
    "packaged_config_dir",
    "repo_root",
    "workspace_dir",
]

DATA_DIR_ENV = "IRIS_DATA_DIR"

_REPO_ROOT_MARKER = "pyproject.toml"


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """The checkout this package was imported from, or its best guess.

    Walks up from this file looking for the ``pyproject.toml`` that marks a
    checkout. Five call sites used to count ``parents[N]`` from their own depth
    instead, and M6.2 broke one of them per layer, silently: the MCP server
    exposed no tools at all after layer 3, and the API could not find the
    governor policy after layer 10. Counting is the bug; a marker is the fix,
    because a marker does not move when the file does.

    Installed from a wheel there is no checkout above the package, the walk finds
    nothing, and this returns the directory the package sits in -- which is what
    the arithmetic returned too. Callers that read repo files must already handle
    their absence.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / _REPO_ROOT_MARKER).is_file():
            return candidate
    return here.parents[2]


def _checkout_root() -> Path | None:
    """The checkout this package was imported from, or ``None`` when installed.

    A checkout is the directory ``repo_root()`` found a ``pyproject.toml`` in, with this
    package's source under its ``src/``. The ``src/`` check is what keeps a wheel
    installed into some other project's ``.venv`` (whose walk up also meets a
    ``pyproject.toml``, the other project's) from reading that project's files.
    """
    root = repo_root()
    here = Path(__file__).resolve()
    if not (root / _REPO_ROOT_MARKER).is_file() or not here.is_relative_to(root / "src"):
        return None
    return root


def _checkout_config_dir() -> Path | None:
    """The checkout's ``config/``, when this package was imported from a checkout."""
    root = _checkout_root()
    if root is None:
        return None
    candidate = root / "config"
    return candidate if candidate.is_dir() else None


def packaged_config_dir() -> Path:
    """The default config the wheel ships, at ``iris_harness/_data/config``.

    pyproject.toml maps the repo's ``config/`` there at build time (``packages``, with
    ``to``), so git holds one copy. In a checkout the directory does not exist, and
    ``config_dir()`` never gets this far.
    """
    return Path(str(files("iris_harness"))) / "_data" / "config"


def config_dir() -> Path:
    """The directory IRIS reads its config from. Every config reader goes through here.

    In order:

    1. ``IRIS_CONFIG_DIR``, when set: an explicit override, as before (a relative value
       is still relative to the current directory; ``~`` expands).
    2. The checkout's ``config/``, when running from a checkout (``repo_root()`` found
       the ``pyproject.toml`` this package's ``src/`` sits under, and ``config/`` is
       beside it). The developer's tree and the Docker image (WORKDIR /app, with
       /app/config, installed editable) resolve here, exactly the files they read
       before.
    3. The defaults shipped inside the package (``packaged_config_dir()``), for a plain
       ``pip install`` with no checkout anywhere.

    Before this, ~25 readers used a bare ``Path("config")``, relative to the current
    directory -- the same directory when run from the repo root, nothing at all from
    anywhere else -- and ~15 used ``repo_root() / "config"``, which from a wheel is
    ``site-packages/config``. Not cached: tests move ``IRIS_CONFIG_DIR`` per test.
    """
    override = os.environ.get("IRIS_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return default_config_dir()


def default_config_dir() -> Path:
    """``config_dir()`` without the ``IRIS_CONFIG_DIR`` override: the checkout's
    ``config/``, else the packaged defaults.

    For the readers that fall back to the shipped file when the override directory
    lacks one (the memory and identity configs): the override may hold only the files
    it changes.
    """
    return _checkout_config_dir() or packaged_config_dir()


def config_path(*parts: str) -> Path:
    """A file or directory under ``config_dir()``: ``config_path("memory", "retention.yaml")``."""
    return config_dir().joinpath(*parts)


def config_root() -> Path:
    """The directory whose ``config/`` is ``config_dir()``: its parent.

    For the APIs that take a repo root and read ``<root>/config/...`` (the skill
    registry, the MCP bridge, the governor): handed this, they read the resolved config.
    In a checkout it is ``repo_root()``; installed from a wheel it is
    ``iris_harness/_data``, which is why the packaged tree keeps the name ``config``.
    ``build_runtime`` has always derived its root the same way (``cfg.parent``).
    """
    return config_dir().parent


def iris_home() -> Path:
    """``$IRIS_HOME`` (tests, the demo and sandboxes relocate it) else ``~/.iris``."""
    from iris_harness.foundation.plugin_dirs import iris_home as _home

    return _home()


def workspace_dir() -> Path:
    """The owner's workspace, ``$IRIS_HOME/workspace``: the identity files and what a
    plugin keeps beside them (an account's category proposals, a hold-out set)."""
    return iris_home() / "workspace"


def data_dir() -> Path:
    """The directory IRIS keeps its local data in: the SQLite stores, ChromaDB, the wiki.

    Every data reader and writer goes through here (``persistence.data_path`` for one
    file). In order:

    1. ``IRIS_DATA_DIR``, when set: an explicit override (tests, eval and sandbox
       instances, ``iris email demo``). A relative value is relative to the current
       directory, as before; ``~`` expands.
    2. ``$IRIS_HOME/data``, when ``IRIS_HOME`` is set: an IRIS relocated as a whole keeps
       its data with it, the way the governance stores already follow ``IRIS_HOME``.
    3. The checkout's ``data/``, when running from a checkout: the developer's tree and
       the Docker image (``/app/data``) read exactly the directory they read before.
    4. ``~/.iris/data``, for an installed wheel with no checkout anywhere.

    Before this the fallback was a bare ``"data"``, relative to the current directory:
    the checkout's own ``data/`` when run from the repo root, and a stray ``data/`` in
    whatever directory an installed ``iris`` happened to be run from.
    """
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override).expanduser()
    if os.environ.get("IRIS_HOME"):
        return iris_home() / "data"
    root = _checkout_root()
    if root is not None:
        return root / "data"
    return iris_home() / "data"


def audit_db_path() -> Path:
    """The governance audit ledger's location.

    One definition for the kernel that appends to it and the trace builder that reads
    it. ``IRIS_GOVERNANCE_AUDIT_DB_PATH`` overrides; otherwise it sits in the
    governance data dir. Both callers used to resolve this themselves, which is how the
    env override came to be parsed in two places (M6.2).
    """
    override = os.environ.get("IRIS_GOVERNANCE_AUDIT_DB_PATH")
    if override:
        return Path(override)
    return governance_data_dir() / "audit.db"
