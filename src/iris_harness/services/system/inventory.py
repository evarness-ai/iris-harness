"""Runtime inventory — what IRIS is currently running (ADR-0069 fast-follow #5).

Local, zero-egress introspection: the IRIS version + git rev, the Python version,
key installed package versions, and the locally-pulled Ollama models. The
"is something newer/better available?" check is deliberately out of scope here —
that crosses the privacy boundary (PyPI / model registries) and is a separate
opt-in, egress-gated concern.
"""

from __future__ import annotations

import os
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

import iris_harness
from iris_harness.foundation.env import probe_host

# Key dependencies worth surfacing (the model/runtime backbone). Names are the
# distribution names as installed.
_KEY_PACKAGES: tuple[str, ...] = (
    "fastapi",
    "pydantic",
    "chromadb",
    "langchain",
    "langchain-ollama",
    "ollama",
    "httpx",
    "opentelemetry-sdk",
)


@dataclass(frozen=True)
class RuntimeInventory:
    iris_version: str
    git_branch: str | None
    git_rev: str | None
    python_version: str
    packages: dict[str, str]
    ollama_models: list[dict[str, object]] = field(default_factory=list)


def _git(args: list[str]) -> str | None:
    """Run a read-only git command; None if git/.git is unavailable."""
    # Fixed, read-only git subcommands; no user input reaches the argv.
    try:
        out = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:  # noqa: BLE001 — no git, not a repo, timeout, etc.
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _version_of(name: str) -> str:
    try:
        return _pkg_version(name)
    except PackageNotFoundError:
        return "—"


def _fetch_ollama_tags(url: str) -> dict[str, object]:
    import httpx

    return httpx.get(url, timeout=1.5).json()  # type: ignore[no-any-return]


def _ollama_models(
    fetch: Callable[[str], dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """List locally-pulled Ollama models (name, byte size, parameter size).

    Best-effort: an unreachable Ollama yields an empty list, never an error.
    ``fetch`` is injectable for tests (default hits the local /api/tags)."""
    host = probe_host(os.environ.get("IRIS_API_HOST", "127.0.0.1"))
    port = os.environ.get("IRIS_OLLAMA_PORT", "11434")
    url = f"http://{host}:{port}/api/tags"
    getter = fetch or _fetch_ollama_tags
    try:
        data = getter(url)
    except Exception:  # noqa: BLE001 — Ollama down / bad payload → no models
        return []
    raw_models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(raw_models, list):
        return []
    models: list[dict[str, object]] = []
    for entry in raw_models:
        if not isinstance(entry, dict):
            continue
        details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
        models.append(
            {
                "name": entry.get("name"),
                "size": entry.get("size"),
                "parameter_size": (details or {}).get("parameter_size"),
            }
        )
    return models


def runtime_inventory(
    *,
    ollama_fetch: Callable[[str], dict[str, object]] | None = None,
) -> RuntimeInventory:
    """Snapshot what IRIS is currently running. Pure, local reads."""
    return RuntimeInventory(
        iris_version=iris_harness.__version__,
        git_branch=_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        git_rev=_git(["rev-parse", "--short", "HEAD"]),
        python_version=platform.python_version(),
        packages={name: _version_of(name) for name in _KEY_PACKAGES},
        ollama_models=_ollama_models(fetch=ollama_fetch),
    )


__all__ = ["RuntimeInventory", "runtime_inventory"]
