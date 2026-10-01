"""Per-session sandbox workspace — a host directory mounted into the container.

Files written by sandboxed commands persist here across calls within the same
session, and surface back to the user as artifact paths.
"""

from __future__ import annotations

import re
from pathlib import Path

_SAFE_ID = re.compile(r"[^A-Za-z0-9_\-]")


class SessionWorkspace:
    """Manages ``~/.iris/sandbox/<session_id>/`` for one chat session."""

    def __init__(self, session_id: str, *, root: Path | None = None) -> None:
        base = root or (Path.home() / ".iris" / "sandbox")
        safe_id = _SAFE_ID.sub("_", session_id) or "default"
        self._path = base / safe_id
        self._path.mkdir(parents=True, exist_ok=True)
        self._known_files: set[str] = set(self._scan())

    @property
    def path(self) -> Path:
        return self._path

    def snapshot_artifacts(self) -> tuple[str, ...]:
        """Return paths of files newly created since the last call.

        Paths are absolute on the host so users can open them directly.
        """
        current = set(self._scan())
        new_files = sorted(current - self._known_files)
        self._known_files = current
        return tuple(str(self._path / rel) for rel in new_files)

    def _scan(self) -> list[str]:
        if not self._path.exists():
            return []
        out: list[str] = []
        for p in self._path.rglob("*"):
            if p.is_file():
                out.append(str(p.relative_to(self._path)))
        return out
