"""Session file management for the IRIS CLI.

Sessions are stored as JSON under $IRIS_HOME/sessions/<id>.json (``~/.iris`` unless
IRIS_HOME relocates it -- tests, the demo and the plugin harness do).
Each session tracks: id, cwd, timestamps, message count.
The session ID is passed to the IRIS API so the server can recall
conversation history for that session.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from iris_harness.foundation.paths import iris_home


def default_sessions_dir() -> Path:
    """Where CLI sessions live: ``$IRIS_HOME/sessions``, resolved when asked."""
    return iris_home() / "sessions"


@dataclass
class Session:
    id: str
    cwd: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    preferred_model: str = ""
    active_provider: str = ""
    router_model: str = ""

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        d["updated_at"] = self.updated_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Session:
        return cls(
            id=str(d["id"]),
            cwd=str(d["cwd"]),
            created_at=datetime.fromisoformat(str(d["created_at"])),
            updated_at=datetime.fromisoformat(str(d["updated_at"])),
            message_count=int(str(d.get("message_count") or 0)),
            preferred_model=str(d.get("preferred_model") or ""),
            active_provider=str(d.get("active_provider") or ""),
            router_model=str(d.get("router_model") or ""),
        )


class SessionManager:
    def __init__(self, sessions_dir: Path | None = None) -> None:
        self._dir = sessions_dir or default_sessions_dir()
        self._dir.mkdir(parents=True, exist_ok=True)

    def create(self, cwd: str | None = None) -> Session:
        now = datetime.now(UTC)
        session = Session(
            id=uuid.uuid4().hex[:12],
            cwd=cwd or os.getcwd(),
            created_at=now,
            updated_at=now,
        )
        self._write(session)
        return session

    def load(self, session_id: str) -> Session | None:
        matches = sorted(self._dir.glob("*.json"))
        for path in matches:
            if path.stem.startswith(session_id):
                try:
                    return Session.from_dict(json.loads(path.read_text()))
                except Exception:  # noqa: BLE001 — an unreadable session file is treated as absent
                    return None
        return None

    def list(self) -> list[Session]:
        sessions: list[Session] = []
        import logging

        _log = logging.getLogger(__name__)
        for path in self._dir.glob("*.json"):
            try:
                sessions.append(Session.from_dict(json.loads(path.read_text())))
            except Exception:  # noqa: BLE001
                _log.debug("skipping malformed session file %s", path)
                continue
        return sorted(sessions, key=lambda s: s.updated_at, reverse=True)

    def continue_recent(self, cwd: str | None = None) -> Session:
        target = cwd or os.getcwd()
        for s in self.list():
            if s.cwd == target:
                return s
        return self.create(target)

    def save(self, session: Session) -> None:
        self._write(session)

    def touch(self, session: Session) -> None:
        session.message_count += 1
        session.updated_at = datetime.now(UTC)
        self._write(session)

    def _write(self, session: Session) -> None:
        path = self._dir / f"{session.id}.json"
        session.updated_at = datetime.now(UTC)
        path.write_text(json.dumps(session.to_dict(), indent=2))
