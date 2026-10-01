"""Cold storage for session logs: monthly, compressed, encrypted (owner decision 2026-09-19).

Session logs (``session-*.jsonl``) hold full prompts and are the chat history the
Sessions view replays. They used to be deleted after ``delete_after_days``; the owner
chose to keep them — but locked — instead:

- one file per month, ``<root>/YYYY-MM.tar.gz.enc``: a gzipped tar of that month's
  session logs, encrypted with Fernet (AES-128-CBC + HMAC-SHA256);
- the key lives in the OS keychain beside the OAuth tokens (``iris-log-archive``),
  made on first use. Lose the keychain entry and the archive is unreadable — the same
  footing the OAuth tokens are on;
- nothing is uploaded anywhere: off-machine copies are the owner's backup's job;
- ``restore`` puts a session (or a month) back into the live log directory;
- ``scrub`` is how ``forget`` reaches the archive: it rewrites the months that mention
  the needle, so a forgotten thing is gone from cold storage too.

A month is read, changed and written back whole (atomically). Months are small — the
whole live log directory's session logs were 6.8 MB when this was written.
"""

from __future__ import annotations

import io
import logging
import os
import re
import tarfile
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

KEY_PROVIDER = "log-archive"
KEY_ACCOUNT = "fernet"
SUFFIX = ".tar.gz.enc"
_MONTH = re.compile(r"^\d{4}-\d{2}$")


def keychain_key() -> bytes:
    """The archive key from the OS keychain, created (and stored) on first use."""
    from cryptography.fernet import Fernet

    from iris_harness.kernel.governance.vault import credentials

    stored = credentials.load_token(KEY_PROVIDER, KEY_ACCOUNT)
    if stored:
        return stored.encode()
    key = Fernet.generate_key()
    credentials.save_token(KEY_PROVIDER, KEY_ACCOUNT, key.decode())
    return key


def month_of(path: Path) -> str:
    """The archive month a log belongs to: its last-modified month, UTC."""
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime("%Y-%m")


@dataclass(frozen=True)
class ArchiveMonth:
    month: str
    files: int
    stored_bytes: int
    members: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "month": self.month,
            "files": self.files,
            "stored_bytes": self.stored_bytes,
            "members": list(self.members),
        }


class LogArchive:
    """Monthly encrypted archives of session logs under ``root``."""

    def __init__(self, root: Path, key: Callable[[], bytes] = keychain_key) -> None:
        self.root = Path(root)
        self._key_source = key
        self._key: bytes | None = None

    # -- crypto + container ------------------------------------------------

    def _fernet(self) -> Any:
        from cryptography.fernet import Fernet

        if self._key is None:
            self._key = self._key_source()
        return Fernet(self._key)

    def _path(self, month: str) -> Path:
        if not _MONTH.match(month):
            raise ValueError(f"month must be YYYY-MM, got {month!r}")
        return self.root / f"{month}{SUFFIX}"

    def _read(self, month: str) -> dict[str, bytes]:
        path = self._path(month)
        if not path.exists():
            return {}
        raw = self._fernet().decrypt(path.read_bytes())
        members: dict[str, bytes] = {}
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
            for info in tar.getmembers():
                if not info.isfile():
                    continue
                handle = tar.extractfile(info)
                if handle is not None:
                    members[Path(info.name).name] = handle.read()
        return members

    def _write(self, month: str, members: dict[str, bytes]) -> None:
        path = self._path(month)
        if not members:
            path.unlink(missing_ok=True)
            return
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for name in sorted(members):
                data = members[name]
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = int(datetime.now(UTC).timestamp())
                tar.addfile(info, io.BytesIO(data))
        token = self._fernet().encrypt(buffer.getvalue())
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=f".{month}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:  # mkstemp made it 0600
                fh.write(token)
            os.replace(tmp, path)  # a crash mid-write never leaves a half month
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # -- the operations ----------------------------------------------------

    def add(self, paths: Iterable[Path]) -> int:
        """Archive ``paths`` into their months (replacing same-named members).

        Returns how many were stored. The caller deletes the originals only after this
        returns — a failure raises and leaves every original in place.
        """
        by_month: dict[str, list[Path]] = {}
        for path in paths:
            by_month.setdefault(month_of(path), []).append(path)
        stored = 0
        for month, files in sorted(by_month.items()):
            members = self._read(month)
            for path in files:
                members[path.name] = path.read_bytes()
                stored += 1
            self._write(month, members)
        return stored

    def months(self) -> list[ArchiveMonth]:
        """Every archived month, oldest first."""
        out = []
        for path in sorted(self.root.glob(f"*{SUFFIX}")):
            month = path.name[: -len(SUFFIX)]
            if not _MONTH.match(month):
                continue
            members = self._read(month)
            out.append(
                ArchiveMonth(month, len(members), path.stat().st_size, tuple(sorted(members)))
            )
        return out

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.glob(f"*{SUFFIX}"))

    def restore(
        self, dest: Path, *, session: str | None = None, month: str | None = None
    ) -> list[Path]:
        """Put archived session logs back into ``dest`` (the live log directory).

        ``session`` restores that one session's log from whichever month holds it;
        ``month`` restores the whole month. An existing live file is never overwritten.
        Restored files get a fresh mtime, so they stay live for another
        ``delete_after_days`` before going back to the archive.
        """
        if (session is None) == (month is None):
            raise ValueError("give exactly one of session= or month=")
        wanted = f"session-{session}.jsonl" if session is not None else None
        months = [month] if month is not None else [m.month for m in self.months()]
        dest.mkdir(parents=True, exist_ok=True)
        restored: list[Path] = []
        for name_month in months:
            for name, data in self._read(name_month).items():
                if wanted is not None and name != wanted:
                    continue
                target = dest / name
                if target.exists():
                    continue
                target.write_bytes(data)
                restored.append(target)
        return restored

    def count_matching(self, needle: str) -> int:
        """Lines across the archive that mention ``needle`` (case-insensitive)."""
        lowered = needle.lower()
        return sum(
            _count_lines(data, lowered)
            for month in self.months()
            for data in self._read(month.month).values()
        )

    def scrub(self, needle: str) -> int:
        """Remove every line mentioning ``needle`` from the archive; lines removed."""
        lowered = needle.lower()
        removed = 0
        for month in self.months():
            members = self._read(month.month)
            changed = False
            for name, data in list(members.items()):
                kept, dropped = _drop_lines(data, lowered)
                if dropped:
                    members[name] = kept
                    removed += dropped
                    changed = True
            if changed:
                self._write(month.month, members)
        return removed

    def enforce_budget(self, max_bytes: int) -> int:
        """Drop the oldest months until the archive fits; 0 means no budget. Months dropped."""
        if max_bytes <= 0:
            return 0
        dropped = 0
        months = sorted(self.root.glob(f"*{SUFFIX}"))
        total = sum(p.stat().st_size for p in months)
        for path in months:
            if total <= max_bytes:
                break
            total -= path.stat().st_size
            path.unlink(missing_ok=True)
            dropped += 1
        return dropped


def _count_lines(data: bytes, lowered: str) -> int:
    return sum(
        1 for line in data.splitlines() if lowered in line.decode("utf-8", "replace").lower()
    )


def _drop_lines(data: bytes, lowered: str) -> tuple[bytes, int]:
    kept: list[bytes] = []
    dropped = 0
    for line in data.splitlines(keepends=True):
        if lowered in line.decode("utf-8", "replace").lower():
            dropped += 1
        else:
            kept.append(line)
    return b"".join(kept), dropped


def scrub_file(path: Path, needle: str) -> int:
    """Remove lines mentioning ``needle`` from one live log file; lines removed."""
    kept, dropped = _drop_lines(path.read_bytes(), needle.lower())
    if dropped:
        path.write_bytes(kept)
    return dropped


def count_in_file(path: Path, needle: str) -> int:
    return _count_lines(path.read_bytes(), needle.lower())


__all__ = [
    "ArchiveMonth",
    "LogArchive",
    "count_in_file",
    "keychain_key",
    "month_of",
    "scrub_file",
]
