"""The encrypted session-log archive (owner decision 2026-09-19: archive, don't delete)."""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path

import pytest
from cryptography.fernet import Fernet, InvalidToken

from iris_harness.memory import log_archive
from iris_harness.memory.log_archive import LogArchive

KEY = Fernet.generate_key()


def _archive(tmp_path: Path, key: bytes = KEY) -> LogArchive:
    return LogArchive(tmp_path / "archive", key=lambda: key)


def _log(dir_: Path, name: str, text: str, *, month_ts: float | None = None) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / name
    path.write_text(text, encoding="utf-8")
    if month_ts is not None:
        os.utime(path, (month_ts, month_ts))
    return path


JUL = time.mktime((2026, 7, 15, 12, 0, 0, 0, 0, -1))
AUG = time.mktime((2026, 8, 15, 12, 0, 0, 0, 0, -1))


def test_logs_are_stored_by_month_and_unreadable_without_the_key(tmp_path: Path) -> None:
    live = tmp_path / "logs"
    a = _log(live, "session-web-a.jsonl", '{"q": "my card ends 4242"}\n', month_ts=JUL)
    b = _log(live, "session-web-b.jsonl", '{"q": "hello"}\n', month_ts=AUG)
    archive = _archive(tmp_path)

    assert archive.add([a, b]) == 2

    files = sorted(p.name for p in (tmp_path / "archive").iterdir())
    assert files == ["2026-07.tar.gz.enc", "2026-08.tar.gz.enc"]
    blob = (tmp_path / "archive" / "2026-07.tar.gz.enc").read_bytes()
    assert b"4242" not in blob and b"session-web-a" not in blob  # names are hidden too
    with pytest.raises(InvalidToken):
        _archive(tmp_path, key=Fernet.generate_key()).months()
    assert [(m.month, m.members) for m in archive.months()] == [
        ("2026-07", ("session-web-a.jsonl",)),
        ("2026-08", ("session-web-b.jsonl",)),
    ]


def test_the_archive_is_private_and_written_atomically(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    archive.add([_log(tmp_path / "logs", "session-web-a.jsonl", "x\n", month_ts=JUL)])
    root = tmp_path / "archive"
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "2026-07.tar.gz.enc").stat().st_mode) == 0o600
    assert not list(root.glob(".*.tmp"))


def test_re_archiving_a_session_replaces_it(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    live = tmp_path / "logs"
    archive.add([_log(live, "session-web-a.jsonl", "old\n", month_ts=JUL)])
    archive.add([_log(live, "session-web-a.jsonl", "old\nnew\n", month_ts=JUL)])
    [month] = archive.months()
    assert month.files == 1
    archive.restore(tmp_path / "out", session="web-a")
    assert (tmp_path / "out" / "session-web-a.jsonl").read_text() == "old\nnew\n"


def test_restore_one_session_or_a_month_never_overwriting_a_live_file(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    src = tmp_path / "src"
    archive.add(
        [
            _log(src, "session-web-a.jsonl", "a\n", month_ts=JUL),
            _log(src, "session-web-b.jsonl", "b\n", month_ts=JUL),
            _log(src, "session-web-c.jsonl", "c\n", month_ts=AUG),
        ]
    )
    live = tmp_path / "logs"
    _log(live, "session-web-b.jsonl", "b, still live\n")

    one = archive.restore(live, session="web-c")
    month = archive.restore(live, month="2026-07")

    assert [p.name for p in one] == ["session-web-c.jsonl"]
    assert [p.name for p in month] == ["session-web-a.jsonl"]  # b was live: left alone
    assert (live / "session-web-b.jsonl").read_text() == "b, still live\n"
    assert time.time() - (live / "session-web-a.jsonl").stat().st_mtime < 60  # fresh again
    with pytest.raises(ValueError):
        archive.restore(live)
    with pytest.raises(ValueError):
        archive.restore(live, month="July")


def test_scrub_removes_the_lines_that_mention_it(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    archive.add(
        [
            _log(
                tmp_path / "src",
                "session-web-a.jsonl",
                '{"q": "Discover card ends 4242"}\n{"q": "weather"}\n',
                month_ts=JUL,
            )
        ]
    )
    assert archive.count_matching("DISCOVER") == 1
    assert archive.scrub("discover") == 1
    assert archive.count_matching("discover") == 0
    archive.restore(tmp_path / "out", session="web-a")
    assert (tmp_path / "out" / "session-web-a.jsonl").read_text() == '{"q": "weather"}\n'


def test_a_budget_drops_the_oldest_months_and_zero_keeps_everything(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    src = tmp_path / "src"
    archive.add([_log(src, "session-a.jsonl", os.urandom(4000).hex(), month_ts=JUL)])
    archive.add([_log(src, "session-b.jsonl", os.urandom(4000).hex(), month_ts=AUG)])
    assert archive.enforce_budget(0) == 0 and len(archive.months()) == 2
    newest = (tmp_path / "archive" / "2026-08.tar.gz.enc").stat().st_size
    assert archive.enforce_budget(newest) == 1
    assert [m.month for m in archive.months()] == ["2026-08"]


def test_the_key_is_made_once_in_the_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_harness.kernel.governance.vault import credentials

    store: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(credentials, "load_token", lambda p, a: store.get((p, a)))
    monkeypatch.setattr(credentials, "save_token", lambda p, a, v: store.__setitem__((p, a), v))

    first = log_archive.keychain_key()
    second = log_archive.keychain_key()

    assert first == second
    assert store == {("log-archive", "fernet"): first.decode()}
    Fernet(first)  # a valid key


def test_the_key_is_only_read_when_the_archive_is_used(tmp_path: Path) -> None:
    def no_keychain() -> bytes:
        raise AssertionError("constructing the archive must not touch the keychain")

    LogArchive(tmp_path / "archive", key=no_keychain)  # the runtime builds one at startup
