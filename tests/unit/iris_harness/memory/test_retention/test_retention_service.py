"""Retention: what ages out automatically, and what only the owner may delete.

The store this was written against held 3,730 turns across 1,029 sessions, 1,420 of
them from playground / eval / test runs that cross-session recall served back as "your
past conversations"; ChromaDB held 2,858 wiki vectors against 2,505 pages, because a
delete in SQLite never removed the vector.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_harness.memory import retention as retention_module
from iris_harness.memory.retention import RetentionService, is_ephemeral_session
from iris_harness.memory.store import MemoryStore, UserFact

# Fact keys such as `bank` and `credit_card` come from the test vocabulary fragment
# (tests/fixtures/test_vocabulary, installed by the `test_vocabulary` fixture), not
# from whichever domain plugin the tree happens to carry.
pytestmark = pytest.mark.usefixtures("test_vocabulary")


@pytest.fixture(autouse=True)
def _fresh_config() -> None:
    retention_module.reset_config_cache()


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    return s


class _FakeIndex:
    """Just enough of SemanticIndex to prove the two halves stay in step."""

    def __init__(self, ids: set[str] | None = None) -> None:
        self.ids = ids or set()
        self.dropped: list[int] = []

    def drop_turns(self, row_ids: list[int]) -> int:
        self.dropped.extend(row_ids)
        self.ids -= {str(i) for i in row_ids}
        return len(row_ids)

    def turn_ids(self) -> set[str]:
        return set(self.ids)

    def index_fact(self, fact: UserFact) -> None:
        pass

    def drop_fact(self, key: str) -> None:
        self.dropped.append(-1)


def _age_turns(store: MemoryStore, session_id: str, days: int) -> None:
    from iris_harness.foundation.persistence import sqlite_conn

    stamp = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    with sqlite_conn(store.db_path) as conn:
        conn.execute("UPDATE conversations SET ts = ? WHERE session_id = ?", (stamp, session_id))
        conn.commit()


class TestEphemeralSessions:
    @pytest.mark.parametrize(
        "sid",
        [
            "playground-brief-inj",
            "eval:cross_domain",
            "exp006-core_spine",
            "verify-fm-3",
            "smoke-1",
            # the 2026-09-17 manual probes that listed as the owner's chats
            "probe",
            "probe3-0",
            "proof-on",
            "proof-off-1",
            "wiki-switch-proof",
            # acceptance and mutation-check runs
            "accept-1",
            "accept4-qwen",
            "accept6-gpt4o",
            "mut",
            "mut3-kill",
        ],
    )
    def test_runs_are_not_memory(self, sid: str) -> None:
        assert is_ephemeral_session(sid) is True

    @pytest.mark.parametrize(
        "sid",
        ["default", "web-3dedfff1", "telegram:123456789", "", "proofread-my-essay", "web-mut1"],
    )
    def test_real_sessions_are(self, sid: str) -> None:
        assert is_ephemeral_session(sid) is False


class TestCooling:
    def test_an_old_session_loses_its_text_but_keeps_its_summary(self, store: MemoryStore) -> None:
        store.save_conversation_turns("old", [("user", "q"), ("assistant", "a")])
        store.save_conversation_summary("old", "Goal: the old session.")
        _age_turns(store, "old", days=200)
        index = _FakeIndex({"1", "2"})

        report = RetentionService(store, index).run()

        assert report.sessions_cooled == 1
        assert report.turns_deleted == 2
        assert report.vectors_deleted == 2
        assert store.fetch_turn_ids("old") == []
        assert store.load_conversation_summary("old") == "Goal: the old session."

    def test_an_old_session_with_no_summary_keeps_its_text(self, store: MemoryStore) -> None:
        """Cold means the summary stands in for the text; with no summary, nothing does.

        On the owner's box 253 sessions were past the window and 10 had summaries —
        cooling on age alone would have deleted the conversations outright.
        """
        store.save_conversation_turns("old", [("user", "q"), ("assistant", "a")])
        _age_turns(store, "old", days=200)
        index = _FakeIndex({"1", "2"})

        report = RetentionService(store, index).run()

        assert report.sessions_cooled == 0
        assert report.sessions_kept_unsummarized == 1
        assert len(store.fetch_turn_ids("old")) == 2
        assert index.dropped == []

    def test_a_recent_session_is_untouched(self, store: MemoryStore) -> None:
        store.save_conversation_turns("recent", [("user", "q"), ("assistant", "a")])

        report = RetentionService(store, _FakeIndex()).run()

        assert report.sessions_cooled == 0
        assert len(store.fetch_turn_ids("recent")) == 2

    def test_a_dry_run_deletes_nothing(self, store: MemoryStore) -> None:
        store.save_conversation_turns("old", [("user", "q"), ("assistant", "a")])
        store.save_conversation_summary("old", "Goal: the old session.")
        _age_turns(store, "old", days=200)
        index = _FakeIndex({"1", "2"})

        report = RetentionService(store, index).run(dry_run=True)

        assert report.turns_deleted == 2  # what WOULD go
        assert len(store.fetch_turn_ids("old")) == 2
        assert index.dropped == []


class TestOrphanSweep:
    def test_a_vector_whose_row_is_gone_is_dropped(self, store: MemoryStore) -> None:
        store.save_conversation_turns("live", [("user", "q"), ("assistant", "a")])
        index = _FakeIndex({"1", "2", "9999"})  # 9999 has no row

        report = RetentionService(store, index).run()

        assert report.orphan_vectors_swept == 1
        assert index.dropped == [9999]


class TestForget:
    def test_preview_finds_turns_summaries_and_facts(self, store: MemoryStore) -> None:
        store.save_conversation_turns(
            "s1", [("user", "Northwind statement please"), ("assistant", "ok")]
        )
        store.save_conversation_summary("s1", "Decisions: paid the Northwind card")
        now = datetime.now(UTC)
        store.upsert_user_fact(
            UserFact(
                key="bank",
                value="Northwind Bank",
                confidence=0.9,
                source="test",
                first_seen=now,
                last_confirmed=now,
                confirmed=True,
            )
        )

        preview = RetentionService(store, _FakeIndex()).preview_forget("Northwind")

        assert len(preview.turns) == 1
        assert len(preview.summaries) == 1
        assert preview.facts == [("bank", "Northwind Bank")]

    def test_nothing_is_deleted_without_confirm(self, store: MemoryStore) -> None:
        store.save_conversation_turns("s1", [("user", "Northwind statement"), ("assistant", "ok")])

        result = RetentionService(store, _FakeIndex()).forget_matching("Northwind")

        assert result["deleted"] is False
        assert len(store.fetch_turn_ids("s1")) == 2

    def test_confirm_deletes_rows_and_vectors_together(self, store: MemoryStore) -> None:
        store.save_conversation_turns("s1", [("user", "Northwind statement"), ("assistant", "ok")])
        store.save_conversation_summary("s1", "Decisions: paid the Northwind card")
        index = _FakeIndex({"1", "2"})

        result = RetentionService(store, index).forget_matching("Northwind", confirm=True)

        assert result["deleted"] is True
        assert result["turns"] == 1 and result["vectors"] == 1
        assert index.dropped == [1]
        assert store.load_conversation_summary("s1") == ""


class TestPurgeSessions:
    def test_preview_lists_only_the_test_runs(self, store: MemoryStore) -> None:
        store.save_conversation_turns("playground-brief-inj", [("user", "q"), ("assistant", "a")])
        store.save_conversation_turns("default", [("user", "q"), ("assistant", "a")])

        result = RetentionService(store, _FakeIndex()).purge_sessions()

        assert result["deleted"] is False
        assert [s["session_id"] for s in result["sessions"]] == ["playground-brief-inj"]
        assert result["total_turns"] == 2

    def test_confirm_removes_them_and_leaves_real_sessions(self, store: MemoryStore) -> None:
        store.save_conversation_turns("eval:x", [("user", "q"), ("assistant", "a")])
        store.save_conversation_turns("default", [("user", "q"), ("assistant", "a")])
        index = _FakeIndex({"1", "2", "3", "4"})

        result = RetentionService(store, index).purge_sessions(confirm=True)

        assert result["deleted"] is True and result["turns"] == 2
        assert store.fetch_turn_ids("eval:x") == []
        assert len(store.fetch_turn_ids("default")) == 2


class TestLogRotation:
    def _log(self, logs: Path, name: str, *, age_days: float, size: int = 1024) -> Path:
        path = logs / name
        path.write_bytes(b"x" * size)
        stamp = time.time() - age_days * 86400
        import os

        os.utime(path, (stamp, stamp))
        return path

    def test_old_logs_are_compressed_then_deleted(self, store: MemoryStore, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        fresh = self._log(logs, "fresh.jsonl", age_days=1)
        middle = self._log(logs, "middle.jsonl", age_days=10)
        ancient = self._log(logs, "ancient.jsonl", age_days=100)  # past service_logs_keep_days

        report = RetentionService(store, _FakeIndex(), logs_dir=logs).run()

        assert report.logs_compressed == 1 and report.logs_deleted == 1
        assert fresh.exists()
        assert not middle.exists() and middle.with_suffix(".jsonl.gz").exists()
        assert not ancient.exists()

    def test_the_size_cap_drops_the_oldest_first(
        self, store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        monkeypatch.setattr(
            retention_module,
            "_CONFIG_CACHE",
            {
                "conversations": {"hot_days": 90, "close_after_idle_days": 0},
                "ephemeral_session_prefixes": [],
                "logs": {"compress_after_days": 99, "delete_after_days": 99, "max_total_mb": 0},
                "housekeeping": {"enabled": True, "vacuum": False},
            },
        )
        self._log(logs, "a.jsonl", age_days=3, size=2048)
        self._log(logs, "b.jsonl", age_days=1, size=2048)

        report = RetentionService(store, _FakeIndex(), logs_dir=logs).run()

        assert report.logs_deleted == 2  # the cap is 0 MB, so everything goes, oldest first
        assert report.log_bytes_reclaimed >= 4096

    def test_a_dry_run_touches_no_file(self, store: MemoryStore, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        logs.mkdir()
        old = self._log(logs, "old.jsonl", age_days=60)

        RetentionService(store, _FakeIndex(), logs_dir=logs).run(dry_run=True)

        assert old.exists()


def test_history_records_each_pass(store: MemoryStore) -> None:
    service = RetentionService(store, _FakeIndex())
    service.run()
    service.run(dry_run=True)

    history = service.history()

    assert len(history) == 2
    assert history[0]["dry_run"] is True  # newest first


# ── the pass that never ran (2026-09-19) ─────────────────────────────────────
# iris_api.log had grown to 100 MB since July: live service logs never age, .pid
# files were treated as logs, session logs (the chat history) were compressed out of
# every reader's reach, and an interval:86400 schedule never fired on a stack that
# restarts more often than daily.


def _config(monkeypatch: pytest.MonkeyPatch, **logs: object) -> None:
    base = {"compress_after_days": 7, "delete_after_days": 30, "max_total_mb": 200}
    monkeypatch.setattr(
        retention_module,
        "_CONFIG_CACHE",
        {
            "conversations": {"hot_days": 90, "close_after_idle_days": 0},
            "ephemeral_session_prefixes": [],
            "logs": {**base, **logs},
            "housekeeping": {"enabled": True, "vacuum": False},
        },
    )


def _file(logs: Path, name: str, *, age_days: float = 0, data: bytes = b"x" * 1024) -> Path:
    import os

    path = logs / name
    path.write_bytes(data)
    stamp = time.time() - age_days * 86400
    os.utime(path, (stamp, stamp))
    return path


class TestLiveLogs:
    def test_a_big_live_log_is_archived_and_emptied_while_its_writer_keeps_going(
        self, store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import gzip

        _config(monkeypatch, rotate_over_mb=1)
        logs = tmp_path / "logs"
        logs.mkdir()
        live = _file(logs, "iris_api.log", data=b"old line\n" * 200_000)  # ~1.8 MB
        writer = live.open("ab")  # the way start_iris.sh's `>>` holds it

        report = RetentionService(store, _FakeIndex(), logs_dir=logs).run()
        writer.write(b"after rotation\n")
        writer.close()

        assert report.logs_rotated == 1 and report.logs_deleted == 0
        [archive] = list(logs.glob("iris_api.log.*.gz"))
        assert gzip.decompress(archive.read_bytes()).count(b"old line") == 200_000
        assert live.read_bytes() == b"after rotation\n"  # appended at the new end, no holes

    def test_a_small_live_log_is_left_alone(
        self, store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _config(monkeypatch, rotate_over_mb=1)
        logs = tmp_path / "logs"
        logs.mkdir()
        live = _file(logs, "governor.log")
        report = RetentionService(store, _FakeIndex(), logs_dir=logs).run()
        assert report.logs_rotated == 0 and live.read_bytes() == b"x" * 1024

    def test_the_size_cap_never_deletes_a_live_log(
        self, store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _config(monkeypatch, max_total_mb=0, rotate_over_mb=99)
        logs = tmp_path / "logs"
        logs.mkdir()
        live = _file(logs, "iris_api.log", age_days=0)
        archive = _file(logs, "iris_api.log.20260901T000000Z.gz", age_days=2)
        RetentionService(store, _FakeIndex(), logs_dir=logs).run()
        assert live.exists()  # unlinking a file a process writes frees nothing
        assert not archive.exists()


def test_pid_files_are_never_touched(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config(monkeypatch, max_total_mb=0)
    logs = tmp_path / "logs"
    logs.mkdir()
    week = _file(logs, "iris_api.pid", age_days=8, data=b"4242")
    month = _file(logs, "governor.pid", age_days=45, data=b"4243")
    RetentionService(store, _FakeIndex(), logs_dir=logs).run()
    assert week.read_bytes() == b"4242" and month.read_bytes() == b"4243"
    assert not list(logs.glob("*.pid.gz"))


def test_session_logs_stay_readable_until_they_expire(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """They are the chat history the Sessions view replays, read uncompressed."""
    _config(monkeypatch)
    logs = tmp_path / "logs"
    logs.mkdir()
    week_old = _file(logs, "session-web-abc.jsonl", age_days=10)
    expired = _file(logs, "session-web-old.jsonl", age_days=40)
    report = RetentionService(store, _FakeIndex(), logs_dir=logs).run()
    assert week_old.exists() and not list(logs.glob("session-web-abc.jsonl.gz"))
    assert not expired.exists()
    assert report.logs_compressed == 0 and report.logs_deleted == 1


class TestPersistedHistory:
    def test_a_restart_remembers_the_last_pass(self, store: MemoryStore, tmp_path: Path) -> None:
        path = tmp_path / "housekeeping_runs.jsonl"
        RetentionService(store, _FakeIndex(), history_path=path).run()
        RetentionService(store, _FakeIndex(), history_path=path).run(dry_run=True)

        fresh = RetentionService(store, _FakeIndex(), history_path=path)
        history = fresh.history()
        assert [h["dry_run"] for h in history] == [True, False]  # newest first
        last = fresh.last_run()
        assert last is not None and last["dry_run"] is False  # a dry run is not a pass

    def test_it_keeps_the_last_fifty(self, store: MemoryStore, tmp_path: Path) -> None:
        path = tmp_path / "housekeeping_runs.jsonl"
        service = RetentionService(store, _FakeIndex(), history_path=path)
        for _ in range(55):
            service.run(dry_run=True)
        assert len(path.read_text().splitlines()) == 50

    def test_never_ran_is_none(self, store: MemoryStore, tmp_path: Path) -> None:
        assert (
            RetentionService(store, _FakeIndex(), history_path=tmp_path / "h.jsonl").last_run()
            is None
        )


class TestDue:
    NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)

    def _last(self, hours_ago: float, backlog: int = 0) -> dict[str, object]:
        started = (self.NOW - timedelta(hours=hours_ago)).isoformat()
        return {"started_at": started, "dry_run": False, "sessions_close_backlog": backlog}

    def test_first_pass_is_due(self) -> None:
        assert retention_module.housekeeping_due(None, min_hours=23, now=self.NOW)[0] is True

    def test_a_recent_pass_is_not(self) -> None:
        due, why = retention_module.housekeeping_due(self._last(5), min_hours=23, now=self.NOW)
        assert due is False and "5.0h ago" in why

    def test_a_day_later_it_is(self) -> None:
        assert retention_module.housekeeping_due(self._last(23), min_hours=23, now=self.NOW)[0]

    def test_a_backlog_runs_again_at_the_next_check(self) -> None:
        due, why = retention_module.housekeeping_due(
            self._last(1, backlog=371), min_hours=23, now=self.NOW
        )
        assert due is True and "backlog" in why


class _CountingSessions:
    def __init__(self, fail: tuple[str, ...] = ()) -> None:
        self.closed: list[str] = []
        self._fail = set(fail)

    def close_session(self, session_id: str) -> dict[str, object]:
        self.closed.append(session_id)
        return {"closed": session_id not in self._fail}


def test_closing_summaries_are_capped_per_pass(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each is a model call; 391 waited on this box, so a pass writes at most N."""
    monkeypatch.setattr(
        retention_module,
        "_CONFIG_CACHE",
        {
            "conversations": {"hot_days": 90, "close_after_idle_days": 1, "close_max_per_pass": 2},
            "ephemeral_session_prefixes": [],
            "logs": {},
            "housekeeping": {"enabled": True, "vacuum": False},
        },
    )
    for sid in ("web-a", "web-b", "web-c", "web-d", "web-e"):
        store.save_conversation_turns(sid, [("user", "q"), ("assistant", "a")])
        _age_turns(store, sid, days=3)
    sessions = _CountingSessions()

    report = RetentionService(store, _FakeIndex(), sessions=sessions).run()

    assert len(sessions.closed) == 2
    assert (report.sessions_closed, report.sessions_close_backlog) == (2, 3)


# ── archive, don't delete (owner decision 2026-09-19) ────────────────────────


@pytest.fixture
def archive(tmp_path: Path):  # type: ignore[no-untyped-def]
    from cryptography.fernet import Fernet

    from iris_harness.memory.log_archive import LogArchive

    key = Fernet.generate_key()
    return LogArchive(tmp_path / "archive", key=lambda: key)


class TestArchiving:
    def test_an_old_real_session_log_moves_to_the_archive(
        self, store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive
    ) -> None:
        _config(monkeypatch)
        monkeypatch.setitem(
            retention_module._CONFIG_CACHE, "ephemeral_session_prefixes", ["playground"]
        )
        logs = tmp_path / "logs"
        logs.mkdir()
        real = _file(logs, "session-web-abc.jsonl", age_days=40, data=b'{"q": "hi"}\n')
        run = _file(logs, "session-playground-x.jsonl", age_days=40)
        recent = _file(logs, "session-web-new.jsonl", age_days=2)

        report = RetentionService(store, _FakeIndex(), logs_dir=logs, archive=archive).run()

        assert not real.exists() and not run.exists() and recent.exists()
        assert report.logs_archived == 1 and report.logs_deleted == 1  # the test run goes
        [month] = archive.months()
        assert month.members == ("session-web-abc.jsonl",)

    def test_nothing_leaves_the_live_dir_if_the_archive_fails(
        self, store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _config(monkeypatch)
        logs = tmp_path / "logs"
        logs.mkdir()
        real = _file(logs, "session-web-abc.jsonl", age_days=40)

        class Broken:
            def add(self, paths: object) -> int:
                raise OSError("disk full")

            def enforce_budget(self, max_bytes: int) -> int:
                return 0

        report = RetentionService(store, _FakeIndex(), logs_dir=logs, archive=Broken()).run()

        assert real.exists()
        assert report.logs_archived == 0 and "archive: disk full" in report.errors

    def test_over_the_cap_a_session_log_is_archived_not_deleted(
        self, store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive
    ) -> None:
        _config(monkeypatch, max_total_mb=0)
        logs = tmp_path / "logs"
        logs.mkdir()
        _file(logs, "session-web-abc.jsonl", age_days=3)
        report = RetentionService(store, _FakeIndex(), logs_dir=logs, archive=archive).run()
        assert report.logs_archived == 1 and report.logs_deleted == 0
        assert archive.months()[0].members == ("session-web-abc.jsonl",)

    def test_a_dry_run_archives_nothing(
        self, store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive
    ) -> None:
        _config(monkeypatch)
        logs = tmp_path / "logs"
        logs.mkdir()
        real = _file(logs, "session-web-abc.jsonl", age_days=40)
        report = RetentionService(store, _FakeIndex(), logs_dir=logs, archive=archive).run(
            dry_run=True
        )
        assert report.logs_archived == 1 and real.exists() and archive.months() == []

    def test_archive_disabled_in_config_falls_back_to_deleting(
        self, store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive
    ) -> None:
        _config(monkeypatch, archive={"enabled": False})
        logs = tmp_path / "logs"
        logs.mkdir()
        real = _file(logs, "session-web-abc.jsonl", age_days=40)
        report = RetentionService(store, _FakeIndex(), logs_dir=logs, archive=archive).run()
        assert not real.exists() and report.logs_deleted == 1 and archive.months() == []

    def test_service_logs_are_kept_ninety_days(
        self, store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive
    ) -> None:
        _config(monkeypatch, service_logs_keep_days=90)
        logs = tmp_path / "logs"
        logs.mkdir()
        sixty = _file(logs, "governor.log.20260701T000000Z.gz", age_days=60)
        hundred = _file(logs, "governor.log.20260601T000000Z.gz", age_days=100)
        RetentionService(store, _FakeIndex(), logs_dir=logs, archive=archive).run()
        assert sixty.exists() and not hundred.exists()
        assert archive.months() == []  # service logs are never archived

    def test_restore_and_list_through_the_service(
        self, store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive
    ) -> None:
        _config(monkeypatch)
        logs = tmp_path / "logs"
        logs.mkdir()
        _file(logs, "session-web-abc.jsonl", age_days=40, data=b"line\n")
        service = RetentionService(store, _FakeIndex(), logs_dir=logs, archive=archive)
        service.run()

        listed = service.archived_logs()
        assert listed["enabled"] is True and listed["months"][0]["files"] == 1
        assert listed["stored_bytes"] > 0
        assert service.restore_logs(session="web-abc") == ["session-web-abc.jsonl"]
        assert (logs / "session-web-abc.jsonl").read_bytes() == b"line\n"

    def test_no_archive_configured(self, store: MemoryStore) -> None:
        service = RetentionService(store, _FakeIndex())
        assert service.archived_logs() == {"enabled": False, "months": [], "stored_bytes": 0}
        with pytest.raises(RuntimeError):
            service.restore_logs(session="x")


def test_forget_reaches_live_and_archived_session_logs(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive
) -> None:
    """Logs hold full prompts: a forget that left them would not have forgotten."""
    _config(monkeypatch)
    logs = tmp_path / "logs"
    logs.mkdir()
    _file(logs, "session-web-old.jsonl", age_days=40, data=b'{"q": "Discover 4242"}\n{"q": "ok"}\n')
    live = _file(logs, "session-web-new.jsonl", data=b'{"q": "my discover card"}\n{"q": "hi"}\n')
    service = RetentionService(store, _FakeIndex(), logs_dir=logs, archive=archive)
    service.run()  # the old one moves to the archive

    preview = service.forget_matching("discover")["preview"]
    assert (preview["log_lines"], preview["archived_log_lines"]) == (1, 1)
    assert preview["total"] >= 2

    done = service.forget_matching("discover", confirm=True)

    assert (done["log_lines"], done["archived_log_lines"]) == (1, 1)
    assert live.read_bytes() == b'{"q": "hi"}\n'
    assert archive.count_matching("discover") == 0


def _idle(store: MemoryStore, *sids: str) -> None:
    for sid in sids:
        store.save_conversation_turns(sid, [("user", "q"), ("assistant", "a")])
        _age_turns(store, sid, days=3)


def _close_config(monkeypatch: pytest.MonkeyPatch, cap: int) -> None:
    monkeypatch.setattr(
        retention_module,
        "_CONFIG_CACHE",
        {
            "conversations": {
                "hot_days": 90,
                "close_after_idle_days": 1,
                "close_max_per_pass": cap,
            },
            "ephemeral_session_prefixes": [],
            "logs": {},
            "housekeeping": {"enabled": True, "vacuum": False},
        },
    )


def test_a_session_counts_as_closed_only_when_a_summary_was_written(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression (first live pass, 2026-09-19): '20 closed' in 1.8 s, 0 summaries —
    compact_now returned "nothing to compact" for short sessions and still counted."""
    _close_config(monkeypatch, cap=5)
    _idle(store, "web-a", "web-b")
    sessions = _CountingSessions(fail=("web-b",))
    service = RetentionService(
        store, _FakeIndex(), sessions=sessions, history_path=tmp_path / "h.jsonl"
    )

    report = service.run()

    assert report.sessions_closed == 1
    assert report.sessions_close_failed == ["web-b"]


def test_a_failed_session_is_skipped_by_the_next_pass_then_retried(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Otherwise the same unsummarizable sessions would take every slot every hour."""
    _close_config(monkeypatch, cap=1)
    _idle(store, "web-a", "web-b")
    sessions = _CountingSessions(fail=("web-a", "web-b"))
    service = RetentionService(
        store, _FakeIndex(), sessions=sessions, history_path=tmp_path / "h.jsonl"
    )

    service.run()
    service.run()
    service.run()

    first = sessions.closed[0]
    assert sessions.closed[1] != first  # the next pass moved on
    assert sessions.closed[2] == first  # and the one after retried it


def test_an_older_sessions_object_falls_back_to_compact_now(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _close_config(monkeypatch, cap=5)
    _idle(store, "web-a", "web-b")

    class Compacting:
        def compact_now(self, session_id: str) -> dict[str, object]:
            return {"compacted": session_id == "web-a"}

    report = RetentionService(store, _FakeIndex(), sessions=Compacting()).run()
    assert report.sessions_closed == 1 and report.sessions_close_failed == ["web-b"]
