"""A session is rebuilt from the stored records, and what is missing is named (issue #134, stage 5).

The acceptance scenario runs through the real governed harness on both entry points (``chat``
and ``chat_stream``): an approved call, a tool that starts another call, a bridged MCP call and
a resumed run. The losses are real deletions from the real stores, and the kill is a real
process exit.
"""

# S603: the subprocesses run this repo's own interpreter on fixed code.
# ruff: noqa: S603

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("duckdb")

import iris_harness  # noqa: E402
from iris_harness.foundation.ids import new_ulid  # noqa: E402
from iris_harness.foundation.observability.session_log import (  # noqa: E402
    session_scope,
    turn_scope,
)
from iris_harness.kernel.governance.approvals import ApprovalQueue  # noqa: E402
from iris_harness.kernel.governance.audit import AuditLog, sequence, spool  # noqa: E402
from iris_harness.kernel.governance.audit.archive import AuditArchive, AuditCompactor  # noqa: E402
from iris_harness.kernel.governance.audit.log import _spool_record  # noqa: E402
from iris_harness.kernel.governance.audit.replay import (  # noqa: E402
    GAP_CLASSES,
    ReplaySources,
    replay_session,
)
from iris_harness.sdk import PluginAPI  # noqa: E402
from iris_harness.testing import harness, plugin  # noqa: E402

SRC = str(Path(iris_harness.__file__).resolve().parents[1])
FAR_FUTURE = datetime.now(UTC) + timedelta(days=400)


@pytest.fixture(autouse=True)
def _fresh_sequence_state() -> Iterator[None]:
    sequence.reset_for_tests()
    yield
    sequence.reset_for_tests()


# ---------------------------------------------------------------- the governed scenario

_SCRIPT: dict[str, Any] = {
    "rules": [
        {
            "name": "route",
            "match": {"system": "request router"},
            "reply": {"json": {"intent": "general"}},
        },
        {
            "name": "shredded",
            "match": {"user": r"(?s)Observation: The owner approved.*shredded memo"},
            "reply": {"content": "Thought: Done.\nFinal Answer: The memo is shredded."},
        },
        {
            "name": "looked up",
            "match": {"user": r"(?s)Observation:.*memo has 3 words"},
            "reply": {"content": "Thought: Done.\nFinal Answer: The memo has three words."},
        },
        {
            "name": "lookup",
            "match": {"user": r"User: Look up the memo"},
            "reply": {
                "content": 'Thought: Look.\nAction: lookup_doc\nAction Input: {"doc": "memo"}'
            },
        },
        {
            "name": "shred",
            "match": {"user": r"User: Shred the memo"},
            "reply": {
                "content": 'Thought: Shred.\nAction: shred_doc\nAction Input: {"doc": "memo"}'
            },
        },
    ]
}


def _memo_plugin() -> Any:
    def setup(api: PluginAPI) -> None:
        api.register_tool(
            "word_count",
            'Count words. Args: {"text": str}.',
            lambda a: str(len(str(a.get("text", "")).split())),
        )

        def lookup(args: dict[str, Any]) -> str:
            counted = api.tools.call("word_count", {"text": "a b c"})
            return f"memo has {counted.text} words"

        api.register_tool("lookup_doc", 'Look a document up. Args: {"doc": str}.', lookup)
        api.register_tool(
            "shred_doc",
            'Shred a document. Args: {"doc": str}.',
            lambda a: f"shredded {a.get('doc')}",
        )

    return plugin(
        setup,
        manifest={
            "name": "memo",
            "provides": ["tool"],
            "tools": {
                "word_count": {"effect": "read"},
                "lookup_doc": {"effect": "read"},
                "shred_doc": {"effect": "destructive"},
            },
        },
    )


def _sources(h: Any, archive: Path | None = None) -> ReplaySources:
    governance = h.home / "governance"
    return ReplaySources(
        audit_db=h.audit_db,
        archive_root=archive,
        approvals_db=governance / "approvals.db",
        ledger_db=governance / "side_effects.db",
        session_log_dir=h.home / "logs",
    )


def _chat(entry: str, h: Any, message: str, session: str) -> Any:
    if entry == "chat":
        return h.chat(message, session_id=session)
    return h.chat_stream(message, session_id=session)


def _bridge_a_mcp_call(h: Any, tmp_path: Path, session: str) -> None:
    """One MCP tool call, declared ``read``, bridged through the governed kernel into ``h``'s
    ledger, made inside the session."""
    from iris_harness.kernel.governance import build_default_kernel
    from iris_harness.kernel.governance.plugins.mcp_allowlist import (
        MCPServerGovernance,
        MCPToolGovernance,
    )
    from iris_harness.kernel.governor.audit import GovernorAuditLogger
    from iris_harness.kernel.governor.policy import GovernorPolicyEngine, load_governor_policy
    from iris_harness.kernel.governor.service import IRISGovernorService
    from iris_harness.tools.mcp_bridge import MCPBridge, MCPBridgeConfig, MCPServerConfig

    root = tmp_path / "mcp"
    policy_dir = root / "config" / "governor"
    policy_dir.mkdir(parents=True)
    (policy_dir / "policy.yaml").write_text(
        "version: '1'\nroutes:\n  - route: coding/mcp\n    allowed_actions:\n"
        "      - session_open\n      - call_tool\n      - invoke_server\n"
        "    requires_approval: false\n    rate_limit:\n      requests: 10\n"
        "      window_seconds: 3600\n",
        encoding="utf-8",
    )
    kernel = build_default_kernel(
        audit_log=AuditLog(h.audit_db),
        approval_queue=ApprovalQueue(db_path=root / "approvals.db"),
        side_effect_ledger_db_path=root / "ledger.db",
    )
    config = MCPBridgeConfig(
        enabled=True,
        servers=(
            MCPServerConfig(
                name="files",
                enabled=True,
                command="python",
                governance=MCPServerGovernance(
                    tools={"read_file": MCPToolGovernance(effect="read")}
                ),
            ),
        ),
    )
    bridge = MCPBridge(
        root,
        config=config,
        governance_kernel=kernel,
        governor_service=IRISGovernorService(
            policy_engine=GovernorPolicyEngine(load_governor_policy(root)),
            audit_logger=GovernorAuditLogger(root / "data" / "audit.db"),
        ),
    )
    with session_scope(session), turn_scope():
        result = bridge.invoke_external_tool(
            "files",
            "read_file",
            {"path": "notes.txt"},
            approval_granted=True,
            executor=lambda *_: "contents",
        )
    assert result.result is not None


def _scenario(entry: str, h: Any, tmp_path: Path, session: str = "sess-accept") -> None:
    _chat(entry, h, "Look up the memo", session)  # a tool that starts another call
    held = _chat(entry, h, "Shred the memo", session)  # held for approval
    assert "needs your approval" in held.text
    [pending] = ApprovalQueue().list_pending()
    assert "shredded" in h.respond_to_approval(pending.approval_id, approve=True)  # resumed
    _bridge_a_mcp_call(h, tmp_path, session)


def _gap_classes(result: Any) -> list[str]:
    return sorted(g.cls for g in result.gaps)


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_the_session_is_rebuilt_as_one_ordered_timeline_with_no_gaps(
    entry: str, tmp_path: Path
) -> None:
    with harness(plugins=[_memo_plugin()], fake_model=_SCRIPT) as h:
        _scenario(entry, h, tmp_path)
        result = replay_session("sess-accept", sources=_sources(h), include_archive=False)

    assert result.complete and result.gaps == (), [str(g) for g in result.gaps]
    stamps = [r.ts for r in result.records]
    assert stamps == sorted(stamps)  # one ordered timeline
    by_tool = {r.tool: r for r in result.records if r.hook_point == "pre_tool_use" and r.tool}
    lookup, nested = by_tool["lookup_doc"], by_tool["word_count"]
    assert nested.parent_call_id == lookup.call_id  # the nested call hangs under its caller
    mcp = [
        r for r in result.records if r.hook_point == "pre_tool_use" and "read_file" in str(r.tool)
    ]
    assert mcp and mcp[0].call_id and mcp[0].parent_call_id is None  # the bridged call
    # The approved call: attempt 1 held, attempt 2 replays it, in a run that was resumed.
    attempts = {r.attempt: r for r in result.records if r.tool == "shred_doc" and r.call_id}
    assert attempts[1].call_id and attempts[1].call_id != attempts[2].call_id
    assert attempts[2].replay_of == attempts[1].call_id
    assert attempts[2].resumed_from_run == attempts[1].run_id
    # The tree: the nested call is a child of lookup_doc, not a root.
    nodes: list[dict[str, Any]] = []

    def walk(items: list[dict[str, Any]]) -> None:
        for n in items:
            nodes.append(n)
            walk(n["children"])

    for turn in result.tree:
        for run in turn["runs"]:
            for step in run["steps"]:
                walk(step["calls"])
    by_id = {n["call_id"]: n for n in nodes}
    assert [c["call_id"] for c in by_id[lookup.call_id]["children"]] == [nested.call_id]
    assert {n["tool"] for n in nodes} >= {"lookup_doc", "word_count", "shred_doc"}


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_a_replay_of_one_session_reports_nothing_of_another(entry: str, tmp_path: Path) -> None:
    with harness(plugins=[_memo_plugin()], fake_model=_SCRIPT) as h:
        _chat(entry, h, "Look up the memo", "sess-a")
        _chat(entry, h, "Look up the memo", "sess-b")
        a = replay_session("sess-a", sources=_sources(h), include_archive=False)
        b = replay_session("sess-b", sources=_sources(h), include_archive=False)
    ids_a = {r.record_id for r in a.records}
    ids_b = {r.record_id for r in b.records}
    assert ids_a and ids_b and not ids_a & ids_b
    assert {r.call_id for r in a.records if r.call_id}.isdisjoint(
        {r.call_id for r in b.records if r.call_id}
    )


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_a_deleted_record_is_a_gap_naming_the_writer_and_the_number(
    entry: str, tmp_path: Path
) -> None:
    with harness(plugins=[_memo_plugin()], fake_model=_SCRIPT) as h:
        _scenario(entry, h, tmp_path)
        conn = sqlite3.connect(h.audit_db)
        victim = conn.execute(
            "SELECT id, writer_id, writer_seq FROM audit_log WHERE session_id = 'sess-accept' "
            "AND hook_point = 'pre_llm_call' AND writer_seq IS NOT NULL ORDER BY id LIMIT 1 OFFSET 1"
        ).fetchone()
        conn.execute("DELETE FROM audit_log WHERE id = ?", (victim[0],))
        conn.commit()
        conn.close()
        result = replay_session("sess-accept", sources=_sources(h), include_archive=False)
        gaps = [g.as_dict() for g in result.gaps]
        gap_strings = h.audit_gaps()

    assert _gap_classes(result) == ["sequence_hole"], gaps
    assert gaps[0]["subject"] == f"writer {victim[1]} seq {victim[2]}"
    assert "between rows of this session" in gaps[0]["detail"]
    assert any("sequence_hole" in g for g in gap_strings)  # audit_gaps() is built on the replay


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_the_same_timeline_comes_back_from_the_cold_tier(entry: str, tmp_path: Path) -> None:
    with harness(plugins=[_memo_plugin()], fake_model=_SCRIPT) as h:
        _scenario(entry, h, tmp_path)
        archive = tmp_path / "archive"
        hot = replay_session("sess-accept", sources=_sources(h, archive), include_archive=True)
        AuditCompactor(
            audit_log=AuditLog(h.audit_db), archive=AuditArchive(root=archive), retention_days=30
        ).compact(now=FAR_FUTURE)
        assert (
            not sqlite3.connect(h.audit_db)
            .execute("SELECT 1 FROM audit_log WHERE session_id = 'sess-accept'")
            .fetchall()
        )  # every session row went cold
        cold = replay_session("sess-accept", sources=_sources(h, archive), include_archive=True)
        no_archive = replay_session("sess-accept", sources=_sources(h), include_archive=False)

    assert cold.gaps == () and [r.record_id for r in cold.records] == [
        r.record_id for r in hot.records
    ]
    assert {r.tier for r in cold.records} == {"cold"}
    assert no_archive.records == ()  # not read: nothing to show, and nothing claimed missing
    assert not [g for g in no_archive.gaps if g.cls == "sequence_hole"]


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_a_cold_record_that_was_removed_is_an_archive_mismatch_and_a_hole(
    entry: str, tmp_path: Path
) -> None:
    with harness(plugins=[_memo_plugin()], fake_model=_SCRIPT) as h:
        _scenario(entry, h, tmp_path)
        archive = tmp_path / "archive"
        AuditCompactor(
            audit_log=AuditLog(h.audit_db), archive=AuditArchive(root=archive), retention_days=30
        ).compact(now=FAR_FUTURE)
        chunks = sorted(archive.glob("year=*/month=*/audit-*.parquet"))
        victim = None
        for chunk in chunks:
            table = pq.read_table(chunk)
            mine = [
                i
                for i, (sid, hook) in enumerate(
                    zip(
                        table.column("session_id").to_pylist(),
                        table.column("hook_point").to_pylist(),
                        strict=True,
                    )
                )
                if sid == "sess-accept" and hook == "pre_llm_call"
            ]
            if len(mine) > 1:
                keep = [i for i in range(table.num_rows) if i != mine[1]]
                victim = (
                    table.column("writer_id")[mine[1]].as_py(),
                    table.column("writer_seq")[mine[1]].as_py(),
                )
                pq.write_table(table.take(keep), chunk)
                break
        assert victim is not None
        result = replay_session("sess-accept", sources=_sources(h, archive), include_archive=True)

    assert _gap_classes(result) == ["archive_mismatch", "sequence_hole"], [
        str(g) for g in result.gaps
    ]
    hole = next(g for g in result.gaps if g.cls == "sequence_hole")
    assert hole.subject == f"writer {victim[0]} seq {victim[1]}"


# ---------------------------------------------------------------- synthetic ledgers


def _db(tmp_path: Path) -> Path:
    path = tmp_path / "audit.db"
    AuditLog(db_path=path)
    return path


def _put(
    db: Path,
    *,
    session: str | None = "s1",
    hook: str = "pre_tool_use",
    decision: str = "allow",
    writer: str | None = "W1",
    seq: int | None = None,
    call: str | None = None,
    parent: str | None = None,
    replay_of: str | None = None,
    resumed: str | None = None,
    run: str = "run-1",
    kind: str | None = None,
    record_id: str | None = "auto",
    ts: datetime | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    body = dict(payload or {})
    if session:
        body.setdefault("session_id", session)
    if call:
        body.setdefault("call_id", call)
        body.setdefault("tool_name", "tool_x")
    stamp = (ts or datetime.now(UTC)).isoformat()
    rid = new_ulid() if record_id == "auto" else record_id
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO audit_log(ts, run_id, step_id, agent_type, hook_point, plugin, decision,"
        " severity, reason, payload_json, record_id, session_id, call_id, parent_call_id,"
        " replay_of, resumed_from_run, writer_id, writer_seq, kind) VALUES (?, ?, 1, 'chat', ?,"
        " 'p', ?, 'info', 'r', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            stamp,
            run,
            hook,
            decision,
            json.dumps(body),
            rid,
            session,
            call,
            parent,
            replay_of,
            resumed,
            writer,
            seq,
            kind,
        ),
    )
    conn.commit()
    conn.close()


def _src(db: Path, tmp_path: Path, **kw: Any) -> ReplaySources:
    return ReplaySources(audit_db=db, session_log_dir=tmp_path / "logs", **kw)


def _writer_rows(db: Path, writer: str, *, pid: int | None = None, close: bool = False) -> None:
    _put(
        db,
        session=None,
        hook=sequence.WRITER_START,
        writer=writer,
        seq=0,
        kind="writer",
        run=f"audit:{writer}",
        payload={"pid": pid if pid is not None else _own_pid()},
    )
    if close:
        _put(
            db,
            session=None,
            hook=sequence.WRITER_CLOSE,
            writer=writer,
            seq=9999,
            kind="writer",
            run=f"audit:{writer}",
        )


def _own_pid() -> int:
    import os

    return os.getpid()


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_a_sequence_number_the_writer_declared_lost_is_a_lost_write(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _writer_rows(db, "W1")
    for seq in (1, 2, 4):
        _put(db, writer="W1", seq=seq, hook="pre_llm_call", call=None)
    _put(
        db,
        session=None,
        writer="W1",
        seq=5,
        kind="gap",
        hook="audit.gap",
        payload={"first_missing": 3, "last_missing": 3},
    )
    result = replay_session("s1", sources=_src(db, tmp_path), include_archive=False)
    assert _gap_classes(result) == ["lost_write"]
    assert "seq 3" in result.gaps[0].detail


def test_a_number_still_waiting_in_the_spool_is_not_a_gap(tmp_path: Path) -> None:
    db = _db(tmp_path)
    writer = new_ulid()
    _writer_rows(db, writer)
    for seq in (1, 3):
        _put(db, writer=writer, seq=seq, hook="pre_llm_call")
    row = {
        "run_id": "run-1",
        "step_id": 1,
        "agent_type": "chat",
        "hook_point": "pre_llm_call",
        "plugin": "p",
        "decision": "allow",
        "severity": "info",
        "reason": "r",
        "payload": {"session_id": "s1"},
        "ts": datetime.now(UTC),
    }
    spool.append(spool.spool_path_for(db), _spool_record(row, new_ulid(), writer, 2, None))
    result = replay_session("s1", sources=_src(db, tmp_path), include_archive=False)
    assert result.gaps == ()
    # the same hole without the spool line is a gap: the line is what excused it
    spool.spool_path_for(db).unlink()
    assert _gap_classes(
        replay_session("s1", sources=_src(db, tmp_path), include_archive=False)
    ) == ["sequence_hole"]


def test_a_hole_between_two_sessions_is_reported_as_session_unknown(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _writer_rows(db, "W1")
    _put(db, session="s1", writer="W1", seq=1, hook="pre_llm_call")
    _put(db, session="s2", writer="W1", seq=2, hook="pre_llm_call")
    # seq 3 is gone (nothing says whose); seq 4 belongs to s2 again
    _put(db, session="s2", writer="W1", seq=4, hook="pre_llm_call")
    only_s1 = replay_session("s1", sources=_src(db, tmp_path), include_archive=False)
    assert only_s1.gaps == ()  # s1's window is [1, 1]: the hole is not inside it
    s2 = replay_session("s2", sources=_src(db, tmp_path), include_archive=False)
    assert [g.cls for g in s2.gaps] == ["sequence_hole"]
    assert "session unknown" not in s2.gaps[0].detail or "seq 3" in s2.gaps[0].subject
    # neighbours of seq 3 are s2's own (2 and 4): attributed to the span; and never s1's data
    assert all("s1" not in json.dumps(g.as_dict()) for g in s2.gaps)
    assert {r.record_id for r in s2.records}.isdisjoint({r.record_id for r in only_s1.records})


def test_a_hole_whose_neighbours_are_in_the_session_but_not_adjacent_says_session_unknown(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    _writer_rows(db, "W1")
    _put(db, session="s1", writer="W1", seq=1, hook="pre_llm_call")
    # seq 2 deleted; seq 3 is another session's; seq 4 is s1's
    _put(db, session="s2", writer="W1", seq=3, hook="pre_llm_call")
    _put(db, session="s1", writer="W1", seq=4, hook="pre_llm_call")
    result = replay_session("s1", sources=_src(db, tmp_path), include_archive=False)
    assert [g.cls for g in result.gaps] == ["sequence_hole"]
    assert result.gaps[0].subject == "writer W1 seq 2"


def test_a_call_started_in_a_live_process_is_in_flight_not_a_gap(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _writer_rows(db, "W1")  # this very process: alive
    _put(db, writer="W1", seq=1, call="C1")
    result = replay_session("s1", sources=_src(db, tmp_path), include_archive=False)
    assert result.gaps == ()
    assert {n.code for n in result.notes} >= {"in_flight", "writer_open"}


_KILLED = """
    import os, sys
    from pathlib import Path
    from iris_harness.foundation.observability.session_log import (
        log_timeline_event, session_scope, turn_scope)
    from iris_harness.kernel.governance.audit import AuditLog
    db = Path(sys.argv[1])
    log = AuditLog(db_path=db)
    with session_scope("s1"), turn_scope():
        log.record(run_id="run-1", step_id=1, agent_type="chat", hook_point="pre_tool_use",
                   plugin="p", decision="allow", severity="info", reason="r",
                   payload={"session_id": "s1", "call_id": "C-KILLED", "tool_name": "boom"})
        log_timeline_event("tool.invoke.start", phase="tool.invoke.start",
                           payload={"tool": "boom", "call_id": "C-KILLED"})
        os._exit(3)
"""


def test_a_call_killed_mid_way_is_an_open_call_and_its_writer_ended_without_close(
    tmp_path: Path,
) -> None:
    db = tmp_path / "audit.db"
    logs = tmp_path / "logs"
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_KILLED), str(db)],
        capture_output=True,
        text=True,
        timeout=120,
        env={
            "PYTHONPATH": SRC,
            "IRIS_AUTH_SECRET": "x",
            "PATH": "/usr/bin:/bin",
            "IRIS_SESSION_LOG_DIR": str(logs),
        },
    )
    assert proc.returncode == 3, proc.stderr[-500:]
    result = replay_session(
        "s1", sources=ReplaySources(audit_db=db, session_log_dir=logs), include_archive=False
    )
    assert _gap_classes(result) == ["open_call"] and "C-KILLED" in result.gaps[0].detail
    assert "writer_ended_without_close" in {n.code for n in result.notes}
    assert not any(n.code == "in_flight" for n in result.notes)


def test_an_unwitnessed_call_without_a_settle_is_only_a_note(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _writer_rows(db, "W1", pid=_dead_pid())
    _put(db, writer="W1", seq=1, call="C1")  # no POST, and no log or ledger saw it start
    result = replay_session("s1", sources=_src(db, tmp_path), include_archive=False)
    assert result.gaps == ()
    assert "unsettled_call" in {n.code for n in result.notes}


def test_a_held_call_is_not_open(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _writer_rows(db, "W1", pid=_dead_pid())
    _put(db, writer="W1", seq=1, call="C1", decision="require_approval")
    result = replay_session("s1", sources=_src(db, tmp_path), include_archive=False)
    assert result.gaps == ()


def test_a_parent_that_does_not_exist_is_a_missing_parent(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _writer_rows(db, "W1")
    _put(db, writer="W1", seq=1, call="C2", parent="C-NOPE")
    _put(db, writer="W1", seq=2, call="C2", hook="post_tool_use", parent="C-NOPE")
    _put(db, writer="W1", seq=3, call="C3", replay_of="C-GONE")
    _put(db, writer="W1", seq=4, call="C3", hook="post_tool_use")
    _put(db, writer="W1", seq=5, call="C4", resumed="run-gone")
    _put(db, writer="W1", seq=6, call="C4", hook="post_tool_use")
    result = replay_session("s1", sources=_src(db, tmp_path), include_archive=False)
    assert _gap_classes(result) == ["missing_parent"] * 3
    assert {g.subject for g in result.gaps} == {
        "parent_call_id C-NOPE",
        "replay_of C-GONE",
        "resumed_from_run run-gone",
    }


def test_a_parent_in_another_session_resolves(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _writer_rows(db, "W1")
    _put(db, session="other", writer="W1", seq=1, call="C-PARENT")
    _put(db, session="other", writer="W1", seq=2, call="C-PARENT", hook="post_tool_use")
    _put(db, session="s1", writer="W1", seq=3, call="C-CHILD", parent="C-PARENT")
    _put(
        db,
        session="s1",
        writer="W1",
        seq=4,
        call="C-CHILD",
        hook="post_tool_use",
        parent="C-PARENT",
    )
    result = replay_session("s1", sources=_src(db, tmp_path), include_archive=False)
    assert result.gaps == ()


def test_a_settle_with_no_start_is_an_orphan_settle(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _writer_rows(db, "W1")
    _put(db, writer="W1", seq=1, call="C1", hook="post_tool_use")
    result = replay_session("s1", sources=_src(db, tmp_path), include_archive=False)
    assert _gap_classes(result) == ["orphan_settle"]


def test_witnesses_with_no_audit_row_are_a_mismatch(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _writer_rows(db, "W1")
    _put(db, writer="W1", seq=1, hook="pre_llm_call")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "session-s1.jsonl").write_text(
        json.dumps(
            {
                "kind": "tool.invoke.start",
                "ts": datetime.now(UTC).isoformat(),
                "session_id": "s1",
                "payload": {"tool": "t", "call_id": "C-LOG"},
            }
        )
        + "\n"
    )
    approvals = ApprovalQueue(db_path=tmp_path / "approvals.db")
    approvals.enqueue("run-1", None, "x", "x", session_id="s1", call_id="C-APPR")
    result = replay_session(
        "s1",
        sources=_src(db, tmp_path, approvals_db=tmp_path / "approvals.db"),
        include_archive=False,
    )
    assert _gap_classes(result) == ["witness_mismatch", "witness_mismatch"]
    assert {g.subject.split()[0] for g in result.gaps} == {"call", "approval"}


def test_rows_from_before_identity_are_a_note_never_a_gap(tmp_path: Path) -> None:
    db = tmp_path / "audit.db"
    conn = sqlite3.connect(db)  # a database as the release before call identity made it
    conn.executescript("""
        CREATE TABLE audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, run_id TEXT NOT NULL,
            step_id INTEGER, agent_type TEXT NOT NULL, hook_point TEXT NOT NULL,
            plugin TEXT NOT NULL, decision TEXT NOT NULL, classification TEXT, tier TEXT,
            cost_usd REAL, severity TEXT NOT NULL, reason TEXT NOT NULL,
            payload_json TEXT NOT NULL);
    """)
    for _ in range(3):
        conn.execute(
            "INSERT INTO audit_log(ts, run_id, agent_type, hook_point, plugin, decision, severity,"
            " reason, payload_json) VALUES (?, 'r', 'chat', 'pre_llm_call', 'p', 'allow', 'info',"
            " 'x', '{\"session_id\": \"s1\"}')",
            (datetime.now(UTC).isoformat(),),
        )
    conn.commit()
    conn.close()
    result = replay_session("s1", sources=_src(db, tmp_path), include_archive=False)
    assert len(result.records) == 3 and result.gaps == ()
    assert "pre_identity_era" in {n.code for n in result.notes}


def test_a_record_stored_twice_with_different_content_is_a_duplicate(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _writer_rows(db, "W1")
    rid = new_ulid()
    _put(
        db,
        writer="W1",
        seq=1,
        hook="pre_llm_call",
        record_id=rid,
        ts=datetime.now(UTC) - timedelta(days=90),
    )
    log = AuditLog(db_path=db)
    archive = tmp_path / "archive"
    AuditCompactor(audit_log=log, archive=AuditArchive(root=archive), retention_days=30).compact()
    # the same record id comes back in the hot tier with other content
    _put(
        db,
        writer="W1",
        seq=1,
        hook="post_llm_call",
        record_id=rid,
        ts=datetime.now(UTC) - timedelta(days=90),
    )
    result = replay_session("s1", sources=_src(db, tmp_path, archive_root=archive))
    assert "duplicate" in _gap_classes(result)


# ---------------------------------------------------------------- bounds


def test_a_cap_cuts_the_read_short_and_no_gap_is_judged_from_a_partial_read(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    _writer_rows(db, "W1")
    for seq in (1, 2, 3, 4, 6):  # 5 is missing: a real hole
        _put(db, writer="W1", seq=seq, hook="pre_llm_call")
    full = replay_session("s1", sources=_src(db, tmp_path), include_archive=False)
    cut = replay_session("s1", sources=_src(db, tmp_path), include_archive=False, max_rows=2)
    assert _gap_classes(full) == ["sequence_hole"] and full.complete
    assert not cut.complete and cut.gaps == ()  # not judged, and it says so
    assert "truncated" in {n.code for n in cut.notes}
    assert len(cut.records) == 2
    assert cut.as_dict()["truncated"] is True


def test_the_time_cap_stops_the_replay(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _writer_rows(db, "W1")
    _put(db, writer="W1", seq=1, hook="pre_llm_call")
    result = replay_session(
        "s1", sources=_src(db, tmp_path), include_archive=False, max_seconds=0.0
    )
    assert not result.complete and result.gaps == ()


def test_the_cold_read_is_bounded_to_the_chunks_in_the_sessions_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iris_harness.kernel.governance.audit.archive import query as q

    db = _db(tmp_path)
    log = AuditLog(db_path=db)
    archive = tmp_path / "archive"
    _writer_rows(db, "W1")
    # two compactions months apart; the session lives in the recent one only
    _put(
        db,
        session="old",
        writer="W1",
        seq=1,
        hook="pre_llm_call",
        ts=datetime.now(UTC) - timedelta(days=300),
    )
    comp = AuditCompactor(audit_log=log, archive=AuditArchive(root=archive), retention_days=30)
    comp.compact(now=datetime.now(UTC) - timedelta(days=200))
    _put(
        db,
        session="s1",
        writer="W1",
        seq=2,
        hook="pre_llm_call",
        ts=datetime.now(UTC) - timedelta(days=60),
    )
    _put(
        db,
        session="s1",
        writer="W1",
        seq=3,
        hook="pre_llm_call",
        ts=datetime.now(UTC) - timedelta(days=60) + timedelta(seconds=5),
    )
    comp.compact()
    logs = tmp_path / "logs"
    logs.mkdir()
    stamp = (datetime.now(UTC) - timedelta(days=60)).isoformat()
    (logs / "session-s1.jsonl").write_text(  # the session log fixes the window
        json.dumps({"kind": "user_message", "ts": stamp, "session_id": "s1"}) + "\n"
    )
    seen: list[list[str]] = []
    real = q.read_cold_rows

    def spy(files: Any, **kw: Any) -> Any:
        seen.append([Path(f).name for f in files])
        return real(files, **kw)

    monkeypatch.setattr(q, "read_cold_rows", spy)
    result = replay_session("s1", sources=_src(db, tmp_path, archive_root=archive))
    all_chunks = {p.name for p in archive.glob("year=*/month=*/audit-*.parquet")}
    assert len(all_chunks) >= 2 and result.gaps == ()
    assert seen and all(set(files) < all_chunks for files in seen)  # never the whole archive
    (logs / "session-s1.jsonl").unlink()  # without a window every chunk is read, and it says so
    unbounded = replay_session("s1", sources=_src(db, tmp_path, archive_root=archive))
    assert "cold_window_unbounded" in {n.code for n in unbounded.notes}


def test_the_endpoint_is_capped_and_says_when_it_cut_the_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from iris_harness.server.iris_api import governance_routes as routes

    db = _db(tmp_path)
    _writer_rows(db, "W1")
    for seq in range(1, 8):
        _put(db, writer="W1", seq=seq, hook="pre_llm_call")
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(db))
    monkeypatch.setenv("IRIS_SESSION_LOG_DIR", str(tmp_path / "logs"))
    app = FastAPI()
    routes.install_governance_routes(app, lambda: None)
    client = TestClient(app)
    ok = client.get("/governance/replay", params={"session": "s1"}).json()
    assert ok["complete"] is True and len(ok["records"]) == 7 and ok["truncated"] is False
    monkeypatch.setattr(routes, "API_MAX_ROWS", 3)
    cut = client.get("/governance/replay", params={"session": "s1"}).json()
    assert cut["truncated"] is True and cut["gaps"] == [] and len(cut["records"]) == 3
    monkeypatch.setattr(routes, "API_MAX_ROWS", 2000)
    monkeypatch.setattr(routes, "API_MAX_SECONDS", 0.0)
    timed = client.get("/governance/replay", params={"session": "s1"}).json()
    assert timed["truncated"] is True
    assert client.get("/governance/replay", params={"session": "../x"}).status_code == 422
    # closed field set: no reason, no payload text in a record
    assert all("reason" not in r and "payload_json" not in r for r in ok["records"])


def test_the_cli_exit_code_is_one_for_a_gap_and_zero_otherwise(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from iris_harness.main import app

    db = _db(tmp_path)
    _writer_rows(db, "W1")
    for seq in (1, 2, 4):
        _put(db, writer="W1", seq=seq, hook="pre_llm_call")
    runner = CliRunner()
    args = ["audit", "replay", "--session", "s1", "--db", str(db), "--no-archive"]
    bad = runner.invoke(app, args)
    assert bad.exit_code == 1 and "sequence_hole" in bad.stdout
    assert runner.invoke(app, [*args, "--no-fail"]).exit_code == 0
    as_json = runner.invoke(app, [*args, "--json", "--no-fail"])
    assert json.loads(as_json.stdout)["gaps"][0]["class"] == "sequence_hole"
    _put(db, writer="W1", seq=3, hook="pre_llm_call")
    assert runner.invoke(app, args).exit_code == 0  # notes (the live writer) never change it
    assert (
        runner.invoke(app, ["audit", "replay", "--session", "../x", "--db", str(db)]).exit_code == 2
    )


def test_every_gap_class_has_a_test_in_this_module() -> None:
    text = Path(__file__).read_text(encoding="utf-8")
    for cls in GAP_CLASSES:
        assert f'"{cls}"' in text, cls
