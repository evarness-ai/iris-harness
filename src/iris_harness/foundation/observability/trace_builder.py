"""Reconstruct end-to-end call traces from session JSONL logs.

Turns the linear event stream written by :mod:`iris_harness.foundation.observability.session_log`
(``~/.iris/logs/session-<id>.jsonl``) into the node/edge graph consumed by the
web console's Agent Call Trace screen.

A *turn* is one request: ``user_message`` … ``agent_response``. Each turn becomes
one :class:`Trace`. Nodes map to the pipeline stages actually present in the log
(intent_router, memory, task_planner, agent(s), llm_call(s), tool_run(s),
response_curator) plus a synthesized ``runtime`` root. Governance hooks are not
in the session log (they live in the governance audit DB) and are omitted here.

Best-effort and read-only: malformed lines/turns are skipped, never raised.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import sqlite3
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from iris_harness.foundation.observability.audit_view import public_payload, tier_locality
from iris_harness.foundation.observability.session_log import TURN_OPENING_KINDS, session_log_dir
from iris_harness.foundation.process_state import track_globals

logger = logging.getLogger(__name__)

# stage key -> (module, method, file) for the verbose view.
_STAGE_META: dict[str, tuple[str, str, str]] = {
    "runtime": (
        "iris_harness.runtime.bootstrap",
        "IrisRuntime.chat",
        "src/iris_harness/runtime/bootstrap.py",
    ),
    "intent_router": (
        "iris_harness.agent.intent_router",
        "IntentRouter.classify",
        "src/iris_harness/agent/intent_router.py",
    ),
    "memory": (
        "iris_harness.memory.retriever",
        "MemoryRetriever.retrieve",
        "src/iris_harness/memory/retriever.py",
    ),
    "task_planner": (
        "iris_harness.agent.task_planner",
        "TaskPlanner.plan",
        "src/iris_harness/agent/task_planner.py",
    ),
    "response_curator": (
        "iris_harness.agent.response_curator",
        "ResponseCurator.curate",
        "src/iris_harness/agent/response_curator.py",
    ),
    # A deterministic handler answered the turn (the intercept stage). The handler itself
    # may be a plugin's, so the node names the dispatch that ran it and labels the handler.
    "handler": (
        "iris_harness.runtime.intercept_dispatch",
        "InterceptDispatch.dispatch",
        "src/iris_harness/runtime/intercept_dispatch.py",
    ),
    # The response check every deterministic answer passes (the guard stage).
    "guard": (
        "iris_harness.agent.response_curator",
        "ResponseCurator.guard",
        "src/iris_harness/agent/response_curator.py",
    ),
    "llm": ("iris_harness.llm.arbiter", "OllamaArbiter.invoke", "src/iris_harness/llm/arbiter.py"),
    "tool": (
        "iris_harness.tools.sandbox_tools",
        "run_shell",
        "src/iris_harness/tools/sandbox_tools.py",
    ),
    "governance": (
        "iris_harness.kernel.governance.kernel",
        "GovernanceKernel.fire",
        "src/iris_harness/kernel/governance/kernel.py",
    ),
}
# agent type -> the handler the agent executor registered for it. The verbose view names
# the module, callable and file that answered, and those are read off the callable
# itself: a literal table here went stale the day the email handler moved, and it
# named private plugin modules from the core (core/SDK boundary plan, PR 1). Filled by
# ``AgentExecutor.register``; process-wide, like the rest of the trace view.
_AGENT_SOURCES: dict[str, Callable[..., Any]] = {}


def _call_id_of(payload: dict[str, Any]) -> Any:
    """The call id of a ``tool.invoke.*`` event: ``call_id``, else the old ``tool_call_id``
    (a session log written before #134 has only that)."""
    return payload.get("call_id") or payload.get("tool_call_id")


def register_agent_source(agent_type: str, handler: Callable[..., Any]) -> None:
    """Record ``handler`` as what answers ``agent_type`` (the latest registration wins)."""
    _AGENT_SOURCES[agent_type] = handler


def clear_agent_sources() -> None:
    """Forget every recorded handler (tests)."""
    _AGENT_SOURCES.clear()


def _component_path(fn: Callable[..., Any], module: str) -> str:
    """The handler's source file, from its package root (``src/…`` in a checkout)."""
    try:
        source = inspect.getsourcefile(fn)
    except TypeError:
        source = None
    if not source:
        return ""
    parts = Path(source).parts
    top = module.split(".", 1)[0]
    if top not in parts:
        return ""
    at = len(parts) - 1 - parts[::-1].index(top)
    rel = "/".join(parts[at:])
    return f"src/{rel}" if at > 0 and parts[at - 1] == "src" else rel


def callable_meta(handler: Callable[..., Any]) -> tuple[str, str, str]:
    """``(module, method, file)`` for a registered handler, read off the callable.

    Follows ``__wrapped__`` (the plugin fault boundary sets it), a ``partial`` to its
    function, and a callable object to its ``__call__``. A closure is named by the
    factory that built it (``_make_code_exec_handler``, not ``….<locals>.handler``),
    which is the name a reader can find in the file.
    """
    fn: Any = inspect.unwrap(handler)
    while isinstance(fn, functools.partial):
        fn = inspect.unwrap(fn.func)
    if not (inspect.isfunction(fn) or inspect.ismethod(fn)) and callable(fn):
        fn = type(fn).__call__
    target = getattr(fn, "__func__", fn)
    module = str(getattr(target, "__module__", "") or "")
    qualname = str(getattr(target, "__qualname__", "") or getattr(target, "__name__", ""))
    method = qualname.split(".<locals>", 1)[0]
    return module, method, _component_path(target, module)


def _agent_meta(agent_type: str) -> tuple[str, str, str]:
    """What answered ``agent_type``: its own handler, else the ``system`` one, else blank."""
    handler = _AGENT_SOURCES.get(agent_type) or _AGENT_SOURCES.get("system")
    if handler is None:
        return "", "", ""
    try:
        return callable_meta(handler)
    except Exception:  # best-effort, like the rest of the trace view
        logger.debug("could not read the source of the %s handler", agent_type, exc_info=True)
        return "", "", ""


ID_SEP = "~"  # trace_id = "<session_id>~<turn_index>" (URL-safe, no encoding needed)


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _ms(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() * 1000.0


def _split_turns(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group a session's events into turns, each started by a ``TURN_OPENING_KINDS`` event:
    what the user said, or a turn the system opened (``turn_open``, ADR-0127)."""
    turns: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for e in events:
        if e.get("kind") in TURN_OPENING_KINDS:
            if cur:
                turns.append(cur)
            cur = [e]
        elif cur:
            cur.append(e)
    if cur:
        turns.append(cur)
    return turns


def _load_session(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return events


def _join_messages(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{m.get('role', '')}] {m.get('content', '')}" for m in messages)


def _tokens(raw: dict[str, Any] | None) -> dict[str, int] | None:
    if not raw:
        return None
    return {
        "prompt": int(raw.get("prompt_tokens", 0) or 0),
        "completion": int(raw.get("completion_tokens", 0) or 0),
        "total": int(raw.get("total_tokens", 0) or 0),
    }


_MEANINGFUL_KINDS = {"llm_call", "intent_router.end", "agent.trace", "tool_run", "handler.end"}


def _is_meaningful(turn: list[dict[str, Any]]) -> bool:
    """A turn worth charting — it actually exercised the request pipeline.

    Skips degenerate turns (e.g. ``approve`` / routine triggers) that log only a
    ``user_message`` + ``agent_response`` and would render as a lone node. A turn a
    deterministic handler answered counts (``handler.end``): it passed the input screen,
    the handler and the response check, and leaving it out hid every such answer from
    Sessions and Call trace.
    """
    return any(e.get("kind") in _MEANINGFUL_KINDS for e in turn)


def _turn_request(turn: list[dict[str, Any]]) -> str:
    """What the turn was asked: the user's words, or a system-opened turn's label."""
    first = turn[0] if turn else {}
    if first.get("kind") == "user_message":
        return str(first.get("text", ""))
    if first.get("kind") == "turn_open":
        return str(first.get("label") or first.get("opener") or "")
    return ""


def _opened_by_system(turn: list[dict[str, Any]]) -> bool:
    """True for a turn the system opened: no one typed anything (ADR-0127)."""
    return bool(turn) and turn[0].get("kind") == "turn_open"


def _turn_total_tokens(turn: list[dict[str, Any]]) -> int:
    total = 0
    for e in turn:
        if e.get("kind") == "llm_call":
            total += int((e.get("tokens") or {}).get("total_tokens", 0) or 0)
    return total


def _turn_started_at(turn: list[dict[str, Any]]) -> str:
    return str(turn[0].get("ts", "")) if turn else ""


def _turn_total_duration_ms(turn: list[dict[str, Any]]) -> float:
    resp = next((e for e in reversed(turn) if e.get("kind") == "agent_response"), None)
    if resp and resp.get("total_duration_ms") is not None:
        return float(resp["total_duration_ms"])
    if turn:
        t0 = _parse_ts(turn[0].get("ts", ""))
        tn = _parse_ts(turn[-1].get("ts", ""))
        if t0 and tn:
            return _ms(t0, tn)
    return 0.0


def _summary(session_id: str, idx: int, turn: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "trace_id": f"{session_id}{ID_SEP}{idx}",
        "request": _turn_request(turn),
        "started_at": _turn_started_at(turn),
        "total_duration_ms": _turn_total_duration_ms(turn),
        "total_tokens": _turn_total_tokens(turn),
    }


def list_traces(limit: int = 50) -> list[dict[str, Any]]:
    """Newest-first summaries of recent turns across all session logs."""
    out: list[dict[str, Any]] = []
    if not session_log_dir().exists():
        return out
    files = sorted(
        session_log_dir().glob("session-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for path in files:
        session_id = path.stem.removeprefix("session-")
        turns = _split_turns(_load_session(path))
        # newest turns within a file first
        for idx in range(len(turns) - 1, -1, -1):
            turn = turns[idx]
            if not _turn_request(turn) or not _is_meaningful(turn):
                continue
            out.append(_summary(session_id, idx, turn))
            if len(out) >= limit:
                return out
    return out


# ── Governance enrichment (from the kernel's audit DB) ─────────────────────────
# Session JSONL has no governance hooks (they live in the audit DB). Each pipeline
# stage stamps its own random run_id, so the kernel also records the active
# session_id in the audit payload (see GovernanceKernel._audit). We correlate by
# that stamped session_id (and run_id == session_id for older curator rows) within
# the turn's timestamp window. Best-effort and read-only: any failure (missing DB,
# locked, no json1, schema drift) yields no gov nodes.

_DECISION_RANK = {"deny": 3, "require_approval": 2, "transform": 1, "allow": 0}

# hook_point -> ordered candidate host kinds (first non-empty match wins).
_HOOK_HOST_KINDS: dict[str, tuple[list[str], ...]] = {
    "pre_classify": (["intent_router"],),
    "pre_llm_call": (["llm"], ["agent"]),
    "pre_tool_use": (["tool"], ["agent"]),
    "post_tool_use": (["tool"], ["agent"]),
    "post_step": (["agent"],),
    # A generated answer is checked in the curator; a deterministic one in the guard.
    "pre_response": (["response_curator"], ["guard"]),
}


def _audit_db_path() -> Path:
    # Foundation resolves it, so reading the ledger costs no import of the kernel that
    # writes it (M6.2). The env override lives there too, parsed once.
    from iris_harness.foundation.paths import audit_db_path

    return audit_db_path()


def _governance_rows(session_id: str, t0_iso: str, t1_iso: str) -> list[dict[str, Any]]:
    db = _audit_db_path()
    if not db.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, ts, run_id, step_id, hook_point, plugin, decision, severity, reason, "
            "classification, tier, payload_json FROM audit_log "
            "WHERE ts >= ? AND ts <= ? "
            "AND (run_id = ? OR json_extract(payload_json, '$.session_id') = ?) "
            "ORDER BY ts ASC, id ASC",
            (t0_iso, t1_iso, session_id, session_id),
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def _resolve_host(hook: str, ts_off: float, nodes: list[dict[str, Any]]) -> str:
    def nearest(kinds: list[str]) -> str | None:
        cands = [n for n in nodes if n["kind"] in kinds]
        if not cands:
            return None
        contained = [
            n for n in cands if n["t_offset_ms"] <= ts_off <= n["t_offset_ms"] + n["duration_ms"]
        ]
        pool = contained or cands
        return str(min(pool, key=lambda n: abs(n["t_offset_ms"] - ts_off))["id"])

    for kinds in _HOOK_HOST_KINDS.get(hook, ()):
        host = nearest(kinds)
        if host is not None:
            return host
    return "runtime"


def _governance_events(rows: list[dict[str, Any]], t0: datetime | None) -> list[dict[str, Any]]:
    """Every hook decision of the turn, in the order the kernel recorded them.

    The graph folds a hook point's rows into one node per host; this keeps each row --
    ``PRE_TURN``, ``PRE_LLM_CALL`` per step, ``PRE/POST_TOOL_USE``, ``PRE_RESPONSE`` --
    with who called, the label, the tier and whether a deterministic handler answered.
    Only the documented payload fields (``audit_view``); the reason is masked by the
    route that serves it, which is the display boundary.
    """
    events: list[dict[str, Any]] = []
    for r in rows:
        ts = _parse_ts(str(r.get("ts", "")))
        tier = r.get("tier")
        events.append(
            {
                "id": r.get("id"),
                "t_offset_ms": round(_ms(t0, ts), 3) if (t0 and ts) else 0.0,
                "run_id": r.get("run_id"),
                "step_id": r.get("step_id"),
                "hook_point": r.get("hook_point"),
                "plugin": r.get("plugin"),
                "decision": r.get("decision"),
                "severity": r.get("severity"),
                "classification": r.get("classification"),
                "tier": tier,
                "locality": tier_locality(str(tier) if tier else None),
                "reason": str(r.get("reason") or ""),
                **public_payload(r.get("payload_json")),
            }
        )
    return events


def _governance_nodes(
    rows: list[dict[str, Any]],
    t0: datetime | None,
    nodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []

    # Group rows by (hook_point, host_node) into one governance node each.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        hook = str(r.get("hook_point", ""))
        ts = _parse_ts(str(r.get("ts", "")))
        ts_off = _ms(t0, ts) if (t0 and ts) else 0.0
        host = _resolve_host(hook, ts_off, nodes)
        groups.setdefault((hook, host), []).append({**r, "_off": ts_off})

    gov_nodes: list[dict[str, Any]] = []
    gov_edges: list[dict[str, Any]] = []
    for i, ((hook, host), grp) in enumerate(groups.items()):
        worst = max(grp, key=lambda r: _DECISION_RANK.get(str(r.get("decision")), 0))
        decision = str(worst.get("decision", "allow"))
        # reason: non-allow plugins first, then the rest; capped.
        ordered = sorted(grp, key=lambda r: -_DECISION_RANK.get(str(r.get("decision")), 0))
        reason = "; ".join(
            f"{r.get('plugin')}: {r.get('reason')}" for r in ordered[:4] if r.get("reason")
        )[:400]
        node_id = f"gov{i}"
        node = _node(
            node_id,
            "governance",
            hook,
            "governance",
            min(r["_off"] for r in grp),
            2.0,
            status="error" if decision == "deny" else "ok",
            method=f"GovernanceKernel.fire({hook.upper()})",
            input=f"hook={hook} plugins={[r.get('plugin') for r in grp]}",
            output=f"{decision} ({len(grp)} plugin{'s' if len(grp) != 1 else ''})",
            governance={"hook": hook, "decision": decision, "reason": reason},
        )
        gov_nodes.append(node)
        gov_edges.append(
            {
                "id": f"e-gov-{node_id}",
                "source": host,
                "target": node_id,
                "kind": "control",
                "label": "guard",
            }
        )
    return gov_nodes, gov_edges


def list_sessions(
    limit: int = 50, *, skip: Callable[[str], bool] | None = None
) -> list[dict[str, Any]]:
    """Newest-first sessions, each with its meaningful turns (for the Sessions screen).

    Groups a session file's turns into one entry: turn count, time span, summed
    tokens/latency, and the per-turn summaries (newest turn first) that link to
    individual traces. ``skip`` leaves out sessions by id (the API passes the retention
    rule, so playground and eval runs stay out of the chat history) — before ``limit``
    is applied, so hiding runs never shortens the list.
    """
    out: list[dict[str, Any]] = []
    if not session_log_dir().exists():
        return out
    files = sorted(
        session_log_dir().glob("session-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if skip is not None:
        files = [p for p in files if not skip(p.stem.removeprefix("session-"))]
    for path in files[: max(1, limit)]:
        session_id = path.stem.removeprefix("session-")
        turns = _split_turns(_load_session(path))
        summaries = [
            _summary(session_id, idx, turn)
            for idx, turn in enumerate(turns)
            if _turn_request(turn) and _is_meaningful(turn)
        ]
        if not summaries:
            continue
        summaries.sort(key=lambda s: s["started_at"], reverse=True)
        # Title = the conversation opener (oldest turn's request), truncated — a
        # human-readable label for the history sidebar instead of the raw id.
        opener = min(summaries, key=lambda s: s["started_at"])["request"]
        out.append(
            {
                "session_id": session_id,
                "title": _title_from_request(opener),
                "turn_count": len(summaries),
                "started_at": min(s["started_at"] for s in summaries),
                "last_at": max(s["started_at"] for s in summaries),
                "total_tokens": sum(s["total_tokens"] for s in summaries),
                "total_duration_ms": round(sum(s["total_duration_ms"] for s in summaries), 3),
                "turns": summaries,
            }
        )
    out.sort(key=lambda s: s["last_at"], reverse=True)
    return out


def _title_from_request(request: str, *, limit: int = 60) -> str:
    text = " ".join(request.split())
    if len(text) <= limit:
        return text or "(untitled)"
    return text[:limit].rstrip() + "…"


def _turn_response(turn: list[dict[str, Any]]) -> str:
    resp = next((e for e in reversed(turn) if e.get("kind") == "agent_response"), None)
    return str((resp or {}).get("response", "")) if resp else ""


def session_messages(session_id: str) -> list[dict[str, Any]]:
    """Chronological user/assistant messages for one session — to replay in chat.

    Each turn with a user request yields a ``user`` message followed by the
    ``assistant`` response (linked to its trace so the UI can deep-link). Unlike
    the Sessions list, this keeps every real exchange (not just pipeline-heavy
    turns) so the resumed conversation reads back faithfully.
    """
    path = session_log_dir() / f"session-{session_id}.jsonl"
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    turns = _split_turns(_load_session(path))
    for idx, turn in enumerate(turns):
        request = _turn_request(turn)
        if not request:
            continue
        started = _turn_started_at(turn)
        # A turn the system opened had no user message, so its replay has none either:
        # only what IRIS said.
        if not _opened_by_system(turn):
            out.append({"role": "user", "text": request, "ts": started})
        out.append(
            {
                "role": "assistant",
                "text": _turn_response(turn),
                "ts": started,
                "trace_id": f"{session_id}{ID_SEP}{idx}",
            }
        )
    return out


# ── Reasoning steps (the "thinking process" narrative) ─────────────────────────
# A chronological, human-readable companion to the graph: request -> intent ->
# memory -> plan -> (llm reasoning / tool calls)* -> stop? -> curator -> answer.


def _first_line(text: str, limit: int = 140) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:limit]
    return (text or "")[:limit]


def build_steps(turn: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ordered reasoning steps for one turn — each with a concise title plus an
    expandable ``detail`` and small ``fields`` (model/tokens/verdicts/…)."""
    if not turn:
        return []
    t0 = _parse_ts(turn[0].get("ts", ""))

    def off(e: dict[str, Any]) -> float:
        ts = _parse_ts(e.get("ts", ""))
        return round(_ms(t0, ts), 3) if (t0 and ts) else 0.0

    steps: list[dict[str, Any]] = []
    pending_tools: dict[Any, dict[str, Any]] = {}

    def add(
        type_: str,
        title: str,
        e: dict[str, Any],
        *,
        status: str = "ok",
        detail: str | None = None,
        fields: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> None:
        steps.append(
            {
                "id": f"s{len(steps)}",
                "seq": len(steps),
                "type": type_,
                "title": title,
                "t_offset_ms": off(e),
                "duration_ms": duration_ms,
                "status": status,
                "iteration": e.get("iteration"),
                "detail": detail,
                "fields": fields or {},
            }
        )

    for e in turn:
        kind = e.get("kind")
        if kind == "user_message":
            text = str(e.get("text", ""))
            add("request", f"Request: {_first_line(text)}", e, detail=text)
        elif kind == "turn_open":
            label = str(e.get("label") or e.get("opener") or "")
            add(
                "request",
                f"Opened by IRIS: {_first_line(label)} (no user message)",
                e,
                detail=f"opener: {e.get('opener')}\nNo one typed anything; IRIS started this turn.",
                fields={"opener": e.get("opener")},
            )
        elif kind == "intent_router.end":
            p = e.get("payload", {}) or {}
            conf = p.get("confidence")
            conf_txt = f", conf {conf:.2f}" if isinstance(conf, int | float) else ""
            add(
                "intent",
                f"Intent → {p.get('agent_type', '?')} ({p.get('intent', '?')}{conf_txt})",
                e,
                detail=json.dumps(p, indent=2),
                fields={
                    "intent": p.get("intent"),
                    "agent": p.get("agent_type"),
                    "confidence": conf,
                    "multi_step": p.get("is_multi_step"),
                },
            )
        elif kind == "memory.context":
            p = e.get("payload", {}) or {}
            add(
                "memory",
                f"Memory loaded (profile {'✓' if p.get('has_user_profile') else '∅'}, "
                f"episodic {p.get('episodic_patterns', 0)}, recent {p.get('recent_turns', 0)})",
                e,
                detail=json.dumps(p, indent=2),
            )
        elif kind == "planner.end":
            p = e.get("payload", {}) or {}
            add(
                "plan",
                f"Plan: {p.get('plan_size', 0)} task(s), first={p.get('first_agent', '?')}",
                e,
                detail=json.dumps(p, indent=2),
                fields={"tasks": p.get("task_ids")},
            )
        elif kind == "llm_call":
            out = (e.get("output") or {}).get("text", "") or ""
            tk = _tokens(e.get("tokens"))
            model = e.get("model", "llm")
            add(
                "llm",
                f"LLM {model}: {_first_line(out) or '(no text output)'}",
                e,
                duration_ms=round(float(e.get("duration_ms") or 0.0), 3),
                detail="PROMPT:\n"
                + _join_messages(e.get("input_messages") or [])
                + "\n\nOUTPUT:\n"
                + out,
                fields={
                    "model": model,
                    "provider": e.get("provider"),
                    "tier": e.get("tier"),
                    "tokens": tk["total"] if tk else None,
                },
            )
        elif kind == "tool.invoke.start":
            pending_tools[_call_id_of(e.get("payload", {}))] = e
        elif kind == "tool.invoke.end":
            p = e.get("payload", {}) or {}
            start = pending_tools.pop(_call_id_of(p), None)
            args = (start or {}).get("payload", {}).get("arguments") if start else None
            ok = p.get("ok", True)
            name = p.get("tool", "tool")
            preview = str(p.get("result_preview") or p.get("error") or "")
            argstr = json.dumps(args) if args is not None else ""
            add(
                "tool",
                f"Tool {name}({_first_line(argstr, 80)}) → {_first_line(preview, 80) or 'ok'}",
                start or e,
                status="ok" if ok else "error",
                duration_ms=(off(e) - off(start)) if start else None,
                detail=f"args: {json.dumps(args, indent=2)}\n\nresult: {preview}",
                fields={"tool": name, "ok": ok},
            )
        elif kind == "tool_run":
            ec = int(e.get("exit_code", 0) or 0)
            add(
                "tool",
                f"$ {_first_line(str(e.get('cmd', '')), 80)} (exit {ec})",
                e,
                status="ok" if ec == 0 else "error",
                duration_ms=round(float(e.get("duration_ms") or 0.0), 3),
                detail=(
                    f"$ {e.get('cmd', '')}\n\nstdout:\n{e.get('stdout', '')}"
                    f"\n\nstderr:\n{e.get('stderr', '')}"
                ),
            )
        elif kind == "agent.trace" and e.get("phase") == "agent.result":
            p = e.get("payload", {}) or {}
            if p.get("success") is False:
                stall = p.get("stall_count")
                add(
                    "stop",
                    (
                        "Stopped: repeated action without progress"
                        if stall
                        else "Stopped (no successful result)"
                    ),
                    e,
                    status="error",
                    detail=json.dumps(p, indent=2),
                    fields={"stall_count": stall, "iterations": p.get("iterations")},
                )
        elif kind == "response_curator.end":
            p = e.get("payload", {}) or {}
            jb = (p.get("metadata", {}) or {}).get("judge_bundle", {}) or {}
            signals = jb.get("signals", []) or []
            verdicts = {s.get("name"): s.get("verdict") for s in signals}
            halted = jb.get("halted")
            summary = (
                "HALTED"
                if halted
                else (", ".join(f"{k}={v}" for k, v in verdicts.items()) or "done")
            )
            non_pass = [
                f"{s.get('name')}: {s.get('verdict')} — {s.get('reason')}"
                for s in signals
                if s.get("verdict") not in ("pass", "skipped")
            ]
            add(
                "curator",
                f"Curator: {summary}",
                e,
                status="error" if (halted or p.get("has_errors")) else "ok",
                detail="\n".join(non_pass) or json.dumps(p, indent=2),
                fields={
                    **verdicts,
                    "has_errors": p.get("has_errors"),
                    "chars": p.get("response_chars"),
                },
            )
        elif kind == "handler.end":
            p = e.get("payload", {}) or {}
            add(
                "handler",
                f"Answered by deterministic handler {p.get('handler') or '?'} (no model)",
                e,
                duration_ms=p.get("duration_ms"),
                detail=json.dumps(p, indent=2),
                fields={"handler": p.get("handler"), "intent": p.get("intent")},
            )
        elif kind == "guard.end":
            p = e.get("payload", {}) or {}
            checks = list(p.get("checks") or [])
            add(
                "guard",
                f"Response check: {p.get('verdict', '?')} ({', '.join(checks) or 'none'})",
                e,
                status="error" if p.get("verdict") == "halt" else "ok",
                duration_ms=p.get("duration_ms"),
                detail=json.dumps(p, indent=2),
                fields={"verdict": p.get("verdict"), "checks": checks},
            )
        elif kind == "agent_response":
            text = str(e.get("response", ""))
            empty = len(text.strip()) == 0
            add(
                "response",
                f"Final answer ({len(text)} chars)" + (" — empty/fallback" if empty else ""),
                e,
                status="error" if (e.get("has_errors") or empty) else "ok",
                detail=text or "(empty)",
                fields={"tokens": e.get("total_tokens"), "duration_ms": e.get("total_duration_ms")},
            )
    return steps


def get_trace(trace_id: str) -> dict[str, Any] | None:
    """Build the full node/edge graph for a single ``<session_id>~<turn>`` id."""
    if ID_SEP not in trace_id:
        return None
    session_id, _, idx_s = trace_id.rpartition(ID_SEP)
    try:
        idx = int(idx_s)
    except ValueError:
        return None
    path = session_log_dir() / f"session-{session_id}.jsonl"
    if not path.exists():
        return None
    turns = _split_turns(_load_session(path))
    if not (0 <= idx < len(turns)):
        return None
    try:
        return _build_trace(session_id, idx, turns[idx])
    except Exception:
        logger.exception("trace build failed for %s", trace_id)
        return None


def _node(
    node_id: str,
    kind: str,
    label: str,
    meta_key: str,
    t_offset_ms: float,
    duration_ms: float,
    *,
    status: str = "ok",
    method: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    module, default_method, file = _STAGE_META.get(meta_key, ("", "", ""))
    node: dict[str, Any] = {
        "id": node_id,
        "kind": kind,
        "label": label,
        "component_path": file,
        "module": module,
        "method": method or default_method,
        "status": status,
        "t_offset_ms": round(max(0.0, t_offset_ms), 3),
        "duration_ms": round(max(0.0, duration_ms), 3),
    }
    node.update({k: v for k, v in extra.items() if v is not None})
    return node


def _build_trace(session_id: str, idx: int, turn: list[dict[str, Any]]) -> dict[str, Any]:
    um = turn[0]
    resp = next((e for e in reversed(turn) if e.get("kind") == "agent_response"), None)
    t0 = _parse_ts(um.get("ts", "")) or _parse_ts((turn[0]).get("ts", ""))

    def off(e: dict[str, Any]) -> float:
        ts = _parse_ts(e.get("ts", ""))
        return _ms(t0, ts) if (t0 and ts) else 0.0

    request_text = _turn_request(turn)
    final_text = str((resp or {}).get("response", ""))
    total_dur = _turn_total_duration_ms(turn)
    has_errors = bool((resp or {}).get("has_errors"))

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    # Synthesized runtime root.
    nodes.append(
        _node(
            "runtime",
            "runtime",
            "IrisRuntime.open_turn" if _opened_by_system(turn) else "IrisRuntime.chat",
            "runtime",
            0.0,
            total_dur,
            status="error" if has_errors else "ok",
            method="IrisRuntime.open_turn" if _opened_by_system(turn) else None,
            input=request_text,
            output=final_text,
        )
    )

    # Main-chain stages, in pipeline order, only if present in the log.
    chain: list[str] = ["runtime"]

    # intent_router
    ir_start = next((e for e in turn if e.get("phase") == "intent_router.start"), None)
    ir_end = next((e for e in turn if e.get("kind") == "intent_router.end"), None)
    if ir_end:
        start_dt = _parse_ts((ir_start or ir_end).get("ts", ""))
        end_dt = _parse_ts(ir_end.get("ts", ""))
        dur = _ms(start_dt, end_dt) if (start_dt and end_dt) else 0.0
        nodes.append(
            _node(
                "intent_router",
                "intent_router",
                "intent_router",
                "intent_router",
                off(ir_start or ir_end),
                dur,
                input=request_text,
                output=json.dumps(ir_end.get("payload", {})),
            )
        )
        chain.append("intent_router")

    # memory (point event; duration estimated as gap to next stage)
    mem = next((e for e in turn if e.get("kind") == "memory.context"), None)
    plan = next((e for e in turn if e.get("kind") == "planner.end"), None)
    if mem:
        dur = off(plan) - off(mem) if plan else 20.0
        nodes.append(
            _node(
                "memory",
                "memory",
                "memory_retriever",
                "memory",
                off(mem),
                dur,
                input=f"session_id={session_id}",
                output=json.dumps(mem.get("payload", {})),
            )
        )
        chain.append("memory")
    if plan:
        nodes.append(
            _node(
                "task_planner",
                "task_planner",
                "task_planner",
                "task_planner",
                off(plan),
                6.0,
                input=json.dumps((ir_end or {}).get("payload", {})),
                output=json.dumps(plan.get("payload", {})),
            )
        )
        chain.append("task_planner")

    # agents (agent.start … agent.result), supports multiple
    agent_spans: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for e in turn:
        if e.get("kind") == "agent.trace" and e.get("phase") == "agent.start":
            pending = e
        elif (
            e.get("kind") == "agent.trace"
            and e.get("phase") == "agent.result"
            and pending is not None
        ):
            agent_spans.append({"start": pending, "result": e})
            pending = None
    if pending is not None:
        agent_spans.append({"start": pending, "result": None})

    for i, span in enumerate(agent_spans):
        start = span["start"]
        result = span["result"]
        atype = str((start.get("payload") or {}).get("agent_type", "system"))
        module, method, file = _agent_meta(atype)
        rpayload = (result or {}).get("payload", {})
        dur = float(rpayload.get("latency_ms") or (off(result) - off(start) if result else 0.0))
        node_id = f"agent{i}"
        node = _node(
            node_id,
            "agent",
            f"{atype}_agent",
            "runtime",
            off(start),
            dur,
            status="error" if rpayload.get("success") is False else "ok",
            input=start.get("text", ""),
            output=(result or {}).get("text", ""),
        )
        node["module"], node["method"], node["component_path"] = module, method, file
        nodes.append(node)
        chain.append(node_id)

    # A deterministic handler, when one answered the turn (stage `intercept`).
    handler_end = next((e for e in turn if e.get("kind") == "handler.end"), None)
    if handler_end is not None:
        hp = handler_end.get("payload", {}) or {}
        h_dur = float(hp.get("duration_ms") or 0.0)
        nodes.append(
            _node(
                "handler",
                "handler",
                str(hp.get("handler") or "handler"),
                "handler",
                off(handler_end) - h_dur,
                h_dur,
                input=request_text,
                output=final_text,
                handler=hp.get("handler"),
            )
        )
        chain.append("handler")

    # response_curator
    rc_start = next((e for e in turn if e.get("kind") == "response_curator.start"), None)
    rc_end = next((e for e in turn if e.get("kind") == "response_curator.end"), None)
    rc_anchor = rc_start or rc_end
    if rc_anchor is not None:
        s_dt = _parse_ts(rc_anchor.get("ts", ""))
        e_dt = _parse_ts((rc_end or rc_anchor).get("ts", ""))
        dur = _ms(s_dt, e_dt) if (s_dt and e_dt) else 0.0
        nodes.append(
            _node(
                "curator",
                "response_curator",
                "response_curator",
                "response_curator",
                off(rc_anchor),
                dur,
                input=json.dumps((rc_start or {}).get("payload", {})),
                output=final_text or json.dumps((rc_end or {}).get("payload", {})),
            )
        )
        chain.append("curator")

    # The response check a deterministic answer passed (stage `guard`).
    guard_end = next((e for e in turn if e.get("kind") == "guard.end"), None)
    if guard_end is not None:
        gp = guard_end.get("payload", {}) or {}
        g_dur = float(gp.get("duration_ms") or 0.0)
        nodes.append(
            _node(
                "guard",
                "guard",
                "response check",
                "guard",
                off(guard_end) - g_dur,
                g_dur,
                status="error" if gp.get("verdict") == "halt" else "ok",
                input=final_text,
                output=f"{gp.get('verdict', '?')} ({', '.join(gp.get('checks') or [])})",
            )
        )
        chain.append("guard")

    # Main chain edges.
    for a, b in zip(chain, chain[1:], strict=False):
        edges.append({"id": f"e-{a}-{b}", "source": a, "target": b, "kind": "data"})

    # LLM and tool branches, attached to their owning stage.
    def owner_for(ts_dt: datetime | None, agent_type: str) -> str:
        if agent_type == "intent_router" and any(n["id"] == "intent_router" for n in nodes):
            return "intent_router"
        # the agent whose span contains ts
        for i, span in enumerate(agent_spans):
            s = _parse_ts(span["start"].get("ts", ""))
            r = _parse_ts((span["result"] or {}).get("ts", "")) if span["result"] else None
            if s and ts_dt and ts_dt >= s and (r is None or ts_dt <= r):
                return f"agent{i}"
        if agent_spans:
            return "agent0"
        return "task_planner" if any(n["id"] == "task_planner" for n in nodes) else "runtime"

    li = 0
    ti = 0
    tool_windows: list[tuple[str, float, float]] = []
    pending_invokes: dict[Any, dict[str, Any]] = {}
    for e in turn:
        kind = e.get("kind")
        if kind == "tool.invoke.start":
            pending_invokes[_call_id_of(e.get("payload", {}))] = e
        elif kind == "tool.invoke.end":
            # ReAct-loop tool call (research, stock_quote, …). Pair with its
            # start by call id to recover args + timing.
            p = e.get("payload", {}) or {}
            start = pending_invokes.pop(_call_id_of(p), None)
            ts_dt = _parse_ts(e.get("ts", ""))
            node_id = f"tool{ti}"
            ti += 1
            ok = bool(p.get("ok", True))
            name = str(p.get("tool", "tool"))
            args = (start or {}).get("payload", {}).get("arguments")
            preview = str(p.get("result_preview") or p.get("error") or "")
            start_off = off(start) if start else off(e)
            tool_windows.append((node_id, start_off, off(e)))
            nodes.append(
                _node(
                    node_id,
                    "tool",
                    name,
                    "tool",
                    start_off,
                    max(0.0, off(e) - start_off),
                    status="ok" if ok else "error",
                    input=json.dumps(args) if args is not None else "",
                    output=preview,
                    tool={
                        "cmd": f"{name}({json.dumps(args)})" if args is not None else name,
                        "exit_code": 0 if ok else 1,
                        "stdout": preview if ok else "",
                        "stderr": "" if ok else preview,
                    },
                )
            )
            edges.append(
                {
                    "id": f"e-tool-{node_id}",
                    "source": owner_for(ts_dt, ""),
                    "target": node_id,
                    "kind": "control",
                    "label": "tool",
                }
            )
        elif kind == "tool_run":
            ts_dt = _parse_ts(e.get("ts", ""))
            node_id = f"tool{ti}"
            ti += 1
            exit_code = int(e.get("exit_code", 0) or 0)
            tool_windows.append((node_id, off(e) - float(e.get("duration_ms") or 0.0), off(e)))
            nodes.append(
                _node(
                    node_id,
                    "tool",
                    str(e.get("cmd", "tool"))[:40],
                    "tool",
                    off(e) - float(e.get("duration_ms") or 0.0),
                    float(e.get("duration_ms") or 0.0),
                    status="ok" if exit_code == 0 else "error",
                    input=e.get("cmd", ""),
                    output=e.get("stdout", ""),
                    tool={
                        "cmd": e.get("cmd", ""),
                        "exit_code": exit_code,
                        "stdout": e.get("stdout", ""),
                        "stderr": e.get("stderr", ""),
                    },
                    resources=e.get("resources"),
                )
            )
            edges.append(
                {
                    "id": f"e-tool-{node_id}",
                    "source": owner_for(ts_dt, ""),
                    "target": node_id,
                    "kind": "control",
                    "label": "tool",
                }
            )

    # Who made each LLM call. The session log names the pipeline stage or agent that
    # was running (``agent_type``); older logs say "unknown", so timing decides there.
    rc_window = (off(rc_anchor), off(rc_end or rc_anchor)) if rc_anchor is not None else None
    stage_owners = {
        "intent_router": "intent_router",
        "task_planner": "task_planner",
        "response_curator": "curator",
        "turn_capture": "runtime",  # the record stage belongs to the runtime itself
    }

    def llm_owner(ts_dt: datetime | None, agent_type: str, start_off: float, end_off: float) -> str:
        owner = stage_owners.get(agent_type)
        if owner is not None and any(n["id"] == owner for n in nodes):
            return owner
        # A tool that calls a model (inbox_digest summarising, …) owns that call.
        for tool_id, t_start, t_end in tool_windows:
            if t_start <= start_off and end_off <= t_end:
                return tool_id
        if agent_type in {"", "unknown"} and rc_window is not None:
            if rc_window[0] <= start_off <= rc_window[1]:
                return "curator"
        return owner_for(ts_dt, agent_type)

    for e in turn:
        if e.get("kind") != "llm_call":
            continue
        ts_dt = _parse_ts(e.get("ts", ""))
        dur = float(e.get("duration_ms") or 0.0)
        start_off = off(e) - dur  # ts is logged at call end
        agent_type = str(e.get("agent_type") or "")
        out = (e.get("output") or {}).get("text", "")
        node_id = f"llm{li}"
        li += 1
        nodes.append(
            _node(
                node_id,
                "llm",
                e.get("model", "llm") or "llm",
                "llm",
                start_off,
                dur,
                input=_join_messages(e.get("input_messages") or []),
                output=out,
                model=e.get("model"),
                provider=e.get("provider"),
                tier=e.get("tier"),
                agent=agent_type or None,
                tokens=_tokens(e.get("tokens")),
                resources=e.get("resources"),
            )
        )
        edges.append(
            {
                "id": f"e-llm-{node_id}",
                "source": llm_owner(ts_dt, agent_type, start_off, start_off + dur),
                "target": node_id,
                "kind": "control",
                "label": "llm",
            }
        )

    # Governance nodes from the audit DB, correlated by run_id==session_id + window.
    t0_iso = str(um.get("ts", ""))
    t1_iso = str((resp or turn[-1]).get("ts", t0_iso))
    gov_rows = _governance_rows(session_id, t0_iso, t1_iso)
    gov_nodes, gov_edges = _governance_nodes(gov_rows, t0, nodes)
    nodes.extend(gov_nodes)
    edges.extend(gov_edges)

    # Execution order. Every edge is numbered by when its target started, so the graph
    # reads as a sequence and not only a hierarchy; ties keep pipeline order.
    order = {n["id"]: (n["t_offset_ms"], i) for i, n in enumerate(nodes)}
    nodes.sort(key=lambda n: order[n["id"]])
    edges.sort(key=lambda e: order.get(e["target"], (float("inf"), len(order))))
    for seq, edge in enumerate(edges, start=1):
        edge["seq"] = seq

    return {
        "session_id": session_id,
        "trace_id": f"{session_id}{ID_SEP}{idx}",
        "request": request_text,
        "opened_by_system": _opened_by_system(turn),
        "started_at": str(um.get("ts", "")),
        "total_duration_ms": round(total_dur, 3),
        "total_tokens": _turn_total_tokens(turn),
        "nodes": nodes,
        "edges": edges,
        "steps": build_steps(turn),
        "governance": _governance_events(gov_rows, t0),
    }


__all__ = ["list_traces", "list_sessions", "get_trace"]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_AGENT_SOURCES")
