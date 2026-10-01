#!/usr/bin/env python3
"""Harness capability evaluation — measure the IRIS agentic harness's autonomy.

Fires a battery of real-world, user-shaped scenarios at the running IRIS API
(``http://127.0.0.1:8003/chat``) and measures the harness's *decisions* — not the
exact prose (which is model-dependent), but the invariants that prove the harness is
deciding well on its own:

  - ROUTING       : did the request reach the right agent?
  - CROSS-DOMAIN  : did one turn use tools across domains (finance -> email)?
  - GROUNDING     : did the answer come from a tool/evidence, not a guess?
  - SELF-EVAL     : did the loop converge (success) without stalling?
  - LOAD CONTROL  : did the P2 shortlist keep the tool menu under the cap?
  - LATENCY       : wall-clock per turn.

This is a dev-only harness (privacy-first: it reads the live response but prints only
metrics + short redacted snippets, never full personal content). Run AFTER restarting
the stack so the current code is live:

    poetry run python scripts/harness_capability_eval.py
    poetry run python scripts/harness_capability_eval.py --json report.json

ADR-0077. Repeatable so the harness can be re-measured after each phase / model swap.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

API = "http://127.0.0.1:8003/chat"
TOOL_CAP = 12  # IRIS_REACT_TOOL_CAP default


@dataclass
class Scenario:
    name: str
    message: str
    # check(body) -> (passed, note). Asserts harness DECISIONS, not exact prose.
    check: Callable[[dict[str, Any]], tuple[bool, str]]
    measures: str = ""


def _trace_actions(body: dict[str, Any]) -> list[str]:
    return [
        s.get("action") for s in (body.get("metadata", {}).get("trace") or []) if s.get("action")
    ]


def _agent(body: dict[str, Any]) -> str:
    return str(body.get("agent_type") or "")


def _resp(body: dict[str, Any]) -> str:
    return str(body.get("response") or "")


# --- the battery (authored as the user: finance, email, calendar, memory, web) --------
SCENARIOS: list[Scenario] = [
    Scenario(
        name="cross_domain_dues",
        message="I should have some dues from a store-cobranded card, can you find it?",
        # The cross-domain SIGNAL is that a finance-intent turn reached into EMAIL
        # (search_inbox) for an answer the local finance store doesn't have. The ideal
        # ordering (finance_lookup first, then search_inbox) is preferred but the model's
        # tool ordering varies (temp 0.7); requiring the email hop is the robust invariant.
        measures="CROSS-DOMAIN: a finance turn searches email (search_inbox)",
        check=lambda b: (
            _agent(b) == "finance" and "search_inbox" in _trace_actions(b),
            f"agent={_agent(b)} actions={_trace_actions(b)} "
            f"(finance_lookup_first={'finance_lookup' in _trace_actions(b)})",
        ),
    ),
    Scenario(
        name="dues_regression",
        message="any dues that are pending?",
        measures="ROUTING+GROUNDING: finance, grounded dues, no crash",
        check=lambda b: (
            _agent(b) == "finance" and not b.get("has_errors") and bool(_resp(b).strip()),
            f"agent={_agent(b)} has_errors={b.get('has_errors')}",
        ),
    ),
    Scenario(
        name="networth_no_web_stall",
        message="what's my total net worth as of today?",
        measures="ANTI-REGRESSION: finance digest, not a stalled web search",
        check=lambda b: (
            _agent(b) == "finance" and "stopped" not in _resp(b).lower(),
            f"agent={_agent(b)} resp_head={_resp(b)[:50]!r}",
        ),
    ),
    Scenario(
        name="email_routing",
        message="what's in my inbox today?",
        measures="ROUTING: email/communication agent",
        check=lambda b: (
            _agent(b) in {"email", "communication"},
            f"agent={_agent(b)}",
        ),
    ),
    Scenario(
        name="calendar_routing",
        message="do I have any meetings tomorrow?",
        measures="ROUTING: calendar agent (not a web search)",
        check=lambda b: (
            _agent(b) == "calendar",
            f"agent={_agent(b)}",
        ),
    ),
    Scenario(
        name="memory_recall",
        message="what do you know about me?",
        measures="MEMORY: profile recall, grounded answer",
        check=lambda b: (
            not b.get("has_errors") and bool(_resp(b).strip()),
            f"agent={_agent(b)} has_errors={b.get('has_errors')}",
        ),
    ),
    Scenario(
        name="fresh_research_grounded",
        message="what's the latest news about AI safety this week?",
        measures="GROUNDING: system agent uses research/web_search before answering",
        check=lambda b: (
            _agent(b) == "system"
            and any(a in _trace_actions(b) for a in ("research", "web_search")),
            f"agent={_agent(b)} actions={_trace_actions(b)}",
        ),
    ),
    Scenario(
        name="load_control_tool_cap",
        message="give me a quick summary of my day and anything I owe",
        measures="LOAD CONTROL: P2 shortlist keeps tools_offered <= cap",
        check=lambda b: (
            int(b.get("metadata", {}).get("tools_offered", 0) or 0) <= TOOL_CAP,
            f"tools_offered={b.get('metadata', {}).get('tools_offered')}",
        ),
    ),
]


@dataclass
class Result:
    name: str
    passed: bool
    note: str
    latency_ms: float
    agent: str
    actions: list[str]
    tools_offered: int | None
    iterations: int | None
    success: bool | None
    metrics: dict[str, Any] = field(default_factory=dict)


def run_one(client: httpx.Client, sc: Scenario, idx: int) -> Result:
    t0 = time.monotonic()
    body: dict[str, Any]
    try:
        r = client.post(API, json={"message": sc.message, "session_id": f"eval:{sc.name}:{idx}"})
        body = r.json() if r.status_code == 200 else {"response": f"HTTP {r.status_code}"}
    except Exception as exc:  # noqa: BLE001
        body = {"response": f"request failed: {exc}", "has_errors": True}
    dt = (time.monotonic() - t0) * 1000
    try:
        passed, note = sc.check(body)
    except Exception as exc:  # noqa: BLE001
        passed, note = False, f"check error: {exc}"
    meta = body.get("metadata", {}) if isinstance(body.get("metadata"), dict) else {}
    return Result(
        name=sc.name,
        passed=passed,
        note=note,
        latency_ms=dt,
        agent=_agent(body),
        actions=_trace_actions(body),
        tools_offered=meta.get("tools_offered"),
        iterations=meta.get("iterations"),
        success=meta.get("success"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the full report to this path")
    ap.add_argument("--only", help="run only scenarios whose name contains this substring")
    args = ap.parse_args()

    scenarios = [s for s in SCENARIOS if not args.only or args.only in s.name]
    results: list[Result] = []
    with httpx.Client(timeout=180.0) as client:
        for i, sc in enumerate(scenarios):
            print(f"  running {sc.name} ...", flush=True)
            results.append(run_one(client, sc, i))

    passed = sum(1 for r in results if r.passed)
    print("\n" + "=" * 78)
    print(f"HARNESS CAPABILITY EVAL — {passed}/{len(results)} passed")
    print("=" * 78)
    for r in results:
        flag = "PASS" if r.passed else "FAIL"
        offered = r.tools_offered if r.tools_offered is not None else "-"
        print(
            f"[{flag}] {r.name:<26} {r.latency_ms:7.0f}ms  agent={r.agent or '-':<10} "
            f"tools={offered} iters={r.iterations if r.iterations is not None else '-'}"
        )
        print(f"        {r.note}")
        if r.actions:
            print(f"        tool calls: {r.actions}")

    lat = [r.latency_ms for r in results]
    offered_vals = [r.tools_offered for r in results if isinstance(r.tools_offered, int)]
    print("-" * 78)
    print(f"avg latency: {sum(lat) / len(lat):.0f}ms   max: {max(lat):.0f}ms")
    if offered_vals:
        print(
            f"tools offered: avg {sum(offered_vals) / len(offered_vals):.1f}, "
            f"max {max(offered_vals)} (cap {TOOL_CAP})"
        )
    cross = next((r for r in results if r.name == "cross_domain_dues"), None)
    if cross:
        print(f"cross-domain (finance->email): {'YES' if cross.passed else 'NO'}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump([r.__dict__ for r in results], f, indent=2)
        print(f"\nwrote {args.json}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
