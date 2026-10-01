"""Phase 3 follow-up — Tier-3 quality battery.

The Tier-3 MoE (qwen3.6:35b-a3b on MLX) was adopted on *throughput*
evidence (exp-004: 4x decode, 118s->30s) plus exp-003's coding grade. It
was never graded for *quality* in-harness on IRIS's own Tier-3 workloads
(complex_task / skill_writing). This closes that gap.

Runs a small graded battery through the **production** Tier-3 client
(`tier_router.get_llm_config("complex_task")` -> `CodingLLMClient`):

- **code** tasks: extract the function, execute it against asserts in a
  subprocess — pass iff the asserts pass (exp-002 methodology).
- **reasoning** tasks: grade by required-substring rubric (every required
  token must appear) — a coarse but objective check that the model
  produced the asked-for structured content.

Every case also checks that no `<think>` block leaked into content (the
exp-004 adoption guard) and records decode wall + completion tokens.

    IRIS_GOVERNANCE_ENABLED=1 poetry run python -m scenarios.run tier3
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from scenarios.common import REPO_ROOT, ScenarioRecord, Timer

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    blocks = _CODE_BLOCK_RE.findall(text)
    return blocks[-1] if blocks else text


def _run_code(code: str, test: str) -> tuple[bool, str]:
    """Exec extracted code + test asserts in a subprocess; True iff exit 0."""
    program = f"{code}\n\n{test}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(program)
        path = fh.name
    try:
        # Grades model-generated code by execution in a throwaway subprocess —
        # a scenario harness, not production; the input is the thing under test.
        proc = subprocess.run(  # noqa: S603
            [sys.executable, path],  # noqa: S603, S607
            capture_output=True,
            text=True,
            timeout=20,
        )
        ok = proc.returncode == 0
        detail = "" if ok else (proc.stderr.strip().splitlines() or [""])[-1][:160]
        return ok, detail
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        Path(path).unlink(missing_ok=True)


# (id, kind, system, prompt, grader-payload)
CODE_TASKS: list[dict[str, Any]] = [
    {
        "id": "code-merge-intervals",
        "system": "You are a senior Python engineer. Return only the function in a code block.",
        "prompt": (
            "Write `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` "
            "that merges all overlapping intervals and returns them sorted by start."
        ),
        "test": (
            "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
            "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n"
            "assert merge_intervals([]) == []"
        ),
    },
    {
        "id": "code-lru-cache",
        "system": "You are a senior Python engineer. Return only the class in a code block.",
        "prompt": (
            "Implement an `LRUCache` class with `__init__(self, capacity: int)`, "
            "`get(self, key) -> int` (returns -1 if absent), and `put(self, key, value)`. "
            "Evict the least-recently-used item when over capacity."
        ),
        "test": (
            "c = LRUCache(2)\n"
            "c.put(1, 1); c.put(2, 2)\n"
            "assert c.get(1) == 1\n"
            "c.put(3, 3)\n"  # evicts 2
            "assert c.get(2) == -1\n"
            "c.put(4, 4)\n"  # evicts 1
            "assert c.get(1) == -1\n"
            "assert c.get(3) == 3 and c.get(4) == 4"
        ),
    },
    {
        "id": "code-parse-duration",
        "system": "You are a senior Python engineer. Return only the function in a code block.",
        "prompt": (
            "Write `parse_duration(s: str) -> int` that parses strings like "
            "'1h30m', '45m', '2h', '90s' into total seconds. Hours=h, minutes=m, "
            "seconds=s. Return 0 for an empty string."
        ),
        "test": (
            "assert parse_duration('1h30m') == 5400\n"
            "assert parse_duration('45m') == 2700\n"
            "assert parse_duration('2h') == 7200\n"
            "assert parse_duration('90s') == 90\n"
            "assert parse_duration('') == 0"
        ),
    },
]

REASONING_TASKS: list[dict[str, Any]] = [
    {
        "id": "reason-plan-decompose",
        "system": "You are a planning assistant. Be concrete and structured.",
        "prompt": (
            "Decompose this into an ordered, numbered plan of 4-7 steps: 'Build a "
            "CLI that watches a folder and indexes new text files for full-text "
            "search.' Mention the watcher, the index, and the query interface."
        ),
        "required": ["watch", "index", "search"],
        "min_steps": 4,
    },
    {
        "id": "reason-tradeoff",
        "system": "You are a technical advisor. Give a clear recommendation with reasons.",
        "prompt": (
            "A user asks whether to store 100k embeddings in SQLite, a flat numpy "
            "file, or a dedicated vector DB on a single laptop. Recommend ONE and "
            "give two concrete reasons. Name the option you recommend explicitly."
        ),
        "required_any": ["sqlite", "numpy", "vector"],
        "min_reasons": 2,
    },
]


def _tier3_client() -> Any:
    from iris_harness.llm.client import CodingLLMClient
    from iris_harness.llm.tier_router import TierRouter

    router = TierRouter.load_from_yaml(REPO_ROOT / "config" / "llm_tiers.yaml")
    cfg = router.get_llm_config("complex_task")
    return CodingLLMClient(cfg), cfg


def run() -> list[ScenarioRecord]:
    client, cfg = _tier3_client()
    records: list[ScenarioRecord] = [
        ScenarioRecord(
            scenario="tier3-quality",
            case_id="tier3-config",
            verdict="pass" if getattr(cfg, "tier_name", "") == "tier3" else "fail",
            expected="complex_task -> tier3",
            actual=f"{cfg.provider}/{cfg.model} (tier_name={getattr(cfg, 'tier_name', '?')})",
            latency_ms=0.0,
        )
    ]

    for task in CODE_TASKS:
        with Timer() as timer:
            out = client.invoke(system_prompt=task["system"], user_prompt=task["prompt"])
        ok, detail = _run_code(_extract_code(out), task["test"])
        records.append(
            ScenarioRecord(
                scenario="tier3-quality",
                case_id=task["id"],
                verdict="pass" if ok else "fail",
                expected="asserts pass on execution",
                actual="executed-pass" if ok else f"fail: {detail}",
                latency_ms=round(timer.elapsed_ms, 2),
                detail={"think_leak": "<think>" in out, "chars": len(out)},
            )
        )

    for task in REASONING_TASKS:
        with Timer() as timer:
            out = client.invoke(system_prompt=task["system"], user_prompt=task["prompt"])
        lowered = out.lower()
        required_ok = all(tok in lowered for tok in task.get("required", []))
        if "required_any" in task:
            required_ok = required_ok and any(tok in lowered for tok in task["required_any"])
        steps = len(re.findall(r"^\s*\d+[.)]", out, re.MULTILINE))
        struct_ok = steps >= task.get("min_steps", 0)
        if "min_reasons" in task:
            # crude: count bullet/numbered reason markers OR "because/reason" cues
            reasons = len(re.findall(r"(?:^\s*[-*]|\bbecause\b|\breason)", lowered, re.MULTILINE))
            struct_ok = reasons >= task["min_reasons"]
        ok = required_ok and struct_ok
        records.append(
            ScenarioRecord(
                scenario="tier3-quality",
                case_id=task["id"],
                verdict="pass" if ok else "fail",
                expected="required tokens present + structured",
                actual=f"required={required_ok} structured={struct_ok}",
                latency_ms=round(timer.elapsed_ms, 2),
                detail={"think_leak": "<think>" in out, "chars": len(out)},
            )
        )
    return records


def extra_summary(records: list[ScenarioRecord]) -> str:
    graded = [r for r in records if r.case_id != "tier3-config"]
    leaks = sum(1 for r in graded if r.detail.get("think_leak"))
    lats = sorted(r.latency_ms for r in graded if r.latency_ms > 0)
    lines = [f"  think-leaks: {leaks}/{len(graded)} (must be 0)"]
    if lats:
        lines.append(f"  latency ms: p50={lats[len(lats) // 2]:.0f} max={lats[-1]:.0f}")
    return "\n".join(lines)
