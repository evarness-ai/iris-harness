"""Routing accuracy scenario — golden set vs the production classifiers.

Two modes, both using the exact production code paths:

- ``keyword``: ``KeywordClassifier`` (deterministic, no LLM, CI-able).
- ``llm``: ``LLMRouterClassifier`` over the configured router tier model,
  built the same way ``bootstrap._llm_router_from_config`` does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from scenarios.common import REPO_ROOT, ScenarioRecord, Timer

GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "routing_golden.yaml"


def load_golden() -> list[dict[str, Any]]:
    raw = yaml.safe_load(GOLDEN_PATH.read_text(encoding="utf-8"))
    return list(raw["cases"])


def _build_classifier(mode: str) -> Any:
    from iris_harness.agent.intent_router import KeywordClassifier, LLMRouterClassifier

    if mode == "keyword":
        return KeywordClassifier()
    if mode == "llm":
        from iris_harness.llm.client import CodingLLMClient
        from iris_harness.llm.tier_router import TierRouter

        router = TierRouter.load_from_yaml(REPO_ROOT / "config" / "llm_tiers.yaml")
        cfg = router.get_llm_config("intent_classification")
        client = CodingLLMClient(cfg)

        def invoke(system_prompt: str, user_prompt: str) -> str:
            return client.invoke(system_prompt=system_prompt, user_prompt=user_prompt)

        return LLMRouterClassifier(invoke)
    raise ValueError(f"unknown routing mode: {mode}")


def run(mode: str) -> list[ScenarioRecord]:
    classifier = _build_classifier(mode)
    records: list[ScenarioRecord] = []
    for case in load_golden():
        with Timer() as timer:
            result = classifier.classify(case["utterance"])
        ok = result.intent == case["expect_intent"] and result.agent_type == case["expect_agent"]
        records.append(
            ScenarioRecord(
                scenario=f"routing-{mode}",
                case_id=case["id"],
                verdict="pass" if ok else "fail",
                expected=f"{case['expect_intent']}/{case['expect_agent']}",
                actual=f"{result.intent}/{result.agent_type}",
                latency_ms=round(timer.elapsed_ms, 2),
                detail={
                    "utterance": case["utterance"],
                    "class": case.get("class", "clear"),
                    "confidence": result.confidence,
                    "source": result.source,
                    "is_multi_step": result.is_multi_step,
                    "note": case.get("note", ""),
                },
            )
        )
    return records


def extra_summary(records: list[ScenarioRecord]) -> str:
    """Accuracy split by case class + misroute direction table."""
    lines: list[str] = []
    by_class: dict[str, list[ScenarioRecord]] = {}
    for record in records:
        by_class.setdefault(record.detail["class"], []).append(record)
    for cls, recs in sorted(by_class.items()):
        passed = sum(1 for r in recs if r.verdict == "pass")
        lines.append(f"  class={cls}: {passed}/{len(recs)} ({passed / len(recs):.0%})")
    sources = {}
    for record in records:
        sources[record.detail["source"]] = sources.get(record.detail["source"], 0) + 1
    lines.append(f"  sources: {sources}")
    latencies = sorted(r.latency_ms for r in records)
    lines.append(
        f"  latency ms: p50={latencies[len(latencies) // 2]:.1f} " f"max={latencies[-1]:.1f}"
    )
    return "\n".join(lines)
