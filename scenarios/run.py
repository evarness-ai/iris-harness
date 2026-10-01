"""Phase 2 scenario harness CLI.

Usage (from repo root):

    poetry run python -m scenarios.run routing-keyword
    poetry run python -m scenarios.run routing-llm        # needs Ollama
    poetry run python -m scenarios.run governance         # needs nothing local
    poetry run python -m scenarios.run email-triage       # loads MiniLM embedder
    poetry run python -m scenarios.run all                # everything above

Results append to docs/testing-program/results/<scenario>/<stamp>.jsonl.
Governance scenarios force IRIS_GOVERNANCE_ENABLED=1 — they are meaningless
without the kernel.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scenarios.common import ScenarioRecord, run_stamp, summarize, write_results  # noqa: E402


def _run_routing(mode: str, stamp: str) -> list[ScenarioRecord]:
    from scenarios import routing

    records = routing.run(mode)
    path = write_results(f"routing-{mode}", stamp, records)
    print(f"\n=== routing-{mode} ===")
    print(summarize(records))
    print(routing.extra_summary(records))
    print(f"results: {path}")
    return records


def _run_governance(stamp: str) -> list[ScenarioRecord]:
    from scenarios import governance

    records = governance.run()
    path = write_results("governance", stamp, records)
    print("\n=== governance ===")
    print(summarize(records))
    print(f"results: {path}")
    return records


def _run_tier3(stamp: str) -> list[ScenarioRecord]:
    from scenarios import tier3_quality

    records = tier3_quality.run()
    path = write_results("tier3-quality", stamp, records)
    print("\n=== tier3-quality (production Tier-3 client) ===")
    print(summarize(records))
    print(tier3_quality.extra_summary(records))
    print(f"results: {path}")
    return records


def _run_multiturn(stamp: str) -> list[ScenarioRecord]:
    from scenarios import multiturn

    records = multiturn.run()
    path = write_results("multiturn", stamp, records)
    print("\n=== multiturn ===")
    print(summarize(records))
    print(f"results: {path}")
    return records


def _run_approvals(stamp: str) -> list[ScenarioRecord]:
    from scenarios import approvals_hitl

    records = approvals_hitl.run()
    path = write_results("approvals-hitl", stamp, records)
    print("\n=== approvals-hitl ===")
    print(summarize(records))
    print(f"results: {path}")
    return records


def _run_email_live(stamp: str) -> list[ScenarioRecord]:
    from scenarios import email_live

    records = email_live.run()
    path = write_results("email-live", stamp, records)
    print("\n=== email-live (READ-ONLY against the real account) ===")
    print(summarize(records))
    print(email_live.extra_summary(records))
    print(f"results: {path}")
    return records


def _run_email_triage(stamp: str) -> list[ScenarioRecord]:
    from scenarios import email_triage

    records = email_triage.run()
    path = write_results("email-triage", stamp, records)
    print("\n=== email-triage ===")
    print(summarize(records))
    print(email_triage.extra_summary(records))
    print(f"results: {path}")
    return records


def _run_finance_body_read(stamp: str) -> list[ScenarioRecord]:
    import json
    from dataclasses import asdict

    from scenarios import finance_body_read

    records = finance_body_read.run()
    print("\n=== finance-body-read (ADR-0121 acceptance, a copy of the owner's data) ===")
    if not records:
        return records
    print(summarize(records))
    # The owner's figures stay on this machine: results go beside the expectations
    # file, not under docs/testing-program/results (which is committed).
    finance_body_read.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = finance_body_read.RESULTS_DIR / f"{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), default=str, sort_keys=True) + "\n")
    print(f"results: {path}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="IRIS scenario harness")
    parser.add_argument(
        "scenario",
        choices=[
            "routing-keyword",
            "routing-llm",
            "governance",
            "email-triage",
            "email-live",
            "approvals",
            "multiturn",
            "tier3",
            "finance-body-read",
            "all",
        ],
    )
    args = parser.parse_args()

    # Governance must be live for scenario runs — the harness measures the
    # governed system, never the legacy ungoverned path.
    os.environ.setdefault("IRIS_GOVERNANCE_ENABLED", "1")

    stamp = run_stamp()
    failures = 0

    if args.scenario in ("routing-keyword", "all"):
        failures += sum(1 for r in _run_routing("keyword", stamp) if r.verdict == "fail")
    if args.scenario in ("routing-llm", "all"):
        failures += sum(1 for r in _run_routing("llm", stamp) if r.verdict == "fail")
    if args.scenario in ("governance", "all"):
        failures += sum(1 for r in _run_governance(stamp) if r.verdict == "fail")
    if args.scenario in ("email-triage", "all"):
        failures += sum(1 for r in _run_email_triage(stamp) if r.verdict == "fail")
    # Deliberately NOT in "all": touches the real account (read-only) and
    # depends on live OAuth state — run explicitly.
    if args.scenario in ("approvals", "all"):
        failures += sum(1 for r in _run_approvals(stamp) if r.verdict == "fail")
    if args.scenario in ("multiturn", "all"):
        failures += sum(1 for r in _run_multiturn(stamp) if r.verdict == "fail")
    # Not in "all": needs the Tier-3 MoE loaded (LM Studio). Run explicitly.
    if args.scenario == "tier3":
        failures += sum(1 for r in _run_tier3(stamp) if r.verdict == "fail")
    if args.scenario == "email-live":
        failures += sum(1 for r in _run_email_live(stamp) if r.verdict == "fail")
    # Not in "all": reads the owner's mailbox (read-only) on a copy of their data, with
    # their expectations file. Run explicitly.
    if args.scenario == "finance-body-read":
        failures += sum(1 for r in _run_finance_body_read(stamp) if r.verdict == "fail")

    print(f"\ntotal failing cases: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
