"""Email triage scenario — synthetic corpus with ground truth, real embeddings.

Builds an isolated sandbox (tmp iris.db / email.db / workspace), seeds the
6-category taxonomy + proposals.jsonl representatives, injects the labeled
synthetic corpus into the EmailStore, then runs the production pure-kNN
triage path (``EmailTriageClassifier.classify``) and grades each prediction
against ground truth. Uses the real MiniLM embedder — this measures actual
classifier quality, not a stub.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from scenarios.common import ScenarioRecord, Timer

CORPUS_PATH = Path(__file__).resolve().parent / "data" / "email_corpus.json"


def _load_corpus() -> dict[str, Any]:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def _seed_sandbox(sandbox: Path, corpus: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Create iris.db categories, proposals.jsonl, and the email store."""
    from iris_harness.data.categories import Category, CategoryStore, make_path
    from iris_harness.email.store import EmailStore
    from iris_harness.skills.contracts.email import EmailMessage

    account_id = corpus["account_id"]
    iris_db = sandbox / "iris.db"
    email_db = sandbox / "email.db"
    workspace = sandbox / "workspace"

    category_store = CategoryStore(db_path=iris_db)
    category_store.ensure_schema()
    proposals: list[dict[str, Any]] = []
    for cat in corpus["categories"]:
        path = make_path("email", cat["root"], cat["branch"], cat["leaf"])
        category_store.upsert_if_new(
            Category(
                path=path,
                type="email",
                root=cat["root"],
                branch=cat["branch"],
                leaf=cat["leaf"],
                account_id=account_id,
                cohesion=cat["cohesion"],
            ),
            source="scenario-harness",
        )
        proposals.append(
            {
                "proposed_root": cat["root"],
                "proposed_branch": cat["branch"],
                "proposed_leaf": cat["leaf"],
                "cohesion": cat["cohesion"],
                "representatives": cat["representatives"],
            }
        )

    slug = account_id.replace(":", "-").replace("@", "-at-")
    proposals_path = workspace / "email" / slug / "proposals.jsonl"
    proposals_path.parent.mkdir(parents=True, exist_ok=True)
    proposals_path.write_text("\n".join(json.dumps(p) for p in proposals) + "\n", encoding="utf-8")

    email_store = EmailStore(db_path=email_db)
    email_store.ensure_schema()
    base_time = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)
    for index, mail in enumerate(corpus["emails"]):
        email_store.upsert(
            EmailMessage(
                id=mail["id"],
                provider="gmail",
                account_id=account_id,
                from_address=mail["from_address"],
                subject=mail["subject"],
                snippet=mail["snippet"],
                received_at=base_time + timedelta(minutes=index),
            )
        )
    return iris_db, email_db, workspace


def run() -> list[ScenarioRecord]:
    from iris_harness.email.triage import EmailTriageClassifier
    from iris_harness.skills.contracts.email import EmailMessage

    corpus = _load_corpus()
    account_id = corpus["account_id"]
    records: list[ScenarioRecord] = []

    with tempfile.TemporaryDirectory(prefix="iris-scenario-email-") as tmp:
        sandbox = Path(tmp)
        iris_db, email_db, workspace = _seed_sandbox(sandbox, corpus)

        classifier = EmailTriageClassifier(
            workspace_dir=workspace,
            db_path=iris_db,
            email_db_path=email_db,
        )
        with Timer() as load_timer:
            centroids = classifier.load_categories(account_id)
        records.append(
            ScenarioRecord(
                scenario="email-triage",
                case_id="load-categories",
                verdict="pass" if len(centroids) == len(corpus["categories"]) else "fail",
                expected=f"{len(corpus['categories'])} centroids",
                actual=f"{len(centroids)} centroids",
                latency_ms=round(load_timer.elapsed_ms, 2),
            )
        )

        base_time = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)
        for mail in corpus["emails"]:
            message = EmailMessage(
                id=mail["id"],
                provider="gmail",
                account_id=account_id,
                from_address=mail["from_address"],
                subject=mail["subject"],
                snippet=mail["snippet"],
                received_at=base_time,
            )
            with Timer() as timer:
                result = classifier.classify(message)
            predicted = result.category_path or ("<queued>" if result.queued else "<none>")
            ok = result.category_path == mail["category"] and not result.queued
            records.append(
                ScenarioRecord(
                    scenario="email-triage",
                    case_id=mail["id"],
                    verdict="pass" if ok else "fail",
                    expected=mail["category"],
                    actual=predicted,
                    latency_ms=round(timer.elapsed_ms, 2),
                    detail={
                        "confidence": result.confidence,
                        "queued": result.queued,
                        "classifier": result.classifier,
                        "error": result.error or "",
                        "subject": mail["subject"],
                    },
                )
            )
    return records


def extra_summary(records: list[ScenarioRecord]) -> str:
    graded = [r for r in records if r.case_id != "load-categories"]
    queued = [r for r in graded if r.detail.get("queued")]
    confidences = [r.detail["confidence"] for r in graded if r.detail.get("confidence") is not None]
    lines = [f"  queued (kNN gate failed): {len(queued)}/{len(graded)}"]
    if confidences:
        lines.append(
            f"  confidence: mean={sum(confidences) / len(confidences):.3f} "
            f"min={min(confidences):.3f} max={max(confidences):.3f}"
        )
    by_category: dict[str, list[ScenarioRecord]] = {}
    for record in graded:
        by_category.setdefault(record.expected, []).append(record)
    for category, recs in sorted(by_category.items()):
        passed = sum(1 for r in recs if r.verdict == "pass")
        lines.append(f"  {category}: {passed}/{len(recs)}")
    return "\n".join(lines)
