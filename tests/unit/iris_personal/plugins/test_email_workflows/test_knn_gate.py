"""Unit tests for the kNN-gate measurement module (Track 1L / ADR-0023)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from iris_personal.email.category_store import Category, CategoryStore
from iris_personal.plugins.email_workflows.holdout import HoldoutLabel, holdout_path
from iris_personal.plugins.email_workflows.knn_gate import (
    DEFAULT_COS_MIN_GRID,
    DEFAULT_MARGIN_MIN_GRID,
    HoldoutPrediction,
    KnnGateRunner,
    SweepCell,
    pick_threshold,
    render_writeup,
    sweep_grid,
)

ACCOUNT = "gmail:user@gmail.com"


# ─── sweep_grid ─────────────────────────────────────────────────────────────


def _pred(*, true: str, pred: str, cos: float, margin: float, mid: str = "x") -> HoldoutPrediction:
    return HoldoutPrediction(
        message_id=mid,
        true_root=true,
        predicted_path=f"email/{pred}/x/y",
        predicted_root=pred,
        cos_top1=cos,
        margin=margin,
    )


def test_sweep_grid_counts_gated_and_correct() -> None:
    preds = [
        _pred(true="finance", pred="finance", cos=0.80, margin=0.10, mid="a"),
        _pred(true="finance", pred="finance", cos=0.75, margin=0.06, mid="b"),
        _pred(true="finance", pred="shopping", cos=0.72, margin=0.05, mid="c"),
        _pred(true="shopping", pred="shopping", cos=0.60, margin=0.03, mid="d"),
        _pred(true="shopping", pred="finance", cos=0.55, margin=0.01, mid="e"),
    ]
    sweep = sweep_grid(
        preds,
        cos_min_grid=(0.70,),
        margin_min_grid=(0.05,),
    )
    assert len(sweep) == 1
    cell = sweep[0]
    # Gated: a, b, c (all have cos≥0.70 AND margin≥0.05)
    assert cell.gated_count == 3
    # Correct: a, b (c was finance → shopping mispredicted)
    assert cell.gated_correct == 2
    assert cell.gated_accuracy == pytest.approx(2 / 3)
    assert cell.queue_rate == pytest.approx(2 / 5)


def test_sweep_grid_uses_default_grid_size() -> None:
    preds = [_pred(true="finance", pred="finance", cos=0.8, margin=0.1)]
    sweep = sweep_grid(preds)
    assert len(sweep) == len(DEFAULT_COS_MIN_GRID) * len(DEFAULT_MARGIN_MIN_GRID)


def test_sweep_grid_empty_predictions_returns_zeroed_grid() -> None:
    sweep = sweep_grid([], cos_min_grid=(0.7,), margin_min_grid=(0.05,))
    assert len(sweep) == 1
    cell = sweep[0]
    assert cell.gated_count == 0
    assert cell.queue_rate == 1.0


# ─── pick_threshold ─────────────────────────────────────────────────────────


def _cell(cos: float, margin: float, total: int, gated: int, correct: int) -> SweepCell:
    return SweepCell(
        cos_min=cos,
        margin_min=margin,
        total=total,
        gated_count=gated,
        gated_correct=correct,
        gated_accuracy=(correct / gated) if gated else 0.0,
        queue_rate=((total - gated) / total) if total else 1.0,
    )


def test_pick_threshold_max_accuracy_subject_to_gated_floor() -> None:
    sweep = [
        # 10 total, all combinations gate ≥5 (meets the 50% floor)
        _cell(0.5, 0.02, 10, 10, 7),  # 70%
        _cell(0.7, 0.05, 10, 6, 5),  # 83.3%
        _cell(0.8, 0.20, 10, 5, 5),  # 100%
    ]
    best, note = pick_threshold(sweep)
    assert best.cos_min == 0.8
    assert best.margin_min == 0.20
    assert note == ""


def test_pick_threshold_excludes_below_floor() -> None:
    """A combination with 100% accuracy but only 30% gated is rejected."""
    sweep = [
        _cell(0.5, 0.02, 10, 10, 7),  # 70%, gated 100%
        _cell(0.9, 0.30, 10, 3, 3),  # 100% but gated only 30%
    ]
    best, _ = pick_threshold(sweep)
    assert best.cos_min == 0.5  # the only one meeting the floor


def test_pick_threshold_widens_when_no_combination_meets_floor() -> None:
    """All combinations have gated < 50% → relax and pick the loosest."""
    sweep = [
        _cell(0.7, 0.05, 10, 3, 2),  # 30% gated, 66% accuracy
        _cell(0.8, 0.20, 10, 1, 1),  # 10% gated, 100% accuracy
    ]
    best, note = pick_threshold(sweep)
    assert best.cos_min == 0.7  # loosest
    assert "no combination meets" in note


def test_pick_threshold_raises_on_empty_sweep() -> None:
    with pytest.raises(ValueError, match="empty sweep"):
        pick_threshold([])


# ─── End-to-end via KnnGateRunner ────────────────────────────────────────────


def _seed_holdout(workspace: Path, labels: list[HoldoutLabel]) -> Path:
    path = holdout_path(workspace, ACCOUNT)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for lbl in labels:
            f.write(lbl.model_dump_json() + "\n")
    return path


def _seed_proposals(workspace: Path, account_id: str, proposals: list[dict]) -> None:
    import json

    slug = account_id.replace(":", "-").replace("@", "-at-")
    path = workspace / "email" / slug / "proposals.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in proposals:
            f.write(json.dumps(p) + "\n")


def _proposal(
    root: str, branch: str, leaf: str, *, rep_domain: str, cohesion: float = 0.85
) -> dict:
    return {
        "cluster_id": hash(f"{root}/{branch}/{leaf}") % 10000,
        "size": 10,
        "cohesion": cohesion,
        "top_domains": [[rep_domain, 10]],
        "representatives": [
            {
                "id": f"rep-{i}",
                "subject": f"subject {i}",
                "from": f"sender@{rep_domain}",
                "snippet": f"snip {i}",
            }
            for i in range(3)
        ],
        "member_ids": [f"member-{i}" for i in range(10)],
        "proposed_root": root,
        "proposed_branch": branch,
        "proposed_leaf": leaf,
        "naming_rationale": "stub",
    }


def _seed_categories(db_path: Path, paths: list[str]) -> None:
    store = CategoryStore(db_path=db_path)
    store.ensure_schema()
    for path in paths:
        type_, root, branch, leaf = path.split("/")
        store.upsert_if_new(
            Category(
                path=path,
                type=type_,
                root=root,
                branch=branch,
                leaf=leaf,
                account_id=ACCOUNT,
                cohesion=0.85,
            )
        )


def _stub_embedder(domain_to_vec: dict[str, np.ndarray]):  # type: ignore[no-untyped-def]
    def _stub(texts: list[str], model_name: str = "stub") -> np.ndarray:
        out: list[np.ndarray] = []
        for t in texts:
            chosen = None
            for k, v in domain_to_vec.items():
                if k in t:
                    chosen = v
                    break
            if chosen is None:
                chosen = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            out.append(chosen.astype(np.float32))
        arr = np.stack(out)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-12)

    return _stub


def test_runner_measure_end_to_end(tmp_path: Path) -> None:
    """Stub embedder + 4 labeled emails → MeasurementReport with sweep
    grid + recommended thresholds + confusion matrix."""
    workspace = tmp_path / "ws"
    _seed_proposals(
        workspace,
        ACCOUNT,
        [
            _proposal("shopping", "apparel", "gap", rep_domain="gap.com"),
            _proposal("finance", "investing", "bonds", rep_domain="bonds.com"),
        ],
    )
    _seed_categories(
        tmp_path / "iris.db",
        ["email/shopping/apparel/gap", "email/finance/investing/bonds"],
    )

    labels = [
        HoldoutLabel(
            message_id=f"m-{i}",
            account_id=ACCOUNT,
            true_root=true,
            from_address=f"x@{dom}",
            from_domain=dom,
            subject="x",
            snippet="y",
            received_at=datetime(2026, 5, 1, tzinfo=UTC),
        )
        for i, (true, dom) in enumerate(
            [
                ("shopping", "gap.com"),
                ("shopping", "gap.com"),
                ("finance", "bonds.com"),
                ("finance", "bonds.com"),
            ]
        )
    ]
    _seed_holdout(workspace, labels)

    runner = KnnGateRunner(
        workspace_dir=workspace,
        db_path=tmp_path / "iris.db",
        embedder=_stub_embedder(
            {
                "gap.com": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "bonds.com": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            }
        ),
    )
    report = runner.measure(ACCOUNT)

    assert report.total_labels == 4
    assert report.label_distribution == {"shopping": 2, "finance": 2}
    # All 4 emails align with their target centroid (cos ≈ 1.0, margin ≈ 1.0)
    # → 100% accuracy at every threshold combination
    for cell in report.sweep:
        assert cell.gated_accuracy == 1.0
    # Confusion is a clean diagonal
    assert report.confusion == {
        "shopping": {"shopping": 2},
        "finance": {"finance": 2},
    }


def test_runner_raises_without_labels(tmp_path: Path) -> None:
    runner = KnnGateRunner(workspace_dir=tmp_path / "ws", db_path=tmp_path / "iris.db")
    with pytest.raises(ValueError, match="no labels"):
        runner.measure(ACCOUNT)


def test_runner_raises_without_categories(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    label = HoldoutLabel(
        message_id="m-1",
        account_id=ACCOUNT,
        true_root="shopping",
        from_address="x@gap.com",
        from_domain="gap.com",
        subject="x",
        snippet="y",
        received_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    _seed_holdout(workspace, [label])
    # Categories DB exists but no rows for this account
    CategoryStore(db_path=tmp_path / "iris.db").ensure_schema()

    runner = KnnGateRunner(workspace_dir=workspace, db_path=tmp_path / "iris.db")
    with pytest.raises(ValueError, match="no active categories"):
        runner.measure(ACCOUNT)


# ─── render_writeup ─────────────────────────────────────────────────────────


# ─── corrections_as_holdout + --include-corrections (ADR-0024) ──────────────


def test_corrections_as_holdout_projects_history_rows(tmp_path: Path) -> None:
    """User corrections become synthetic HoldoutLabel rows."""
    from iris_personal.email.contracts import EmailMessage
    from iris_personal.email.store import EmailStore
    from iris_personal.plugins.email_workflows.knn_gate import corrections_as_holdout

    iris_db = tmp_path / "iris.db"
    email_db = tmp_path / "email.db"

    cat_store = CategoryStore(db_path=iris_db)
    cat_store.ensure_schema()
    cat_store.upsert_if_new(
        Category(
            path="email/finance/banking/northwind-savings",
            type="email",
            root="finance",
            branch="banking",
            leaf="northwind-savings",
            account_id=ACCOUNT,
        )
    )

    email_store = EmailStore(db_path=email_db)
    email_store.ensure_schema()
    email_store.upsert(
        EmailMessage(
            id="m-1",
            provider="gmail",
            account_id=ACCOUNT,
            from_address="bank@northwind.test",
            from_domain="northwind.test",
            subject="Statement",
            snippet="Your statement",
            received_at=datetime(2026, 5, 1, tzinfo=UTC),
        )
    )

    cat_store.record_correction(
        message_id="m-1",
        account_id=ACCOUNT,
        old_path="email/shopping/apparel/gap",
        new_path="email/finance/banking/northwind-savings",
        previous_classifier="pure-knn",
    )

    holdouts = corrections_as_holdout(ACCOUNT, category_store=cat_store, email_store=email_store)
    assert len(holdouts) == 1
    lbl = holdouts[0]
    assert lbl.message_id == "m-1"
    assert lbl.true_root == "finance"
    assert lbl.label_source == "user-classification-correction"
    assert lbl.from_address == "bank@northwind.test"


def test_corrections_as_holdout_drops_missing_emails(tmp_path: Path) -> None:
    """If a corrected message is no longer in email.db, skip it."""
    from iris_personal.email.store import EmailStore
    from iris_personal.plugins.email_workflows.knn_gate import corrections_as_holdout

    iris_db = tmp_path / "iris.db"
    cat_store = CategoryStore(db_path=iris_db)
    cat_store.ensure_schema()
    cat_store.upsert_if_new(
        Category(
            path="email/finance/banking/x",
            type="email",
            root="finance",
            branch="banking",
            leaf="x",
            account_id=ACCOUNT,
        )
    )
    cat_store.record_correction(
        message_id="ghost",
        account_id=ACCOUNT,
        old_path=None,
        new_path="email/finance/banking/x",
        previous_classifier=None,
    )

    email_store = EmailStore(db_path=tmp_path / "email.db")
    email_store.ensure_schema()  # but never seed 'ghost'

    holdouts = corrections_as_holdout(ACCOUNT, category_store=cat_store, email_store=email_store)
    assert holdouts == []


def test_runner_include_corrections_augments_holdout(tmp_path: Path) -> None:
    """When --include-corrections is set, corrections join the measurement set."""
    from iris_personal.email.contracts import EmailMessage
    from iris_personal.email.store import EmailStore

    workspace = tmp_path / "ws"
    _seed_proposals(
        workspace,
        ACCOUNT,
        [
            _proposal("shopping", "apparel", "gap", rep_domain="gap.com"),
            _proposal("finance", "banking", "northwind-savings", rep_domain="northwind.test"),
        ],
    )
    _seed_categories(
        tmp_path / "iris.db",
        ["email/shopping/apparel/gap", "email/finance/banking/northwind-savings"],
    )

    # No holdout JSONL — measurement should still find labels via corrections
    email_store = EmailStore(db_path=tmp_path / "email.db")
    email_store.ensure_schema()
    email_store.upsert(
        EmailMessage(
            id="m-correction",
            provider="gmail",
            account_id=ACCOUNT,
            from_address="bank@northwind.test",
            from_domain="northwind.test",
            subject="Statement",
            snippet="Your statement",
            received_at=datetime(2026, 5, 1, tzinfo=UTC),
        )
    )

    cat_store = CategoryStore(db_path=tmp_path / "iris.db")
    cat_store.record_correction(
        message_id="m-correction",
        account_id=ACCOUNT,
        old_path="email/shopping/apparel/gap",
        new_path="email/finance/banking/northwind-savings",
        previous_classifier="pure-knn",
    )

    runner = KnnGateRunner(
        workspace_dir=workspace,
        db_path=tmp_path / "iris.db",
        email_db_path=tmp_path / "email.db",
        embedder=_stub_embedder(
            {
                "gap.com": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "northwind.test": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            }
        ),
    )
    report = runner.measure(ACCOUNT, include_corrections=True)
    assert report.total_labels == 1
    assert report.label_distribution == {"finance": 1}


def test_render_writeup_contains_summary_sections() -> None:
    from iris_personal.plugins.email_workflows.knn_gate import MeasurementReport

    report = MeasurementReport(
        account_id=ACCOUNT,
        measured_at=datetime(2026, 5, 25, tzinfo=UTC),
        total_labels=4,
        label_distribution={"shopping": 2, "finance": 2},
        predictions=[],
        sweep=[_cell(0.7, 0.05, 4, 4, 4)],
        recommended_cos_min=0.7,
        recommended_margin_min=0.05,
        recommended_accuracy=1.0,
        recommended_gated_count=4,
        confusion={"shopping": {"shopping": 2}, "finance": {"finance": 2}},
    )
    md = render_writeup(report)
    assert "kNN-gate measurement" in md
    assert "Holdout size: 4" in md
    assert "shopping" in md
    assert "## Sweep grid" in md
    assert "## Recommendation" in md
    assert "## Confusion matrix" in md
    assert "0.70" in md  # recommended cos_min
