"""Tests for ``iris email accept-categories`` (Track 1F)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iris_harness.main import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _proposal(
    cluster_id: int = 0,
    root: str = "shopping",
    branch: str = "apparel",
    leaf: str = "outlet-brand",
    **extra,
) -> dict:
    base = {
        "cluster_id": cluster_id,
        "size": 15,
        "cohesion": 0.87,
        "top_domains": [["gap.com", 15]],
        "representatives": [],
        "member_ids": [f"m-{i}" for i in range(15)],
        "proposed_root": root,
        "proposed_branch": branch,
        "proposed_leaf": leaf,
        "naming_rationale": "dominated by gap.com",
    }
    base.update(extra)
    return base


def _write_proposals(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _invoke_accept(runner: CliRunner, *, account: str, proposals: Path, db: Path):  # type: ignore[no-untyped-def]
    return runner.invoke(
        app,
        [
            "email",
            "accept-categories",
            "--account",
            account,
            "--proposals",
            str(proposals),
            "--db-path",
            str(db),
        ],
    )


# ─── Happy path ─────────────────────────────────────────────────────────────


def test_accept_happy_path_persists_rows(runner: CliRunner, tmp_path: Path) -> None:
    """End-to-end: 3 proposals → 3 categories rows + 3 history rows."""
    from iris_personal.email.category_store import CategoryStore

    proposals = tmp_path / "proposals.jsonl"
    _write_proposals(
        proposals,
        [
            _proposal(cluster_id=0, root="shopping", branch="apparel", leaf="gap"),
            _proposal(cluster_id=1, root="finance", branch="investing", leaf="bonds"),
            _proposal(cluster_id=2, root="social", branch="facebook", leaf="updates"),
        ],
    )
    db = tmp_path / "iris.db"

    result = _invoke_accept(runner, account="gmail:user@gmail.com", proposals=proposals, db=db)
    assert result.exit_code == 0, result.output

    store = CategoryStore(db_path=db)
    rows = store.list()
    assert len(rows) == 3
    paths = {r.path for r in rows}
    assert paths == {
        "email/shopping/apparel/gap",
        "email/finance/investing/bonds",
        "email/social/facebook/updates",
    }
    # Each category has exactly one history row (the insert)
    for r in rows:
        history = store.history(r.path)
        assert len(history) == 1
        assert history[0]["op"] == "insert"
    # Output mentions "3 new" and at least one accepted path
    assert "3 new" in result.output
    assert "shopping" in result.output


def test_accept_is_idempotent_on_rerun(runner: CliRunner, tmp_path: Path) -> None:
    """Re-running with the same JSONL is a no-op — no new rows, no
    new history entries."""
    from iris_personal.email.category_store import CategoryStore

    proposals = tmp_path / "proposals.jsonl"
    _write_proposals(proposals, [_proposal()])
    db = tmp_path / "iris.db"

    # First run inserts
    r1 = _invoke_accept(runner, account="gmail:user@gmail.com", proposals=proposals, db=db)
    assert r1.exit_code == 0
    # Second run is a no-op
    r2 = _invoke_accept(runner, account="gmail:user@gmail.com", proposals=proposals, db=db)
    assert r2.exit_code == 0
    assert "0 new" in r2.output
    assert "1 unchanged" in r2.output

    store = CategoryStore(db_path=db)
    assert len(store.list()) == 1
    assert len(store.history("email/shopping/apparel/outlet-brand")) == 1


# ─── Strict validation ──────────────────────────────────────────────────────


def test_accept_aborts_when_any_row_invalid(runner: CliRunner, tmp_path: Path) -> None:
    """Per ADR-0019 §5: a single invalid row → exit 4, NO writes to DB."""
    from iris_personal.email.category_store import CategoryStore

    proposals = tmp_path / "proposals.jsonl"
    _write_proposals(
        proposals,
        [
            _proposal(cluster_id=0, root="shopping", branch="apparel", leaf="gap"),
            _proposal(cluster_id=1, root="retail", branch="apparel", leaf="shopmart"),  # bad root
            _proposal(cluster_id=2, root="social", branch="facebook", leaf="updates"),
        ],
    )
    db = tmp_path / "iris.db"

    result = _invoke_accept(runner, account="gmail:user@gmail.com", proposals=proposals, db=db)
    assert result.exit_code == 4, result.output
    # The store may or may not exist depending on if the schema file is created.
    # Assertion: no rows were inserted.
    if db.exists():
        store = CategoryStore(db_path=db)
        store.ensure_schema()
        assert store.list() == []
    # Error output lists the offending cluster
    assert "cluster 1" in result.output
    assert "retail" in result.output


def test_accept_collects_all_errors_before_aborting(runner: CliRunner, tmp_path: Path) -> None:
    """The user should see EVERY invalid row in one pass, not stop on the first."""
    proposals = tmp_path / "proposals.jsonl"
    _write_proposals(
        proposals,
        [
            _proposal(cluster_id=0, root="bogus", branch="x", leaf="y"),
            _proposal(cluster_id=1, root="shopping", branch="", leaf="y"),
            _proposal(cluster_id=2, root="shopping", branch="x", leaf=""),
        ],
    )
    db = tmp_path / "iris.db"

    result = _invoke_accept(runner, account="gmail:user@gmail.com", proposals=proposals, db=db)
    assert result.exit_code == 4
    assert "3 invalid" in result.output
    assert "cluster 0" in result.output
    assert "cluster 1" in result.output
    assert "cluster 2" in result.output


# ─── Missing file / empty file ──────────────────────────────────────────────


def test_accept_exits_2_when_proposals_missing(runner: CliRunner, tmp_path: Path) -> None:
    proposals = tmp_path / "does-not-exist.jsonl"
    db = tmp_path / "iris.db"
    result = _invoke_accept(runner, account="gmail:user@gmail.com", proposals=proposals, db=db)
    assert result.exit_code == 2
    assert "not found" in result.output


def test_accept_exits_3_when_proposals_empty(runner: CliRunner, tmp_path: Path) -> None:
    proposals = tmp_path / "proposals.jsonl"
    proposals.write_text("\n\n   \n")  # just whitespace
    db = tmp_path / "iris.db"
    result = _invoke_accept(runner, account="gmail:user@gmail.com", proposals=proposals, db=db)
    assert result.exit_code == 3
    assert "empty" in result.output


def test_accept_exits_4_on_malformed_json(runner: CliRunner, tmp_path: Path) -> None:
    proposals = tmp_path / "proposals.jsonl"
    proposals.write_text('{"valid": true}\nnot-json\n{"valid": true}\n')
    db = tmp_path / "iris.db"
    result = _invoke_accept(runner, account="gmail:user@gmail.com", proposals=proposals, db=db)
    assert result.exit_code == 4
    assert "line 2" in result.output


# ─── Path D — duplicate-leaf collision detection (ADR-0020 amendment) ───────


def test_accept_exits_5_when_jsonl_has_path_collisions(runner: CliRunner, tmp_path: Path) -> None:
    """Two distinct clusters with identical (root, branch, leaf) must
    abort before any DB write (per ADR-0020 amendment / Path D)."""
    from iris_personal.email.category_store import CategoryStore

    proposals = tmp_path / "proposals.jsonl"
    _write_proposals(
        proposals,
        [
            # cluster 0 + 1 collide on email/learning/assignments/instructure
            _proposal(cluster_id=0, root="learning", branch="assignments", leaf="instructure"),
            _proposal(cluster_id=1, root="learning", branch="assignments", leaf="instructure"),
            # cluster 2 + 3 collide on email/shopping/apparel/dsw
            _proposal(cluster_id=2, root="shopping", branch="apparel", leaf="dsw"),
            _proposal(cluster_id=3, root="shopping", branch="apparel", leaf="dsw"),
            # cluster 4 is unique
            _proposal(cluster_id=4, root="social", branch="facebook", leaf="updates"),
        ],
    )
    db = tmp_path / "iris.db"

    result = _invoke_accept(runner, account="gmail:user@gmail.com", proposals=proposals, db=db)
    assert result.exit_code == 5, result.output

    # Crucially: NO DB writes happened. Even the unique cluster 4 is rejected
    # because Path D's strict-validation contract is all-or-nothing.
    if db.exists():
        store = CategoryStore(db_path=db)
        store.ensure_schema()
        assert store.list() == []

    assert "2 path collision" in result.output
    assert "email/learning/assignments/instructure" in result.output
    assert "email/shopping/apparel/dsw" in result.output
    # The cluster ids for each colliding pair are surfaced
    assert "0" in result.output and "1" in result.output
    assert "2" in result.output and "3" in result.output


def test_accept_succeeds_after_collision_is_disambiguated(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The user's expected workflow after Path D fires: edit the JSONL
    to differentiate the colliding leaves, then re-run successfully."""
    from iris_personal.email.category_store import CategoryStore

    proposals = tmp_path / "proposals.jsonl"
    _write_proposals(
        proposals,
        [
            _proposal(
                cluster_id=0,
                root="learning",
                branch="assignments",
                leaf="instructure-homework",
            ),
            _proposal(
                cluster_id=1,
                root="learning",
                branch="assignments",
                leaf="instructure-attendance",
            ),
        ],
    )
    db = tmp_path / "iris.db"

    result = _invoke_accept(runner, account="gmail:user@gmail.com", proposals=proposals, db=db)
    assert result.exit_code == 0, result.output

    store = CategoryStore(db_path=db)
    paths = {c.path for c in store.list()}
    assert paths == {
        "email/learning/assignments/instructure-homework",
        "email/learning/assignments/instructure-attendance",
    }
