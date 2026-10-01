"""Tests for the holdout-labeling I/O layer (Track 1L / ADR-0023)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from iris_personal.plugins.email_workflows.holdout import (
    ALLOWED_HOLDOUT_LABELS,
    SPIKE_LABEL_TO_ROOT,
    HoldoutLabel,
    _account_slug_for_path,
    append_label,
    holdout_path,
    import_from_spike,
    labeled_message_ids,
    load_holdout,
)


def _label(
    message_id: str = "msg-1",
    *,
    account_id: str = "gmail:user@gmail.com",
    true_root: str = "shopping",
    from_address: str = "x@gap.com",
) -> HoldoutLabel:
    return HoldoutLabel(
        message_id=message_id,
        account_id=account_id,
        true_root=true_root,
        from_address=from_address,
        from_domain=from_address.split("@", 1)[1],
        subject="some subject",
        snippet="some snippet",
        received_at=datetime(2026, 5, 25, tzinfo=UTC),
    )


# ─── HoldoutLabel contract ──────────────────────────────────────────────────


def test_label_minimal_valid() -> None:
    label = _label()
    assert label.true_root == "shopping"
    assert label.label_source == "user"  # default


def test_label_rejects_non_allowed_root() -> None:
    with pytest.raises(ValidationError, match="true_root must be one of"):
        _label(true_root="retail")


def test_label_is_frozen() -> None:
    label = _label()
    with pytest.raises(ValidationError):
        label.true_root = "finance"  # type: ignore[misc]


def test_label_serializable_via_model_dump_json() -> None:
    """Roundtrip through JSONL must preserve all fields."""
    label = _label(true_root="finance", from_address="cust@northwind.test")
    raw = label.model_dump_json()
    reconstructed = HoldoutLabel.model_validate_json(raw)
    assert reconstructed == label


# ─── Path helpers ───────────────────────────────────────────────────────────


def test_account_slug_for_path_matches_other_clis() -> None:
    """The slug must match what bootstrap-categories / triage-batch use."""
    assert _account_slug_for_path("gmail:user@gmail.com") == "gmail-user-at-gmail.com"


def test_holdout_path_anchors_under_workspace(tmp_path: Path) -> None:
    p = holdout_path(tmp_path, "gmail:a@b.com")
    assert p == tmp_path / "email" / "gmail-a-at-b.com" / "holdout-labels.jsonl"


# ─── Append + load ──────────────────────────────────────────────────────────


def test_append_creates_parent_dir(tmp_path: Path) -> None:
    p = tmp_path / "nested" / "deep" / "holdout-labels.jsonl"
    assert append_label(p, _label()) is True
    assert p.exists()


def test_append_skips_existing_unless_force(tmp_path: Path) -> None:
    p = tmp_path / "holdout-labels.jsonl"
    assert append_label(p, _label()) is True
    # Same (account_id, message_id) again → no-op
    assert append_label(p, _label(true_root="finance")) is False
    # But force=True writes another row
    assert append_label(p, _label(true_root="finance"), force=True) is True
    # File now has two rows
    assert len(p.read_text().splitlines()) == 2


def test_load_holdout_returns_empty_for_missing(tmp_path: Path) -> None:
    assert load_holdout(tmp_path / "does-not-exist.jsonl") == []


def test_load_holdout_returns_all_rows_in_order(tmp_path: Path) -> None:
    p = tmp_path / "holdout-labels.jsonl"
    append_label(p, _label(message_id="a"))
    append_label(p, _label(message_id="b", true_root="finance"))
    append_label(p, _label(message_id="c", true_root="social"))
    loaded = load_holdout(p)
    assert [lbl.message_id for lbl in loaded] == ["a", "b", "c"]


def test_labeled_message_ids_filters_by_account(tmp_path: Path) -> None:
    p = tmp_path / "holdout-labels.jsonl"
    append_label(p, _label(message_id="a", account_id="gmail:x@y.com"))
    append_label(p, _label(message_id="b", account_id="gmail:other@z.com"))
    assert labeled_message_ids(p, "gmail:x@y.com") == {"a"}
    assert labeled_message_ids(p, "gmail:other@z.com") == {"b"}


# ─── Spike import ───────────────────────────────────────────────────────────


def test_spike_label_mapping_covers_all_six(tmp_path: Path) -> None:
    """All 6 spike labels must map to a valid ADR-0022 root."""
    assert set(SPIKE_LABEL_TO_ROOT.keys()) == {
        "personal",
        "marketing",
        "social",
        "news-digest",
        "finance",
        "learning",
    }
    for root in SPIKE_LABEL_TO_ROOT.values():
        assert root in ALLOWED_HOLDOUT_LABELS


def test_import_from_spike_converts_rows(tmp_path: Path) -> None:
    spike_path = tmp_path / "phase1_emails_labeled.jsonl"
    rows = [
        {
            "uid": "100",
            "user_category": "finance",
            "from_address": "bank@example.com",
            "from_domain": "example.com",
            "subject": "Statement",
            "snippet": "Your statement is ready",
            "received_at": "2026-05-01T12:00:00+00:00",
        },
        {
            "uid": "101",
            "user_category": "marketing",
            "from_address": "shop@gap.com",
            "from_domain": "gap.com",
            "subject": "Sale",
            "snippet": "60% off",
            "received_at": "2026-05-02T12:00:00+00:00",
        },
    ]
    with spike_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    target = tmp_path / "holdout-labels.jsonl"
    imported, skipped = import_from_spike(
        spike_path, account_id="gmail:user@gmail.com", target_path=target
    )
    assert imported == 2
    assert skipped == 0

    loaded = load_holdout(target)
    assert {lbl.message_id for lbl in loaded} == {"imap:100", "imap:101"}
    # Mapping applied
    by_id = {lbl.message_id: lbl for lbl in loaded}
    assert by_id["imap:100"].true_root == "finance"
    assert by_id["imap:101"].true_root == "shopping"  # marketing → shopping
    # Source tag
    assert all(lbl.label_source == "imported-from-spike" for lbl in loaded)


def test_import_from_spike_skips_user_skipped_rows(tmp_path: Path) -> None:
    spike_path = tmp_path / "phase1.jsonl"
    rows = [
        {
            "uid": "200",
            "user_category": None,  # user skipped this one during the spike
            "from_address": "a@b.com",
            "from_domain": "b.com",
            "subject": "x",
            "snippet": "y",
            "received_at": "2026-05-01T00:00:00+00:00",
        },
        {
            "uid": "201",
            "user_category": "<skipped>",
            "from_address": "a@b.com",
            "from_domain": "b.com",
            "subject": "x",
            "snippet": "y",
            "received_at": "2026-05-01T00:00:00+00:00",
        },
        {
            "uid": "202",
            "user_category": "finance",
            "from_address": "bank@example.com",
            "from_domain": "example.com",
            "subject": "Statement",
            "snippet": "Your statement",
            "received_at": "2026-05-01T00:00:00+00:00",
        },
    ]
    with spike_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    target = tmp_path / "holdout.jsonl"
    imported, skipped = import_from_spike(
        spike_path, account_id="gmail:u@x.com", target_path=target
    )
    assert imported == 1  # only the finance row
    assert skipped == 2


def test_import_from_spike_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="spike file not found"):
        import_from_spike(
            tmp_path / "missing.jsonl",
            account_id="gmail:u@x.com",
            target_path=tmp_path / "out.jsonl",
        )


def test_import_from_spike_is_idempotent_on_rerun(tmp_path: Path) -> None:
    """Running import twice with the same source → second run imports 0."""
    spike_path = tmp_path / "phase1.jsonl"
    with spike_path.open("w") as f:
        f.write(
            json.dumps(
                {
                    "uid": "300",
                    "user_category": "finance",
                    "from_address": "bank@example.com",
                    "from_domain": "example.com",
                    "subject": "x",
                    "snippet": "y",
                    "received_at": "2026-05-01T00:00:00+00:00",
                }
            )
            + "\n"
        )

    target = tmp_path / "holdout.jsonl"
    first = import_from_spike(spike_path, account_id="gmail:u@x.com", target_path=target)
    assert first == (1, 0)
    second = import_from_spike(spike_path, account_id="gmail:u@x.com", target_path=target)
    assert second == (0, 1)
