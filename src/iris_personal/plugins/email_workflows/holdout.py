"""Held-out labeled emails for kNN-gate measurement (Track 1L / ADR-0023).

The holdout is the user's ground-truth signal: a small set of emails the
user has manually labeled with the correct root category. Track 1L's
``iris email knn-gate`` consumes this set to measure pure-kNN gate
accuracy and recommend threshold values.

Lives at ``$IRIS_HOME/workspace/email/<account_slug>/holdout-labels.jsonl``
per ADR-0023 §1. Append-only JSONL, one ``HoldoutLabel`` per line.
Idempotent on (account_id, message_id) — re-labeling overwrites in
practice (last write wins), enforced at append time by skipping
already-labeled ids unless the caller passes ``force=True``.

Schema is intentionally denormalized (carries from_address, subject,
snippet, received_at) so labels survive even if the email.db row ages
out or the user moves accounts.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from iris_personal.email.category_store import ALLOWED_ROOTS

# Allowed labels are the 14 ADR-0022 roots. Duplicated here rather than
# imported through CategoryStore to keep iris_personal.plugins.email_workflows.holdout independent
# of the categories table (the holdout is a labeling artifact, not a
# downstream consumer of the taxonomy).
ALLOWED_HOLDOUT_LABELS = ALLOWED_ROOTS


def _account_slug_for_path(account_id: str) -> str:
    """Translate ``gmail:user@gmail.com`` → ``gmail-user-at-gmail.com``.

    Matches the bootstrap-categories / triage-batch slug used elsewhere.
    """
    return account_id.replace(":", "-").replace("@", "-at-")


def holdout_path(workspace_dir: Path, account_id: str) -> Path:
    """Canonical workspace path per ADR-0023 §1."""
    return workspace_dir / "email" / _account_slug_for_path(account_id) / "holdout-labels.jsonl"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_received_at(raw: object) -> datetime:
    """Tolerant date parse. Spike data uses IMAP-style strings (RFC 2822
    via email.utils); the production fetch uses ISO 8601. Try both.
    """
    if not isinstance(raw, str) or not raw:
        return _utc_now()
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return _utc_now()


class HoldoutLabel(BaseModel):
    """One user-labeled email in the held-out set."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    # ─── Identity ────────────────────────────────────────────────────
    message_id: str = Field(..., min_length=1)
    account_id: str = Field(..., min_length=1)

    # ─── Label ───────────────────────────────────────────────────────
    true_root: str = Field(..., min_length=1)
    label_source: str = Field(default="user")  # 'user' | 'imported-from-spike'

    # ─── Denormalized envelope ───────────────────────────────────────
    from_address: str = Field(..., min_length=1)
    from_domain: str | None = None
    subject: str = ""
    snippet: str = ""
    received_at: datetime

    # ─── Lifecycle ───────────────────────────────────────────────────
    labeled_at: datetime = Field(default_factory=_utc_now)

    @field_validator("true_root")
    @classmethod
    def _check_root(cls, v: str) -> str:
        if v not in ALLOWED_HOLDOUT_LABELS:
            raise ValueError(f"true_root must be one of {ALLOWED_HOLDOUT_LABELS}: got {v!r}")
        return v


# ─── Store ───────────────────────────────────────────────────────────────────


def load_holdout(path: Path) -> list[HoldoutLabel]:
    """Read the holdout JSONL. Missing file → empty list."""
    if not path.exists():
        return []
    out: list[HoldoutLabel] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(HoldoutLabel.model_validate_json(line))
    return out


def append_label(path: Path, label: HoldoutLabel, *, force: bool = False) -> bool:
    """Append one label to the JSONL.

    Returns True if the label was written, False if a label for
    ``(account_id, message_id)`` already exists and ``force=False``.

    Idempotent in practice: the loader returns the *last* row written
    for each (account_id, message_id) when consumers de-dup. Use
    ``force=True`` to re-label an email (the previous label is left
    in the file as audit trail).
    """
    if not force:
        existing = load_holdout(path)
        for prior in existing:
            if prior.account_id == label.account_id and prior.message_id == label.message_id:
                return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(label.model_dump_json() + "\n")
    return True


def labeled_message_ids(path: Path, account_id: str) -> set[str]:
    """Return the set of ``message_id`` already labeled for one account."""
    return {label.message_id for label in load_holdout(path) if label.account_id == account_id}


# ─── Spike import ────────────────────────────────────────────────────────────

# Mapping from the spike's 6 flat labels to ADR-0022 roots.
# See ADR-0023 §3.
SPIKE_LABEL_TO_ROOT: dict[str, str] = {
    "personal": "personal",
    "marketing": "shopping",
    "social": "social",
    "news-digest": "news",
    "finance": "finance",
    "learning": "learning",
}


def import_from_spike(
    spike_path: Path,
    *,
    account_id: str,
    target_path: Path,
    skip_skipped: bool = True,
) -> tuple[int, int]:
    """One-shot import of ``data/spike/phase1_emails_labeled.jsonl`` (flat
    schema with ``uid``, ``user_category``) into a HoldoutLabel JSONL.

    The spike rows lack a Gmail message id; we synthesize a stable id
    of the form ``imap:<uid>`` so the holdout entry stays addressable.
    Rows with ``user_category=None`` (user marked them skipped during
    the spike) are excluded unless ``skip_skipped=False``.

    Returns ``(imported, skipped)``.
    """
    if not spike_path.exists():
        raise FileNotFoundError(f"spike file not found: {spike_path}")

    imported = 0
    skipped = 0
    for line in spike_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row: dict[str, Any] = json.loads(line)
        spike_label = row.get("user_category")
        if spike_label in (None, "<skipped>"):
            if skip_skipped:
                skipped += 1
                continue
        if spike_label not in SPIKE_LABEL_TO_ROOT:
            skipped += 1
            continue
        true_root = SPIKE_LABEL_TO_ROOT[spike_label]
        uid = row.get("uid")
        if not uid:
            skipped += 1
            continue
        received_at = _parse_received_at(row.get("received_at"))
        label = HoldoutLabel(
            message_id=f"imap:{uid}",
            account_id=account_id,
            true_root=true_root,
            label_source="imported-from-spike",
            from_address=row.get("from_address") or "unknown@unknown",
            from_domain=row.get("from_domain"),
            subject=row.get("subject", ""),
            snippet=row.get("snippet", ""),
            received_at=received_at,
        )
        if append_label(target_path, label):
            imported += 1
        else:
            skipped += 1
    return imported, skipped
