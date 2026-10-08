"""Generic cross-subsystem surface-feedback + suppression learning (issue 0028).

When IRIS proactively surfaces an item to the user — an email reply-followup, a
finance bill, a portfolio mover, a system-health alert, a low-trust research
source — the user can mark it "not useful". We record that verdict keyed by a
stable *suppression key* ``(subsystem, surface_kind, dims)`` and let each
subsystem consult :meth:`SurfaceFeedbackStore.should_suppress` before surfacing
similar items again.

This store is **subsystem-agnostic on purpose**. The proactive surfaces are
heterogeneous — Tasks (ADR-0073), read-time projections (system health), brief
items (finance bills, portfolio movers), and a rank signal (research trust) — so
a single spine that rides the Task/PendingAction seam would not reach all of
them. Instead every consult point calls this store directly.

Verdict vocabulary (``verdict`` column):

``not_useful``
    The user judged a surfaced item to be noise. One explicit ``not_useful``
    suppresses that exact key (``min_fp`` default 1).

``useful``
    A positive counter-signal — the surfaced item was wanted. Lifts suppression
    (counted against ``not_useful`` so a later "useful" un-suppresses a key).

Feedback is also mirrored into the digital-twin Layer-2 behavior signals
(``user_behavior_signals``) as ground truth for how the user steers IRIS.

Refs: a surfaced item carries a self-describing :func:`encode_ref` token so any
channel (CLI ``iris feedback``, ``POST /feedback``, chat) can record feedback on
it without a server-side lookup.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from iris_harness.foundation.logsafe import log_safe
from iris_harness.foundation.paths import data_dir
from iris_harness.foundation.persistence import sqlite_conn

logger = logging.getLogger(__name__)

# Verdict vocabulary. "not_useful" suppresses; "useful" is the positive
# counter-signal that lifts suppression.
NOT_USEFUL = "not_useful"
USEFUL = "useful"
VERDICTS = (NOT_USEFUL, USEFUL)

# Feedback provenance. "user" = explicit user verdict (strong); "auto" =
# heuristic/system-derived (weaker — callers can require a higher min_fp).
SOURCE_USER = "user"
SOURCE_AUTO = "auto"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS surface_feedback (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    subsystem    TEXT NOT NULL,
    surface_kind TEXT NOT NULL,
    key_hash     TEXT NOT NULL,
    dims_json    TEXT NOT NULL,
    verdict      TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'user',
    session_id   TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_surface_feedback_key
    ON surface_feedback(subsystem, surface_kind, key_hash);
"""


def _default_db_path() -> Path:
    """Resolve learning.db under the data dir (``paths.data_dir``; never Path.home()).

    Mirrors ``iris_harness.runtime.agent_console.data_dir`` so tests that relocate
    ``IRIS_DATA_DIR`` never touch the real profile (CLAUDE.md gotcha 0026).
    """
    return data_dir() / "learning.db"


def _canonical_dims(dims: Mapping[str, object]) -> dict[str, str]:
    """Lowercase + strip dims, dropping empty values, so trivially-different
    surfacings of the same item collapse onto one key."""
    out: dict[str, str] = {}
    for key, value in dims.items():
        if value is None:
            continue
        text = str(value).strip().lower()
        if not text:
            continue
        out[str(key).strip().lower()] = text
    return out


def _key_hash(subsystem: str, surface_kind: str, dims: Mapping[str, object]) -> str:
    payload = json.dumps(
        [subsystem.lower(), surface_kind.lower(), _canonical_dims(dims)],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]  # noqa: S324 — non-crypto id


def encode_ref(subsystem: str, surface_kind: str, dims: Mapping[str, object]) -> str:
    """Self-describing feedback token a surfaced item carries.

    ``fb:<base64url(json)>`` — decodes to ``(subsystem, surface_kind, dims)`` so
    any channel can record feedback without a server-side lookup. Stable across
    processes; no registry needed.
    """
    raw = json.dumps(
        {"s": subsystem, "k": surface_kind, "d": _canonical_dims(dims)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    token = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"fb:{token}"


def decode_ref(ref: str) -> tuple[str, str, dict[str, str]]:
    """Inverse of :func:`encode_ref`. Raises ValueError on a malformed token."""
    if not ref.startswith("fb:"):
        raise ValueError(f"not a feedback ref: {ref!r}")
    token = ref[3:]
    padding = "=" * (-len(token) % 4)
    try:
        raw = base64.urlsafe_b64decode(token + padding)
        data = json.loads(raw)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"malformed feedback ref: {ref!r}") from exc
    subsystem = str(data.get("s", ""))
    surface_kind = str(data.get("k", ""))
    dims = data.get("d", {})
    if not subsystem or not surface_kind or not isinstance(dims, dict):
        raise ValueError(f"incomplete feedback ref: {ref!r}")
    return subsystem, surface_kind, {str(k): str(v) for k, v in dims.items()}


@dataclass(frozen=True)
class SuppressionStat:
    """Aggregated feedback for one suppression key."""

    subsystem: str
    surface_kind: str
    dims: dict[str, str]
    not_useful: int
    useful: int

    @property
    def net(self) -> int:
        return self.not_useful - self.useful


@dataclass(frozen=True)
class SuppressionSummary:
    """Roll-up across the whole feedback ledger — for the context-health surface."""

    total_feedback: int  # every verdict ever recorded (in window)
    active_suppressions: int  # distinct keys currently suppressing (net >= min_fp)
    by_subsystem: dict[str, int]  # subsystem -> active-suppression count

    def as_dict(self) -> dict[str, object]:
        return {
            "total_feedback": self.total_feedback,
            "active_suppressions": self.active_suppressions,
            "by_subsystem": dict(self.by_subsystem),
        }


@dataclass(frozen=True)
class FeedbackEntry:
    """One recorded verdict, as :meth:`SurfaceFeedbackStore.feedback_between` reads it."""

    subsystem: str
    surface_kind: str
    dims: dict[str, str]
    verdict: str
    source: str
    at: datetime


@dataclass
class SurfaceFeedbackStore:
    """SQLite-backed feedback ledger keyed by ``(subsystem, surface_kind, dims)``.

    Lives in ``learning.db`` alongside the other steering signals. All writes are
    best-effort and never raise into the caller — recording feedback must not
    break a user action, and a suppression-store hiccup must not block surfacing.
    """

    db_path: Path | None = None

    def __post_init__(self) -> None:
        if self.db_path is None:
            self.db_path = _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def _db(self) -> Path:
        # __post_init__ guarantees db_path is set; narrows Path | None -> Path.
        assert self.db_path is not None
        return self.db_path

    def ensure_schema(self) -> None:
        with sqlite_conn(self._db) as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        subsystem: str,
        surface_kind: str,
        dims: Mapping[str, object],
        verdict: str,
        *,
        source: str = SOURCE_USER,
        session_id: str = "",
        emit_signal: bool = True,
    ) -> None:
        """Append one feedback verdict. Best-effort; never raises.

        On ``emit_signal`` (default), also mirrors the verdict into the
        digital-twin Layer-2 behavior signals as steering ground truth.
        """
        if verdict not in VERDICTS:
            logger.debug("ignoring unknown surface-feedback verdict %r", log_safe(verdict))
            return
        canonical = _canonical_dims(dims)
        key_hash = _key_hash(subsystem, surface_kind, canonical)
        dims_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        try:
            with sqlite_conn(self._db) as conn:
                conn.execute(
                    "INSERT INTO surface_feedback"
                    "(subsystem, surface_kind, key_hash, dims_json, verdict, source,"
                    " session_id, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        subsystem,
                        surface_kind,
                        key_hash,
                        dims_json,
                        verdict,
                        source,
                        session_id,
                        datetime.now(UTC).isoformat(),
                    ),
                )
        except Exception:  # feedback capture must never break a user action
            logger.debug(
                "surface-feedback capture failed (%s/%s)",
                log_safe(subsystem),
                log_safe(surface_kind),
                exc_info=True,
            )
            return
        if emit_signal:
            self._emit_layer2_signal(subsystem, surface_kind, dims_json, verdict, session_id)

    def _emit_layer2_signal(
        self, subsystem: str, surface_kind: str, dims_json: str, verdict: str, session_id: str
    ) -> None:
        """Mirror the verdict into user_behavior_signals (digital-twin L2)."""
        try:
            from iris_harness.services.learning.store import LearningMetricsStore

            store = LearningMetricsStore(db_path=self._db)
            store.ensure_schema()  # idempotent; the L2 table may not exist yet
            store.record_user_behavior_signal(
                f"surface_feedback_{verdict}",
                subject=f"{subsystem}/{surface_kind}",
                detail=dims_json,
                session_id=session_id,
            )
        except Exception:  # signal mirror is best-effort
            logger.debug("surface-feedback L2 signal mirror failed", exc_info=True)

    # ------------------------------------------------------------------
    # Read / consult
    # ------------------------------------------------------------------

    def should_suppress(
        self,
        subsystem: str,
        surface_kind: str,
        dims: Mapping[str, object],
        *,
        min_fp: int = 1,
        window_days: int | None = 365,
    ) -> bool:
        """True when this key has net ``not_useful`` feedback at/above ``min_fp``.

        ``net = #not_useful - #useful`` so a later "useful" verdict lifts a prior
        suppression. ``window_days=None`` considers all history.
        """
        not_useful, useful = self._counts(subsystem, surface_kind, dims, window_days=window_days)
        return (not_useful - useful) >= min_fp

    def summary(self, *, min_fp: int = 1, window_days: int | None = 365) -> SuppressionSummary:
        """Ledger-wide roll-up: total verdicts + how many keys are actively suppressing.

        A key is *actively suppressing* when its net (``#not_useful − #useful``) is at
        least ``min_fp`` — the same rule :meth:`should_suppress` applies per surface.
        Best-effort; returns zeros on a read error so the health surface never crashes.
        """
        sql = (
            "SELECT subsystem,"
            " SUM(CASE WHEN verdict = ? THEN 1 ELSE 0 END) AS nu,"
            " SUM(CASE WHEN verdict = ? THEN 1 ELSE 0 END) AS u,"
            " COUNT(*) AS total"
            " FROM surface_feedback"
        )
        params: list[object] = [NOT_USEFUL, USEFUL]
        if window_days is not None:
            cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
            sql += " WHERE created_at >= ?"
            params.append(cutoff)
        sql += " GROUP BY subsystem, surface_kind, key_hash"
        try:
            with sqlite_conn(self._db) as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error:
            logger.debug("surface-feedback summary read failed", exc_info=True)
            return SuppressionSummary(total_feedback=0, active_suppressions=0, by_subsystem={})
        total = 0
        active = 0
        by_subsystem: dict[str, int] = {}
        for subsystem, nu, u, key_total in rows:
            total += int(key_total)
            if int(nu) - int(u) >= min_fp:
                active += 1
                by_subsystem[str(subsystem)] = by_subsystem.get(str(subsystem), 0) + 1
        return SuppressionSummary(
            total_feedback=total, active_suppressions=active, by_subsystem=by_subsystem
        )

    def stat(
        self,
        subsystem: str,
        surface_kind: str,
        dims: Mapping[str, object],
        *,
        window_days: int | None = 365,
    ) -> SuppressionStat:
        not_useful, useful = self._counts(subsystem, surface_kind, dims, window_days=window_days)
        return SuppressionStat(
            subsystem=subsystem,
            surface_kind=surface_kind,
            dims=_canonical_dims(dims),
            not_useful=not_useful,
            useful=useful,
        )

    def _counts(
        self,
        subsystem: str,
        surface_kind: str,
        dims: Mapping[str, object],
        *,
        window_days: int | None,
    ) -> tuple[int, int]:
        key_hash = _key_hash(subsystem, surface_kind, dims)
        sql = (
            "SELECT verdict, COUNT(*) FROM surface_feedback "
            "WHERE subsystem = ? AND surface_kind = ? AND key_hash = ?"
        )
        params: list[object] = [subsystem, surface_kind, key_hash]
        if window_days is not None:
            cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
            sql += " AND created_at >= ?"
            params.append(cutoff)
        sql += " GROUP BY verdict"
        try:
            with sqlite_conn(self._db) as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error:
            logger.debug(
                "surface-feedback read failed (%s/%s)", subsystem, surface_kind, exc_info=True
            )
            return (0, 0)
        counts = {str(v): int(n) for v, n in rows}
        return counts.get(NOT_USEFUL, 0), counts.get(USEFUL, 0)

    def feedback_between(
        self, start: datetime, end: datetime, *, source: str | None = SOURCE_USER
    ) -> list[FeedbackEntry]:
        """Every verdict recorded in ``[start, end)``, oldest first.

        What the digest's ``learned yesterday`` line reads (loop-proof D17): the owner's
        verdicts of one local day. ``source`` narrows to explicit user verdicts by
        default; ``None`` returns every provenance. Best-effort: a read error is empty.
        """
        sql = (
            "SELECT subsystem, surface_kind, dims_json, verdict, source, created_at"
            " FROM surface_feedback WHERE created_at >= ? AND created_at < ?"
        )
        params: list[object] = [start.astimezone(UTC).isoformat(), end.astimezone(UTC).isoformat()]
        if source is not None:
            sql += " AND source = ?"
            params.append(source)
        sql += " ORDER BY id"
        try:
            with sqlite_conn(self._db) as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error:
            logger.debug("surface-feedback window read failed", exc_info=True)
            return []
        entries: list[FeedbackEntry] = []
        for subsystem, surface_kind, dims_json, verdict, src, created_at in rows:
            try:
                dims = {str(k): str(v) for k, v in json.loads(dims_json).items()}
            except (ValueError, AttributeError):
                dims = {}
            entries.append(
                FeedbackEntry(
                    subsystem=str(subsystem),
                    surface_kind=str(surface_kind),
                    dims=dims,
                    verdict=str(verdict),
                    source=str(src),
                    at=datetime.fromisoformat(str(created_at)),
                )
            )
        return entries


# ── Email surface keys ───────────────────────────────────────────────────────
#
# The email surfaces a user can answer "not useful" on, and the dimensions that key
# them. The vocabulary is email's (``iris_personal.email.feedback_keys``, which email's
# own readers use); this copy is for the core paths that still RECORD a verdict until
# PR 7 of the core/SDK boundary plan moves them to email: the `record_feedback` ReAct
# tool and `iris feedback` (both rebuild a followup's key from a core Task's
# ``wait_for`` payload, ``kind="reply_from"``), the digest's ``/not-useful`` route and
# the "learned yesterday" phrases. A verdict recorded here must hide what email reads,
# so the two copies are pinned equal by
# ``tests/unit/iris_personal/test_email/test_feedback_keys.py``; PR 7 deletes this one.

EMAIL_SEARCH_SUBSYSTEM = "email"
EMAIL_SEARCH_SURFACE = "search_result"
EMAIL_FOLLOWUP_SURFACE = "followup"
# A line of the morning digest's Focus section (👎 "not useful" hides the sender).
EMAIL_FOCUS_SURFACE = "focus"


def _domain_of(value: str) -> str:
    """The domain in a bare domain, an address, or a 'Name <addr>' pair."""
    s = value.strip()
    if "<" in s and ">" in s:
        s = s[s.rfind("<") + 1 : s.rfind(">")]
    return (s.rsplit("@", 1)[-1] if "@" in s else s).strip().lower()


def email_search_dims(from_domain: str) -> dict[str, str]:
    """Sender-scoped suppression key for an email search result."""
    return {"from_domain": (from_domain or "").strip().lower()}


def email_search_dims_from_sender(sender: str) -> dict[str, str]:
    """Search suppression key from a user-named sender — a bare domain
    ('acme.com'), an address ('x@acme.com'), or 'Name <x@acme.com>'."""
    return email_search_dims(_domain_of(sender))


def _address_of(value: str) -> str:
    """The bare address in an address or a 'Name <addr>' pair, lowercased."""
    s = value.strip()
    if "<" in s and ">" in s:
        s = s[s.rfind("<") + 1 : s.rfind(">")]
    return s.strip().lower()


def email_focus_dims(sender: str) -> dict[str, str]:
    """Suppression key for a line in the digest's Focus section (loop-proof D17).

    Keyed on the sender's full address, not the domain: a bank's marketing sender and
    its relationship-manager sender share a domain, and hiding one must not hide the
    other. Accepts a bare address or 'Name <addr>'.
    """
    return {"sender": _address_of(sender)}


def email_followup_dims_from(account_id: str, from_value: str) -> dict[str, str]:
    """Suppression-key dimensions for an email followup, from raw fields.

    Keyed on ``account · from_domain`` — who it's from, not the per-message subject or
    topical category. So a user "not useful" on one Quant Academy blast suppresses the
    rest, and the key is reconstructible from a followup Task's ``wait_for`` payload
    (which carries account + from but not the category). The category root is handled
    separately by the actionable-root gate.
    """
    domain = _domain_of(from_value) if "@" in from_value else ""
    return {"account": account_id, "from_domain": domain}


__all__ = [
    "EMAIL_FOCUS_SURFACE",
    "EMAIL_FOLLOWUP_SURFACE",
    "EMAIL_SEARCH_SUBSYSTEM",
    "EMAIL_SEARCH_SURFACE",
    "NOT_USEFUL",
    "USEFUL",
    "VERDICTS",
    "SOURCE_USER",
    "SOURCE_AUTO",
    "FeedbackEntry",
    "SuppressionStat",
    "SuppressionSummary",
    "SurfaceFeedbackStore",
    "encode_ref",
    "decode_ref",
    "email_focus_dims",
    "email_followup_dims_from",
    "email_search_dims",
    "email_search_dims_from_sender",
]
