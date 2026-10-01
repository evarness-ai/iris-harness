"""The proof bundle (OSS plan R14): the audit ledger's evidence for three invariants,
in one versioned JSON document that verifies offline.

1. ``no-private-to-cloud`` -- no model call whose data was classified ``personal`` or
   ``secret`` was allowed to a cloud tier.
2. ``mailbox-write-approved`` -- no mailbox write without an approved approval row for
   that account: the latest ``mailbox_writes`` row (approval or revoke) at or before
   the write's time must be an approval, so a write after a revoke fails.
3. ``every-call-and-answer-audited`` -- every model call and every answer has an audit
   row.

A bundle holds two kinds of evidence, kept apart on purpose:

* ``ledger`` -- the audit rows in scope, cut down to ids, timestamps, hook points,
  decisions, classifications, tiers and keyed digests. Never a reason, a prompt, an
  argument or a result: the row schema is closed, and ``verify`` rejects any other key.
* ``observations`` -- what happened, recorded at the point it happened rather than at
  a governance decision: the model calls and answers the session logs saw
  (``llm_call`` / ``agent_response`` events, written by the transport side after a call
  returns), and the mailbox writes -- each a ``mailbox_write_performed`` ledger row the
  email library writes once a provider's write has reached the mailbox, read here as an
  observation (plus any a caller reports as JSON). Invariants 2 and 3 are claims about
  these against the ledger's decisions; a bundle with no observations proves only
  invariant 1.

Identifiers that name a person (a mailbox account) are pseudonymised with a key made
for the export and thrown away after it: equal within one bundle, so the checks can
match an approval to a write, and linkable to nothing outside it.

``content_sha256`` is the SHA-256 of the canonical JSON of everything else, so an edit
that does not recompute it fails ``verify``. It is integrity, not authenticity: whoever
edits a bundle can recompute it. A signed bundle is a later format version.

The JSON schema is ``proof_bundle.schema.json`` beside this module; ``verify`` checks
the same closed structure in code (no schema library at runtime). The format is
documented in ``docs/reference/proof-bundle.md``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from iris_harness.kernel.governance.audit.log import AuditLog, AuditRow

FORMAT = "iris-proof-bundle"
SCHEMA_VERSION = 1
SCHEMA_FILE = Path(__file__).with_name("proof_bundle.schema.json")

#: Model tiers as the ledger names them: ``tier_3`` is the cloud tier (a provider that
#: ``runs: cloud``). Fixed by format version 1, never read from the bundle.
CLOUD_TIERS = frozenset({"tier_3"})
#: Data that must never reach a cloud tier.
PRIVATE = frozenset({"personal", "secret"})

LLM_CALL_HOOK = "pre_llm_call"
ANSWER_HOOK = "pre_response"
#: Where the owner's approval of an account's mailbox writes lands (the email plugin's
#: ``write_approvals.AUDIT_HOOK``; the core names it, it does not import it).
MAILBOX_WRITES_HOOK = "mailbox_writes"
#: Where the email library records a write that reached a mailbox (its
#: ``write_approvals.WRITE_HOOK``): the payload's ``account`` and ``count``. Read as a
#: mailbox-write observation, never as an approval.
MAILBOX_WRITE_PERFORMED_HOOK = "mailbox_write_performed"

NO_PRIVATE_TO_CLOUD = "no-private-to-cloud"
MAILBOX_WRITE_APPROVED = "mailbox-write-approved"
EVERY_CALL_AUDITED = "every-call-and-answer-audited"

#: The three statements, verbatim. A bundle carries them so a reader knows what it
#: proves; ``verify`` checks them against these and fails a bundle that altered one.
INVARIANTS: Mapping[str, str] = {
    NO_PRIVATE_TO_CLOUD: (
        "No model call whose data was classified personal or secret was allowed to a "
        "cloud tier (tier_3)."
    ),
    MAILBOX_WRITE_APPROVED: (
        "No mailbox write without an approved approval row: every account written to has "
        "a mailbox_writes approval in the ledger, recorded before the write when the write "
        "has a time."
    ),
    EVERY_CALL_AUDITED: (
        "Every model call and every answer has an audit row: at least one pre_llm_call "
        "(run_id, step_id) per model call observed in a session, and a pre_response row in "
        "each session that answered."
    ),
}

# The closed row and observation shapes (the schema file says the same).
_ROW_FIELDS: Mapping[str, tuple[type, ...]] = {
    "id": (int,),
    "ts": (str,),
    "run_id": (str,),
    "step_id": (int, type(None)),
    "agent_type": (str,),
    "hook_point": (str,),
    "plugin": (str,),
    "decision": (str,),
    "classification": (str, type(None)),
    "tier": (str, type(None)),
    "severity": (str,),
    "session_id": (str, type(None)),
    "tool_name": (str, type(None)),
    "account_ref": (str, type(None)),
    "args_digest": (str, type(None)),
    "result_digest": (str, type(None)),
    "digest_alg": (str, type(None)),
}
_OBSERVATION_FIELDS: Mapping[str, Mapping[str, tuple[type, ...]]] = {
    "model_calls": {"session_id": (str, type(None)), "tier": (str, type(None))},
    "answers": {"session_id": (str,)},
    "mailbox_writes": {"account_ref": (str,), "count": (int,), "at": (str, type(None))},
}
_TOP_FIELDS = (
    "format",
    "schema_version",
    "subject",
    "created_at",
    "producer",
    "scope",
    "invariants",
    "ledger",
    "observations",
    "content_sha256",
)


class ProofBundleError(ValueError):
    """A bundle that cannot be read at all (not JSON, not an object)."""


@dataclass(frozen=True)
class MailboxWrite:
    """A mailbox write the caller observed: ``count`` changes to ``account``, at ``at``
    (ISO 8601) when known. ``account`` is pseudonymised on export."""

    account: str
    count: int
    at: str | None = None


@dataclass(frozen=True)
class Observations:
    """Evidence from outside the ledger, for invariants 2 and 3.

    ``model_calls`` are ``(session_id, tier)`` pairs, one per model call that returned
    (``session_id`` None when the observer knows no session); ``answers`` the session ids
    that produced an answer; ``mailbox_writes`` what reached a mailbox.
    """

    model_calls: tuple[tuple[str | None, str | None], ...] = ()
    answers: tuple[str, ...] = ()
    mailbox_writes: tuple[MailboxWrite, ...] = ()

    def __add__(self, other: Observations) -> Observations:
        return Observations(
            model_calls=self.model_calls + other.model_calls,
            answers=self.answers + other.answers,
            mailbox_writes=self.mailbox_writes + other.mailbox_writes,
        )


@dataclass(frozen=True)
class Violation:
    """One way a bundle failed: the invariant (or ``format`` / ``integrity``) and why."""

    invariant: str
    detail: str

    def __str__(self) -> str:
        return f"{self.invariant}: {self.detail}"


@dataclass
class _Pseudonyms:
    """Account ids -> refs under a key made for one export and never stored."""

    key: bytes = field(default_factory=lambda: secrets.token_bytes(32))

    def ref(self, value: str) -> str:
        return "acct-" + hmac.new(self.key, value.encode(), hashlib.sha256).hexdigest()[:24]


# --------------------------------------------------------------------------- observe


def observations_from_session_logs(
    log_dir: Path,
    *,
    since: datetime | str | None = None,
    until: datetime | str | None = None,
    sessions: Iterable[str] | None = None,
) -> Observations:
    """The model calls and answers the session logs under ``log_dir`` recorded.

    Reads only each event's ``kind``, ``session_id``, ``ts`` and ``tier`` -- never its
    text. ``sessions`` narrows to those session ids.
    """
    lo, hi = _when(since), _when(until)
    wanted = set(sessions) if sessions is not None else None
    calls: list[tuple[str | None, str | None]] = []
    answers: list[str] = []
    if not log_dir.is_dir():
        return Observations()
    for path in sorted(log_dir.glob("session-*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("kind")
            if kind not in ("llm_call", "agent_response"):
                continue
            session = event.get("session_id")
            if not isinstance(session, str) or (wanted is not None and session not in wanted):
                continue
            ts = _when(event.get("ts"))
            if ts is None and (lo is not None or hi is not None):
                continue
            if ts is not None and ((lo is not None and ts < lo) or (hi is not None and ts > hi)):
                continue
            if kind == "llm_call":
                tier = event.get("tier")
                calls.append((session, str(tier) if tier is not None else None))
            else:
                answers.append(session)
    return Observations(model_calls=tuple(calls), answers=tuple(answers))


def observations_from_json(raw: Mapping[str, Any]) -> Observations:
    """Observations a caller wrote as JSON: ``{"model_calls": [{"session_id", "tier"}],
    "answers": [{"session_id"}], "mailbox_writes": [{"account", "count", "at"}]}``."""
    calls = tuple(
        (_opt_str(item.get("session_id")), _opt_str(item.get("tier")))
        for item in raw.get("model_calls") or ()
    )
    answers = tuple(str(item["session_id"]) for item in raw.get("answers") or ())
    writes = tuple(
        MailboxWrite(
            account=str(item["account"]),
            count=int(item["count"]),
            at=_opt_str(item.get("at")),
        )
        for item in raw.get("mailbox_writes") or ()
    )
    return Observations(model_calls=calls, answers=answers, mailbox_writes=writes)


# ---------------------------------------------------------------------------- export


def export_bundle(
    ledger: AuditLog | Path,
    *,
    since: datetime | str | None = None,
    until: datetime | str | None = None,
    run_ids: Sequence[str] = (),
    observations: Observations | None = None,
    subject: str = "",
) -> dict[str, Any]:
    """The proof bundle for the ledger rows in scope, with ``observations``.

    Scope: rows at or after ``since`` and at or before ``until``, of ``run_ids`` when
    given. Every ``mailbox_writes`` approval row up to ``until`` is included whatever
    the scope -- an approval granted before the window still authorises a write in it.
    Each ``mailbox_write_performed`` row in scope is also a mailbox-write observation
    (its account, count and time), added to ``observations``.
    """
    log = ledger if isinstance(ledger, AuditLog) else AuditLog(db_path=ledger)
    rows = _rows_in_scope(log, since=since, until=until, run_ids=run_ids)
    seen = {row.id for row in rows}
    approvals = [
        row
        for row in log.query(until=until)
        if row.hook_point == MAILBOX_WRITES_HOOK and row.id not in seen
    ]
    names = _Pseudonyms()
    ledger_rows = [_bundle_row(row, names) for row in sorted(rows + approvals, key=_order)]
    seen_obs = (observations or Observations()) + _ledger_writes(rows)
    body: dict[str, Any] = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "subject": subject,
        "created_at": datetime.now(UTC).isoformat(),
        "producer": {"name": "iris-harness", "version": _version()},
        "scope": {
            "since": _iso_or_none(since),
            "until": _iso_or_none(until),
            "run_ids": list(run_ids),
        },
        "invariants": [{"id": key, "statement": text} for key, text in INVARIANTS.items()],
        "ledger": ledger_rows,
        "observations": {
            "model_calls": [
                {"session_id": session, "tier": tier} for session, tier in seen_obs.model_calls
            ],
            "answers": [{"session_id": session} for session in seen_obs.answers],
            "mailbox_writes": [
                {"account_ref": names.ref(write.account), "count": write.count, "at": write.at}
                for write in seen_obs.mailbox_writes
            ],
        },
    }
    body["content_sha256"] = content_sha256(body)
    return body


def content_sha256(bundle: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of ``bundle`` without its ``content_sha256``."""
    body = {key: value for key, value in bundle.items() if key != "content_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_bundle(bundle: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_bundle(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ProofBundleError(f"{path}: not JSON ({exc})") from exc
    if not isinstance(doc, dict):
        raise ProofBundleError(f"{path}: a proof bundle is a JSON object")
    return doc


# ---------------------------------------------------------------------------- verify


def verify_bundle(bundle: Mapping[str, Any]) -> list[Violation]:
    """Every way ``bundle`` fails; empty when it is well formed, intact and all three
    invariants hold. Offline: reads nothing but the bundle."""
    problems = _structure(bundle)
    if problems:
        return problems  # nothing below can be read reliably
    if bundle["content_sha256"] != content_sha256(bundle):
        problems.append(
            Violation("integrity", "content_sha256 does not match the content (edited?)")
        )
    rows: list[Mapping[str, Any]] = list(bundle["ledger"])
    observations: Mapping[str, Any] = bundle["observations"]
    problems += _no_private_to_cloud(rows)
    problems += _mailbox_writes_approved(rows, observations["mailbox_writes"])
    problems += _every_call_and_answer_audited(
        rows, observations["model_calls"], observations["answers"]
    )
    return problems


def _structure(bundle: Mapping[str, Any]) -> list[Violation]:
    out: list[Violation] = []

    def bad(detail: str) -> None:
        out.append(Violation("format", detail))

    if bundle.get("format") != FORMAT:
        bad(f"format is {bundle.get('format')!r}, not {FORMAT!r}")
        return out
    if bundle.get("schema_version") != SCHEMA_VERSION:
        bad(f"schema_version {bundle.get('schema_version')!r} is not supported ({SCHEMA_VERSION})")
        return out
    keys = set(bundle)
    if keys != set(_TOP_FIELDS):
        missing, extra = set(_TOP_FIELDS) - keys, keys - set(_TOP_FIELDS)
        bad(f"top-level keys: missing {sorted(missing)}, unexpected {sorted(extra)}")
        return out
    stated = bundle["invariants"]
    if not isinstance(stated, list) or {
        (i.get("id"), i.get("statement")) for i in stated if isinstance(i, dict)
    } != set(INVARIANTS.items()):
        bad("the invariant statements are not format version 1's, verbatim")
    if not isinstance(bundle["content_sha256"], str):
        bad("content_sha256 is not a string")
    for name in ("producer", "scope"):
        if not isinstance(bundle[name], dict):
            bad(f"{name} is not an object")
    ledger = bundle["ledger"]
    if not isinstance(ledger, list):
        bad("ledger is not a list")
    else:
        for index, row in enumerate(ledger):
            problem = _shape(row, _ROW_FIELDS)
            if problem:
                bad(f"ledger[{index}]: {problem}")
    observations = bundle["observations"]
    if not isinstance(observations, dict) or set(observations) != set(_OBSERVATION_FIELDS):
        bad(f"observations must have exactly {sorted(_OBSERVATION_FIELDS)}")
        return out
    for name, shape in _OBSERVATION_FIELDS.items():
        items = observations[name]
        if not isinstance(items, list):
            bad(f"observations.{name} is not a list")
            continue
        for index, item in enumerate(items):
            problem = _shape(item, shape)
            if problem:
                bad(f"observations.{name}[{index}]: {problem}")
    return out


def _shape(item: Any, shape: Mapping[str, tuple[type, ...]]) -> str | None:
    if not isinstance(item, dict):
        return "not an object"
    if set(item) != set(shape):
        missing, extra = set(shape) - set(item), set(item) - set(shape)
        return f"missing {sorted(missing)}, unexpected {sorted(extra)}"
    for key, types in shape.items():
        value = item[key]
        # bool is an int subclass; a JSON true is never an id or a count.
        if isinstance(value, bool) or not isinstance(value, types):
            return f"{key} has the wrong type"
    return None


def _no_private_to_cloud(rows: Sequence[Mapping[str, Any]]) -> list[Violation]:
    return [
        Violation(
            NO_PRIVATE_TO_CLOUD,
            f"ledger row {row['id']}: a {row['classification']} model call was allowed to "
            f"{row['tier']}",
        )
        for row in rows
        if row["hook_point"] == LLM_CALL_HOOK
        and row["decision"] == "allow"
        and row["classification"] in PRIVATE
        and row["tier"] in CLOUD_TIERS
    ]


def _mailbox_writes_approved(
    rows: Sequence[Mapping[str, Any]], writes: Sequence[Mapping[str, Any]]
) -> list[Violation]:
    """Invariant 2, time-ordered: a timed write holds only when the latest
    ``mailbox_writes`` row for its account at or before the write's time is an ``allow``.
    A revoke (``deny``) between an approval and a write makes the write a violation; a
    later re-approval authorises again. A row at exactly the write's time counts as
    before it, and rows sharing one time are ordered by ledger id, the last deciding --
    so a revoke stamped with the write's own time fails it (fail closed). A write with no
    time cannot be ordered against a revoke, so it is refused (fail closed)."""
    history: dict[str, list[tuple[datetime | None, int, str]]] = {}
    for row in rows:
        if row["hook_point"] == MAILBOX_WRITES_HOOK and row["account_ref"]:
            history.setdefault(row["account_ref"], []).append(
                (_when(row["ts"]), row["id"], row["decision"])
            )
    out: list[Violation] = []
    for write in writes:
        if write["count"] <= 0:
            continue
        ref = write["account_ref"]
        decisions = history.get(ref, [])
        if not any(decision == "allow" for _, _, decision in decisions):
            out.append(
                Violation(
                    MAILBOX_WRITE_APPROVED,
                    f"{write['count']} write(s) to {ref} with no approval row",
                )
            )
            continue
        problem = _write_not_authorised(ref, write["at"], decisions)
        if problem:
            out.append(Violation(MAILBOX_WRITE_APPROVED, problem))
    return out


def _write_not_authorised(
    ref: str, at_text: str | None, decisions: Sequence[tuple[datetime | None, int, str]]
) -> str | None:
    """Why a write to ``ref`` at ``at_text`` is not authorised by ``decisions`` (the
    account's ``mailbox_writes`` rows: time, ledger id, decision), or None when it is."""
    if at_text is None:
        return (
            f"write(s) to {ref} have no time, so they cannot be ordered against a revoke "
            "(an observation must carry 'at')"
        )
    at = _when(at_text)
    if at is None:
        return f"write time {at_text!r} is not ISO 8601"
    if any(moment is None for moment, _, _ in decisions):
        # An unreadable row could be the revoke that decides: fail closed.
        return f"an approval row for {ref} has a time that is not ISO 8601"
    before = sorted(
        (moment, row_id, decision)
        for moment, row_id, decision in decisions
        if moment is not None and moment <= at
    )
    if not before:
        return f"write(s) to {ref} at {at_text} came before its approval"
    if before[-1][2] != "allow":
        return f"write(s) to {ref} at {at_text} came after its approval was revoked"
    return None


def _every_call_and_answer_audited(
    rows: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any]],
    answers: Sequence[Mapping[str, Any]],
) -> list[Violation]:
    out: list[Violation] = []
    keys_by_session: dict[str | None, set[tuple[str, int | None]]] = {}
    for row in rows:
        if row["hook_point"] == LLM_CALL_HOOK:
            key = (row["run_id"], row["step_id"])
            keys_by_session.setdefault(row["session_id"], set()).add(key)
            keys_by_session.setdefault(None, set()).add(key)
    observed = Counter(call["session_id"] for call in calls)
    sessionless = observed.pop(None, 0)
    for session, count in sorted(observed.items()):
        audited = len(keys_by_session.get(session, ()))
        if audited < count:
            out.append(
                Violation(
                    EVERY_CALL_AUDITED,
                    f"session {session}: {count} model call(s) observed, {audited} audited "
                    f"at {LLM_CALL_HOOK}",
                )
            )
    if sessionless:
        audited = len(keys_by_session.get(None, ()))
        if audited < sum(observed.values()) + sessionless:
            out.append(
                Violation(
                    EVERY_CALL_AUDITED,
                    f"{sum(observed.values()) + sessionless} model call(s) observed, "
                    f"{audited} audited at {LLM_CALL_HOOK}",
                )
            )
    answered = {row["session_id"] for row in rows if row["hook_point"] == ANSWER_HOOK}
    for session in sorted({answer["session_id"] for answer in answers}):
        if session not in answered:
            out.append(
                Violation(
                    EVERY_CALL_AUDITED, f"session {session}: an answer with no {ANSWER_HOOK} row"
                )
            )
    return out


# ---------------------------------------------------------------------------- check

#: The most violation details one check reports per invariant (the count is exact).
CHECK_DETAIL_LIMIT = 20


def check_window(
    ledger: AuditLog | Path,
    *,
    since: datetime | str | None = None,
    until: datetime | str | None = None,
    session_logs: Path | None = None,
    observations: Observations | None = None,
) -> dict[str, Any]:
    """Export the window's bundle in memory and verify it, per invariant.

    What ``export`` then ``verify`` say, without a file: for each invariant whether it
    holds, why not (at most ``CHECK_DETAIL_LIMIT`` details, and the exact count), and
    the evidence it was judged on -- because an invariant over no observations holds
    trivially, and a reader has to be able to tell that from one that held over many.
    ``format`` / ``integrity`` problems (none for a bundle made here) are reported apart.
    The session logs are read for model calls and answers; mailbox writes are the
    window's ``mailbox_write_performed`` ledger rows (``export_bundle`` reads them), plus
    any ``observations`` brings.
    """
    seen = (
        observations_from_session_logs(session_logs, since=since, until=until)
        if session_logs is not None
        else Observations()
    )
    if observations is not None:
        seen = seen + observations
    bundle = export_bundle(ledger, since=since, until=until, observations=seen)
    by_invariant: dict[str, list[str]] = {}
    for violation in verify_bundle(bundle):
        by_invariant.setdefault(violation.invariant, []).append(violation.detail)
    rows: list[Mapping[str, Any]] = bundle["ledger"]
    obs: Mapping[str, Any] = bundle["observations"]
    evidence: dict[str, dict[str, int]] = {
        NO_PRIVATE_TO_CLOUD: {
            "model_call_rows": sum(1 for r in rows if r["hook_point"] == LLM_CALL_HOOK),
        },
        MAILBOX_WRITE_APPROVED: {
            "approval_rows": sum(1 for r in rows if r["hook_point"] == MAILBOX_WRITES_HOOK),
            "writes_observed": sum(int(w["count"]) for w in obs["mailbox_writes"]),
        },
        EVERY_CALL_AUDITED: {
            "model_calls_observed": len(obs["model_calls"]),
            "answers_observed": len(obs["answers"]),
        },
    }
    integrity = by_invariant.pop("format", []) + by_invariant.pop("integrity", [])
    invariants = [
        {
            "id": key,
            "statement": statement,
            "ok": key not in by_invariant,
            "violation_count": len(by_invariant.get(key, ())),
            "violations": by_invariant.get(key, [])[:CHECK_DETAIL_LIMIT],
            "evidence": evidence[key],
        }
        for key, statement in INVARIANTS.items()
    ]
    return {
        "since": bundle["scope"]["since"],
        "until": bundle["scope"]["until"],
        "ledger_rows": len(rows),
        "ok": not integrity and all(item["ok"] for item in invariants),
        "integrity": integrity[:CHECK_DETAIL_LIMIT],
        "invariants": invariants,
    }


# ---------------------------------------------------------------------------- helpers


def _rows_in_scope(
    log: AuditLog,
    *,
    since: datetime | str | None,
    until: datetime | str | None,
    run_ids: Sequence[str],
) -> list[AuditRow]:
    if not run_ids:
        return list(log.query(since=since, until=until))
    rows: dict[int, AuditRow] = {}
    for run_id in run_ids:
        for row in log.query(run_id=run_id, since=since, until=until):
            rows[row.id] = row
    return list(rows.values())


def _ledger_writes(rows: Iterable[AuditRow]) -> Observations:
    """The mailbox writes the email library recorded in ``rows`` (hook
    ``mailbox_write_performed``), as observations timed at their row."""
    writes: list[MailboxWrite] = []
    for row in sorted(rows, key=_order):
        if row.hook_point != MAILBOX_WRITE_PERFORMED_HOOK:
            continue
        payload = _payload(row)
        account, count = payload.get("account"), payload.get("count")
        if not isinstance(account, str) or not account:
            continue
        # A write row that does not say how many is still a write: count it as one.
        number = count if isinstance(count, int) and not isinstance(count, bool) else 1
        writes.append(MailboxWrite(account=account, count=number, at=row.ts))
    return Observations(mailbox_writes=tuple(writes))


def _order(row: AuditRow) -> tuple[str, int]:
    return row.ts, row.id


def _bundle_row(row: AuditRow, names: _Pseudonyms) -> dict[str, Any]:
    payload = _payload(row)
    account = (
        payload.get("account")
        if row.hook_point in (MAILBOX_WRITES_HOOK, MAILBOX_WRITE_PERFORMED_HOOK)
        else None
    )
    return {
        "id": row.id,
        "ts": row.ts,
        "run_id": row.run_id,
        "step_id": row.step_id,
        "agent_type": row.agent_type,
        "hook_point": row.hook_point,
        "plugin": row.plugin,
        "decision": row.decision,
        "classification": row.classification,
        "tier": row.tier,
        "severity": row.severity,
        "session_id": _opt_str(payload.get("session_id")),
        "tool_name": _opt_str(payload.get("tool_name")),
        "account_ref": names.ref(str(account)) if account else None,
        "args_digest": _opt_str(payload.get("args_digest")),
        "result_digest": _opt_str(payload.get("result_digest")),
        "digest_alg": _opt_str(payload.get("digest_alg")),
    }


def _payload(row: AuditRow) -> dict[str, Any]:
    try:
        payload = json.loads(row.payload_json)
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _opt_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _iso_or_none(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    return value.isoformat() if isinstance(value, datetime) else str(value)


def _when(value: datetime | str | None | Any) -> datetime | None:
    """``value`` as an aware datetime (a naive one is UTC), or None when it is not one."""
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value
    else:
        try:
            moment = datetime.fromisoformat(str(value))
        except ValueError:
            return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _version() -> str:
    try:
        return metadata.version("iris-harness")
    except metadata.PackageNotFoundError:
        return "unknown"


__all__ = [
    "CHECK_DETAIL_LIMIT",
    "CLOUD_TIERS",
    "EVERY_CALL_AUDITED",
    "FORMAT",
    "INVARIANTS",
    "MAILBOX_WRITES_HOOK",
    "MAILBOX_WRITE_APPROVED",
    "MAILBOX_WRITE_PERFORMED_HOOK",
    "NO_PRIVATE_TO_CLOUD",
    "SCHEMA_FILE",
    "SCHEMA_VERSION",
    "MailboxWrite",
    "Observations",
    "ProofBundleError",
    "Violation",
    "check_window",
    "content_sha256",
    "export_bundle",
    "observations_from_json",
    "observations_from_session_logs",
    "read_bundle",
    "verify_bundle",
    "write_bundle",
]
