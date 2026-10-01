"""The chat ``email`` agent -- the inbox digest, and the loop's degrade path over it.

Two handlers. ``make_email_handler`` is the deterministic read-only digest of the local
store: the ``email`` lane when the harness's governed loop is off. With the loop on, the
harness answers ``email`` on that loop (the plugin claims it with
``api.register_loop_intent``), over the email tools this plugin registers, and
``make_email_fallback_handler`` is its degrade path: the answers code can give without a
model, then the digest, so a weak model's miss never regresses below the digest.

Lifted from ``bootstrap.py`` at M6.1b, when the email library left the core with this
plugin (OSS plan M6, decision 2). Until email slice step 5 the degrade path was a second
governed loop the plugin built itself; the loop is the harness's alone now.

``_bill_email_digest`` is the deterministic answer to "did a bill arrive by email": the
email judge's own ``bill`` bucket (the owner's correction wins over the judge's call)
decides what a bill is, so email answers about statement emails on its own. Finance
depends on email as one of its sources, never the other way round: this plugin imports
nothing of the finance domain.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from iris_harness.sdk.time import local_now
from iris_harness.sdk.tools import BoundTools
from iris_harness.sdk.types import ActivityChunk, AgentTask, HandlerResult, StreamChunk, TraceChunk

logger = logging.getLogger(__name__)


def _email_inbox_digest(
    llm_call: Callable[[str], str] | None, data_dir: Path
) -> tuple[str, dict[str, object]]:
    """Summarise the local email store (read-only) — the chat email agent's answer.

    Reads the synced ``EmailStore`` (kept fresh by the Gmail sweep heartbeat) and
    renders a grounded inbox digest with an optional LLM narrative on top, same
    deterministic-first contract as the Planner. Never raises: on any error it
    returns a short, honest message rather than the generic pipeline error.
    """

    from iris_personal.email.digest import build_inbox_digest, narrate_digest
    from iris_personal.email.store import EmailStore

    try:
        store = EmailStore(db_path=data_dir / "email.db")
        store.ensure_schema()
        digest = build_inbox_digest(store, now=local_now())
        text = narrate_digest(digest, llm_call=llm_call)
        return text, {
            "email_capability_registered": True,
            "accounts": len(digest.accounts),
            "today_total": digest.today_total,
            "grand_total": digest.grand_total,
        }
    except Exception:  # degrade gracefully, never 500 the turn
        logger.exception("email digest failed")
        return (
            "I couldn't read your local email store just now. If this is the first "
            "run, connect an account with `iris auth gmail login --user <address>` "
            "and let a sync complete.",
            {"email_capability_registered": True, "error": "digest_failed"},
        )


# An attachment / document retrieval ask ("get my passport copy", "find the invoice
# PDF"). The local model often picks search_inbox over find_attachment, so the email
# handler deterministically routes these to the attachment tool (issue 0024). Two
# signals: (a) a file extension / "attachment"; (b) a specific document noun — these
# fire verb-free, because the planner often strips the verb ("get my passport copy" →
# "passport copy") before the handler sees it; (c) a fetch verb + a generic doc noun.
_EMAIL_ATTACHMENT_INTENT_RE = re.compile(
    r"\.(?:pdf|docx?|xlsx?|pptx?|jpe?g|png)\b|\battachments?\b|\battached\b"
    r"|\b(?:passport|visa|resume|cv|invoice|receipt|boarding\s+pass|certificate|"
    r"aadhaar|aadhar|pan\s+card|offer\s+letter)\b"
    r"|\b(?:get|find|fetch|download|locate|send\s+me|pull\s+up|show\s+me|where(?:'?s| is))\b"
    r"[^?]*\b(?:copy|document|file)\b",
    re.IGNORECASE,
)

_EMAIL_CONTINUATION_RE = re.compile(
    r"^\s*(?:yes|yes\s+please|please\s+do\s+it|do\s+it|go\s+ahead|continue|"
    r"yeah|yep|sure|ok|okay)\b",
    re.IGNORECASE,
)
_EMAIL_READ_REQUEST_RE = re.compile(
    r"\b(?:read|open|summari[sz]e|show|check|see\s+what\s+is\s+in|what(?:'s|\s+is)\s+in)\b"
    r"[^?]*\b(?:email|emails|message|messages|mail|body)\b",
    re.IGNORECASE,
)
_EMAIL_ADDRESS_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_EMAIL_SUBJECT_CONTAINS_RE = re.compile(
    r"subjects?\s+containing\s+[\"'`“”]?([^\"'`“”]+)[\"'`“”]?",
    re.IGNORECASE,
)
# "any bill / statement emails today?" — generic words only; which emails ARE bills is
# the email judge's call (its `bill` bucket), not a list of institutions here.
_BILL_EMAIL_REQUEST_RE = re.compile(
    r"\b(?:finance|financial|bank|banking|credit\s*card|card|statement|payment|"
    r"bill|billing|due|interest|brokerage|investment)\b"
    r"[^?]*\b(?:email|emails|mail|inbox|message|messages)\b"
    r"|\b(?:email|emails|mail|inbox|message|messages)\b[^?]*\b(?:finance|financial|bank|banking|credit\s*card|card|statement|payment|bill|billing|due|interest|brokerage|investment)\b",
    re.IGNORECASE,
)


# FileManager-domain cues that share the "file/document" vocabulary with the
# attachment detector but clearly mean the on-disk catalog, not an email
# attachment ("show me the file catalog", "files in my folder"). These defer to
# the intent router's files→filemanager rule instead of the inbox attachment
# search. Kept narrow so genuine attachment asks ("get my passport copy", "the
# invoice pdf") still route to email.
_FILEMANAGER_DOMAIN_RE = re.compile(
    r"\b(?:file\s*manager|catalog(?:ue)?|file\s+vault|"
    r"my\s+(?:files|folders?|downloads?)|"
    r"files?\s+(?:in|inside|under|within)\b|"
    r"folders?)\b",
    re.IGNORECASE,
)


def _is_email_attachment_intent(query: str) -> bool:
    """True for a 'get my <document> from email' style attachment ask.

    FileManager-domain phrasings (the catalog, a folder, the file vault) share
    the file/document vocabulary but mean on-disk files — yield those to the
    intent router so they reach the FileManager agent.
    """
    q = query or ""
    if _FILEMANAGER_DOMAIN_RE.search(q):
        return False
    return bool(_EMAIL_ATTACHMENT_INTENT_RE.search(q))


def _previous_user_turn(recent_turns: Sequence[str], current_query: str) -> str:
    """Most recent prior user turn, excluding the current terse follow-up."""
    current = (current_query or "").strip().lower()
    skipped_current = False
    for turn in reversed(tuple(recent_turns or ())):
        if not turn.lower().startswith("user:"):
            continue
        _, _, content = turn.partition(":")
        message = content.strip()
        if not message:
            continue
        if not skipped_current and message.lower() == current:
            skipped_current = True
            continue
        return message
    return ""


def _effective_email_query(task: AgentTask) -> str:
    """Carry the previous scoped ask across a terse "yes" follow-up.

    A numbered pick ("1", "the first one") is deliberately NOT carried here. Rebuilding
    it from the previous user turn lost the number whenever that turn did not read as
    a read request, and re-ran a search instead of reading what was on screen. The
    shortlist is a ``choice`` continuation now, and a pick arrives resolved on
    ``AgentTask.selected_choice``.
    """
    query = (task.query or "").strip()
    if not _EMAIL_CONTINUATION_RE.match(query):
        return query
    ctx = task.memory_context
    if ctx is None or not ctx.recent_turns:
        return query
    previous = _previous_user_turn(ctx.recent_turns, query)
    return previous or query


def _looks_like_email_read_request(query: str) -> bool:
    return bool(_EMAIL_READ_REQUEST_RE.search(query or ""))


def _looks_like_bill_email_request(query: str) -> bool:
    return bool(_BILL_EMAIL_REQUEST_RE.search(query or ""))


def _email_read_lookup_query(query: str) -> str:
    """Condense a verbose read-email ask into sender + subject terms."""
    parts: list[str] = []
    if sender := _EMAIL_ADDRESS_RE.search(query or ""):
        parts.append(sender.group(0))
    if subject := _EMAIL_SUBJECT_CONTAINS_RE.search(query or ""):
        value = subject.group(1).strip().strip("\"'`“”")
        if value:
            parts.append(value)
    compact = " ".join(parts).strip()
    return compact or (query or "").strip()


def _email_query_since(query: str) -> datetime | None:
    from datetime import timedelta

    lowered = (query or "").lower()
    days: int | None = None
    if re.search(r"\btoday\b", lowered):
        days = 1
    elif re.search(r"\byesterday\b", lowered):
        days = 2
    elif re.search(r"\bthis\s+week\b", lowered):
        days = 7
    elif re.search(r"\bthis\s+month\b", lowered):
        days = 31
    if days is None:
        return None
    return local_now() - timedelta(days=days)


def _bill_email_digest(query: str, *, data_dir: Path) -> tuple[str, dict[str, object]] | None:
    """Recent emails the judge filed as bills, or ``None`` to leave the turn to the loop.

    A bill is an email whose effective bucket is ``bill``: the owner's correction when
    there is one, else the judge's. Mail the judge has not sorted yet is held from every
    reader (``email.store``: no reader sees half-processed mail), so it is read here only
    to be counted, never shown: the answer says how many are still being sorted rather
    than implying there is no bill among them.
    """
    from iris_personal.email.contracts import EmailMessage
    from iris_personal.email.store import EmailStore

    from .judgments import JUDGED, WAITING, JudgmentStore

    since = _email_query_since(query)
    store = EmailStore(db_path=data_dir / "email.db")
    store.ensure_schema()
    judgments = JudgmentStore(db_path=store.db_path)
    judgments.ensure_schema()

    bills: list[EmailMessage] = []
    being_sorted = 0
    seen_ids: set[str] = set()
    for account_id in store.list_accounts():
        for msg in store.list_recent(account_id, limit=80, since=since, include_held=True):
            if msg.id in seen_ids:
                continue
            seen_ids.add(msg.id)
            judgment = judgments.get(msg.id)
            if judgment is None:
                continue  # never judged (the judge is off, or older mail): not known a bill
            if judgment.status == WAITING:
                being_sorted += 1  # held: counted, never shown
                continue
            if judgment.status == JUDGED and judgment.effective_bucket == "bill":
                bills.append(msg)

    if not bills:
        return None

    lines: list[str] = []
    for msg in bills[:8]:
        when = msg.received_at.strftime("%b %d %H:%M")
        snippet = (msg.snippet or "").strip()
        snippet = re.sub(r"\s+", " ", snippet)
        preview = f" — {snippet[:120].rstrip()}" if snippet else ""
        lines.append(f"- {when} · {msg.from_address}: {msg.subject}{preview}")

    window = " today" if re.search(r"\btoday\b", query, re.IGNORECASE) else ""
    text = f"I found {len(bills[:8])} bill email(s){window}:\n" + "\n".join(lines)
    if being_sorted:
        text += (
            f"\n\n{being_sorted} newer email(s) in that window are still being sorted, "
            "so a bill among them would not show here yet."
        )
    return text, {
        "agent_type": "email",
        "email_capability_registered": True,
        "bill_email_recall": True,
        "matched_count": len(bills),
        "being_sorted_count": being_sorted,
    }


def make_email_handler(
    *,
    llm_call: Callable[[str], str] | None = None,
    data_dir: Path,
) -> tuple[Callable[[AgentTask], HandlerResult], Callable[[AgentTask], Iterator[StreamChunk]]]:
    """Build the chat email agent: a read-only inbox digest from the local store."""

    def handler(task: AgentTask) -> HandlerResult:  # digest is "my inbox"
        return _email_inbox_digest(llm_call, data_dir)

    def stream_handler(task: AgentTask) -> Iterator[StreamChunk]:
        yield ActivityChunk("reading your inbox")
        text, metadata = _email_inbox_digest(llm_call, data_dir)
        yield TraceChunk(
            json.dumps(
                {
                    "event": "email.digest",
                    "agent_type": "email",
                    "accounts": metadata.get("accounts", 0),
                    "today_total": metadata.get("today_total", 0),
                },
                sort_keys=True,
            )
        )
        yield text
        yield metadata

    return handler, stream_handler


def make_email_fallback_handler(
    tools: BoundTools | None,
    *,
    data_dir: Path,
    llm_call: Callable[[str], str] | None,
) -> Callable[[AgentTask], HandlerResult]:
    """The ``email`` intent's degrade path under the harness's governed loop.

    The loop answers ``email`` turns (``api.register_loop_intent``); this runs when a
    turn errors, answers nothing, or answers without reading. It answers what code can
    answer without a model -- the pick from a shortlist this agent showed, an attachment
    ask, an explicit read by sender/subject, a bill-email question -- and otherwise the
    deterministic inbox digest, so the worst case is never worse than the digest.

    ``tools`` is the plugin's ``api.tools``: every email tool this answer runs goes
    through the governed runner as ``plugin:email_workflows`` (PRE_TOOL_USE, the approval
    rules, POST_TOOL_USE, an audit row), never by calling the tool's function. A call
    governance holds or refuses is said as such, then the digest. Without a runtime
    (``tools`` None) only the answers that call no tool remain.

    Until email slice step 5 this built a second governed loop of its own, over its own
    copy of the tools: a loop as the degrade path of the loop. That needed the agentic
    core and the governance kernel, which a plugin may not import, and it is gone.
    """

    def _digest(note: str = "") -> tuple[str, dict[str, object]]:
        text, meta = _email_inbox_digest(llm_call, data_dir)
        if not note:
            return text, meta
        return f"{note}\n\n{text}", {**meta, "tool_held": True}

    def _run(name: str, args: dict[str, Any]) -> tuple[str | None, str]:
        """``(answer, note)``: the tool's output when it ran, else why not (held/refused)."""
        if tools is None:
            return None, ""
        result = tools.call(name, args)
        if result.ok and not result.held:
            return result.text, ""
        if result.held:
            return None, f"I couldn't run {name} for this: {result.text}"
        logger.debug("email fallback: %s did not run: %s", name, result.text)
        return None, ""

    def handler(task: AgentTask) -> HandlerResult:
        effective_query = _effective_email_query(task)
        # The pick from a shortlist this agent showed: the harness resolved it against
        # the ids the shortlist recorded, so this is a direct read by id -- no search to
        # re-run, no model to ask what "the 1 st one" means.
        message_id = str((task.selected_choice or {}).get("message_id") or "").strip()
        if message_id:
            out, note = _run("read_email", {"message_id": message_id})
            if note:
                return _digest(note)
            if out is not None:
                return out, {
                    "agent_type": "email",
                    "email_capability_registered": True,
                    "read_email_recall": True,
                    "selected_choice": True,
                }
        # An attachment ask goes straight to the attachment finder (issue 0024).
        if _is_email_attachment_intent(effective_query):
            out, note = _run("find_attachment", {"query": effective_query})
            if note:
                return _digest(note)
            if out and out.strip():
                return out, {
                    "agent_type": "email",
                    "email_capability_registered": True,
                    "attachment_recall": True,
                }
        if _looks_like_email_read_request(effective_query):
            out, note = _run("read_email", {"query": _email_read_lookup_query(effective_query)})
            if note:
                return _digest(note)
            if out and out.strip() and not out.lower().startswith(("error:", "no email found")):
                return out, {
                    "agent_type": "email",
                    "email_capability_registered": True,
                    "read_email_recall": True,
                    "effective_query": effective_query,
                }
        if _looks_like_bill_email_request(effective_query):
            try:
                bill_digest = _bill_email_digest(effective_query, data_dir=data_dir)
                if bill_digest is not None:
                    text, digest_meta = bill_digest
                    return text, {**digest_meta, "effective_query": effective_query}
            except Exception:  # fall through to the digest
                logger.debug("bill email deterministic recall failed", exc_info=True)
        return _digest()

    return handler


# ADR-0072 slice 3 — judge-gated clarify nudge at the answer boundary.
