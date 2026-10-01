"""ReAct tools for the chat email agent (issue 0001).

The email intent used to dispatch to a single fixed inbox digest, ignoring the
query. These tools make it query-aware the agentic-OS way: the LLM in the email
ReAct loop *selects* the right tool and fills its args from the natural-language
ask. Each tool aggregates across all connected accounts and returns grounded
text; the loop composes the final answer from the observation.

- ``inbox_digest`` — "what's my inbox" (unchanged grounded digest).
- ``search_inbox`` — "latest AI article emails" → FTS5 text search.
- ``list_by_category`` — "show my Newsletters" → category browse.
- ``analyze_inbox`` — "which credit cards do I have" → every match, grouped by sender.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from iris_harness.sdk.time import local_now
from iris_harness.sdk.types import ToolSpec
from iris_personal.email.categories import (
    LabelsFor,
    ResolvedCategory,
    category_menu,
    provider_labels,
    resolve_category,
)
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.feedback_keys import (
    EMAIL_SEARCH_SURFACE,
    EMAIL_SUBSYSTEM,
    email_search_dims,
)
from iris_personal.email.store import CategoryFilter, EmailStore, SearchHit

logger = logging.getLogger(__name__)


def _log_degraded(op: str, exc: BaseException) -> None:
    """Log a search-path failure the caller degrades past (#668 follow-up).

    The agent sees "no results" or an error line; without this, a broken index, store
    or token is invisible to operators. Operation + exception type only, never the
    query or any email content.
    """
    logger.warning("email %s failed (%s); degrading", op, type(exc).__name__, exc_info=True)


# Cap how many hits we hand back to the loop — enough to answer, small enough to
# keep the observation (and the model's context) tight.
_MAX_RESULTS = 15
# An analysis reads every match, not the top few; this only bounds a runaway query.
_ANALYSIS_MAX_HITS = 1000

# --- Search-result feedback (issue 0006 close-the-loop, on the issue-0028 spine) ---
# Marking a search result "not what I meant" suppresses that SENDER from future
# search results — keyed on from_domain. Search is pull (the user asked), so a
# suppressed sender is *downranked to the bottom*, never hidden, mirroring the
# research engine's source downrank.
# The suppression keys are email's vocabulary (``iris_personal.email.feedback_keys``).
# Callers import them from there, never through this module: a re-export would be the
# shim decision 11 forbids.


def _sender_domain(from_domain: str | None, from_address: str) -> str:
    """Best-effort sender domain (the SearchHit/EmailMessage from_domain, else derived)."""
    if from_domain:
        return from_domain.strip().lower()
    addr = from_address
    if "<" in addr and ">" in addr:
        addr = addr[addr.rfind("<") + 1 : addr.rfind(">")]
    return addr.rsplit("@", 1)[-1].strip().lower() if "@" in addr else ""


def _sender_matches_tokens(from_domain: str | None, from_address: str, tokens: list[str]) -> bool:
    if not tokens:
        return False
    domain = _sender_domain(from_domain, from_address)
    sender = (from_address or "").lower()
    return any(tok and (tok in domain or tok in sender) for tok in tokens)


def _is_suppressed_sender(feedback_store: Any, from_domain: str) -> bool:
    if feedback_store is None or not from_domain:
        return False
    try:
        return bool(
            feedback_store.should_suppress(
                EMAIL_SUBSYSTEM, EMAIL_SEARCH_SURFACE, email_search_dims(from_domain)
            )
        )
    except Exception as exc:  # noqa: BLE001 — suppression must never break search
        _log_degraded("search suppression check", exc)
        return False


_SEARCH_FEEDBACK_HINT = (
    '\n(Not relevant? Tell me e.g. "ignore <sender> in search" and I\'ll stop '
    "surfacing that sender here.)"
)

# Invisible padding bulk-mailers inject (combining grapheme joiner, zero-widths,
# soft hyphen, line/para separators) — strip so snippets read cleanly.
_INVISIBLE = re.compile("[\u034f\u200b-\u200f\u2028\u2029\u202f\u2060\ufeff\u00ad]")


def _clean(text: str, *, limit: int = 140) -> str:
    text = _INVISIBLE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit].rstrip()


def _store(data_dir: Path) -> EmailStore:
    store = EmailStore(db_path=data_dir / "email.db")
    store.ensure_schema()
    return store


# Words that carry no topic in an inbox ask — used to tell a real search term
# ("AI", "Amazon") from filler ("how is my inbox today"). Backstop only; the
# model's tool choice + the tool descriptions are the primary mechanism.
_TOPICLESS = frozenset(
    {
        "a",
        "an",
        "the",
        "my",
        "me",
        "i",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "any",
        "new",
        "all",
        "some",
        "what",
        "whats",
        "show",
        "tell",
        "give",
        "about",
        "from",
        "with",
        "for",
        "to",
        "of",
        "in",
        "on",
        "and",
        "or",
        "how",
        "today",
        "now",
        "this",
        "that",
        "these",
        "those",
        "latest",
        "recent",
        "email",
        "emails",
        "mail",
        "mails",
        "message",
        "messages",
        "inbox",
        "mailbox",
        "gmail",
        "unread",
        "please",
        "got",
        "have",
        "look",
        "looks",
        "looking",
    }
)


def _content_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _TOPICLESS]


def _fts_safe_lookup_queries(text: str) -> list[str]:
    """Candidate lookup queries that won't trip FTS5 syntax on punctuation-heavy asks."""
    raw = (text or "").strip()
    candidates: list[str] = []
    if raw:
        candidates.append(raw)
    compact = " ".join(re.findall(r"[a-z0-9]+", raw.lower()))
    if compact and compact not in candidates:
        candidates.append(compact)
    return candidates


def _query_grounded(model_query: str, user_turn: str) -> bool:
    """True if the model's search term is actually present in the user's message.

    A weak local model sometimes invents a topic (e.g. "finance") for a plain
    "how is my inbox today" ask — pulling an example word from the tool docs.
    When none of the query's content words appear in the user's own turn, the
    term is hallucinated; the caller falls back to the digest instead of running
    a misleading search. Permissive prefix match so finance/financial/finances
    all count as grounded. When the user turn is unknown, never block (return
    True) — this is a safety net, not a gate.
    """
    if not user_turn:
        return True
    q_tokens = _content_tokens(model_query)
    if not q_tokens:
        return False  # query was only filler words — no real topic
    user_tokens = set(re.findall(r"[a-z0-9]+", user_turn.lower()))

    def _present(tok: str) -> bool:
        # Both sides need a real stem for a prefix match: a user token like "i" or "my"
        # is a prefix of almost any word, which made every invented topic "grounded".
        return any(
            tok == u or (len(tok) >= 4 and len(u) >= 4 and (u.startswith(tok) or tok.startswith(u)))
            for u in user_tokens
        )

    return any(_present(t) for t in q_tokens)


_STATEMENT_DETAILS_REQUEST_RE = re.compile(
    r"\bstatement\b[^?]*\b(?:account\s*details?|account\s*number|balances?|closing\s*balance|available\s*balance)\b"
    r"|\b(?:account\s*details?|account\s*number|balances?|closing\s*balance|available\s*balance)\b[^?]*\bstatement\b",
    re.IGNORECASE,
)
_STATEMENT_SUBJECT_RE = re.compile(
    r"\b(statement|e-?statement|account\s*statement|balance\s*summary)\b",
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(
    r"(?i)\b(?:INR|USD|EUR|GBP|AED|SAR|Rs\.?|\$|€|£)?\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\b"
)
# Email addresses in a user's search query must not be reflected verbatim in the
# tool's display strings (privacy-first: don't echo PII the reply doesn't need —
# the search already used it). Redacted to a placeholder in every "matched
# '<query>'" message. (2026-07-06 red-team finding 3.)
_QUERY_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def _display_query(query: str) -> str:
    """Sanitize a raw search query for echoing back to the user: redact any
    email address so a reflected query never surfaces PII in the response."""
    return _QUERY_EMAIL_RE.sub("<email>", query)


def _wants_statement_details(query: str) -> bool:
    return bool(_STATEMENT_DETAILS_REQUEST_RE.search(query or ""))


def _mask_account_numbers(text: str) -> str:
    """Mask long numeric account identifiers while preserving last-4 digits."""

    def _mask(match: re.Match[str]) -> str:
        digits = match.group(0)
        return f"••{digits[-4:]}"

    return re.sub(r"(?<!\d)\d{8,18}(?!\d)", _mask, text or "")


def _extract_statement_details(text: str) -> list[str]:
    """Best-effort extraction of account identifiers and balances from statement text."""
    source = text or ""
    details: list[str] = []

    account_match = re.search(
        r"(?i)\b(?:account|a/c|acct)\s*(?:number|no\.?|#)\s*[:\-]?\s*([A-Z0-9*X\-]*\d{4,20}|••\d{4})",
        source,
    )
    if account_match is None:
        account_match = re.search(
            r"(?i)\b(?:account|a/c|acct)\b[^\n]{0,30}\b(••\d{4}|\d{8,18})\b",
            source,
        )
    if account_match:
        value = account_match.group(1).strip()
        value = _mask_account_numbers(value)
        details.append(f"Account: {value}")

    for label, pattern in (
        (
            "Available balance",
            r"(?i)\bavailable\s*balance\b\s*[:\-]?\s*([^\n;]{1,40})",
        ),
        (
            "Closing balance",
            r"(?i)\bclosing\s*balance\b\s*[:\-]?\s*([^\n;]{1,40})",
        ),
        (
            "Current balance",
            r"(?i)\bcurrent\s*balance\b\s*[:\-]?\s*([^\n;]{1,40})",
        ),
    ):
        match = re.search(pattern, source)
        if not match:
            continue
        raw = match.group(1).strip().rstrip(".")
        amount_match = _AMOUNT_RE.search(raw)
        value = amount_match.group(0).strip() if amount_match else raw
        details.append(f"{label}: {value}")

    # Keep order but remove duplicates from noisy bodies/summaries.
    seen: set[str] = set()
    out: list[str] = []
    for item in details:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _since(args: dict[str, Any]) -> datetime | None:
    raw = args.get("since_days")
    if raw in (None, ""):
        return None
    try:
        days = int(str(raw))
    except (TypeError, ValueError):
        return None
    if days <= 0:
        return None
    return local_now() - timedelta(days=days)


def _infer_since_from_query(user_turn: str) -> datetime | None:
    """Best-effort time window from the user's phrasing when the model omits it."""
    days = _infer_days_from_query(user_turn)
    if days is None:
        return None
    return local_now() - timedelta(days=days)


def _infer_days_from_query(user_turn: str) -> int | None:
    """The window ``_infer_since_from_query`` reads, in days, or None."""
    lowered = (user_turn or "").lower()
    days: int | None = None
    if re.search(r"\btoday\b", lowered):
        days = 1
    elif re.search(r"\byesterday\b", lowered):
        days = 2
    elif re.search(r"\bthis\s+week\b", lowered):
        days = 7
    elif re.search(r"\bthis\s+month\b", lowered):
        days = 31
    return days


def _fmt_hit(hit: SearchHit) -> str:
    when = hit.received_at.strftime("%Y-%m-%d")
    cat = f" [{hit.classified_category}]" if hit.classified_category else ""
    raw = hit.snippet_highlighted.replace("<mark>", "").replace("</mark>", "")
    snip = _clean(raw)
    snip = f" — {snip}" if snip else ""
    # The id lets the model name exact emails to a tool that acts on them (trash_email);
    # the owner chose to show it over trashing "whatever the last search found".
    return (
        f"- {when}  {_clean(hit.subject) or '(no subject)'}  · from {hit.from_address}{cat}{snip}"
        f"  [id {hit.id}]"
    )


def _fmt_msg(msg: EmailMessage) -> str:
    # EmailMessage doesn't carry the category (that's on SearchHit); for a
    # category browse every result is already in the asked-for category anyway.
    when = msg.received_at.strftime("%Y-%m-%d")
    snip = _clean(msg.snippet or "")
    snip = f" — {snip}" if snip else ""
    return (
        f"- {when}  {_clean(msg.subject) or '(no subject)'}  · from {msg.from_address}{snip}"
        f"  [id {msg.id}]"
    )


def _fmt_read_candidate(hit: SearchHit, *, idx: int) -> str:
    when = hit.received_at.strftime("%Y-%m-%d")
    return f"{idx}. {when}  {_clean(hit.subject) or '(no subject)'}  · from {hit.from_address}"


# Generic document words + file extensions that over-constrain an attachment search —
# the distinctive noun ("passport", "invoice") is what matters (issue 0024).
_ATTACHMENT_STOPWORDS = frozenset(
    {
        "copy",
        "copies",
        "document",
        "documents",
        "doc",
        "docs",
        "file",
        "files",
        "scan",
        "scanned",
        "attachment",
        "attachments",
        "attached",
        "get",
        "fetch",
        "find",
        "download",
        "send",
        "pull",
        "show",
        "locate",
        "please",
        "the",
        "pdf",
        "docx",
        "xlsx",
        "png",
        "jpg",
        "jpeg",
        "image",
        # query filler that isn't a topic ("can you get …")
        "can",
        "could",
        "would",
        "will",
        "you",
        "your",
        "need",
        "want",
        "give",
        "there",
        "any",
    }
)


def _term_hit(term: str, text: str) -> bool:
    """Alphanumeric-boundary match: a noise token ("you") can't match inside "your",
    but "passport" still matches "passport_copy.pdf" (``_``/``.`` are
    boundaries, unlike ``\\b`` which treats ``_`` as a word char)."""
    return (
        re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text, re.IGNORECASE) is not None
    )


def find_email_attachment(*, data_dir: Path, query: str, since_days: Any = None) -> str:
    """Locate email ATTACHMENTS the user is after ("get my passport copy", "the
    invoice PDF"). Plain harness logic (NOT a governed ToolSpec) so the email handler
    can call it deterministically without tripping the pre_tool_use gate (issue 0024).

    Checks the local store (which usually has no attachment metadata — the sweep
    fetches headers only) then falls back to a TARGETED live Gmail search
    (``has:attachment <terms>``). Matches on filename + subject/sender; honest
    "couldn't find" when none; graceful re-auth nudge on a revoked token."""
    query = (query or "").strip()
    if not query:
        return "Error: find_attachment requires a 'query' (e.g. 'passport', 'invoice')."
    # Drop generic document words + file extensions: they over-constrain the Gmail
    # AND-query ("has:attachment passport copy" misses "Kutty Passport.pdf" because
    # the file isn't named "copy"). Keep only distinctive nouns (issue 0024).
    terms = [t for t in _content_tokens(query) if len(t) >= 3 and t not in _ATTACHMENT_STOPWORDS]
    since = _since({"since_days": since_days})
    try:
        store = _store(data_dir)
        accounts = store.list_accounts()
    except Exception as exc:  # noqa: BLE001
        _log_degraded("find_attachment account list", exc)
        return f"find_attachment failed: {exc}"

    seen: set[str] = set()
    out: list[str] = []
    candidates: list[EmailMessage] = []
    # FTS5 MATCH rejects raw punctuation ("...email?"), so search on the cleaned
    # terms, not the raw query.
    fts_q = " ".join(terms)
    for account_id in accounts:
        if fts_q:
            try:
                for hit in store.search(fts_q, account_id=account_id, limit=20):
                    msg = store.get(hit.id)
                    if msg is not None:
                        candidates.append(msg)
            except Exception:  # FTS quirks never block the live fallback
                logger.debug("find_attachment: store search failed for %r", fts_q, exc_info=True)
        candidates.extend(store.list_recent(account_id, limit=400, since=since))

    for msg in candidates:
        if msg.id in seen or not msg.attachments:
            continue
        subject_match = any(_term_hit(t, msg.subject) for t in terms)
        for att in msg.attachments:
            if subject_match or any(_term_hit(t, att.filename) for t in terms):
                seen.add(msg.id)
                when = msg.received_at.strftime("%Y-%m-%d")
                kb = max(1, att.size_bytes // 1024)
                out.append(
                    f"- {att.filename} ({kb} KB) — from {msg.from_address}, {when} "
                    f"(subject: {_clean(msg.subject, limit=80)})"
                )
                break
        if len(out) >= 10:
            break

    if not out:
        auth_lost = False
        try:
            from iris_personal.email.providers import mail_provider_for

            for account_id in accounts:
                provider = mail_provider_for(account_id)
                if provider is None:
                    continue  # no mailbox plugin for this account: the local index was it
                try:
                    live = provider.list_attachment_candidates(account_id, terms, limit=15)
                except Exception as exc:  # noqa: BLE001 — token revoked / network
                    _log_degraded("live attachment search", exc)
                    auth_lost = True
                    continue
                for cand in live:
                    subj_match = any(_term_hit(t, cand.subject) for t in terms)
                    for att in cand.attachments:
                        if (
                            not terms
                            or subj_match
                            or any(_term_hit(t, att.filename) for t in terms)
                        ):
                            kb = max(1, att.size_bytes // 1024)
                            out.append(
                                f"- {att.filename} ({kb} KB) — from {cand.from_address} "
                                f"(subject: {_clean(cand.subject, limit=80)})"
                            )
                            break
                    if len(out) >= 10:
                        break
                if out:
                    break
        except Exception:  # attachment search is best-effort
            logger.debug("find_attachment: live mailbox search failed", exc_info=True)
        if not out and auth_lost:
            return (
                f"I couldn't search for '{query}' — your Gmail connection may need "
                "re-authentication (`iris auth gmail login`)."
            )

    if not out:
        return (
            f"I couldn't find any email attachment matching '{_display_query(query)}'. It may "
            "not have arrived yet, or the file name doesn't mention it — try the sender or "
            "subject."
        )
    head = f"Found {len(out)} attachment(s) matching '{_display_query(query)}':"
    return head + "\n" + "\n".join(out)


def build_email_tools(
    *,
    data_dir: Path,
    llm_call: Callable[[str], str] | None,
    summarize_llm: Callable[[str], str] | None = None,
    current_query: Callable[[], str] | None = None,
    current_session_id: Callable[[], str] | None = None,
    continuations: Any = None,
    labels_for: LabelsFor = provider_labels,
) -> list[ToolSpec]:
    """Build the email ReAct tool set (aggregating across all accounts).

    ``summarize_llm`` (issue 0002 item C) is a capable single-shot summariser used
    by ``read_email`` to summarise a fetched body — done in the tool, not left to
    the ReAct loop (which often surfaced the raw body). Falls back to ``llm_call``,
    then to the raw body, if absent.

    ``current_query`` returns the user's original turn for the live request; when
    given, ``search_inbox`` uses it to reject hallucinated search terms (a topic
    the model invented that the user never typed) and fall back to the digest.

    ``continuations`` + ``current_session_id`` (ADR-0106 ``choice``): when both are
    given, the numbered shortlist ``read_email`` shows is recorded against the session
    with the message ids it listed, so a reply of "1" or "the first one" reads the
    email that was on screen. The harness routes that reply to the ``email`` agent,
    which reads the picked id — without them a pick has nothing to resolve against.

    Unless ``IRIS_EMAIL_SEMANTIC_SEARCH=0`` (ADR-0071; on by default since ADR-0121
    PR 4) ``search_inbox`` is hybrid:
    it fuses lexical FTS5 with semantic recall over the local MiniLM/Chroma index via
    RRF, so conceptual asks match mail that lacks the literal words. Off by default —
    then ``search_inbox`` is purely lexical (with the issue-0006 broaden/offer gate).
    One search tool either way, so the model never has to choose lexical vs semantic.

    A ``category`` argument is a plain name ("promo", "credit cards") resolved against
    the categories that exist (:mod:`iris_personal.email.categories`); ``labels_for``
    gives each account's provider label → path table (the registered provider's by
    default; tests pass their own)."""

    semantic_index = None
    from iris_personal.email.semantic_index import semantic_search_enabled

    if semantic_search_enabled():
        try:
            from iris_personal.email.semantic_index import EmailSemanticIndex

            semantic_index = EmailSemanticIndex(persist_dir=data_dir / "email_semantic")
        except Exception as exc:  # noqa: BLE001 — semantic search is optional polish
            _log_degraded("semantic index open", exc)
            semantic_index = None

    # Surface-feedback spine (issue 0028): downrank senders the user marked "not
    # what I meant" in past searches. Default store; best-effort.
    feedback_store: Any = None
    try:
        from iris_harness.sdk.learning import SurfaceFeedbackStore

        feedback_store = SurfaceFeedbackStore()
        feedback_store.ensure_schema()
    except Exception as exc:  # noqa: BLE001 — feedback is optional polish
        _log_degraded("feedback store open", exc)
        feedback_store = None

    def _digest_text() -> str:
        from iris_personal.email.digest import build_inbox_digest, narrate_digest

        try:
            store = _store(data_dir)
            digest = build_inbox_digest(store, now=local_now())
            return narrate_digest(digest, llm_call=llm_call)
        except Exception as exc:  # noqa: BLE001 — tools never raise into the loop
            _log_degraded("inbox_digest", exc)
            return f"inbox_digest unavailable: {exc}"

    def _inbox_digest(_args: dict[str, Any]) -> str:
        return _digest_text()

    def _search_inbox(args: dict[str, Any]) -> str:
        query = str(args.get("query") or args.get("input") or "").strip()
        if not query:
            # "search my inbox" with no term means "show my inbox" — be forgiving
            # (the local model sometimes picks search_inbox for a plain inbox ask)
            # and return the digest instead of an error the model would surface.
            return _digest_text()
        # Backstop: a weak model sometimes invents a topic (e.g. "finance") for a
        # plain "how is my inbox today" ask. If the term isn't in the user's own
        # words, treat it like an empty query and return the digest rather than a
        # misleading "N email(s) matched 'finance'". Only fires when we know the
        # original turn; never blocks a genuinely grounded search.
        user_turn = current_query() if current_query is not None else ""
        if not _query_grounded(query, user_turn):
            return _digest_text()
        category = args.get("category")
        category = str(category).strip() if category else None
        prefer_sender_tokens: list[str] = []
        raw_prefer = args.get("prefer_sender_tokens")
        if isinstance(raw_prefer, str):
            prefer_sender_tokens = [raw_prefer.strip().lower()] if raw_prefer.strip() else []
        elif isinstance(raw_prefer, list):
            prefer_sender_tokens = [
                str(tok).strip().lower() for tok in raw_prefer if str(tok).strip()
            ]
        since = _since(args)
        if since is None:
            since = _infer_since_from_query(user_turn)
        try:
            store = _store(data_dir)
            accounts = store.list_accounts()
        except Exception as exc:  # noqa: BLE001
            _log_degraded("search_inbox account list", exc)
            return f"search_inbox failed: {exc}"

        # A category name the mail does not use can only return nothing, and "no emails
        # matched" then reads as "you have no such mail": search everything and say so.
        unknown_category = ""
        filters: dict[str, CategoryFilter] = {}
        if category:
            try:
                resolved = {
                    a: resolve_category(store, category, a, labels_for=labels_for) for a in accounts
                }
            except Exception as exc:  # noqa: BLE001
                _log_degraded("search_inbox category resolve", exc)
                return f"search_inbox failed: {exc}"
            if any(r.found for r in resolved.values()):
                filters = {a: r.filter for a, r in resolved.items()}
            else:
                unknown_category = (
                    f"No category matches '{category}' ({category_menu(store)}), "
                    "so I searched the whole inbox.\n"
                )
                category = None

        def _run(fts_query: str) -> list[SearchHit]:
            found: list[SearchHit] = []
            for account_id in accounts:
                if category and account_id not in filters:
                    continue
                found.extend(
                    store.search(
                        fts_query,
                        account_id=account_id,
                        category_prefix=filters[account_id] if category else None,
                        since=since,
                        limit=_MAX_RESULTS,
                    )
                )
            return found

        try:
            hits = _run(query)
        except Exception as exc:  # noqa: BLE001
            _log_degraded("search_inbox", exc)
            return f"search_inbox failed: {exc}"

        # Hybrid retrieval (ADR-0071 slice 2): when the semantic index is on, fuse the
        # lexical hits with vector recall via RRF so a conceptual ask ("pending
        # payments") still surfaces mail that never says the literal words. Category
        # filters lean lexical (the index keeps no category metadata yet).
        auto_block = ""

        def _auto_statement_block(message_id: str | None) -> str:
            if not _wants_statement_details(query) or not message_id:
                return ""
            # Read and summarize the top matched statement email immediately, so users
            # asking for account details + balances get one-shot answers.
            top = _mask_account_numbers(_read_email({"message_id": message_id}))
            if top.lower().startswith(("read_email failed", "error:", "no email found")):
                return ""
            extracted = _extract_statement_details(top)
            if extracted:
                return "\n\nTop statement details (auto-read):\n" + "\n".join(
                    f"- {line}" for line in extracted
                )
            return "\n\nTop statement details (auto-read):\n" + _clean(top, limit=320)

        if semantic_index is not None and semantic_index.is_ready and category is None:
            try:
                sem_ids = [
                    mid for mid, _ in semantic_index.search(query, k=_MAX_RESULTS, since=since)
                ]
            except Exception as exc:  # noqa: BLE001 — semantic leg is best-effort
                _log_degraded("search_inbox semantic leg", exc)
                sem_ids = []
            if sem_ids:
                from iris_personal.email.semantic_index import rrf_fuse

                hit_by_id = {h.id: h for h in hits}
                fused = rrf_fuse([[h.id for h in hits], sem_ids])[:_MAX_RESULTS]

                def _mid_domain(mid: str) -> str:
                    h = hit_by_id.get(mid)
                    if h is not None:
                        return _sender_domain(h.from_domain, h.from_address)
                    m = store.get(mid)
                    return _sender_domain(m.from_domain, m.from_address) if m is not None else ""

                # Downrank suppressed senders to the bottom (stable — keeps fused order
                # within each group); never hide, since the user asked for these.
                if feedback_store is not None:
                    fused.sort(
                        key=lambda f: _is_suppressed_sender(feedback_store, _mid_domain(f[0]))
                    )
                if prefer_sender_tokens:

                    def _sender_sort_key(f: Any) -> bool:
                        hit = hit_by_id.get(f[0])
                        return not _sender_matches_tokens(
                            hit.from_domain if hit else None,
                            hit.from_address if hit else "",
                            prefer_sender_tokens,
                        )

                    fused.sort(key=_sender_sort_key)
                lines = []
                for mid, _ in fused:
                    hit = hit_by_id.get(mid)
                    if hit is not None:
                        lines.append(_fmt_hit(hit))
                    elif (msg := store.get(mid)) is not None:
                        lines.append(_fmt_msg(msg))
                if lines:
                    top_statement_id = next(
                        (
                            mid
                            for mid, _ in fused
                            if (
                                ((hit := hit_by_id.get(mid)) is not None)
                                and _STATEMENT_SUBJECT_RE.search(hit.subject or "")
                            )
                        ),
                        fused[0][0] if fused else None,
                    )
                    auto_block = _auto_statement_block(top_statement_id)
                    return (
                        f"{len(lines)} email(s) matching '{query}' (by keyword + meaning):\n"
                        + "\n".join(lines)
                        + auto_block
                        + _SEARCH_FEEDBACK_HINT
                    )

        # Dead-end gate (quick win — see docs/issues/0006): FTS5 ANDs the terms, so
        # a conceptual ask like "pending payment" misses mail that says only one of
        # the words ("payment due", "statement ready"). Before giving up, broaden to
        # an OR of the query's content words (prefix-matched) — generic, not
        # finance-specific. Proper semantic retrieval is the follow-up feature.
        broadened = False
        if not hits:
            tokens = _content_tokens(query)
            if len(tokens) > 1:
                or_query = " OR ".join(f"{t}*" for t in tokens)
                try:
                    hits = _run(or_query)
                except Exception as exc:  # noqa: BLE001 — broaden is best-effort
                    _log_degraded("search_inbox broaden", exc)
                    hits = []
                broadened = bool(hits)

        # Nothing in the category matched: the model's category guess must not turn a
        # real email into "you have no such mail". Seen on the cloud trial (2026-09-20):
        # a real Anthropic receipt was reported missing because the model passed
        # category='bills'. Drop the filter and say so, rather than answer a filter
        # artefact as fact.
        dropped_category = unknown_category
        if not hits and category:

            def _run_uncategorised(fts_query: str) -> list[SearchHit]:
                found: list[SearchHit] = []
                for account_id in accounts:
                    found.extend(
                        store.search(
                            fts_query, account_id=account_id, since=since, limit=_MAX_RESULTS
                        )
                    )
                return found

            for attempt in (query, " OR ".join(f"{t}*" for t in _content_tokens(query))):
                if not attempt:
                    continue
                try:
                    hits = _run_uncategorised(attempt)
                except Exception as exc:  # noqa: BLE001 — best-effort, the empty result stands
                    _log_degraded("search_inbox uncategorised retry", exc)
                    hits = []
                if hits:
                    broadened = attempt != query
                    dropped_category = (
                        f"No '{category}' email matched, so I searched the whole inbox.\n"
                    )
                    break

        if not hits:
            # Still nothing — offer next steps / invite a clarification instead of a
            # flat "no results" the model would just echo as a dead end.
            cat = f" in category {category}" if category else ""
            return unknown_category + (
                f"No emails matched '{_display_query(query)}'{cat}. I couldn't find a direct "
                "match — want me to broaden the search, try a related category, or did you "
                "mean something more specific?"
            )

        hits.sort(key=lambda h: h.rank)  # bm25: smaller = better
        # Downrank senders the user marked "not what I meant" (stable — keeps the
        # bm25 order within each group; suppressed last, never hidden).
        if feedback_store is not None:
            hits.sort(
                key=lambda h: _is_suppressed_sender(
                    feedback_store, _sender_domain(h.from_domain, h.from_address)
                )
            )
        if prefer_sender_tokens:
            hits.sort(
                key=lambda h: not _sender_matches_tokens(
                    h.from_domain,
                    h.from_address,
                    prefer_sender_tokens,
                )
            )
        lines = [_fmt_hit(h) for h in hits[:_MAX_RESULTS]]
        top_statement_hit = next(
            (h for h in hits if _STATEMENT_SUBJECT_RE.search(h.subject or "")),
            hits[0] if hits else None,
        )
        auto_block = _auto_statement_block(top_statement_hit.id if top_statement_hit else None)
        shown_query = _display_query(query)
        header = (
            f"No exact match for '{shown_query}', but {len(hits)} email(s) mention related terms:\n"
            if broadened
            else f"{len(hits)} email(s) matched '{shown_query}':\n"
        )
        header = dropped_category + header
        return header + "\n".join(lines) + auto_block + _SEARCH_FEEDBACK_HINT

    def _list_by_category(args: dict[str, Any]) -> str:
        category = str(args.get("category") or args.get("input") or "").strip()
        if not category:
            return "Error: list_by_category requires a 'category' name (e.g. Newsletters)."
        since = _since(args)
        try:
            store = _store(data_dir)
            resolved: list[tuple[str, ResolvedCategory]] = [
                (a, resolve_category(store, category, a, labels_for=labels_for))
                for a in store.list_accounts()
            ]
            resolved = [(a, r) for a, r in resolved if r.found]
            if not resolved:
                return f"No category matches '{category}'. {category_menu(store)}"
            total = 0
            msgs: list[EmailMessage] = []
            for account_id, r in resolved:
                total += store.count_by_category(account_id, r.filter, since=since)
                msgs.extend(
                    store.list_by_category(account_id, r.filter, limit=_MAX_RESULTS, since=since)
                )
        except Exception as exc:  # noqa: BLE001
            _log_degraded("list_by_category", exc)
            return f"list_by_category failed: {exc}"
        paths = sorted({p for _, r in resolved for p in r.filter.paths})
        where = f"'{category}' ({', '.join(paths)})" if paths else f"'{category}'"
        window = f" from the last {args.get('since_days')} days" if since else ""
        if not msgs:
            return f"No emails in {where}{window}."
        msgs.sort(key=lambda m: m.received_at, reverse=True)
        shown = msgs[:_MAX_RESULTS]
        lines = [_fmt_msg(m) for m in shown]
        newest = f", newest {len(shown)} shown" if total > len(shown) else ""
        return f"{total} email(s) in {where}{window}{newest}:\n" + "\n".join(lines)

    def _offer_shortlist(question: str, hits: list[SearchHit]) -> None:
        """Record the shortlist as this session's open ``choice`` (ADR-0106).

        The options are what makes a later "1" answerable: the ids in the order shown.
        Owned by the ``email`` agent, which reads the picked id. Bookkeeping only — a
        shortlist that could not be recorded is still the right answer to show.
        """
        if continuations is None or current_session_id is None:
            return
        session_id = current_session_id()
        if not session_id:
            return
        try:
            continuations.ask(
                session_id,
                "email",
                kind="choice",
                question=question,
                intent="communication",
                payload={
                    "choices": [
                        {"message_id": hit.id, "subject": _clean(hit.subject)} for hit in hits
                    ]
                },
            )
        except Exception:  # never fail the answer over its bookkeeping
            logger.warning("read_email: could not record the shortlist choice", exc_info=True)

    def _read_email(args: dict[str, Any]) -> str:
        # Fetch ONE specific email's FULL body on demand (issue 0002 item C) so the
        # loop can summarize it — only the snippet is stored, so this needs a live
        # Gmail call. Find the best local match, then fetch its body by id.
        query = str(args.get("query") or args.get("input") or "").strip()
        sender_contains = str(args.get("from") or args.get("from_contains") or "").strip().lower()
        subject_contains = (
            str(args.get("subject") or args.get("subject_contains") or "").strip().lower()
        )
        message_id = str(args.get("message_id") or "").strip()
        pick_raw = args.get("pick")
        pick: int | None = None
        if pick_raw not in (None, ""):
            try:
                pick = int(str(pick_raw))
            except (TypeError, ValueError):
                return "'pick' must be an integer (1 = top result, 2 = second, etc.)."
            if pick <= 0:
                return "'pick' must be 1 or greater."

        if not query and not message_id:
            return (
                "Tell me which email to read by subject/sender (or a keyword), e.g. "
                "query='payment reminder' and optionally from='robinhood.com'."
            )
        try:
            store = _store(data_dir)
            matches: list[tuple[str, SearchHit]] = []
            if message_id:
                msg = store.get(message_id)
                if msg is None:
                    return f"No email found with message_id '{message_id}'."
                # Synthetic rank 0.0 for explicit-id selection.
                matches.append(
                    (
                        msg.account_id,
                        SearchHit(
                            id=msg.id,
                            received_at=msg.received_at,
                            subject=msg.subject,
                            from_address=msg.from_address,
                            from_domain=msg.from_domain,
                            classified_category=msg.classified_category,
                            snippet_highlighted=msg.snippet,
                            rank=0.0,
                        ),
                    )
                )
            else:
                queries = _fts_safe_lookup_queries(query)
                for account_id in store.list_accounts():
                    for candidate in queries:
                        try:
                            hits = store.search(candidate, account_id=account_id, limit=7)
                        except Exception:  # fall through to the next safer query form
                            logger.debug(
                                "read_email: search failed for %r on %s",
                                candidate,
                                account_id,
                                exc_info=True,
                            )
                            continue
                        for hit in hits:
                            matches.append((account_id, hit))
                        if hits:
                            break
        except Exception as exc:  # noqa: BLE001
            _log_degraded("read_email search", exc)
            return f"read_email failed: {exc}"
        if not matches:
            return f"No email found matching '{query}'."

        # De-duplicate by message-id (a message can appear via multiple candidate queries).
        best_by_id: dict[str, tuple[str, SearchHit]] = {}
        for account_id, hit in matches:
            cur = best_by_id.get(hit.id)
            if cur is None or hit.rank < cur[1].rank:
                best_by_id[hit.id] = (account_id, hit)
        deduped = list(best_by_id.values())

        def _passes_filters(hit: SearchHit) -> bool:
            if (
                sender_contains
                and sender_contains not in hit.from_address.lower()
                and (sender_contains not in (hit.from_domain or "").lower())
            ):
                return False
            if subject_contains and subject_contains not in hit.subject.lower():
                return False
            return True

        filtered = [pair for pair in deduped if _passes_filters(pair[1])]
        if not filtered:
            parts: list[str] = []
            if sender_contains:
                parts.append(f"from='{sender_contains}'")
            if subject_contains:
                parts.append(f"subject='{subject_contains}'")
            suffix = f" with {', '.join(parts)}" if parts else ""
            return f"I found emails for '{query}' but none{suffix}. Try a different sender/subject filter."

        filtered.sort(key=lambda ah: ah[1].rank)  # bm25: smaller = better
        if len(filtered) > 1 and pick is None and not message_id:
            top = filtered[:5]
            lines = [_fmt_read_candidate(h, idx=i) for i, (_a, h) in enumerate(top, start=1)]
            shortlist = (
                f"I found multiple emails matching '{query}'. Which one should I read?\n"
                + "\n".join(lines)
                + "\nReply with the number (e.g. 1), or refine with from/subject."
            )
            _offer_shortlist(shortlist, [h for _a, h in top])
            return shortlist

        if pick is not None:
            if pick > len(filtered):
                return f"Only {len(filtered)} match(es) found for '{query}'. Choose a number between 1 and {len(filtered)}."
            account_id, hit = filtered[pick - 1]
        else:
            account_id, hit = filtered[0]
        when = hit.received_at.strftime("%Y-%m-%d")
        preview = _clean(hit.snippet_highlighted.replace("<mark>", "").replace("</mark>", ""))

        from iris_personal.email.providers import mail_provider_for

        provider = mail_provider_for(account_id)
        if provider is None:
            return (
                f"Found '{hit.subject}' (from {hit.from_address}, {when}) but no mail provider "
                f"is mounted for this account, so I can't fetch its full content. "
                f"Preview: {preview}"
            )
        try:
            body = provider.fetch_message_body(account_id, hit.id)
        except Exception as exc:  # noqa: BLE001 — most likely a revoked token; degrade gracefully
            _log_degraded("read_email body fetch", exc)
            return (
                f"Found '{hit.subject}' (from {hit.from_address}, {when}) but couldn't fetch its "
                f"full content — your Gmail connection may need re-authentication "
                f"(`iris auth gmail login`). Preview: {preview}"
            )
        if not body:
            return (
                f"'{hit.subject}' (from {hit.from_address}, {when}) has no readable text body. "
                f"Preview: {preview}"
            )
        header = f"Email: {_clean(hit.subject)}\nFrom: {hit.from_address}  ({when})"
        # Summarise the body HERE (single-shot, reliable) rather than leaving it to
        # the ReAct loop, which often surfaced the raw body verbatim (issue 0002 C).
        summariser = summarize_llm or llm_call
        if summariser is None:
            return f"{header}\n\n{body}"
        prompt = (
            "Summarize this email in 2-3 sentences for the user. Use ONLY the content "
            "below — do not invent or add anything beyond what the email says.\n\n"
            f"Subject: {hit.subject}\nFrom: {hit.from_address}\n\n{body}\n\nSummary:"
        )
        try:
            summary = (summariser(prompt) or "").strip()
        except Exception as exc:  # noqa: BLE001 — fall back to the grounded body, never raise
            _log_degraded("read_email summary", exc)
            summary = ""
        return f"{header}\n\nSummary: {summary}" if summary else f"{header}\n\n{body}"

    def _analyze_inbox(args: dict[str, Any]) -> str:
        """Every email matching the topic, grouped by sender (see ``email.analysis``)."""
        from iris_personal.email.analysis import group_by_sender, render_groups

        query = str(args.get("query") or args.get("topic") or args.get("input") or "").strip()
        if not query:
            return (
                "Error: analyze_inbox needs a 'query' — the topic the user asked about "
                "(e.g. 'credit card', 'subscription', 'statement')."
            )
        user_turn = current_query() if current_query is not None else ""
        if not _query_grounded(query, user_turn):
            return (
                f"Error: '{_display_query(query)}' is not in the user's question. Use the "
                "topic words the user actually said."
            )
        since = _since(args) or _infer_since_from_query(user_turn)
        try:
            store = _store(data_dir)
            accounts = store.list_accounts()
        except Exception as exc:  # noqa: BLE001
            _log_degraded("analyze_inbox account list", exc)
            return f"analyze_inbox failed: {exc}"

        def _run(fts_query: str) -> list[SearchHit]:
            found: list[SearchHit] = []
            for account_id in accounts:
                found.extend(
                    store.search(
                        fts_query, account_id=account_id, since=since, limit=_ANALYSIS_MAX_HITS
                    )
                )
            return found

        hits: list[SearchHit] = []
        for candidate in _fts_safe_lookup_queries(query):
            try:
                hits = _run(candidate)
            except Exception:  # punctuation can trip FTS5; try the next form
                logger.debug("analyze_inbox: FTS query %r failed", candidate, exc_info=True)
                continue
            if hits:
                break
        broadened = False
        if not hits:
            tokens = _content_tokens(query)
            if len(tokens) > 1:
                try:
                    hits = _run(" OR ".join(f"{t}*" for t in tokens))
                except Exception as exc:  # noqa: BLE001 — broaden is best-effort
                    _log_degraded("analyze_inbox broaden", exc)
                    hits = []
                broadened = bool(hits)
        return render_groups(_display_query(query), group_by_sender(hits), broadened=broadened)

    def _find_attachment(args: dict[str, Any]) -> str:
        query = str(args.get("query") or args.get("input") or "").strip()
        return find_email_attachment(
            data_dir=data_dir, query=query, since_days=args.get("since_days")
        )

    tools = [
        ToolSpec(
            name="inbox_digest",
            description=(
                "Summarise the whole inbox — counts + the most attention-worthy messages "
                "across all accounts. This is the DEFAULT for any general inbox status ask "
                "that names NO specific topic, sender, or category — e.g. 'how is my inbox "
                "today', 'what's in my inbox', 'any new mail', 'anything important'. If the "
                "user did not name a subject to filter on, use this, NOT search_inbox. No "
                "arguments."
            ),
            call=_inbox_digest,
        ),
        ToolSpec(
            name="search_inbox",
            description=(
                "Find emails about a TOPIC the user named — by keyword AND by meaning "
                "(e.g. 'AI article', 'invoice from Amazon', 'any pending payments'). "
                "Matches even when the exact words aren't in the email, so it's the right "
                "tool for conceptual asks too. Use ONLY when the user's message names the "
                "subject/sender/keyword — do NOT invent a query; if no topic was named, use "
                'inbox_digest instead. Args: {"query": str, "since_days"?: int, '
                '"category"?: str}. Returns matching emails, most-relevant first.'
            ),
            call=_search_inbox,
        ),
        ToolSpec(
            name="read_email",
            description=(
                "Fetch and read the FULL content of ONE specific email so you can summarize "
                "it or answer questions about what it says. Use when the user refers to a "
                "particular email — 'what does it say', 'summarize that email', 'what's in the "
                "email from X'. Bodies aren't stored, so this fetches the full text live. "
                "If multiple matches exist, it returns a numbered shortlist and asks which "
                "one to read. "
                'Args: {"query": str, "pick"?: int, "from"?: str, "subject"?: str, "message_id"?: str}.'
            ),
            call=_read_email,
        ),
        ToolSpec(
            name="list_by_category",
            description=(
                "List recent emails in a CATEGORY the user named — 'my promotions', "
                "'promo emails', 'newsletters', 'credit card emails' — without a text query. "
                "Pass the user's own word as the category; it is matched against the "
                "categories in their mail, and a name that matches none lists them. "
                'Args: {"category": str, "since_days"?: int}.'
            ),
            call=_list_by_category,
        ),
        ToolSpec(
            name="analyze_inbox",
            description=(
                "ANALYSE the user's email to answer a which / how many / list-all question "
                "about their MAIL — 'how many subscriptions am I paying for', 'which "
                "companies send me invoices', 'which banks email me statements', 'who "
                "emails me about insurance'. Reads EVERY matching email (not just the top "
                "few) and groups them by sender with counts, dates and example subjects, so "
                "you can name each distinct subscription, company or sender once. "
                "Use this instead of search_inbox whenever the answer is a list or a count "
                'of things rather than one email. Args: {"query": str (the topic words the '
                'user said, e.g. "credit card"), "since_days"?: int}.'
            ),
            call=_analyze_inbox,
        ),
        ToolSpec(
            name="find_attachment",
            description=(
                "Locate an email ATTACHMENT / document the user wants — 'get my passport "
                "copy', 'find the invoice PDF', 'where's the resume I was sent'. Matches the "
                "attachment file name and the email's subject/sender. Use this (NOT "
                "search_inbox) whenever the user asks for a file/document/copy/attachment from "
                'email. Args: {"query": str, "since_days"?: int}. Returns each matching '
                "attachment with its sender and date so the user can open it."
            ),
            call=_find_attachment,
        ),
    ]
    # ADR-0118 step 5: the first destructive tool, and its undo.
    from iris_personal.email.trash_tools import build_trash_tools

    def _window_days(args: dict[str, Any]) -> int | None:
        since = _since(args)
        if since is not None:
            return int(str(args.get("since_days")))
        return _infer_days_from_query(current_query() if current_query is not None else "")

    tools.extend(
        build_trash_tools(data_dir=data_dir, labels_for=labels_for, window_days=_window_days)
    )
    # ADR-0118 amendment: a write approved per call on the pinned card.
    from iris_personal.email.send_tools import build_send_tools

    tools.extend(build_send_tools(data_dir=data_dir, current_query=current_query))
    return tools


__all__ = ["build_email_tools"]
