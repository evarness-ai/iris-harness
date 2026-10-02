"""Gmail-side fetch logic for the ``gmail-inbox`` skill (Phase 1 Track 1C).

Two entry points for callers:

  fetch_new_emails(account_id, ...)
    The high-level operation. Resolves credentials, picks delta-vs-
    cold-start, parses Gmail API responses into EmailMessage rows,
    persists them via EmailStore, and rotates the sync_cursor.

  FetchResult
    The return shape — count, new cursor, and a flag indicating whether
    we fell back from delta sync to cold-start (cursor was older than
    Gmail's history retention).

Sync strategy:
  - On startup with no cursor → cold-start: messages.list(q="newer_than:30d"),
    paginate up to max_messages, batch-get(format="metadata"). Write the
    profile's current historyId as the new cursor.
  - On startup with a cursor → delta: history.list(startHistoryId=cursor),
    collect messagesAdded ids, batch-get(format="metadata"). Use the
    response's historyId as the new cursor.
  - If Gmail returns 410 ("history not available") → fall back to
    cold-start automatically.

What's deliberately NOT here (deferred per canonical §3.1):
  - Body fetching (format="full"). Bodies are fetched on demand by
    downstream consumers, not pre-cached.
  - Attachment metadata. Requires format="full"; defer until Phase 3
    finance-statements skill needs it.
  - Archive operations. Trash and restore (ADR-0118 step 5), a plain-text send
    (``send_message``, ADR-0118 amendment) and the email judge's labels
    (``ensure_labels`` / ``modify_labels``, loop-proof PR 5) are the only writes, below.

Mailbox writes are gated (OSS plan R4): creating a label, adding or removing labels,
trash, untrash and the restore's label put-back each run inside the email library's one
per-account gate (``iris_personal.email.write_approvals.mailbox_write``): the approval is
checked before any Gmail request, raising ``PermissionError`` until the owner approved
writes for the account (``iris email writes approve``, or email setup's label preview),
and what reached the mailbox leaves one audit row (the proof bundle's observation, R14).
Sending is not one of them: it adds a message rather than changing the owner's mail,
and every send already waits for the owner's approval of that message (``send_email``).

Label read-back (loop-proof PR 5): the delta also asks history.list for
``labelAdded`` / ``labelRemoved``. A message already in email.db whose *user* labels
changed has its current labels re-read (``format=minimal``), written to
``emails.labels``, and returned in ``FetchResult.label_changes``; the sweep hands them
on as ``email.labels_changed``. A cold start (no cursor, or a stale one) reads no
label changes: there is no "since" to compare with.
"""

from __future__ import annotations

import base64
import html as html_lib
import json
import logging
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.header import decode_header
from email.message import EmailMessage as MIMEMessage
from email.policy import SMTP
from email.utils import getaddresses
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from iris_harness.sdk.process_state import track_globals
from iris_personal.email.provider_api import (
    EmailMessage,
    FetchResult,
    MailSyncStore,
    ProgressFn,
    default_sync_store,
)

# Every write goes through the email library's gate-and-record (R4 + R14): the owner's
# approval first, then one audit row of what reached the mailbox.
from iris_personal.email.write_approvals import mailbox_write

from .gmail_oauth import load_credentials
from .vendor_categories import vendor_category_for

logger = logging.getLogger(__name__)

GMAIL_PROVIDER = "gmail"
GMAIL_HISTORY_CURSOR_KIND = "gmail_history_id"
DEFAULT_MAX_MESSAGES = 100
DEFAULT_COLD_START_DAYS = 30
GMAIL_LIST_PAGE_SIZE = 100  # cap per page; Gmail allows up to 500
GMAIL_METADATA_HEADERS_OF_INTEREST = ("Message-ID", "In-Reply-To", "References")


class StaleCursor(Exception):
    """Gmail returned 404/410 — historyId is too old, fall back to cold-start."""


# ``FetchResult`` is the core interface's type (``email.providers``); re-exported here
# so the name keeps working for the tests and callers that imported it from this module.


def fetch_new_emails(
    account_id: str,
    *,
    store: MailSyncStore | None = None,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    cold_start_days: int = DEFAULT_COLD_START_DAYS,
    progress: ProgressFn | None = None,
) -> FetchResult:
    """Sync new mail for one account. Idempotent on re-runs."""
    s = store if store is not None else default_sync_store()
    s.ensure_schema()

    creds = load_credentials(_address_from_account_id(account_id))
    if creds is None:
        raise RuntimeError(
            f"No Gmail credentials for {account_id}. "
            "Run `iris auth gmail login --user <address>` first."
        )

    service = build("gmail", "v1", credentials=creds, cache_discovery=False)

    existing_cursor = s.get_cursor(GMAIL_PROVIDER, account_id, GMAIL_HISTORY_CURSOR_KIND)
    fell_back = False

    relabelled: list[str] = []
    if existing_cursor:
        try:
            message_ids, new_cursor, relabelled = _history_delta(
                service, existing_cursor, max_messages
            )
        except StaleCursor:
            logger.info(
                "gmail-fetch: cursor %s for %s is stale; falling back to cold-start",
                existing_cursor,
                account_id,
            )
            fell_back = True
            message_ids, new_cursor = _cold_start(service, max_messages, cold_start_days)
    else:
        message_ids, new_cursor = _cold_start(service, max_messages, cold_start_days)

    messages = _batch_get_metadata(service, message_ids, account_id, progress=progress)
    persisted = s.upsert_many(messages)
    label_changes = _read_label_changes(service, s, relabelled, exclude=set(message_ids))

    if new_cursor:
        s.set_cursor(GMAIL_PROVIDER, account_id, GMAIL_HISTORY_CURSOR_KIND, new_cursor)

    return FetchResult(
        account_id=account_id,
        fetched=persisted,
        new_message_ids=tuple(m.id for m in messages),
        new_cursor=new_cursor,
        fell_back_to_cold_start=fell_back,
        label_changes=label_changes,
    )


def fetch_message_body(account_id: str, message_id: str, *, max_chars: int = 4000) -> str:
    """Fetch ONE message's text body from Gmail on demand (issue 0002 item C).

    Bodies are NOT persisted (only the snippet is — ADR-0026 §3.1), so reading a
    specific email's full content requires a live API call (``format=full``).
    Returns the decoded text/plain body, falling back to stripped HTML, truncated
    to ``max_chars``. Raises ``RuntimeError`` on a missing/revoked credential so
    the caller can surface a re-auth hint instead of a stack trace.
    """
    creds = load_credentials(_address_from_account_id(account_id))
    if creds is None:
        raise RuntimeError(
            f"No valid Gmail credentials for {account_id}. "
            "Run `iris auth gmail login --user <address>` to reconnect."
        )
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    payload = service.users().messages().get(userId="me", id=message_id, format="full").execute()
    return _extract_body_text(payload.get("payload", {}))[:max_chars].strip()


# ─── Trash and restore (ADR-0118 step 5) ─────────────────────────────


class GmailScopeError(PermissionError):
    """The account's token predates gmail.modify, so it can read but not trash."""


def _service_for(account_id: str) -> Any:
    address = _address_from_account_id(account_id)
    creds = load_credentials(address)
    if creds is None:
        raise RuntimeError(
            f"No valid Gmail credentials for {account_id}. "
            "Run `iris auth gmail login --user <address>` to reconnect."
        )
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


# Requests per Gmail batch HTTP call. The items of a batch run concurrently on
# Google's side, and 50 at once drew "429 Too many concurrent requests for user"
# for most of a 200-message trash (2026-09-22); 10 stays under the per-user limit.
_BATCH_SIZE = 10
# Rate-limited items are retried, waiting this long (seconds) before each round.
_RETRY_WAITS = (1.0, 2.0, 4.0, 8.0)
# Gmail signals a rate limit with 429, or with 403 and one of these reasons; a 403
# for any other reason (insufficientPermissions) is the grant, and retrying is futile.
_RATE_LIMIT_REASONS = frozenset({"rateLimitExceeded", "userRateLimitExceeded"})


def _status(exc: HttpError) -> int | None:
    code = getattr(exc, "status_code", None) or getattr(exc.resp, "status", None)
    return int(code) if code is not None else None


def _reasons(exc: HttpError) -> set[str]:
    """Gmail's ``errors[].reason`` values. googleapiclient parses them into
    ``error_details`` only when the body also has a ``message``; read the body as well."""
    details = getattr(exc, "error_details", None)
    if not isinstance(details, list) or not details:
        try:
            details = json.loads(exc.content or b"{}").get("error", {}).get("errors", [])
        except (ValueError, AttributeError):
            details = []
    return {str(d.get("reason")) for d in details if isinstance(d, dict) and d.get("reason")}


def _is_rate_limited(exc: HttpError) -> bool:
    return _status(exc) == 429 or (
        _status(exc) == 403 and bool(_reasons(exc) & _RATE_LIMIT_REASONS)
    )


def _is_forbidden(exc: HttpError) -> bool:
    """The grant cannot do this (read-only token): 403 that is not a rate limit."""
    return _status(exc) == 403 and not _is_rate_limited(exc)


def _batched(
    account_id: str, message_ids: list[str], request_for: Callable[[Any, str], Any], *, what: str
) -> dict[str, Any]:
    """Send one request per id in Gmail batch HTTP calls; return ``{id: response}``.

    A round trip per ``_BATCH_SIZE`` messages instead of one per message: the owner's
    first real trash of 200 promotions took 114 s and its restore 160 s one call at a
    time, past the chat's time limit (2026-09-22). Rate-limited items are retried with
    a growing wait. A read-only grant (403 insufficientPermissions) is refused for the
    whole account, so it surfaces on the first batch, before any message moves, and
    says how to fix it. Any other per-message error skips that message (it may already
    be gone) and the caller reports the difference.
    """
    service = _service_for(account_id)
    messages = service.users().messages()
    results: dict[str, Any] = {}
    forbidden: list[HttpError] = []
    limited: list[str] = []

    def _done(request_id: str, response: Any, exception: Exception | None) -> None:
        if exception is None:
            results[request_id] = response
        elif isinstance(exception, HttpError) and _is_rate_limited(exception):
            limited.append(request_id)
        elif isinstance(exception, HttpError) and _is_forbidden(exception):
            forbidden.append(exception)
        else:
            logger.warning("gmail-%s: skipping %s (%s)", what, request_id, exception)

    pending = list(message_ids)
    for wait in (0.0, *_RETRY_WAITS):
        if not pending:
            break
        if wait:
            time.sleep(wait)
        limited.clear()
        for start in range(0, len(pending), _BATCH_SIZE):
            batch = service.new_batch_http_request(callback=_done)
            for mid in pending[start : start + _BATCH_SIZE]:
                batch.add(request_for(messages, mid), request_id=mid)
            batch.execute()
            if forbidden:
                raise GmailScopeError(
                    f"{account_id} was connected read-only. Run "
                    f"`iris auth gmail login --user {_address_from_account_id(account_id)}` "
                    "again and allow IRIS to manage your mail, then retry."
                ) from forbidden[0]
        pending = list(limited)
    if pending:
        logger.warning("gmail-%s: %d still rate-limited after retries", what, len(pending))
    return results


def message_labels(account_id: str, message_ids: list[str]) -> dict[str, list[str]]:
    """Each message's current Gmail labels, read just before it is trashed so a restore
    can put back what the trash takes away."""
    got = _batched(
        account_id,
        message_ids,
        lambda m, mid: m.get(userId="me", id=mid, format="minimal"),
        what="labels",
    )
    return {mid: list(resp.get("labelIds", [])) for mid, resp in got.items()}


def trash_messages(account_id: str, message_ids: list[str]) -> list[str]:
    """Move messages to Gmail's Trash (kept 30 days). Returns the ids moved.

    Gmail's trash also takes the message out of INBOX (or SPAM); ``untrash`` does not
    put it back, which is why ``message_labels`` is read first.
    """
    with mailbox_write(account_id, "move mail to Trash", op="trash") as tally:
        done = _batched(
            account_id, message_ids, lambda m, mid: m.trash(userId="me", id=mid), what="trash"
        )
        moved = [mid for mid in message_ids if mid in done]
        tally.add(len(moved))
    return moved


def untrash_messages(account_id: str, message_ids: list[str]) -> dict[str, list[str]]:
    """Take messages back out of Trash. Returns ``{id: labels after}`` for those restored."""
    with mailbox_write(account_id, "restore mail from Trash", op="restore") as tally:
        done = _batched(
            account_id, message_ids, lambda m, mid: m.untrash(userId="me", id=mid), what="untrash"
        )
        tally.add(len(done))
    return {mid: list(resp.get("labelIds", [])) for mid, resp in done.items()}


def add_labels(account_id: str, labels_by_message: dict[str, set[str]]) -> None:
    """Add labels back, one ``batchModify`` per distinct label set (up to 1,000 ids each)."""
    with mailbox_write(account_id, "put labels back", op="restore_labels") as tally:
        groups: dict[frozenset[str], list[str]] = {}
        for mid, labels in labels_by_message.items():
            if labels:
                groups.setdefault(frozenset(labels), []).append(mid)
        if not groups:
            return
        messages = _service_for(account_id).users().messages()
        for label_set, ids in groups.items():
            for start in range(0, len(ids), 1000):
                chunk = ids[start : start + 1000]
                messages.batchModify(
                    userId="me", body={"ids": chunk, "addLabelIds": sorted(label_set)}
                ).execute()
                tally.add(len(chunk))


# ─── Labels (loop-proof PR 5: the email judge's IRIS/* labels) ───────

# account id → {label name: label id}, filled by ``ensure_labels``. In-process only: a
# restart re-lists once per account.
_LABEL_IDS: dict[str, dict[str, str]] = {}
# Gmail takes at most 1,000 ids per users.messages.batchModify.
_MODIFY_MAX_IDS = 1000


def _scope_error(account_id: str) -> GmailScopeError:
    return GmailScopeError(
        f"{account_id} was connected read-only. Run "
        f"`iris auth gmail login --user {_address_from_account_id(account_id)}` "
        "again and allow IRIS to manage your mail, then retry."
    )


def forget_labels(account_id: str | None = None) -> None:
    """Drop the cached label ids (one account, or all): the next call re-lists."""
    if account_id is None:
        _LABEL_IDS.clear()
    else:
        _LABEL_IDS.pop(account_id, None)


def ensure_labels(
    account_id: str, names: list[str], *, service: Any | None = None
) -> dict[str, str]:
    """``{name: label id}`` for every name, creating the labels that do not exist yet.

    Lists the account's labels once and caches the ids in-process. A nested name
    (``Parent/Child``) is how Gmail nests labels; the API accepts the child without
    its parent, but Gmail then shows it flat until a ``Parent`` label exists, so each
    missing parent is created first. Returns the requested names only.

    Listing is a read; creating a missing label is a mailbox write, so the write gate
    is asked right before the first create (read-back of labels that already exist
    needs no approval).
    """
    cached = _LABEL_IDS.get(account_id, {})
    if all(n in cached for n in names):
        return {n: cached[n] for n in names}
    svc = service if service is not None else _service_for(account_id)
    labels_api = svc.users().labels()
    try:
        listing = labels_api.list(userId="me").execute()
        known = {str(lb["name"]): str(lb["id"]) for lb in listing.get("labels", [])}
        wanted: list[str] = []
        for name in names:
            parts = name.split("/")
            for depth in range(1, len(parts) + 1):
                prefix = "/".join(parts[:depth])
                if prefix not in wanted:
                    wanted.append(prefix)
        for name in wanted:
            if name in known:
                continue
            with mailbox_write(account_id, "create the label " + name, op="create_label") as t:
                created = labels_api.create(
                    userId="me",
                    body={
                        "name": name,
                        "labelListVisibility": "labelShow",
                        "messageListVisibility": "show",
                    },
                ).execute()
                t.add(1)
            known[name] = str(created["id"])
            logger.info("gmail-labels: created %s for %s", name, account_id)
    except HttpError as exc:
        if _is_forbidden(exc):
            raise _scope_error(account_id) from exc
        raise
    _LABEL_IDS[account_id] = known
    return {n: known[n] for n in names}


def modify_labels(
    account_id: str,
    message_ids: list[str],
    add_ids: list[str],
    remove_ids: list[str],
    *,
    service: Any | None = None,
) -> int:
    """Add and remove labels on messages: one ``batchModify`` per 1,000 ids. Returns
    how many ids were sent. A rate limit is retried with the trash's growing waits; a
    403 that is not one is the grant (``GmailScopeError``). A 400/404 (a label deleted
    in Gmail since it was cached) drops the cache so the next run re-lists, and raises.
    """
    with mailbox_write(account_id, "change labels", op="label") as tally:
        if not message_ids or not (add_ids or remove_ids):
            return 0
        svc = service if service is not None else _service_for(account_id)
        messages = svc.users().messages()
        body_labels: dict[str, list[str]] = {}
        if add_ids:
            body_labels["addLabelIds"] = list(add_ids)
        if remove_ids:
            body_labels["removeLabelIds"] = list(remove_ids)
        for start in range(0, len(message_ids), _MODIFY_MAX_IDS):
            chunk = message_ids[start : start + _MODIFY_MAX_IDS]
            for attempt, wait in enumerate((0.0, *_RETRY_WAITS)):
                if wait:
                    time.sleep(wait)
                try:
                    messages.batchModify(userId="me", body={"ids": chunk, **body_labels}).execute()
                    break
                except HttpError as exc:
                    if _is_rate_limited(exc) and attempt < len(_RETRY_WAITS):
                        continue
                    if _is_forbidden(exc):
                        raise _scope_error(account_id) from exc
                    if _status(exc) in (400, 404):
                        forget_labels(account_id)
                    raise
            tally.add(len(chunk))
    return tally.count


def fetch_messages_metadata(account_id: str, message_ids: list[str]) -> list[EmailMessage]:
    """Header metadata for specific messages, to put a restored message back in the store."""
    got = _batched(
        account_id,
        message_ids,
        lambda m, mid: m.get(userId="me", id=mid, format="metadata"),
        what="metadata",
    )
    out: list[EmailMessage] = []
    for mid in message_ids:
        if mid not in got:
            continue
        try:
            out.append(_parse_message(got[mid], account_id))
        except (ValueError, KeyError) as exc:
            logger.debug("gmail-fetch: skipping %s (parse error: %s)", mid, exc)
    return out


# ─── Send (ADR-0118 amendment: send_email) ───────────────────────────


def build_raw_message(
    *,
    sender: str,
    to: list[str],
    cc: list[str],
    subject: str,
    body: str,
    in_reply_to: str | None = None,
    references: str | None = None,
) -> str:
    """A plain-text RFC 2822 message, base64url-encoded — the ``raw`` Gmail's
    ``users.messages.send`` takes. No HTML part, no attachments. ``in_reply_to`` and
    ``references`` thread a reply for every mail client, not only Gmail."""
    msg = MIMEMessage(policy=SMTP)
    msg["From"] = sender
    msg["To"] = ", ".join(to)
    if cc:
        msg["Cc"] = ", ".join(cc)
    msg["Subject"] = subject
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
        msg["References"] = f"{references} {in_reply_to}".strip() if references else in_reply_to
    msg.set_content(body)  # text/plain; utf-8, transfer-encoded as needed
    return base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")


def send_message(
    account_id: str,
    *,
    to: list[str],
    cc: list[str],
    subject: str,
    body: str,
    in_reply_to: str | None = None,
    references: str | None = None,
    thread_id: str | None = None,
    service: Any | None = None,
) -> str:
    """Send one plain-text email from ``account_id``; return Gmail's id for it.

    ``thread_id`` puts a reply in the original's Gmail thread. gmail.modify covers
    ``messages.send`` (Google's discovery document lists it among send's scopes), so the
    grant trash already asks for is enough; a token from before it answers 403, which
    becomes the same "connect again" message trash gives.
    """
    raw = build_raw_message(
        sender=_address_from_account_id(account_id),
        to=to,
        cc=cc,
        subject=subject,
        body=body,
        in_reply_to=in_reply_to,
        references=references,
    )
    request: dict[str, Any] = {"raw": raw}
    if thread_id:
        request["threadId"] = thread_id
    svc = service if service is not None else _service_for(account_id)
    try:
        sent = svc.users().messages().send(userId="me", body=request).execute()
    except HttpError as exc:
        if getattr(exc, "status_code", None) == 403 or getattr(exc.resp, "status", None) == 403:
            raise GmailScopeError(
                f"{account_id} was connected read-only. Run "
                f"`iris auth gmail login --user {_address_from_account_id(account_id)}` "
                "again and allow IRIS to manage your mail, then retry."
            ) from exc
        raise
    return str(sent.get("id") or "")


# ─── Internals ────────────────────────────────────────────────────────


# A text/plain part with fewer words than this share of the HTML part's is a stub.
_PLAIN_STUB_RATIO = 0.5


def _extract_body_text(payload: dict[str, Any]) -> str:
    """Text content of a Gmail message payload: text/plain, unless it is a stub.

    A card issuer's statement email can carry a text/plain part that only says "Please
    visit the following link to view your message"; the figures (statement balance,
    minimum, due date) are in the HTML part alone. Preferring text/plain whenever it
    existed made the finance reader record "statement notice states no figures" and chat
    answer "check your card account" (2026-09-28). So when the plain part is much
    shorter than the HTML's text, the HTML's text is the body. Recurses into
    multipart payloads.
    """
    plain = _find_part_text(payload, "text/plain")
    html = _find_part_text(payload, "text/html")
    html_text = _strip_html(html) if html else ""
    if plain and _word_count(plain) >= _PLAIN_STUB_RATIO * _word_count(html_text):
        return plain
    return html_text or plain


_URL_RE = re.compile(r"https?://\S+")


def _word_count(text: str) -> int:
    """Words a reader sees, links left out: the stub's long tracking URLs made it
    look as long as the HTML text by characters."""
    return len(re.findall(r"\w+", _URL_RE.sub(" ", text)))


def _find_part_text(part: dict[str, Any], mime: str) -> str:
    if part.get("mimeType") == mime:
        data = (part.get("body") or {}).get("data")
        if data:
            return _b64url_decode(data)
    for sub in part.get("parts") or []:
        found = _find_part_text(sub, mime)
        if found:
            return found
    return ""


def _b64url_decode(data: str) -> str:
    try:
        return base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — malformed body data is non-fatal
        return ""


def _strip_html(html: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    # Entities after the tags are gone: "&#36;12.34" must read as "$12.34".
    return re.sub(r"\s+", " ", html_lib.unescape(text)).strip()


def _address_from_account_id(account_id: str) -> str:
    """email_accounts.id slugs look like 'gmail:user@gmail.com' — return the
    bare email portion. See ADR-0016 for the slug shape."""
    if ":" in account_id:
        return account_id.split(":", 1)[1]
    return account_id


def _history_delta(
    service: Any, start_history_id: str, max_messages: int
) -> tuple[list[str], str | None, list[str]]:
    """Walk history.list to gather messagesAdded ids, a new cursor, and the ids whose
    user labels were added or removed (the label read-back, in first-seen order).

    Raises StaleCursor on Gmail 404/410 (cursor too old).
    """
    message_ids: list[str] = []
    relabelled: dict[str, None] = {}
    page_token: str | None = None
    last_history_id: str | None = start_history_id

    while len(message_ids) < max_messages:
        request = (
            service.users()
            .history()
            .list(
                userId="me",
                startHistoryId=start_history_id,
                historyTypes=_HISTORY_TYPES,
                pageToken=page_token,
                maxResults=GMAIL_LIST_PAGE_SIZE,
            )
        )
        try:
            response = request.execute()
        except HttpError as exc:
            status = getattr(getattr(exc, "resp", None), "status", None)
            if status in (404, 410):
                raise StaleCursor(str(exc)) from exc
            raise

        for event in response.get("history", []):
            for key in ("labelsAdded", "labelsRemoved"):
                for change in event.get(key, []):
                    msg_id = (change.get("message") or {}).get("id")
                    if msg_id and any(_is_user_label(x) for x in change.get("labelIds", [])):
                        relabelled[msg_id] = None
            for added in event.get("messagesAdded", []):
                msg_ref = added.get("message", {})
                msg_id = msg_ref.get("id")
                if msg_id:
                    message_ids.append(msg_id)
                    if len(message_ids) >= max_messages:
                        break
            if len(message_ids) >= max_messages:
                break

        last_history_id = response.get("historyId", last_history_id)
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return message_ids, last_history_id, list(relabelled)


# What the delta asks history.list for: new mail, and label changes for the read-back.
_HISTORY_TYPES = ["messageAdded", "labelAdded", "labelRemoved"]
# Gmail's system label ids (INBOX, UNREAD, CATEGORY_PROMOTIONS, ...) are upper case;
# a label the account created has a generated id (``Label_12``). Reading a message is
# a change to UNREAD, so only user-label changes are read back.
_SYSTEM_LABEL = re.compile(r"^[A-Z_]+$")


def _is_user_label(label_id: str) -> bool:
    return bool(label_id) and not _SYSTEM_LABEL.match(label_id)


def _read_label_changes(
    service: Any, store: MailSyncStore, message_ids: list[str], *, exclude: set[str]
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Current labels of relabelled messages already in email.db, written back to
    ``emails.labels``. New mail (``exclude``) was just fetched whole; a message not in
    the store, or gone from Gmail, is skipped."""
    out: list[tuple[str, tuple[str, ...]]] = []
    for mid in message_ids:
        if mid in exclude or store.get(mid) is None:
            continue
        try:
            payload = (
                service.users().messages().get(userId="me", id=mid, format="minimal").execute()
            )
        except HttpError as exc:
            logger.debug("gmail-fetch: label read-back skipping %s (HttpError: %s)", mid, exc)
            continue
        labels = tuple(payload.get("labelIds", []))
        store.set_labels(mid, labels)
        out.append((mid, labels))
    return tuple(out)


def _cold_start(
    service: Any, max_messages: int, cold_start_days: int
) -> tuple[list[str], str | None]:
    """messages.list paginate + read current profile.historyId for the cursor."""
    query = f"newer_than:{cold_start_days}d"
    message_ids: list[str] = []
    page_token: str | None = None

    while len(message_ids) < max_messages:
        request = (
            service.users()
            .messages()
            .list(
                userId="me",
                q=query,
                pageToken=page_token,
                maxResults=min(GMAIL_LIST_PAGE_SIZE, max_messages - len(message_ids)),
            )
        )
        response = request.execute()
        for m in response.get("messages", []):
            if len(message_ids) >= max_messages:
                break
            mid = m.get("id")
            if mid:
                message_ids.append(mid)
        page_token = response.get("nextPageToken")
        if not page_token or len(message_ids) >= max_messages:
            break

    profile = service.users().getProfile(userId="me").execute()
    new_cursor = profile.get("historyId")
    return message_ids, new_cursor


def _batch_get_metadata(
    service: Any,
    message_ids: list[str],
    account_id: str,
    *,
    progress: ProgressFn | None = None,
) -> list[EmailMessage]:
    """Fetch metadata for each id; skip individual failures with a debug log."""
    out: list[EmailMessage] = []
    total = len(message_ids)
    for i, mid in enumerate(message_ids, start=1):
        try:
            payload = (
                service.users().messages().get(userId="me", id=mid, format="metadata").execute()
            )
        except HttpError as exc:
            logger.debug("gmail-fetch: skipping %s (HttpError: %s)", mid, exc)
            continue
        finally:
            if progress is not None:
                progress(i / total, f"fetched {i}/{total}")
        try:
            out.append(_parse_message(payload, account_id))
        except (ValueError, KeyError) as exc:
            logger.debug("gmail-fetch: skipping %s (parse error: %s)", mid, exc)
            continue
    return out


def _parse_message(payload: dict[str, Any], account_id: str) -> EmailMessage:
    """Gmail API response → EmailMessage. Bodies + attachments are skipped."""
    headers = {h["name"]: h["value"] for h in payload.get("payload", {}).get("headers", [])}
    from_raw = headers.get("From", "")
    subject = _decode_subject(headers.get("Subject", ""))

    internal_ms = int(payload.get("internalDate", "0"))
    received_at = (
        datetime.fromtimestamp(internal_ms / 1000, tz=UTC) if internal_ms > 0 else datetime.now(UTC)
    )

    labels = tuple(payload.get("labelIds", []))
    return EmailMessage(
        id=payload["id"],
        provider="gmail",
        account_id=account_id,
        thread_id=payload.get("threadId"),
        from_address=from_raw or "unknown@unknown",
        from_domain=_extract_domain(from_raw) if from_raw else None,
        to=_parse_addresses(headers.get("To", "")),
        cc=_parse_addresses(headers.get("Cc", "")),
        subject=subject,
        received_at=received_at,
        snippet=payload.get("snippet", "")[:400],
        body_text=None,
        body_html=None,
        labels=labels,
        attachments=(),
        headers_subset={
            k: v for k, v in headers.items() if k in GMAIL_METADATA_HEADERS_OF_INTEREST
        },
        # Gmail's tab (Promotions, Social, ...) as an IRIS path, from the plugin's YAML.
        vendor_category=vendor_category_for(labels),
    )


def _decode_subject(raw: str) -> str:
    """Decode RFC 2047 MIME-encoded subjects (e.g. '=?UTF-8?B?...?=')."""
    if not raw:
        return ""
    parts = []
    for fragment, charset in decode_header(raw):
        if isinstance(fragment, bytes):
            try:
                parts.append(fragment.decode(charset or "utf-8", errors="replace"))
            except (LookupError, UnicodeDecodeError):
                parts.append(fragment.decode("utf-8", errors="replace"))
        else:
            parts.append(fragment)
    return "".join(parts).strip()


def _parse_addresses(raw: str) -> tuple[str, ...]:
    """Comma-separated 'Name <email>' list → tuple of bare email strings.

    Drops entries with no '@' (unparseable).
    """
    if not raw:
        return ()
    pairs = getaddresses([raw])
    return tuple(addr for _name, addr in pairs if "@" in addr)


def _extract_domain(from_raw: str) -> str | None:
    """Pull bare domain from a From header value."""
    if "@" not in from_raw:
        return None
    tail = from_raw.split("@", 1)[1]
    return tail.split(">", 1)[0].split()[0].strip().lower() or None


# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_LABEL_IDS")
