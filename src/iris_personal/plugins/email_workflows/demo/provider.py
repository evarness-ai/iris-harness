"""The demo mailbox: a ``MailProvider`` (and ``LabellingProvider``) over the corpus.

It serves ``corpus.json`` the way a real provider serves a mailbox, so everything
downstream -- the sweep, the judge's queue, the judge's live body read, the digest --
runs unchanged. It talks to nothing: no credentials, no network.

The mailbox itself is read-only. ``trash_messages`` and ``restore_messages`` refuse
with ``PermissionError`` (what a read-only grant raises), and ``send_message`` is not
offered at all. Labels are the one thing it keeps, in its own state file beside the
store it syncs (``demo_mailbox.json`` in the data dir), never anywhere else -- and only
once the owner approved mailbox writes for the account (``email.write_approvals``, OSS
plan R4), exactly as a real mailbox: the demo is where the approval step is tried.

Times are relative to an *anchor*: the moment this mailbox was first opened, kept in
the state file, so the inbox reads as current on first run and reads the same on every
run after.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta, tzinfo
from pathlib import Path
from threading import Lock
from typing import Any

from iris_harness.sdk.persistence import data_path
from iris_personal.email.provider_api import (
    AttachmentCandidate,
    DownloadedAttachment,
    EmailAttachment,
    EmailMessage,
    FetchResult,
    MailSyncStore,
    ProgressFn,
    default_sync_store,
)

# The approval check and the write's audit row (R4 + R14), with the approval store
# injectable (tests hand the demo their own).
from iris_personal.email.write_approvals import WriteApprovalStore, mailbox_write

from .corpus import CORPUS_PATH, OWNER_ADDRESS, load_corpus

DEMO_PROVIDER = "demo"
DEMO_ACCOUNT = f"{DEMO_PROVIDER}:{OWNER_ADDRESS}"
CURSOR_KIND = "corpus_offset"
STATE_FILENAME = "demo_mailbox.json"
_SNIPPET_CHARS = 200

_DATE_TOKEN = re.compile(r"\{(date|nice)([+-]\d+)\}")
_WHEN_TOKEN = re.compile(r"\{when([+-]\d+)@(\d{2}):(\d{2})\}")


def render_body(body: str, today: date) -> str:
    """Fill the corpus's date tokens against ``today`` (the anchor's local date)."""

    def _date(match: re.Match[str]) -> str:
        day = today + timedelta(days=int(match.group(2)))
        return day.isoformat() if match.group(1) == "date" else day.strftime("%a %b %d")

    def _when(match: re.Match[str]) -> str:
        day = today + timedelta(days=int(match.group(1)))
        return f"{day.isoformat()}T{match.group(2)}:{match.group(3)}"

    return _WHEN_TOKEN.sub(_when, _DATE_TOKEN.sub(_date, body))


def _snippet(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:_SNIPPET_CHARS]


def _domain(sender: str) -> str | None:
    match = re.search(r"@([^>\s]+)", sender)
    return match.group(1).lower() if match else None


class DemoMailProvider:
    """The ``demo`` provider: 200 synthetic emails, read-only, labels kept locally."""

    name = DEMO_PROVIDER

    def __init__(
        self,
        *,
        corpus_path: Path = CORPUS_PATH,
        state_path: Path | None = None,
        tz: tzinfo | None = None,
        clock: Any = None,
        write_approvals: WriteApprovalStore | None = None,
    ) -> None:
        self._rows = load_corpus(corpus_path)
        self._approvals = write_approvals
        self._by_id = {str(r["id"]): r for r in self._rows}
        self._state_path = state_path
        self._tz = tz
        self._clock = clock
        self._anchor: datetime | None = None
        self._lock = Lock()

    # -- state -------------------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self._state_path or data_path(STATE_FILENAME)

    def _load_state(self) -> dict[str, Any]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_state(self, state: Mapping[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)

    def anchor(self) -> datetime:
        """When this mailbox was first opened (kept, so every run reads the same)."""
        with self._lock:
            if self._anchor is not None:
                return self._anchor
            state = self._load_state()
            raw = state.get("anchor")
            anchor: datetime | None = None
            if isinstance(raw, str):
                try:
                    anchor = datetime.fromisoformat(raw)
                except ValueError:
                    anchor = None
            if anchor is None:
                anchor = self._clock() if self._clock is not None else datetime.now(UTC)
                state["anchor"] = anchor.isoformat()
                self._save_state(state)
            self._anchor = anchor
            return anchor

    def _zone(self) -> tzinfo:
        if self._tz is not None:
            return self._tz
        from iris_harness.sdk.time import iris_timezone

        return iris_timezone()

    def _today(self) -> date:
        return self.anchor().astimezone(self._zone()).date()

    # -- messages ----------------------------------------------------------------

    def _labels(self, message_id: str, row: Mapping[str, Any]) -> tuple[str, ...]:
        extra = self._load_state().get("labels", {}).get(message_id, [])
        return tuple(dict.fromkeys([*row.get("labels", []), *extra]))

    def _message(self, row: Mapping[str, Any], account_id: str) -> EmailMessage:
        body = render_body(str(row["body"]), self._today())
        return EmailMessage(
            id=str(row["id"]),
            provider=DEMO_PROVIDER,
            account_id=account_id,
            thread_id=str(row["thread_id"]),
            from_address=str(row["from"]),
            from_domain=_domain(str(row["from"])),
            to=tuple(row.get("to") or ()),
            cc=tuple(row.get("cc") or ()),
            subject=str(row["subject"]),
            received_at=self.anchor() - timedelta(minutes=int(row["minutes_ago"])),
            snippet=_snippet(body),
            body_text=body,
            labels=self._labels(str(row["id"]), row),
            attachments=tuple(EmailAttachment(**a) for a in row.get("attachments") or ()),
            vendor_category=row.get("vendor_category"),
        )

    def _row(self, message_id: str) -> Mapping[str, Any]:
        row = self._by_id.get(message_id)
        if row is None:
            raise LookupError(f"no demo message {message_id!r}")
        return row

    # -- MailProvider ------------------------------------------------------------

    def fetch_new(
        self,
        account_id: str,
        *,
        store: MailSyncStore | None = None,
        max_messages: int = 100,
        cold_start_days: int = 30,
        progress: ProgressFn | None = None,
    ) -> FetchResult:
        """Deliver the next ``max_messages`` of the corpus, oldest first. Idempotent: the
        cursor (in the store, like any provider's) remembers how far it got."""
        del cold_start_days  # the corpus is the whole mailbox
        s = store if store is not None else default_sync_store()
        s.ensure_schema()
        raw = s.get_cursor(DEMO_PROVIDER, account_id, CURSOR_KIND)
        start = int(raw) if raw and raw.isdigit() else 0
        batch = self._rows[start : start + max(0, max_messages)]
        messages = [self._message(r, account_id) for r in batch]
        persisted = s.upsert_many(messages)
        end = start + len(batch)
        s.set_cursor(DEMO_PROVIDER, account_id, CURSOR_KIND, str(end))
        if progress is not None:
            # In-memory, no network -- nothing to report mid-way through.
            progress(1.0, f"fetched {len(batch)}")
        return FetchResult(
            account_id=account_id,
            fetched=persisted,
            new_message_ids=tuple(m.id for m in messages),
            new_cursor=str(end),
            fell_back_to_cold_start=start == 0 and bool(batch),
        )

    def reset_cursor(self, account_id: str, *, store: MailSyncStore) -> None:
        store.set_cursor(DEMO_PROVIDER, account_id, CURSOR_KIND, "0")

    def fetch_message_body(self, account_id: str, message_id: str, *, max_chars: int = 4000) -> str:
        del account_id
        return render_body(str(self._row(message_id)["body"]), self._today())[:max_chars]

    def trash_messages(self, account_id: str, message_ids: Sequence[str]) -> list[str]:
        raise PermissionError("the demo mailbox is read-only; nothing was moved to Trash")

    def restore_messages(
        self,
        account_id: str,
        message_ids: Sequence[str],
        *,
        labels_before: Mapping[str, Sequence[str]] | None = None,
    ) -> list[EmailMessage]:
        raise PermissionError("the demo mailbox is read-only; nothing is in its Trash")

    def fetch_message_attachments(
        self,
        account_id: str,
        message_id: str,
        *,
        mime_types: tuple[str, ...] | None = None,
        service: Any | None = None,
    ) -> list[DownloadedAttachment]:
        """The message's attachments; the bytes are a short synthetic text naming the
        file (the demo ships no real documents)."""
        del service
        message = self._message(self._row(message_id), account_id)
        out: list[DownloadedAttachment] = []
        for attachment in message.attachments:
            if mime_types and attachment.mime_type not in mime_types:
                continue
            content = (
                f"Demo attachment {attachment.filename} from {message.from_address}: "
                f"{message.subject}\n"
            ).encode()
            out.append(DownloadedAttachment(meta=attachment, content=content))
        return out

    def list_attachment_candidates(
        self, account_id: str, terms: Sequence[str], *, limit: int = 15
    ) -> list[AttachmentCandidate]:
        words = [t.lower() for t in terms if t.strip()]
        out: list[AttachmentCandidate] = []
        for row in reversed(self._rows):  # newest first, like a mailbox search
            if not row.get("attachments"):
                continue
            message = self._message(row, account_id)
            haystack = " ".join(
                [message.subject, message.from_address, *(a.filename for a in message.attachments)]
            ).lower()
            if words and not all(w in haystack for w in words):
                continue
            out.append(
                AttachmentCandidate(
                    subject=message.subject,
                    from_address=message.from_address,
                    attachments=message.attachments,
                )
            )
            if len(out) >= limit:
                break
        return out

    # -- LabellingProvider (kept in the demo's own state file) -------------------

    def ensure_labels(self, account_id: str, names: Sequence[str]) -> dict[str, str]:
        del account_id
        with self._lock:
            state = self._load_state()
            ids: dict[str, str] = state.setdefault("label_ids", {})
            for name in names:
                ids.setdefault(name, f"Label_{len(ids) + 1}")
            self._save_state(state)
            return {name: ids[name] for name in names}

    def modify_labels(
        self,
        account_id: str,
        message_ids: Sequence[str],
        add_ids: Sequence[str],
        remove_ids: Sequence[str],
    ) -> int:
        with (
            mailbox_write(account_id, "change labels", op="label", store=self._approvals) as tally,
            self._lock,
        ):
            state = self._load_state()
            names = {v: k for k, v in state.get("label_ids", {}).items()}
            labels: dict[str, list[str]] = state.setdefault("labels", {})
            remove = {names.get(i, i) for i in remove_ids}
            for message_id in message_ids:
                self._row(message_id)
                current = [x for x in labels.get(message_id, []) if x not in remove]
                for label_id in add_ids:
                    if names.get(label_id, label_id) not in current:
                        current.append(names.get(label_id, label_id))
                labels[message_id] = current
            self._save_state(state)
            tally.add(len(message_ids))
        return len(message_ids)


__all__ = [
    "CURSOR_KIND",
    "DEMO_ACCOUNT",
    "DEMO_PROVIDER",
    "DemoMailProvider",
    "render_body",
]
