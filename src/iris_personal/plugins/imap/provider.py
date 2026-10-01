"""The IMAP ``MailProvider`` + ``LabellingProvider`` (OSS plan R3: IMAP + app password).

**Sync.** One folder per account (``INBOX`` unless the account says otherwise), opened
read-only (``EXAMINE``) and read with ``BODY.PEEK`` so syncing never marks mail read.
The cursor is ``(folder, UIDVALIDITY, highest UID seen)`` in the email store's
``sync_cursors``, like every provider's. A run asks for UIDs above the mark, oldest
first, so a capped batch never leaves a gap. A first sync (no cursor) takes the newest
``max_messages`` of the last ``cold_start_days``. When the server's UIDVALIDITY changes,
every UID it handed out is void: the provider forgets the mark and cold-starts
(``fell_back_to_cold_start``); ids are Message-ID based (``mime.py``), so the re-sync
upserts the same rows rather than doubling them.

**Labels.** IRIS labels are ``$``-keywords (``keywords.py`` says why not folders). The
owner moving one in their mail client comes back on the next sync as
``FetchResult.label_changes``: the provider asks the server which messages carry each
IRIS keyword and compares with what the last sync saw.

**Writes are gated (R4).** Keywords, moves to Trash and restores are refused with
``PermissionError`` -- before any connection -- until the account has a recorded write
approval in the email library's one gate for every provider
(``iris_personal.email.write_approvals.approve_mailbox_writes``, which the onboarding
flow's label-preview approval calls with its approval id). The callers are the same governed
paths Gmail's writes go through: the judge's ``sync_labels`` and the approval-gated
trash/restore tools. Trash is a move to the server's Trash folder, never an expunge of
the owner's mail.

**Egress.** Every method that talks to the server opens one session and writes one
``EGRESS`` line with the host (``connection.open_session``); none logs credentials.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from iris_harness.sdk.vault import SecretStore
from iris_personal.email.provider_api import (
    AttachmentCandidate,
    DownloadedAttachment,
    EmailMessage,
    FetchResult,
    MailSyncStore,
    default_sync_store,
)

# The approval check and the audit row of each write (R4 + R14), with the approval store
# injectable (tests hand the provider their own).
from iris_personal.email.write_approvals import WriteApprovalStore, WriteTally, mailbox_write

from .account import IMAP_PROVIDER, ImapAccount, address_of, load_account
from .connection import (
    DEFAULT_TIMEOUT_S,
    FetchedItem,
    FolderInfo,
    ImapAuthError,
    ImapConnectionError,
    ImapError,
    ImapSession,
    SslContextFactory,
    open_session,
    quote,
)
from .keywords import is_writable_keyword, keyword_for
from .mime import iter_attachments, parse_bytes, parse_mail, readable_text, to_email_message
from .state import ImapState, Location

logger = logging.getLogger(__name__)

CURSOR_KIND = "imap_uid"
#: A message larger than this is synced from its headers alone (no snippet, no
#: attachment list) rather than downloaded whole on every sync.
DEFAULT_MAX_FULL_FETCH_BYTES = 10 * 1024 * 1024
_FETCH_CHUNK = 50
_TRASH_NAMES = (
    "Trash",
    "Deleted Items",
    "Deleted Messages",
    "INBOX.Trash",
    "INBOX/Trash",
    "[Gmail]/Trash",
    "[Google Mail]/Trash",
)


@dataclass(frozen=True)
class Cursor:
    folder: str
    uidvalidity: int
    uid: int

    def dump(self) -> str:
        return json.dumps({"folder": self.folder, "uidvalidity": self.uidvalidity, "uid": self.uid})

    @classmethod
    def load(cls, raw: str | None) -> Cursor | None:
        if not raw:
            return None
        try:
            data = json.loads(raw)
            return cls(str(data["folder"]), int(data["uidvalidity"]), int(data["uid"]))
        except (ValueError, KeyError, TypeError):
            return None


class ImapProvider:
    """An IMAP mailbox behind the core's ``MailProvider`` interface, with labels."""

    name = IMAP_PROVIDER

    def __init__(
        self,
        *,
        secret_store: SecretStore | None = None,
        state: ImapState | None = None,
        ssl_context_factory: SslContextFactory | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_full_fetch_bytes: int = DEFAULT_MAX_FULL_FETCH_BYTES,
        clock: Any = None,
        write_approvals: WriteApprovalStore | None = None,
    ) -> None:
        self._secrets = secret_store
        self.state = state or ImapState()
        self._ssl = ssl_context_factory
        self._timeout = timeout
        self._max_full = max_full_fetch_bytes
        self._clock = clock
        self._approvals = write_approvals

    # -- plumbing ----------------------------------------------------------------

    def _now(self) -> datetime:
        return self._clock() if self._clock is not None else datetime.now(UTC)

    def account(self, account_id: str) -> ImapAccount:
        found = load_account(account_id, store=self._secrets)
        if found is None:
            address = address_of(account_id)
            raise ImapError(
                f"No IMAP credentials for {account_id}. "
                f"Run `iris auth imap login --user {address}` first."
            )
        return found

    @contextmanager
    def _session(self, account: ImapAccount, purpose: str) -> Iterator[ImapSession]:
        """A logged-in session; the login outcome is recorded for System Health."""
        try:
            with open_session(
                account, purpose=purpose, ssl_context_factory=self._ssl, timeout=self._timeout
            ) as session:
                self.state.login_ok(account.account_id)
                yield session
        except ImapAuthError as exc:
            self.state.login_failed(account.account_id, str(exc), auth=True)
            raise
        except ImapConnectionError as exc:
            self.state.login_failed(account.account_id, str(exc), auth=False)
            raise

    def check_login(self, account: ImapAccount) -> FolderInfo:
        """Connect, log in and open the account's folder read-only (``iris auth imap
        login`` and the health net probe). Raises :class:`ImapError` subclasses."""
        with self._session(account, "imap.check_login") as session:
            return session.select(account.folder, readonly=True)

    # -- write approval (R4): the email library's gate, one for every provider ------

    def _write(self, account_id: str, what: str, op: str) -> AbstractContextManager[WriteTally]:
        """The gate, then the write's audit row (what reached the mailbox, by ``op``)."""
        return mailbox_write(account_id, what, op=op, store=self._approvals)

    # -- locating messages -------------------------------------------------------

    def _stored_message_id(self, message_id: str) -> str | None:
        """The ``Message-ID`` header the email store kept for a message the plugin's own
        map does not know (state lost, or a row synced before it existed)."""
        stored = default_sync_store().get(message_id, include_held=True)
        return stored.headers_subset.get("Message-ID") if stored is not None else None

    def _by_folder(
        self, account: ImapAccount, ids: Sequence[str]
    ) -> dict[str, list[tuple[str, Location | None]]]:
        locations = self.state.locations(account.account_id, ids)
        grouped: dict[str, list[tuple[str, Location | None]]] = {}
        for mid in dict.fromkeys(ids):
            loc = locations.get(mid)
            grouped.setdefault(loc.folder if loc else account.folder, []).append((mid, loc))
        return grouped

    def _find_by_header(self, session: ImapSession, rfc_message_id: str | None) -> int | None:
        if not rfc_message_id:
            return None
        found = session.uid_search("HEADER", "Message-ID", quote(rfc_message_id))
        return found[-1] if found else None

    def _each_folder(
        self,
        session: ImapSession,
        account: ImapAccount,
        ids: Sequence[str],
        *,
        readonly: bool,
    ) -> Iterator[tuple[FolderInfo, list[tuple[str, int, Location | None]]]]:
        """Open each folder the messages are in and yield ``(folder, [(id, uid, loc)])``
        while it is selected. A UID from an older UIDVALIDITY is re-found by its
        ``Message-ID``; a message that cannot be found is left out (and logged)."""
        for folder, group in self._by_folder(account, ids).items():
            info = session.select(folder, readonly=readonly)
            found: list[tuple[str, int, Location | None]] = []
            for mid, loc in group:
                uid: int | None = None
                if loc is not None and loc.uidvalidity == info.uidvalidity:
                    uid = loc.uid
                else:
                    rfc = loc.rfc_message_id if loc else self._stored_message_id(mid)
                    uid = self._find_by_header(session, rfc)
                    if uid is not None:
                        self.state.record(
                            account.account_id,
                            [
                                (
                                    mid,
                                    folder,
                                    info.uidvalidity,
                                    uid,
                                    rfc,
                                    loc.keywords if loc else (),
                                )
                            ],
                        )
                if uid is None:
                    logger.info("imap %s: %s not found in %s", account.account_id, mid, folder)
                    continue
                found.append((mid, uid, loc))
            if found:
                yield info, found

    # -- MailProvider: sync ------------------------------------------------------

    def fetch_new(
        self,
        account_id: str,
        *,
        store: MailSyncStore | None = None,
        max_messages: int = 100,
        cold_start_days: int = 30,
    ) -> FetchResult:
        s = store if store is not None else default_sync_store()
        s.ensure_schema()
        account = self.account(account_id)
        cursor = Cursor.load(s.get_cursor(IMAP_PROVIDER, account_id, CURSOR_KIND))
        label_changes: tuple[tuple[str, tuple[str, ...]], ...] = ()

        with self._session(account, "imap.fetch_new") as session:
            info = session.select(account.folder, readonly=True)
            incremental = (
                cursor is not None
                and cursor.folder == account.folder
                and cursor.uidvalidity == info.uidvalidity
            )
            fell_back = cursor is not None and not incremental
            if fell_back:
                logger.info(
                    "imap %s: UIDVALIDITY of %s changed; cold-starting", account_id, account.folder
                )
            if incremental and cursor is not None:
                above = [
                    u for u in session.uid_search("UID", f"{cursor.uid + 1}:*") if u > cursor.uid
                ]
                batch = above[: max(0, max_messages)]
                high = batch[-1] if batch else cursor.uid
            else:
                since = (self._now() - timedelta(days=cold_start_days)).strftime("%d-%b-%Y")
                window = session.uid_search("SINCE", since)
                batch = window[-max_messages:] if max_messages > 0 else []
                top = session.uid_search("UID", "*")
                high = max([*top, *batch, 0])
            messages = self._fetch_messages(session, account, info, batch)
            if incremental:
                label_changes = self._read_back(session, account, info, s, exclude=set(batch))

        persisted = s.upsert_many(messages)
        new_cursor = Cursor(account.folder, info.uidvalidity, high)
        s.set_cursor(IMAP_PROVIDER, account_id, CURSOR_KIND, new_cursor.dump())
        return FetchResult(
            account_id=account_id,
            fetched=persisted,
            new_message_ids=tuple(m.id for m in messages),
            new_cursor=new_cursor.dump(),
            fell_back_to_cold_start=fell_back,
            label_changes=label_changes,
        )

    def _download(
        self, session: ImapSession, uids: Sequence[int]
    ) -> tuple[list[FetchedItem], dict[int, bytes]]:
        """Flags/date/size for each UID, then the bytes: whole messages up to the size
        cap, headers only above it. ``BODY.PEEK`` throughout: nothing is marked read."""
        meta: list[FetchedItem] = []
        raw: dict[int, bytes] = {}
        for start in range(0, len(uids), _FETCH_CHUNK):
            chunk = list(uids[start : start + _FETCH_CHUNK])
            items = session.uid_fetch(chunk, "(UID FLAGS INTERNALDATE RFC822.SIZE)")
            meta.extend(items)
            small = [i.uid for i in items if (i.size or 0) <= self._max_full]
            large = [i.uid for i in items if (i.size or 0) > self._max_full]
            for got in session.uid_fetch(small, "(UID BODY.PEEK[])"):
                if got.body is not None:
                    raw[got.uid] = got.body
            for got in session.uid_fetch(large, "(UID BODY.PEEK[HEADER])"):
                if got.body is not None:
                    raw[got.uid] = got.body
        return meta, raw

    def _fetch_messages(
        self,
        session: ImapSession,
        account: ImapAccount,
        info: FolderInfo,
        uids: Sequence[int],
    ) -> list[EmailMessage]:
        if not uids:
            return []
        known = self.state.label_keywords(account.account_id)
        meta, raw = self._download(session, uids)
        out: list[EmailMessage] = []
        rows: list[tuple[str, str, int, int, str | None, tuple[str, ...]]] = []
        for item in sorted(meta, key=lambda i: i.uid):
            body = raw.get(item.uid)
            if body is None:
                logger.debug("imap %s: uid %s returned no bytes", account.account_id, item.uid)
                continue
            try:
                mail = parse_mail(body)
                message = to_email_message(
                    mail,
                    account_id=account.account_id,
                    folder=info.name,
                    flags=item.flags,
                    internaldate=item.internaldate,
                )
            except (ValueError, TypeError, LookupError) as exc:
                logger.debug("imap %s: skipping uid %s (%s)", account.account_id, item.uid, exc)
                continue
            out.append(message)
            iris_keywords = tuple(f for f in item.flags if f in known)
            rows.append(
                (message.id, info.name, info.uidvalidity, item.uid, mail.message_id, iris_keywords)
            )
        self.state.record(account.account_id, rows)
        return out

    def _read_back(
        self,
        session: ImapSession,
        account: ImapAccount,
        info: FolderInfo,
        store: MailSyncStore,
        *,
        exclude: set[int],
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """The owner's relabels since the last sync: which messages carry each IRIS
        keyword now, against what the last sync saw. New mail (``exclude``) was just
        fetched with its flags, so it is not a change."""
        known = self.state.label_keywords(account.account_id)
        if not known:
            return ()
        now: dict[int, set[str]] = {}
        for keyword in sorted(known):
            for uid in session.uid_search("KEYWORD", keyword):
                now.setdefault(uid, set()).add(keyword)
        mapped = self.state.in_folder(account.account_id, info.name, info.uidvalidity)
        changed: dict[str, tuple[str, ...]] = {}
        for uid, keywords in now.items():
            loc = mapped.get(uid)
            if loc is None or uid in exclude:
                continue
            if set(loc.keywords) != keywords:
                changed[loc.id] = tuple(sorted(keywords))
        for loc in self.state.with_keywords(account.account_id, info.name).values():
            if loc.uidvalidity == info.uidvalidity and loc.uid not in now:
                if loc.uid not in exclude:
                    changed[loc.id] = ()
        out: list[tuple[str, tuple[str, ...]]] = []
        for mid, now_keywords in changed.items():
            stored = store.get(mid, include_held=True)
            if stored is None:
                continue
            labels = tuple(x for x in stored.labels if x not in known) + now_keywords
            store.set_labels(mid, labels)
            out.append((mid, labels))
        self.state.set_keywords(account.account_id, changed)
        return tuple(out)

    def reset_cursor(self, account_id: str, *, store: MailSyncStore) -> None:
        """Forget the UID mark so the next ``fetch_new`` cold-starts."""
        store.clear_cursor(IMAP_PROVIDER, account_id, CURSOR_KIND)

    # -- MailProvider: reads -----------------------------------------------------

    def _raw_message(self, session: ImapSession, account: ImapAccount, message_id: str) -> bytes:
        for _info, found in self._each_folder(session, account, [message_id], readonly=True):
            _mid, uid, _loc = found[0]
            items = session.uid_fetch([uid], "(UID BODY.PEEK[])")
            if items and items[0].body is not None:
                return items[0].body
        raise LookupError(f"message {message_id} is not in {account.account_id} any more")

    def fetch_message_body(self, account_id: str, message_id: str, *, max_chars: int = 4000) -> str:
        account = self.account(account_id)
        with self._session(account, "imap.fetch_body") as session:
            raw = self._raw_message(session, account, message_id)
        return readable_text(parse_mail(raw))[:max_chars].strip()

    def fetch_message_attachments(
        self,
        account_id: str,
        message_id: str,
        *,
        mime_types: tuple[str, ...] | None = None,
        service: Any | None = None,
    ) -> list[DownloadedAttachment]:
        del service  # no client handle to inject: the session is opened here
        account = self.account(account_id)
        with self._session(account, "imap.fetch_attachments") as session:
            raw = self._raw_message(session, account, message_id)
        mail = parse_mail(raw)
        by_id = {a.attachment_id: a for a in mail.attachments}
        out: list[DownloadedAttachment] = []
        for attachment_id, part in iter_attachments(parse_bytes(raw)):
            meta = by_id.get(attachment_id)
            if meta is None or (mime_types and meta.mime_type not in mime_types):
                continue
            payload = part.get_payload(decode=True)
            out.append(
                DownloadedAttachment(
                    meta=meta, content=payload if isinstance(payload, bytes) else b""
                )
            )
        return out

    def list_attachment_candidates(
        self, account_id: str, terms: Sequence[str], *, limit: int = 15
    ) -> list[AttachmentCandidate]:
        """Newest messages matching every term (server-side ``TEXT`` for ASCII terms;
        every term is also checked here) that carry attachments. Raises
        :class:`ImapAuthError` on a refused login, so the caller can say reconnect."""
        account = self.account(account_id)
        words = [t.strip() for t in terms if t.strip()]
        criteria: list[str] = []
        for word in words:
            if word.isascii():
                criteria += ["TEXT", quote(word)]
        out: list[AttachmentCandidate] = []
        with self._session(account, "imap.attachment_search") as session:
            session.select(account.folder, readonly=True)
            uids = session.uid_search(*(criteria or ["ALL"]))
            newest = list(reversed(uids))[: max(limit * 4, limit)]
            _meta, raw = self._download(session, newest)
        for uid in newest:
            body = raw.get(uid)
            if body is None:
                continue
            mail = parse_mail(body)
            if not mail.attachments:
                continue
            haystack = " ".join(
                [mail.subject, mail.from_raw, readable_text(mail)]
                + [a.filename for a in mail.attachments]
            ).lower()
            if not all(w.lower() in haystack for w in words):
                continue
            out.append(
                AttachmentCandidate(
                    subject=mail.subject,
                    from_address=mail.from_raw or "unknown",
                    attachments=mail.attachments,
                )
            )
            if len(out) >= limit:
                break
        return out

    # -- MailProvider: Trash (reversible moves, never an expunge of owner mail) -----

    def _trash_folder(self, session: ImapSession) -> str:
        folders = session.list_folders()
        for attrs, name in folders:
            if any(a.lower() == "\\trash" for a in attrs):
                return name
        names = {name.lower(): name for _attrs, name in folders}
        for candidate in _TRASH_NAMES:
            if candidate.lower() in names:
                return names[candidate.lower()]
        raise ImapError(
            f"{session.account.host} has no Trash folder IRIS can find; IRIS never deletes "
            "mail outright, so nothing was moved"
        )

    def _settle(
        self,
        session: ImapSession,
        account: ImapAccount,
        folder: str,
        moved: list[tuple[str, int, Location | None, str | None]],
        mapping: dict[int, int],
    ) -> FolderInfo:
        """Record where moved messages now are: the UIDPLUS mapping when the server
        gave one, else a ``Message-ID`` search in the destination."""
        info = session.select(folder, readonly=True)
        for mid, old_uid, loc, home in moved:
            new_uid = mapping.get(old_uid)
            if new_uid is None:
                rfc = loc.rfc_message_id if loc else self._stored_message_id(mid)
                new_uid = self._find_by_header(session, rfc)
            if new_uid is None:
                continue
            if loc is None:
                rfc = self._stored_message_id(mid)
                self.state.record(
                    account.account_id, [(mid, folder, info.uidvalidity, new_uid, rfc, ())]
                )
            self.state.moved(
                account.account_id,
                mid,
                folder=folder,
                uidvalidity=info.uidvalidity,
                uid=new_uid,
                home_folder=home,
            )
        return info

    def trash_messages(self, account_id: str, message_ids: Sequence[str]) -> list[str]:
        with self._write(account_id, "move mail to Trash", "trash") as tally:
            return self._trash(account_id, message_ids, tally)

    def _trash(self, account_id: str, message_ids: Sequence[str], tally: WriteTally) -> list[str]:
        account = self.account(account_id)
        done: list[str] = []
        with self._session(account, "imap.trash") as session:
            trash = self._trash_folder(session)
            pending: list[tuple[str, int, Location | None, str | None]] = []
            mapping: dict[int, int] = {}
            for info, found in self._each_folder(session, account, message_ids, readonly=False):
                if info.name == trash:
                    continue
                mapping.update(session.uid_move([uid for _m, uid, _l in found], trash))
                tally.add(len(found))
                pending.extend((mid, uid, loc, info.name) for mid, uid, loc in found)
                done.extend(mid for mid, _u, _l in found)
            if pending:
                self._settle(session, account, trash, pending, mapping)
        return done

    def restore_messages(
        self,
        account_id: str,
        message_ids: Sequence[str],
        *,
        labels_before: Mapping[str, Sequence[str]] | None = None,
    ) -> list[EmailMessage]:
        """Move messages back from Trash to the folder they left. ``labels_before`` is
        not needed: an IMAP move keeps a message's flags and keywords."""
        del labels_before
        with self._write(account_id, "restore mail from Trash", "restore") as tally:
            return self._restore(account_id, message_ids, tally)

    def _restore(
        self, account_id: str, message_ids: Sequence[str], tally: WriteTally
    ) -> list[EmailMessage]:
        account = self.account(account_id)
        restored: list[EmailMessage] = []
        with self._session(account, "imap.restore") as session:
            by_home: dict[str, list[tuple[str, int, Location | None, str | None]]] = {}
            mapping: dict[int, int] = {}
            for _info, found in self._each_folder(session, account, message_ids, readonly=False):
                groups: dict[str, list[tuple[str, int, Location | None]]] = {}
                for mid, uid, loc in found:
                    home = (loc.home_folder if loc else None) or account.folder
                    groups.setdefault(home, []).append((mid, uid, loc))
                for home, items in groups.items():
                    mapping.update(session.uid_move([uid for _m, uid, _l in items], home))
                    tally.add(len(items))
                    by_home.setdefault(home, []).extend((m, u, loc, None) for m, u, loc in items)
            for home, moved in by_home.items():
                info = self._settle(session, account, home, moved, mapping)
                locs = self.state.locations(account.account_id, [m for m, *_ in moved])
                uids = [locs[m].uid for m, *_ in moved if m in locs]
                restored.extend(self._fetch_messages(session, account, info, uids))
        return restored

    # -- LabellingProvider -------------------------------------------------------

    def ensure_labels(self, account_id: str, names: Sequence[str]) -> dict[str, str]:
        """``{name: keyword}``. A keyword needs no creating on the server, so this is
        local: it only remembers which keywords are IRIS's, for the read-back."""
        mapping = {name: keyword_for(name) for name in names}
        self.state.remember_keywords(account_id, mapping)
        return mapping

    def modify_labels(
        self,
        account_id: str,
        message_ids: Sequence[str],
        add_ids: Sequence[str],
        remove_ids: Sequence[str],
    ) -> int:
        """Add and remove keywords. Exactly what is passed, minus anything that is not a
        keyword: IRIS never touches ``\\Seen``, ``\\Deleted`` or the folder."""
        with self._write(account_id, "change labels", "label") as tally:
            return self._modify_labels(account_id, message_ids, add_ids, remove_ids, tally)

    def _modify_labels(
        self,
        account_id: str,
        message_ids: Sequence[str],
        add_ids: Sequence[str],
        remove_ids: Sequence[str],
        tally: WriteTally,
    ) -> int:
        add = [k for k in dict.fromkeys(add_ids) if is_writable_keyword(k)]
        remove = [k for k in dict.fromkeys(remove_ids) if is_writable_keyword(k) and k not in add]
        if not message_ids or not (add or remove):
            return 0
        account = self.account(account_id)
        sent = 0
        with self._session(account, "imap.labels") as session:
            for info, found in self._each_folder(session, account, message_ids, readonly=False):
                refused = [k for k in add if not info.accepts_keyword(k)]
                if refused:
                    raise PermissionError(
                        f"{account.host} does not keep custom keywords in {info.name}, so "
                        f"IRIS cannot label mail there ({', '.join(refused)}); nothing was "
                        "changed in that folder"
                    )
                uids = [uid for _m, uid, _l in found]
                session.uid_store(uids, "+FLAGS.SILENT", add)
                session.uid_store(uids, "-FLAGS.SILENT", remove)
                sent += len(found)
                tally.add(len(found))
        return sent


__all__ = ["CURSOR_KIND", "DEFAULT_MAX_FULL_FETCH_BYTES", "Cursor", "ImapProvider"]
