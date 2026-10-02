"""What a mail provider is written against: the stable tier's email surface (OSS R16).

A third-party mailbox plugin (another IMAP flavour, a JMAP server, an Exchange bridge)
imports from this module and nothing else in the email slice. It holds exactly what
such a plugin needs to implement the interface and register itself:

* the interface -- :class:`MailProvider`, and the optional :class:`LabellingProvider`;
* the types its methods take and return -- :class:`EmailMessage` and
  :class:`EmailAttachment` (the message record), :class:`FetchResult`,
  :class:`DownloadedAttachment`, :class:`AttachmentCandidate`;
* :class:`MailSyncStore` -- the part of the local record a sync writes (its cursor, the
  messages it fetched, a stored message's labels). A provider is handed one; the record
  itself (``EmailStore``) is not part of the surface. :func:`default_sync_store` is the
  one a provider uses when a caller passes none;
* :func:`register_mail_provider` -- called from the plugin's ``setup()``;
* :func:`connect_account` -- records the owner's account at the provider, so the sweep
  syncs it;
* :func:`mailbox_write` -- how a method that changes the mailbox (labels, Trash,
  restore) makes its write: the owner's write approval (R4) is checked first, and what
  reached the mailbox is recorded in the governance ledger after (the proof bundle's
  mailbox-write observation, R14). It yields a :class:`WriteTally`; the write calls its
  ``add`` as each batch lands;
* :func:`require_mailbox_writes` -- the approval check alone, for a method that must
  refuse before it can know what it will write. It records nothing; a write still goes
  through :func:`mailbox_write`.

Everything else in ``iris_personal.email`` is the slice's own and may change in any
release. The names here change only after a deprecation cycle
(docs/reference/stable-api.md).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from iris_personal.email import write_approvals
from iris_personal.email.contracts import EmailAttachment, EmailMessage
from iris_personal.email.providers import (
    AttachmentCandidate,
    DownloadedAttachment,
    FetchResult,
    LabellingProvider,
    MailProvider,
    MailSyncStore,
    ProgressFn,
    register_mail_provider,
)


def default_sync_store() -> MailSyncStore:
    """The owner's mail record, ready to sync into: for a ``fetch_new`` called with no
    ``store`` (callers in IRIS always pass one)."""
    from iris_personal.email.store import EmailStore

    store = EmailStore()
    store.ensure_schema()
    return store


def connect_account(provider: str, address: str) -> str:
    """Record the owner's mailbox ``address`` at ``provider`` as a connected account and
    return its account id -- the id every :class:`MailProvider` method is called with.

    Call it once the provider holds what it needs to reach the mailbox (its credentials
    in the vault, say): from then on the email sweep syncs the account. Connecting an
    account that is already connected returns its id; one that was disconnected is
    connected again. ``provider`` is the provider's ``name``, compared case-insensitively.
    The record lives in the IRIS data directory (``$IRIS_DATA_DIR`` / ``$IRIS_HOME``).
    Raises ``ValueError`` for an address without an ``@``.
    """
    from iris_personal.email.accounts import EmailAccountStore

    store = EmailAccountStore()
    store.ensure_schema()
    existing = store.get_by_address(provider, address)
    if existing is None:
        return store.add(provider=provider, address=address).id
    if not existing.active:
        store.activate(existing.id)
    return existing.id


def require_mailbox_writes(account_id: str, what: str) -> None:
    """Raise ``PermissionError`` unless the owner approved mailbox writes for
    ``account_id``. ``what`` is the write in the owner's words ("change labels", "move
    mail to Trash"); the error tells the owner the one command that approves it."""
    write_approvals.require_mailbox_writes(account_id, what)


#: What one :func:`mailbox_write` changed: call ``add(n)`` with how many messages or
#: labels each batch changed, as it lands; ``count`` is the total so far.
WriteTally = write_approvals.WriteTally


@contextmanager
def mailbox_write(account_id: str, what: str, *, op: str) -> Iterator[WriteTally]:
    """Make one write to ``account_id``'s mailbox, the governed way.

    On entry, the owner's approval is checked (:func:`require_mailbox_writes`: with
    none, ``PermissionError`` and the body never runs). The body makes the write and
    passes what reached the mailbox to the yielded :class:`WriteTally`'s ``add`` as each
    batch lands. On exit -- also when the body raises part way, so a partial write is seen
    too -- one governance ledger row records the total (hook
    ``mailbox_write_performed``; nothing when the total is 0). ``what`` is the write in
    the owner's words ("move mail to Trash"), for the refusal; ``op`` names its kind
    (``trash``, ``restore``, ``label``, ``create_label``, ``restore_labels``)::

        with mailbox_write(account_id, "move mail to Trash", op="trash") as tally:
            moved = service.trash(message_ids)
            tally.add(len(moved))
    """
    with write_approvals.mailbox_write(account_id, what, op=op) as tally:
        yield tally


__all__ = [
    "AttachmentCandidate",
    "DownloadedAttachment",
    "EmailAttachment",
    "EmailMessage",
    "FetchResult",
    "LabellingProvider",
    "MailProvider",
    "MailSyncStore",
    "ProgressFn",
    "WriteTally",
    "connect_account",
    "default_sync_store",
    "mailbox_write",
    "register_mail_provider",
    "require_mailbox_writes",
]
