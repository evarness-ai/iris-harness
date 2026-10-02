"""Mail providers — the interface behind every read that reaches a mailbox app.

The core owns the local record (``EmailStore``) and every read over it. It does not
own a mailbox: fetching new mail, a message body, or an attachment means talking to
Gmail (or, later, anything else), and the owner's rule is that anything talking to an
external app is a plugin (OSS plan M5.7 track A). So the core declares the interface
here and keeps a registry; a provider plugin registers an implementation at
``setup()``, keyed by the ``provider`` column of ``email_accounts`` (``"gmail"``).

Callers in the core and in other plugins — the sweep heartbeat, the ``read_email``
and ``find_attachment`` tools, finance's statement ingest, the category-discovery
bootstrap — look the provider up by account and call the interface. With no provider
registered the lookup returns ``None`` and each caller says so in its own words; a
missing plugin is an honest "not connected", never a crash and never a silent no-op.

This is the same shape as ``tasks.pending_actions.register_provider`` (M2.6) and
``health.service.register_check_provider``: a keyed core registry a plugin plugs into,
not a seventh registration kind.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from iris_harness.sdk.process_state import track_globals

if TYPE_CHECKING:
    from iris_personal.email.contracts import EmailAttachment, EmailMessage

logger = logging.getLogger(__name__)

# ``(fraction 0..1, short message) -> None``: live progress during a long fetch, the
# same shape ``iris_harness.services.activities.ActivityRunner`` hands a submitted
# job. Optional everywhere it appears: a provider that doesn't call it just shows an
# indeterminate "still working" state to whatever is watching, never a stuck one.
ProgressFn = Callable[[float, str], None]


@runtime_checkable
class MailSyncStore(Protocol):
    """The part of the local mail record a provider writes while it syncs.

    A provider is handed one (``fetch_new``, ``reset_cursor``); it never opens the
    record itself or reaches past these methods. ``EmailStore`` is the implementation;
    this is all of it a provider may rely on: its sync cursor, the messages it fetched,
    and the labels of messages already stored (a label read-back).
    """

    def ensure_schema(self) -> None:
        """Make the store ready. Idempotent; a store a caller hands over already is."""
        ...

    def get_cursor(self, provider: str, account_id: str, cursor_kind: str) -> str | None:
        """The provider's saved sync position for an account, or ``None`` (cold start)."""
        ...

    def set_cursor(
        self, provider: str, account_id: str, cursor_kind: str, cursor_value: str
    ) -> None:
        """Save the sync position reached."""
        ...

    def clear_cursor(self, provider: str, account_id: str, cursor_kind: str) -> None:
        """Forget the sync position, so the next sync cold-starts."""
        ...

    def upsert_many(self, messages: Iterable[EmailMessage]) -> int:
        """Insert or update fetched messages; return how many were written."""
        ...

    def get(self, message_id: str, *, include_held: bool = False) -> EmailMessage | None:
        """A stored message by id (``include_held``: also one held back from reads)."""
        ...

    def set_labels(self, message_id: str, labels: Sequence[str]) -> bool:
        """Replace a stored message's labels; ``False`` when it is not stored."""
        ...


@dataclass(frozen=True)
class FetchResult:
    """Summary of one ``fetch_new`` call.

    ``new_message_ids`` carries the provider-native ids of the messages that were just
    upserted — subscribers to ``email.new_arrived`` receive the same set in their
    payload, so a sweep can hand off work to triage without round-tripping through
    email.db.
    """

    account_id: str
    fetched: int
    new_message_ids: tuple[str, ...]
    new_cursor: str | None
    fell_back_to_cold_start: bool
    # ``(message id, its labels now)`` for messages already in the store whose labels
    # the owner (or IRIS) changed since the last sync; the sweep emits them as
    # ``email.labels_changed``. Empty for a provider without a label read-back.
    label_changes: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class DownloadedAttachment:
    """An attachment plus its decoded bytes."""

    meta: EmailAttachment
    content: bytes


@dataclass(frozen=True)
class AttachmentCandidate:
    """One message that carries attachments, as a live provider search returns it.

    The provider does the mailbox query; the caller does the matching, so the ranking
    stays in one place whatever mailbox the message came from.
    """

    subject: str
    from_address: str
    attachments: tuple[EmailAttachment, ...]


@runtime_checkable
class MailProvider(Protocol):
    """What a mailbox plugin supplies. Every method reads the mailbox except
    ``trash_messages`` and ``restore_messages``, which move mail to the provider's Trash
    and back: reversible by design, never a permanent delete (ADR-0118)."""

    @property
    def name(self) -> str:
        """The ``email_accounts.provider`` value this serves, e.g. ``"gmail"``."""
        ...

    def fetch_new(
        self,
        account_id: str,
        *,
        store: MailSyncStore | None = None,
        max_messages: int = 100,
        cold_start_days: int = 30,
        progress: ProgressFn | None = None,
    ) -> FetchResult:
        """Sync new mail for one account into the store. Idempotent on re-runs.

        ``progress``, when given, is called with how far through the fetch this
        call is. Optional: a provider that doesn't call it just runs exactly as
        it always has."""
        ...

    def reset_cursor(self, account_id: str, *, store: MailSyncStore) -> None:
        """Forget the incremental sync cursor so the next ``fetch_new`` cold-starts."""
        ...

    def fetch_message_body(self, account_id: str, message_id: str, *, max_chars: int = 4000) -> str:
        """The readable text body of one message, truncated to ``max_chars``."""
        ...

    def trash_messages(self, account_id: str, message_ids: Sequence[str]) -> list[str]:
        """Move messages to Trash; return the ids moved. Raises ``PermissionError`` when
        the account's grant cannot modify mail, with how to fix it in the message."""
        ...

    def restore_messages(
        self,
        account_id: str,
        message_ids: Sequence[str],
        *,
        labels_before: Mapping[str, Sequence[str]] | None = None,
    ) -> list[EmailMessage]:
        """Take messages out of Trash; return them, ready to put back in the store.

        ``labels_before`` is each message's labels when it was trashed (from the
        provider's optional ``current_labels(account_id, ids)``, called by the trash
        tool just before trashing); a provider whose trash removes labels puts them
        back from it."""
        ...

    def fetch_message_attachments(
        self,
        account_id: str,
        message_id: str,
        *,
        mime_types: tuple[str, ...] | None = None,
        service: Any | None = None,
    ) -> list[DownloadedAttachment]:
        """List and download a message's attachments (optionally MIME-filtered).

        ``service`` is an opaque, provider-specific client handle a caller may inject
        (tests do); a provider that has no such thing ignores it.
        """
        ...

    def list_attachment_candidates(
        self, account_id: str, terms: Sequence[str], *, limit: int = 15
    ) -> list[AttachmentCandidate]:
        """Messages with attachments that a live mailbox search for ``terms`` returns.

        Raises on an authentication failure so the caller can say "reconnect".
        """
        ...


@runtime_checkable
class LabellingProvider(Protocol):
    """An optional capability: a provider that can put named labels on mail (the email
    judge's IRIS/* labels, loop-proof PR 5). Callers check for it rather than require
    it of every mailbox provider (``getattr(provider, "modify_labels", None)``): a
    provider without it simply does not label."""

    def ensure_labels(self, account_id: str, names: Sequence[str]) -> dict[str, str]:
        """``{name: label id}``, creating the labels that do not exist yet."""
        ...

    def modify_labels(
        self,
        account_id: str,
        message_ids: Sequence[str],
        add_ids: Sequence[str],
        remove_ids: Sequence[str],
    ) -> int:
        """Add and remove labels (by id) on messages; return how many were sent. Raises
        ``PermissionError`` when the grant cannot modify mail, with how to fix it."""
        ...


_lock = Lock()
_providers: dict[str, MailProvider] = {}


def register_mail_provider(provider: MailProvider) -> None:
    """Install (or replace) the provider for ``provider.name``.

    Keyed, not appended, so a process that builds a second runtime (the playground
    does, and so does every runtime test) replaces the provider instead of stacking a
    stale one under a live one.
    """
    with _lock:
        _providers[provider.name] = provider


def mail_provider_for(account: str) -> MailProvider | None:
    """The provider for an account id (``"gmail:user@x"``) or a bare provider name."""
    key = account.split(":", 1)[0] if ":" in account else account
    with _lock:
        return _providers.get(key)


def registered_mail_providers() -> tuple[str, ...]:
    with _lock:
        return tuple(sorted(_providers))


def clear_mail_providers() -> None:
    """Drop every provider (tests)."""
    with _lock:
        _providers.clear()


# -- the CLI seam's providers ------------------------------------------------------
#
# ``iris`` commands run with no runtime built, so no plugin's ``setup()`` has mounted a
# provider. A provider plugin's ``cli:`` function (which runs at every ``iris`` start)
# *offers* its provider here as a factory; a command that talks to a mailbox (``iris
# email setup``) calls :func:`mount_cli_mail_providers` to build and register them.
# Offering costs nothing -- the factory imports the provider only when called -- so
# ``iris --help`` pays nothing, and a command that never asks mounts nothing.

_cli_factories: dict[str, Callable[[], MailProvider]] = {}


def offer_cli_mail_provider(name: str, factory: Callable[[], MailProvider]) -> None:
    """Offer ``factory`` (building the provider for ``name``) to CLI commands. Keyed:
    offering ``name`` again replaces it."""
    with _lock:
        _cli_factories[name] = factory


def mount_cli_mail_providers() -> tuple[str, ...]:
    """Build and register every offered provider not registered yet; returns the names
    now mounted. A factory that fails (its extra is not installed) is skipped: that
    provider stays unmounted and its callers say "not connected", as with no plugin."""
    with _lock:
        pending = {n: f for n, f in _cli_factories.items() if n not in _providers}
    mounted: list[str] = []
    for name, factory in sorted(pending.items()):
        try:
            register_mail_provider(factory())
        except Exception:  # one provider missing must not stop the command
            logger.warning("mail provider %r could not be built for the CLI", name, exc_info=True)
            continue
        mounted.append(name)
    return tuple(mounted)


def clear_cli_mail_providers() -> None:
    """Forget every offered factory (tests)."""
    with _lock:
        _cli_factories.clear()


__all__ = [
    "AttachmentCandidate",
    "DownloadedAttachment",
    "FetchResult",
    "LabellingProvider",
    "MailProvider",
    "MailSyncStore",
    "clear_cli_mail_providers",
    "clear_mail_providers",
    "mail_provider_for",
    "mount_cli_mail_providers",
    "offer_cli_mail_provider",
    "register_mail_provider",
    "registered_mail_providers",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_providers", "_cli_factories")
