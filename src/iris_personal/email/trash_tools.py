"""``trash_email`` and ``restore_email``: the first destructive tool (ADR-0118 step 5).

``trash_email`` moves messages to the provider's Trash, which Gmail keeps for 30 days.
It is declared ``effect: destructive`` with ``undo: restore_email``, so every call
waits for the owner's approval. The card describes it in their words ("Trash 3 emails",
one line per subject and sender), looked up here from the local store by id, so they
approve mail they can recognise rather than ids.

A trashed message leaves the local store (it is no longer in the mailbox as the owner
sees it), and a small ledger remembers its account, subject and sender so that
``restore_email`` — "undo that" — can bring it back without asking which account.
Nothing here deletes permanently; the Gmail grant IRIS holds cannot.

``trash_category`` is the owner's other way in (promo-classification grill, 2026-09-22):
"delete my promo emails this week" names a category and a window, and CODE picks the
emails — every one filed under that category (or carrying the provider label that maps
to it) in the window, newest first, at most ``_MAX_PER_CATEGORY``. Nothing is guessed,
so the owner chose to skip the card for it: it is a reversible ``write`` that runs at
once and says how to undo it (ADR-0118 amendment). ``trash_email`` keeps the card,
because there the model chose the emails.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import timedelta
from pathlib import Path
from typing import Any

from iris_harness.sdk.time import local_now
from iris_harness.sdk.types import ToolDescription, ToolSpec
from iris_personal.email.categories import (
    LabelsFor,
    category_menu,
    provider_labels,
    resolve_category,
)
from iris_personal.email.store import CategoryFilter, EmailStore

logger = logging.getLogger(__name__)

_MAX_PER_CALL = 50
# One category request moves at most this many, newest first; the reply says how many
# more match. The owner gets ~60 promotions a day, so a week is 300-400 (2026-09-22).
_MAX_PER_CATEGORY = 200
# How many of the moved emails the reply names; the count covers the rest.
_REPLY_SAMPLE = 10


def _ids(args: dict[str, Any]) -> list[str]:
    raw = args.get("ids") or args.get("message_ids") or []
    if isinstance(raw, str):
        raw = [raw]
    seen: list[str] = []
    for value in raw if isinstance(raw, list) else []:
        mid = str(value).strip()
        if mid and mid not in seen:
            seen.append(mid)
    return seen


def _days(args: dict[str, Any]) -> int | None:
    try:
        days = int(str(args.get("since_days")))
    except (TypeError, ValueError):
        return None
    return days if days > 0 else None


def _covers_all(in_use: Iterable[str], paths: Iterable[str]) -> bool:
    """True when ``paths`` is every category the account uses (and there is more than one)."""
    used = set(in_use)
    return len(used) > 1 and used <= set(paths)


def _plural(n: int) -> str:
    return f"{n} email" if n == 1 else f"{n} emails"


def build_trash_tools(
    *,
    data_dir: Path,
    provider_for: Callable[[str], Any] | None = None,
    labels_for: LabelsFor = provider_labels,
    window_days: Callable[[dict[str, Any]], int | None] | None = None,
) -> list[ToolSpec]:
    """The trash tools over the store at ``data_dir``. ``provider_for`` maps an account
    to its mail provider (the registered one by default; tests pass a fake), and
    ``labels_for`` to its label → path table. ``window_days`` reads the call's time
    window in days (the ``since_days`` argument, else the owner's own words); without
    it only ``since_days`` counts."""

    def _store() -> EmailStore:
        store = EmailStore(db_path=data_dir / "email.db")
        store.ensure_schema()
        return store

    def _provider(account_id: str) -> Any:
        if provider_for is not None:
            return provider_for(account_id)
        from iris_personal.email.providers import mail_provider_for

        return mail_provider_for(account_id)

    def describe_trash(args: dict[str, Any]) -> ToolDescription:
        store = _store()
        lines: list[str] = []
        for mid in _ids(args):
            msg = store.get(mid)
            if msg is None:
                lines.append(f"(not in your mail any more: {mid})")
                continue
            subject = (msg.subject or "(no subject)").strip()
            lines.append(f"{subject} — {msg.from_address} · {msg.received_at:%d %b}")
        return ToolDescription(title=f"Trash {_plural(len(lines))}", lines=tuple(lines))

    def validate_trash(args: dict[str, Any]) -> str | None:
        """Refuse a call the owner could only approve to do nothing: no ids, too many,
        or ids that are not in their mail (the model invents plausible ones when a
        search found nothing, or passes a query instead)."""
        ids = _ids(args)
        if not ids:
            return (
                "trash_email was given no email ids, so nothing has been looked up yet: "
                "this is NOT a search result and says nothing about what is in the mail. "
                "First find the emails with search_inbox or list_by_category, then call "
                'trash_email with {"ids": [...]} — the [id …] values those results print.'
            )
        if len(ids) > _MAX_PER_CALL:
            return f"trash_email takes at most {_MAX_PER_CALL} emails at a time."
        store = _store()
        unknown = [mid for mid in ids if store.get(mid) is None]
        if unknown:
            return (
                f"{_plural(len(unknown))} named here {'is' if len(unknown) == 1 else 'are'} "
                f"not in the owner's mail: {', '.join(unknown[:10])}"
                f"{' …' if len(unknown) > 10 else ''}. Ids cannot be guessed: use only the "
                "[id …] values that search_inbox or list_by_category printed in this "
                "conversation, and look the emails up first if you have none."
            )
        return None

    def _move(store: EmailStore, msgs: list[Any]) -> tuple[list[Any], str | None]:
        """Trash ``msgs`` account by account and record each batch for restore_email.

        Returns what moved and, when some of it could not move, the owner-facing
        reason. Every account's provider is checked first, so a missing one moves
        nothing at all. An account whose grant cannot modify mail is skipped with its
        fix, and the other accounts still move: on the owner's first real run
        (2026-09-22) a read-only second account came first and stopped the whole
        batch, so none of the 287 promotions in the main account moved.
        """
        by_account: dict[str, list[Any]] = defaultdict(list)
        for msg in msgs:
            by_account[msg.account_id].append(msg)
        providers: dict[str, Any] = {}
        for account_id in by_account:
            provider = _provider(account_id)
            if provider is None or not hasattr(provider, "trash_messages"):
                return [], (
                    f"Error: no mail provider can trash mail for {account_id}. "
                    "Nothing was trashed."
                )
            providers[account_id] = provider
        batch_id = uuid.uuid4().hex
        trashed: list[Any] = []
        refused: list[str] = []
        for account_id, batch in by_account.items():
            provider = providers[account_id]
            ids = [m.id for m in batch]
            try:
                # Read first: a provider's trash may drop labels (Gmail: INBOX) that its
                # restore cannot know to put back (2026-09-22: 200 restored promotions
                # came back archived).
                current = getattr(provider, "current_labels", None)
                labels_before = current(account_id, ids) if current is not None else {}
                if current is not None:
                    # Never trash what could not be read: its restore would not know
                    # what to put back.
                    ids = [mid for mid in ids if mid in labels_before]
                done = set(provider.trash_messages(account_id, ids)) if ids else set()
            except PermissionError as exc:
                refused.append(f"Not trashed ({_plural(len(batch))}): {exc}")
                continue
            moved = [m for m in batch if m.id in done]
            store.move_to_trashed(moved, batch_id=batch_id, labels_before=labels_before)
            trashed.extend(moved)
        return trashed, "\n".join(refused) or None

    def trash(args: dict[str, Any]) -> str:
        ids = _ids(args)
        if not ids:
            return 'Error: give the emails to trash as {"ids": [...]}, from search results.'
        if len(ids) > _MAX_PER_CALL:
            return f"Error: at most {_MAX_PER_CALL} emails at a time."
        store = _store()
        found: list[Any] = []
        missing: list[str] = []
        for mid in ids:
            msg = store.get(mid)
            if msg is None:
                missing.append(mid)
            else:
                found.append(msg)
        trashed, problem = _move(store, found)
        if problem is not None and not trashed:
            return problem
        parts = [
            f"Moved {_plural(len(trashed))} to Trash (kept 30 days; restore_email brings them back)."
        ]
        parts.extend(f"- {m.subject or '(no subject)'} — {m.from_address}" for m in trashed)
        if problem is not None:
            parts.append(problem)
        if missing:
            parts.append(f"Not found, so not trashed: {', '.join(missing)}")
        skipped = len(ids) - len(trashed) - len(missing)
        if skipped and problem is None:
            parts.append(f"{_plural(skipped)} could not be trashed (already gone?).")
        return "\n".join(parts)

    def _no_card_scope(
        account_id: str, resolved: CategoryFilter
    ) -> tuple[CategoryFilter, tuple[str, ...]]:
        """What a category trash may touch, and the paths to name in the reply.

        With no card, the mailbox's own bucket decides whenever it has one (owner,
        2026-09-22): only mail carrying the provider's label for the category, wherever
        triage filed it. IRIS's own paths can be wrong — "promo" also matched
        ``email/shopping/deals-promotions/amazon``, where triage had put Amazon order
        and delivery notices and an Apple receipt that Gmail itself filed as Updates.
        A category the provider has no bucket for is matched by its paths.
        """
        if not resolved.labels:
            return resolved, resolved.paths
        bucket = labels_for(account_id)
        shown = tuple(sorted({bucket[label] for label in resolved.labels if label in bucket}))
        return CategoryFilter(labels=resolved.labels), shown

    def trash_category(args: dict[str, Any]) -> str:
        name = str(args.get("category") or args.get("input") or "").strip()
        if not name:
            return (
                'Error: trash_category needs {"category": str} — the kind of email the '
                'owner named, e.g. "promotions". Nothing was trashed.'
            )
        days = window_days(args) if window_days is not None else _days(args)
        since = local_now() - timedelta(days=days) if days else None
        when = f" from the last {days} days" if days else ""
        store = _store()
        resolved = [
            (a, resolve_category(store, name, a, labels_for=labels_for))
            for a in store.list_accounts()
        ]
        resolved = [(a, r) for a, r in resolved if r.found]
        if not resolved:
            return f"No category matches '{name}', so nothing was trashed. {category_menu(store)}"
        if any(_covers_all(store.category_counts(a), r.filter.paths) for a, r in resolved):
            # "email" or "everything" names all the mail, not a kind of it; with no
            # card in front of this tool, that must not become the newest 200 of all.
            return (
                f"'{name}' covers every category of email, not one kind, so nothing was "
                f"trashed. Name one kind. {category_menu(store)}"
            )
        targets = [(a, *_no_card_scope(a, r.filter)) for a, r in resolved]
        total = sum(store.count_by_category(a, f, since=since) for a, f, _ in targets)
        if total == 0:
            return f"No {name} emails{when}; nothing to trash."
        candidates: list[Any] = []
        for account_id, scope, _ in targets:
            candidates.extend(
                store.list_by_category(account_id, scope, limit=_MAX_PER_CATEGORY, since=since)
            )
        candidates.sort(key=lambda m: m.received_at, reverse=True)
        trashed, problem = _move(store, candidates[:_MAX_PER_CATEGORY])
        if not trashed:
            return problem or "Nothing was trashed: the mail provider moved none of them."
        paths = sorted({p for _, _, shown in targets for p in shown})
        parts = [
            f"Moved {_plural(len(trashed))} filed under '{name}' ({', '.join(paths)}){when} "
            "to Trash. Gmail keeps them 30 days; restore_email with no ids brings this whole "
            'batch back ("undo that").'
        ]
        parts.extend(
            f"- {m.subject or '(no subject)'} — {m.from_address} · {m.received_at:%d %b}"
            for m in trashed[:_REPLY_SAMPLE]
        )
        if len(trashed) > _REPLY_SAMPLE:
            parts.append(f"(+{len(trashed) - _REPLY_SAMPLE} more)")
        if problem is not None:
            parts.append(problem)
        left = total - len(trashed)
        if left > 0:
            parts.append(
                f"{left} more {name} emails{when} are still in the inbox; asking again moves "
                f"the next {min(left, _MAX_PER_CATEGORY)}."
            )
        return "\n".join(parts)

    def restore(args: dict[str, Any]) -> str:
        store = _store()
        ids = _ids(args)
        entries = store.trashed(ids or None)
        if not entries:
            return "Nothing to restore: IRIS has not trashed those emails."
        by_account: dict[str, list[str]] = defaultdict(list)
        for mid, account_id, _subject, _sender in entries:
            by_account[account_id].append(mid)
        restored: list[Any] = []
        for account_id, mids in by_account.items():
            provider = _provider(account_id)
            if provider is None or not hasattr(provider, "restore_messages"):
                return f"Error: no mail provider can restore mail for {account_id}."
            try:
                messages = provider.restore_messages(
                    account_id, mids, labels_before=store.trashed_labels(mids)
                )
            except PermissionError as exc:
                return f"Not restored: {exc}"
            store.restore_from_trashed(messages)
            restored.extend(messages)
        lines = [f"Restored {_plural(len(restored))} from Trash."]
        # A sample, not the batch: this reply is the answer (answers_directly), and a
        # 200-line list cost 79 s of a local model copying it out (2026-09-22).
        lines.extend(
            f"- {m.subject or '(no subject)'} — {m.from_address}" for m in restored[:_REPLY_SAMPLE]
        )
        if len(restored) > _REPLY_SAMPLE:
            lines.append(f"(+{len(restored) - _REPLY_SAMPLE} more)")
        return "\n".join(lines)

    return [
        ToolSpec(
            name="trash_email",
            description=(
                "Move emails to the Trash (Gmail keeps them 30 days; restore_email brings "
                "them back). For when the user asks to delete, trash or clean up specific "
                'emails. Args: {"ids": [str]} — the [id …] values from search_inbox or '
                "list_by_category results, all the emails in ONE call. Every call waits for "
                "the owner's approval, which shows them each email; do not ask them first. "
                "When the user names a KIND of email ('my promo emails', 'social updates "
                "from this week') rather than specific ones, use trash_category instead."
            ),
            call=trash,
            describe=describe_trash,
            validate=validate_trash,
        ),
        ToolSpec(
            name="trash_category",
            description=(
                "Move every email of a KIND the user named to the Trash — 'delete my promo "
                "emails for this week', 'clear out social notifications'. Code picks the "
                "emails (all filed under that category in the window, newest first, at most "
                f"{_MAX_PER_CATEGORY}); it runs at once, and restore_email undoes it. Do not "
                "search first and do not ask the user which ones or whether to go ahead. "
                'Args: {"category": str (the user\'s own word, e.g. "promo"), '
                '"since_days"?: int (7 for "this week")}.'
            ),
            call=trash_category,
        ),
        ToolSpec(
            name="restore_email",
            description=(
                "Bring back emails that trash_email moved to the Trash — for 'undo that', "
                "'restore them', 'bring those back'. Args: {\"ids\"?: [str]}; with no ids it "
                "restores the most recent batch."
            ),
            call=restore,
        ),
    ]


__all__ = ["build_trash_tools"]
