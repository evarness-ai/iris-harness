"""The judge's Gmail labels: write them, and read the owner's relabels back.

**Write** (:func:`sync_labels`, after each judge run and right after a correction from
the card, chat or web): every judged email carries EXACTLY ONE IRIS/* label, the one for
its effective bucket (``judge.yaml`` names the labels). The rows due
(``JudgmentStore.labels_due``) are grouped by bucket and written with one
``modify_labels`` call per (account, bucket): add that bucket's label, remove every other
IRIS/* label. An email the owner called ``promo`` loses all of them. Nothing else is ever
touched: IRIS never archives (INBOX) and never marks read (UNREAD). The row then records
the label now on it (``mark_labelled``). The ``IRIS_EMAIL_JUDGE_LABELS`` setting turns this
off; the caller checks it (``labels_enabled``). Both must allow a write: the setting says
whether IRIS labels at all, and the account's mailbox-write approval (R4,
``iris_personal.email.write_approvals``) says whether the owner let IRIS change that
mailbox. An account without the approval is skipped before its provider is asked
(its rows stay due, so the first run after ``iris email writes approve`` labels them).

**Read back** (:func:`read_back`, on ``email.labels_changed`` from the sweep): the owner
moving an IRIS/* label in Gmail is a correction (``source="gmail"``). IRIS's own write
comes back the same way, so a label equal to the row's ``label_bucket`` is not the owner.
Two IRIS labels on one email (the owner added one and left IRIS's) means the one that is
not ``label_bucket``; none at all (the owner removed it) records nothing.

A provider without the labelling capability (``LabellingProvider``) is skipped.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from iris_personal.email.write_approvals import require_mailbox_writes

from .judge_config import JudgeConfig, labels_enabled
from .judge_corrections import Emit, apply_correction
from .judgments import PROMO, Correction, JudgmentStore

logger = logging.getLogger(__name__)

# Labels IRIS must never remove (it does not archive or mark read). The removes are
# built from judge.yaml's labels only; this is the belt to that brace.
_NEVER_REMOVE = frozenset({"INBOX", "UNREAD"})


@dataclass
class AccountLabels:
    """One account's label write: emails given a label, stripped of all IRIS labels
    (``promo``), and not written (``error`` says why)."""

    written: int = 0
    removed: int = 0
    failed: int = 0
    error: str = ""


@dataclass
class LabelSync:
    """What :func:`sync_labels` did, per account. Accounts whose provider cannot label
    are in ``skipped``."""

    accounts: dict[str, AccountLabels] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    @property
    def written(self) -> int:
        return sum(a.written for a in self.accounts.values())

    @property
    def removed(self) -> int:
        return sum(a.removed for a in self.accounts.values())

    @property
    def failed(self) -> int:
        return sum(a.failed for a in self.accounts.values())

    @property
    def errors(self) -> list[str]:
        return [f"{acct}: {a.error}" for acct, a in self.accounts.items() if a.error]

    def summary(self) -> str:
        text = f"labels: {self.written} written, {self.removed} removed, {self.failed} failed"
        return text + (f" ({'; '.join(self.errors)})" if self.errors else "")


def _labelling(provider: Any) -> bool:
    return callable(getattr(provider, "ensure_labels", None)) and callable(
        getattr(provider, "modify_labels", None)
    )


def sync_labels(
    store: JudgmentStore, config: JudgeConfig, providers: Mapping[str, Any]
) -> LabelSync:
    """Write the IRIS/* label of every row whose label is due, per account
    (``providers``: account id → mail provider). Never raises: an account's error is
    reported in the result and the other accounts still run."""
    result = LabelSync()
    names = config.labels  # bucket key → label name
    for account_id, provider in providers.items():
        if not _labelling(provider):
            result.skipped.append(account_id)
            continue
        due = store.labels_due(account_id)
        if not due:
            continue
        counts = result.accounts.setdefault(account_id, AccountLabels())
        try:
            # The owner's per-account approval (R4), asked before the provider is: an
            # account without one gets no request at all, and one clear log line.
            require_mailbox_writes(account_id, "label the mail it judged")
        except PermissionError as exc:
            logger.warning("judge-labels: %d left unlabelled: %s", len(due), exc)
            counts.failed += len(due)
            counts.error = str(exc)
            continue
        try:
            ids_by_name = provider.ensure_labels(account_id, list(names.values()))
            label_id = {key: ids_by_name[name] for key, name in names.items()}
        except Exception as exc:  # noqa: BLE001 — one account's failure is reported
            logger.warning("judge-labels: %s: could not resolve labels: %s", account_id, exc)
            counts.failed += len(due)
            counts.error = str(exc)
            continue
        every = set(label_id.values())

        groups: dict[str | None, list[str]] = {}
        for row in due:
            bucket = row.effective_bucket
            if bucket == PROMO:
                groups.setdefault(None, []).append(row.message_id)
            elif bucket in label_id:
                groups.setdefault(bucket, []).append(row.message_id)
            else:
                logger.warning(
                    "judge-labels: %s has bucket %r, not in judge.yaml; left unlabelled",
                    row.message_id,
                    bucket,
                )

        ordered = list(groups.items())
        for index, (bucket, ids) in enumerate(ordered):
            add = [label_id[bucket]] if bucket is not None else []
            remove = sorted((every - set(add)) - _NEVER_REMOVE)
            try:
                provider.modify_labels(account_id, ids, add, remove)
            except PermissionError as exc:
                # The grant: every group left would be refused the same way.
                counts.failed += sum(len(rest) for _, rest in ordered[index:])
                counts.error = str(exc)
                break
            except Exception as exc:  # noqa: BLE001 — reported, next group still runs
                logger.warning("judge-labels: %s: %s failed: %s", account_id, bucket, exc)
                counts.failed += len(ids)
                counts.error = str(exc)
                continue
            store.mark_labelled(ids, bucket)
            if bucket is None:
                counts.removed += len(ids)
            else:
                counts.written += len(ids)
    return result


def read_back(
    store: JudgmentStore,
    config: JudgeConfig,
    changes: Iterable[tuple[str, Sequence[str]]],
    emit: Emit | None,
    *,
    label_names: Mapping[str, str] | None = None,
) -> list[Correction]:
    """Turn the owner's Gmail relabels into corrections (``source="gmail"``).

    ``changes`` is ``(message id, its labels now)``, as ``email.labels_changed`` carries
    it; ``label_names`` maps a provider label id to its name (the inverse of
    ``ensure_labels``). A label not in the map is taken as a name already. Returns the
    corrections that changed a bucket.
    """
    by_name = {name: key for key, name in config.labels.items()}
    id_to_name = dict(label_names or {})
    out: list[Correction] = []
    for message_id, labels in changes:
        row = store.get(message_id)
        if row is None or row.bucket is None:
            continue  # never judged (or still waiting): not IRIS's label to read
        present = [
            by_name[name]
            for name in (id_to_name.get(label, label) for label in labels)
            if name in by_name
        ]
        if not present:
            continue  # the owner took the label off and put none on: nothing to learn
        owners = [key for key in present if key != row.label_bucket]
        if not owners:
            continue  # the label IRIS wrote, echoed back by the sync
        chosen = next(key for key in config.keys if key in owners)
        correction = apply_correction(store, config, message_id, chosen, source="gmail", emit=emit)
        if len(present) == 1:
            # Gmail already shows exactly this label: nothing left to write.
            store.mark_labelled([message_id], chosen)
        if correction is not None and correction.changed:
            logger.info(
                "judge-labels: %s moved %s -> %s in Gmail",
                message_id,
                correction.previous,
                chosen,
            )
            out.append(correction)
    return out


# ─── Bus handlers (wired in plugin.py) ─────────────────────────────────


def _default_emit(topic: str, payload: Any) -> None:
    from iris_harness.sdk.events import get_default_bus

    get_default_bus().emit_sync(topic, payload)


@dataclass
class LabelHandlers:
    """The two subscriptions, sharing how they find the config, store and provider
    (each resolved per event, so a settings change or a test's patch takes effect)."""

    # Read per event (the host's services may be filled in after setup).
    config_dir_for: Callable[[], Path | None] = field(default=lambda: None)
    store_factory: Callable[[], JudgmentStore] = JudgmentStore
    provider_for: Callable[[str], Any] | None = None
    emit: Emit | None = None

    def _provider(self, account_id: str) -> Any:
        if self.provider_for is not None:
            return self.provider_for(account_id)
        from iris_personal.email.providers import mail_provider_for

        return mail_provider_for(account_id)

    def _store(self) -> JudgmentStore:
        store = self.store_factory()
        store.ensure_schema()
        return store

    def on_labels_changed(self, payload: Any) -> list[Correction]:
        """``email.labels_changed`` → the owner's Gmail relabels as corrections."""
        if not labels_enabled():
            return []  # IRIS writes no labels, so none are read back either
        provider = self._provider(payload.account_id)
        if not _labelling(provider):
            return []
        config = JudgeConfig.load(self.config_dir_for())
        ids_by_name = provider.ensure_labels(payload.account_id, list(config.labels.values()))
        return read_back(
            self._store(),
            config,
            payload.changes,
            self.emit or _default_emit,
            label_names={label_id: name for name, label_id in ids_by_name.items()},
        )

    def on_corrected(self, payload: Any) -> LabelSync | None:
        """``email.judgment_corrected`` from the card, chat or web → move that email's
        Gmail label now, so the surface can truthfully say it did. A ``gmail``
        correction already shows in Gmail; the next judge run tidies any leftover."""
        if payload.source == "gmail" or not labels_enabled():
            return None
        provider = self._provider(payload.account_id)
        if provider is None:
            return None
        result = sync_labels(
            self._store(), JudgeConfig.load(self.config_dir_for()), {payload.account_id: provider}
        )
        if result.errors:
            logger.warning("judge-labels: after a correction: %s", result.summary())
        return result
