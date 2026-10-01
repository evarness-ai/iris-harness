"""Tests for the generic pending-actions mechanism (ADR-0073 slice 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.services.tasks import DesiredAction, TaskAction, TaskStore
from iris_harness.services.tasks.pending_actions import invoke_and_reconcile, reconcile


@pytest.fixture
def store(tmp_path: Path) -> TaskStore:
    s = TaskStore(db_path=tmp_path / "tasks.db")
    s.ensure_schema()
    return s


class _FakeProvider:
    """A test provider over an arbitrary 'source-kind' slice."""

    source_kind = "routine"  # any SourceKind that isn't used by another provider

    def __init__(self, desired: list[DesiredAction]) -> None:
        self._desired = desired

    def desired_actions(self) -> list[DesiredAction]:
        return self._desired

    def invoke(self, task: object) -> str:  # pragma: no cover - not exercised here
        return "ok"


def _da(key: str) -> DesiredAction:
    return DesiredAction(
        dedup_key=key,
        title=f"do {key}",
        description="",
        source_id=key,
        action=TaskAction(kind="review", label="Review", target_id=key, safe=True),
    )


def test_reconcile_raises_desired(store: TaskStore) -> None:
    summary = reconcile(_FakeProvider([_da("a"), _da("b")]), store)
    assert summary.raised == 2 and summary.open_total == 2
    assert {t.dedup_key for t in store.list(has_action=True)} == {"a", "b"}


def test_reconcile_is_idempotent(store: TaskStore) -> None:
    p = _FakeProvider([_da("a")])
    reconcile(p, store)
    second = reconcile(p, store)
    assert second.raised == 0
    assert len(store.list(has_action=True)) == 1


def test_reconcile_resolves_dropped_desired(store: TaskStore) -> None:
    # First pass raises a + b; second pass only wants a => b is completed.
    reconcile(_FakeProvider([_da("a"), _da("b")]), store)
    summary = reconcile(_FakeProvider([_da("a")]), store)
    assert summary.resolved == 1
    open_keys = {t.dedup_key for t in store.list(status="open", has_action=True)}
    assert open_keys == {"a"}


class _ResolvingProvider(_FakeProvider):
    """Invoking an action clears the blocker it was raised for, as a real domain does."""

    def invoke(self, task: object) -> str:
        key = getattr(task, "dedup_key", None)
        self._desired = [d for d in self._desired if d.dedup_key != key]
        return "done"


def test_invoke_and_reconcile_closes_the_invoked_action(store: TaskStore) -> None:
    provider = _ResolvingProvider([_da("a"), _da("b")])
    reconcile(provider, store)
    task = next(t for t in store.list(has_action=True) if t.dedup_key == "a")

    assert invoke_and_reconcile(provider, task, store) == "done"

    open_keys = {t.dedup_key for t in store.list(status="open", has_action=True)}
    assert open_keys == {"b"}


def test_invoke_and_reconcile_keeps_the_result_when_reconcile_fails(
    store: TaskStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _ResolvingProvider([_da("a")])
    reconcile(provider, store)
    task = store.list(has_action=True)[0]

    def _boom() -> list[DesiredAction]:
        raise RuntimeError("provider scan failed")

    monkeypatch.setattr(provider, "desired_actions", _boom)
    assert invoke_and_reconcile(provider, task, store) == "done"


def test_reconcile_only_touches_its_own_source_kind(store: TaskStore) -> None:
    # A pending action owned by a DIFFERENT provider must survive reconciliation.
    store.create(
        title="finance action",
        source_kind="finance-statements",
        action=TaskAction(kind="re_extract", label="Re-extract", target_id="s1", safe=True),
    )
    reconcile(_FakeProvider([]), store)  # routine provider with nothing desired
    survivors = store.list(source_kind="finance-statements", has_action=True)
    assert len(survivors) == 1 and survivors[0].status == "open"
