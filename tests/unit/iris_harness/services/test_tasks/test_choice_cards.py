"""Choice cards on pending actions (ADR-0121): one card, several answers, one option."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.services.tasks import (
    ActionCard,
    ActionChoice,
    ActionFact,
    ActionOptions,
    ActionOptionValue,
    Task,
    TaskAction,
    TaskStore,
)
from iris_harness.services.tasks.pending_actions import DesiredAction, invoke_and_reconcile

TYPES = ActionOptions(
    name="type",
    label="Type",
    values=(
        ActionOptionValue(value="bank", label="Bank"),
        ActionOptionValue(value="card", label="Card"),
    ),
    default="bank",
)
CHOICES = (
    ActionChoice(value="confirm", label="Yes, it's mine", primary=True, needs_option=True),
    ActionChoice(value="ignore", label="Ignore"),
)


def _card(**over: object) -> TaskAction:
    base: dict[str, object] = {
        "kind": "execute",
        "label": "Answer",
        "target_id": "x",
        "safe": True,
        "choices": CHOICES,
        "options": TYPES,
        "card": ActionCard(
            tag="new sender", facts=(ActionFact(label="IRIS read", value="Woodgrove"),)
        ),
    }
    base.update(over)
    return TaskAction(**base)  # type: ignore[arg-type]


def test_a_choice_card_must_be_safe_unique_and_have_its_option() -> None:
    with pytest.raises(ValueError, match="must be safe"):
        _card(safe=False)
    with pytest.raises(ValueError, match="unique"):
        _card(choices=(CHOICES[0], CHOICES[0]))
    with pytest.raises(ValueError, match="requires options"):
        _card(options=None)
    with pytest.raises(ValueError, match="not one of"):
        ActionOptions(name="t", label="T", values=TYPES.values, default="loan")


def test_an_answer_is_checked_against_the_card() -> None:
    card = _card()
    card.check_answer("confirm", "card")
    card.check_answer("confirm", None)  # the default (bank) stands in
    card.check_answer("ignore", None)
    with pytest.raises(ValueError, match="answer with one of: confirm, ignore"):
        card.check_answer(None, None)
    with pytest.raises(ValueError, match="pick a type"):
        card.check_answer("confirm", "loan")
    with pytest.raises(ValueError, match="pick a type"):
        _card(options=TYPES.model_copy(update={"default": None})).check_answer("confirm", None)
    with pytest.raises(ValueError, match="takes no choice"):
        TaskAction(kind="re_extract", label="Retry", safe=True).check_answer("confirm", None)


def test_a_card_survives_the_task_store(tmp_path: Path) -> None:
    store = TaskStore(db_path=tmp_path / "tasks.db")
    store.ensure_schema()
    store.upsert(
        dedup_key="k1",
        title="Is Woodgrove yours?",
        description="",
        source_kind="other",
        action=_card(),
    )
    task = store.get_by_dedup_key("k1")
    assert task is not None and task.action == _card()


class _Provider:
    source_kind = "other"

    def __init__(self) -> None:
        self.answers: list[tuple[str, str | None]] = []

    def desired_actions(self) -> list[DesiredAction]:
        return []

    def invoke(self, task: Task) -> str:  # pragma: no cover - a card never lands here
        raise AssertionError("a choice card is answered by invoke_choice")

    def invoke_choice(self, task: Task, choice: str, option: str | None) -> str:
        self.answers.append((choice, option))
        return f"{choice}:{option}"


class _OneButtonProvider:
    source_kind = "other"

    def desired_actions(self) -> list[DesiredAction]:
        return []

    def invoke(self, task: Task) -> str:
        return "ran"


def _task(tmp_path: Path, action: TaskAction) -> tuple[Task, TaskStore]:
    store = TaskStore(db_path=tmp_path / "tasks.db")
    store.ensure_schema()
    store.upsert(dedup_key="k", title="t", description="", source_kind="other", action=action)
    task = store.get_by_dedup_key("k")
    assert task is not None
    return task, store


def test_the_answer_reaches_the_provider_with_its_option(tmp_path: Path) -> None:
    task, store = _task(tmp_path, _card())
    provider = _Provider()
    assert invoke_and_reconcile(provider, task, store, choice="confirm") == "confirm:bank"
    assert (
        invoke_and_reconcile(provider, task, store, choice="confirm", option="card")
        == "confirm:card"
    )
    assert (
        invoke_and_reconcile(provider, task, store, choice="ignore", option="card") == "ignore:None"
    )


def test_a_provider_without_invoke_choice_cannot_answer_a_card(tmp_path: Path) -> None:
    task, store = _task(tmp_path, _card())
    with pytest.raises(ValueError, match="cannot answer a choice card"):
        invoke_and_reconcile(_OneButtonProvider(), task, store, choice="ignore")


def test_a_one_button_action_is_unchanged(tmp_path: Path) -> None:
    task, store = _task(tmp_path, TaskAction(kind="re_extract", label="Retry", safe=True))
    assert invoke_and_reconcile(_OneButtonProvider(), task, store) == "ran"
