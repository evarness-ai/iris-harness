"""Pins the sizing invariant in `config/llm_tiers.yaml`.

`num_ctx` is the whole window — prompt plus generation — and `budget_for` lets
the prompt take 75% of it, so only a quarter is guaranteed to the answer. A tier
promising more `max_tokens` than that quarter gets cut off mid-sentence once the
prompt actually fills its budget.

That is not hypothetical: `code_exec` shipped at `num_ctx: 4096` with
`max_tokens: 2048`, guaranteeing 1024, and a planner ran out of room mid-JSON —
leaving a `run_shell` tool call that never closed and a turn that aborted. The
config is where that was decided, so the config is where it is checked.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from iris_harness.llm.budget import budget_for

_CONFIG = Path(__file__).resolve().parents[4] / "config" / "llm_tiers.yaml"
_PROMPT_FRACTION = 0.75  # must match the call in bootstrap.py


def _tiers() -> dict[str, dict]:
    return dict(yaml.safe_load(_CONFIG.read_text())["tiers"])


def _local_tiers_with_a_window() -> list[tuple[str, dict]]:
    """Only tiers that pin a `num_ctx` — cloud tiers manage their own window."""
    return [(name, tier) for name, tier in _tiers().items() if tier.get("num_ctx")]


def test_there_are_local_tiers_to_check() -> None:
    """Guard the guard: a rename must not turn this file into a no-op."""
    assert len(_local_tiers_with_a_window()) >= 4


@pytest.mark.parametrize("name,tier", _local_tiers_with_a_window(), ids=lambda v: v)
def test_generation_room_covers_max_tokens(name: str, tier: dict) -> None:
    num_ctx = int(tier["num_ctx"])
    max_tokens = int(tier.get("max_tokens", 2048))
    generation_room = num_ctx - budget_for(num_ctx, fraction=_PROMPT_FRACTION)

    assert generation_room >= max_tokens, (
        f"tier {name!r} ({tier.get('model')}) promises max_tokens={max_tokens} but "
        f"num_ctx={num_ctx} only guarantees {generation_room} after the prompt takes "
        f"its {_PROMPT_FRACTION:.0%}. Raise num_ctx to >= {4 * max_tokens}, or lower "
        "max_tokens — otherwise the model is cut off mid-answer."
    )


def test_code_exec_runs_a_code_model() -> None:
    """A general 3B degenerated into a repeated-import loop writing a heredoc.

    Not a pin on one model id — a pin on the property that mattered: whatever
    `code_exec` routes to should be built for writing code.
    """
    model = str(_tiers()["code_exec"]["model"])

    assert "coder" in model or "code" in model, (
        f"code_exec routes to {model!r}, which is not a code model. "
        "That tier writes shell and Python inside JSON tool calls."
    )


def test_email_judge_is_a_pinned_local_model() -> None:
    """The judge's accuracy is only comparable run to run on one model, and it reads
    raw email bodies: always the Mac's Ollama, never downshifted or moved."""
    tier = _tiers()["email_judge"]
    assert tier["provider"] == "ollama"
    assert tier["pinned"] is True
    assert tier["use_for"] == ["email_judge"]
    assert int(tier["max_tokens"]) <= 2048
