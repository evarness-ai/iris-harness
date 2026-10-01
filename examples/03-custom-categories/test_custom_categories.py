"""Your email categories, live in the email assistant: offline, on a scripted model.

The test builds a config directory -- IRIS's own, plus ``email/judge.yaml`` from this
example -- and runs the ``email`` profile on it. In chat, "the carpool email is money"
is a correction in your words ("money" is your name for the ``bill`` bucket): the
judge's correction handler claims it, deterministically, with no model call. With the
shipped categories the same sentence is not a correction and goes on to the assistant.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from iris_harness.sdk.config import config_dir
from iris_harness.testing import harness

HERE = Path(__file__).parent
CORRECTION = "the carpool email is money"
REBUCKET = "email_rebucket"  # the judge's chat-correction handler
SCRIPT = {"default": {"content": "Scripted answer."}}


@pytest.fixture
def your_config(tmp_path: Path) -> Path:
    """IRIS's config directory with your ``email/judge.yaml`` in it."""
    target = tmp_path / "config"
    shutil.copytree(config_dir(), target)
    (target / "email").mkdir(exist_ok=True)
    shutil.copy(HERE / "judge.yaml", target / "email" / "judge.yaml")
    return target


@pytest.fixture
def shipped_config(tmp_path: Path) -> Path:
    """IRIS's config directory as shipped."""
    target = tmp_path / "shipped"
    shutil.copytree(config_dir(), target)
    return target


def test_your_words_for_a_category_work_in_chat(your_config: Path) -> None:
    with harness(
        profile="email",
        config_dir=your_config,
        env={"IRIS_CONFIG_DIR": str(your_config)},
        fake_model=SCRIPT,
    ) as h:
        assert h.plugin_loaded("email_workflows")
        result = h.chat(CORRECTION)

        # Your categories file made this a correction: answered by the judge's handler
        # (no judged email matches "carpool" in this empty mailbox, and it says so).
        assert 'find a judged email like "carpool"' in result.text
        assert h.model_calls() == ()
        rows = h.audit_rows(hook_point="pre_response", session_id=result.session_id)
        assert [row.handler for row in rows if row.deterministic] == [REBUCKET]


def test_with_the_shipped_categories_the_same_words_mean_nothing(shipped_config: Path) -> None:
    with harness(
        profile="email",
        config_dir=shipped_config,
        env={"IRIS_CONFIG_DIR": str(shipped_config)},
        fake_model=SCRIPT,
    ) as h:
        result = h.chat(CORRECTION)

        # Not a correction: the turn went on to the email assistant, which answered it
        # from the (empty) mailbox.
        assert "judged email" not in result.text
        rows = h.audit_rows(hook_point="pre_response", session_id=result.session_id)
        assert REBUCKET not in [row.handler for row in rows if row.deterministic]
