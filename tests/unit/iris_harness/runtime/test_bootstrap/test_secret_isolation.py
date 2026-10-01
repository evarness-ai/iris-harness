"""The test process must not hold the developer's live credentials.

`conftest.py` neutralizes `load_dotenv` so the repo-root `.env` cannot reach the
suite. That closes the file path but not the inherited one: a developer who
sources `.env` in their shell profile hands every test the live values anyway.

The sharpest case is Telegram. With `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
both present, the only thing between a runtime-building test and a live
long-poll against the real bot is one env-var default. A test that flipped it
would start polling for real, take the `getUpdates` slot from a running
channel_gateway (Telegram allows one consumer per token) and reply to the
developer's actual messages.
"""

from __future__ import annotations

import os

import pytest

# Credentials that reach a real service, or make a run machine-dependent.
# `IRIS_AUTH_SECRET` is excluded on purpose: the suite requires it, and conftest
# sets it to a documented dummy.
LIVE_SECRETS = (
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "IRIS_SPIKE_GMAIL_PASSWORD",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "LM_STUDIO_API_KEY",
    "EXA_API_KEY",
    "HF_TOKEN",
    "GITHUB_TOKEN",
)


@pytest.mark.parametrize("name", LIVE_SECRETS)
def test_live_secret_is_not_visible_to_tests(name: str) -> None:
    assert os.getenv(name) is None, (
        f"{name} is set inside the test process. conftest strips it precisely so a "
        "test cannot reach a real service; something re-populated it, or the name "
        "was dropped from conftest's list."
    )


def test_the_auth_secret_survives_because_the_suite_needs_it() -> None:
    """Guard the guard: over-stripping would break every service test."""
    assert os.getenv("IRIS_AUTH_SECRET")


def test_a_runtime_built_in_tests_cannot_start_a_telegram_poller() -> None:
    """The property that actually matters, not just the absent variable.

    Asserted against the real predicate so it keeps holding if the flag defaults
    are ever rearranged.
    """
    from iris_harness.runtime.bootstrap import _runtime_telegram_poller_enabled

    assert _runtime_telegram_poller_enabled() is False


def test_no_telegram_poller_even_if_a_test_claims_the_gateway_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scenario the strip defends: a test flips the flag that gates the poller.

    Enabling it is legitimate (`test_telegram_poller_policy.py` does), so the
    safety cannot rest on the flag. It rests on there being no real token to poll
    with.
    """
    monkeypatch.setenv("IRIS_CHANNEL_GATEWAY_TELEGRAM_ENABLED", "0")

    from iris_harness.runtime.bootstrap import _runtime_telegram_poller_enabled

    assert _runtime_telegram_poller_enabled() is True  # the flag really did flip
    assert not os.getenv("TELEGRAM_BOT_TOKEN")  # and there is still nothing to poll with
