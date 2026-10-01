"""Tests for running REPL slash commands on behalf of an HTTP caller.

Covers the two things that make this safe: the read-only table actually refuses
writes, and warming a model is suppressed when the API is the one running the
handler (there the call would re-enter the same process and block on an Ollama
load for up to 30s).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from iris_harness.cli.commands import _warm_model, _warmup_url
from iris_harness.cli.web_commands import (
    WEB_READONLY,
    ModelOverrides,
    WebCommandError,
    is_web_supported,
    run_web_command,
    web_supported_names,
)

# ── the warmup opt-out ────────────────────────────────────────────────────────


def test_repl_context_still_warms() -> None:
    """The REPL wants the 5-30s load paid now, with a spinner, not on next turn."""
    ctx = SimpleNamespace(api_url="http://localhost:8003")

    assert _warmup_url(ctx) == "http://localhost:8003"


def test_a_caller_that_opts_out_gets_no_url() -> None:
    ctx = SimpleNamespace(api_url="http://localhost:8003", warm_models=False)

    assert _warmup_url(ctx) == ""


def test_warm_model_without_a_url_does_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must return before any network setup, not fail inside it."""
    import urllib.request

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("warmup attempted a request with no api_url")

    monkeypatch.setattr(urllib.request, "urlopen", _explode)
    _warm_model("", role="executor", model="qwen3.5:latest")


# ── the read-only table ───────────────────────────────────────────────────────


def test_read_subcommands_are_allowed_and_writes_are_not() -> None:
    assert is_web_supported("/reminders", "list") is True
    assert is_web_supported("/reminders", "due") is True
    assert is_web_supported("/reminders", "missed") is True
    assert is_web_supported("/reminders", "tick") is False
    assert is_web_supported("/reminders", "add 2026-01-01 09:00 rent") is False
    assert is_web_supported("/routines", "approve 3") is False
    assert is_web_supported("/queue", "promote some-slug") is False
    assert is_web_supported("/active", "done 2") is False


def test_commands_outside_the_table_are_not_supported() -> None:
    for name in ("/exit", "/export", "/compact", "/nonsense"):
        assert is_web_supported(name) is False


def test_free_form_argument_commands_accept_anything() -> None:
    """`/session 20` and `/replay 3` take a count; there is nothing to gate."""
    assert WEB_READONLY["/session"] is None
    assert is_web_supported("/session", "20") is True
    assert is_web_supported("/replay", "3") is True


def test_running_a_write_subcommand_raises_and_says_what_is_allowed() -> None:
    with pytest.raises(WebCommandError) as excinfo:
        run_web_command(
            "/reminders",
            "add 2026-01-01 09:00 rent",
            session_id="s1",
            overrides=ModelOverrides(),
            provider_manager=None,
        )

    message = str(excinfo.value)
    assert "read-only" in message
    assert "list" in message


def test_running_an_unsupported_command_raises() -> None:
    with pytest.raises(WebCommandError):
        run_web_command(
            "/export",
            "dump.txt",
            session_id="s1",
            overrides=ModelOverrides(),
            provider_manager=None,
        )


def test_every_listed_command_is_a_real_command() -> None:
    """A typo in the table would silently make a command unreachable."""
    from iris_harness.cli.commands import visible_commands

    known = {c.name for c in visible_commands()}
    assert web_supported_names() <= known
