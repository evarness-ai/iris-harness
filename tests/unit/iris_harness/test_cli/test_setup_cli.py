"""``iris setup``: step orchestration, resumability, and the auth secret.

Each mandatory step is stubbed at the ``_step_*`` boundary rather than re-exercising
``iris doctor`` itself (that has its own tests); this file is about the wizard's own
logic -- order, resumption, abort-on-failure, ``--status``/``--reset``, and the
optional services/Telegram/email prompts. The Telegram ``getUpdates`` poll itself is
``services/system/telegram_pairing.py``'s own module, tested in
``tests/unit/iris_harness/services/test_system/test_telegram_pairing.py``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from iris_harness.cli import setup as setup_cli
from iris_harness.main import app
from iris_harness.runtime import welcome as welcome_mod
from iris_harness.services.system import setup_state

runner = CliRunner()


@pytest.fixture()
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("IRIS_HOME", str(home))
    return home


@pytest.fixture()
def mandatory_steps_ok(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Preflight and home/secret both succeed, instantly, with no real side effects."""
    calls = {"preflight": 0, "home_secret": 0}

    def preflight(*, interactive: bool) -> bool:
        calls["preflight"] += 1
        return True

    def home_secret() -> None:
        calls["home_secret"] += 1

    monkeypatch.setattr(setup_cli, "_step_preflight", preflight)
    monkeypatch.setattr(setup_cli, "_step_home_secret", home_secret)
    return calls


def _state_after() -> setup_state.SetupState:
    return setup_state.load_state()


def test_fresh_run_records_all_five_steps_non_interactively(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(setup_cli, "_interactive", lambda: False)
    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0, result.output
    state = _state_after()
    assert state.steps["preflight"].status == "done"
    assert state.steps["home_secret"].status == "done"
    assert state.steps["services"].status == "skipped"
    assert state.steps["services"].detail == "not a terminal"
    assert state.steps["telegram"].status == "skipped"
    assert state.steps["telegram"].detail == "not a terminal"
    assert state.steps["email"].status == "skipped"
    assert mandatory_steps_ok == {"preflight": 1, "home_secret": 1}


def test_closing_summary_names_real_commands_and_the_webui_port(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live bug: the summary said `iris chat` (no such command -- the REPL is
    bare `iris`) and pointed the web UI at IRIS_API_URL/8003 (the raw API, which
    rejects a browser with no Authorization header as "invalid bearer token")
    instead of the Vite dev console's actual port, IRIS_WEBUI_PORT/5181."""
    monkeypatch.setattr(setup_cli, "_interactive", lambda: False)
    monkeypatch.delenv("IRIS_WEBUI_PORT", raising=False)
    monkeypatch.delenv("IRIS_API_HOST", raising=False)
    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0, result.output
    assert "iris chat" not in result.output
    assert "chat now:            iris" in result.output
    assert "http://127.0.0.1:5181" in result.output
    assert ":8003" not in result.output


def test_fresh_run_marks_welcome_as_shown(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(setup_cli, "_interactive", lambda: False)
    runner.invoke(app, ["setup"])
    marker = welcome_mod.welcome_marker_path()
    assert marker.exists()
    record = json.loads(marker.read_text(encoding="utf-8"))
    assert record["skipped"] is True
    assert "iris setup" in record["reason"]


def test_resume_skips_steps_already_recorded(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    setup_state.record_step("preflight", "done")
    setup_state.record_step("home_secret", "done")

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("mandatory step re-ran on resume")

    monkeypatch.setattr(setup_cli, "_step_preflight", boom)
    monkeypatch.setattr(setup_cli, "_step_home_secret", boom)
    monkeypatch.setattr(setup_cli, "_interactive", lambda: False)

    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0, result.output
    state = _state_after()
    assert state.steps["services"].status == "skipped"
    assert state.steps["telegram"].status == "skipped"
    assert state.steps["email"].status == "skipped"


def test_not_ready_preflight_aborts_before_later_steps(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(setup_cli, "_step_preflight", lambda *, interactive: False)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("home & secret ran after preflight failed")

    monkeypatch.setattr(setup_cli, "_step_home_secret", boom)
    monkeypatch.setattr(setup_cli, "_interactive", lambda: False)

    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 1
    state = _state_after()
    assert "preflight" not in state.steps


def test_status_flag_reports_without_calling_any_step(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_state.record_step("preflight", "done")

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("--status ran a step")

    monkeypatch.setattr(setup_cli, "_step_preflight", boom)
    monkeypatch.setattr(setup_cli, "_step_home_secret", boom)

    result = runner.invoke(app, ["setup", "--status"])
    assert result.exit_code == 0
    assert "Preflight" in result.output


def test_status_flag_on_empty_state_says_nothing_recorded(
    home: Path,
) -> None:
    result = runner.invoke(app, ["setup", "--status"])
    assert result.exit_code == 0
    assert "No setup progress recorded" in result.output


def test_reset_clears_state_then_the_wizard_runs_again(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_state.record_step("preflight", "done")
    setup_state.record_step("home_secret", "done")
    setup_state.record_step("services", "skipped", detail="declined")
    setup_state.record_step("telegram", "skipped", detail="declined")
    setup_state.record_step("email", "skipped", detail="declined")
    monkeypatch.setattr(setup_cli, "_interactive", lambda: False)

    result = runner.invoke(app, ["setup", "--reset"])
    assert result.exit_code == 0, result.output
    assert mandatory_steps_ok == {"preflight": 1, "home_secret": 1}


def test_interactive_optional_steps_are_skipped_on_decline(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)
    result = runner.invoke(app, ["setup"], input="n\nn\nn\n")
    assert result.exit_code == 0, result.output
    state = _state_after()
    assert state.steps["services"].status == "skipped"
    assert state.steps["services"].detail == "declined"
    assert state.steps["telegram"].status == "skipped"
    assert state.steps["telegram"].detail == "declined"
    assert state.steps["email"].status == "skipped"
    assert state.steps["email"].detail == "declined"


def test_services_accept_but_script_missing_is_skipped_with_reason(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)  # no scripts/start_iris.sh here
    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)
    result = runner.invoke(app, ["setup"], input="y\nn\nn\n")
    assert result.exit_code == 0, result.output
    state = _state_after()
    assert state.steps["services"].status == "skipped"
    assert state.steps["services"].detail == "start_iris.sh not found"


def test_services_accept_and_script_succeeds_is_recorded_done(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    script = scripts_dir / "start_iris.sh"
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)

    class FakeResult:
        returncode = 0

    monkeypatch.setattr(setup_cli.subprocess, "run", lambda argv, **kw: FakeResult())
    result = runner.invoke(app, ["setup"], input="y\nn\nn\n")
    assert result.exit_code == 0, result.output
    state = _state_after()
    assert state.steps["services"].status == "done"


def test_services_script_nonzero_exit_is_recorded_failed_not_skipped(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "start_iris.sh").write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)

    class FakeResult:
        returncode = 1

    monkeypatch.setattr(setup_cli.subprocess, "run", lambda argv, **kw: FakeResult())
    result = runner.invoke(app, ["setup"], input="y\nn\nn\n")
    assert result.exit_code == 0, result.output
    state = _state_after()
    assert state.steps["services"].status == "failed"
    assert "exited 1" in state.steps["services"].detail


def test_services_script_cannot_even_start_does_not_crash_the_wizard(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The bug: a venv script with a blocked exec permission (seen live: macOS's
    com.apple.provenance denying exec on an rwxr-xr-x file) raised PermissionError
    straight out of subprocess.run and crashed the whole CLI. One optional step
    failing to even start must not take down the other steps' summary."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "start_iris.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)

    def boom(argv: list[str], **kw: Any) -> Any:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(setup_cli.subprocess, "run", boom)
    result = runner.invoke(app, ["setup"], input="y\nn\nn\n")
    assert result.exit_code == 0, result.output
    assert "Setup complete" in result.output
    state = _state_after()
    assert state.steps["services"].status == "failed"
    assert "Permission denied" in state.steps["services"].detail


def test_telegram_empty_token_is_skipped_not_a_hang(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare Enter at the token prompt must be accepted as empty input and
    recorded as skipped -- not re-prompt forever (typer.prompt needs an explicit
    default="" for that; the same bug the email-address prompt had)."""

    def boom(token: str) -> None:
        raise AssertionError("no lookup should run with no token")

    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)
    monkeypatch.setattr(setup_cli, "get_telegram_bot_username", boom)
    result = runner.invoke(app, ["setup"], input="n\ny\n\nn\n")
    assert result.exit_code == 0, result.output
    state = _state_after()
    assert state.steps["telegram"].status == "skipped"
    assert state.steps["telegram"].detail == "no token entered"


def test_telegram_accept_but_no_message_is_recorded_failed_not_skipped(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)
    monkeypatch.setattr(setup_cli, "get_telegram_bot_username", lambda token: "TestBot")
    monkeypatch.setattr(setup_cli, "poll_telegram_chat_id", lambda token: None)
    monkeypatch.setattr(setup_cli, "_env_path", lambda: home / ".env")
    result = runner.invoke(app, ["setup"], input="n\ny\nfake-bot-token\nn\n")
    assert result.exit_code == 0, result.output
    state = _state_after()
    # "failed", not "skipped": a wrong token or a missed message is not a choice,
    # so --- unlike a declined step --- it must come back up on the next run.
    assert state.steps["telegram"].status == "failed"
    assert state.steps["telegram"].detail == "no message received"


def test_a_failed_telegram_attempt_is_retried_on_the_next_run(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug: a wrong bot token timed out, got recorded, and every later `iris
    setup` silently skipped Telegram forever -- even though the CLI's own message
    says "re-run `iris setup` to try again". `needs_run()` is the fix."""
    setup_state.record_step("preflight", "done")
    setup_state.record_step("home_secret", "done")
    setup_state.record_step("services", "skipped", detail="declined")
    setup_state.record_step("telegram", "failed", detail="no message received")

    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)
    monkeypatch.setattr(setup_cli, "get_telegram_bot_username", lambda token: "TestBot")
    monkeypatch.setattr(setup_cli, "poll_telegram_chat_id", lambda token: "42")
    monkeypatch.setattr(setup_cli, "_env_path", lambda: home / ".env")

    # Only the still-pending steps prompt: Telegram (retried) then email.
    result = runner.invoke(app, ["setup"], input="y\ncorrected-token\nn\n")
    assert result.exit_code == 0, result.output
    state = _state_after()
    assert state.steps["telegram"].status == "done"
    assert state.steps["telegram"].detail == "chat_id=42"
    # The earlier, genuinely declined step was NOT re-prompted.
    assert state.steps["services"].status == "skipped"


def test_telegram_accept_and_paired_writes_env_and_records_chat_id(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = home / ".env"
    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)
    monkeypatch.setattr(setup_cli, "get_telegram_bot_username", lambda token: "TestBot")
    monkeypatch.setattr(setup_cli, "poll_telegram_chat_id", lambda token: "987654321")
    monkeypatch.setattr(setup_cli, "_env_path", lambda: env_path)
    result = runner.invoke(app, ["setup"], input="n\ny\nfake-bot-token\nn\n")
    assert result.exit_code == 0, result.output
    assert "https://t.me/TestBot?start=setup" in result.output
    state = _state_after()
    assert state.steps["telegram"].status == "done"
    assert state.steps["telegram"].detail == "chat_id=987654321"
    contents = env_path.read_text(encoding="utf-8")
    assert "TELEGRAM_BOT_TOKEN" in contents
    assert "987654321" in contents


def test_telegram_username_lookup_fails_falls_back_to_manual_instructions(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)
    monkeypatch.setattr(setup_cli, "get_telegram_bot_username", lambda token: None)
    monkeypatch.setattr(setup_cli, "poll_telegram_chat_id", lambda token: "987654321")
    monkeypatch.setattr(setup_cli, "_env_path", lambda: home / ".env")
    result = runner.invoke(app, ["setup"], input="n\ny\nfake-bot-token\nn\n")
    assert result.exit_code == 0, result.output
    assert "t.me/" not in result.output
    assert "couldn't look up the bot's username" in result.output
    state = _state_after()
    assert state.steps["telegram"].status == "done"


def test_email_demo_choice_runs_email_demo_and_records_success(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []

    class FakeResult:
        returncode = 0

    def fake_run(argv: list[str], **kwargs: Any) -> FakeResult:
        captured.append(argv)
        return FakeResult()

    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)
    monkeypatch.setattr(setup_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(setup_cli, "_iris_argv", lambda: ["iris"])
    result = runner.invoke(app, ["setup"], input="n\nn\ny\n1\n")
    assert result.exit_code == 0, result.output
    assert captured == [["iris", "email", "demo"]]
    state = _state_after()
    assert state.steps["email"].status == "done"


def test_email_demo_subprocess_nonzero_exit_is_recorded_failed_not_skipped(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeResult:
        returncode = 2

    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)
    monkeypatch.setattr(setup_cli.subprocess, "run", lambda argv, **kw: FakeResult())
    monkeypatch.setattr(setup_cli, "_iris_argv", lambda: ["iris"])
    result = runner.invoke(app, ["setup"], input="n\nn\ny\n1\n")
    assert result.exit_code == 0, result.output
    state = _state_after()
    assert state.steps["email"].status == "failed"
    assert "exited 2" in state.steps["email"].detail


def test_email_demo_subprocess_cannot_even_start_does_not_crash_the_wizard(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(argv: list[str], **kw: Any) -> Any:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)
    monkeypatch.setattr(setup_cli.subprocess, "run", boom)
    monkeypatch.setattr(setup_cli, "_iris_argv", lambda: ["iris"])
    result = runner.invoke(app, ["setup"], input="n\nn\ny\n1\n")
    assert result.exit_code == 0, result.output
    assert "Setup complete" in result.output
    state = _state_after()
    assert state.steps["email"].status == "failed"
    assert "Permission denied" in state.steps["email"].detail


def test_email_gmail_choice_runs_auth_login_then_setup(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []

    class FakeResult:
        returncode = 0

    def fake_run(argv: list[str], **kwargs: Any) -> FakeResult:
        captured.append(argv)
        return FakeResult()

    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)
    monkeypatch.setattr(setup_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(setup_cli, "_iris_argv", lambda: ["iris"])
    # No OAuth client at this fresh $IRIS_HOME, so the "Continue anyway?" prompt
    # fires first (y = go ahead and let the real `iris auth gmail login` surface
    # its own FileNotFoundError, same as today).
    result = runner.invoke(app, ["setup"], input="n\nn\ny\n2\ny\nowner@example.com\n")
    assert result.exit_code == 0, result.output
    assert captured == [
        ["iris", "auth", "gmail", "login", "--user", "owner@example.com"],
        ["iris", "email", "setup"],
    ]
    state = _state_after()
    assert state.steps["email"].status == "done"


def test_email_gmail_choice_declines_continuing_without_oauth_client(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(argv: list[str], **kw: Any) -> Any:
        raise AssertionError("no subprocess should run once the owner declines")

    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)
    monkeypatch.setattr(setup_cli.subprocess, "run", boom)
    result = runner.invoke(app, ["setup"], input="n\nn\ny\n2\nn\n")
    assert result.exit_code == 0, result.output
    assert "docs/usage-guides/gmail-auth.md" in result.output
    state = _state_after()
    assert state.steps["email"].status == "skipped"
    assert state.steps["email"].detail == "no Google OAuth client configured yet"


def test_email_imap_choice_runs_auth_login_then_setup(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []

    class FakeResult:
        returncode = 0

    def fake_run(argv: list[str], **kwargs: Any) -> FakeResult:
        captured.append(argv)
        return FakeResult()

    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)
    monkeypatch.setattr(setup_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(setup_cli, "_iris_argv", lambda: ["iris"])
    result = runner.invoke(app, ["setup"], input="n\nn\ny\n3\nowner@example.com\n")
    assert result.exit_code == 0, result.output
    assert "app password" in result.output
    assert "docs/getting-started/connect-your-mailbox.md" in result.output
    assert captured == [
        ["iris", "auth", "imap", "login", "--user", "owner@example.com"],
        ["iris", "email", "setup"],
    ]
    state = _state_after()
    assert state.steps["email"].status == "done"


def test_imap_choice_shows_a_domain_specific_app_password_hint(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reminder prints right before the password prompt fires (inside the
    subprocess), not just once in the earlier generic paragraph -- by the time the
    real prompt appears, that earlier text has scrolled past."""

    class FakeResult:
        returncode = 0

    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)
    monkeypatch.setattr(setup_cli.subprocess, "run", lambda argv, **kw: FakeResult())
    monkeypatch.setattr(setup_cli, "_iris_argv", lambda: ["iris"])
    result = runner.invoke(app, ["setup"], input="n\nn\ny\n3\nowner@gmail.com\n")
    assert result.exit_code == 0, result.output
    assert "myaccount.google.com" in result.output.split()


def test_imap_choice_falls_back_to_generic_hint_for_an_unknown_domain(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeResult:
        returncode = 0

    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)
    monkeypatch.setattr(setup_cli.subprocess, "run", lambda argv, **kw: FakeResult())
    monkeypatch.setattr(setup_cli, "_iris_argv", lambda: ["iris"])
    result = runner.invoke(app, ["setup"], input="n\nn\ny\n3\nowner@example.com\n")
    assert result.exit_code == 0, result.output
    assert "account security settings" in result.output


def test_email_gmail_login_failure_stops_before_running_setup(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []

    class FakeResult:
        returncode = 1

    def fake_run(argv: list[str], **kwargs: Any) -> FakeResult:
        captured.append(argv)
        return FakeResult()

    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)
    monkeypatch.setattr(setup_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(setup_cli, "_iris_argv", lambda: ["iris"])
    result = runner.invoke(app, ["setup"], input="n\nn\ny\n2\ny\nowner@example.com\n")
    assert result.exit_code == 0, result.output
    # email setup was never reached -- nothing to walk through since login failed.
    assert captured == [["iris", "auth", "gmail", "login", "--user", "owner@example.com"]]
    state = _state_after()
    assert state.steps["email"].status == "failed"
    assert "gmail login failed" in state.steps["email"].detail


def test_email_empty_choice_is_skipped_declined(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(argv: list[str], **kw: Any) -> Any:
        raise AssertionError("no subprocess should run for an empty choice")

    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)
    monkeypatch.setattr(setup_cli.subprocess, "run", boom)
    result = runner.invoke(app, ["setup"], input="n\nn\ny\n\n")
    assert result.exit_code == 0, result.output
    state = _state_after()
    assert state.steps["email"].status == "skipped"
    assert state.steps["email"].detail == "declined"


def test_email_invalid_choice_is_skipped_declined(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(argv: list[str], **kw: Any) -> Any:
        raise AssertionError("no subprocess should run for an invalid choice")

    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)
    monkeypatch.setattr(setup_cli.subprocess, "run", boom)
    result = runner.invoke(app, ["setup"], input="n\nn\ny\n9\n")
    assert result.exit_code == 0, result.output
    assert "not a valid choice" in result.output
    state = _state_after()
    assert state.steps["email"].status == "skipped"
    assert state.steps["email"].detail == "declined"


def test_email_no_address_entered_is_skipped(
    home: Path, mandatory_steps_ok: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(argv: list[str], **kw: Any) -> Any:
        raise AssertionError("no subprocess should run with no address")

    monkeypatch.setattr(setup_cli, "_interactive", lambda: True)
    monkeypatch.setattr(setup_cli.subprocess, "run", boom)
    # Choice 3 (IMAP), not Gmail: IMAP has no "Continue anyway?" prerequisite
    # prompt in between, so this isolates the address-prompt behavior cleanly.
    result = runner.invoke(app, ["setup"], input="n\nn\ny\n3\n\n")
    assert result.exit_code == 0, result.output
    state = _state_after()
    assert state.steps["email"].status == "skipped"
    assert state.steps["email"].detail == "no address entered"


def test_iris_argv_uses_the_current_interpreter_with_dash_m() -> None:
    """Not the venv's bin/iris script: that file's own exec permission can be
    blocked by the OS (macOS's com.apple.provenance, seen live) even when its
    Unix permission bits look fine. sys.executable is always executable."""
    assert setup_cli._iris_argv() == [sys.executable, "-m", "iris_harness.main"]


def test_ensure_auth_secret_generates_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_path = tmp_path / ".env"
    monkeypatch.delenv("IRIS_AUTH_SECRET", raising=False)
    monkeypatch.setattr(setup_cli, "_env_path", lambda: env_path)
    setup_cli._ensure_auth_secret()
    assert env_path.exists()
    contents = env_path.read_text(encoding="utf-8")
    assert "IRIS_AUTH_SECRET=" in contents
    assert os.environ["IRIS_AUTH_SECRET"]


def test_ensure_auth_secret_noop_when_already_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_path = tmp_path / ".env"
    monkeypatch.setenv("IRIS_AUTH_SECRET", "already-set-secret")
    monkeypatch.setattr(setup_cli, "_env_path", lambda: env_path)
    setup_cli._ensure_auth_secret()
    assert not env_path.exists()
