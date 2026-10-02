"""``iris auth imap login``'s hidden-input cleanup (issue #69)."""

from __future__ import annotations

from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from iris_personal.plugins.imap import cli as imap_cli
from iris_personal.plugins.imap.cli import _clean_hidden_input, cmd_auth_imap_login


def test_clean_hidden_input_drops_control_characters_and_outer_whitespace() -> None:
    assert _clean_hidden_input("  s3cret\x1b[200~  \n") == "s3cret[200~"
    assert _clean_hidden_input("plain-value") == "plain-value"


def test_login_prompt_passes_the_cleaned_password_to_connect_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_connect_account(account: Any, **kw: Any) -> Any:
        captured["password"] = account.password
        return type("Row", (), {"address": account.address, "id": "acct-1"})()

    monkeypatch.setattr(imap_cli, "connect_account", fake_connect_account)
    app = typer.Typer()
    app.command()(cmd_auth_imap_login)

    result = CliRunner().invoke(
        app,
        ["--user", "owner@example.com", "--host", "imap.example.com"],
        input="  s3cret\x1b[200~  \n",
    )

    assert result.exit_code == 0, result.output
    assert captured["password"] == "s3cret[200~"
