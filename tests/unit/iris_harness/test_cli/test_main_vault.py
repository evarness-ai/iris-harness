"""``iris vault add``'s hidden-input cleanup (issue #69)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from iris_harness.main import _clean_hidden_input, app


def test_clean_hidden_input_drops_control_characters_and_outer_whitespace() -> None:
    # The escape byte a bracketed-paste marker starts with is dropped; the printable
    # rest of the marker (issue #69's leading suspect) is not parsed out, by design --
    # this is a cheap, safe-regardless mitigation, not a bracketed-paste parser.
    assert _clean_hidden_input("  s3cret\x1b[200~  \n") == "s3cret[200~"
    assert _clean_hidden_input("plain-value") == "plain-value"
    assert _clean_hidden_input("\x07\x00tabbed\x7f") == "tabbed"


def test_vault_add_stores_the_cleaned_value_not_the_raw_capture() -> None:
    store = MagicMock()
    with patch("iris_harness.kernel.governance.vault.VaultStore", return_value=store):
        result = CliRunner().invoke(
            app, ["vault", "add", "a-handle", "--value", "  s3cret\x1b[200~\n"]
        )
    assert result.exit_code == 0, result.output
    kwargs: dict[str, Any] = store.add.call_args.kwargs
    assert kwargs["secret_value"] == "s3cret[200~"
