"""The email slice's part of the stable tier (OSS plan R16) is real code.

``iris_harness/sdk/stable_tier.yaml`` names the mail-provider facade
(``email.provider_api``) -- the core cannot re-export it -- and the ``iris email``
quickstart commands. The core's
test (tests/unit/test_stable_tier.py) cannot import the email slice, so the names are
checked here, beside it.
"""

from __future__ import annotations

import importlib

import pytest

from iris_harness.testing import stable_tier


@pytest.mark.parametrize("module", sorted(stable_tier().names))
def test_every_declared_name_exists(module: str) -> None:
    mod = importlib.import_module(module)
    missing = sorted(name for name in stable_tier().names[module] if not hasattr(mod, name))
    assert missing == [], f"{module} lacks declared stable names: {missing}"


def test_the_provider_protocols_are_the_declared_ones() -> None:
    from iris_personal.email.provider_api import LabellingProvider, MailProvider, MailSyncStore

    declared = stable_tier().names["iris_personal.email.provider_api"]
    assert {MailProvider.__name__, LabellingProvider.__name__, MailSyncStore.__name__} <= declared


def test_the_write_approval_command_is_stable_with_the_options_the_scaffold_uses() -> None:
    """``iris email writes approve --account <id> --yes`` is what every refused write
    names and what the ``mail-provider`` scaffold's generated test runs, so its name and
    those two options are part of the tier."""
    from typer.testing import CliRunner

    from iris_harness.main import app

    assert ("email", "writes", "approve") in stable_tier().quickstart_cli
    result = CliRunner().invoke(app, ["email", "writes", "approve", "--help"])
    assert result.exit_code == 0, result.output
    assert "--account" in result.output and "--yes" in result.output


def test_the_governed_write_is_stable_without_its_internals() -> None:
    """``mailbox_write`` is the stable way a provider writes; its signature carries no
    store or ledger seam (those are the slice's own, for its tests)."""
    import inspect

    from iris_personal.email import provider_api, write_approvals

    declared = stable_tier().names["iris_personal.email.provider_api"]
    assert {"mailbox_write", "WriteTally"} <= declared
    assert provider_api.WriteTally is write_approvals.WriteTally
    params = inspect.signature(provider_api.mailbox_write).parameters
    assert list(params) == ["account_id", "what", "op"]
    assert params["op"].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_mail_record_is_not_stable() -> None:
    """A provider gets ``MailSyncStore``; ``EmailStore`` stays the slice's own."""
    declared = stable_tier().names
    assert "iris_personal.email.store" not in declared
    assert all("EmailStore" not in names for names in declared.values())


@pytest.mark.parametrize(
    "command", [cmd for cmd in stable_tier().quickstart_cli if cmd[0] == "email"]
)
def test_the_email_quickstart_commands_exist(command: tuple[str, ...]) -> None:
    from typer.testing import CliRunner

    from iris_harness.main import app

    result = CliRunner().invoke(app, [*command, "--help"])
    assert result.exit_code == 0, result.output
