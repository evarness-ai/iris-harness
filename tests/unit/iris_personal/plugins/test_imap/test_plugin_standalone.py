"""The IMAP plugin, pinned from both sides: it plugs into the core's keyed registries
(mail provider + credential row), imports nothing of the core but the SDK, and its CLI
adds an account against the fake server with the password going only to the vault."""

from __future__ import annotations

import ast
from pathlib import Path

import typer
from typer.testing import CliRunner

from iris_harness.runtime.plugin_host.api import HarnessServices, PluginAPI
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus
from iris_personal.email import providers
from iris_personal.email.accounts import EmailAccountStore

from .conftest import PASSWORD, USER
from .fake_imap_server import FakeImapServer, FakeMailbox

PLUGIN_DIR = Path("src/iris_personal/plugins/imap")


def _mount(tmp_path: Path) -> PluginRegistry:
    from iris_personal.plugins.imap import plugin

    registry = PluginRegistry()
    registry.add_plugin(PluginRecord(name="imap", source="builtin", status=PluginStatus.LOADED))
    api = PluginAPI(
        plugin="imap",
        services=HarnessServices(
            config_dir=tmp_path / "config",
            data_dir=tmp_path / "data",
            tier_router=None,
            agent_executor=None,
            heartbeats=None,
            channels=None,
            deterministic_reply=lambda **kw: None,
            events=None,
        ),
        registry=registry,
    )
    plugin.setup(api)
    return registry


def test_setup_registers_the_mail_provider_and_the_credential_row(tmp_path: Path) -> None:
    from iris_harness.services.health import credentials

    providers.clear_mail_providers()
    credentials.clear_credential_checks()
    try:
        registry = _mount(tmp_path)
        provider = providers.mail_provider_for("imap:someone@example.test")
        assert provider is not None and provider.name == "imap"
        assert isinstance(provider, providers.MailProvider)
        assert isinstance(provider, providers.LabellingProvider)
        assert "imap_credentials" in credentials._registered
        assert ("imap", "credential_check", "imap_credentials") in registry.seams()
        rec = registry.get("imap")
        assert rec is not None and rec.registrations == []
    finally:
        providers.clear_mail_providers()
        credentials.clear_credential_checks()


def test_the_plugin_imports_only_the_sdk_from_the_core() -> None:
    for path in PLUGIN_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for name in names:
                if name.startswith("iris_harness"):
                    assert name.startswith("iris_harness.sdk"), f"{path.name}: {name}"
                assert not name.startswith("iris_personal.plugins."), f"{path.name}: {name}"


def test_manifest_declares_no_registration_kind_on_purpose() -> None:
    from iris_harness.runtime.plugin_host.manifest import load_manifest

    manifest = load_manifest(PLUGIN_DIR / "manifest.yaml")
    assert manifest.name == "imap" and manifest.provides == ()


def test_the_personal_profile_mounts_imap_before_the_workflows() -> None:
    # Every shipped profile that mounts both: `email` in every tree, `personal-assistant`
    # where the private domains are (the public tree, OSS plan R1, has only `email`).
    mounting = [
        text
        for text in (
            q.read_text(encoding="utf-8") for q in sorted(Path("config/profiles").glob("*.yaml"))
        )
        if "name: imap" in text and "name: email_workflows" in text
    ]
    assert mounting, "no shipped profile mounts imap with email_workflows"
    for text in mounting:
        assert text.index("name: imap") < text.index("name: email_workflows")


def _cli() -> typer.Typer:
    from iris_personal.plugins.imap import cli

    app = typer.Typer()
    app.command("login")(cli.cmd_auth_imap_login)
    app.command("status")(cli.cmd_auth_imap_status)
    app.command("logout")(cli.cmd_auth_imap_logout)
    return app


def test_cli_login_status_logout_against_the_fake_server() -> None:
    from iris_personal.plugins.imap.account import load_account

    box = FakeMailbox(users={USER: PASSWORD})
    runner = CliRunner()
    with FakeImapServer(box) as srv:
        args = ["login", "--user", USER, "--host", srv.host, "--port", str(srv.port)]
        args += ["--security", "plain", "--password-stdin"]

        wrong = runner.invoke(_cli(), args, input="not-it\n")
        assert wrong.exit_code == 3, wrong.output
        assert load_account(f"imap:{USER}") is None  # nothing saved on a failed login

        ok = runner.invoke(_cli(), args, input=PASSWORD + "\n")
        assert ok.exit_code == 0, ok.output
        assert PASSWORD not in ok.output

        saved = load_account(f"imap:{USER}")
        assert saved is not None and saved.password == PASSWORD and saved.port == srv.port
        row = EmailAccountStore().get(f"imap:{USER}")
        assert row is not None and row.provider == "imap" and row.active

        status = runner.invoke(_cli(), ["status"])
        assert status.exit_code == 0 and USER in status.output
        assert "not approved" in status.output and PASSWORD not in status.output

        out = runner.invoke(_cli(), ["logout", "--user", USER])
        assert out.exit_code == 0
        assert load_account(f"imap:{USER}") is None
        gone = EmailAccountStore().get(f"imap:{USER}")
        assert gone is not None and not gone.active

        again = runner.invoke(_cli(), args, input=PASSWORD + "\n")
        assert again.exit_code == 0, again.output
        back = EmailAccountStore().get(f"imap:{USER}")
        assert back is not None and back.active  # a logout's row comes back to life


def test_cli_login_refuses_plaintext_to_a_remote_host() -> None:
    result = CliRunner().invoke(
        _cli(),
        ["login", "--user", USER, "--host", "imap.example.test", "--security", "plain",
         "--password-stdin"],
        input="x\n",
    )  # fmt: skip
    assert result.exit_code == 2 and "plaintext" in result.output
