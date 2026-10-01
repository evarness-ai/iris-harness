"""The Gmail plugin, pinned from both sides (OSS plan M5.7, track A slice 2).

Core = mechanisms: the store, the reads over it, the provider interface and the sweep
that drives whichever provider is registered. The plugin = everything that talks to
Gmail. It registers through two keyed core registries, not a ``PluginAPI`` kind — the
same shape as pending-action providers and health check providers.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

from iris_harness.runtime.plugin_host.api import HarnessServices, PluginAPI
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus
from iris_personal.email import providers


def _mount(tmp_path: Path) -> PluginRegistry:
    from iris_personal.plugins.gmail import plugin

    registry = PluginRegistry()
    registry.add_plugin(PluginRecord(name="gmail", source="builtin", status=PluginStatus.LOADED))
    api = PluginAPI(
        plugin="gmail",
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


# ─── The core's share imports nothing from the plugin ─────────────────────────


def test_core_email_modules_import_nothing_from_the_plugin() -> None:
    from iris_personal.email import agent_tools, digest, events, providers, store, sweep

    for module in (store, digest, agent_tools, sweep, events, providers):
        source = inspect.getsource(module)
        assert "plugins_builtin" not in source, f"{module.__name__} imports a plugin"
        assert "googleapiclient" not in source, f"{module.__name__} talks to Google itself"


def test_importing_the_core_store_and_sweep_loads_no_google_client() -> None:
    probe = (
        "import sys; from iris_personal.email.store import EmailStore; "
        "from iris_personal.email.sweep import build_email_sweep_handler; "
        "print(sorted(n for n in sys.modules if n.startswith('iris_harness.plugins_builtin') "
        "or n.startswith('googleapiclient')))"
    )
    cmd = [sys.executable, "-c", probe]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    assert proc.stdout.strip() == "[]", proc.stdout


def test_core_credential_checks_no_longer_name_gmail() -> None:
    from iris_harness.services.health import credentials

    source = inspect.getsource(credentials)
    assert "Gmail" not in source and "gmail" not in source


def test_the_core_cli_no_longer_defines_the_gmail_auth_commands() -> None:
    from iris_harness import main

    assert not hasattr(main, "gmail_app")
    assert not hasattr(main, "cmd_auth_gmail_login")


# ─── The plugin plugs into the core's registries ──────────────────────────────


def test_setup_registers_the_mail_provider_and_the_credential_row(tmp_path: Path) -> None:
    from iris_harness.services.health import credentials

    providers.clear_mail_providers()
    credentials.clear_credential_checks()
    try:
        registry = _mount(tmp_path)

        provider = providers.mail_provider_for("gmail:user@gmail.com")
        assert provider is not None and provider.name == "gmail"
        assert isinstance(provider, providers.MailProvider)
        assert "gmail_credentials" in credentials._registered
        assert ("gmail", "credential_check", "gmail_credentials") in registry.seams()
        rec = registry.get("gmail")
        assert rec is not None and rec.registrations == []  # no PluginAPI kind: by design
    finally:
        providers.clear_mail_providers()
        credentials.clear_credential_checks()


def test_the_provider_delegates_at_call_time_so_patches_land(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Tests of the core tools patch ``gmail_fetch.fetch_message_body``; the provider
    must read the module attribute when called, not capture the function at import."""
    from iris_personal.plugins.gmail import gmail_fetch
    from iris_personal.plugins.gmail.provider import GmailProvider

    monkeypatch.setattr(gmail_fetch, "fetch_message_body", lambda _a, _m, **_k: "patched body")
    assert GmailProvider().fetch_message_body("gmail:u@x", "m1") == "patched body"


def test_manifest_declares_no_registration_kind_on_purpose() -> None:
    from iris_harness.runtime.plugin_host.manifest import load_manifest

    path = Path("src/iris_personal/plugins/gmail/manifest.yaml")
    assert load_manifest(path).provides == ()
    assert "provides: []" in path.read_text(encoding="utf-8")


def test_the_personal_profile_mounts_gmail_before_the_workflows_that_need_it() -> None:
    """Order matters: the provider has to register before a workflow looks it up.

    `email` and `personal-assistant` carry them -- `default` is core-only (decision 10),
    so a core-only install has neither gmail nor email_workflows.
    """
    # Every shipped profile that mounts both: `email` in every tree, `personal-assistant`
    # where the private domains are (the public tree, OSS plan R1, has only `email`).
    mounting = [
        text
        for text in (
            q.read_text(encoding="utf-8") for q in sorted(Path("config/profiles").glob("*.yaml"))
        )
        if "name: gmail" in text and "name: email_workflows" in text
    ]
    assert mounting, "no shipped profile mounts gmail with email_workflows"
    for text in mounting:
        assert text.index("name: gmail") < text.index("name: email_workflows")

    default = Path("config/profiles/default.yaml").read_text(encoding="utf-8")
    assert "name: gmail" not in default and "name: email_workflows" not in default


def test_setup_registers_the_token_refresh_repair(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """ADR-0116: the plugin that owns the credential owns its first fix."""
    from iris_harness.services.health import credentials, repair
    from iris_harness.services.health import service as health_service
    from iris_harness.services.health.models import CheckKind, HealthCheck, HealthState
    from iris_personal.plugins.gmail import gmail_oauth

    providers.clear_mail_providers()
    health_service.clear_check_providers()
    repair.clear_repairers()
    try:
        _mount(tmp_path)
        refreshed: list[str] = []

        def fake_refresh(account: str) -> bool:
            refreshed.append(account)
            return False  # revoked

        monkeypatch.setattr(gmail_oauth, "force_refresh", fake_refresh)
        red = HealthCheck(
            "Gmail", CheckKind.CREDENTIAL, HealthState.RED, "revoked", subject="user@gmail.com"
        )
        outcome = repair.run_repairs(red, repair.registered_repairers())
        assert refreshed == ["user@gmail.com"]
        assert outcome is not None and outcome.final and not outcome.ok
    finally:
        providers.clear_mail_providers()
        health_service.clear_check_providers()
        credentials.clear_credential_checks()
        repair.clear_repairers()


def test_a_live_refresh_probes_the_plugins_gmail_rows(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Regression (revoked-Gmail simulation, 2026-09-19): the plugin's provider read the
    env flag, so refresh(net_probe=True) never probed Gmail and a revoked token stayed
    green through the watch's diagnosis and GET /health/connectors?live=true."""
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    from iris_harness.kernel.governance.vault.credentials import CredentialRevokedError
    from iris_harness.services.health import credentials as health_credentials
    from iris_harness.services.health import service as health_service
    from iris_personal.connections import google
    from iris_personal.plugins.gmail import gmail_oauth

    monkeypatch.delenv("IRIS_HEALTH_NET_PROBE", raising=False)
    health_credentials.clear_credential_checks()
    google.reset_probe_cache()
    account = SimpleNamespace(
        email_account=SimpleNamespace(address="a@b.com"),
        has_keychain_token=True,
        token_expiry=datetime.now(UTC) + timedelta(days=30),
        refresh_token_present=True,
    )
    monkeypatch.setattr(gmail_oauth, "status", lambda: [account])

    def revoked(addr: str) -> object:
        raise CredentialRevokedError("Gmail", addr, f"iris auth gmail login --user {addr}")

    monkeypatch.setattr(gmail_oauth, "load_credentials", revoked)
    monkeypatch.setattr(health_credentials, "audit_key_checks", lambda: [])
    # Only the credential rows: the refresh's probe choice must reach the plugin's check.
    monkeypatch.setattr(
        health_service,
        "build_snapshot",
        lambda **kw: health_service.HealthSnapshot(
            tuple(health_credentials.credential_checks(net_probe=kw["net_probe"])), "t"
        ),
    )
    try:
        _mount(tmp_path)
        local = health_service.refresh(net_probe=False)
        live = health_service.refresh(net_probe=True)
        after = health_service.refresh(net_probe=False)  # the cached verdict holds
    finally:
        health_credentials.clear_credential_checks()
        google.reset_probe_cache()

    def gmail(snap):  # type: ignore[no-untyped-def]
        return next(c for c in snap.checks if c.target == "Gmail").state.value

    assert (gmail(local), gmail(live), gmail(after)) == ("green", "red", "red")
