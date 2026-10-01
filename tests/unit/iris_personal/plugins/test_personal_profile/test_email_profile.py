"""The `email` profile on an install that has the email plugins (OSS plan R1/R2/R6).

The core half -- the selection rule, and that ``email.yaml`` is ``default`` plus the two
email plugins -- is in ``tests/unit/iris_harness/runtime/test_plugin_host/test_profile.py``
and holds on a core-only install. What is pinned here needs the plugins installed: an
unnamed run comes up as the email assistant, with ``iris email demo`` on its CLI.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import pytest
import typer

from iris_harness.foundation.paths import default_config_dir
from iris_harness.runtime.plugin_host import discover_plugin, load_profile
from iris_harness.sdk import PluginCLI, register_plugin_commands

CONFIG = default_config_dir()


def test_an_unnamed_run_is_the_email_assistant(tmp_path: Path) -> None:
    prof = load_profile(CONFIG, home_dir=tmp_path, env={})
    assert prof.name == "email"
    assert {"gmail", "email_workflows"} <= {p.name for p in prof.enabled_plugins()}


def _shipped(name: str) -> bool:
    return (CONFIG / "profiles" / f"{name}.yaml").is_file()


def test_iris_profile_wins(tmp_path: Path) -> None:
    # `personal-assistant` mounts the private domains, so the public tree does not ship it
    # (OSS plan R1); every profile this tree ships must win when named.
    names = [n for n in ("default", "minimal", "email", "personal-assistant") if _shipped(n)]
    assert {"default", "minimal", "email"} <= set(names)
    for name in names:
        assert load_profile(CONFIG, home_dir=tmp_path, env={"IRIS_PROFILE": name}).name == name


def test_personal_assistant_is_unchanged(tmp_path: Path) -> None:
    if not _shipped("personal-assistant"):
        pytest.skip("personal-assistant is not shipped in this tree (OSS plan R1)")
    prof = load_profile(CONFIG, home_dir=tmp_path, env={"IRIS_PROFILE": "personal-assistant"})
    # imap is an optional row: it mounts exactly when its entry point is installed, and a
    # venv that has not re-run `poetry install` since it was added skips it, not fails.
    imap_installed = any(ep.name == "imap" for ep in entry_points(group="iris_harness.plugins"))
    assert [p.name for p in prof.plugins] == [
        "system",
        "gmail",
        *(["imap"] if imap_installed else []),
        "file_organizer",
        "email_workflows",
        "calendar",
        "finance_workflows",
        "planner",
        "telegram_channel",
        "web_channel",
        "web_push_channel",
        "research",
        "code_exec",
    ]
    assert not any(layer.startswith("selected:") for layer in prof.layers)


def test_the_gmail_plugin_declares_the_packages_the_email_extra_installs() -> None:
    """The signal the selection reads: without the Google API client, no `email`."""
    source = discover_plugin("gmail")
    assert source is not None
    assert set(source.manifest.requires.packages) == {
        "google-api-python-client",
        "google-auth",
        "google-auth-oauthlib",
    }


def _commands(app: typer.Typer, group: str) -> set[str]:
    for info in app.registered_groups:
        if info.name == group and info.typer_instance is not None:
            sub: Any = info.typer_instance
            return {cmd.name or "" for cmd in sub.registered_commands}
    return set()


def test_iris_email_demo_is_on_the_cli_of_an_unnamed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IRIS_PROFILE", raising=False)
    root = typer.Typer()
    added = register_plugin_commands(PluginCLI(root=root), config_dir=CONFIG, home_dir=tmp_path)
    assert {"gmail", "email_workflows"} <= set(added)
    assert "demo" in _commands(root, "email")
