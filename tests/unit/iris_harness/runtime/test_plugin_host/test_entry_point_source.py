"""An entry-point plugin carries its real manifest, and its CLI module resolves.

Both matter from M6.1b on, when the six domain plugins stop being builtins and arrive
as `iris_harness.plugins` entry points (OSS plan M6, decisions 2 and 10). Neither
failure is loud: a synthesized manifest loads fine and simply registers less — no
`cli:` commands, no `provides` — and a CLI module resolved against the wrong package
raises inside a fault boundary that logs and moves on. So they are pinned here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import typer

from iris_harness.runtime.plugin_host.loader import _entry_point_source
from iris_harness.sdk import PluginCLI
from iris_harness.sdk.cli import _import_cli


@dataclass
class _FakeEntryPoint:
    name: str
    value: str
    dist: object | None = None


@pytest.fixture
def nested_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A plugin package nested two levels deep, like iris_personal.plugins.<name>."""
    root = tmp_path / "acme_domain" / "plugins" / "widgets"
    root.mkdir(parents=True)
    for pkg in (tmp_path / "acme_domain", tmp_path / "acme_domain" / "plugins", root):
        (pkg / "__init__.py").write_text("")
    (root / "manifest.yaml").write_text(
        "name: widgets\n"
        "version: 9.9.9\n"
        "description: a nested plugin with a real manifest\n"
        "entrypoint: plugin:setup\n"
        "flavor: python\n"
        "trust: in-process\n"
        "cli: cli:register\n"
        "provides:\n"
        "  - intercept\n"
    )
    (root / "plugin.py").write_text("def setup(api):\n    return None\n")
    (root / "cli.py").write_text(
        "def register(cli):\n    cli.group('widgets', help='nested').command('ping')(lambda: None)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "acme_domain.plugins.widgets"
    for mod in [m for m in sys.modules if m.startswith("acme_domain")]:
        del sys.modules[mod]


def _source(monkeypatch: pytest.MonkeyPatch, ep: _FakeEntryPoint):
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group=None: [ep])
    return _entry_point_source(ep.name)


def test_the_manifest_beside_the_module_wins_over_a_synthetic_one(
    nested_plugin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _source(monkeypatch, _FakeEntryPoint("widgets", f"{nested_plugin}.plugin:setup"))

    assert src is not None
    assert src.kind == "entry_point"
    assert src.manifest.version == "9.9.9"  # the file's, not the distribution's
    assert src.manifest.cli == "cli:register"  # the synthetic manifest has no cli at all
    assert "intercept" in src.manifest.provides


def test_a_plugin_without_a_manifest_still_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The minimum contract for a third-party plugin: a name and an entry point."""
    pkg = tmp_path / "bare_plugin"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("def setup(api):\n    return None\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    src = _source(monkeypatch, _FakeEntryPoint("bare", "bare_plugin:setup"))

    assert src is not None
    assert src.manifest.entrypoint == "bare_plugin:setup"
    assert src.manifest.cli is None


def test_the_cli_module_resolves_inside_the_nested_package(
    nested_plugin: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old rule took the TOP-LEVEL package: acme_domain.cli, which does not exist."""
    src = _source(monkeypatch, _FakeEntryPoint("widgets", f"{nested_plugin}.plugin:setup"))
    assert src is not None

    register = _import_cli(src)

    assert callable(register)
    root = typer.Typer()
    register(PluginCLI(root=root))
    assert any(g.name == "widgets" for g in root.registered_groups)
