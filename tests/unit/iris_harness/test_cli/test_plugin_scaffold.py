"""``iris plugin new``: the scaffold's output passes its own tests (OSS plan L2 exit).

For every kind, a plugin is generated into a temporary directory and treated as a third
party's project: its manifest is validated against the schema, its tree is held to the
stable tier, its lint config is applied to it, and its own test suite runs -- in a
subprocess, from the plugin's directory, on this interpreter, with the IRIS code under
test first on the path and no ``IRIS_*`` setting or real home directory to lean on.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib
from typer.testing import CliRunner

import iris_harness
from iris_harness.cli.plugin_scaffold import (
    KINDS,
    PLUGIN_KINDS,
    TEMPLATE_ROOT,
    ScaffoldError,
    new_plugin,
    plugin_names,
)
from iris_harness.runtime.plugin_host.manifest import (
    PluginManifest,
    RegistrationKind,
    load_manifest,
)
from iris_harness.testing import check_stable_imports, stable_tier

# The source tree the running tests import: the generated plugin must be tested against it.
_SRC = Path(iris_harness.__file__).resolve().parents[1]


def _generated_env(home: Path) -> dict[str, str]:
    """A plugin author's environment: no IRIS settings, a home of its own."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("IRIS_", "PYTEST_"))}
    env["HOME"] = str(home)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_SRC), env.get("PYTHONPATH")]))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run(args: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - this interpreter, fixed arguments
        [sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=600
    )


@pytest.mark.parametrize("kind", KINDS)
def test_the_generated_plugin_passes_its_own_tests(kind: str, tmp_path: Path) -> None:
    if kind == "mail-provider":
        # Its plugin depends on the email slice (iris-harness[email]), which a core-only
        # tree does not have.
        pytest.importorskip("iris_personal.email.provider_api")
    result = new_plugin(f"demo-{kind}", kind, parent=tmp_path)
    root = result.root

    # Every placeholder was filled, in contents and in paths.
    for path in root.rglob("*"):
        assert "__tmpl_" not in str(path.relative_to(root)), path
        if path.is_file():
            assert "__tmpl_" not in path.read_text(encoding="utf-8"), path

    # A third party's plugin may import only the stable tier.
    assert check_stable_imports([root]) == []

    # Installable: the entry point IRIS discovers, pointing at setup(); the manifest ships.
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    entry_points = project["project"]["entry-points"][stable_tier().entry_point_group]
    expected = PLUGIN_KINDS[kind].entry_point.format(package=result.names.package)
    assert entry_points == {result.names.name: expected}
    if kind == "skill":  # declarative: the entry point is the package, there is no setup()
        assert expected == result.names.package
        assert result.manifest.flavor == "declarative"
        assert not (result.package_dir / "plugin.py").exists()
    assert project["tool"]["setuptools"]["package-data"][result.names.package] == ["*.yaml"]

    home = tmp_path / "home"
    home.mkdir()
    env = _generated_env(home)
    for lint in (["-m", "ruff", "check", "."], ["-m", "black", "--check", "--quiet", "."]):
        linted = _run(lint, root, env)
        assert linted.returncode == 0, linted.stdout + linted.stderr

    ran = _run(["-m", "pytest", "-q", "-p", "no:cacheprovider"], root, env)
    assert ran.returncode == 0, ran.stdout[-4000:] + ran.stderr[-2000:]
    assert " passed" in ran.stdout and "failed" not in ran.stdout
    # The plugin's tests left the author's home alone.
    assert not (home / ".iris").exists()
    assert not (home / ".local" / "share" / "iris").exists()


def _declared(kind: str, tmp_path: Path) -> tuple[str, PluginManifest]:
    result = new_plugin(f"x-{kind}", kind, parent=tmp_path)
    return result.names.tool, load_manifest(result.package_dir / "manifest.yaml")


def test_each_kind_declares_what_it_registers(tmp_path: Path) -> None:
    tool, manifest = _declared("tool", tmp_path)
    assert manifest.provides == (RegistrationKind.TOOL,)
    decl = manifest.tools[tool]
    assert (decl.effect, decl.content) == ("read", "internal")

    _tool, manifest = _declared("channel", tmp_path)
    assert manifest.provides == (RegistrationKind.CHANNEL,)

    _tool, manifest = _declared("mail-provider", tmp_path)
    assert manifest.provides == ()
    assert manifest.identity.provides == ("email",)

    # A search provider joins the research tool's chain: no tool of its own, and the
    # provider's name declared (register_search_provider refuses an undeclared one).
    tool, manifest = _declared("research-provider", tmp_path)
    assert (manifest.provides, manifest.tools) == ((), {})
    assert manifest.search_providers == (tool,)

    tool, manifest = _declared("skill", tmp_path)
    assert set(manifest.tools) == {f"{tool}_convert_length"}


def test_every_template_tree_is_a_declared_kind() -> None:
    trees = {p.name for p in TEMPLATE_ROOT.iterdir() if p.is_dir() and p.name != "_common"}
    assert trees == set(KINDS)


def test_the_templates_are_held_to_the_stable_tier() -> None:
    root = Path(__file__).resolve()
    while not (root / "src" / "iris_harness").is_dir():
        root = root.parent
    assert str(TEMPLATE_ROOT.relative_to(root)) in stable_tier().enforced_roots


@pytest.mark.parametrize(
    ("name", "package", "tool", "dist"),
    [
        ("weather-now", "iris_plugin_weather_now", "weather_now", "iris-plugin-weather-now"),
        ("email", "iris_plugin_email", "email", "iris-plugin-email"),
        ("a2_b", "iris_plugin_a2_b", "a2_b", "iris-plugin-a2-b"),
    ],
)
def test_names_map_to_a_package_a_tool_and_a_distribution(
    name: str, package: str, tool: str, dist: str
) -> None:
    names = plugin_names(name)
    assert (names.package, names.tool, names.dist) == (package, tool, dist)


@pytest.mark.parametrize(
    "bad", ["", "Weather", "2fast", "-x", "x-", "a--b", "a_-b", "has space", "dot.ted", "x" * 41]
)
def test_a_bad_name_is_refused_with_the_rule(bad: str) -> None:
    with pytest.raises(ScaffoldError, match="not valid|longer than"):
        plugin_names(bad)


def test_a_keyword_is_refused() -> None:
    with pytest.raises(ScaffoldError, match="keyword"):
        plugin_names("class")


def test_an_unknown_kind_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ScaffoldError, match="unknown kind"):
        new_plugin("x", "judge", parent=tmp_path)


def test_an_existing_directory_needs_force(tmp_path: Path) -> None:
    first = new_plugin("again", "tool", parent=tmp_path)
    own = first.root / "NOTES.md"
    own.write_text("mine", encoding="utf-8")
    (first.package_dir / "plugin.py").write_text("changed", encoding="utf-8")

    with pytest.raises(ScaffoldError, match="--force"):
        new_plugin("again", "tool", parent=tmp_path)

    again = new_plugin("again", "tool", parent=tmp_path, force=True)
    assert "def setup" in (again.package_dir / "plugin.py").read_text(encoding="utf-8")
    assert own.read_text(encoding="utf-8") == "mine"  # --force leaves other files alone


@pytest.mark.parametrize("group", ["plugin", "plugins"])
def test_the_cli_writes_a_plugin(group: str, tmp_path: Path) -> None:
    from iris_harness.main import app

    runner = CliRunner()
    result = runner.invoke(
        app, [group, "new", "cli-made", "--kind", "tool", "--dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "cli-made" / "src" / "iris_plugin_cli_made" / "plugin.py").is_file()
    assert "pytest" in result.output

    again = runner.invoke(app, [group, "new", "cli-made", "--kind", "tool", "--dir", str(tmp_path)])
    assert again.exit_code == 1 and "--force" in again.output

    unknown = runner.invoke(app, [group, "new", "y", "--kind", "judge", "--dir", str(tmp_path)])
    assert unknown.exit_code == 2 and "unknown kind" in unknown.output

    bad = runner.invoke(app, [group, "new", "Bad Name", "--kind", "tool", "--dir", str(tmp_path)])
    assert bad.exit_code == 1 and "not valid" in bad.output
