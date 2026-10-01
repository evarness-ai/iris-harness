"""`iris email` never writes into the directory it runs from (the IRIS_HOME rule).

Every store default used to be a bare ``Path("data/email.db")``: from an installed
``iris`` that is a stray ``data/`` wherever the owner happens to be. Now the stores
resolve through ``sdk.persistence`` (``$IRIS_DATA_DIR``, else ``$IRIS_HOME/data``, ...)
and the workspace files through ``sdk.config.workspace_dir``. Each command here runs
with only ``IRIS_HOME`` set, from an unrelated working directory, which must stay empty.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from iris_harness.foundation.paths import default_config_dir
from iris_harness.sdk import PluginCLI, register_plugin_commands

ACCOUNT = "gmail:owner@example.com"
SRC = Path(__file__).resolve().parents[5] / "src"


@pytest.fixture
def places(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    home = tmp_path / "home"
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.delenv("IRIS_DATA_DIR", raising=False)
    monkeypatch.delenv("IRIS_PROFILE", raising=False)  # the email profile, by selection
    monkeypatch.setenv("IRIS_HOME", str(home))
    monkeypatch.chdir(cwd)
    return home, cwd


def _iris(home: Path) -> typer.Typer:
    root = typer.Typer()
    register_plugin_commands(PluginCLI(root=root), config_dir=default_config_dir(), home_dir=home)
    return root


@pytest.mark.parametrize(
    ("args", "exit_code"),
    [
        (["email", "search", "invoice"], 0),
        (["email", "repair-domains"], 0),
        (["email", "judgments"], 0),
        (["email", "corrections", "list", "--account", ACCOUNT], 3),  # 3: none captured yet
    ],
    ids=["search", "repair-domains", "judgments", "corrections-list"],
)
def test_email_commands_keep_their_stores_in_the_home(
    places: tuple[Path, Path], args: list[str], exit_code: int
) -> None:
    home, cwd = places
    result = CliRunner().invoke(_iris(home), args)
    assert result.exit_code == exit_code, result.output
    assert list(cwd.iterdir()) == []
    assert any((home / "data").iterdir())


def test_the_workspace_follows_the_home(places: tuple[Path, Path]) -> None:
    home, cwd = places
    from iris_personal.plugins.email_workflows.knn_gate import KnnGateRunner
    from iris_personal.plugins.email_workflows.triage import EmailTriageClassifier
    from iris_personal.plugins.gmail.gmail_oauth import default_client_secrets_path

    for cls in (EmailTriageClassifier, KnnGateRunner):
        defaults = {f.name: f.default_factory for f in dataclasses.fields(cls)}
        assert defaults["workspace_dir"]() == home / "workspace"  # type: ignore[misc]
        assert defaults["email_db_path"]() == home / "data" / "email.db"  # type: ignore[misc]
    assert default_client_secrets_path().is_relative_to(home / "workspace")
    assert list(cwd.iterdir()) == []


def test_iris_email_demo_writes_only_its_own_home(
    places: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    home, cwd = places
    # The demo's child process imports this checkout's code, not whatever else is on
    # the machine.
    monkeypatch.setenv("PYTHONPATH", str(SRC))
    demo = home.parent / "demo"
    result = CliRunner().invoke(_iris(home), ["email", "demo", "--home", str(demo)])
    assert result.exit_code == 0, result.output
    assert list(cwd.iterdir()) == []
    assert (demo / "data" / "email.db").is_file()
