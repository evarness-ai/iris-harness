"""Tests for the workspace path migration (2026-05-19).

Identity moved from ``~/.iris/identity/soul.md`` + ``~/.iris/memory/user.md``
to a consolidated ``~/.iris/workspace/`` location. The loader must:

- Prefer the workspace copy when present.
- Fall back to the legacy path so pre-migration installs keep working.
- Auto-migrate legacy -> workspace on first ``bootstrap_identity_files``.
- Strip frontmatter from both SOUL.md and USER.md before injecting body
  text into prompts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.memory.identity import loader


@pytest.fixture()
def iris_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / ".iris"
    monkeypatch.setattr(loader, "IRIS_HOME", home)
    monkeypatch.setattr(loader, "WORKSPACE_DIR", home / "workspace")
    monkeypatch.setattr(loader, "IDENTITY_DIR", home / "identity")
    monkeypatch.setattr(loader, "MEMORY_DIR", home / "memory")
    monkeypatch.setattr(loader, "BEHAVIORS_DIR", home / "behaviors")
    monkeypatch.setattr(loader, "SOUL_PATH", home / "workspace" / "SOUL.md")
    monkeypatch.setattr(loader, "USER_MD_PATH", home / "workspace" / "USER.md")
    monkeypatch.setattr(loader, "_LEGACY_SOUL_PATH", home / "identity" / "soul.md")
    monkeypatch.setattr(loader, "_LEGACY_USER_MD_PATH", home / "memory" / "user.md")
    monkeypatch.setattr(loader, "ACTIVE_MD_PATH", home / "memory" / "active.md")
    monkeypatch.setattr(loader, "EPISODIC_MD_PATH", home / "memory" / "episodic.md")
    return home


def test_load_soul_prefers_workspace_copy(iris_home: Path) -> None:
    workspace = iris_home / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "SOUL.md").write_text(
        "---\nname: IRIS\nclassification: internal\n---\n\nworkspace body\n",
        encoding="utf-8",
    )

    assert loader.load_soul() == "workspace body"


def test_load_soul_falls_back_to_legacy_path(iris_home: Path) -> None:
    legacy = iris_home / "identity"
    legacy.mkdir(parents=True)
    (legacy / "soul.md").write_text(
        "---\nname: IRIS\n---\n\nlegacy body\n",
        encoding="utf-8",
    )
    # Workspace dir doesn't exist yet — loader must still find legacy.
    assert loader.load_soul() == "legacy body"


def test_load_soul_returns_none_when_neither_exists(iris_home: Path) -> None:
    assert loader.load_soul() is None


def test_load_user_md_prefers_workspace_and_strips_frontmatter(
    iris_home: Path,
) -> None:
    workspace = iris_home / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "USER.md").write_text(
        "---\nclassification: personal\negress: local-only\n---\n\n"
        "# User Profile\n\nname: Robin\n",
        encoding="utf-8",
    )

    body = loader.load_user_md()
    assert body is not None
    assert body.startswith("# User Profile")
    assert "classification:" not in body


def test_load_user_md_falls_back_to_legacy(iris_home: Path) -> None:
    legacy = iris_home / "memory"
    legacy.mkdir(parents=True)
    (legacy / "user.md").write_text("# Legacy profile\n", encoding="utf-8")

    assert loader.load_user_md() == "# Legacy profile"


def test_bootstrap_migrates_legacy_soul_into_workspace(iris_home: Path) -> None:
    legacy = iris_home / "identity"
    legacy.mkdir(parents=True)
    (legacy / "soul.md").write_text("---\nname: IRIS\n---\n\nlegacy soul body\n", encoding="utf-8")

    loader.bootstrap_identity_files()

    workspace_soul = iris_home / "workspace" / "SOUL.md"
    assert workspace_soul.exists()
    assert "legacy soul body" in workspace_soul.read_text(encoding="utf-8")


def test_bootstrap_migrates_legacy_user_md_into_workspace(iris_home: Path) -> None:
    legacy = iris_home / "memory"
    legacy.mkdir(parents=True)
    (legacy / "user.md").write_text(
        "# Legacy user\n\n## Auto-detected\n- **name**: Robin  <!-- auto: confidence=0.95 -->\n",
        encoding="utf-8",
    )

    loader.bootstrap_identity_files()

    workspace_user = iris_home / "workspace" / "USER.md"
    assert workspace_user.exists()
    text = workspace_user.read_text(encoding="utf-8")
    assert "# Legacy user" in text
    assert "name**: Robin" in text


def test_bootstrap_does_not_clobber_existing_workspace_files(iris_home: Path) -> None:
    workspace = iris_home / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "SOUL.md").write_text("workspace authored\n", encoding="utf-8")

    legacy = iris_home / "identity"
    legacy.mkdir(parents=True)
    (legacy / "soul.md").write_text("legacy authored\n", encoding="utf-8")

    loader.bootstrap_identity_files()

    assert (workspace / "SOUL.md").read_text(encoding="utf-8") == "workspace authored\n"


def test_append_user_fact_to_md_writes_workspace_path_with_frontmatter(
    iris_home: Path,
) -> None:
    """First-time fact append on a clean install must seed the workspace
    file with the classification frontmatter, not the legacy plain header."""

    written = loader.append_user_fact_to_md("name", "Robin", 0.95)

    assert written is True
    workspace_user = iris_home / "workspace" / "USER.md"
    assert workspace_user.exists()
    text = workspace_user.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "classification: personal" in text
    assert "name**: Robin" in text


# ---------------------------------------------------------------------------
# AGENTS.md + iris-harness.md loaders (on-demand reads)
# ---------------------------------------------------------------------------


def test_load_agents_md_strips_frontmatter(
    iris_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = iris_home / "workspace"
    workspace.mkdir(parents=True)
    agents_path = workspace / "AGENTS.md"
    agents_path.write_text(
        "---\nclassification: internal\nload: on-demand\n---\n\n"
        "# IRIS Agent Registry\n\n## Chat agent\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "AGENTS_MD_PATH", agents_path)

    body = loader.load_agents_md()
    assert body is not None
    assert body.startswith("# IRIS Agent Registry")
    assert "classification:" not in body


def test_load_agents_md_returns_none_when_missing(
    iris_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loader, "AGENTS_MD_PATH", iris_home / "workspace" / "AGENTS.md")
    assert loader.load_agents_md() is None


def test_load_harness_md_reads_repo_doc(
    iris_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness_path = tmp_path / "iris-harness.md"
    harness_path.write_text("# IRIS Harness\n\nOperational framework body.\n", encoding="utf-8")
    monkeypatch.setattr(loader, "HARNESS_DOC_PATH", harness_path)

    body = loader.load_harness_md()
    assert body is not None
    assert "# IRIS Harness" in body
    assert "Operational framework body." in body


def test_load_harness_md_returns_none_when_missing(
    iris_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loader, "HARNESS_DOC_PATH", tmp_path / "missing.md")
    assert loader.load_harness_md() is None


# ---------------------------------------------------------------------------
# iris_doc tool — exposed to the ReAct loop for on-demand doc reads
# ---------------------------------------------------------------------------


def _iris_doc_spec():
    from iris_harness.runtime.react_tools import builtin_react_tools

    specs = builtin_react_tools(semantic_index=None, wiki=None, repo_root=None)
    matches = [spec for spec in specs if getattr(spec, "name", None) == "iris_doc"]
    assert len(matches) == 1, f"expected exactly one iris_doc tool, got {len(matches)}"
    return matches[0]


def test_iris_doc_tool_loads_workspace_doc_by_name(
    iris_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tool dispatch resolves the name to the right loader and returns body."""

    workspace = iris_home / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "AGENTS.md").write_text(
        "---\nclassification: internal\n---\n\n# Agents\n\nentry body\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "AGENTS_MD_PATH", workspace / "AGENTS.md")

    spec = _iris_doc_spec()
    result = spec.call({"name": "AGENTS"})

    assert "# Agents" in result
    assert "entry body" in result
    assert "classification:" not in result


def test_iris_doc_tool_returns_error_on_unknown_name() -> None:
    spec = _iris_doc_spec()
    result = spec.call({"name": "BOGUS"})
    assert result.startswith("Error: iris_doc unknown name")
    assert "AGENTS" in result and "HARNESS" in result  # error lists valid names


def test_iris_doc_tool_returns_error_when_name_missing() -> None:
    spec = _iris_doc_spec()
    result = spec.call({})
    assert result.startswith("Error: iris_doc requires a 'name'")


def test_iris_doc_tool_handles_missing_file_gracefully(
    iris_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loader, "AGENTS_MD_PATH", iris_home / "workspace" / "AGENTS.md")
    spec = _iris_doc_spec()
    result = spec.call({"name": "AGENTS"})
    assert "file not present" in result


def test_iris_doc_tool_normalises_case(iris_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Names are case-insensitive — agents, AGENTS, Agents all work."""

    workspace = iris_home / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "AGENTS.md").write_text("body\n", encoding="utf-8")
    monkeypatch.setattr(loader, "AGENTS_MD_PATH", workspace / "AGENTS.md")

    spec = _iris_doc_spec()
    assert "body" in spec.call({"name": "agents"})
    assert "body" in spec.call({"name": "Agents"})
