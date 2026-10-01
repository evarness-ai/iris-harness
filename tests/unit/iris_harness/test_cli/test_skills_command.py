"""Tests for the `/skills` slash command."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from iris_harness.cli.commands import _cmd_skills, _list_skill_entries


def _write_skill_package(repo_root: Path, skill_name: str) -> None:
    skill_dir = repo_root / "config" / "skills" / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "manifest.yaml").write_text(
        f"name: {skill_name}\n"
        "version: 1.0.0\n"
        "description: Test skill package\n"
        "author: iris-tests\n"
        "license: Apache-2.0\n"
        "tools:\n"
        "  - name: fetch_top_repos\n"
        "    description: Fetch top GitHub repositories.\n"
        "    governor_route: system/read\n"
        "requires:\n"
        "  python: '>=3.12'\n"
        "  packages:\n"
        "    - pydantic>=2.0\n"
        "  env_vars: []\n"
        "  config_files: []\n"
        "  agents: []\n",
        encoding="utf-8",
    )
    (skill_dir / "agent.md").write_text("# Fetch Top Repos\n", encoding="utf-8")
    (skill_dir / "tools.py").write_text(
        "from langchain_core.tools import BaseTool\n"
        "from pydantic import BaseModel, Field\n\n"
        "class FetchArgs(BaseModel):\n"
        "    limit: int = Field(default=10)\n\n"
        "class FetchTool(BaseTool):\n"
        "    name: str = 'fetch_top_repos'\n"
        "    description: str = 'Fetch top repositories.'\n"
        "    args_schema: type[BaseModel] = FetchArgs\n\n"
        "    def _run(self, limit: int = 10) -> str:\n"
        "        return str(limit)\n\n"
        "    async def _arun(self, limit: int = 10) -> str:\n"
        "        return str(limit)\n\n"
        "SKILL_TOOLS = [FetchTool]\n",
        encoding="utf-8",
    )


def test_list_skill_entries_reads_runtime_config_skills(tmp_path: Path) -> None:
    _write_skill_package(tmp_path, "fetch-top-repos")
    auto_dir = tmp_path / "config" / "skills" / "auto" / "draft-skill"
    auto_dir.mkdir(parents=True)
    (auto_dir / "manifest.yaml").write_text("name: draft-skill\n", encoding="utf-8")

    entries = _list_skill_entries(tmp_path)

    assert len(entries) == 1
    assert entries[0].name == "fetch-top-repos"
    assert entries[0].source == "config/skills"
    assert entries[0].status == "loadable"
    assert entries[0].tools == "fetch_top_repos"
    assert entries[0].agents == "all"
    assert entries[0].location == "config/skills/fetch-top-repos"


def test_skills_list_command_shows_promoted_skill(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_skill_package(tmp_path, "fetch-top-repos")
    output = io.StringIO()
    monkeypatch.setattr(
        "iris_harness.cli.render.console",
        Console(file=output, force_terminal=False, width=200),
    )
    ctx = SimpleNamespace(session=SimpleNamespace(cwd=str(tmp_path)))

    assert _cmd_skills(ctx, "list") is True

    rendered = output.getvalue()
    assert "fetch-top-repos" in rendered
    assert "fetch_top_repos" in rendered
    assert "config/skills/fetch-top-repos" in rendered
