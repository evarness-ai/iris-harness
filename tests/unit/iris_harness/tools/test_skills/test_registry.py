"""Unit tests for the minimal code-first skills subsystem."""

from __future__ import annotations

from pathlib import Path

from iris_harness.tools.skills.loader import (
    discover_skill_dirs,
    load_skill_manifest,
    load_skill_package,
    resolve_auto_skills_root,
    resolve_skills_root,
    scaffold_skill_proposal,
    validate_skill_prerequisites,
)
from iris_harness.tools.skills.models import SkillProposal
from iris_harness.tools.skills.registry import SkillRegistry


def write_skill_package(
    repo_root: Path,
    *,
    skill_name: str,
    agents: tuple[str, ...] = ("email",),
    env_vars: tuple[str, ...] = (),
    packages: tuple[str, ...] = ("pydantic>=2.0",),
) -> Path:
    skill_dir = repo_root / "config" / "skills" / skill_name
    skill_dir.mkdir(parents=True)
    agents_yaml = "\n".join(f"    - {agent}" for agent in agents)
    env_yaml = "\n".join(f"    - {env_var}" for env_var in env_vars) or "    []"
    packages_yaml = "\n".join(f"    - {package}" for package in packages)
    (skill_dir / "manifest.yaml").write_text(
        f"name: {skill_name}\n"
        "version: 1.0.0\n"
        "description: Test skill package\n"
        "author: iris-tests\n"
        "license: Apache-2.0\n"
        "tools:\n"
        f"  - name: {skill_name}_tool\n"
        "    description: Test tool\n"
        "    governor_route: system/read\n"
        "requires:\n"
        "  python: '>=3.12'\n"
        "  packages:\n"
        f"{packages_yaml}\n"
        "  env_vars:\n"
        f"{env_yaml}\n"
        "  config_files:\n"
        "    - message_templates.yaml\n"
        "  agents:\n"
        f"{agents_yaml}\n",
        encoding="utf-8",
    )
    (skill_dir / "agent.md").write_text(
        f"# {skill_name}\n\nUse the test skill when needed.\n",
        encoding="utf-8",
    )
    (skill_dir / "tools.py").write_text(
        "from langchain_core.tools import BaseTool\n"
        "from pydantic import BaseModel, Field\n\n"
        "class EchoArgs(BaseModel):\n"
        "    message: str = Field(description='Message to echo')\n\n"
        "class EchoTool(BaseTool):\n"
        f"    name: str = '{skill_name}_tool'\n"
        "    description: str = 'Echo test tool'\n"
        "    args_schema: type[BaseModel] = EchoArgs\n\n"
        "    def _run(self, message: str) -> str:\n"
        "        return message\n\n"
        "    async def _arun(self, message: str) -> str:\n"
        "        return message\n\n"
        "SKILL_TOOLS = [EchoTool]\n",
        encoding="utf-8",
    )
    config_dir = repo_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "message_templates.yaml").write_text("templates: {}\n", encoding="utf-8")
    return skill_dir


def test_resolve_and_discover_skill_directories(tmp_path: Path) -> None:
    write_skill_package(tmp_path, skill_name="email_triage")
    auto_root = resolve_auto_skills_root(tmp_path)
    auto_skill_dir = auto_root / "quarantined_skill"
    auto_skill_dir.mkdir(parents=True)
    (auto_skill_dir / "manifest.yaml").write_text(
        "name: quarantined_skill\n"
        "version: 0.1.0\n"
        "description: quarantined\n"
        "author: iris\n"
        "license: Apache-2.0\n"
        "tools: []\n"
        "requires: {}\n",
        encoding="utf-8",
    )

    skills_root = resolve_skills_root(tmp_path)
    skill_dirs = discover_skill_dirs(tmp_path)

    assert skills_root == (tmp_path / "config" / "skills").resolve()
    assert skill_dirs == ((tmp_path / "config" / "skills" / "email_triage").resolve(),)


def test_load_skill_package_reads_manifest_tools_and_agent_context(tmp_path: Path) -> None:
    skill_dir = write_skill_package(tmp_path, skill_name="email_triage")

    manifest = load_skill_manifest(skill_dir)
    package = load_skill_package(tmp_path, skill_dir)

    assert manifest.name == "email_triage"
    assert package.is_loadable is True
    assert package.manifest.tools[0].name == "email_triage_tool"
    assert len(package.tool_classes) == 1
    assert package.agent_context == "# email_triage\n\nUse the test skill when needed."


def test_validate_skill_prerequisites_reports_missing_env_vars(tmp_path: Path) -> None:
    skill_dir = write_skill_package(
        tmp_path,
        skill_name="finance_monitor",
        env_vars=("FINANCE_SKILL_ENABLED",),
    )
    manifest = load_skill_manifest(skill_dir)

    missing = validate_skill_prerequisites(tmp_path, manifest, environment={})

    assert missing == ("env:FINANCE_SKILL_ENABLED",)


def test_skill_registry_filters_packages_and_tools_by_agent(tmp_path: Path) -> None:
    write_skill_package(tmp_path, skill_name="email_triage", agents=("email",))
    write_skill_package(tmp_path, skill_name="finance_monitor", agents=("finance",))

    registry = SkillRegistry(tmp_path)
    packages = registry.discover()

    assert len(packages) == 2
    assert {
        package.manifest.name
        for package in registry.list_packages(agent_name="email", only_loadable=True)
    } == {"email_triage"}
    assert set(registry.list_tool_classes(agent_name="finance")) == {"finance_monitor_tool"}


def test_skill_registry_keeps_unloadable_packages_out_of_tool_index(tmp_path: Path) -> None:
    write_skill_package(
        tmp_path,
        skill_name="community_stock_tracker",
        packages=("package-that-does-not-exist>=1.0",),
    )

    registry = SkillRegistry(tmp_path)
    packages = registry.discover()

    assert packages[0].is_loadable is False
    assert packages[0].missing_prerequisites == ("package:package-that-does-not-exist",)
    assert registry.list_tool_classes() == {}


def test_scaffold_skill_proposal_creates_quarantined_package(tmp_path: Path) -> None:
    proposal = SkillProposal(
        proposal_id="proposal-task-1-demo-skill",
        task_id="task-1",
        skill_name="demo-skill",
        skill_slug="demo-skill",
        scope="platform",
        source_description="Create a demo skill proposal",
        source_changed_files=("src/iris_code/cli.py",),
        tool_usage=("read_file", "edit_file"),
        skill_usage=(),
        reward_summary="bootstrap reward signals captured",
        proposal_dir="config/skills/auto/demo-skill",
        manifest_path="config/skills/auto/demo-skill/manifest.yaml",
    )

    proposal_dir, created = scaffold_skill_proposal(tmp_path, proposal)

    assert created is True
    assert proposal_dir == tmp_path / "config" / "skills" / "auto" / "demo-skill"
    assert (proposal_dir / "manifest.yaml").exists()
    assert (proposal_dir / "agent.md").exists()
    assert (proposal_dir / "tools.py").exists()
    assert discover_skill_dirs(tmp_path) == ()


# No automatic discovery should occur until the proposal is explicitly approved.


def test_discover_isolates_a_skill_whose_import_fails(tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
    """One undeclared-dependency skill is skipped; the skills after it still load (#110)."""
    write_skill_package(tmp_path, skill_name="a_broken")
    write_skill_package(tmp_path, skill_name="z_fine")
    (tmp_path / "config" / "skills" / "a_broken" / "tools.py").write_text(
        "import no_such_optional_dependency_xyz\n", encoding="utf-8"
    )

    registry = SkillRegistry(tmp_path)
    with caplog.at_level("DEBUG"):
        packages = registry.discover()
        registry.discover()  # per-turn re-discovery must not repeat the warning

    assert [p.manifest.name for p in packages] == ["z_fine"]
    ((failed_dir, reason),) = registry.load_failures.items()
    assert failed_dir.name == "a_broken"
    assert reason == "ModuleNotFoundError: No module named 'no_such_optional_dependency_xyz'"
    skipped = [r for r in caplog.records if "failed to load" in r.getMessage()]
    assert len(skipped) == 1
    # A genuine fault keeps its traceback (#110 follow-up): the one-line reason is for the
    # `iris skills list` status, the log carries the stack an operator needs to fix it.
    assert skipped[0].levelname == "ERROR"
    assert skipped[0].exc_info is not None


def test_missing_declared_package_blocks_the_skill_without_importing_it(tmp_path: Path) -> None:
    skill_dir = write_skill_package(
        tmp_path, skill_name="needs_extra", packages=("no-such-dist-xyz>=1",)
    )
    (skill_dir / "tools.py").write_text("raise RuntimeError('imported')\n", encoding="utf-8")

    registry = SkillRegistry(tmp_path)
    (package,) = registry.discover()

    assert package.is_loadable is False
    assert package.missing_prerequisites == ("package:no-such-dist-xyz",)
    assert registry.load_failures == {}


def test_shipped_gmail_inbox_declares_its_google_dependency() -> None:
    manifest = load_skill_manifest(
        Path(__file__).resolve().parents[5] / "config" / "skills" / "email" / "gmail-inbox"
    )
    # Every distribution its import chain (gmail_fetch -> gmail_oauth) needs, not one of them.
    assert set(manifest.requires.packages) >= {
        "google-api-python-client",
        "google-auth",
        "google-auth-oauthlib",
    }
    assert manifest.requires.extra == "email"


def test_shipped_skills_on_a_core_only_install(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """No Google client installed: gmail-inbox is blocked, nothing fails, the rest load (#110)."""
    import importlib.metadata as metadata
    import sys

    real_version = metadata.version

    def fake_version(name: str) -> str:
        if name == "google-api-python-client":
            raise metadata.PackageNotFoundError(name)
        return real_version(name)

    monkeypatch.setattr(metadata, "version", fake_version)
    monkeypatch.setitem(sys.modules, "googleapiclient", None)  # import raises ImportError

    registry = SkillRegistry(Path(__file__).resolve().parents[5])
    by_name = {p.manifest.name: p for p in registry.discover()}

    assert registry.load_failures == {}
    assert by_name["gmail-inbox"].missing_prerequisites == ("package:google-api-python-client",)
    assert by_name["system-status"].is_loadable
