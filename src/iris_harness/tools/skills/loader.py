"""Manifest parsing and tool loading helpers for code-first skills."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import re
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml
from langchain_core.tools import BaseTool

from .models import SkillManifest, SkillPackage, SkillProposal

SKILLS_ROOT = Path("config/skills")
AUTO_SKILLS_DIR = "auto"
MANIFEST_FILE = "manifest.yaml"
TOOLS_FILE = "tools.py"
AGENT_CONTEXT_FILE = "agent.md"
PROPOSAL_FILE = "proposal.yaml"
RUNS_FILE = "runs.jsonl"
DEFAULT_SANDBOX_SCRIPT_NAME = "sandbox_script.py"
PACKAGE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+")


def resolve_skills_root(repo_root: Path) -> Path:
    """Resolve the configured skills root under the repository."""
    return (repo_root / SKILLS_ROOT).resolve()


def resolve_auto_skills_root(repo_root: Path) -> Path:
    """Resolve the quarantine root for proposed or generated skills."""
    return resolve_skills_root(repo_root) / AUTO_SKILLS_DIR


def discover_skill_dirs(repo_root: Path) -> tuple[Path, ...]:
    """Discover all skill package directories containing a manifest."""
    skills_root = resolve_skills_root(repo_root)
    auto_root = resolve_auto_skills_root(repo_root)
    if not skills_root.exists():
        return ()
    return tuple(
        sorted(
            manifest_path.parent.resolve()
            for manifest_path in skills_root.rglob(MANIFEST_FILE)
            if auto_root.resolve() not in manifest_path.resolve().parents
        )
    )


def load_skill_manifest(skill_dir: Path) -> SkillManifest:
    """Load and validate a skill manifest from disk."""
    manifest_path = skill_dir / MANIFEST_FILE
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"skill manifest must decode to a mapping: {manifest_path}")
    return SkillManifest.model_validate(payload)


def validate_skill_prerequisites(
    repo_root: Path,
    manifest: SkillManifest,
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return missing skill prerequisites without raising for runtime discovery."""
    missing: list[str] = []
    if not _python_requirement_satisfied(manifest.requires.python):
        missing.append(f"python:{manifest.requires.python}")

    for package_spec in manifest.requires.packages:
        package_name = _extract_package_name(package_spec)
        try:
            importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(f"package:{package_name}")

    source = dict(environment or {})
    for env_var in manifest.requires.env_vars:
        if not source.get(env_var):
            missing.append(f"env:{env_var}")

    for config_file in manifest.requires.config_files:
        if not _resolve_config_path(repo_root, config_file).exists():
            missing.append(f"config:{config_file}")

    for credential in manifest.requires.required_credentials:
        if not _vault_handle_available(credential.handle):
            missing.append(f"credential:{credential.handle}")

    return tuple(missing)


def _vault_handle_available(handle: str) -> bool:
    """Return True iff the vault has a secret for ``handle``.

    Vault import is lazy so the skills package stays importable on hosts
    that don't have the cryptography stack installed. Any failure
    (vault offline, missing master key) is treated as "credential
    missing" — the skill registers as not-loadable, which is the
    behavior we want for an operator who hasn't run `iris vault add`
    yet.
    """
    try:
        from iris_harness.kernel.governance.vault import resolve_secret_value
    except Exception:  # noqa: BLE001
        return False
    try:
        return resolve_secret_value(handle) is not None
    except Exception:  # noqa: BLE001
        return False


def load_skill_package(
    repo_root: Path,
    skill_dir: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> SkillPackage:
    """Load one skill package, preserving prerequisite failures for registry reporting."""
    manifest = load_skill_manifest(skill_dir)
    missing_prerequisites = validate_skill_prerequisites(
        repo_root,
        manifest,
        environment=environment,
    )
    tool_classes: tuple[type[BaseTool], ...] = ()
    if not missing_prerequisites and manifest.kind != "brief":
        tool_classes = load_skill_tool_classes(skill_dir)
    return SkillPackage(
        manifest=manifest,
        skill_dir=skill_dir.resolve(),
        tools_module_path=(skill_dir / TOOLS_FILE).resolve(),
        tool_classes=tool_classes,
        agent_context=load_skill_agent_context(skill_dir),
        missing_prerequisites=missing_prerequisites,
    )


def load_skill_tool_classes(skill_dir: Path) -> tuple[type[BaseTool], ...]:
    """Import the skill's tools module and return the declared tool classes."""
    tools_path = skill_dir / TOOLS_FILE
    module = _load_module_from_path(
        module_name=_build_module_name(skill_dir, suffix="tools"),
        file_path=tools_path,
    )
    tool_classes = getattr(module, "SKILL_TOOLS", None)
    if not isinstance(tool_classes, (list, tuple)) or not tool_classes:
        raise ValueError(
            f"skill tools module must expose a non-empty SKILL_TOOLS list: {tools_path}"
        )
    normalized = tuple(tool_classes)
    if not all(
        isinstance(tool_class, type) and issubclass(tool_class, BaseTool)
        for tool_class in normalized
    ):
        raise ValueError(f"SKILL_TOOLS must contain BaseTool subclasses: {tools_path}")
    return normalized


def load_skill_agent_context(skill_dir: Path) -> str | None:
    """Load optional prompt-context text for a skill package."""
    context_path = skill_dir / AGENT_CONTEXT_FILE
    if not context_path.exists():
        return None
    return context_path.read_text(encoding="utf-8").strip() or None


def scaffold_skill_proposal(
    repo_root: Path,
    proposal: SkillProposal,
    *,
    sandbox_script: str | None = None,
    overwrite: bool = False,
    manifest_overrides: dict[str, Any] | None = None,
) -> tuple[Path, bool]:
    """Create a quarantined skill proposal package under config/skills/auto/.

    When ``proposal.source_kind == "sandbox"``, ``sandbox_script`` (the verbatim
    script body) and ``proposal.sandbox_script_path`` (its filename inside the
    proposal_dir) are required; the sandbox script and an empty ``runs.jsonl``
    are written instead of the placeholder ``tools.py`` / ``__init__.py``.
    """
    proposal_dir = repo_root / proposal.proposal_dir
    manifest_path = repo_root / proposal.manifest_path
    created = overwrite or not manifest_path.exists()
    if manifest_path.exists() and not overwrite:
        return proposal_dir, False

    if proposal.source_kind == "sandbox":
        if sandbox_script is None:
            raise ValueError("sandbox_script is required when source_kind='sandbox'")
        if not proposal.sandbox_script_path:
            raise ValueError("proposal.sandbox_script_path is required when source_kind='sandbox'")

    proposal_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "name": proposal.skill_slug,
        "version": "0.1.0",
        "description": f"Proposed from coding task {proposal.task_id}: {proposal.source_description}",
        "author": "iris-coding-agent",
        "license": "Apache-2.0",
        "tools": [
            {
                "name": f"{proposal.skill_slug}_tool",
                "description": f"TODO: implement the capability proposed from task {proposal.task_id}",
                "governor_route": "system/read",
            }
        ],
        "requires": {
            "python": ">=3.12",
            "packages": ["pydantic>=2.0"],
            "env_vars": [],
            "config_files": [],
            "agents": [],
        },
    }
    if manifest_overrides:
        # Agentic synthesis (crystallizer) supplies real description / tools /
        # when_to_use, replacing the TODO defaults above.
        manifest_payload.update(manifest_overrides)
    manifest_path.write_text(
        yaml.safe_dump(manifest_payload, sort_keys=False),
        encoding="utf-8",
    )
    (proposal_dir / AGENT_CONTEXT_FILE).write_text(
        _render_skill_proposal_context(proposal),
        encoding="utf-8",
    )
    save_skill_proposal(repo_root, proposal)

    if proposal.source_kind == "sandbox":
        # sandbox_script and sandbox_script_path were validated above.
        assert sandbox_script is not None
        assert proposal.sandbox_script_path is not None
        (proposal_dir / proposal.sandbox_script_path).write_text(sandbox_script, encoding="utf-8")
        runs_path = proposal_dir / RUNS_FILE
        if not runs_path.exists():
            runs_path.write_text("", encoding="utf-8")
    else:
        (proposal_dir / TOOLS_FILE).write_text(
            _render_skill_proposal_tools(proposal),
            encoding="utf-8",
        )
        (proposal_dir / "__init__.py").write_text(
            "from .tools import SKILL_TOOLS\n",
            encoding="utf-8",
        )
    return proposal_dir, created


def save_skill_proposal(repo_root: Path, proposal: SkillProposal) -> Path:
    """Persist proposal lifecycle metadata to ``proposal.yaml`` in the proposal_dir."""
    proposal_dir = repo_root / proposal.proposal_dir
    proposal_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = proposal_dir / PROPOSAL_FILE
    proposal_path.write_text(
        yaml.safe_dump(proposal.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return proposal_path


def load_skill_proposal(repo_root: Path, slug: str) -> SkillProposal:
    """Load proposal lifecycle metadata for an auto-quarantined skill by slug."""
    proposal_path = resolve_auto_skills_root(repo_root) / slug / PROPOSAL_FILE
    if not proposal_path.exists():
        raise FileNotFoundError(f"proposal metadata not found: {proposal_path}")
    payload = yaml.safe_load(proposal_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"proposal.yaml must decode to a mapping: {proposal_path}")
    return SkillProposal.model_validate(payload)


def record_sandbox_run(
    repo_root: Path,
    slug: str,
    *,
    args: Mapping[str, object] | None = None,
    exit_code: int = 0,
    duration_ms: int | None = None,
) -> SkillProposal:
    """Append a sandbox-run entry, bump run_count, and flip status to 'ready' at threshold."""
    proposal = load_skill_proposal(repo_root, slug)
    if proposal.source_kind != "sandbox":
        raise ValueError(
            f"record_sandbox_run requires source_kind='sandbox', got '{proposal.source_kind}'"
        )

    now = datetime.now(UTC)
    proposal_dir = repo_root / proposal.proposal_dir
    runs_path = proposal_dir / RUNS_FILE
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": now.isoformat(),
        "args": dict(args) if args is not None else None,
        "exit_code": exit_code,
        "duration_ms": duration_ms,
    }
    with runs_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    new_run_count = proposal.run_count + (1 if exit_code == 0 else 0)
    new_status = proposal.status
    if (
        proposal.status == "proposed"
        and exit_code == 0
        and new_run_count >= proposal.promotion_threshold
    ):
        new_status = "ready"

    updated = proposal.model_copy(
        update={
            "run_count": new_run_count,
            "last_run_at": now,
            "status": new_status,
        }
    )
    save_skill_proposal(repo_root, updated)
    return updated


def _build_module_name(skill_dir: Path, *, suffix: str) -> str:
    """Create a deterministic module name for dynamically loaded skill code."""
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", skill_dir.as_posix()).strip("_")
    return f"iris_dynamic_skill_{sanitized}_{suffix}"


def _load_module_from_path(module_name: str, file_path: Path) -> ModuleType:
    """Import a Python module directly from a file path."""
    if not file_path.exists():
        raise FileNotFoundError(f"skill module does not exist: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to build import spec for: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _extract_package_name(package_spec: str) -> str:
    """Extract the importlib metadata package name from a versioned requirement string."""
    match = PACKAGE_NAME_PATTERN.match(package_spec.strip())
    if match is None:
        raise ValueError(f"invalid package requirement: {package_spec}")
    return match.group(0)


def _python_requirement_satisfied(requirement: str) -> bool:
    """Evaluate a minimal version constraint for the active interpreter."""
    operator, required_version = _split_version_requirement(requirement)
    current_version = (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    if operator == ">=":
        return current_version >= required_version
    if operator == ">":
        return current_version > required_version
    if operator == "<=":
        return current_version <= required_version
    if operator == "<":
        return current_version < required_version
    if operator == "==":
        return current_version == required_version
    raise ValueError(f"unsupported python requirement operator: {operator}")


def _split_version_requirement(requirement: str) -> tuple[str, tuple[int, int, int]]:
    """Split a simple version requirement into operator and numeric tuple."""
    for operator in (">=", "<=", "==", ">", "<"):
        if requirement.startswith(operator):
            raw_version = requirement[len(operator) :].strip()
            parts = [int(part) for part in raw_version.split(".") if part]
            while len(parts) < 3:
                parts.append(0)
            return operator, (parts[0], parts[1], parts[2])
    raise ValueError(f"unsupported python requirement: {requirement}")


_REPO_TOP_LEVEL_DIRS: frozenset[str] = frozenset(
    {"config", "data", "scripts", "src", "tests", "docs"}
)


def _resolve_config_path(repo_root: Path, config_file: str) -> Path:
    """Resolve a manifest ``config_files`` entry to a concrete filesystem path.

    Resolution rules, in order:

    1. ``~``-prefixed paths are expanded against the user's home (e.g.
       ``~/.iris/workspace/.../proposals.jsonl`` → real home path).
    2. Absolute paths are returned untouched.
    3. Relative paths whose first component is a known top-level repo
       directory (``config``, ``data``, ``scripts``, ``src``,
       ``tests``, ``docs``) are joined with ``repo_root`` directly —
       so ``data/email.db`` resolves to ``<repo>/data/email.db``, not
       ``<repo>/config/data/email.db``.
    4. All other relative paths fall back to ``<repo>/config/<path>``
       for backward compatibility with bare-filename skill prereqs
       (e.g. ``message_templates.yaml``).
    """
    expanded = Path(config_file).expanduser()
    if expanded.is_absolute():
        return expanded
    if expanded.parts and expanded.parts[0] in _REPO_TOP_LEVEL_DIRS:
        return repo_root / expanded
    return repo_root / "config" / expanded


def _render_skill_proposal_context(proposal: SkillProposal) -> str:
    """Render the agent-context markdown for a proposed skill package."""
    changed_files = (
        "\n".join(f"- {path}" for path in proposal.source_changed_files) or "- none captured"
    )
    tool_usage = "\n".join(f"- {tool}" for tool in proposal.tool_usage) or "- none captured"
    skill_usage = "\n".join(f"- {skill}" for skill in proposal.skill_usage) or "- none captured"
    reward_summary = proposal.reward_summary or "No reward summary captured yet."
    return (
        f"# Proposed Skill: {proposal.skill_name}\n\n"
        f"Derived from task `{proposal.task_id}`. This package is quarantined under `config/skills/auto/` and is not auto-registered.\n\n"
        f"## Source Task\n\n{proposal.source_description}\n\n"
        f"## Changed Files\n\n{changed_files}\n\n"
        f"## Tool Usage\n\n{tool_usage}\n\n"
        f"## Skill Usage\n\n{skill_usage}\n\n"
        f"## Reward Summary\n\n{reward_summary}\n"
    )


def _render_skill_proposal_tools(proposal: SkillProposal) -> str:
    """Render a placeholder tools module for a proposed skill package."""
    tool_name = f"{proposal.skill_slug}_tool"
    class_name = "".join(part.capitalize() for part in proposal.skill_slug.split("-")) + "Tool"
    return (
        "from langchain_core.tools import BaseTool\n"
        "from pydantic import BaseModel, Field\n\n"
        "class ProposedSkillArgs(BaseModel):\n"
        "    request: str = Field(description='Task-specific input for the proposed skill')\n\n"
        f"class {class_name}(BaseTool):\n"
        f"    name: str = '{tool_name}'\n"
        f"    description: str = 'TODO: implement the capability proposed from task {proposal.task_id}'\n"
        "    args_schema: type[BaseModel] = ProposedSkillArgs\n\n"
        "    def _run(self, request: str) -> str:\n"
        "        raise NotImplementedError('Skill proposal scaffolding only; implement before activation')\n\n"
        "    async def _arun(self, request: str) -> str:\n"
        "        raise NotImplementedError('Skill proposal scaffolding only; implement before activation')\n\n"
        f"SKILL_TOOLS = [{class_name}]\n"
    )
