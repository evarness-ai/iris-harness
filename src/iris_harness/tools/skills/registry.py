"""Registry for loading and filtering code-first skill packages."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from langchain_core.tools import BaseTool

from .loader import discover_skill_dirs, load_skill_package
from .models import SkillPackage


class SkillRegistry:
    """Discover and index skill packages available in the repository."""

    def __init__(
        self,
        repo_root: Path,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.environment = dict(environment or {})
        self._packages: tuple[SkillPackage, ...] = ()

    def discover(self) -> tuple[SkillPackage, ...]:
        """Discover and load all skill packages from config/skills."""
        packages: list[SkillPackage] = []
        for skill_dir in discover_skill_dirs(self.repo_root):
            packages.append(
                load_skill_package(self.repo_root, skill_dir, environment=self.environment)
            )
        self._packages = tuple(packages)
        return self._packages

    def list_packages(
        self,
        *,
        agent_name: str | None = None,
        only_loadable: bool = False,
    ) -> tuple[SkillPackage, ...]:
        """List discovered packages filtered by agent and loadability."""
        packages = self._packages
        if agent_name is None and not only_loadable:
            return packages

        normalized_agent = agent_name.strip() if isinstance(agent_name, str) else None
        filtered: list[SkillPackage] = []
        for package in packages:
            if only_loadable and not package.is_loadable:
                continue
            if (
                normalized_agent is not None
                and normalized_agent not in package.manifest.requires.agents
            ):
                continue
            filtered.append(package)
        return tuple(filtered)

    def list_tool_classes(self, *, agent_name: str | None = None) -> dict[str, type[BaseTool]]:
        """Return loadable tool classes indexed by manifest tool name."""
        indexed: dict[str, type[BaseTool]] = {}
        for package in self.list_packages(agent_name=agent_name, only_loadable=True):
            for tool_manifest, tool_class in zip(
                package.manifest.tools, package.tool_classes, strict=False
            ):
                indexed[tool_manifest.name] = tool_class
        return indexed


# Backwards-compatible alias used by earlier drafts and generated code.
SkillsRegistry = SkillRegistry
