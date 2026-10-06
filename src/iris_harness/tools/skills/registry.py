"""Registry for loading and filtering code-first skill packages."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

from langchain_core.tools import BaseTool

from .loader import discover_skill_dirs, load_skill_package
from .models import SkillPackage

logger = logging.getLogger(__name__)


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
        self._load_failures: dict[Path, str] = {}
        self._reported_failures: set[tuple[Path, str]] = set()
        self._reported_unavailable: set[tuple[str, tuple[str, ...]]] = set()

    @property
    def load_failures(self) -> dict[Path, str]:
        """Skills the last :meth:`discover` skipped, as ``skill_dir -> one-line reason``."""
        return dict(self._load_failures)

    def discover(self) -> tuple[SkillPackage, ...]:
        """Discover and load all skill packages from config/skills.

        One skill that cannot load (invalid manifest, a tools module whose import
        fails) is skipped and recorded in :attr:`load_failures`; it never aborts the
        pass, so the skills after it still load. A skill whose *declared* prerequisites
        are missing is not a failure: it loads as a blocked package
        (``missing_prerequisites``) without its tools module being imported.
        """
        packages: list[SkillPackage] = []
        failures: dict[Path, str] = {}
        for skill_dir in discover_skill_dirs(self.repo_root):
            try:
                packages.append(
                    load_skill_package(self.repo_root, skill_dir, environment=self.environment)
                )
            except Exception as exc:  # isolate one skill's fault from the rest
                reason = _one_line_reason(exc)
                failures[skill_dir] = reason
                # discover() also runs per chat turn and on hot reload: say it once. A
                # skill that is genuinely broken (a tools.py that raises, a bad manifest,
                # an import its manifest does not declare) is a fault, so it is logged
                # with its traceback; ``load_failures`` carries only the one-line reason.
                if (skill_dir, reason) not in self._reported_failures:
                    self._reported_failures.add((skill_dir, reason))
                    logger.exception("skill %s failed to load; skipping it", skill_dir.name)
            else:
                self._note_unavailable(packages[-1])
        self._packages = tuple(packages)
        self._load_failures = failures
        return self._packages

    def _note_unavailable(self, package: SkillPackage) -> None:
        """One INFO line for a skill that a missing declared package leaves blocked.

        Not a fault: the skill declared what it needs and the install lacks it. Said
        once (``discover()`` re-runs per turn), naming the extra that installs it.
        """
        missing = tuple(
            item.removeprefix("package:")
            for item in package.missing_prerequisites
            if item.startswith("package:")
        )
        key = (package.manifest.name, missing)
        if not missing or key in self._reported_unavailable:
            return
        self._reported_unavailable.add(key)
        extra = package.manifest.requires.extra
        fix = f"; install it with: pip install 'iris-harness[{extra}]'" if extra else ""
        logger.info(
            "skill %s unavailable: missing package %s%s",
            package.manifest.name,
            ", ".join(missing),
            fix,
        )

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


def _one_line_reason(exc: BaseException) -> str:
    """``ExcType: first line of message`` -- no traceback, no multi-line payload."""
    message = (str(exc).strip().splitlines() or [""])[0]
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


# Backwards-compatible alias used by earlier drafts and generated code.
SkillsRegistry = SkillRegistry
