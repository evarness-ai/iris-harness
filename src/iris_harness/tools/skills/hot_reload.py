"""Filesystem watcher that re-discovers skills when manifests change.

Disabled by default. Enabled when ``IRIS_SKILL_HOTRELOAD=1`` is set in the
environment. Uses :mod:`watchdog` to observe ``config/skills/`` and triggers a
fresh :meth:`SkillRegistry.discover` on any change to ``manifest.yaml`` or
``tools.py``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from iris_harness.tools.skills.loader import MANIFEST_FILE, TOOLS_FILE, resolve_skills_root
from iris_harness.tools.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

ENV_FLAG = "IRIS_SKILL_HOTRELOAD"


def is_hot_reload_enabled() -> bool:
    """Return ``True`` when hot reload should be activated."""
    return os.environ.get(ENV_FLAG, "").strip() in {"1", "true", "TRUE", "yes"}


class SkillHotReloader:
    """Watch the skills directory and re-discover on change.

    The reloader is a no-op until :meth:`start` is called, and is safe to call
    when hot reload is disabled (``start`` returns ``False`` in that case).
    """

    def __init__(
        self,
        *,
        registry: SkillRegistry,
        repo_root: Path,
    ) -> None:
        self._registry = registry
        self._repo_root = repo_root.resolve()
        self._observer: Any | None = None

    def start(self) -> bool:
        """Begin watching. Returns ``True`` if the watcher was started."""
        if not is_hot_reload_enabled():
            return False
        if self._observer is not None:
            return True
        skills_root = resolve_skills_root(self._repo_root)
        if not skills_root.exists():
            logger.info("hot reload skipped — %s does not exist", skills_root)
            return False
        observer: Any = Observer()
        observer.schedule(
            _SkillChangeHandler(registry=self._registry),
            str(skills_root),
            recursive=True,
        )
        observer.daemon = True
        observer.start()
        self._observer = observer
        logger.info("skill hot reload started for %s", skills_root)
        return True

    def stop(self) -> None:
        if self._observer is None:
            return
        try:
            self._observer.stop()
            self._observer.join(timeout=2.0)
        finally:
            self._observer = None


class _SkillChangeHandler(FileSystemEventHandler):
    """Re-run discovery when a skill manifest or tools module changes."""

    _WATCHED_NAMES = frozenset({MANIFEST_FILE, TOOLS_FILE})

    def __init__(self, *, registry: SkillRegistry) -> None:
        super().__init__()
        self._registry = registry

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(str(event.src_path))
        if path.name not in self._WATCHED_NAMES:
            return
        try:
            self._registry.discover()
            logger.info("skill registry re-discovered after change to %s", path)
        except Exception:  # never crash the watcher thread
            logger.exception("hot reload discovery failed for %s", path)
