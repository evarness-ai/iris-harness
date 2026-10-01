"""FSJail — Phase 4 ``PreToolUse`` symlink-aware write jail.

Story 12.gov-4.4. Enforces the coding agent's per-persona
``fs_write_jail`` prefixes after resolving the requested target path to
its real absolute path. This closes the classic ".." traversal and
symlink-escape holes:

- ``src/../../../etc/passwd`` denies with ``prefix_mismatch`` because the
  normalized absolute path leaves every allowed prefix.
- ``src/link_out/file.py`` where ``link_out`` points outside the repo
  denies with ``symlink_escape`` because the lexical path looked inside
  the jail but the resolved real path does not.

Priority **30** at ``PreToolUse`` — after ``PersonaSurface`` (15),
``ToolPolicyHook`` (20), and ``CommandSandbox`` (25). By the time this
hook runs the caller persona is already known; FSJail only needs to
verify that the write target stays inside the persona's filesystem jail
and away from globally protected governance files.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final

from iris_harness.kernel.governance.hooks.types import HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.plugins.persona_surface import (
    PersonaPolicy,
    PersonaPolicyDocument,
)

logger = logging.getLogger(__name__)

DEFAULT_WRITE_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "write_file",
        "create_file",
        "edit_file",
        "mkdir",
        "rmtree",
        "git_add",
        "rename",
    }
)
_CONFIG_DENY_FILENAMES: Final[frozenset[str]] = frozenset(
    {"vault.db", "evaluator.db", "approvals.db", "evaluator-policy.yaml"}
)
_LOCAL_DATA_DENY_FILENAMES: Final[frozenset[str]] = frozenset(
    {"audit.db", "evaluator.db", "approvals.db", "evaluator-policy.yaml"}
)


@dataclass(frozen=True)
class _JailPrefix:
    raw: str
    lexical: Path
    resolved: Path


@dataclass(frozen=True)
class _ResolvedTarget:
    label: str
    requested_path: str
    requested_abs: Path
    resolved_path: Path


class FSJail:
    """Enforce persona-scoped write prefixes and protected global paths."""

    name: str = "fs_jail"
    hook_point: HookPoint = HookPoint.PRE_TOOL_USE
    priority: int = 30

    ALLOW_REASON_NOT_WRITE_TOOL: ClassVar[str] = "fs_jail: not a write tool"
    ALLOW_REASON_NON_CODING: ClassVar[str] = "fs_jail: not a coding-agent dispatch"
    ALLOW_REASON_DEGRADED: ClassVar[str] = "fs_jail: no policy loaded (degraded)"

    def __init__(
        self,
        *,
        policy: PersonaPolicyDocument | None = None,
        write_tools: frozenset[str] = DEFAULT_WRITE_TOOLS,
        workspace_root: Path | None = None,
        iris_config_root: Path | None = None,
        iris_local_data_root: Path | None = None,
    ) -> None:
        self._policy = policy
        self._write_tools = frozenset(write_tools)
        self._workspace_root = (
            workspace_root.resolve(strict=False) if workspace_root is not None else None
        )
        self._iris_config_root = (
            iris_config_root.resolve(strict=False)
            if iris_config_root is not None
            else (Path.home() / ".config" / "iris").resolve(strict=False)
        )
        self._iris_runs_root = (self._iris_config_root / "runs").resolve(strict=False)
        self._iris_local_data_root = (
            iris_local_data_root.resolve(strict=False)
            if iris_local_data_root is not None
            else (Path.home() / ".local" / "share" / "iris").resolve(strict=False)
        )
        self._warned_no_policy = False

    async def __call__(self, ctx: HookContext) -> HookDecision:
        tool_name = ctx.payload.get("tool_name")
        if not isinstance(tool_name, str) or tool_name not in self._write_tools:
            return HookDecision(outcome="allow", reason=self.ALLOW_REASON_NOT_WRITE_TOOL)

        if ctx.agent_type != "coding":
            return HookDecision(outcome="allow", reason=self.ALLOW_REASON_NON_CODING)

        if ctx.persona is None:
            return HookDecision(
                outcome="deny",
                reason="fs_jail: coding-agent dispatch missing persona",
                severity="error",
            )

        if ctx.persona == "orchestrator":
            return HookDecision(
                outcome="deny",
                reason="fs_jail: orchestrator persona cannot invoke write tools",
                severity="error",
            )

        if self._policy is None:
            if not self._warned_no_policy:
                logger.warning(
                    "FSJail: no persona-policy loaded; write tools will be allowed "
                    "without jail enforcement"
                )
                self._warned_no_policy = True
            return HookDecision(outcome="allow", reason=self.ALLOW_REASON_DEGRADED)

        persona_policy = self._policy.personas.get(ctx.persona)
        if persona_policy is None:
            return HookDecision(
                outcome="deny",
                reason=f"fs_jail: unknown persona {ctx.persona!r}",
                severity="error",
            )

        workspace_root = _resolve_workspace_root(ctx, default_workspace_root=self._workspace_root)
        targets = _extract_targets(
            payload=ctx.payload,
            tool_name=tool_name,
            workspace_root=workspace_root,
        )
        if not targets:
            return HookDecision(
                outcome="deny",
                reason="fs_jail: missing target path in write-tool payload",
                severity="error",
                audit_metadata={"workspace_root": str(workspace_root)},
            )

        jail_prefixes = _resolve_jail_prefixes(
            workspace_root=workspace_root,
            persona_policy=persona_policy,
        )
        if not jail_prefixes and not persona_policy.fs_read_only:
            return HookDecision(
                outcome="deny",
                reason=(f"fs_jail: persona {ctx.persona!r} has no fs_write_jail configured"),
                severity="error",
                audit_metadata=_audit_metadata_for_target(
                    targets[0],
                    workspace_root=workspace_root,
                    trip_reason="prefix_mismatch",
                ),
            )

        for target in targets:
            if _is_globally_denied(
                target.resolved_path,
                iris_config_root=self._iris_config_root,
                iris_runs_root=self._iris_runs_root,
                iris_local_data_root=self._iris_local_data_root,
            ):
                return HookDecision(
                    outcome="deny",
                    reason=(
                        f"fs_jail: target {target.requested_path!r} resolves to a "
                        "globally protected governance path"
                    ),
                    severity="critical",
                    audit_metadata=_audit_metadata_for_target(
                        target,
                        workspace_root=workspace_root,
                        trip_reason="global_deny",
                    ),
                )

            if persona_policy.fs_read_only:
                return HookDecision(
                    outcome="deny",
                    reason=(
                        f"fs_jail: persona {ctx.persona!r} is read-only and cannot "
                        f"invoke write tool {tool_name!r}"
                    ),
                    severity="error",
                    audit_metadata=_audit_metadata_for_target(
                        target,
                        workspace_root=workspace_root,
                        trip_reason="read_only",
                    ),
                )

            if _is_within_any(target.resolved_path, (jail.resolved for jail in jail_prefixes)):
                continue

            inside_lexical = _is_within_any(
                target.requested_abs, (jail.lexical for jail in jail_prefixes)
            )
            trip_reason = "symlink_escape" if inside_lexical else "prefix_mismatch"
            return HookDecision(
                outcome="deny",
                reason=(
                    f"fs_jail: target {target.requested_path!r} resolves outside persona "
                    f"{ctx.persona!r} fs_write_jail ({trip_reason})"
                ),
                severity="error",
                audit_metadata=_audit_metadata_for_target(
                    target,
                    workspace_root=workspace_root,
                    trip_reason=trip_reason,
                ),
            )

        return HookDecision(
            outcome="allow",
            reason=f"fs_jail: write targets permitted for persona {ctx.persona!r}",
            audit_metadata=_audit_metadata_for_targets(targets, workspace_root=workspace_root),
        )


def _resolve_workspace_root(ctx: HookContext, *, default_workspace_root: Path | None) -> Path:
    raw = ctx.metadata.get("workspace_root")
    if isinstance(raw, str) and raw.strip():
        return Path(raw).expanduser().resolve(strict=False)
    if isinstance(raw, Path):
        return raw.expanduser().resolve(strict=False)
    if default_workspace_root is not None:
        return default_workspace_root
    return Path.cwd().resolve(strict=False)


def _extract_targets(
    *,
    payload: dict[str, Any],
    tool_name: str,
    workspace_root: Path,
) -> tuple[_ResolvedTarget, ...]:
    containers = _payload_containers(payload)
    targets: list[tuple[str, str]] = []

    if tool_name == "rename":
        source_value = _first_string(containers, "source", "src", "from_path", "old_path", "path")
        target_value = _first_string(
            containers, "target", "destination", "dest", "dst", "to_path", "new_path"
        )
        if source_value is not None:
            targets.append(("source", source_value))
        if target_value is not None:
            targets.append(("target", target_value))
        return tuple(
            _resolve_target(label=label, requested_path=path, workspace_root=workspace_root)
            for label, path in targets
        )

    if tool_name == "git_add":
        raw_paths = _first_string_sequence(containers, "paths")
        if raw_paths:
            return tuple(
                _resolve_target(
                    label=f"paths[{index}]",
                    requested_path=path_text,
                    workspace_root=workspace_root,
                )
                for index, path_text in enumerate(raw_paths)
            )

    path_value = _first_string(containers, "path", "target")
    if path_value is None:
        return ()
    return (
        _resolve_target(label="path", requested_path=path_value, workspace_root=workspace_root),
    )


def _payload_containers(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    args = payload.get("args")
    if isinstance(args, dict):
        return (args, payload)
    return (payload,)


def _first_string(containers: tuple[dict[str, Any], ...], *keys: str) -> str | None:
    for container in containers:
        for key in keys:
            raw = container.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    return None


def _first_string_sequence(containers: tuple[dict[str, Any], ...], *keys: str) -> tuple[str, ...]:
    for container in containers:
        for key in keys:
            raw = container.get(key)
            if isinstance(raw, (list, tuple)):
                normalized = tuple(
                    item.strip() for item in raw if isinstance(item, str) and item.strip()
                )
                if normalized:
                    return normalized
    return ()


def _resolve_target(*, label: str, requested_path: str, workspace_root: Path) -> _ResolvedTarget:
    candidate = _candidate_path(workspace_root, requested_path)
    requested_abs = Path(os.path.abspath(os.fspath(candidate)))
    resolved_path = candidate.resolve(strict=False)
    return _ResolvedTarget(
        label=label,
        requested_path=requested_path,
        requested_abs=requested_abs,
        resolved_path=resolved_path,
    )


def _resolve_jail_prefixes(
    *, workspace_root: Path, persona_policy: PersonaPolicy
) -> tuple[_JailPrefix, ...]:
    prefixes: list[_JailPrefix] = []
    for raw_prefix in persona_policy.fs_write_jail:
        raw = raw_prefix.strip()
        if not raw:
            continue
        candidate = _candidate_path(workspace_root, raw)
        prefixes.append(
            _JailPrefix(
                raw=raw,
                lexical=Path(os.path.abspath(os.fspath(candidate))),
                resolved=candidate.resolve(strict=False),
            )
        )
    return tuple(prefixes)


def _candidate_path(workspace_root: Path, raw_path: str) -> Path:
    raw = Path(raw_path).expanduser()
    return raw if raw.is_absolute() else workspace_root / raw


def _is_globally_denied(
    path: Path,
    *,
    iris_config_root: Path,
    iris_runs_root: Path,
    iris_local_data_root: Path,
) -> bool:
    resolved = path.resolve(strict=False)
    if _is_relative_to(resolved, iris_config_root):
        return not _is_relative_to(resolved, iris_runs_root)

    protected_exact_paths = {
        (iris_local_data_root / name).resolve(strict=False) for name in _LOCAL_DATA_DENY_FILENAMES
    }
    protected_exact_paths.update(
        (iris_config_root / name).resolve(strict=False) for name in _CONFIG_DENY_FILENAMES
    )
    return resolved in protected_exact_paths


def _audit_metadata_for_target(
    target: _ResolvedTarget,
    *,
    workspace_root: Path,
    trip_reason: str,
) -> dict[str, Any]:
    payload = _audit_metadata_for_targets((target,), workspace_root=workspace_root)
    payload["trip_reason"] = trip_reason
    payload["target_label"] = target.label
    return payload


def _audit_metadata_for_targets(
    targets: tuple[_ResolvedTarget, ...] | list[_ResolvedTarget],
    *,
    workspace_root: Path,
) -> dict[str, Any]:
    rows = [
        {
            "label": target.label,
            "requested_path": target.requested_path,
            "requested_abs": str(target.requested_abs),
            "resolved_path": str(target.resolved_path),
        }
        for target in targets
    ]
    metadata: dict[str, Any] = {
        "workspace_root": str(workspace_root),
        "targets": rows,
    }
    if len(rows) == 1:
        row = rows[0]
        metadata["requested_path"] = row["requested_path"]
        metadata["requested_abs"] = row["requested_abs"]
        metadata["resolved_path"] = row["resolved_path"]
    return metadata


def _is_within_any(path: Path, roots: Any) -> bool:
    return any(_is_relative_to(path, root) for root in roots)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
