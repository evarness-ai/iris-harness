"""Unit tests for FSJail (story 12.gov-4.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.kernel.governance import HookContext, HookPoint
from iris_harness.kernel.governance.plugins.fs_jail import DEFAULT_WRITE_TOOLS, FSJail
from iris_harness.kernel.governance.plugins.persona_surface import (
    PersonaPolicy,
    PersonaPolicyDocument,
)


def _policy() -> PersonaPolicyDocument:
    return PersonaPolicyDocument(
        personas={
            "developer": PersonaPolicy(
                name="developer",
                allowed_tools=("create_file", "edit_file"),
                fs_write_jail=("./src/", "./tests/", "./docs/"),
            ),
            "analyst": PersonaPolicy(
                name="analyst",
                allowed_tools=("read_file",),
                fs_read_only=True,
                fs_write_jail=("./src/",),
            ),
        }
    )


def _ctx(
    *,
    workspace_root: Path,
    tool: str = "create_file",
    persona: str | None = "developer",
    agent_type: str = "coding",
    path: str | None = None,
    flat_path: bool = False,
) -> HookContext:
    payload: dict[str, object] = {"tool_name": tool}
    if path is not None:
        if flat_path:
            payload["path"] = path
        else:
            payload["args"] = {"path": path}
    return HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        run_id="r-1",
        agent_type=agent_type,
        persona=persona,
        payload=payload,
        metadata={"workspace_root": str(workspace_root)},
    )


def test_plugin_metadata() -> None:
    jail = FSJail(policy=None)
    assert jail.name == "fs_jail"
    assert jail.hook_point == HookPoint.PRE_TOOL_USE
    assert jail.priority == 30


def test_default_write_tools_cover_runtime_create_file_and_story_aliases() -> None:
    assert {"create_file", "edit_file", "write_file", "git_add"} <= DEFAULT_WRITE_TOOLS


async def test_ac1_developer_write_inside_jail_allowed(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    (workspace_root / "src" / "iris").mkdir(parents=True)
    jail = FSJail(policy=_policy())

    decision = await jail(_ctx(workspace_root=workspace_root, path="src/iris_harness/foo.py"))

    assert decision.outcome == "allow"
    assert decision.audit_metadata["requested_path"] == "src/iris_harness/foo.py"
    assert decision.audit_metadata["resolved_path"].endswith("src/iris_harness/foo.py")


async def test_ac2_path_traversal_outside_jail_denied_with_prefix_mismatch(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    (workspace_root / "src").mkdir(parents=True)
    jail = FSJail(policy=_policy())

    decision = await jail(_ctx(workspace_root=workspace_root, path="src/../../../etc/passwd"))

    assert decision.outcome == "deny"
    assert decision.audit_metadata["trip_reason"] == "prefix_mismatch"
    assert "prefix_mismatch" in decision.reason


async def test_ac3_symlink_escape_denied(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    (workspace_root / "src").mkdir(parents=True)
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    (workspace_root / "src" / "link_out").symlink_to(outside_root, target_is_directory=True)
    jail = FSJail(policy=_policy())

    decision = await jail(_ctx(workspace_root=workspace_root, path="src/link_out/owned.py"))

    assert decision.outcome == "deny"
    assert decision.audit_metadata["trip_reason"] == "symlink_escape"
    assert decision.audit_metadata["resolved_path"].startswith(str(outside_root))


async def test_ac4_global_deny_path_is_critical(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    iris_config_root = tmp_path / "iris-config"
    (iris_config_root / "runs").mkdir(parents=True)
    jail = FSJail(policy=_policy(), iris_config_root=iris_config_root)

    decision = await jail(
        _ctx(
            workspace_root=workspace_root,
            tool="edit_file",
            path=str(iris_config_root / "vault.db"),
        )
    )

    assert decision.outcome == "deny"
    assert decision.severity == "critical"
    assert decision.audit_metadata["trip_reason"] == "global_deny"


async def test_ac5_read_only_persona_denied_even_inside_notional_jail(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    (workspace_root / "src").mkdir(parents=True)
    jail = FSJail(policy=_policy())

    decision = await jail(
        _ctx(workspace_root=workspace_root, persona="analyst", path="src/report.md")
    )

    assert decision.outcome == "deny"
    assert decision.audit_metadata["trip_reason"] == "read_only"
    assert "read-only" in decision.reason


async def test_ac6_brand_new_leaf_under_existing_parent_allowed(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    (workspace_root / "src" / "iris").mkdir(parents=True)
    jail = FSJail(policy=_policy())

    decision = await jail(_ctx(workspace_root=workspace_root, path="src/iris_harness/brand_new.py"))

    assert decision.outcome == "allow"
    assert decision.audit_metadata["resolved_path"].endswith("src/iris_harness/brand_new.py")


async def test_flat_payload_path_shape_is_supported(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    (workspace_root / "docs").mkdir(parents=True)
    jail = FSJail(policy=_policy())

    decision = await jail(
        _ctx(
            workspace_root=workspace_root,
            tool="edit_file",
            path="docs/notes.md",
            flat_path=True,
        )
    )

    assert decision.outcome == "allow"


async def test_non_write_tool_short_circuits_allow(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    jail = FSJail(policy=_policy())

    decision = await jail(_ctx(workspace_root=workspace_root, tool="read_file", path="src/foo.py"))

    assert decision.outcome == "allow"
    assert decision.reason == FSJail.ALLOW_REASON_NOT_WRITE_TOOL


async def test_non_coding_agent_short_circuits_allow(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    jail = FSJail(policy=_policy())

    decision = await jail(
        _ctx(
            workspace_root=workspace_root,
            agent_type="chat",
            path="src/foo.py",
        )
    )

    assert decision.outcome == "allow"
    assert decision.reason == FSJail.ALLOW_REASON_NON_CODING


async def test_missing_path_denies_loudly(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    jail = FSJail(policy=_policy())

    decision = await jail(_ctx(workspace_root=workspace_root, path=None))

    assert decision.outcome == "deny"
    assert "missing target path" in decision.reason


async def test_no_policy_degrades_to_allow_with_single_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    jail = FSJail(policy=None)

    with caplog.at_level("WARNING"):
        first = await jail(_ctx(workspace_root=workspace_root, path="src/foo.py"))
        second = await jail(_ctx(workspace_root=workspace_root, path="src/bar.py"))

    assert first.outcome == "allow"
    assert first.reason == FSJail.ALLOW_REASON_DEGRADED
    assert second.outcome == "allow"
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
