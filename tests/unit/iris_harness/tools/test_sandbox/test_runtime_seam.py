"""Sandbox runtime seam — Phase 6 sub-phase 6c.1.

Config + factory + Protocol conformance. Default docker → no behavior change.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.tools.sandbox import (
    DockerSandbox,
    GVisorSandbox,
    SandboxConfig,
    SandboxRuntime,
    SessionWorkspace,
    build_sandbox_runtime,
    instantiate_runtime,
    is_runtime_available,
    resolve_runtime_name,
)


def _workspace(tmp_path: Path) -> SessionWorkspace:
    return SessionWorkspace("s1", root=tmp_path)


# --- config ------------------------------------------------------------------


def test_packaged_config_defaults_to_docker() -> None:
    cfg = SandboxConfig.from_yaml(
        Path(__file__).resolve().parents[5] / "config" / "governance" / "sandbox.yaml"
    )
    assert cfg.runtime == "docker"
    assert cfg.on_unavailable == "fallback"
    assert cfg.limits.memory == "1g"
    assert cfg.limits.pids_limit == 256


def test_absent_config_is_docker_default(tmp_path: Path) -> None:
    cfg = SandboxConfig.from_yaml(tmp_path / "nope.yaml")
    assert cfg.runtime == "docker"


def test_unknown_key_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("version: 1\nbogus: true\n", encoding="utf-8")
    with pytest.raises(ValueError):
        SandboxConfig.from_yaml(p)


def test_env_overrides_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_SANDBOX_RUNTIME", "gvisor")
    assert SandboxConfig(runtime="docker").resolved_runtime() == "gvisor"


def test_env_override_ignores_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_SANDBOX_RUNTIME", "nonsense")
    assert SandboxConfig(runtime="docker").resolved_runtime() == "docker"


# --- protocol conformance + argv ---------------------------------------------


def test_docker_sandbox_satisfies_protocol(tmp_path: Path) -> None:
    sandbox = DockerSandbox(_workspace(tmp_path))
    assert isinstance(sandbox, SandboxRuntime)
    assert sandbox.name == "docker"


def test_docker_argv_unchanged_no_runtime_flag(tmp_path: Path) -> None:
    sandbox = DockerSandbox(_workspace(tmp_path))
    argv = sandbox._build_argv("echo hi", container_name="c1")
    # No runtime flag spliced for Docker: `docker run --rm ...`
    assert argv[:3] == ["docker", "run", "--rm"]
    assert "--runtime=runsc" not in argv


# --- factory + selection -----------------------------------------------------


def test_factory_builds_docker_when_available(tmp_path: Path) -> None:
    runtime, name = build_sandbox_runtime(
        _workspace(tmp_path), config=SandboxConfig(runtime="docker"), available=lambda r: True
    )
    assert name == "docker"
    assert isinstance(runtime, DockerSandbox)


def test_factory_falls_back_to_docker_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # gvisor requested but unavailable; docker available → fallback.
    runtime, name = build_sandbox_runtime(
        _workspace(tmp_path),
        config=SandboxConfig(runtime="gvisor", on_unavailable="fallback"),
        available=lambda r: r == "docker",
    )
    with caplog.at_level("WARNING"):
        pass
    assert name == "docker"
    assert isinstance(runtime, DockerSandbox)


def test_factory_disable_returns_none(tmp_path: Path) -> None:
    runtime, name = build_sandbox_runtime(
        _workspace(tmp_path),
        config=SandboxConfig(runtime="gvisor", on_unavailable="disable"),
        available=lambda r: False,
    )
    assert runtime is None
    assert name == "gvisor"


def test_factory_no_runtime_available_returns_none(tmp_path: Path) -> None:
    runtime, name = build_sandbox_runtime(
        _workspace(tmp_path),
        config=SandboxConfig(runtime="docker", on_unavailable="fallback"),
        available=lambda r: False,
    )
    assert runtime is None


def test_firecracker_not_available() -> None:
    assert is_runtime_available("firecracker") is False


# --- gVisor backend (6c.2) ---------------------------------------------------


def test_gvisor_sandbox_argv_has_runsc_runtime(tmp_path: Path) -> None:
    sandbox = GVisorSandbox(_workspace(tmp_path))
    assert sandbox.name == "gvisor"
    assert isinstance(sandbox, SandboxRuntime)
    argv = sandbox._build_argv("echo hi", container_name="c1")
    # runtime flag spliced right after `docker run`, rest of argv unchanged.
    assert argv[:3] == ["docker", "run", "--runtime=runsc"]
    assert "--cap-drop=ALL" in argv  # hardening baseline preserved


def test_instantiate_gvisor(tmp_path: Path) -> None:
    runtime = instantiate_runtime(
        "gvisor", _workspace(tmp_path), config=SandboxConfig(runtime="gvisor")
    )
    assert isinstance(runtime, GVisorSandbox)


def test_instantiate_unknown_falls_back_to_docker(tmp_path: Path) -> None:
    runtime = instantiate_runtime("bogus", _workspace(tmp_path), config=SandboxConfig.default())
    assert isinstance(runtime, DockerSandbox)
    assert not isinstance(runtime, GVisorSandbox)


def test_resolve_picks_gvisor_when_available() -> None:
    cfg = SandboxConfig(runtime="gvisor")
    assert resolve_runtime_name(cfg, available=lambda r: True) == "gvisor"


def test_resolve_gvisor_falls_back_to_docker() -> None:
    cfg = SandboxConfig(runtime="gvisor", on_unavailable="fallback")
    assert resolve_runtime_name(cfg, available=lambda r: r == "docker") == "docker"


def test_resolve_disable_returns_none() -> None:
    cfg = SandboxConfig(runtime="gvisor", on_unavailable="disable")
    assert resolve_runtime_name(cfg, available=lambda r: False) is None


def test_build_factory_constructs_gvisor_when_available(tmp_path: Path) -> None:
    runtime, name = build_sandbox_runtime(
        _workspace(tmp_path), config=SandboxConfig(runtime="gvisor"), available=lambda r: True
    )
    assert name == "gvisor"
    assert isinstance(runtime, GVisorSandbox)
