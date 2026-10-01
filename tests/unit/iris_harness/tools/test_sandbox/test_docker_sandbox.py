"""Tests for DockerSandbox argv shape + error handling.

These tests do NOT exercise Docker itself — they verify the argv we hand to
``subprocess.run`` and the wrapping behavior around timeouts / missing image.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from iris_harness.tools.sandbox.docker_sandbox import DEFAULT_IMAGE, DockerSandbox
from iris_harness.tools.sandbox.exceptions import (
    DockerUnavailableError,
    SandboxImageMissingError,
)
from iris_harness.tools.sandbox.workspace import SessionWorkspace


@pytest.fixture
def sandbox(tmp_path: Path) -> DockerSandbox:
    ws = SessionWorkspace("s", root=tmp_path)
    return DockerSandbox(ws)


def test_build_argv_contains_required_isolation_flags(sandbox: DockerSandbox) -> None:
    argv = sandbox._build_argv("echo hi", container_name="iris-sandbox-test")
    assert argv[0] == "docker"
    assert argv[1] == "run"
    assert "--rm" in argv
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges" in argv
    assert any(a.startswith("--memory=") for a in argv)
    assert any(a.startswith("--cpus=") for a in argv)
    assert any(a.startswith("--pids-limit=") for a in argv)
    assert any(a.startswith("--tmpfs=/tmp") for a in argv)


def test_build_argv_names_container_for_timeout_cleanup(sandbox: DockerSandbox) -> None:
    argv = sandbox._build_argv("ls", container_name="iris-sandbox-abc123")
    name_idx = argv.index("--name")
    assert argv[name_idx + 1] == "iris-sandbox-abc123"


def test_build_argv_mounts_workspace_rw(sandbox: DockerSandbox, tmp_path: Path) -> None:
    argv = sandbox._build_argv("ls", container_name="iris-sandbox-test")
    mount_idx = argv.index("-v")
    mount_spec = argv[mount_idx + 1]
    host, container, mode = mount_spec.split(":")
    assert Path(host) == sandbox.workspace.path
    assert container == "/workspace"
    assert mode == "rw"


def test_build_argv_uses_default_image_with_bash_lc(sandbox: DockerSandbox) -> None:
    argv = sandbox._build_argv("echo hi", container_name="iris-sandbox-test")
    image_idx = argv.index(DEFAULT_IMAGE)
    assert argv[image_idx + 1 : image_idx + 4] == ["bash", "-lc", "echo hi"]


def test_build_argv_omits_network_none_when_network_on(sandbox: DockerSandbox) -> None:
    argv = sandbox._build_argv("ls", container_name="iris-sandbox-test")
    assert "--network=none" not in argv


def test_build_argv_adds_network_none_when_disabled(tmp_path: Path) -> None:
    ws = SessionWorkspace("s", root=tmp_path)
    sandbox = DockerSandbox(ws, network=False)
    argv = sandbox._build_argv("ls", container_name="iris-sandbox-test")
    assert "--network=none" in argv


def test_run_shell_rejects_empty_command(sandbox: DockerSandbox) -> None:
    result = sandbox.run_shell("   ")
    assert result.exit_code == 2
    assert "empty command" in result.stderr


def test_run_shell_raises_when_docker_missing(sandbox: DockerSandbox) -> None:
    with patch("iris_harness.tools.sandbox.docker_sandbox.shutil.which", return_value=None):
        with pytest.raises(DockerUnavailableError):
            sandbox.run_shell("echo hi")


def test_run_shell_raises_when_image_missing(sandbox: DockerSandbox) -> None:
    with patch(
        "iris_harness.tools.sandbox.docker_sandbox.shutil.which", return_value="/usr/bin/docker"
    ):
        # First subprocess.run = `docker info` (success); second = image inspect (failure)
        info_ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="28.0", stderr="")
        inspect_fail = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="No such image"
        )
        with patch(
            "iris_harness.tools.sandbox.docker_sandbox.subprocess.run",
            side_effect=[info_ok, inspect_fail],
        ):
            with pytest.raises(SandboxImageMissingError):
                sandbox.run_shell("echo hi")


def test_run_shell_returns_result_on_success(sandbox: DockerSandbox) -> None:
    with patch(
        "iris_harness.tools.sandbox.docker_sandbox.shutil.which", return_value="/usr/bin/docker"
    ):
        info_ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="28.0", stderr="")
        inspect_ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr="")
        run_ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="hello\n", stderr="")
        with patch(
            "iris_harness.tools.sandbox.docker_sandbox.subprocess.run",
            side_effect=[info_ok, inspect_ok, run_ok],
        ):
            r = sandbox.run_shell("echo hello")
        assert r.ok is True
        assert r.stdout == "hello\n"
        assert r.exit_code == 0


def test_run_shell_marks_timeout(sandbox: DockerSandbox) -> None:
    with patch(
        "iris_harness.tools.sandbox.docker_sandbox.shutil.which", return_value="/usr/bin/docker"
    ):
        info_ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="28.0", stderr="")
        inspect_ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr="")
        timeout = subprocess.TimeoutExpired(cmd="docker run ...", timeout=5)
        rm_ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch(
            "iris_harness.tools.sandbox.docker_sandbox.subprocess.run",
            side_effect=[info_ok, inspect_ok, timeout, rm_ok],
        ):
            r = sandbox.run_shell("sleep 999", timeout=5)
        assert r.timed_out is True
        assert r.exit_code == 124
        assert "timeout" in r.stderr.lower() or "exceeded" in r.stderr.lower()


def test_run_shell_force_removes_container_on_timeout(sandbox: DockerSandbox) -> None:
    """After a timeout, the daemon-side container must be forcibly removed."""
    with patch(
        "iris_harness.tools.sandbox.docker_sandbox.shutil.which", return_value="/usr/bin/docker"
    ):
        info_ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="28.0", stderr="")
        inspect_ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr="")
        timeout = subprocess.TimeoutExpired(cmd="docker run ...", timeout=5)
        rm_ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with patch(
            "iris_harness.tools.sandbox.docker_sandbox.subprocess.run",
            side_effect=[info_ok, inspect_ok, timeout, rm_ok],
        ) as run_mock:
            sandbox.run_shell("sleep 999", timeout=5)

        run_args = run_mock.call_args_list[2].args[0]
        name_idx = run_args.index("--name")
        container_name = run_args[name_idx + 1]
        assert container_name.startswith("iris-sandbox-")

        rm_call_args = run_mock.call_args_list[3].args[0]
        assert rm_call_args == ["docker", "rm", "-f", container_name]


def test_ensure_ready_auto_builds_when_image_missing(sandbox: DockerSandbox) -> None:
    with patch(
        "iris_harness.tools.sandbox.docker_sandbox.shutil.which", return_value="/usr/bin/docker"
    ):
        info_ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="28.0", stderr="")
        inspect_fail = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="No such image"
        )
        build_ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="built", stderr="")
        inspect_ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="[]", stderr="")
        with patch(
            "iris_harness.tools.sandbox.docker_sandbox.subprocess.run",
            side_effect=[info_ok, inspect_fail, build_ok, inspect_ok],
        ) as run_mock:
            sandbox.ensure_ready(auto_build_image=True)
        assert run_mock.call_count == 4


def test_ensure_ready_auto_build_failure_raises(sandbox: DockerSandbox) -> None:
    with patch(
        "iris_harness.tools.sandbox.docker_sandbox.shutil.which", return_value="/usr/bin/docker"
    ):
        info_ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="28.0", stderr="")
        inspect_fail = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="No such image"
        )
        build_fail = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="failed to solve"
        )
        with patch(
            "iris_harness.tools.sandbox.docker_sandbox.subprocess.run",
            side_effect=[info_ok, inspect_fail, build_fail],
        ):
            with pytest.raises(SandboxImageMissingError, match="auto-build failed"):
                sandbox.ensure_ready(auto_build_image=True)
