"""Regression: sandbox catastrophic-command guard (exp-007 S3b).

The chat code_exec path issues `run_shell` commands inside Docker; the governance
kernel can't see them. This guard refuses catastrophic commands at the sandbox tool
layer (defense-in-depth + audit; Docker already contains them). Patterns must be NARROW
so legitimate code-exec (relative rm, pip, python, git) is never blocked.
"""

from __future__ import annotations

import pytest

from iris_harness.tools.sandbox_tools import _destructive_match


@pytest.mark.parametrize(
    "cmd",
    [
        "rm -rf /",
        "rm -rf /*",
        "rm -rf ~",
        "rm -rf $HOME",
        "sudo rm -rf / --no-preserve-root",
        ":(){ :|:& };:",
        "mkfs.ext4 /dev/sda1",
        "dd if=/dev/zero of=/dev/sda",
        "echo x > /dev/sda",
        "shred -n 3 /dev/nvme0n1",
        "chmod -R 777 /",
    ],
)
def test_blocks_catastrophic(cmd):
    assert _destructive_match(cmd) is not None


@pytest.mark.parametrize(
    "cmd",
    [
        "rm -rf build/",
        "rm -rf ./node_modules",
        "rm -rf /workspace/tmp",  # absolute but a real subpath, not root
        "pip install pandas",
        "python script.py",
        "git add -A && git commit -m wip",
        "chmod 644 notes.txt",
        "dd if=input.bin of=output.bin",
        "mkdir -p data/out",
    ],
)
def test_allows_legitimate(cmd):
    assert _destructive_match(cmd) is None


def test_run_shell_blocks_without_executing(monkeypatch):
    from iris_harness.tools import sandbox_tools

    host = object.__new__(sandbox_tools.SandboxToolHost)  # skip Docker preflight
    host._session_id = "t"

    called = {"ran": False}

    class _Stub:
        def run_shell(self, *_a, **_k):
            called["ran"] = True
            raise AssertionError("sandbox must not run a blocked command")

    host._sandbox = _Stub()
    result = host.run_shell("rm -rf / --no-preserve-root")
    assert result.exit_code == 126
    assert "command guard" in result.stderr
    assert called["ran"] is False
