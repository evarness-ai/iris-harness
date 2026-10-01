"""Driving the harness's isolated execution sandbox.

The sandbox runtime is core by design -- it is a containment boundary, not a
plugin's business to reimplement. A plugin that needs to run generated code
drives this one, and `propose_skill_from_sandbox` is how a successful run
becomes a skill proposal instead of being thrown away.
"""

from __future__ import annotations

from iris_harness.tools.propose_skill_from_sandbox import propose_skill_from_sandbox
from iris_harness.tools.sandbox import SandboxConfig, resolve_runtime_name
from iris_harness.tools.sandbox_tools import SandboxToolHost

__all__ = [
    "SandboxConfig",
    "SandboxToolHost",
    "propose_skill_from_sandbox",
    "resolve_runtime_name",
]
