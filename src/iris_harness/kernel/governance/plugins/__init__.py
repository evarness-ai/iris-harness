"""Governance plugins.

Each plugin implements the ``Hook`` protocol (``name``, ``hook_point``,
``priority``, ``__call__``) and is registered with the
``GovernanceKernel`` at process start. Plugins ship in this package; v1
forbids runtime registration and skill-side hook registration (design
§5.3 trust boundary).
"""

from iris_harness.kernel.governance.plugins.classifier import (
    ClassificationResult,
    DataClassifier,
    DataClassifierHook,
)
from iris_harness.kernel.governance.plugins.command_sandbox import CommandSandbox
from iris_harness.kernel.governance.plugins.cost_limiter import CostLimiter
from iris_harness.kernel.governance.plugins.credential_broker import CredentialBroker
from iris_harness.kernel.governance.plugins.destructive_approval import DestructiveApprovalHook
from iris_harness.kernel.governance.plugins.egress_gate import EgressGate
from iris_harness.kernel.governance.plugins.fs_jail import FSJail
from iris_harness.kernel.governance.plugins.mcp_allowlist import (
    MCPAllowlistHook,
    MCPPersonaGrant,
    MCPServerGovernance,
)
from iris_harness.kernel.governance.plugins.network_egress import NetworkEgress
from iris_harness.kernel.governance.plugins.output_classifier import OutputClassifierHook
from iris_harness.kernel.governance.plugins.persona_surface import (
    PersonaPolicy,
    PersonaPolicyDocument,
    PersonaSurface,
)
from iris_harness.kernel.governance.plugins.post_tool_use_ledger import PostToolUseLedgerHook
from iris_harness.kernel.governance.plugins.redaction import RedactionFilterHook
from iris_harness.kernel.governance.plugins.tool_policy import ToolPolicyHook

__all__ = [
    "ClassificationResult",
    "CommandSandbox",
    "CostLimiter",
    "CredentialBroker",
    "DataClassifier",
    "DataClassifierHook",
    "DestructiveApprovalHook",
    "EgressGate",
    "FSJail",
    "MCPAllowlistHook",
    "MCPPersonaGrant",
    "MCPServerGovernance",
    "NetworkEgress",
    "PersonaPolicy",
    "PersonaPolicyDocument",
    "PersonaSurface",
    "OutputClassifierHook",
    "PostToolUseLedgerHook",
    "RedactionFilterHook",
    "ToolPolicyHook",
]
