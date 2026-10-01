# kernel/

The governance layer. Everything above calls in; it imports only `foundation/`.

## Map

- `governance/kernel.py` — `GovernanceKernel`: hooks fire in priority order at the hook points in `governance/hooks/types.py`; the registry freezes at `init_lock()`.
- `governance/wiring.py` — how the kernel is assembled when the runtime is built.
- `governance/egress_policy.py` — the class-to-tier table, from `config/governance/egress.yaml`, enforced by the egress gate in `governance/plugins/`.
- `governance/` also holds classification, judges (`governance/evaluator/`), the audit ledger (`governance/audit/`), approvals, the side-effect ledger, cost, devices, threat detection, MCP signing and the vault.
- `governor/` — the older policy engine behind the Governor service (route guards, rate limits). New checks do not go here.

## Local rules

- Every change here needs a CODEOWNER review.
- New policy-sensitive behaviour goes in `governance/` and is reached in-process; a server route that touches it must call in (`tests/security/test_no_bypass.py`).
- Never weaken a default (egress, approval, audit) to make a test pass; fix the caller.
- Files here are large: search first, read in windows.

## Related

- Tests: `tests/unit/iris_harness/kernel/`, `tests/security/`.
