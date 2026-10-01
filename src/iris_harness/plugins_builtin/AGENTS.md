# plugins_builtin/

The reference plugins, written exactly as an outside plugin would be.

## Map

- `system/` — deterministic time/date answers and the `system_health` tool.
- `research/` — the `research` tool over a pluggable provider chain, with its own egress guards.
- `code_exec/` — a bounded model and sandbox loop.
- `telegram_channel/`, `web_channel/`, `web_push_channel/` — delivery channels.
- `graphiti_import/` — imports a Graphiti export into memris.

Each is a package with a `manifest.yaml` and a `plugin.py` whose `setup(api)` registers its capabilities. Profiles in `config/profiles/` decide which mount.

## Local rules

- Import `iris_harness.sdk` and nothing else from the core; no plugin imports another plugin. `poetry run lint-imports` enforces both, and `tests/unit/test_import_contract_coverage.py` fails until a new plugin is listed in the contracts.
- A plugin never calls a tool or an agent directly; it registers, and the kernel runs it.
- A failure degrades the capability (the fault boundary), it never takes the turn down.
- A new plugin gets `tests/unit/iris_harness/plugins_builtin/test_<name>/`, including a standalone test that the core still answers with it unmounted.

## Related

- Contract: `docs/architecture/plugin-contract.md`. Scaffold: `iris plugin new`.
