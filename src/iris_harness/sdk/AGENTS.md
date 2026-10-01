# sdk/

The author-facing API: what a plugin imports. Part of the stable tier.

## Map

- `__init__.py` — the plugin contract: `setup(api: PluginAPI)` and the registration kinds. `PluginAPI` itself lives in `src/iris_harness/runtime/plugin_host/api.py`; the SDK re-exports what an author writes against.
- One module per capability a plugin may use (`tools.py`, `channels.py`, `research.py`, `approvals.py`, `audit.py`, `memory.py`, `llm.py`, ...). Services reach a plugin through the typed `HarnessServices` fields, never through untyped extras.
- `stable_tier.yaml` — the declaration of what is stable.

## Local rules

- A name exported in a module's `__all__` is a promise. Adding one is fine; removing or renaming one needs a deprecation cycle (`docs/reference/stable-api.md`) and a matching change to `tests/fixtures/stable_tier/names.txt`.
- Do not re-export internals for convenience; a plugin that needs a core capability gets a typed seam here.
- The SDK may import the layers below it; nothing below may import the SDK.
- Nothing in the SDK may offer a plugin a path around the governance kernel.

## Related

- Tests: `tests/unit/iris_harness/test_sdk/`, `tests/unit/test_stable_tier.py`.
- Contract: `docs/architecture/plugin-contract.md`.
