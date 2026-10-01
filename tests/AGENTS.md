# tests/

The pytest suite. `unit/` mirrors the source tree: `src/iris_harness/<layer>/<pkg>/bar.py` is tested in `tests/unit/iris_harness/<layer>/test_<pkg>/test_bar.py` (a leaf keeps the `test_` prefix so it cannot shadow a stdlib module).

## Map

- `conftest.py` — relocates `IRIS_HOME` and `IRIS_DATA_DIR` to temp dirs before any import; blocks model ports, IRIS service ports and non-loopback hosts.
- `unit/` — per-package tests. `unit/test_stable_tier.py` pins the stable tier.
- `integration/` — cross-component flows; deterministic, clean state.
- `security/` — `test_no_bypass.py` fails any model or tool call site that skips the governance kernel.
- `fixtures/` — shared data, including `fixtures/stable_tier/names.txt`.

## Local rules

- `asyncio_mode = "auto"`: never add `@pytest.mark.asyncio`. `S101` (assert) is allowed.
- No test reaches a model or the network: use the scripted fake model in `iris_harness.testing`, or the `offline_*` fixtures, or mark it `real_llm` (deselected by default).
- A test never writes the real profile; resolve state through `IRIS_HOME`.
- A bug fix comes with the test that would have caught it; never weaken an assertion to go green.

## Running

`IRIS_AUTH_SECRET="test-secret-for-testing" IRIS_DISABLE_WARMUP=1 poetry run pytest <narrow path>`. Markers `real_llm`, `real_embeddings` and `smoke` are deselected by default. `scripts/changed_tests.sh` picks the tests for a change.
