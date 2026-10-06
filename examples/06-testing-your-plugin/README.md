# 06 · Testing your plugin

How to test a plugin with `iris_harness.testing`, on a small `convert_units` tool. The
test file is the example; read it top to bottom.

| Layer | What the test does | Tools |
|---|---|---|
| 1. Logic | The conversion as plain functions, no IRIS. Fast, and where most of your cases belong. | plain pytest |
| 2. In IRIS | A real governed IRIS in a throwaway home, the plugin mounted in-process, one chat turn on a scripted model: the tool is called, the answer comes from its result, both `chat` and `chat_stream`. An undeclared tool is refused and the plugin shows `degraded`. | `harness`, `plugin`, `Script` |
| 3. Guarantees | Every model call and every answer has its audit row; nothing reached the network; the plugin imports only the stable API; the conformance suite passes. | `audit_gaps`, `audit_rows`, `no_network`, `check_stable_imports`, `assert_conformant` |

| File | What it is |
|---|---|
| `unit_converter.py` | The plugin: `convert` (pure) and the `convert_units` tool. |
| `manifest.yaml` | Declares the tool (`effect: read`). |
| `fake_model.yaml` | The scripted model: an ordered list of `match` -> `reply` rules. |
| `test_unit_converter.py` | The tests, in the three layers above. |

## Run it

```bash
pytest examples/06-testing-your-plugin -q
```

Expected output: `12 passed` in about 30 s (15 s with `--no-cov`).

## The pieces

- **`harness(plugins=[...], fake_model=...)`** builds the same runtime `iris` runs --
  routing, the governed loop, the curator, the audit ledger -- in a temporary
  `IRIS_HOME`, with your `IRIS_*` settings and credentials out of the environment, a
  throwaway vault key, a keyring that refuses every call and no network. On exit it puts
  back every piece of process-wide state the run touched.
- **`plugin(setup, manifest=...)`** mounts your plugin without packaging it, through the
  same loader and manifest checks as an installed one.
- **The scripted model** (`fake_model.yaml`) is data: rules matched in order against the
  call's `system` / `user` text (regexes), answering with `content`, `json` or
  `tool_calls`. `{name}` in a reply is a named group from the match. Leave out
  `default:` so an unscripted call fails loudly. `h.model_calls()` lists which rule
  answered each call.
- **`TurnResult`** -- `text`, `agent`, `intent`, `answered`, `error`, `session_id`,
  `audit_refs`, and for `chat_stream` the `events`.
- **`h.audit_rows(hook_point=..., session_id=...)`** returns `TurnAuditRow`s:
  `hook_point`, `plugin` (the check), `decision`, `reason`, `run_id`, `step_id`,
  `classification`, `tier`, `session_id`, `tool`, `deterministic`, `handler`.
- **`h.audit_gaps()`** is empty when every model call and every answer was audited.
- **`h.respond_to_approval(id, approve=...)`** answers an approval as the owner (see
  example 02).
- **`check_stable_imports([...])`** lists every import outside the stable API
  (`docs/reference/stable-api.md`); hold your plugin to it in CI.
- **`assert_conformant(plugin(...), tools={...}, capabilities={...})`** is the governance
  conformance suite. Give it one example call per declared tool (and per method of each
  capability you provide); it runs each from code as your plugin and checks the audit
  ledger: a `pre_tool_use` and `post_tool_use` row for every call, all naming the caller
  the harness stamped, a destructive tool or confirming write held for the owner, run
  once with the queued arguments when approved and never when rejected. A declared tool
  with no example fails it, and so does an approval check with nothing to check (no queued
  approval, no argument digests to compare). Whether a tool must be held is read from your
  manifest, so declare its `effect` honestly. `check_conformance` returns the `Violation`s
  instead.

## Try changing

Add a unit and test it in the fastest layer first. In `unit_converter.py`, add yards to
the length table:

```python
_TO_METRES = {"m": 1.0, "km": 1000.0, "mi": 1609.344, "ft": 0.3048, "yd": 0.9144}
```

and a case to `test_convert`'s parameters in `test_unit_converter.py`:

```python
        (1, "yd", "ft", 3.0),
```

`13 passed`: the new unit is covered in milliseconds, with no IRIS built. Only what
needs the harness (a turn, its tool call, its audit rows) belongs in layer 2.

## Next

[`07-verify-with-evarness`](../07-verify-with-evarness/): prove governance properties
from the audit ledger, offline. Background:
[Test your plugin](../../docs/guides/test-your-plugin.md).
