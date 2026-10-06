# Test your plugin

`iris_harness.testing` builds the same runtime `iris` runs (routing, the governed loop,
the curator, the audit ledger) inside your test, on a scripted model, offline. A plugin
test proves three things: the logic works, the plugin works inside a governed IRIS, and
the guarantees hold.

## A first test

```python
from pathlib import Path

from iris_harness.testing import harness, plugin

from my_plugin import setup

MANIFEST = Path(__file__).with_name("manifest.yaml")


def test_my_plugin_answers() -> None:
    script = {"default": {"content": "Scripted answer."}}
    with harness(plugins=[plugin(setup, manifest=MANIFEST)], fake_model=script) as h:
        assert h.plugin_loaded("my-plugin")
        result = h.chat("hello")          # or h.chat_stream("hello")
        assert result.text == "Scripted answer."
        assert h.audit_gaps() == []       # every model call and every answer audited
```

- **`harness(plugins=[...], fake_model=...)`** runs in a throwaway `IRIS_HOME`, with your
  own `IRIS_*` settings and credentials out of the environment, a throwaway vault master
  key, a keyring that refuses every call, and no network. On exit it puts back every
  piece of process-wide state the run filled, so two harnesses in one process never see
  each other's plugins.
- **`plugin(setup, manifest=...)`** mounts your plugin without packaging it, through the
  same loader and manifest checks as an installed one. For a declarative plugin, pass
  only the manifest: `plugin(manifest=PATH)`.

## The scripted model

The fake model is data: rules matched in order against each model call's `system` and
`user` text (regular expressions), answering with `content`, `json` or `tool_calls`. A
named group in the match fills `{name}` in the reply. Leave out `default:` in a real
test, so a call you did not script fails loudly; `h.model_calls()` lists which rule
answered each call. The [testing example](https://github.com/evarness-ai/iris-harness/tree/main/examples/06-testing-your-plugin)
has a full `fake_model.yaml` that drives a tool call through the governed loop.

## What a turn gives you

`chat` and `chat_stream` return a `TurnResult`: `text`, `agent`, `intent`, `sources`,
`answered` (false only when a streamed turn ended in an error), `error`, `session_id`,
`audit_refs` (the ids of the ledger rows the turn wrote) and, for `chat_stream`,
`events` (`TurnEvent(kind, text)`, the terminal `done` or `error` last).

## Plugin state and degraded plugins

`h.plugins()` is `{name: (status, load_error)}`. `h.plugin_states()` returns a frozen
`PluginState(status, load_error, degraded_reason, failure_count)` per plugin. A mounted
plugin giving degraded answers has a `degraded_reason`: guarded calls that failed, or an
optional `capabilities: uses` that no mounted plugin provides (`optional capability
weather.forecast unavailable (degraded)`). The same text is the plugin's yellow Health
line, and it clears once a provider mounts.

## Assert on governance

```python
rows = h.audit_rows(hook_point="pre_response", session_id=result.session_id)
assert [row.handler for row in rows if row.deterministic] == ["my_handler"]
assert h.model_calls() == ()   # a deterministic handler answered: no model was asked
```

`audit_rows` returns `TurnAuditRow`s: `hook_point`, `plugin` (the governance check
that wrote the row), `decision`, `reason`, `run_id`, `step_id`, `classification`,
`tier`, `session_id`, `tool`, and `deterministic` and `handler` on a deterministic
handler's answer row. `h.audit_gaps()` is empty when every model call and every answer
has its row.

A turn that stops for the owner's approval (a destructive tool, a write declared
`approval: pinned`) leaves a pending approval; `h.respond_to_approval(approval_id,
approve=True)` answers it as the owner would, the halted run resumes, and the approved
call runs, governed again. The
[governed-tool example](https://github.com/evarness-ai/iris-harness/tree/main/examples/02-governed-tool)
approves one call and rejects another.

## No network, stable imports only

```python
from pathlib import Path

from iris_harness.testing import check_stable_imports, no_network


def test_my_plugin_uses_only_the_stable_api() -> None:
    assert check_stable_imports([Path("src/my_plugin")]) == []


def test_offline() -> None:
    with no_network() as attempts:
        ...  # an outbound connection in here raises NetworkBlockedError
    assert attempts == []
```

## Module-level state

A plugin that keeps a module-level registry or cache declares it once, at the bottom of
its module, so the harness can restore it between tests:

```python
from iris_harness.sdk.process_state import track_globals

_seen: dict[str, str] = {}

track_globals(__name__, "_seen")
```
