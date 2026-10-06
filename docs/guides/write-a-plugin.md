# Write a plugin

Everything outside the governed loop and the governance kernel is a plugin: the email
assistant, the channels, the research tool, the code sandbox. Yours registers through
the same API and runs inside the same governance, with no side door.

## Start from the scaffold

```bash
iris plugins new weather-now --kind tool
cd weather-now
pip install -e ".[test]"
pytest
```

`iris plugins new NAME --kind KIND [--dir PATH] [--force]` writes a package that installs
and passes its own tests offline: a `pyproject.toml` with the `iris_harness.plugins`
entry point, the package with its `manifest.yaml` and `setup(api)`, a README, and
tests that mount the plugin in a real governed harness on the scripted model, plus a
check that it imports only the [stable API](../reference/stable-api.md).

| `--kind` | What it generates |
|---|---|
| `tool` | One read-only tool on the governed ReAct loop, called by the scripted model in its test. |
| `channel` | A delivery channel (`IChannelConnector`: `name`, `send`, `healthy`) on the harness's gateway. |
| `mail-provider` | A `MailProvider` the email sweep syncs: connecting an account, incremental sync, the mailbox-write gate. Needs `iris-harness[email]`. |
| `research-provider` | A web-search backend (`SearchProvider` returning `SearchHit`s) in the `research` tool's provider chain. |
| `skill` | A declarative plugin: tools declared in the manifest and bound to plain functions, with no `plugin.py`. |

## What a plugin is

A directory, or an installed package, with a manifest and a `setup(api)`:

```yaml
# manifest.yaml
name: weather-now
version: 0.1.0
description: One line.
entrypoint: plugin:setup
party: untrusted     # who wrote it: first-party | trusted-third-party | untrusted
provides: [tool]
tools:
  weather_now:
    effect: read        # read | write (asks once per run) | destructive (approved per call)
    content: internal   # external: a third party wrote it; scanned for injected instructions
```

```python
# plugin.py
from iris_harness.sdk import PluginAPI


def weather_now(args: dict) -> str:
    return "Sunny, 21 C"  # what the model reads as the tool's observation


def setup(api: PluginAPI) -> None:
    api.register_tool("weather_now", "Current weather. Args: {}.", weather_now)
```

A tool must be declared under `tools:` with its effect, or the plugin is refused when
it mounts. The declaration is what the governance kernel enforces: a `write` tool asks
the owner once before the first write of a run, a `destructive` one waits for approval
on every call, and an `external` tool's result is scanned before the model sees it.

## The six registration kinds

| Kind | Call | What it is for |
|---|---|---|
| intercept | `api.register_intercept(name, fn, ...)` | A [deterministic handler](../concepts/deterministic-handlers.md): answer a turn with no model, or return `None` to pass it on. |
| tool | `api.register_tool(name, description, call)` | A tool on the governed ReAct loop. |
| intent handler | `api.register_intent_handler(agent_type, fn, ...)` | An agent: the turns the router sends to your intent. |
| heartbeat | `api.register_heartbeat(name, fn, schedule="interval:60", ...)` | A scheduled job. |
| channel | `api.register_channel(connector)` | A delivery surface (Telegram, web push, your own). |
| confirmation executor | `api.register_confirmation_executor(kind, fn)` | What runs when the owner approves a pending action of your kind. |

Every registration is wrapped in one fault boundary: a failing plugin shows up as a
degraded or failed `plugin:<name>` row in System Health instead of taking the process
down.

## Run it in your IRIS

IRIS finds plugins in this order: built in, then installed packages (entry-point group
`iris_harness.plugins`), then `$IRIS_HOME/plugins/<name>/` (a directory with a
`manifest.yaml`, no packaging needed). A **profile** says which ones mount and in what
order: the shipped one is chosen by `IRIS_PROFILE` (unset, it is `email` when the email
plugins are installed, else `default`), and `$IRIS_HOME/profile.yaml` adds yours:

```yaml
# ~/.iris/profile.yaml
plugins:
  - name: weather-now
```

`iris --dump-config` prints the effective profile, where each plugin was found and what
it provides, without starting anything. On a running IRIS, `iris plugins` shows each
plugin's status and `iris plugins show NAME` its registrations.

## What you may import

Only the stable tier: `iris_harness.sdk`, `iris_harness.testing`, and the mail-provider
facade `iris_personal.email.provider_api`. Everything else is internal and may change
in any release. The list, the deprecation rule and the check your own CI can run are
in [Stable API](../reference/stable-api.md).

## Next

- [Test your plugin](test-your-plugin.md) against a real governed harness.
- The [examples](examples.md): a deterministic handler, a governed tool with
  approvals, your own agent, a scheduled routine, MCP.
- The full contract (capabilities, services, events, CLI commands, web console
  screens): [the plugin contract](https://github.com/evarness-ai/iris-harness/blob/main/docs/architecture/plugin-contract.md).
