# Plugins and profiles

The core of IRIS is mechanisms: the turn pipeline, the agentic loop, the governance
kernel, the scheduler, memory. Everything that knows about a particular app, data
source or delivery surface is a plugin, the email assistant included.

## One API, one passage

A plugin's `setup(api)` registers what it adds (deterministic handlers, tools, agents,
heartbeats, channels, approval executors) through `PluginAPI`, the plugin SDK
(`src/iris_harness/sdk/`). Each registration lands on the same machinery the core
uses: a plugin tool joins the same ReAct tool pool and runs through the same
`pre_tool_use` path; an agent runs on the same executor; a heartbeat on the same
scheduler; a channel on the same gateway. There is no side door, so there is nothing
extra to govern.

A plugin runs in the IRIS process (`trust: in-process`). That is a contract, not a
sandbox: the plugin imports only the [stable API](../reference/stable-api.md), and an
import-linter contract holds the reference plugins to it.

## Capabilities: plugins that use each other

When one plugin should use another's service in code, it does not import it. The
provider declares a **capability** (a `domain.verb` name such as `mail.read`, with a
`Protocol` published in the SDK) and registers an implementation; a consumer declares
that it uses the capability and asks for it:

```yaml
capabilities:
  provides: [mail.read]   # you implement it
  uses: []                # you work without these, degraded
  requires: []            # you are not loaded without these
```

```python
api.provide("mail.read", MyReader())       # provider
reader = api.capability("mail.read")       # consumer: the implementation, or None
```

Every capability call is governed like a tool call: the kernel checks the consumer's
manifest allows it, applies the method's declared effect, audits it, and masks the
owner's identity out of the result. Undeclared use is refused both ways.

## Profiles

A profile (`config/profiles/<name>.yaml`) lists the plugins that mount, in order. The
shipped ones:

| Profile | What it mounts |
|---|---|
| `minimal` | The governed core and the `system` plugin. |
| `default` | `minimal` plus the reference plugins: Telegram, web and web-push channels, `research`, `code_exec`. |
| `email` | `default` plus the email assistant: the `gmail` and `imap` providers (each optional) and `email_workflows`. |

`IRIS_PROFILE` picks one. Unset, IRIS runs `email` when the email plugins are installed
and loadable, otherwise `default`. Your own `$IRIS_HOME/profile.yaml` layers on top
(add plugins, change the handler order), and `IRIS_PLUGINS_DISABLE=a,b` /
`IRIS_PLUGINS_ENABLE=c,d` override both. `iris --dump-config` prints the result without
starting anything.

## Discovery

1. **Built in**: `iris_harness.plugins_builtin.<name>`.
2. **Installed**: a package with an entry point in the `iris_harness.plugins` group
   (what `iris plugins new` scaffolds).
3. **Home**: `$IRIS_HOME/plugins/<name>/manifest.yaml`, no packaging needed.

## Failure stays local

Every registration runs inside one fault boundary. A plugin whose requirements are not
met is not loaded; one that fails at setup or at run time is recorded against its name
and shows as a yellow (degraded) or red (failed) `plugin:<name>` row in System Health
(the `system_health` tool, `GET /health`, the web console's Health screen). The rest of IRIS keeps running.
