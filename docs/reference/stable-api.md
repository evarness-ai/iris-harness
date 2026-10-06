# Stable API (0.1.0)

What a plugin may import and rely on. Anything here changes only after a deprecation
cycle; everything else in the tree is internal and may change in any release (OSS plan
R16).

The declaration is data: [`src/iris_harness/sdk/stable_tier.yaml`](https://github.com/evarness-ai/iris-harness/blob/main/src/iris_harness/sdk/stable_tier.yaml).

## The stable tier

| Surface | What is stable |
|---|---|
| `iris_harness.sdk` | Every module in the package; in each, exactly the names in its `__all__`. |
| `iris_harness.testing` | Same rule. The scripted fake model, `no_network`, the governed harness (`harness`, `plugin` -- `plugin(manifest=...)` with no setup for a declarative plugin, and the harness's own `TurnResult` / `TurnEvent` / `TurnRecord` / `TurnAuditRow`), `check_stable_imports`, `check_network_imports` (plugin source that imports a raw network library), `fake_http` (answers `api.http` requests from a script, replacing only the transport), the governance conformance suite (`iris_harness.testing.conformance`: `check_conformance` / `assert_conformant`, `Violation`, `ConformanceError`), and `iris_harness.testing.proof_bundle` (export and verify the R14 proof bundle, [proof-bundle.md](proof-bundle.md)). The runtime's internal result types and the audit ledger's own row type are not stable. |
| `TurnAuditRow` | What `Harness.audit_rows()` returns, frozen: `id`, `created_at` (a datetime), `hook_point`, `plugin` (the governance check that wrote the row), `decision`, `reason`, `run_id`, `step_id`, `classification`, `tier`, `session_id`, `tool` (on a tool-use row), `deterministic` and `handler` (on the answer row of a deterministic handler), `caller` (who invoked a tool or capability call: `model:system`, `plugin:<consumer>`, `mcp:<client>`) and `tool_plugin` (who owns the tool a tool-use row is about: the plugin's name, `skill:<name>`, `mcp:<server>`, or `system` for a core tool). The last two are `None` on a row they do not describe. `egress` is what a `pre_egress` / `post_egress` row records (host, port, scheme, method, the declared data class, and on the outcome status, byte counts, duration or the error class; never a path, query or body) and `None` on any other row. The raw payload is not part of it: its keys belong to each check. |
| Extension protocols | `iris_harness.sdk.research`: `SearchProvider` (a backend `api.register_search_provider` adds to the `research` tool's chain), `SearchHit` (the frozen hit it returns; the engine's `SearchResult` left the tier before 0.1.0), `SearchType`, `Freshness`, `ChainLink`, `search_provider_chain`. `iris_harness.sdk.channels.IChannelConnector` (what `api.register_channel` takes). |
| External content | `iris_harness.sdk.content.wrap_external_content(text, *, source, tool=None)`: the kernel's own external-content tripwire and `<external_content>` envelope, for plugin code that puts third-party text into a prompt of its own. Idempotent, offline. `redact_external_content(text)` is the tripwire alone, no envelope, for text the plugin keeps or shows the owner (for example the output of a script it ran over third-party data). See [governance](../concepts/governance.md#the-external-content-floor). |
| `PluginAPI` kinds | `intercept`, `tool`, `intent_handler`, `heartbeat`, `channel`, `confirmation_executor`. |
| Governed HTTP | `iris_harness.sdk.http`: `GovernedHttp` (what `api.http` is), `EgressDenied` (a `PermissionError`) and `current_http()` (for a declarative plugin). See [plugin egress](../architecture/plugin-egress.md). |
| Manifest schema | The top-level keys of `manifest.yaml` (`PluginManifest`, including `egress`), and `flavor: declarative` with its tool keys (`description`, `impl`, `args` of `type`/`description`/`required`/`default`/`options`; docs/architecture/plugin-contract.md). |
| Entry-point group | `iris_harness.plugins`. |
| Mail providers | One facade, `iris_personal.email.provider_api`: `MailProvider`, `LabellingProvider`, the types their methods use (`EmailMessage`, `EmailAttachment`, `FetchResult`, `DownloadedAttachment`, `AttachmentCandidate`), `MailSyncStore` (the part of the mail record a sync writes: its cursor, the fetched messages, a stored message's labels) with `default_sync_store`, `register_mail_provider`, `connect_account` (records the owner's account at the provider, so the sweep syncs it), `mailbox_write(account_id, what, *, op)` (a context manager every method that changes the mailbox writes inside: it checks the owner's write approval first, yields a `WriteTally` the write `add`s each landed batch to, and records the total as a `mailbox_write_performed` ledger row, the proof bundle's write observation) and `require_mailbox_writes` (the approval check alone, recording nothing). The record itself (`EmailStore`) is not stable. Imported from the facade: the core never imports the email slice, so the SDK cannot re-export it. |
| Settings | The environment variables `stable_tier.yaml` lists under `settings`, with their defaults: the ones an email install needs ([Environment variables](environment-variables.md) lists them first). Every other `IRIS_*` setting is configuration, not stable. |
| Quickstart CLI | `iris doctor`, `iris serve` (the IRIS API, on `127.0.0.1:8003` unless `--host`/`--port` say otherwise), `iris email demo`, `iris email setup`, and `iris email writes approve --account <id> [--yes]` (the owner's one-time mailbox-write approval: every refused write names it, and the `mail-provider` scaffold's generated test runs it). |

A name in a stable module but not in its `__all__` is internal.

## The deprecation rule

- Adding a name, a registration kind, a manifest key or a command is not breaking.
- Removing or renaming one, or changing its signature or meaning incompatibly, is.
  Before that lands: keep the old name working for at least one minor release, have it
  warn (`DeprecationWarning`) with the replacement, and list it in the release notes.
- Until 1.0 (after about three outside plugins survive without breaks) a minor release
  may carry a deprecation's removal; a patch release never does.

## How it is enforced

- `tests/unit/test_stable_tier.py` holds every tree `stable_tier.yaml` lists under
  `enforced_roots` (the examples and the `iris plugin new` templates) to
  the tier, with `iris_harness.testing.check_stable_imports` (an AST walk of every
  import). import-linter cannot: `examples/00-quickstart` is not a package name, and a
  forbidden contract cannot say "every module but these names".
- The same file pins the kinds, the entry-point group, the manifest keys and the core
  quickstart command to the declaration, and every stable name to
  `tests/fixtures/stable_tier/names.txt`: removing a name fails there.
- `tests/unit/iris_personal/test_email/test_stable_tier_names.py` checks the mail-provider
  names and the `iris email` commands.
- The examples run as tests (`examples/` is in `testpaths`, and ci-linux runs them on
  every push) and `mypy examples/` checks them strictly, as a plugin author's code:
  `iris_harness` and `iris_personal` ship a `py.typed` marker, so the stable tier's
  annotations are the ones a plugin is checked against.

Your plugin's CI can hold itself to the tier the same way:

```python
from pathlib import Path

from iris_harness.testing import check_stable_imports

def test_my_plugin_uses_only_the_stable_api() -> None:
    assert check_stable_imports([Path("src/my_plugin")]) == []
```

## Testing against a real harness

```python
from iris_harness.testing import harness, plugin

def setup(api):
    ...  # your plugin's setup

def test_my_plugin_answers() -> None:
    script = {"default": {"content": "Scripted answer."}}
    with harness(plugins=[plugin(setup, manifest="manifest.yaml")], fake_model=script) as h:
        result = h.chat("hello")          # or h.chat_stream("hello")
        assert result.text == "Scripted answer."
        assert h.audit_gaps() == []       # every model call and every answer audited
```

`harness()` builds the same runtime `iris` runs, in a throwaway `IRIS_HOME`: the
owner's `IRIS_*` settings and credentials out of the environment, a throwaway vault
master key, a keyring that refuses every call, no network. See
`iris_harness/testing/harness.py`.

`chat` and `chat_stream` return a `TurnResult`: `text`, `agent`, `intent`, `sources`,
`answered` (false only when a streamed turn ended in an error), `error`, `session_id`,
`audit_refs` (the ids of the ledger rows the turn wrote) and, for `chat_stream`,
`events` (`TurnEvent(kind, text)`, the terminal `done` or `error` last).
`audit_rows(hook_point=..., session_id=...)` returns the ledger as `TurnAuditRow`s.

A turn that stops for the owner's approval (a destructive tool, a write declared
`approval: pinned`) leaves a pending row on `ApprovalQueue().list_pending()`
(`iris_harness.sdk.approvals`); `respond_to_approval(approval_id, approve=True)` answers
it as the owner would -- the halted run resumes and the approved call runs, governed
again -- and returns what the owner is told. It also runs a call a plugin's code queued
through `api.tools`.

On exit the harness puts back every piece of process-wide state the run filled -- the
mail-provider registry, API routes, health checks, the process bus's subscribers,
caches read from the throwaway home -- so two harnesses in one process never see each
other's plugins. A plugin that keeps a module-level registry or cache declares it once,
at the bottom of its module, so the harness can do the same for it:

```python
from iris_harness.sdk.process_state import track_globals

_seen: dict[str, str] = {}
...
track_globals(__name__, "_seen")
```
