# Examples

Runnable examples of building on IRIS, in
[`examples/`](https://github.com/evarness-ai/iris-harness/tree/main/examples). Each is a
small directory (the code, its manifest, a README and a test) that imports only the
[stable API](../reference/stable-api.md), runs offline on a scripted model in about half a
minute or less (20 to 32 seconds each on an Apple M4 Max with the repository's coverage
report on, 5 to 16 without it; the per-example times are in the examples' README), and is
tested in CI on every change.

| Example | What it shows |
|---|---|
| `00-quickstart` | `iris doctor`, `iris email demo`, and a first governed turn from Python. |
| `01-deterministic-handler` | A handler that answers without a model (`register_intercept`), audited as `deterministic`. |
| `02-governed-tool` | Tools whose manifest declares their effect, content and confirmation; an approval-gated call, approved and rejected. |
| `03-custom-categories` | Your own email categories (`judge.yaml`), live in the email assistant. |
| `04-routine-to-telegram` | A scheduled job delivered to a channel; Telegram faked at the HTTP layer. |
| `05-your-own-agent` | A domain agent on the governed loop with its own tools and a deterministic fallback. |
| `06-testing-your-plugin` | How to test a plugin: the harness, a scripted model, audit assertions, `no_network`, the stable-import check. |
| `07-verify-with-evarness` | The email-onboarding invariants (no personal data to the cloud, no write without approval, everything audited), exported as a [proof bundle](../reference/proof-bundle.md) and verified offline. |
| `08-mcp` | Serve IRIS tools to an MCP client; allowlist and sign an external MCP server. |

## Run them

From a clone, with the development dependencies installed (`poetry install --extras email`):

```bash
pytest examples -q                   # all of them
pytest examples/02-governed-tool -q  # one
```

## Use one in your IRIS

A plugin example runs from your IRIS home without packaging: copy its directory to
`~/.iris/plugins/<name>/` and list `<name>` under `plugins:` in `~/.iris/profile.yaml`.
To ship one, package it with an entry point in the `iris_harness.plugins` group, as
`iris plugins new` does ([Write a plugin](write-a-plugin.md)).
