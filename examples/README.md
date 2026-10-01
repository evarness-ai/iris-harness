# Examples

Runnable examples of building on IRIS. Each one is a small directory -- the code, its
manifest, a README and a test -- that imports only the stable API
(`iris_harness.sdk`, `iris_harness.testing`, the mail-provider facade; see
`docs/reference/stable-api.md`), runs offline on a scripted model in well under five
minutes, and is tested in CI.

| Example | What it shows |
|---|---|
| [`00-quickstart`](00-quickstart/) | `iris doctor`, `iris email demo`, and a first governed turn from Python. |
| [`01-deterministic-handler`](01-deterministic-handler/) | A handler that answers without a model (`register_intercept`), audited as `deterministic`. |
| [`02-governed-tool`](02-governed-tool/) | Tools whose manifest declares their effect, content and confirmation; an approval-gated call, approved and rejected. |
| [`03-custom-categories`](03-custom-categories/) | Your own email categories (`judge.yaml`), live in the email assistant. |
| [`04-routine-to-telegram`](04-routine-to-telegram/) | A scheduled job delivered to a channel; Telegram faked at the HTTP layer. |
| [`05-your-own-agent`](05-your-own-agent/) | A domain agent on the governed loop (`register_loop_intent`) with its own tools and a deterministic fallback. |
| [`06-testing-your-plugin`](06-testing-your-plugin/) | How to test a plugin: the harness, a scripted model, audit assertions, `no_network`, the stable-import check. |
| [`07-verify-with-evarness`](07-verify-with-evarness/) | The email-onboarding invariants (no personal data to the cloud, no write without approval, everything audited), exported as a proof bundle and verified offline. |
| [`08-mcp`](08-mcp/) | Serve IRIS tools to an MCP client; allowlist and sign an external MCP server. |

## Run them

```bash
pytest examples -q                   # all of them
pytest examples/02-governed-tool -q  # one
```

They run in the full suite too (`pyproject.toml` lists `examples` beside `tests`), and
`tests/unit/test_stable_tier.py` fails if any of them imports outside the stable API.

## Use one in your IRIS

A plugin example runs from your IRIS home without packaging: copy its directory to
`~/.iris/plugins/<name>/` and list `<name>` under `plugins:` in `~/.iris/profile.yaml`.
To ship one, package it with an entry point in the `iris_harness.plugins` group.
Each README has the details.

Skills (tool packages under `config/skills/`) are covered by the guide
`docs/guides/writing-a-skill.md`.
