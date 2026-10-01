# Examples

Runnable examples of building on IRIS. Each one is a small directory -- the code, its
manifest, a README and a test -- that imports only the stable API
(`iris_harness.sdk`, `iris_harness.testing`, the mail-provider facade; see
`docs/reference/stable-api.md`), runs offline on a scripted model in about half a
minute or less ([measured](#how-long-they-take)), and is tested in CI.

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

### How long they take

Measured on 2026-10-01 on an Apple M4 Max: wall-clock time of one
`pytest examples/<dir> -q` per example, Python start-up included. The repository's
pytest options turn on coverage and write an HTML report, which is most of the time;
`--no-cov` leaves it out.

| Example | Tests | `pytest examples/<dir> -q` | with `--no-cov` |
|---|---|---|---|
| `00-quickstart` | 3 | 26 s | 9 s |
| `01-deterministic-handler` | 2 | 21 s | 7 s |
| `02-governed-tool` | 4 | 29 s | 12 s |
| `03-custom-categories` | 2 | 23 s | 7 s |
| `04-routine-to-telegram` | 2 | 20 s | 5 s |
| `05-your-own-agent` | 3 | 28 s | 11 s |
| `06-testing-your-plugin` | 11 | 30 s | 14 s |
| `07-verify-with-evarness` | 3 | 31 s | 11 s |
| `08-mcp` | 8 | 32 s | 16 s |

## Use one in your IRIS

A plugin example runs from your IRIS home without packaging: copy its directory to
`~/.iris/plugins/<name>/` and list `<name>` under `plugins:` in `~/.iris/profile.yaml`.
To ship one, package it with an entry point in the `iris_harness.plugins` group.
Each README has the details.

Skills (tool packages under `config/skills/`) are covered by the guide
`docs/guides/writing-a-skill.md`.
