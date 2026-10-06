# AGENTS.md

Working notes for coding agents (and people) changing this repository. Read this first;
each folder with its own `AGENTS.md` adds local rules. Human-facing process is in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## What IRIS is

A deterministic, governed agent harness for small local models. A turn runs one pipeline
(intercept, classify, resolve, route, execute, curate, record); every model call and every
tool call passes a governance kernel that classifies data, gates egress, asks for approval
where required and writes an audit row. Everything outside that loop and kernel is a
plugin. Email is the proof domain shipped in release 1.

Python 3.12 or 3.13, Poetry, FastAPI services, a Typer CLI (`iris`), a React web console.

## Layout

- `src/iris_harness/` — the harness core, layered top-down; each package imports only
  layers below it (`[tool.importlinter]` in `pyproject.toml`):
  `server` → `cli` → `playground` → `plugins_builtin` → `testing` → `sdk` → `runtime` →
  `agent` → `services` → `memory` → `tools`/`llm` → `kernel` → `foundation`.
  - `runtime/turn/` — the one turn pipeline; `chat` drains what `chat_stream` yields.
  - `runtime/plugin_host/` — discovery, the plugin registry, the fault boundary, profiles.
  - `agent/tool_runner.py` — `GovernedToolRunner`, the only way a tool runs.
  - `kernel/governance/` — classification, egress, judges, approvals, audit, vault.
  - `sdk/` — the author-facing API a plugin imports. `testing/` — the scripted fake model
    and the governed test harness.
  - `plugins_builtin/` — reference plugins (system, research, code_exec, channels).
- `src/memris/` — the memory graph; imports nothing from IRIS.
- `src/iris_personal/` — the email domain: `email/` (the record), `connections/`, and
  `plugins/email_workflows/`, `plugins/gmail/` (the plugins). It imports only the SDK; the
  core never imports it.
- `config/` — runtime YAML: profiles (`config/profiles/`), model tiers, governance policy.
- `examples/` — runnable plugin examples, run as tests.
- `webui/` — the web console (a thin renderer over the API).
- `tests/` — the suite; mirrors `src/` under `tests/unit/`.

## Rules that matter

1. **Fix the root cause.** No band-aids, no workaround that makes a test pass. Trace the
   failure to its mechanism and fix it there. If the cause or the intended design is
   unclear, say so and ask rather than guess.
2. **Everything goes through governance.** Tools run through `GovernedToolRunner`; model
   calls go through the kernel. `tests/security/test_no_bypass.py` fails a direct call
   site. Deterministic handlers (intercepts) still pass the model-free guards and leave
   an audit row.
3. **Personal data stays local.** Anything that could egress is gated and off by default.
   Never log or trace message contents, credentials or identity.
4. **Import contracts are design, not lint.** `poetry run lint-imports` must pass. Do not
   add an `ignore_imports` entry to get green; move the code to the right layer or add a
   typed seam in the SDK.
5. **Plugins use the stable tier only** (`docs/reference/stable-api.md`):
   `iris_harness.sdk`, `iris_harness.testing`, the `PluginAPI` kinds, the manifest
   schema, the `iris_harness.plugins` entry-point group. Breaking a stable name needs a
   deprecation cycle; `tests/unit/test_stable_tier.py` pins it.
6. **Cover every path.** A behaviour change lands on every surface that reaches it (CLI,
   API, streaming chat, channels, web UI) and is tested on each. Logic lives in
   `src/iris_harness/`; the web UI and CLI only render it.
7. **User state resolves `IRIS_HOME`** (or `IRIS_DATA_DIR`) via
   `iris_harness.foundation.paths`, never `Path.home()` directly. The suite relocates
   both before import; a module that ignores them writes into the real profile.
8. **YAML drives behaviour.** Intents, keywords, profiles and policy are configuration,
   not hard-coded lists.
9. **No AI attribution.** No `Co-Authored-By` trailers for tools, no "generated with"
   footers in commits, PRs, docs or credits.
10. No emojis in code, docs, commits or output.

## Tests

```bash
export IRIS_AUTH_SECRET="test-secret-for-testing" IRIS_DISABLE_WARMUP=1
poetry run pytest tests/unit/<narrow path>
```

- Run the narrowest path that exercises the change; the gate runs the rest.
- `asyncio_mode = "auto"`: never add `@pytest.mark.asyncio`.
- Tests never reach a model, an IRIS port or the network (`tests/conftest.py` fails them);
  use the scripted fake model in `iris_harness.testing` or the offline fixtures.
- A bug fix ships with the test that would have caught it.

## The gate

```bash
scripts/ci_local.sh --fast   # while iterating: ruff, black, mypy, lint-imports, changed tests
scripts/ci_local.sh          # before a PR: full suite, playground smoke, secret + PII scan
```

Hosted CI runs changed-scope tests on Python 3.12 for PRs and pushes; the full suite on 3.12
and 3.13 is on demand (`gh workflow run ci -f full=true`).

Style: Black (100 columns), Ruff, MyPy strict with the Pydantic plugin.

## Commits and PRs

- `<type>(<scope>): <summary>`; type one of feat, fix, docs, style, refactor, test,
  chore, ci, perf, build. The PR title follows the same format; PRs are squash-merged.
- Sign off every commit (`git commit -s`, DCO).
- Commit at task boundaries; keep each PR to one concern.

## Working efficiently

- Search (`grep`) before reading; read large files in windows, not end to end.
- Verify a claim against the code, not against a doc or a comment.
- When unsure about a design decision, surface it instead of deciding implicitly.
