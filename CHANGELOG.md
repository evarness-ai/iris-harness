# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

First public release of the IRIS harness.

### Added

- The governed agent harness (`iris_harness`): intent routing, planning, a ReAct
  tool loop, response curation, and tiered LLM routing across local (Ollama, LM
  Studio) and opt-in cloud models.
- The governance kernel every model and tool call passes through: data
  classification, an egress gate, a Fernet vault, hooks, an evaluator, human
  approvals and an audit ledger.
- A plugin SDK (`iris_harness.sdk`) with profiles and three discovery paths
  (built-in, `iris_harness.plugins` entry points, `$IRIS_HOME/plugins`), plus the
  reference plugins `system`, `research`, `code_exec`, `telegram_channel`,
  `web_channel`, `web_push_channel` and `graphiti_import`.
- memris (`memris`), the memory graph of claims the harness remembers with.
- The IRIS API and Governor services, the `iris` CLI, a React web console, and a
  Docker Compose stack with a bundled Ollama.

### Fixed

- A ReAct step's `pre_llm_call` audit row names the model and provider the step
  actually called: the step resolves its model once and the row and the call share
  it, where the row used to carry a snapshot from when the loop was built and the call
  asked the tier router again (a governor downshift or a tier edit could make them
  disagree, and each step cost a second governor acquire). `model` and `provider` now
  appear in `GET /governance/audit`, `iris governance audit` and the trace, and the
  CLI table has a `by` column (tool owner, capability provider or model).
- A core-only start (no `email` extra, no domain plugins) stays quiet without going
  blind. A skill whose tools module genuinely fails to load is logged once with its
  traceback (as before #118's one-line warning), while a skill blocked by a declared
  missing package gets one INFO line naming the extra to install; `gmail-inbox` now
  declares all three Google packages its import chain needs and `requires.extra: email`.
  A `heartbeats.yaml` entry names its owning plugin with `plugin:`: an unmounted owner is
  skipped quietly (one INFO summary, "plugin not mounted"), but a mounted plugin that
  registered no handler, a mistyped handler with no `plugin:`, or no plugin lookup bound
  still warns. Refs #110.
- A typo in a heartbeat's `plugin:` no longer reads as "plugin not installed":
  `config/heartbeats.yaml` declares every owner it may name (`plugins_in_tree`,
  `plugins_external`), a test checks each `plugin:` against them (and the in-tree names
  against the shipped manifests), and an undeclared owner with a missing handler warns.
  A skill blocked by a missing env var, config file or credential, not only a package,
  now logs the same one INFO line and shows `blocked: env:NAME` in `iris skills list`;
  `requires.env_vars` is now read from the process environment (it was checked against
  an empty mapping, so a declared variable always counted as missing). Refs #110.
