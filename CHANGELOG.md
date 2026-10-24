# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `evarness setup` asks where the Evarness home goes and whether the data alone should go elsewhere, creates the directory, writes `EVARNESS_HOME` and `EVARNESS_DATA_DIR` to the settings `.env` and prints the `export` line for the shell profile. Before, the wizard printed the resolved home and set neither variable anywhere, and a tester who wanted the data on another disk had nothing to go on.
- The Gmail OAuth client guide is on the documentation site, under Getting started ([Gmail: the OAuth client](https://evarness-ai.github.io/evarness/getting-started/gmail-oauth-client/)). It lived only in the repository (`docs/usage-guides/gmail-auth.md`), which the site excluded.
- `evarness email setup` with no mailbox connected says, for each provider, what the login needs first (an app password; a Google OAuth client) and the page that explains it, beside the login command; `onboarding.yaml` carries both. The API's overview and the web console's Setup screen show the same.
- `evarness doctor` reports an Accounts row: what is connected, or, with nothing connected, the two login commands and what each needs first. Information only; it never changes the verdict.

### Changed

- Every `evarness` command reads its settings from named files, in order: the checkout's `.env` in a source checkout, then the Evarness home's `.env` (`foundation/env_files.py`); the shell wins over both. An installed `evarness` used to search upward from its own site-packages directory, so the `.env` that `evarness setup` wrote in the current directory was read only from that directory. The wizard now writes to the home's file, and its closing lines name the home, the data directory and the file.
- `install.sh` accepts Python 3.12 or 3.13, looks for `python3.12` and `python3.13` by name when `python3` is another version, and points Poetry at the one it found. It used to require exactly 3.12 from `python3`, and so refused a supported 3.13. `.python-version` says `3.12` rather than a patch release.
- Every CLI message that pointed at a file under `docs/` (the Gmail login's "client secrets not found", `evarness setup`'s email step, `evarness auth gmail --help`) names the page's address on the documentation site instead; an installed `evarness` has no `docs/` tree.
- Counting connected accounts (`evarness system status`, doctor) no longer creates the database in a home that has none.

### Fixed

- `evarness plugin new` pinned the generated plugin to `evarness>=0.1,<0.2`, so it could not be installed beside 0.2.x: the stable tier's version, which the pin is derived from, was never bumped at 0.2.0. The tier is now 0.2.0, and a test holds the generated pin to the running package version.
- `evarness auth gmail login`'s success line said the account was registered in `data/evarness.db`, a checkout path; it names the Evarness home's database and the next command.

## [0.2.3] - 2026-10-20

### Added

- `pyproject.toml` declares the project under `[project]` (PEP 621): name, version, `requires-python`, the console script, the plugin entry points, the extras and the URLs. The dependency table stays Poetry's, listed as dynamic. `uv sync` in a clone now installs the package and the `evarness` command instead of resolving an empty workspace (#373); `uv build` and `poetry build` produce the same wheel metadata as before, with the license as a `License-Expression`. The version is read from `[project]` by the release workflow, the TestPyPI stamper and the version test.
- `scripts/getting_started.py`, given an Evarness command such as `doctor`, says it is CI's runner of the docs' shell blocks and names the CLI and the two ways to install it. In a clone where `uv sync` installed no `evarness` command it was the nearest runnable thing, and argparse's bare "invalid choice" sent a tester nowhere. The README's contributing section says the same about `uv sync`.
- `evarness governance proof-bundle export --home HOME`: one flag for the ledger and the session logs of one Evarness home (`HOME/governance/audit.db` and `HOME/logs`), the demo's home first of all. `--db` and `--session-logs` still override each one.
- `evarness email demo` ends with the command that verifies the run it just made, ledger and session logs from its own home.

### Fixed

- The install line is `uv tool install "evarness[email]" --upgrade`: on a machine that installed the verifier's pre-release, which shipped as `evarness` 0.1.0a1 on PyPI before the harness took the name, the plain line answered "already installed" and left the verifier's CLI in place, so `evarness doctor` printed its `invalid choice`. The setup-gotchas page names the symptom.
- `scripts/ci_local.sh` runs its playground smoke step in a scratch home it removes afterwards. Run with the developer's environment, the playground wrote its session logs into the developer's own `~/.evarness/logs`, where the owner's proof bundle then read them (#363).
- `proof-bundle export --db` with a ledger inside a home's `governance/` now reads that home's `logs/` as well, instead of the owner's. Before, the demo's ledger was checked against the owner's own chat sessions and `every-call-and-answer-audited` failed for sessions the demo never ran (#362). When `--db` and `--session-logs` name two homes, the export says so.

## [0.2.2] - 2026-10-19

### Changed

- The README, which PyPI renders as the project page, carries its diagram and its recording as images committed in the repository. The 0.2.1 page showed the diagram as Mermaid source and the recording as a broken image: PyPI renders no Mermaid, and its image proxy refuses the release-asset URL. The diagram is a hand-drawn PNG with its Excalidraw source beside the architecture drawing, which is re-exported with the Evarness collection names.
- The README's proof-bundle command after the demo passes `--session-logs` for the demo home as well as `--db`. With `--db` alone the export read the owner's own session logs and failed `every-call-and-answer-audited` against sessions the demo never ran (#362).

## [0.2.1] - 2026-10-19

### Changed

- The README, which is also the PyPI page: shorter, one tagline and one line under it (local by default, deterministic where it can be, governed before action, learns in the open), a diagram with the governance kernel on both the deterministic and the agent path, the hook-point and learning-flag tables replaced by links to the concept pages, and the proof-bundle step after the demo passing `--db` for the demo home, without which the export read the owner's empty ledger. The launch recording is two commands (the demo and the proof bundle) and is attached to each release instead of committed.

### Fixed

- `evarness email demo` no longer says the run recorded signals for the learning loop. A scripted run has no corrections or confirmations to learn from, so nothing was recorded; the summary now says where the loop's signals come from.

## [0.2.0] - 2026-10-19

### Added

- The public preview: the README with the hook lifecycle, the three pillars and the limits list; the Learning concept page; the architecture decision log published on the site (120 records, each with a status line from a pass on 2026-10-18); the contributor on-ramp (`CONTRIBUTING.md`, pre-commit, the DCO check); the suite on `windows-latest` in CI; the launch recording as a VHS tape.

### Upgrade notes

- **IRIS is now Evarness.** The package on PyPI is `evarness` (`uv tool install "evarness[email]"`), the command is `evarness`, the import path is `evarness` (the domains `evarness_personal`), the environment variables are `EVARNESS_*`, and the home is `~/.evarness` (`~/.local/share/evarness` and `~/.config/evarness` for the governance stores). Each 0.1.x name keeps working for one cycle and warns once: `import iris_harness` and `import iris_personal` resolve to the new packages (the same module objects), an `IRIS_*` variable is read as its `EVARNESS_*` name when that is unset, a plugin registered under the `iris_harness.plugins` entry-point group still mounts, and `~/.iris` (and the two governance directories) are read where the new directory does not exist yet. All of it is removed in 0.3.0. The `iris` command is gone: it collided with the InterSystems IRIS instance-control command of the same name. Memory collection names, the `urn:evarness:` ontology ids, the `evarness_doc` tool and the `evarnessd_` device-token marker are renamed with no fallback; a 0.1.x semantic index is rebuilt on first use.
- **Docker and the Windows script read `.env` by the new names.** Compose needs `EVARNESS_AUTH_SECRET` (its `${...:?}` check does not know the old key), so rename the keys in `.env` or add the new ones; `setup-windows.ps1` carries every `IRIS_*` line over to its `EVARNESS_*` name itself, keeping the existing vault master key. Inside the containers the old names are read with a warning as everywhere else.
- The two builtin skills `iris-core` and `iris-tasks` are now `harness-core` and `harness-tasks`, not `evarness-*`: the semantic router embeds a skill's name with its description, so a brand word in the name is noise that decides whether `list_due_today` makes the tool shortlist. A profile or routine that names them by the old name must be edited.
- The three email plugins, the other skills and the console carry the new name; the architecture diagram still says IRIS until it is redrawn.

## [0.1.5] - 2026-10-15

### Upgrade notes

- A server signed under `enforce` as `npx pkg`, `uvx pkg` or `python -m installed_pkg` now reads `invalid` until its args pin a release (`pkg@1.2.3`, `pkg==1.2.3`, `image@sha256:...`) or it runs from a script file or local package, and it is re-signed. Shadow mode only records it.
- A broken `config/governance/mcp-signing.yaml` now stops the MCP bridge with an error naming the file, and a missing one runs signing in `shadow` instead of off. Only `enabled: false` turns it off.
- An MCP result larger than 1 MiB is refused unless the server entry raises `max_result_bytes`.
- A tool name that is not in a server's `tools/list` and not in `governance.tools` is refused once the server has been listed; a never-listed server is now listed first.
- The streaming chat loop stops after three consecutive failed tool calls, as the non-streaming loop already did.
- A call that fails is now recorded as failed (`tool_error` on its audit rows, a write settles `error`): an MCP `isError` result, and about 100 tools that returned `Error:` or failure text. The text the model reads is unchanged.

### Security

- `MCPBridge.invoke_external_tool(..., executor=fn)` now checks the server's launch signature before it runs `fn`. The executor returned before the transport call, where the check lives, so an unsigned, tampered or revoked server ran under `enforce` with `unsigned_policy: deny`; in `shadow` it logged no verdict. Nothing in the harness passes an executor today, so this closes a trap for a host built on the method. Closes M4c of #113.
- An MCP server entry can pin the values of non-secret `env:` entries into its signature with `pinned_env`. Env values were never signed, so re-pointing an upstream URL after `iris mcp sign` still verified and sent the traffic to the new host. A pinned value that changes now reads `invalid` in `iris mcp verify` and is refused before the server starts under `enforce`. A name that reads as a credential (it contains `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `PASSWD`, `CREDENTIAL`, `AUTH`, `DSN`, `BEARER`, `COOKIE` or `PRIVATE`) cannot be pinned, and the error says why: keep those in the vault. The check is on the name, so a secret under another name is not caught. An entry that pins nothing keeps its signature. The plugin `server:` block takes `pinned_env` too. Closes M4b of #113.
- A launch refused for its signature is now a `PermissionError` and leaves one `deny` row. `MCPSignatureError` was a `RuntimeError`, so the HTTP route answered 400 instead of 403, and the refusal came after `PRE_TOOL_USE` had written ten `allow` rows for a call that never ran, with nothing in the ledger saying why it stopped. The bridge now checks the signature before any hook runs; a failing call writes one `pre_tool_use` row from `mcp_signature` (`deny`, with the signature status and signer key id in its payload), the governor records nothing, and `tools/list` is refused before the governor's guard. Part of M4d of #113; the verdict on allowed rows, the no-file default and `iris mcp revoke|remove|list` are not in this change.
- The ledger rows of an MCP call now say what its launch signature was. Every `pre_tool_use` and `post_tool_use` row of a call carries `signature_status` (`verified`, `unsigned`, `untrusted`, `invalid`, or `not_checked` when no signing policy is in force), the signer's `signature_key_id` and `signature_spec_sha256`, the hash of the spec that was checked. Before, the verdict went to a logger warning only, so a call that ran under `shadow` or with signing off left nothing in the ledger saying its server was unsigned. Part of M4d of #113; the no-file default and `iris mcp revoke|remove|list` are not in this change.
- A checkout with no `config/governance/mcp-signing.yaml` now runs MCP launch signing in the shipped default (`shadow`, `warn`) instead of silently off: every call is verified and its verdict is on the ledger rows, nothing is blocked. Only a policy file that says `enabled: false` turns it off. A policy file that cannot be read (a typo in `mode`, bad YAML) now stops the bridge with an error that names the file; it used to turn signing off, which a policy set to `enforce` could not have meant. An unreadable trust store is logged as a warning, not at debug level, and trusts no key. Part of M4d of #113.
- A stdio server whose launcher names no script file is now signed only if the signature binds its code. `python -m pkg` hashes the `.py` files of a local `pkg/` into the spec; `npx pkg@1.2.3`, `uvx pkg==1.2.3`, `pipx run pkg==1.2.3` and `docker run image@sha256:<digest>` are bound by the pin in the args. Before, those launchers verified whatever release they fetched, so the code could change after `iris mcp sign` and still run under `enforce`. `iris mcp sign` refuses an unpinned launcher (`pkg`, `@latest`, a range, a tag) and says what to use; a signed entry that does not bind its code reads `invalid` in `verify`, `list` and at launch, so `unsigned_policy` applies under `enforce` and shadow records it. A module installed outside the server's directory cannot be hashed; run it from a script file. Entries with a script file are unchanged. Closes #287.
- The text of a bridged MCP server's result now reaches the injection guard and the output classifier. A `tools/call` result is `{"content": [{"type": "text", "text": ...}], "isError": ...}`, and both read only a string or a string under a known key, so the guard reported "no scannable text" and the classifier "tool returned no text" on every `MCPBridge` call: an injected instruction in a server's output was never scanned, and the owner's email in it never raised the run's classification. Both now read a result's text through one helper (`kernel/governance/hooks/result_text.py`) that joins the text blocks and skips the others; the guard scans and redacts each block where it sits, so the caller still gets the `content`/`isError` shape, and a `deny` still withholds the result. Part of #112.
- An MCP server's own script is now part of its signed launch spec. For `python server.py` or `node server.js` the signature hashed only the interpreter, so editing `server.py` after `iris mcp sign` still verified and ran under `enforce`. The first argument that names a local file is hashed into `entrypoint_sha256`; an entry with no such argument keeps its signature. `iris mcp sign` now prints what it covers, and says when the server's code is not covered (`-m pkg`, `npx pkg`). Closes M4a of #113.

### Added

- `FailedResult` in `iris_harness.sdk.tools`: a tool returns `FailedResult("Error: ...")` (a `str`) to report that a call failed without raising. The call is recorded as failed (`ToolResult.ok` false, `tool_error: ToolReportedError` on its audit rows, the failure streak counts it) and the model still reads the text; unlike an exception it does not count against the plugin as a fault. The class moved from `agent/tool_runner.py` to the kernel (`hooks/tool_payload.py`) so tools, memory, services and runtime can use it too; `agent.tool_runner.FailedResult` still resolves. Plugins that return `Error:` text keep working: the loop's `Error:` prefix check stays until the convention has shipped. Part of #301 (the tools that return `Error:` text move in a follow-up).
- `max_result_bytes` on an MCP server entry and on a plugin `server:` block (default 1 MiB). A `tools/call` or `tools/list` result whose JSON is larger is refused with an `MCPTransportError` that names the size and the setting, instead of being handed to the hooks and the model whole; a stdio server is also cut off once it has sent that much plus the JSON-RPC envelope without completing a message, so a server cannot make the exchange hold up to 32 MiB first. A 3 MB result passed the bridge uncapped before. Raise the limit on an entry that legitimately returns more. Closes #298; keeping a session alive is #299.
- `iris mcp list`, `iris mcp revoke` and `iris mcp remove`. Revoking a key was a hand edit of the trust store, and removing a server left its key trusted with nothing saying so. `list` shows each server's verdict and the trust store's keys; `revoke <key_id>` (or `--signature <sig>`) distrusts a key, keeps it on record and names the servers it now refuses; `remove <server>` deletes the entry, says when its key signs nothing else, and with `--revoke-signature` stops the same signed entry being pasted back. Closes the CLI item of #113.
- `TurnAuditRow.tool_error`: on a `post_tool_use` row of a call that failed, the class name of what the tool raised, or `ToolReportedError` for a result it marked failed (an MCP `isError: true`); `None` on a call that succeeded and on every other row. A plugin's test had to read the ledger's raw payload to see it. The field is appended with a default, so the stable tier only grows (`tests/unit/test_stable_tier.py` pins the order). `docs/architecture/unified-governance-layer.md` 8.5 now says that, once a server has been listed through the bridge, a tool absent from its last `tools/list` and from `governance.tools` is refused before any hook (`mcp_undeclared_tool`, one `deny` row, a 403 on the HTTP route), that the #180 default still applies before any listing, and that an `isError` result is recorded as a failed call. Closes #297.
- `iris_harness.testing.plugin(manifest=..., server=[command, *args])` mounts a `trust: mcp` plugin under the governed harness with no `setup`, so a plugin author's own CI can run a scripted turn against it and assert on the audit rows. `harness(seed={...})` writes files into the home before the runtime is built. M3 of #111.
- `trust: mcp` plugins are mounted. A manifest with a `server:` block (command, args, env, `inherit_env`, `signature`) and declared `tools:` registers each tool into the governed tool pool, so a call gets one `PRE_TOOL_USE`/`POST_TOOL_USE` pair, the stamped caller and the manifest's effect and approval rules; its output is `content: external`. The server starts per call after the launch-signature check, and a profile cannot flip how a plugin runs. Before, such a plugin was marked unsupported and never started. M2 of #111; 
- `iris_harness.sdk.mcp` puts MCP server signing in the stable tier: `build_server_spec`, `sign_server_spec`, `verify_server_signature`, `TrustStore` and the types they use. A plugin author can sign and verify a `trust: mcp` server with `iris_harness.sdk` alone. This is the stable-tier part of M1 in #111; the plugin host for `trust: mcp` (M2, M3) is not built.

### Fixed

- About 70 more tool failures are now recorded as failed calls. After the `Error:` text was moved to `FailedResult` (#308), the same tools still returned a failure in other words as an ordinary result: an exception caught and handed back (`memory_forget failed: ...`, `search_inbox failed: ...`, `Stock quote unavailable: ...`), a capability that is not configured (`... unavailable (memory store not configured)`, `Wiki unavailable.`), a target that does not exist (`No pending action ...`, `No email found matching ...`, `No recommendation #N to promote`), bad input (`'pick' must be an integer`, `No category matches ...`) and a refused action (`Not sent: ...`, `Not restored: ...`). Each now returns `FailedResult`, so its audit rows carry `tool_error`, a write settles `error`, and the failure streak counts it by outcome. The text the model reads is unchanged. Empty answers stay successes (`Nothing stored matches that.`, `No emails in <category>`, `Nothing to compact`), as do the pre-approval validation messages of `trash_email`. A new scan fails on any `except` block that returns a `failed` or `unavailable` message without the wrapper. Part of #309; the `Error:` prefix check stays.
- The built-in tools and the email tools now report a failed call with `FailedResult` instead of returning `Error: ...` text. About 30 tools returned a usage or refusal message (a missing `query`, an unknown `iris_doc` name, "no mail provider can send from this account") as an ordinary result, so the call was recorded as a success: no `tool_error` on its audit rows, a write settled `completed`, and only the loop's text-prefix check counted it toward the failure streak. The text the model reads is unchanged. Covers `stock_quote`, `memory_graph`, the `react_tools` set (`memory_*`, `wiki_search`, `iris_doc`, `recall_conversation`, `promote_recommendation`, `propose_skill_from_sandbox`), `research`, `code_exec`, `search_docs`, and the email tools (`send_email`, `trash_*`, `restore_email`, `find_attachment`, `list_by_category`, `analyze_inbox`). A new scan test fails on any function that returns `Error:` text without the wrapper. Tools that report a failure in other words (`... failed: ...`, "takes at most N") are not converted here. The loop's `Error:` prefix check stays for plugins that still return that text. Part of #301.
- The streaming chat loop now stops after three consecutive failed tool calls, as the non-streaming loop already did. `tool_failure_streak` was recorded on `run_stream` but not enforced (ADR-0107 left failure streaks to the local guards), and no local guard counts failures: the stall guard trips only on an identical action and input repeated, so a model that kept calling a failing tool (an MCP server answering `isError`, a tool that raised) with different arguments ran to `max_iterations`, ten calls by default. It now halts at the streak with the existing interruption message (what ran, why it stopped, the next step left to the user); an identical repeat still stops at the stall guard first, so no run is halted twice. A model that would have recovered on its fourth attempt is stopped at the third. The threshold stays the signal's default of 3, like the other evaluator signals. Closes #300.
- A tool call that fails now counts toward the loop's `tool_failure_streak`, whatever its text says. The loop told the evaluator a step failed only when its observation started `Error:`, so an MCP tool answering `isError: true` (its text starts `error:`) and a tool that raised (`Tool error: ...`) were never counted, and a model that kept calling either was never stopped by the streak. The step now carries the call's outcome from the governed runner (`ok` false, and not a result governance withheld) and `PostStep` reports it as `tool_error` on the sync and the streaming loop alike; a tool that returns a usage error as ordinary `Error: ...` text still counts. The streaming loop records the streak but, as before (ADR-0107), does not halt on it. Part of #295.
- The stdio MCP transport no longer hangs on servers that behave like ordinary ones. It read through a buffered line reader the deadline could not see, so a server that wrote half a line blocked well past `timeout_seconds`, and one stdout banner line made it read the next JSON line as a header. It also read stderr only after a timeout, so a server writing 400 KB to stderr filled the pipe and stalled, and it counted id-bearing messages instead of checking ids, so a stray message ended the wait and a later lookup raised a bare `KeyError`. Reads now go through the exchange's own buffer under one deadline; a line that is not JSON-RPC (a banner) is skipped; stderr is drained on a thread and its tail kept; a server `ping` is answered and other server requests are declined; a response to an id the session never sent raises `MCPTransportError` naming it; and a server that exits or times out reports its exit code and the end of its stderr (a `ValueError` subclass, so existing callers still catch it). `Content-Length` framing is still read, under the same deadline. Part of #112 (M6, M7, M8, M9, M10).
- An MCP tool call the server answers with `isError: true` is now recorded as a failed call, and the bridge no longer forwards a tool the server never declared. Nothing read `isError`: the result went back as a success, its `post_tool_use` rows looked like any other, and the side-effect ledger settled a write as `completed`. The result is still returned, but on the bridge and for a `trust: mcp` plugin the call's `post_tool_use` rows now carry `tool_error: ToolReportedError`, a write-ahead row is settled `error`, `iris mcp serve` answers `isError` for it, and the server's text is still wrapped as untrusted. A tool that raised now has its exception class on those rows too. Any tool name the caller sent went through the `pre_tool_use` hooks to the server. After the server has answered `tools/list` through the bridge, a name it did not list and `governance.tools` does not declare is refused before any hook runs: a `PermissionError` (HTTP 403) and one `pre_tool_use` `deny` row from `mcp_undeclared_tool`, and the server is never reached. A `trust: mcp` plugin already reached only the tools its manifest declares. Part of #112.
- The MCP bridge no longer forwards an undeclared tool name to a server it has not listed. The check against the server's `tools/list` ran only after something had listed the server through the bridge, so before that a name `governance.tools` does not declare reached the server's `tools/call`, and with `undeclared_tools: read` it ran with no approval. A call naming an undeclared tool on a server the bridge has not listed now lists the server first, once for the bridge's life, through the governed listing (the signature check and the governor's guard, with the call's own approval), then applies the same refusal: a name the server does not list gets a `PermissionError` (HTTP 403) and one `pre_tool_use` `deny` row from `mcp_undeclared_tool`, and the server is never called. A listing that fails or is refused refuses the call the same way (one `deny` row from `mcp_undeclared_tool`; refused for want of approval, the call is reported as needing approval, which the HTTP route answers with its 403) and is tried again on the next call. A tool declared under `governance.tools` needs no listing, and `undeclared_tools: read` still sets the effect of a tool the server lists. Closes #296.
- A stdio MCP call no longer waits out its timeout when the server answers fast. The reader was buffered, so both replies arrived in one read and `select` then waited for a byte already in the buffer. Part of #112.

## [0.1.4] - 2026-10-14

### Security

- An MCP stdio server no longer starts with IRIS's whole environment. The bridge copied `os.environ` into the child, so a server you configured held every `IRIS_*` setting, `IRIS_VAULT_MASTER_KEY` when it was set as a variable, and every key or token in your shell. It now starts with `PATH`, `HOME`, `USER`, `LOGNAME`, `LANG`, `LC_*`, `TERM`, `TMPDIR`, `TZ`, the names its entry lists under the new `inherit_env`, and its own `env:`. `inherit_env` refuses `IRIS_*` names, and its names are part of the signed launch spec (an entry that lists none keeps its signature). A server that relied on an inherited variable stops seeing it until you list it. This closes the environment part of #113 only; the bridge is off by default and this is not an operating-system boundary. (#113)

### Changed

- `README.md` and the setup gotchas guide now say what to do when the install picks an unsupported Python (`uv tool install --python 3.13`), that `uv sync` installs nothing because IRIS is a Poetry project, and how to recover when the email judge's model was not pulled (`ollama pull`, then `iris email setup --restart` on 0.1.3 or later).

## [0.1.3] - 2026-10-13

### Fixed

- `iris email setup` no longer judges every email against a model Ollama has not pulled. Ollama answers `404 model not found` for each call; the judge took that for a per-email failure, printed a traceback and recorded an Unsure verdict for every email, then asked to change the mailbox. It now stops at the first 404 and says `model 'qwen3.5:4b-q4_K_M' is not pulled: run `ollama pull qwen3.5:4b-q4_K_M` on the Ollama host`, with no verdicts recorded, so the next run judges the same mail.
- Emails that 0.1.0 to 0.1.2 judged against an Ollama model that was not pulled were saved as Unsure verdicts carrying the 404 and never judged again, so `ollama pull` did not repair them. `iris email setup` now judges those again, and `iris email judge --retry-failed` re-judges every email whose last verdict was an error, before any new mail. They are judged as the visible mail they are: never held again, never released twice.

### Changed

- `README.md` now says the MCP bridge is experimental and off by default, and links the issues that track its known gaps (#111, #112, #113, #114).

## [0.1.2] - 2026-10-11

### Fixed

- `iris --version`, the web console's version line, the four services' API docs and the MCP `clientInfo` and `serverInfo` now report the installed version. Each held its own hard-coded copy, so 0.1.1 printed `IRIS 0.1.0` (and the MCP server reported `0.2.0`). They all read `iris_harness.__version__` now, and `tests/unit/test_version.py` pins it to `pyproject.toml`.

## [0.1.1] - 2026-10-11

### Fixed

- Reconnecting an account no longer fails on macOS with `PasswordSetError: Can't store password on keychain: (-25244, 'Unknown Error')`. That error is the keychain refusing to let one program edit an entry another created, which happens after a reinstall under a different Python (`uv tool` after `poetry`, or the reverse). Saving an account token (IMAP and plugin secrets, the Gmail and Google OAuth tokens) now deletes and rewrites the entry on that one error. The vault master key is never touched: its service is refused even when asked. If the delete fails too, the error names the `security delete-generic-password` command that clears the entry. (#261)
- `IRIS_GOVERNOR_BASE_URL`, which the health check reads, is now declared in the settings catalog and listed in the environment-variable reference. (#262)

### Changed

- `README.md` now says to connect an account (`iris auth imap login` or `iris auth gmail login`) before `iris email setup`, and to pull the email judge's model, which `iris doctor --fix` does not. Without the connect step `iris email setup` stops at "No mailbox is connected yet". (#260)
- Dependency updates, all minor or patch: `cryptography` 50.0.2, `langchain-openai` 1.6.7, `black` 26.10.0, `ruff` 0.16.10, `trafilatura` 2.3.0, `pypdfium2` 5.14.0, `openinference-instrumentation-langchain` 0.1.78, `markupsafe` 3.0.4 and `platformdirs` 4.12.3; in the web console `vite` 8.3.2, `lucide-react` 1.52.0, `@tanstack/react-query` 5.104.1, `@vitejs/plugin-react` 6.1.2, `postcss` 8.5.29 and `typescript-eslint` 8.71.0. (#241, #243)

### Added

- A contributor quickstart in the README, a setup gotchas guide (`docs/guides/setup-gotchas.md`), and a concept-to-code table that labels the architecture diagram as conceptual. (#257, #258, #259)

## [0.1.0] - 2026-10-11, added after the 2026-10-07 notes below

These changes were merged after the notes in the next section were written and before the
0.1.0 tag was cut, so they shipped in 0.1.0. They were kept under "Unreleased" until now.

### Changed

- The sandbox egress proxy (`egress_proxy.py`) now listens on `127.0.0.1` by default, not `0.0.0.0`. The sidecar the harness starts is unaffected: it is started with `IRIS_EGRESS_BIND=0.0.0.0` because the sandbox reaches it from a sibling container. Anyone who runs `egress_proxy.py` by hand or from their own compose file and relied on it listening on every interface must set `IRIS_EGRESS_BIND=0.0.0.0` (or a specific address). A value that is not an IP address is refused, and a non-loopback bind logs one warning.
- `POST /memory/export` takes `name`, a single folder name, instead of `out_dir`, an absolute path, and writes that folder under the new `IRIS_EXPORT_DIR` (default `~/.iris/exports`, created owner-only). A name with a separator, `..` or more than 64 characters, one that resolves outside the folder through a symlink, or one that is an existing file is refused with 400. A caller that sent `out_dir` now gets 422: send `name` instead. To export straight into an Obsidian vault, set `IRIS_EXPORT_DIR` to the vault's parent folder and name the vault's folder, or use `iris memory export --to PATH`, which is unchanged and writes wherever you type.

### Added

- Audit compaction now keeps what the ledger knows and accounts for what it moved (issue #134,
  stage 4b). The Parquet archive keeps the identity and sequence columns (`id`, `record_id`,
  `session_id`, `turn_id`, `call_id`, `parent_call_id`, `attempt`, `replay_of`,
  `resumed_from_run`, `writer_id`, `writer_seq`, `kind`) and archives the ledger's own
  `writer` and `gap` rows. Each run writes and fsyncs its chunks, then ONE transaction inserts a
  `compaction` marker (chunk files, SHA-256, row counts, id range, each writer's lowest and
  highest sequence number and count) and deletes exactly the archived rows by id. A crash between
  the two leaves chunks no marker names: the next run adopts them (their rows are still in the
  ledger) or moves them to `<archive>/.orphans/` (never deleted); a chunk written by an older
  release, or any file it does not recognise, is never touched. `audit_archive` de-duplicates a
  row found in both tiers or in two chunks by `record_id`, hides the ledger's own rows by default
  (`iris audit query|export --include-store-rows` shows them), and reads old and new chunks
  together. `iris audit verify` checks every marker against its chunks. Compactions are
  serialised by a lock in the archive directory. Known limit: a spool line that survives a crash
  between being applied and being removed can re-insert a row archived in the meantime; the view
  collapses the copy.
- A session can be replayed from the stored records, and what is missing is named (issue #134,
  stage 5). `iris audit replay --session S` and `GET /governance/replay?session=S` rebuild one
  session's ordered timeline from the audit ledger (hot and archive), the approval queue, the
  side-effect ledger and the session log, with its calls hung on their parents, and list the gaps
  found: a sequence number no store holds, an archive chunk that no longer matches its marker, a
  call that started and never settled in a process that is gone, a parent or held call that does not
  exist, a witness with no counterpart, a settle with no start, a record stored twice with
  different content. Rows from before records had ids, a writer that never closed and a call still
  running are notes, not gaps. The command exits 1 on a gap (`--no-fail` for scripts) and the
  endpoint is capped (rows and seconds) and says when it cut the read short. `Harness.audit_gaps()`
  still returns a list of strings and is empty for a healthy turn; it now also lists what the
  replay proves lost. `writer.start` rows now carry the process's start time so a recycled pid is
  not mistaken for a running writer (older rows fall back to the pid alone). The egress witness is
  not part of this stage.

### Fixed

- Opening a memris database no longer fails when several processes open a new or older file at the
  same instant. `SQLiteGraphStore` read the schema version and then wrote it (or upgraded the tables)
  in a plain transaction, so two processes could both insert the version row
  (`UNIQUE constraint failed: memris_meta.key`) or both upgrade, and a brand-new file could refuse
  the second process's switch to WAL ("database is locked"). The initialisation now takes the write
  lock first (`BEGIN IMMEDIATE`) and waits out a concurrent WAL switch. The earlier statement that
  memris was measured and not affected (#223) held only for a database that already had its version
  row; an empty one was not covered.

### Security

- An unexpected failure in `/chat/stream` or a web slash command now tells the client the exception class and to see the server log, not the exception text. The full traceback is logged. Curated errors (validation, removal, allow-list) are unchanged.
- Request-derived values in log lines are escaped (new stable `iris_harness.sdk.logging.log_safe`); the shared ingress line escapes method, path, source and session id.
- Session ids that contain a path separator or leave the session log directory are refused.
- Copilot token fragments are no longer logged.

## [0.1.0] - 2026-10-07

First public release of the IRIS harness.

### Added

- The audit ledger can show what is missing (issue #134, stage 4a). Every process that writes to
  `audit.db` is a writer with its own id and numbers its rows before writing them
  (`writer_id`, `writer_seq`), so a write that fails leaves a hole in the sequence. A row the
  database refuses is kept in a local spool (`audit-spool.jsonl`, beside the database, `0600`) and
  put into the ledger on the next successful write or restart; the next row that lands also records a
  `gap` row (the missing numbers and the failure class, never a message). A call that guards an
  effect (a write or destructive tool, a cloud-tier model call, an outbound request) is refused
  when neither the database nor the spool could keep its row. `writer.start` / `writer.close` rows
  bound each writer's life; a writer with no close ended without saying so. A spool line can add
  a missing row but cannot rewrite one that exists, and malformed lines are set aside and counted.
  The lost and waiting counts show in System Health (red when a row was kept nowhere),
  `iris system status`, `GET /governance/state` and `GET /governance/audit` (`write_health`), and
  the Governance screen. A database an earlier release created gets the columns when it is opened
  (several processes opening it at once are safe); old rows are never backfilled. The archive
  compaction leaves the ledger's own rows in place until the marker work lands (stage 4b).

- A stored conversation summary records whether it absorbed third-party text (issue #145,
  summaries). `conversation_summaries.has_external` is 1 once any external-origin turn was folded
  into the summary (and stays 1: a paraphrase cannot be un-marked), 0 when every folded turn is known
  not to be external, and NULL (unknown) for every summary written before the column. A flagged
  summary comes back into a prompt inside the untrusted-content envelope, as well as scanned, from the
  session summary, `memory_search` over sessions and `recall_conversation`'s summary fallback;
  unflagged and unknown summaries are scanned as before. A database an earlier release created gets
  the column when it is opened (several processes opening it at once are safe).

- A stored assistant turn records where it came from (issue #145, steps 2-3). The loop notes
  whether the run read third-party text (a `content: external` tool's result) before answering,
  and `conversations.turn_origin` stores `external` or `internal` on the assistant row (NULL
  is unknown: every row written before this change, never backfilled, and any answer the loop
  did not produce). A turn marked `external` comes back into a prompt inside the
  untrusted-content envelope, as well as scanned, from the conversation window, the history
  reload after a restart, the related turns from other sessions, `recall_conversation`,
  `memory_search` over sessions and the intent router's recent-turns block. Internal and
  unknown turns are scanned as before; the owner's own turns come back verbatim. A database an
  earlier release created gets the column when it is opened (several processes opening it at once
  are safe).

- `iris_harness.sdk.persistence.ensure_columns(db_path, table, {column: declaration}, *, indexes=(),
  on_added=None)` adds the columns an older table lacks, safely when several processes open it at
  once: a plugin that reads `PRAGMA table_info` and then runs `ALTER TABLE` itself loses that race
  with `duplicate column name` (issue #201). It looks first, then takes the write lock on a
  connection of its own (`BEGIN IMMEDIATE`) and decides again inside it; `on_added` runs in that
  transaction, only in the process that added a column, for a one-time backfill. It is the harness's
  own function, re-exported as a new stable name.

- The audit stores carry an identity in real columns (issue #134, stage 3). `audit_log` gains
  `record_id` (minted by the store, unique), `session_id`, `turn_id`, `call_id`,
  `parent_call_id`, `attempt`, `replay_of` and `resumed_from_run`; the side-effect ledger gains the
  call's identity and an append-only `side_effect_events` table; the approval queue gains
  `step_id` and `turn_id`; the router and governor audit tables gain their ids. A database an
  earlier release created is migrated when it is opened: columns are added, nothing is rewritten
  or backfilled, a row without a `record_id` is one written before identity (an `audit_meta` row
  records the boundary), and several processes opening the file at once are safe.

- The plugin raw-network lint also names `paramiko`, `boto3`, `openai`, `redis`, `pymongo` and
  `aiosmtplib` (issue #175, item 4). `NETWORK_MODULES` grows by six entries; no name is added or
  removed, so the stable-tier snapshot (`tests/fixtures/stable_tier/names.txt`) is unchanged. A
  plugin that imports one of them now fails `check_network_imports` the way one that imports
  `httpx` does; route the call through `api.http`, or declare why it cannot be. The scan finds
  one first-party file, `research/cache.py` (the opt-in Redis cache, a plain TCP connection to
  the operator's own server), pinned as listed debt with its reason in the research manifest and
  `plugin-egress.md`.

- An audit row says where its call sits (issue #134, stage 2): `parent_call_id` (the governed
  call it ran inside: a tool a tool called, a capability a tool called, a bridged MCP call),
  `attempt` (2 on the approved re-execution of a held call, with `replay_of` naming the held
  attempt), `turn_id` (the chat turn the row was written in) and `resumed_from_run` (the run
  id, on the rows a halted run writes after it is re-entered). The harness writes them, never a
  caller, an argument or a hook. `TurnAuditRow` gains `call_id`, `held_call_id`, `turn_id`,
  `parent_call_id`, `attempt`, `replay_of` and `resumed_from_run`, all optional and `None` on a
  row they do not describe or on one written before calls carried an identity. No existing
  field, `run_id` included, changes, and no database migration is involved.

- An operator can declare what an MCP tool does: `governance.tools: {<name>: {effect:
  read|write|destructive}}` under a server in `mcp-servers.yaml` (issue #102). A destructive
  tool is refused before the server is reached unless `invoke_external_tool` is given
  `approved_by`, an approved queue row that pinned exactly that call, and it leaves a pending
  write-ahead ledger row before it runs. A tool not listed behaves as before.

- The governed HTTP client's response cap is configurable per plugin (issue #175). The manifest
  key `egress.max_response_bytes` sets the most decoded body bytes a response may have: default
  10 MiB, lowering is always allowed, and the harness ceiling is 64 MiB (a larger value fails
  manifest validation, so the plugin does not mount). A cap other than the default is recorded as
  `cap` on every `pre_egress` row and shown by `iris plugins`. Adding a manifest key is not a
  breaking change to the stable surface; the key sits under `egress`, which is already in it.

- A capability method can declare `sends_to` (`MethodSpec.sends_to`, as a tool does), and
  `CapabilityCall` carries it: every governed call of the method stamps where its arguments go
  at `PRE_TOOL_USE`, so the owner-PII guards read them. `weather.forecast` declares
  `external_service` for its `location` (issue #100).

- Health shows the model guard when it is not doing what the owner may believe (issue #136):
  a yellow `governance` row when `IRIS_GOVERNANCE_PROMPT_GUARD` is on but its classifier cannot
  run (packages missing, weights not in the local cache, or a load already failed), and one when
  it is off while tools that return third-party text are mounted, naming how many and which
  plugins. Silent when the posture is fine. The probe never loads the model, and the always-on
  floor is named as still running. `GET /governance/state` gains a `model_guard` object from the
  same probe. New `CheckKind.GOVERNANCE`; the web Health screen gains Plugins and Governance
  groups (plugin rows had no group there).

- `ToolResult` (what `api.tools.call` returns) gains `external`, `source` and `tool`, set by the
  tool service from the tool's declared `content`, and `for_model()`: the text inside the
  untrusted-content envelope when the tool is external, unchanged when it is not (issue #147).
  `result.text` stays the owner-facing form. The conformance suite adds a `content` check: for
  every external tool with an example call, the result must say it is external and
  `for_model()` must wrap it. `wrap_external_content` and `for_model()` share one
  implementation (`wrap_scanned`).

- The owner is told when text they are reading was redacted by the external-content floor, and
  can allow known false positives (issue #139). A shared turn stage appends one plain sentence
  once to an answer that shows the redaction marker (generated and deterministic, `chat` and
  `chat_stream`, before the turn is recorded); a brief and a directly answered skill add it
  where they render. `iris governance redactions` and `GET /governance/redactions` list what
  the floor cut by pattern id, count, tool and source, never the text. The allow-list
  (`config/governance/external-content.yaml`, `iris governance allow add|remove|list`) names one
  floor pattern id and a namespaced source (`plugin:`, `skill:`, `mcp:` or `core:`), optionally
  narrowed to a tool: no wildcard, never a hidden-character pattern,
  optional `until`, ignored when expired, and a file with a refused entry allows nothing. Each
  edit is a ledger row, each use is recorded on the floor's row, and `GET /governance/state`
  gains `external_content_allow`. `ToolResult.for_model()` honours the allow-list for the scope the
  tool service stamped, so an allowed source's text is not re-redacted there. A plugin cannot edit it, and the plugin name `system` (the stamp
  of the core's own tools) is reserved for the builtin: any other plugin of that name fails to
  load.

- A model-called memory write after the run read outside text needs the owner's approval
  (issue #149). A run is tainted once a step ran a tool declared `content: external`; in a
  tainted run, `memory_correct`, `memory_forget` and `memory_restore` (the list is
  `config/governance/taint-policy.yaml`, `approval_when_tainted`) are held on the existing
  approval card, which says why. The loop derives the taint from the run's recorded steps, so a
  run resumed after an approval keeps it; an ordinary correction in a run that read nothing
  from outside is unchanged, and the next message starts clean. New defaulted
  `ToolCall.tainted`. A missing or malformed file falls back to the three memory writes. Taint
  comes only from a tool step declared external in the run: external text that reaches the prompt
  with no tool step (recalled memory, injected retrieval context, stored text from earlier
  turns) does not taint it.

- `search_docs`: the core can search its own shipped documentation (architecture, concepts,
  guides, reference, usage-guides) by keyword at section level, as an internal read tool of
  the `system` plugin. The corpus is an allow-list in `config/docs_search.yaml`; it cannot
  reach the identity files, the vault, `.env` files, the owner's data or any store, and a
  document that classifies `secret` is dropped. An install without the docs says so.

- Plugin manifests can declare `egress:` (issue #103): the hosts a plugin's code may
  contact through the SDK's governed HTTP client (#103b; see below). The host
  grammar refuses IP literals in any spelling, `localhost`, newlines, repeated dots,
  suffix-only wildcards (`*.com`) and over-long hosts; `iris plugins show` and
  `--dump-config` print `none declared (raw network calls by this plugin are not governed)` for a plugin
  without one.

- Governed outbound HTTP for plugins (issue #103, part 2): `api.http` (and
  `iris_harness.sdk.http.current_http()` for a declarative plugin) sends a request only to a
  host the plugin's manifest `egress:` declares, fires the new `PRE_EGRESS` / `POST_EGRESS`
  hook points and writes a ledger row per request (host, port, method, plugin, tool, run,
  status, bytes, duration; never a path, query, header or body). A host that is not declared,
  an IP literal, a missing kernel or a missing `plugin_egress` hook is refused with
  `EgressDenied`. This governs calls made through that client only: it does not stop a plugin
  that opens its own socket, and no shipped plugin uses the client yet. Also new:
  `testing.fake_http`, `testing.check_network_imports` and a conformance check `egress`.
  Each request has one total time budget (default 10 s, at most 60) and a 10 MiB decoded-body
  cap, ignores proxy and netrc environment variables, refuses `Host` and `Proxy-*` headers and
  names such as `localhost`, `*.local` and `*.internal` (even under `open_web`), connects only
  to a checked public address (the name is resolved once), records a malformed URL as a
  denial, and is not sent when its `pre_egress` ledger row cannot be written. A name an
  operator or attacker points at an internal address is refused; see
  `docs/architecture/plugin-egress.md` for the limits. A request is made only inside a
  governed tool or capability call and acts only for the plugin whose tool is running (a
  tool of plugin `evil` using `GovernedHttp("weather")` is denied and recorded against
  `evil`); one made outside any call, such as from a thread the tool started, is denied.
  `EgressDenied` is a `RuntimeError`, not an `OSError` (a stable name whose base changed
  before release). `check_network_imports` also reports attribute use of an imported
  package (`urllib.request.urlopen`), `asyncio.open_connection` / `start_server`, `httpcore`,
  `h11` and `http.server`, and an unparsable file as a finding.
  The response body is decoded by the client with a bounded decompressor, never by httpx:
  it asks for `identity`, accepts at most one `gzip` or `deflate` layer, and refuses any other
  encoding (layered, `zstd`, `br`) before reading it, so a compression bomb cannot exceed the
  10 MiB cap in memory. A `Host`, `Proxy-*`, `Connection`, `Upgrade`, `TE`, `Transfer-Encoding`
  or `Content-Length` request header is refused in any spelling (dict, pairs, bytes keys).

### Changed

- **Default flipped (issue #180): an MCP tool nobody declared is now treated as
  `destructive`.** Before, a tool with no `governance.tools` entry ran with no effect
  stamp, no write-ahead row and no approval. Now every such call is refused before the
  server is reached unless `approved_by` names an approved queue row that pinned exactly
  that call, and it leaves a write-ahead row. **What breaks:** every current MCP server
  whose tools are not listed under `governance.tools` will need an approval on each call
  (over HTTP it is refused with a 403, since that surface grants no approval). **The
  one-line fix:** for a server that only reads, add `governance: {undeclared_tools: read}`
  to its entry in `mcp-servers.yaml`; or declare each tool under `governance.tools`. After
  `tools/list` the bridge logs a warning naming the undeclared tools. A server's own
  `readOnlyHint` never relaxes the default.
- **Default flipped (issue #104): text a tool, capability or MCP server declares
  `content: external` is now marked and scanned on every install.** The new
  external-content floor needs no model, no weights and no network. An external result
  reaches the model inside an `<external_content source=... trust="untrusted">` envelope
  (plugin or core code calling through `api.tools` gets the redaction only, no envelope),
  and a short list of deterministic patterns (instruction overrides, chat-template and
  tool-call syntax, exfiltration instructions, hidden characters) is redacted with a ledger
  row naming the pattern ids, the tool and the source, never the text. It is on by default
  (`IRIS_GOVERNANCE_EXTERNAL_CONTENT_FLOOR`, a plain boolean: unset, blank and any
  unrecognised value leave it on (an unrecognised one logs a warning), only
  `0`/`false`/`no`/`off` turns it off, with a warning; with the floor off, text a third party wrote reaches the model unmarked and
  unscanned) and acts at `POST_TOOL_USE`, so the agent loop, `api.tools`, `iris mcp serve`,
  the MCP bridge and capability results all get it. A text that quotes an attack phrase is
  redacted too: see "The external-content floor" in `docs/concepts/governance.md` for the
  pattern list and its limits. The model guard (`IRIS_GOVERNANCE_PROMPT_GUARD`) is
  unchanged: opt-in, shadow-first, fail-open.
- **`code_exec` output is treated as external content (issue #140).** The tool stays
  `effect: read`, behind its Docker mount and sandbox limits, and its egress allowlist is
  unchanged; what changed is that what a script printed is no longer assumed clean (a
  script can print a page it fetched). The tool is declared `content: external`, so the
  governed loop marks and scans its result; and the paths that do not run through the
  loop's runner apply the same tripwire (`iris_harness.sdk.content.redact_external_content`,
  new): the intent route's answer and trace, the nested planner's view of `run_shell`
  stdout and stderr (inside the envelope), the session log's `tool_run` record, and a
  code_exec lesson before it is stored and again before it is re-injected into a planner
  prompt. The streamed answer is scanned with a sliding overlap (the last two lines are held
  back and rescanned with each new line, a long line is force-scanned at 64 KB), so a phrase
  split across up to two line breaks is caught; the planner's progress text, the command in
  the activity hint, trace and log, and artifact file names (including the "Artifacts:"
  block on every path) are scanned too. These calls honour
  `IRIS_GOVERNANCE_EXTERNAL_CONTENT_FLOOR` (off: verbatim) and a run writes at most five
  counts-only ledger rows however many lines match (later matches are redacted and logged
  once). A redaction cut by the 64 KB forced flush also drops the rest of its sentence when
  that arrives, instead of emitting it raw. Known limit: the session log's `llm_call` record of
  the planner's raw prompt and reply is a log, not a prompt, and is not scanned.
- The retrieved-content guard's `guard unavailable` ledger row now records the tool, how
  many segments went unscanned, and the classifier's backend and detail. Docs and manifest
  comments that said external content is always scanned now say what is on by default.
- **Behavior change:** `config/governance/threat-detection.yaml` no longer has a `fail_mode` key. It was
  parsed and read by nothing, so `closed` promised what never happened; a guard that
  cannot run still lets the text through and writes a `guard unavailable` row. A config
  override that still sets `fail_mode` now stops startup: when a guard that reads the file is
  requested (`IRIS_GOVERNANCE_PROMPT_GUARD`, `IRIS_GOVERNANCE_INPUT_SAFETY` or
  `IRIS_CURATOR_OUTPUT_SAFETY`) the build raises an error that names the file and the key and
  says to delete the line, instead of running on with the guards silently off (there is no
  deprecation shim). Other config errors still turn the guards off with a warning, as before.
- New stable name: `iris_harness.sdk.content.wrap_external_content(text, *, source, tool=None)`
  applies the floor's tripwire and envelope (the kernel's own implementation) to external text
  that plugin code puts into a prompt of its own. Idempotent and offline.
  `iris_harness.sdk.content.redact_external_content(text)` is the same tripwire without the
  envelope, for text a plugin shows the owner or logs rather than hands to a model. It goes
  through the kernel's `redact_text`, so it honours `IRIS_GOVERNANCE_EXTERNAL_CONTENT_FLOOR`
  (off: the text comes back unchanged) and a matching call writes a counts-only ledger row,
  for the first five matches per scope (a run for a plugin that opens one, else a session);
  later matches are still redacted and only counted in one warning.
- All nine email skill tools (`email-triage`: `email_inbox_summary`, `email_focus`,
  `email_needs_reply`, `email_judged_yesterday`, `classify_email_by_id`, `run_email_triage`;
  `gmail-inbox`: `fetch_new_emails`; `email-followup`: `detect_followups`,
  `list_open_followups`) declare `content: external`. Everything they return is derived from
  the mailbox pipeline, so they are external as a class, not because each returns email text:
  six of them carry only counts, ids or status today, so the envelope is conservative there,
  and a later change that adds free text to one cannot silently skip the floor. Code callers
  get no envelope. `rag/docs-search` is undecided and unchanged.
- A skill package's tool can declare `content: internal | external` in its manifest
  (`web-fetch`'s `fetch_web_content` is external), served and in the loop alike.

- Destructive tools and pinned writes now get a durable side-effect ledger row before
  they run (issue #73). This needs no flag: with `IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER`
  unset the ledger covers that high-risk class only, and plain writes and reads are
  unchanged (no row, no file). Behavior change for deployments that set
  `IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER=0`: a destructive tool or pinned write is now denied
  rather than run with no durable record. Remove the setting, or set it to `1`, to run
  them again.

- Behavior change: `IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER=1` (or `true`/`yes`/`on`) is now
  the same as leaving it unset: the high-risk class only. It used to also record every
  non-read call, so an explicit `true` silently widened audit scope compared with the
  default. That scope is now its own boolean, `IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_ALL`
  (default off). If you set `IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER=1` expecting every
  non-read call to be recorded, also set `IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_ALL=1`.
  `IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER=0` is unchanged (no ledger; destructive tools and
  pinned writes are denied), and `..._ALL` set while the ledger is off logs a warning that
  it has no effect. An explicit `IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER=1` (or `true`/`yes`/`on`)
  without `..._ALL` logs one warning at startup saying it covers high-risk calls only;
  leaving the setting unset logs nothing.

### Added

- **Stored assistant text and summaries are scanned when they re-enter a prompt (issue
  #145, step one).** The external-content floor screens a tool result once; the model's
  restatement of it, stored as a turn, came back unscreened through the conversation window,
  the session summary, `recall_conversation`, `memory_search` over sessions, the related
  earlier turns and the intent router's recent-turns context. `kernel/governance/reentry.py` now runs the floor's tripwire over assistant
  turns and summaries at those readers (the owner's own turns are returned verbatim), cuts
  anything over 16 KB per text or 128 KB per read with a visible `[not scanned: ...]` marker,
  and writes one counts-only `audit_log` row per read when something matched or a cap was hit.
  It follows `IRIS_GOVERNANCE_EXTERNAL_CONTENT_FLOOR`; there is no new setting. Phrase-level
  only; the recalled text is redacted (with the marker `[redacted: instruction-like text in stored content]`, since it is the model's own earlier text and not external content) but not yet marked as untrusted. See "Stored text
  coming back into a prompt" in `docs/concepts/governance.md`.

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
- The plugin loader logs a warning when it mounts a plugin whose manifest omits
  `party` (the plugin is treated as `untrusted`), including an entry-point plugin
  with no manifest; `iris plugins` and `iris plugins show` print `party`; the
  examples and `iris plugin new` scaffolds declare `party: untrusted` explicitly.

### Fixed

- The compaction summariser reads stored text through the re-entry scan (issue #161). The
  summary the model writes is stored, so a reworded instruction would defeat a scan at read
  time; the assistant turns and the previous summary now go through the scan on the way INTO
  the summariser (the owner's own turns stay verbatim, nothing is enveloped, the summary's
  provenance is still tracked by its flag). One fix point, `_summary_prompt`, covers `compact`,
  `summarize_all` and the legacy dict API. Audited as reader `compactor`, counts only.

- `httpcore` is a declared dependency (issue #174): the governed client's pinned transport imports it directly and replaces httpx's private `_pool`. A new test pins that attribute, so an httpx upgrade that moves it fails in CI rather than at runtime.

- The behavior miner reads stored turns through the re-entry scan (issue #162). The turns it
  mines are stored text, so an assistant turn that held third-party text is redacted before it is
  put in the miner's prompt and the owner's own turns stay verbatim. The scan sits in
  `mine_behavior_patterns`, the one entry the console preview, the periodic run and the
  compaction-archived span all pass through. Audited as reader `behavior_miner`, counts only.
  The miner's output was already propose-only and human-approved.

- The general lane's direct skill answer is a governed call (issue #155, second part). When a
  request matches a skill, the lane answers it before any model by running the skill's first
  tool; that call used to run in-process with no `PRE_TOOL_USE` / `POST_TOOL_USE`, no approval
  check and no audit row. It now runs through `ToolService` under the `core:general_lane` caller
  (a `core:` caller needs no caller-policy entry, and a test pins that): audit rows with one call
  id, the external-content floor (one scan, where there were two), and the tool-policy and
  approval checks. The owner's answer is unchanged byte for byte for a structured result
  (the tool's result travels as JSON inside the governed call). **Visible change for flag-off
  and shadow installs:** a skill tool that governance will not run without the owner's approval
  is now refused with the governance message instead of running unasked (nothing is queued: a
  skill tool is not a registered tool, so an approval for it could never execute). Governed text
  that is not the tool's JSON (a result governance replaced or withheld) is shown as the text it
  is and records no pending actions. The brief slots (`direct_brief_response`) are a separate
  follow-up, #199.

- The legacy general lane (`IRIS_AGENTIC_CORE_ENABLED` off, or `shadow` where the lane's answer is
  the one read) runs every tool through the governed runner (issue #155, #134 D7). Its
  `memory_search`, `memory_graph`, `wiki_search`, `propose_skill_from_sandbox` and every skill
  tool, `fetch_web_content` included, used to run through a private dispatch table with no
  `PRE_TOOL_USE` / `POST_TOOL_USE`, no external-content floor and no audit row, so an injected
  instruction in a fetched page reached the next prompt, the session log, the answer and the memory
  stores raw. The table is gone: the lane's tool set is built from the loop's own builders and each
  tool carries the loop's declaration (`effect`, `confirm`, `content`). What changes for a
  flag-off or shadow install: a page or wiki result reaches the model inside the untrusted-content
  envelope with the instruction redacted and leaves audit rows; `memory_search` is the loop's
  scoped search (`facts`, `patterns`, `behaviors`, `sessions`); and `propose_skill_from_sandbox`,
  a write that needs the owner's approval, is refused in this lane (it has no checkpoint to pause
  on) instead of writing a draft unasked. With governance switched off
  (`IRIS_GOVERNANCE_ENABLED`) the lane's tools still run ungoverned, as the operator's explicit
  opt-out; one startup warning says so.

- `httpcore` is a declared dependency (issue #174): the governed client's pinned transport imports it directly and replaces httpx's private `_pool`. A new test pins that attribute, so an httpx upgrade that moves it fails in CI rather than at runtime.

- The governed HTTP client's name lookup is inside the request's deadline (issue #173).
  `getaddrinfo` has no timeout, so a resolver that hangs held the request before the deadline
  even started. The lookup now runs on a daemon thread and the request waits for it at most
  what is left of the deadline, then fails as the usual deadline abort. A thread stuck in the C
  resolver cannot be cancelled: it ends when the resolver returns, never keeps the process
  alive, and at most 8 can be stuck at once (a further lookup fails closed at once, "name
  lookups are backed up"). The decompression half of the issue was already done in #171
  (`_decoded` decodes at most 64 KiB at a time).

- The email store and the email onboarding store add their columns the same safe way (issue #201,
  second part): opening an older `email.db` from several processes at once no longer fails with
  `duplicate column name`, and the one-time stamp of old classifications as IRIS's (`classified_source`)
  runs once, in the process that added the column. The `memris` versioned upgrades were measured
  (twelve interpreters, fifteen rounds) and already tolerate the race, so they are unchanged.

- The email judge's user message is no longer wrapped in the untrusted-content envelope
  (undoing that part of the issue #148 change). Measured on a real mailbox, 100 emails judged
  by the local `email_judge` model before and after the wrapping: about 5 stably changed bucket
  at the same confidence, and by reading them the wrapping looked slightly worse (two
  promotional bank loan offers became `bill`). The judge's reply is a schema-constrained
  verdict, so the envelope only guarded against a manipulated label. The digest narration and the
  triage picker keep their envelope.

- Opening an older database from several processes at once no longer fails in the stores that
  migrate their tables (issue #201). The checkpoint, continuation, task, activity, learning-signal,
  document (RAG), reminder and approval stores each read `PRAGMA table_info` and then ran
  `ALTER TABLE ... ADD COLUMN`; two processes (the API, the CLI, a heartbeat) opening the same older
  file both saw the column missing, and the loser's `ALTER` raised `duplicate column name`, which
  stopped that process from starting its store. Measured with twelve real interpreters opening an
  older file at once: every one of these stores failed. They now add their columns through
  `ensure_columns` (a cheap look, then `BEGIN IMMEDIATE` on a connection of its own), keeping
  their original column declarations; the three one-time backfills (document classification,
  reminder lifecycle) run inside the transaction that adds the column, so only the process that added
  it does them, once. The email and onboarding stores and the `memris` versioned upgrades are the
  follow-up.

- The kernel's content classifier runs in linear time on hostile text (issue #156). The `email`
  pattern backtracked quadratically (a 200 KB `"a."*100000 + "@b."` took 19.5 s, and
  `"+1-"*130000` and `"123-45-"*57000` over 20 s each) and so did the voice-transcript marker;
  anything that classifies untrusted text (tool and MCP results, files, RAG ingest, the docs
  search) could be stalled with a small input. Both are now small linear matchers that mean what
  the old patterns meant (checked against them on 40,000 random strings); the same inputs take
  under 0.1 s.

- The external-content scan is idempotent (issue #166). A redaction could expose a phrase the
  first pass had no word boundary for (`...?q=1disregard the above prompt` after a swallowed
  URL), so scanning the scan's own output could still redact. `scan` now re-scans until
  nothing matches (at most 8 passes; spans and ids accumulate; `ScanResult.passes` and
  `.exhausted` report it; reaching the cap logs a warning and still returns the redacted text).
  Text that needed one pass is returned as before. Found by fuzzing 200,000 inputs, 7 of which
  needed a second pass; they are pinned as regression cases beside a seeded CI fuzz.

- An MCP call a `PreToolUse` hook answers with `require_approval` no longer reaches the
  server: the bridge refused only `deny` before the call, though the post step already refused
  both (issue #181).
- The timing test for the linear-time classifier no longer flakes on a loaded runner (issue #156
  follow-up). Its absolute one-second bound failed once at 1.2 s on a hosted runner for a case that
  takes about 0.05 s on a laptop. The bound is now a generous eight seconds (the old quadratic
  patterns took 20-50 s), and linearity is proven by a runner-independent check: doubling the input
  must not much more than double the time, best of three on each side, which a quadratic pattern
  fails whatever the machine; a second test shows the check catches the original email regex.

- `iris docs sync` (`sync_all`) removes a registered file that is gone from disk: its chunks
  and index entries are dropped and the file domain is told (`removed`), the same as
  `iris docs remove`. Before, a deleted file stayed searchable and the file domain kept
  listing it as indexed (issue #130). It prunes at once, but only when the file's parent
  directory exists and can be read: if the parent is missing or unreadable (an unmounted
  volume, a dropped share, a denied folder) nothing is pruned or reported, the entry is
  counted in `sources_unavailable` (the summary says "N skipped: location unavailable") and
  one warning says how many. Only RAG's own entries are removed, never a file. `IngestResult`
  gains defaulted `sources_removed` and `sources_unavailable` counts.

- The re-entry scan's audit rows are deduplicated (issue #164). A poisoned turn left in the
  history window used to write a `reentry_scan` row on every later read (about two a turn: 100
  reads, 100 rows). Rows are now keyed on session, reader, origin and the hashes of the matched
  or cut texts, in this process: the first sighting writes a row and a later one is written at
  the 2nd, 4th, 8th, 16th... sighting, each carrying `sightings` (the running count) in its
  payload. The audit log is append-only, so the count rides on new rows, not on an edit of the
  first. A different poisoned text is a different key and always gets its own row. The counts
  are in-process: a restart resets them. The existing payload fields are unchanged. This is a
  deliberate change to the kernel's audit behaviour.

- `iris run resume` reads the process's shared side-effect ledger for the database instead of
  building a second handle of its own, and opens it before it lists pending rows, so an
  unusable ledger database stops the command there (issue #102).

- Raw mail no longer reaches a model unmarked (issue #148). The email judge's user message
  (sender, subject, body), the inbox digest narration prompt and the triage picker's email
  envelope each go in one untrusted-content envelope with instruction-like spans redacted,
  through `wrap_external_content`; the owner's own instructions stay outside it, and the plain
  digest list the owner reads is unchanged. The offline demo's scripted model rule that matched
  the end of the prompt now allows for the closing tag. Nothing is scanned at ingest: stored
  text is not rewritten, and retrieval is marked where it is read.

- Learned memory is scanned where it is read back into a prompt (issue #163). `active.md`,
  `episodic.md` and a lesson's body are written from conversations, some of which held third-party
  text, and used to reach the model unscanned. The retriever (`active`, the episodic digest, the
  matched lesson and its pointers, the semantic episodic patterns), `memory_search` (`patterns`,
  `behaviors`), the intention-rollup context and the mission proposer now scan them after the size
  cut. Text the scan redacts also comes back inside the untrusted-content envelope
  (`source="learned_memory"`); clean text is unchanged and not enveloped, because a lesson is a
  recipe the model follows. Audit rows carry origin `learned_memory`.

- A document index made under a different embedding model can be repaired (issue #144).
  Chroma refuses to reopen the persisted collection under a new embedder, so the index was
  unavailable, retrieval fell back to keyword search and `iris docs reindex` could not
  rebuild it. `iris docs reindex` now says so in plain words and names the repair, the new
  explicit `iris docs reindex --reset-collection`: it deletes only the vector collection and
  rebuilds it from `rag.db`, keeping every source and every label. Plain `iris docs reindex`
  is unchanged. A rebuild that fails part-way (disk full, embedder error) leaves the index
  partial; the command says so, `rag.db` is untouched, and rerunning it is idempotent. The
  command cannot detect a running IRIS server, so stop the server before
  the reset and restart it after. A server that was left running no longer fails silently:
  on its next request it logs one warning, reopens the collection and retries once (and
  falls back to keyword search with a warning if the reopen fails).
- A governed request whose outcome row cannot be written is no longer invisible (issue #175).
  A failed `post_egress` ledger write used to be a log line only. The request has already
  happened by then, so the response is still returned (failing it would report a POST that took
  effect as failed and invite a retry); the loss is now counted (`unrecorded_outcomes()`),
  logged at error with the call id, and shown as a red `Egress ledger` row in System Health, so
  `GET /health`, the web Health screen, the system_health tool and the CLI all see it. The same
  ledger being down fails the next `pre_egress` write, so later requests are refused. The count
  is per process and clears on restart.

- The external-content floor bounds what a hostile text can grow to without erasing what
  follows it. The first 64 redacted spans in a text keep the full-size marker; each further
  span is redacted with the 3-character `[~]` and the legitimate text between spans is kept
  (an earlier cap collapsed the whole rest of the text into one marker, so one hostile item
  could erase the content after it). Output is asymptotically at most 2x the input plus a fixed 7 KB (the first 64 spans of
  each pass carry the full marker), every span is still redacted, and the span and pattern counts in the ledger row stay exact. A `source` or
  `tool` label with a newline or other control or invisible character can no longer inject a log line or
  ledger value: `redact_text` and the tool-result floor hook clean and cap both (200 characters).
- The research plugin's keyed providers and page fetch go through the governed HTTP client
  (issue #172). Brave, Exa and Tavily, and the page fetch behind `fetch_content`, used to open
  their own connections; each request is now checked against the plugin's `egress` declaration,
  made to the address the client checked (a changing DNS answer no longer matters) and recorded
  as a PRE and a POST ledger row. Redirects are followed one governed request per hop, capped at
  5, with each hop's scheme checked first; a redirect to a loopback, private, link-local or
  metadata address is refused and never connected to, and so is a redirect from https to http
  (http to https and http to http are followed). The fetch carries the tool call's context
  onto its worker threads, and a fetch with no tool call running is refused, not sent. Behaviour
  changes: a bare IP address in a URL is denied (even `open_web` contacts host names only), the
  timeout is the request's total budget rather than per operation, and a page body is read up
  to 10 MiB. The extracted text of a golden set of pages is byte-identical to before. SearXNG
  (the operator's own server), the DuckDuckGo library and the Crawl4AI browser are unchanged.
  The raw-network debt list shrinks by four pairs.

- The side-effect ledger is one shared handle per database, not one per kernel, and a ledger
  key that already holds a row is no longer ignored silently (issue #102). A destructive or
  pinned call whose key is taken is denied before it runs; a call's post-run record that
  finds its key taken is reported as a warning instead of confirmed.
  If the database file is removed while the process runs, the ledger creates it again,
  empty, and logs one warning (pending write-ahead rows it held are gone). Not covered by
  this change: `iris run resume` (`main.py`) still builds its own `SideEffectLedger`
  instead of the shared handle, and MCP-bridge calls get no write-ahead record (they declare
  no effect today, and calls with no call id need a key scheme first; see #134).
- A wildcard egress declaration over a public suffix is refused (issue #175). `*.co.uk`,
  `*.com.au` and the like passed because no public-suffix list shipped, so a plugin could
  declare every registrant under them. The harness now vendors Mozilla's Public Suffix List,
  ICANN section only (`kernel/governance/public_suffix_icann.dat`, MPL-2.0, the notice, version
  and source commit kept in its header; refreshed by `scripts/refresh_public_suffixes.py`), and
  a wildcard whose base is a public suffix fails manifest validation, so the plugin does not
  mount; `*.example.co.uk` and `*.github.io`-style declarations (the PRIVATE section is not
  vendored) are accepted. A manifest that declared such a wildcard needs the domain it means.

- A ReAct step's `pre_llm_call` audit row names the model and provider the step
  actually called: the step resolves its model once and the row and the call share
  it, where the row used to carry a snapshot from when the loop was built and the call
  asked the tier router again (a governor downshift or a tier edit could make them
  disagree, and a turn paid one extra router ask, a governor acquire, for the snapshot).
  A router answer of the wrong type still fails loudly rather than yielding a row with
  no model. `model` and `provider` now
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
