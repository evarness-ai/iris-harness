# Plugin egress: declared, allowed, recorded (issue #103)

Status: implemented (issue #103). Epic #85, exit criterion G3. Sequenced
before the MCP rung (#111-#114), which reuses the declaration designed here.

## The problem

A plugin's own outbound HTTP passed through no governance and left no ledger row. The
only governance on a tool's network use is the `network_egress` `PRE_TOOL_USE` hook, which
inspects tool *names* in a built-in list and a coding persona's domain allowlist; for a
plugin's tool its row reads `network_egress: not a network tool`. The manifest had no
place to say which hosts a plugin talks to (`egress:` was rejected as an extra field),
`sends_to` accepted only `search_engine`, and `config/governance/egress.yaml` is the
LLM-prompt class-to-tier policy, not a plugin network policy. The operator's only record
was whatever the plugin chose to log through `sdk.logging.log_egress`.

## The invariant

> Every outbound network call a plugin makes **through the SDK** is declared in its
> manifest, allowed by policy, and recorded in the governance ledger.

The invariant is about calls that go through the governed client. It is not a claim about
the process (see "What this does not prove").

## Shape

1. **Manifest declaration.** `egress:` lists the hosts a plugin may contact:

   ```yaml
   egress:
     hosts:
       - api.open-meteo.com                   # shorthand: https, port 443, data: internal
       - host: geocoding-api.open-meteo.com
         data: personal                       # the highest data class this host receives
       - host: "*.example.org"                # a subdomain pattern, never a bare "*"
         schemes: [https]
         ports: [443, 8443]
     open_web: false                          # true: any host (a page fetcher); see below
   ```

   Unknown keys are rejected. `data` is `public | internal | personal` (never `secret`);
   the default is `internal`, so a plugin that sends something personal must say so.
   `open_web: true` is the explicit, loud form of "any host": it is for a tool whose job
   is to fetch pages the owner or model choose (the research plugin). It still records
   every call.

   The harness compiles every mounted manifest into one `PluginEgressPolicy`
   (`runtime/egress_access.py`) and registers it after plugins mount, exactly as it does
   the caller policy (`runtime/tool_access.py`, `kernel/governance/caller_policy.py`).
   Config, not code: no plugin name is hard-coded in the kernel.

2. **A governed HTTP client** (`iris_harness.sdk.http`). A plugin gets one from
   `api.http` (bound to its own name by the harness) or, in a declarative plugin's function, from `current_http()` (bound to the
   tool being run). Each request:
   - is held to the call it is made inside: the harness stamps the running tool's plugin, run,
     step and data class around every tool and capability call, and the client acts only for
     that plugin. `GovernedHttp("other")` can be constructed (the class is a stable name), but
     a request from a tool of another plugin is a recorded deny (`pre_egress`, attributed to
     the plugin whose tool is running, `egress.plugin` = the name the client was built with),
     so a plugin cannot borrow another's declaration or `open_web`. A request made with no
     call in progress (a `threading.Thread` the tool started loses the call's context, as does
     code run at import) is a recorded deny too: it has no run, parent call or data class to be
     held to. Do the request on the tool's own thread;
   - fires a new kernel hook point `PRE_EGRESS`; the `plugin_egress` hook allows only a
     scheme/host/port the plugin declared and refuses a run whose data class is above the
     host's declared `data` (fail closed, no policy or no kernel means deny);
   - runs the owner-PII shadow observation over what leaves (same `egress` column the
     network tools use; shadow only, like the column itself, ADR-0125);
   - makes the request: redirects are not followed (each hop is a new, governed call); the
     environment is not read (`trust_env=False`: no proxy variables, netrc or CA-bundle
     variables); a `Host`, `Proxy-*`, `Connection`, `Upgrade`, `TE`, `Transfer-Encoding` or `Content-Length` header is refused (judged on the normalised names, any spelling) with a recorded deny; the name is
     resolved once and the connection goes to the address that was checked (below); the whole
     transfer has one time budget and the decoded body a size cap (below);
   - fires `POST_EGRESS`, which records the outcome.

   The ledger rows carry host, port, scheme, method, the declared data class, the plugin
   (`tool_plugin`), the tool (`tool_name`), the caller, the run id and session, and on
   the outcome row status, bytes sent and received, duration and an exception class on
   failure. **Never** a path, query, header, body or secret. A `log_egress` line is also
   written, so `iris.egress` stays one trail.

   The harness stamps the call's run, step, tool, caller and classification around the
   tool invocation (`EgressScope`, a `ContextVar` set in `GovernedToolRunner` for both
   tool calls and capability provider calls), so the rows join the turn's other rows on
   `chat` and `chat_stream` alike: both run the one tool runner.

   Tests: `iris_harness.testing.fake_http(...)` replaces only the transport, so the
   policy, the hooks and the rows run exactly as in production. This replaces each
   plugin's own `make_setup(transport)` (GAP-4). `no_network()` still refuses real
   sockets.

3. **Drift check.** At runtime, a host the manifest does not declare is denied and
   recorded (a `deny` row naming the host). In the conformance suite,
   `check_conformance` fails a plugin whose example call was denied for an undeclared
   host, and `testing.check_network_imports(paths)` flags plugin source that imports a
   raw network library (httpx, requests, urllib.request, http.client, socket, ssl,
   aiohttp, urllib3, smtplib, imaplib, ftplib, websockets, ...). A plugin CI runs it as it
   runs `check_stable_imports`.

4. **A generic `sends_to` destination.** `external_service` joins `search_engine`: a tool
   declaring it hands its arguments to a service outside the machine, so the owner-PII
   `egress` column treats its `PRE_TOOL_USE` arguments as leaving (destination = the tool).
   `search_engine` stays: it names a destination with its own guard column
   (`web_search`). A tool declaring `external_service` must belong to a plugin that
   declares `egress` hosts (or `open_web`), checked at manifest load.

5. **Not solved.** See "What this does not prove".

## The host grammar

A declared host is strict, and refused at manifest load rather than guessed at: ASCII letters,
digits and hyphens only (an internationalised name is declared in its `xn--` form, which must
decode); at most one trailing dot, which is dropped; labels of at most 63 characters, a name of
at most 253, at most 256 hosts per plugin; no whitespace or control character (a newline
included). Refused outright, fail closed, and an owner may relax it later: IP literals in any
spelling (dotted, short `127.1`, hex, octal, IPv6), `localhost`, any name whose last label is
numeric, and a wildcard whose base is a single label (`*.com`).

The host of a request is held to the same rules before it is compared (one trailing dot at
most, ASCII, no whitespace or repeated dots). A request whose host is an IP literal in any
spelling, or not a valid host name, is denied by the policy even for `open_web: true`, with a
`pre_egress` row; it is never sent.

Names that mean this machine or the local network are refused as a request's target even
for `open_web: true` (and cannot be declared): `localhost`, `*.localhost`, `*.local`,
`*.internal`, `*.localdomain`, with or without a trailing dot.

A wildcard over a public suffix (`*.co.uk`, `*.com.au`, `*.ck`) is refused at manifest load: it
would cover every registrant under the suffix, which `*.example.co.uk` does not. The suffixes
are Mozilla's Public Suffix List, **ICANN section only**, vendored in
`kernel/governance/public_suffix_icann.dat` (MPL-2.0; the file keeps the notice, version and
source commit in its header) and refreshed with `scripts/refresh_public_suffixes.py`. The
PRIVATE section (hosting providers such as `github.io`) is left out on purpose, so
`*.github.io`-style declarations are accepted and the owner reads them. A stale list is
fail-closed only in the sense that nothing already refused becomes allowed: a suffix created
since the last refresh is simply not known yet. The header of the file says which list version
it is; `python scripts/refresh_public_suffixes.py --check` (exit 1) says whether it differs from
the live list, and running it without `--check` refreshes it.

## What a request may reach, and how much it may take

A name that passes the policy can still resolve to an address inside the owner's machine or
network (an attacker-chosen name pointing at `127.0.0.1` or the cloud metadata address; a
name that answers with a public address first and an internal one later: DNS rebinding). The
check therefore sits at the connection (`runtime/egress_transport.py`): the name is resolved
**once**; if any answer is loopback, private (RFC 1918, `fc00::/7`), link-local
(`169.254.0.0/16`, `fe80::/10`), shared (`100.64.0.0/10`), unspecified (`0.0.0.0/8`, `::`),
multicast or reserved, or an IPv6 form embedding such an IPv4 address (v4-mapped, NAT64,
6to4), the connection is refused; otherwise it goes to the address that was checked, with the
TLS server name, certificate verification and `Host` header still those of the host name. A
refusal is a `post_egress` row (`error: EgressDenied`, `aborted: address`) and raises
`EgressDenied`; nothing was sent. The check runs on the real transport only (`fake_http`
replaces the transport, so there is no socket to check).

Bounds: the whole request has one wall-clock budget, enforced on every socket operation and
between body chunks (`timeout` seconds, default 10, at most 60; `None`, zero, negative or NaN
mean the default; this is a total, where httpx's own timeout is per operation), and the
decoded body is read up to 10 MiB (fixed: not manifest-configurable). Either limit ends the
request with a `post_egress` row (`error: EgressDenied`, `aborted: max_bytes | deadline`,
`bytes_in` = the decoded bytes actually read) and an `EgressDenied` with a fixed message.
The returned response holds the decoded body, without `Content-Encoding` and length headers.
httpx never decodes: it would expand a whole wire chunk in one step, before any size check, so
a small compressed body (layered gzip, zstd, brotli) could take gigabytes. The client asks for
`Accept-Encoding: identity`; a server that compresses anyway is accepted only for a single
`gzip` or `deflate` layer, decoded by a bounded decompressor at most 64 KiB per step (memory
stays near the cap), and any other `Content-Encoding` is refused before its body is read
(`aborted: encoding`; an undecodable body is `aborted: decode`).

A request whose URL cannot be parsed is a `pre_egress` deny row (`malformed: the URL is not
valid`) and an `EgressDenied` with a fixed message: the URL is never echoed. A request whose
`pre_egress` row cannot be written is not sent: the kernel withdraws the allow at that one
hook point (a failed ledger write elsewhere still never raises).

## Record identity (issue #134)

Built on #134 stage 1 (the runner mints one ULID `call_id` per call attempt, the kernel
stamps it on every audit row from metadata). An egress request is a call of its own, so it
gets the same identity, through the same path:

| Field | On the row (top level of the payload, `GET /governance/audit`, the CLI `call` column) | Where it comes from |
|---|---|---|
| `call_id` | the request's own id: a 26-character ULID, shared by its `pre_egress` and `post_egress` rows (and every hook that fired for them) | minted by the governed client with `iris_harness.foundation.ids.new_ulid`, the runner's minter; there is no argument to pass one; carried in the hook context's **metadata** and written by `kernel._audit`, never read from the payload |
| `parent_call_id` | the calling tool's (or capability call's) runner-minted `call_id`, i.e. the id on that call's own `pre_tool_use` / `post_tool_use` rows. Absent for a request made outside a governed call | read by the client from the harness's call scope (`EgressScope.tool_call_id`, set around the tool or provider invocation by the runner), carried as metadata `parent_call_id`, stamped by `kernel._audit` |
| `attempt`, `replay_of` | inside the nested `egress` value: a repeat of the same method, host and port inside one governed call is attempt 2, 3, ... and `replay_of` names the first request's `call_id` | the client |

`parent_call_id` is a public payload field next to `call_id` and `held_call_id`
(`audit_view.PUBLIC_PAYLOAD_FIELDS`), so the audit API and CLI show it. `TurnAuditRow.egress`
(additive) is the nested record with the two stamped ids merged in, so a plugin's test reads
`row.egress["call_id"]` and `row.egress["parent_call_id"]` without a new row field. The row
already carries `run_id`, `step_id`, `session_id`, `caller`, `tool_name` and `tool_plugin`
(#124/#131). A denied request is a `pre_egress` deny row; a malformed one (no host, a URL
with credentials) is a deny row too, flagged `malformed`; a failed one has a `post_egress`
row with `error`. Each carries the same ids. None is silent.

What #134 stage 1 provides: the ULID format and minter, kernel-side stamping from metadata
only, `held_call_id`, and the `call_id` public field. What stage 2 will standardise: the
ids as first-class audit columns rather than payload keys (so `parent_call_id` and `attempt`
become queryable and survive Parquet compaction), a record id minted inside the store write,
`attempt` / `replay_of` as the generic retry edge (today only the client sets them, for
egress), a parent edge for every child call (plugin `api.tools`, capability calls), and
replay across a resumed run. Not covered today: a request made by a streamed capability
method after it returns (its generator body runs outside the call scope), which has an id
but no parent.

## Interplay with the external-content floor (#137)

The floor applies to **tool results** (`content: external` tools): the kernel's
`external_content_floor` hook scans and envelopes what a tool returns to the loop's model.
It does not see the governed client's response: `http.get(...)` hands the plugin the
third-party body as an ordinary return value, and nothing is scanned, wrapped or redacted on
that path. What is and is not covered:

- Covered: a tool that returns the fetched text and declares `content: external` (the
  `forecast` example in the tests) is enveloped and scanned by the floor on its way to the
  model, as before. The egress rows record the request (host, status, byte counts), never
  the body.
- Not covered: text the plugin itself puts into a prompt of its own, shows to the owner, or
  stores, before or without returning it. A plugin that does this should pass the body
  through `iris_harness.sdk.content.wrap_external_content(body, source=...)` first. The
  governed client does not do it for the plugin, because the same body may be JSON the
  plugin parses; wrapping is a decision about prose.
- The client never wraps or redacts a response (it does cap its size, below); `fake_http`
  bodies are returned untouched. A plugin that needs redaction beyond the floor's tripwire waits for the
  redaction helpers being consolidated separately; this design adds none.

## Decisions

**Default for a plugin that declares no `egress`: deny all, through the governed client.**
An empty declaration is a closed door, not "unrestricted". Nothing breaks on migration:
the client is opt-in, so a plugin that does not use it is unaffected by the policy, and
a plugin that adopts it must declare first. The only nudge for plugins that stay on raw
libraries is the lint (run by the author) and the conformance check (which sees only calls made through the client); no mount-time notice (the loader's
existing notices cover `party`).

**`party` does not change the policy.** `party` is a declaration of provenance; nothing
enforces it. The same rule applies to first-party and untrusted plugins; what differs is
the lint. `testing.check_network_imports` is a function a plugin author runs in the
plugin's own CI (it is not part of `check_conformance`, and the harness does not run it at
mount). First-party plugins that import raw libraries today are on an explicit, tested,
shrinking list (below), so a *new* raw import in a first-party plugin fails this repo's
`tests/` instead of passing silently. That is not an exemption: it is a visible debt.

**First-party network use today** (verified in source; `tests/unit/iris_harness/test_testing/
test_network_imports.py` pins the files):

| Plugin | What it connects to | In this change |
|---|---|---|
| `research` | SearXNG (operator URL), `api.tavily.com`, `api.exa.ai`, `api.search.brave.com`, `duckduckgo.com`, and any page it fetches (urllib, Trafilatura, Crawl4AI) | `egress: {open_web: true}` declared; its SSRF checks stay; five files on the debt list |
| `gmail` | `gmail.googleapis.com`, `www.googleapis.com`, `oauth2.googleapis.com`, `accounts.google.com` (Google API client, OAuth) | the four hosts declared (`data: personal`); three files on the debt list |
| `imap` | the owner's IMAP server over TCP (`imaplib`, `ssl`) | not an HTTP host, so nothing to declare; one file on the debt list |
| `email_workflows` | job-posting fetch (`httpx`), discovery (`requests`), a demo that guards sockets | three files on the debt list; no declaration yet |
| `telegram_channel`, `web_push_channel` | `api.telegram.org`, browser push endpoints, through `services/channels` | the plugin files import nothing raw, so the lint is silent; no declaration yet |

Migrating each onto the governed client is separate work (their SSRF checks, DNS handling,
OAuth transports and streaming need care); declaring their hosts now makes `iris plugins
show` truthful and gives the debt list a name per file.

**Stable-tier additions** (each is a public name a plugin or its CI needs):
`sdk.http.GovernedHttp` (the client's type, for annotations), `sdk.http.EgressDenied`
(what a denied call raises: a `RuntimeError` subclass, deliberately not an `OSError`, so a
plugin's `except OSError` around its network code cannot swallow a governance denial; it was a
`PermissionError` before release, and the name is unchanged), `sdk.http.current_http` (declarative
plugins have no `api`), `testing.fake_http` (the one injection point for tests),
`testing.check_network_imports`, `testing.NetworkImportViolation` and
`testing.NETWORK_MODULES` (the lint and what it reads). `PluginAPI.http` is an attribute of
an already stable class. Also new: the manifest key `egress`, `TurnAuditRow.egress`, and
the conformance check name `egress`.

**Fail closed in three more places.** No policy registered: every host denied. No kernel
bound, or a kernel without the `plugin_egress` hook (an empty kernel would otherwise
answer "allow, no hooks registered"): the client raises before sending. A host the policy
does not know, or a plugin that is not mounted: denied.

**Built here, left for later.** An operator-side narrowing file (the egress counterpart of
`config/governance/tool-access.yaml`), a capability `MethodSpec.sends_to` (#100), egress
rows for the items a *streamed* capability method yields after it returns (the generator
runs outside the call's scope; such a request is still checked and recorded, attributed to
the plugin without the tool and run), and moving the first-party files above onto the
client.

**Relation to `sends_to: search_engine`.** Orthogonal. `sends_to` says which owner-PII
column reads a tool's *arguments*; `egress` says which *hosts* the plugin's code may
contact and records each contact. A tool can have both; the research plugin does.

**Capability `MethodSpec`** (#100, third bullet): a capability method cannot yet declare
that its arguments leave the machine. Not done here; the governed client already records
the host of the call such a method makes.

## What this does not prove

An in-process plugin (`trust: in-process`) is a contract, not a sandbox. It runs in the
harness's interpreter and can open its own socket, import `urllib`, or shell out. The
governed client proves: *a call made through it* was declared, allowed and recorded, and a
denied host was not contacted by it. The lint proves the plugin's *own source files*
import (or use through an imported package: `urllib.request.urlopen`, `asyncio.open_connection`)
no raw network library it names, and reports an unparsable file as a finding; it does not see
dynamic imports, a library the list does not name (`paramiko`, `aiosmtplib`, `boto3`, `openai`,
`redis`, ...), an event loop's own `create_connection`, a dependency's own
network use, or `subprocess`. A ledger with no row for a host is therefore not proof the
host was never contacted by a plugin; it is proof the governed client never contacted it.
`no_network()` proves a test path made no socket in the test process.

| Not enforced | Why, and what holds instead |
|---|---|
| A plugin opening its own socket, or shelling out | In-process code is a contract, not a sandbox; the lint sees only static imports. Boundary: the MCP rung's process isolation (#111-#114) |
| A plugin that bypasses the call scope | Inside a call the client acts only for the running tool's plugin (enforced). An in-process plugin that goes around the client altogether (own socket) is the row above |
| A request made on a thread the tool started | Denied, not governed: it has no call scope. The plugin must request on the tool's thread |
| The lookup's integrity | A resolver that lies about a public name is the operator's DNS concern. The returned addresses ARE checked, and the lookup's duration IS bounded: it runs on a daemon thread inside the request's total deadline (a thread stuck in the C resolver cannot be cancelled; it ends when the resolver returns, and at most 8 can be stuck at once, after which a lookup fails closed at once) |
| Memory just past the cap | Decoding is bounded to 64 KiB per step, so memory is about the cap plus one step |
| Ledger completeness | A request through the client with no `pre_egress` row is not sent; a request that does not go through the client leaves no row. A `post_egress` row that fails to write is logged and not retried |
| A wildcard over a public suffix (`*.co.uk`) | Refused at manifest load (the vendored ICANN list); a suffix newer than the last refresh is not known; the PRIVATE section (`github.io`) is not refused |

OS-level enforcement (a launch wrapper, proxy environment, sandbox profile or network
namespace) belongs to the MCP rung, which isolates the process. #111-#114 build on this
design: the same manifest `egress` block, compiled by the same `PluginEgressPolicy`, and
the same `PRE_EGRESS` / `POST_EGRESS` rows for contacts a launch wrapper reports.

## Behaviour matrix

| Situation | Result | Rows |
|---|---|---|
| Host declared (scheme, port, data class fit) | request made | `pre_egress` allow, `post_egress` outcome |
| Tool of plugin `evil` uses `GovernedHttp("weather")` or `("research")` | `EgressDenied`, no request | `pre_egress` deny, `tool_plugin: evil` |
| No governed call in progress (a bare thread) | `EgressDenied`, no request | `pre_egress` deny |
| Host not declared | `EgressDenied`, no request | `pre_egress` deny naming the host |
| Plugin declares no `egress` | every host denied | `pre_egress` deny |
| `open_web: true` | any host allowed | both rows |
| Run classified above the host's `data` | denied | `pre_egress` deny |
| First-party vs untrusted | same policy | same rows |
| No policy or kernel bound | denied (fail closed) | none possible; raises |
| `fake_http` active | transport faked, everything else as production | both rows |
| Host resolves to an internal address (any answer) | `EgressDenied`, no connection | `post_egress` with `aborted: address` |
| `localhost`, `*.local`, `*.internal`, ... | denied, even for `open_web` | `pre_egress` deny |
| Body over 10 MiB decoded, or past the time budget | `EgressDenied`, cut off | `post_egress` with `aborted`, `bytes_in` |
| `Host` / `Proxy-*` header, or an unparsable URL | `EgressDenied`, no request | `pre_egress` deny |
| `pre_egress` ledger write fails | `EgressDenied`, no request | none (logged) |
| `no_network()` active, real transport | the socket is refused; outcome row records the error | both rows |
