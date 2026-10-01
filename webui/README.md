# IRIS Console — Web UI

React + Vite + TypeScript + Tailwind + [React Flow](https://reactflow.dev) console for IRIS.
The first screen is the **Agent Call Trace**: a graph-based, end-to-end visualization of what
happens when a request flows through IRIS — which component fires, when, how long it takes, how the
LLM is invoked (input/output/tokens), and CPU/GPU/memory usage — with an animated replay of the
data flow in sequence.

## Run

```bash
cd webui
npm install
npm run dev      # http://localhost:5181
npm run build    # type-check + production build
```

## What's here (MVP)

- **App shell** — dark console with sidebar nav. Call Trace is live; other screens are stubs.
- **Call Trace screen** (`src/screens/CallTrace.tsx`):
  - React Flow graph (custom `IRISNode`, dagre top-to-bottom layout in `src/lib/layout.ts`).
  - **Replay** transport (play/pause/step/speed) that reveals nodes/edges in chronological order
    and pulses the active component — the "data flow in sequence" animation.
  - **Node detail panel** — timing, LLM model/provider/tokens, input/output, tool stdout/stderr,
    governance decision, and a CPU/GPU/RAM resource panel.
  - **Waterfall** — per-step duration bars.
  - Trace selector over the recent turns.

## Data source — the live API, nothing else

The screen calls the IRIS API (proxied via `/api`):

- `GET /api/traces` — newest-first summaries of recent turns
- `GET /api/traces/:trace_id` — the full node/edge graph for one turn

These are served by `src/iris_harness/server/iris_api` and built by `src/iris_harness/observability/trace_builder.py`,
which reconstructs the graph from session JSONL (`~/.iris/logs/session-<id>.jsonl`). A *turn*
(`user_message` … `agent_response`) becomes one trace; `trace_id` is `"<session_id>~<turn_index>"`.
Degenerate turns (e.g. `approve`/routine triggers with no pipeline activity) are skipped.

The console ships no mock data. If the API is unreachable, every screen shows the "API
unavailable" notice naming `iris serve`. If it answers but has no logs yet (a fresh install),
Call Trace and Sessions say so and link to Chat: start a conversation and its turns appear
here. A 401 goes to `/pair` (see *Pairing a browser*). `tests/real-data-only.spec.ts` holds
all three.

To run against live data, start the API and the dev server:

```bash
# terminal 1 — IRIS API (default port 8003; override with IRIS_API_PORT)
cd .. && poetry run uvicorn iris_harness.server.iris_api.main:app --port 8003
# terminal 2 — web UI (proxies /api -> 8003)
cd webui && npm run dev
```

### Served by the API (no Vite)

A deployment has no dev server, so the IRIS API can serve the production build itself
(`src/iris_harness/server/iris_api/static_ui.py`). It applies the same rule as the dev proxy in
`vite.config.ts`: `/assets/*` are files, and a browser navigation (`Sec-Fetch-Mode: navigate`)
gets `index.html`, so a deep link or refresh on `/health` is the console while a `fetch` to
`/health` is still the API.

```bash
cd webui && npm run build
cd .. && IRIS_WEBUI_DIST="$PWD/webui/dist" \
  poetry run uvicorn iris_harness.server.iris_api.main:app --port 8003
# open http://127.0.0.1:8003/
```

The shell and its assets load without a token; every data call still needs a credential.
Unlike the dev proxy, nothing injects the secret for the browser: the browser **pairs**
instead (next section). Day-to-day development keeps using `npm run dev`.

### Pairing a browser (ADR-0117)

A browser served by the API authenticates as a paired device
(`docs/architecture/mobile-cloud-ui-plan.md`, track 1 PR 3):

1. Any data call that answers **401** sends the console to `/pair?next=<where you were>`.
2. Get a code — `iris device pair` on the machine running IRIS, or **Devices → Pair a new
   device** on a browser that is already paired. Codes look like `ABCD-EFGH`, last 5 minutes,
   work once, and are voided after 5 wrong tries.
3. Enter the code and a device name on `/pair`. The API answers with an **HttpOnly**
   `iris_device` cookie (`SameSite=Strict`, `Secure` over HTTPS) and the console returns to
   `next`. `next` is only honoured when it is a same-origin path (`safeNext` in `lib/http.ts`).

**Devices** (System group, beside Governance; `src/screens/Devices.tsx`) lists paired devices
with scope, last seen and a "this device" marker, revokes them (revoking the current one is
"Sign out"), and starts a new pairing with a `read` or `control` scope. A `read` device can
look but not pair or revoke others; the server enforces that, the UI only hides the dead
buttons. Device administration is **not** behind `IRIS_WEBUI_ALLOW_WRITES`.

Under `npm run dev` the Vite proxy injects the service secret, so nothing 401s, `/pair` is
never reached on its own, and Devices reports "service secret" rather than a device.

### The HTTP rule: never call `fetch` directly

Every API call goes through `apiFetch` in `src/lib/http.ts` — same signature as `fetch`,
same `Response` back (streaming included). It is what turns a 401 into the pairing screen,
once, for every lib module, React or not. Two rules ride on it:

- **No tokens in JS.** The cookie is HttpOnly; never read, store or send a device token or
  the service secret from browser code (no `localStorage`, no `Authorization` header).
- **A 401 is never hidden behind data.** No read falls back to canned data, so a 401 is an
  error like any other and `apiFetch` sends the browser to `/pair`.

`tests/unit/test_webui/test_webui_pairing.py` holds these lines (there is no JS test runner).

The server image does this build itself: the `Dockerfile` has a `node:22-slim` stage that runs
`npm ci && npm run build` and copies the result to `/app/webui/dist`, the default of
`IRIS_WEBUI_DIST`, so a deployment serves the console with nothing to set. `npm ci` refuses a
`package-lock.json` that is out of step with `package.json`, so commit the lockfile with every
dependency change. `webui/dist/` is in `.dockerignore`: a local build never reaches the image.

The release wheel carries the build too (OSS plan R6): the release workflow runs
`scripts/bundle_webui.py` after `npm run build`, which copies `webui/dist` into
`src/iris_harness/server/iris_api/webui_dist/` (gitignored), and pyproject's `include`
ships it. With `IRIS_WEBUI_DIST` unset and no `/app/webui/dist`, the API serves that copy,
so `pip install iris-harness` gets the console with no Node.

That one build serves every deployment, so it cannot carry the deployment's name. The
"Remote harness" badge (`src/components/TargetBadge.tsx`) reads `deployment_label` from
`/capabilities` at runtime (`IRIS_DEPLOYMENT_LABEL` on the server) and falls back to the
build-time `VITE_IRIS_TARGET_LABEL` (from `IRIS_API_LABEL`, see `vite.config.ts`). Neither set: no badge.

Governance hooks are not in the session log (they live in the governance audit DB), so they are
not yet shown for live traces — a follow-up.

## Resource metrics

`session_log.py` now attaches a `resources` block (CPU%, RAM free/total, thermal) to `llm_call`,
`tool_run`, and `agent_response` events. **GPU%** is not obtainable via psutil on macOS / Apple
Silicon (would need privileged `powermetrics`), so it is logged as `null` and rendered as
"n/a".

## Stack

- React 18, Vite 8 (rolldown-based — no esbuild dependency), TypeScript 5
- `@xyflow/react` (React Flow v12) + `@dagrejs/dagre` for layout
- Tailwind (PostCSS build) + React Router + TanStack Query + lucide-react
- `@/*` path alias → `src/*`

## Design system

`design.md` is the single source of truth for design tokens (colors, type,
spacing, radius, elevation), in the [google-labs-code/design.md](https://github.com/google-labs-code/design.md)
format. The CSS variables in `src/styles/tokens.css` realize those tokens for
**light + dark** (dark is the default); Tailwind maps token names to them, so
components use `bg-surface` / `text-fg-muted` / `rounded-lg` and never raw hex or
inline CSS. Theme toggle persists to `localStorage`.

```bash
npm run tokens:lint   # validate tokens + WCAG-AA contrast (errors fail)
npm run tokens:gen    # export design.md tokens to a Tailwind JSON (reference)
npm run build         # tsc + vite (no CDN)
```
