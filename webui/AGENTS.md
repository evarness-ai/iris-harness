# webui/

The web console: React 18, Vite, TypeScript, Tailwind CSS. The IRIS API serves its production build (`npm run build` writes it to dist, which is not committed).

## Map

- `src/screens/` — one file per screen. `src/components/` — shared UI.
- `src/lib/` — API client (`http.ts`, `client.ts`), queries and types.
- `src/lib/nav.ts` — renders the navigation the API returns: the core screens (`config/webui/nav.yaml`) plus those of mounted plugins (`src/iris_harness/runtime/plugin_host/nav.py`). The web UI never decides which screen exists.
- `tests/` — Playwright viewport tests.

## Local rules

- A thin renderer, never a dependency: no harness behaviour lives only here. Logic belongs in `src/iris_harness/`, behind the API, with a CLI path too.
- Ad-hoc writes are off by default (`IRIS_WEBUI_ALLOW_WRITES`); Setup's confirmed actions go through the approval queue.
- Check `npm run build` (type-check + build) and `npm run lint` before a PR.

## Running

`npm install && npm run dev` serves on :5181 and proxies the API (default :8003; `IRIS_API_PORT` or `IRIS_API_URL` override it).
