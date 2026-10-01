import path from "node:path";
import type { IncomingMessage } from "node:http";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server. /api (traces/sessions) and /chat (streaming chat, NDJSON) are
// proxied to src/iris_harness/server/iris_api (default port 8003, override with IRIS_API_PORT).
//
// IRIS_API_URL points the whole UI at a DIFFERENT harness — a server over Tailscale,
// say — so the local UI can watch the deployment that is actually serving the owner's
// phone. The secret must then be that harness's own IRIS_AUTH_SECRET, since each
// deployment has its own.
const apiPort = process.env.IRIS_API_PORT ?? "8003";
const target = process.env.IRIS_API_URL ?? `http://127.0.0.1:${apiPort}`;

// Shown as a badge in the UI. Mistaking the cloud's data for the laptop's would make
// every number on screen a lie, so the origin is always visible when it is not local.
const targetLabel = process.env.IRIS_API_LABEL ?? (process.env.IRIS_API_URL ? target : "");

// The API requires `Authorization: Bearer $IRIS_AUTH_SECRET` on every route
// (security floor, phase 0). The proxy injects it server-side so the secret
// never reaches browser code; without it, API calls answer 401/503.
const authSecret = process.env.IRIS_AUTH_SECRET;
const authHeaders = authSecret ? { Authorization: `Bearer ${authSecret}` } : undefined;

// Several backend resource paths (/portfolio, /routines, /health, …) share a name
// with an SPA route. A full-page browser navigation (deep-link or refresh) to one
// of those must serve the SPA shell so React Router can render it client-side —
// only XHR/fetch/streaming calls should proxy to the API. ``Sec-Fetch-Mode:
// navigate`` is sent by browsers ONLY on top-level navigations (never by fetch /
// XHR), so it's the precise discriminator: on a navigation we bypass the proxy
// and let Vite serve index.html; everything else proxies as before.
function api() {
  return {
    target,
    changeOrigin: true,
    headers: authHeaders,
    bypass(req: IncomingMessage): string | undefined {
      if (req.headers["sec-fetch-mode"] === "navigate") return "/index.html";
      return undefined;
    },
  };
}

export default defineConfig({
  define: {
    "import.meta.env.VITE_IRIS_TARGET_LABEL": JSON.stringify(targetLabel),
  },
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
  server: {
    port: 5181,
    proxy: {
      "/api": api(),
      // Streaming endpoint — http-proxy streams NDJSON tokens live by default.
      "/chat": api(),
      // Read-only control surfaces (Phase 3): llm mode/pressure, learning
      // metrics, routines, heartbeats, in-runtime memory.
      "/llm": api(),
      "/observability": api(),
      "/routines": api(),
      "/portfolio": api(),
      "/market": api(),
      "/heartbeat": api(),
      "/memory": api(),
      // RAG uploads/search (Phase 4) + unified knowledge graph (Phase 5).
      "/rag": api(),
      "/knowledge": api(),
      // Governance audit + settings (Phase 6, read-only).
      "/governance": api(),
      "/settings": api(),
      // System Health snapshot (ADR-0069, read-only).
      "/health": api(),
      // Runtime inventory (#5) + self-learning experiments (#4), read-only.
      "/runtime": api(),
      "/learning": api(),
      // Write-capability flag (IRIS_WEBUI_ALLOW_WRITES).
      "/capabilities": api(),
      // Playground — scenario suites, run, config↔runtime drift (Phase 1).
      "/playground": api(),
      // Action Center (ADR-0073) + per-agent console/settings/metrics (ADR-0074).
      "/actions": api(),
      // Async Activity feed (P1) — background system jobs (read-only).
      "/activities": api(),
      "/agents": api(),
      // Plugin inventory — profile, registrations, manifests + YAML (read-only).
      "/plugins": api(),
      "/tasks": api(),
      "/reminders": api(),
      "/finance": api(),
    },
  },
});
