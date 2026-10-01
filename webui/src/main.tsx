import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider } from "react-router-dom";
import { router } from "./routes";
import { Toaster } from "@/components/ui/sonner";
import { onUnauthorized } from "@/lib/http";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 15_000, refetchOnWindowFocus: false, retry: 1 } },
});

// A 401 from any lib/* call becomes a client-side navigation to /pair (lib/http.ts).
onUnauthorized((to) => void router.navigate(to, { replace: true }));

/* Register the service worker (Track 2b PR 8).
 *
 * Production only: in development Vite serves `public/` too, and a worker
 * that survives across HMR reloads is a confusing thing to debug for no gain.
 *
 * Registration failure is logged, not surfaced. Nothing in the console needs
 * the worker yet — it is here so the app can be added to the Home Screen, and
 * so PR 9 has somewhere to put the push handler. */
if (import.meta.env.PROD && "serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker
      .register("/sw.js", { scope: "/" })
      .catch((err) => console.warn("service worker registration failed", err));
  });
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
      <Toaster />
    </QueryClientProvider>
  </StrictMode>,
);
