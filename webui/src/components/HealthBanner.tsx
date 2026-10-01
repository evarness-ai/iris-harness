/* Global health banner (ADR-0069) — shows red, actionable alerts on every screen
 * so a background failure (a revoked token, a down service) surfaces without
 * opening the Health screen. Reads the same snapshot the screen does.
 *
 * Re-auth reuses the governed path (no bespoke OAuth endpoint): the button hands
 * the exact remediation command to the agent via the chat composer, which runs
 * it under the normal approval flow. */
import { Link } from "react-router-dom";
import { AlertTriangle } from "lucide-react";
import { useHealth } from "@/lib/queries";
import type { HealthCheck } from "@/lib/control";

/* A credential is re-authenticated; a service is restarted. Saying
 * "Re-authenticate" for a down service sent the agent a command that does not
 * match the fix (an unreachable Ollama is not a revoked token). */
function actionLabel(alert: HealthCheck): string {
  return alert.kind === "credential" ? "Re-authenticate" : "How to fix";
}

function askFor(alert: HealthCheck): string {
  if (!alert.action) return `Look into the ${alert.target} health alert: ${alert.detail}`;
  return alert.kind === "credential"
    ? `Please re-authenticate ${alert.target} by running: ${alert.action}`
    : `The ${alert.target} health check says "${alert.detail}". The suggested fix is: ${alert.action}. Check it and tell me what is wrong.`;
}

export function HealthBanner() {
  const { data } = useHealth();
  const alerts = data?.alerts ?? [];
  if (alerts.length === 0) return null;

  return (
    // Desktop only: on a phone the same alerts are in the header bell
    // (AttentionBell), which also carries a count this banner never had.
    <div className="hidden border-b border-danger/40 bg-danger/10 px-4 py-2 md:block md:px-6">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-3 gap-y-1 text-xs">
        <span className="inline-flex items-center gap-1.5 font-semibold text-danger">
          <AlertTriangle size={13} /> Needs attention
        </span>
        {alerts.map((a, i) => (
          <span key={`${a.target}:${i}`} className="inline-flex items-center gap-1.5">
            <span className="font-medium text-fg">{a.target}</span>
            <span className="text-fg-muted">{a.detail}</span>
            {a.fix_url?.startsWith("/") ? (
              // The page that fixes it (Settings > Connections for a revoked Google
              // token): no agent turn, no command to type.
              <Link
                to={a.fix_url}
                className="font-medium text-primary underline-offset-2 hover:underline"
              >
                {a.reconnect ? "Reconnect" : "Fix it"}
              </Link>
            ) : a.action ? (
              <Link
                to={`/chat?ask=${encodeURIComponent(askFor(a))}`}
                className="font-medium text-primary underline-offset-2 hover:underline"
              >
                {actionLabel(a)}
              </Link>
            ) : (
              <Link
                to="/health"
                className="font-medium text-primary underline-offset-2 hover:underline"
              >
                View Health
              </Link>
            )}
          </span>
        ))}
      </div>
    </div>
  );
}
