/* Which harness is this UI showing?
 *
 * The cloud trial (docs/architecture/cloud-trial-plan.md) runs a second harness on a
 * VM. Every number on screen — sessions, costs, health, heartbeats — belongs to the
 * machine that answers the data calls. Reading the VM's memory growth as the laptop's
 * would be a silent, expensive mistake, so the origin is always on screen when it is
 * not the ordinary local harness.
 *
 * The name comes from two places:
 *
 * - at runtime, `deployment_label` from /capabilities (IRIS_DEPLOYMENT_LABEL on the
 *   server). A console served by the API out of the server image is built once for
 *   every deployment, so only the server can say which one it is.
 * - at build time, VITE_IRIS_TARGET_LABEL: the local Vite UI pointed at another
 *   harness with IRIS_API_URL (and IRIS_API_LABEL). It is the fallback for
 *   a server that sets no label, or one whose /capabilities cannot be read yet.
 *
 * Nothing renders in the ordinary local case, so the usual UI is untouched. */
import { useCapabilities } from "@/lib/queries";

const buildLabel = import.meta.env.VITE_IRIS_TARGET_LABEL as string | undefined;

export function TargetBadge() {
  const runtimeLabel = useCapabilities().data?.deployment_label;
  const label = runtimeLabel || buildLabel;
  if (!label) return null;

  return (
    <div
      // Desktop only. On a phone this row cost 29px on every screen to say
      // something that never changes; the header shows a dot instead, with
      // the label in its tooltip (App.tsx).
      className="hidden items-center gap-2 border-b border-accent/40 bg-accent/10 px-4 py-1.5 text-xs md:flex md:px-6"
      role="status"
    >
      <span className="font-medium uppercase tracking-wide">Remote harness</span>
      <span className="opacity-80">
        showing data from <code>{label}</code>
        {/* Only the local Vite UI aimed elsewhere is known to be running on the Mac. */}
        {buildLabel ? ", not this Mac" : null}
      </span>
    </div>
  );
}

/** The same fact as the badge, in 7px, for the phone header.
 *
 * "Which harness am I looking at" is worth a row on a desktop and worth a dot
 * on a phone — the answer never changes during a session, but reading the
 * VM's numbers as the laptop's stays an expensive mistake, so it is never
 * absent. The label rides in the tooltip and the accessible name. */
export function TargetDot() {
  const runtimeLabel = useCapabilities().data?.deployment_label;
  const label = runtimeLabel || buildLabel;
  if (!label) return null;

  return (
    <span
      role="status"
      aria-label={`Remote harness: ${label}`}
      title={`Remote harness — showing data from ${label}`}
      className="h-[7px] w-[7px] shrink-0 rounded-full bg-accent ring-4 ring-accent/15 md:hidden"
    />
  );
}
