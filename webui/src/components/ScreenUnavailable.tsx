/* What a direct link to a screen this install does not have shows (OSS plan R17).
 *
 * The nav only lists screens whose owner is mounted, but a bookmark, a push tap or a
 * pasted URL can still reach a route whose plugin is not. Drawing that screen would
 * call an API nobody serves and show a broken page; this says what is missing and,
 * when the harness knows the plugin, why it is not running. */
import { Link } from "react-router-dom";
import { Puzzle } from "lucide-react";
import type { UnavailableScreen } from "@/lib/nav";

export function ScreenUnavailable({ entry }: { entry: UnavailableScreen | undefined }) {
  return (
    <div
      role="status"
      className="mx-auto mt-8 max-w-md rounded-xl border border-border bg-surface p-5 text-sm"
    >
      <div className="flex items-center gap-2 text-fg">
        <Puzzle size={18} className="shrink-0 text-fg-muted" aria-hidden />
        <h2 className="font-semibold">
          {entry ? `${entry.label} isn't available` : "This screen isn't part of this install"}
        </h2>
      </div>
      {entry ? (
        <p className="mt-2 break-words text-fg-muted">
          It belongs to the <code className="font-mono text-fg">{entry.plugin}</code> plugin,
          which isn't installed in this profile ({entry.reason}).
        </p>
      ) : (
        <p className="mt-2 text-fg-muted">
          No installed plugin provides it, so there is nothing to show here.
        </p>
      )}
      <p className="mt-2 text-fg-subtle">
        Plugins and the profile that mounts them are listed under{" "}
        <Link to="/agents" className="text-primary underline-offset-2 hover:underline">
          Agents
        </Link>
        .
      </p>
      <Link
        to="/chat"
        className="mt-4 inline-flex min-h-[44px] items-center rounded-lg border border-border px-3 text-fg hover:bg-surface-raised"
      >
        Back to Chat
      </Link>
    </div>
  );
}
