/* The phone's page list (plan decision 19).
 *
 * Every nav group (GET /api/v1/webui/nav), in sidebar order, so a screen added to
 * the nav reaches the phone without touching the bottom bar. The four pinned
 * routes stay listed and are marked, rather than hidden: a list that silently
 * omits what the bar already holds makes the owner hunt for a screen that is one
 * tap away.
 *
 * Phone-only. The desktop sidebar already shows all of this, so /more is not a
 * sidebar entry (`nav: false` in config/webui/nav.yaml) and redirects above `md`. */
import { useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ChevronRight } from "lucide-react";
import { iconFor } from "@/routes";
import { useActionCount } from "@/lib/queries";
import { useNav } from "@/lib/nav";
import { useIsDesktop } from "@/lib/breakpoint";

/** Above `md` the sidebar is showing, and this page would be a dead end. */
function useRedirectOnDesktop() {
  const navigate = useNavigate();
  const isDesktop = useIsDesktop();
  useEffect(() => {
    if (isDesktop) navigate("/chat", { replace: true });
  }, [isDesktop, navigate]);
}

export function MoreScreen() {
  useRedirectOnDesktop();
  const actionCount = useActionCount();
  const groups = useNav().data?.groups ?? [];

  return (
    <div className="space-y-5">
      {groups.map((group) => (
        <div key={group.id}>
          {group.label && (
            <h2 className="px-0.5 pb-1.5 text-[10px] font-bold uppercase tracking-widest text-fg-subtle">
              {group.label}
            </h2>
          )}
          <div className="space-y-1.5">
            {group.items.map((item) => {
              const Icon = iconFor(item.icon);
              const pinned = item.mobile_tab?.label;
              const badge = item.route === "/actions" ? actionCount : 0;
              return (
                <Link
                  key={item.route}
                  to={item.route}
                  className="flex min-h-[46px] items-center gap-3 rounded-lg border border-border bg-surface px-3 py-2.5 text-fg transition-colors hover:bg-surface-raised"
                >
                  <Icon size={17} className="shrink-0 text-fg-muted" aria-hidden />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[13px] font-medium">{item.label}</span>
                    {item.subtitle && (
                      <span className="block truncate text-[10.5px] text-fg-subtle">
                        {item.subtitle}
                      </span>
                    )}
                  </span>
                  {badge > 0 && (
                    <span className="shrink-0 rounded-full bg-danger px-1.5 text-[10px] font-bold leading-[17px] text-white">
                      {badge > 99 ? "99+" : badge}
                    </span>
                  )}
                  {pinned ? (
                    <span className="shrink-0 rounded-full bg-primary/15 px-2 py-0.5 text-[9px] font-bold uppercase tracking-wide text-primary">
                      {pinned}
                    </span>
                  ) : (
                    <ChevronRight size={16} className="shrink-0 text-fg-subtle" aria-hidden />
                  )}
                </Link>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
