/* The phone's bottom tab bar (plan decisions 19 and 26).
 *
 * Four pinned routes plus More, below `md` only — the desktop keeps its sidebar.
 * The pins are the nav's `mobile_tabs` (a core screen's `mobile_tab` in
 * config/webui/nav.yaml), so pinning a screen is one YAML field, and More lists
 * every group, so a screen added to the nav reaches the phone with no edit here.
 *
 * It is the last child of the shell's `h-[100dvh]` flex column rather than
 * `position: fixed`. Fixed would sit on top of the content, and the screen that
 * asks for the whole viewport (Chat) would put its composer underneath it — the
 * #562/#564 bug, one layer down. As a `flex-none` sibling it takes its height
 * out of the column and the content area gets what is left. */
import { NavLink } from "react-router-dom";
import { Menu } from "lucide-react";
import { iconFor } from "@/routes";
import { useActionCount } from "@/lib/queries";
import { allScreens, useNav } from "@/lib/nav";

/** Unread-style count, capped so a runaway number cannot widen the tab. */
function TabBadge({ count }: { count: number }) {
  if (count <= 0) return null;
  return (
    <span
      aria-hidden
      className="absolute right-[calc(50%-18px)] top-1 min-w-[15px] rounded-full bg-danger px-1 text-center text-[9px] font-bold leading-[15px] text-white"
    >
      {count > 99 ? "99+" : count}
    </span>
  );
}

const TAB_CLASS =
  "relative flex flex-col items-center gap-0.5 py-2 text-fg-subtle transition-colors";

export function BottomTabs() {
  const actionCount = useActionCount();
  const nav = useNav().data;
  const icons = new Map((nav ? allScreens(nav) : []).map((s) => [s.route, s.icon]));

  return (
    <nav
      aria-label="Sections"
      role="tablist"
      className="flex-none grid grid-cols-5 border-t border-border bg-sidebar pb-[env(safe-area-inset-bottom,0px)] md:hidden"
    >
      {(nav?.mobile_tabs ?? []).map((item) => {
        const Icon = iconFor(icons.get(item.route) ?? "");
        const label = item.label;
        // The badge belongs to what the tab opens, not to its name: Activity is
        // /actions, so it carries the pending-action count the sidebar shows.
        const badge = item.route === "/actions" ? actionCount : 0;
        return (
          <NavLink
            key={item.route}
            to={item.route}
            role="tab"
            aria-label={label}
            className={({ isActive }) => `${TAB_CLASS} ${isActive ? "text-primary" : ""}`}
          >
            {({ isActive }) => (
              <>
                <Icon size={21} strokeWidth={isActive ? 2.2 : 1.7} aria-hidden />
                <span className="text-[10px]">{label}</span>
                <TabBadge count={badge} />
              </>
            )}
          </NavLink>
        );
      })}
      <NavLink
        to="/more"
        role="tab"
        aria-label="More"
        className={({ isActive }) => `${TAB_CLASS} ${isActive ? "text-primary" : ""}`}
      >
        {({ isActive }) => (
          <>
            <Menu size={21} strokeWidth={isActive ? 2.2 : 1.7} aria-hidden />
            <span className="text-[10px]">More</span>
          </>
        )}
      </NavLink>
    </nav>
  );
}
