import { useEffect, useRef, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { ChevronRight, Lock } from "lucide-react";
import { toast } from "sonner";
import { iconFor } from "./routes";
import { ThemeToggle } from "@/components/ThemeToggle";
import { HealthBanner } from "@/components/HealthBanner";
import { TargetBadge, TargetDot } from "@/components/TargetBadge";
import { AttentionBell } from "@/components/AttentionBell";
import { BottomTabs } from "@/components/BottomTabs";
import { SessionHistoryPanel } from "@/components/SessionHistoryPanel";
import { newSession } from "@/lib/chat";
import { useActionCount, useActivities, useActivityBadge, useWritesEnabled } from "@/lib/queries";
import { findScreen, routeOf, unavailableFor, useNav, type NavGroup } from "@/lib/nav";
import { ScreenUnavailable } from "@/components/ScreenUnavailable";

/** Toast when a background Activity finishes, wherever the user is in the app —
 * a categorize/cleanup/organize job can complete while they're not on Chat or
 * Activity. Primed on first load so we don't toast for pre-existing history. */
function useActivityCompletionToasts() {
  const { data } = useActivities();
  const navigate = useNavigate();
  const seen = useRef<Map<string, string>>(new Map());
  const primed = useRef(false);
  useEffect(() => {
    if (!data) return;
    const acts = data.activities ?? [];
    if (!primed.current) {
      for (const a of acts) seen.current.set(a.id, a.status);
      primed.current = true;
      return;
    }
    for (const a of acts) {
      const prev = seen.current.get(a.id);
      if (prev === a.status) continue;
      seen.current.set(a.id, a.status);
      if (a.status === "completed" && prev !== "completed") {
        toast.success(`${a.title} — done`, {
          description: "Open Activity to review the result.",
          action: { label: "Activity", onClick: () => navigate("/activity") },
        });
      } else if (a.status === "failed" && prev !== "failed") {
        toast.error(`${a.title} failed`, { description: a.error || undefined });
      }
    }
  }, [data, navigate]);
}

const COLLAPSED_KEY = "iris.nav.collapsed";
const CHAT_HISTORY_KEY = "iris.nav.chatHistoryOpen";

function readCollapsed(): Set<string> {
  try {
    const raw = localStorage.getItem(COLLAPSED_KEY);
    return new Set(raw ? (JSON.parse(raw) as string[]) : []);
  } catch {
    return new Set();
  }
}

function readChatHistoryOpen(): boolean {
  try {
    return localStorage.getItem(CHAT_HISTORY_KEY) === "1";
  } catch {
    return false;
  }
}

function CountBadge({ count }: { count: number }) {
  if (count <= 0) return null;
  return (
    <span className="ml-auto inline-flex min-w-[1.25rem] items-center justify-center rounded-full bg-primary/15 px-1.5 py-0.5 text-[10px] font-semibold text-primary">
      {count}
    </span>
  );
}

function NavList({
  onPick,
  chatHistoryOpen,
  setChatHistoryOpen,
}: {
  onPick?: () => void;
  chatHistoryOpen: boolean;
  setChatHistoryOpen: (value: boolean) => void;
}) {
  const actionCount = useActionCount();
  const activityCount = useActivityBadge();
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const nav = useNav();
  const activeGroup = nav.data ? findScreen(nav.data, pathname)?.screen.group : undefined;
  const [collapsed, setCollapsed] = useState<Set<string>>(readCollapsed);
  const onChatRoute = pathname.startsWith("/chat");
  const activeSessionId = pathname.match(/^\/chat\/([^/]+)/)?.[1] ?? "";

  const badgeFor = (path: string) =>
    path === "/actions" ? actionCount : path === "/activity" ? activityCount : 0;

  const setGroupCollapsed = (id: string, value: boolean) =>
    setCollapsed((prev) => {
      if (prev.has(id) === value) return prev;
      const next = new Set(prev);
      if (value) next.add(id);
      else next.delete(id);
      try {
        localStorage.setItem(COLLAPSED_KEY, JSON.stringify([...next]));
      } catch {
        // storage unavailable — collapse state just won't persist
      }
      return next;
    });

  // Navigating into a collapsed group (link, toast, deep URL) reopens it.
  useEffect(() => {
    if (activeGroup) setGroupCollapsed(activeGroup, false);
  }, [activeGroup]);

  // Entering Chat (not every session switch within it, since onChatRoute only
  // flips on the way in) opens its history the same way an active group
  // reopens above -- a manual collapse afterward sticks, same as a group's.
  useEffect(() => {
    if (onChatRoute) setChatHistoryOpen(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onChatRoute]);

  const renderGroup = (group: NavGroup) => {
    const open = !group.label || !collapsed.has(group.id);
    const groupBadge = group.items.reduce((n, item) => n + badgeFor(item.route), 0);
    return (
      <div key={group.id} className={group.label ? "mt-3" : undefined}>
        {group.label && (
          <button
            type="button"
            onClick={() => setGroupCollapsed(group.id, open)}
            aria-expanded={open}
            className={`flex w-full items-center gap-1 rounded-md px-2 py-1 text-[10.5px] font-semibold uppercase tracking-wider transition-colors hover:text-fg ${
              !open && group.id === activeGroup ? "text-primary" : "text-fg-subtle"
            }`}
          >
            <ChevronRight
              size={12}
              className={`shrink-0 transition-transform ${open ? "rotate-90" : ""}`}
            />
            <span className="truncate">{group.label}</span>
            {!open && <CountBadge count={groupBadge} />}
          </button>
        )}
        {open && (
          <div className={`flex flex-col gap-0.5 ${group.label ? "mt-1" : ""}`}>
            {group.items.map((item) => {
              const Icon = iconFor(item.icon);
              const link = (
                <NavLink
                  to={item.route}
                  onClick={onPick}
                  className={({ isActive }) =>
                    `flex min-w-0 flex-1 items-center gap-2.5 rounded-lg border-l-2 px-3 py-1.5 text-left text-sm transition-colors ${
                      isActive
                        ? "border-l-primary bg-primary/10 font-medium text-primary"
                        : "border-l-transparent text-fg-muted hover:bg-surface hover:text-fg"
                    }`
                  }
                >
                  <Icon size={15} className="shrink-0" aria-hidden />
                  <span className="truncate">{item.label}</span>
                  <CountBadge count={badgeFor(item.route)} />
                </NavLink>
              );
              // Chat alone gets an expand chevron: its session history nests
              // right under it instead of living as its own column in the
              // Chat screen (full width for the conversation there instead).
              if (item.route !== "/chat") return <div key={item.route}>{link}</div>;
              return (
                <div key={item.route}>
                  <div className="flex items-center">
                    {link}
                    <button
                      type="button"
                      onClick={() => setChatHistoryOpen(!chatHistoryOpen)}
                      aria-expanded={chatHistoryOpen}
                      aria-label={chatHistoryOpen ? "Hide chat history" : "Show chat history"}
                      className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-fg-subtle hover:bg-surface hover:text-fg"
                    >
                      <ChevronRight
                        size={14}
                        className={`transition-transform ${chatHistoryOpen ? "rotate-90" : ""}`}
                      />
                    </button>
                  </div>
                  {chatHistoryOpen && (
                    <SessionHistoryPanel
                      active={activeSessionId}
                      onSelect={(id) => navigate(`/chat/${id}`)}
                      onNew={() => navigate(`/chat/${newSession()}`)}
                    />
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  return (
    <nav className="flex flex-col" aria-label="Main">
      <div className="mb-3 px-2 pt-1">
        <div className="text-base font-bold tracking-tight text-primary">IRIS</div>
        <div className="text-[10px] uppercase tracking-widest text-fg-subtle">Console</div>
      </div>
      {nav.data?.groups.map(renderGroup)}
      {nav.isError && (
        <div className="mt-3 px-2 text-xs text-fg-subtle">
          Navigation unavailable: the IRIS API did not answer.{" "}
          <button
            type="button"
            onClick={() => void nav.refetch()}
            className="text-primary underline-offset-2 hover:underline"
          >
            Retry
          </button>
        </div>
      )}
    </nav>
  );
}

export function AppLayout() {
  const { pathname } = useLocation();
  const writesEnabled = useWritesEnabled();
  useActivityCompletionToasts();
  const nav = useNav();
  const found = nav.data ? findScreen(nav.data, pathname) : undefined;
  const current = found?.screen;
  const currentGroup = found?.group;
  // A route whose owner is not mounted is not drawn: nothing serves its API. Until
  // the nav answers nothing is drawn either, so such a screen never flashes up. If
  // the nav cannot be read at all, screens render as before and report their own
  // API errors.
  const gated = nav.data !== undefined && routeOf(pathname) !== "/" && current === undefined;

  // Owned here, not in NavList: the sidebar's own width (below) needs it too.
  // Widens only while open, so every other screen's nav keeps its usual 212px.
  const [chatHistoryOpen, setChatHistoryOpenState] = useState(readChatHistoryOpen);
  const setChatHistoryOpen = (value: boolean) => {
    setChatHistoryOpenState(value);
    try {
      localStorage.setItem(CHAT_HISTORY_KEY, value ? "1" : "0");
    } catch {
      // storage unavailable — the open/closed state just won't persist
    }
  };

  return (
    <div
      className={`min-h-[100dvh] bg-bg text-fg md:grid ${
        chatHistoryOpen ? "md:grid-cols-[280px_1fr]" : "md:grid-cols-[212px_1fr]"
      }`}
    >
      {/* Desktop sidebar */}
      <aside className="hidden border-r border-border bg-sidebar p-3 md:sticky md:top-0 md:block md:h-screen md:overflow-y-auto">
        <NavList chatHistoryOpen={chatHistoryOpen} setChatHistoryOpen={setChatHistoryOpen} />
      </aside>

      {/* No mobile drawer: below `md` the bottom bar pins four routes and its
          More tab opens the full grouped list (/more, plan decision 19). A
          hamburger as well would be a second, hidden way to reach the same
          pages. */}

      {/* A column of *definite* height, so the one screen that wants the whole
          viewport (Chat) can ask for what is left instead of guessing the
          chrome's height. `min-h` is not enough: a `flex-1` child resolves to
          its content against a minimum, so Chat's internally-scrolling panes
          would grow the page instead of scrolling. Long screens scroll in the
          wrapper below rather than in the document. */}
      <main className="flex h-[100dvh] min-w-0 flex-col">
        {/* Tight on a phone, unchanged above `md`. py-3 plus a 44px control
            made this 69px, on the screen where vertical room is scarcest; the
            padding gives way, the tap target does not. */}
        <header className="sticky top-0 z-30 flex items-center gap-2 border-b border-border bg-bg px-3 py-1 backdrop-blur md:gap-3 md:px-6 md:py-3">
          <div className="flex min-w-0 items-center gap-2 md:items-baseline">
            <TargetDot />
            {currentGroup?.label && (
              <span className="hidden shrink-0 text-xs text-fg-subtle sm:inline">
                {currentGroup.label} /
              </span>
            )}
            <h1 className="truncate text-sm font-semibold text-fg">{current?.title ?? ""}</h1>
          </div>
          <div className="hidden min-w-0 truncate text-xs text-fg-muted lg:block">
            {current?.subtitle}
          </div>
          {!writesEnabled && (
            <span
              title="Editing is off. Set IRIS_WEBUI_ALLOW_WRITES=1 to enable."
              className="ml-2 inline-flex shrink-0 items-center gap-1 rounded-full border border-border px-2 py-0.5 text-[10.5px] font-medium text-fg-subtle"
            >
              <Lock size={11} /> read-only
            </span>
          )}
          <div className="ml-auto flex shrink-0 items-center gap-1.5">
            <AttentionBell />
            <ThemeToggle />
          </div>
        </header>
        <TargetBadge />
        <HealthBanner />
        {/* Definite height (flex-1 of a definite-height main) and its own
            scrollbar. Deliberately NOT a flex column: a screen taller than the
            viewport would then be a flex item and get *shrunk* to fit instead
            of scrolling. As a block it simply overflows, and a screen that
            wants the full height asks for `h-full`. */}
        <div
          className={`min-h-0 flex-1 overflow-y-auto md:p-5 ${
            // Chat is a fixed-height column whose padding comes straight out
            // of the conversation; every other screen scrolls and wants the
            // margin.
            pathname.startsWith("/chat") ? "px-3 pb-3 pt-2" : "p-4"
          }`}
        >
          {gated && nav.data ? (
            <ScreenUnavailable entry={unavailableFor(nav.data, pathname)} />
          ) : nav.isPending ? null : (
            <Outlet />
          )}
        </div>
        <BottomTabs />
      </main>
    </div>
  );
}
