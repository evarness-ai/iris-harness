import { createBrowserRouter, Navigate } from "react-router-dom";
import type { LucideIcon } from "lucide-react";
import {
  Activity,
  AlarmClock,
  Bot,
  Brain,
  CalendarClock,
  ChartLine,
  FileText,
  FlaskConical,
  HeartPulse,
  History,
  Inbox,
  LayoutDashboard,
  ListChecks,
  Mail,
  Menu,
  MessageSquare,
  MonitorSmartphone,
  Newspaper,
  Puzzle,
  Settings,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  UserRound,
  Waypoints,
} from "lucide-react";
import { AppLayout } from "./App";

/* Which screens exist, their labels, groups and order are NOT here: they come from
 * GET /api/v1/webui/nav (lib/nav.ts), the core's config/webui/nav.yaml plus every
 * mounted plugin's manifest (OSS plan R17). This file only knows how to draw them:
 * the icons a screen may name, and the component behind each route. */

/** The icons a nav declaration may name (kebab-case lucide names). */
const ICONS: Record<string, LucideIcon> = {
  activity: Activity,
  "alarm-clock": AlarmClock,
  bot: Bot,
  brain: Brain,
  "calendar-clock": CalendarClock,
  "chart-line": ChartLine,
  "file-text": FileText,
  "flask-conical": FlaskConical,
  "heart-pulse": HeartPulse,
  history: History,
  inbox: Inbox,
  "layout-dashboard": LayoutDashboard,
  "list-checks": ListChecks,
  mail: Mail,
  menu: Menu,
  "message-square": MessageSquare,
  "monitor-smartphone": MonitorSmartphone,
  newspaper: Newspaper,
  puzzle: Puzzle,
  settings: Settings,
  "shield-check": ShieldCheck,
  sparkles: Sparkles,
  stethoscope: Stethoscope,
  "user-round": UserRound,
  waypoints: Waypoints,
};

/** A declared icon, or a generic one for a name this bundle does not carry. */
export function iconFor(name: string): LucideIcon {
  return ICONS[name] ?? Puzzle;
}

// Each screen is its own lazy chunk (route.lazy) so the heavy graph screens
// (React Flow + dagre) don't load until visited — keeps the initial bundle lean.
export const router = createBrowserRouter([
  // Outside the shell on purpose: an unpaired browser lands here after a 401
  // (lib/http.ts), and the shell's own data calls would 401 again.
  { path: "pair", lazy: async () => ({ Component: (await import("./screens/Pair")).PairScreen }) },
  // Outside the shell on purpose: the first thing a new install shows, not a
  // screen inside the console you've already set up (own header, no sidebar).
  {
    path: "onboarding",
    lazy: async () => ({ Component: (await import("./screens/Onboarding")).OnboardingScreen }),
  },
  {
    element: <AppLayout />,
    children: [
      { index: true, element: <Navigate to="/chat" replace /> },
      { path: "chat", lazy: async () => ({ Component: (await import("./screens/Chat")).ChatScreen }) },
      { path: "more", lazy: async () => ({ Component: (await import("./screens/More")).MoreScreen }) },
      {
        path: "chat/:sessionId",
        lazy: async () => ({ Component: (await import("./screens/Chat")).ChatScreen }),
      },
      {
        path: "actions",
        lazy: async () => ({
          Component: (await import("./screens/ActionCenter")).ActionCenterScreen,
        }),
      },
      {
        path: "reminders/:reminderId",
        lazy: async () => ({ Component: (await import("./screens/Reminder")).ReminderScreen }),
      },
      {
        path: "digest",
        lazy: async () => ({ Component: (await import("./screens/Digest")).DigestScreen }),
      },
      {
        path: "digest/:digestId",
        lazy: async () => ({ Component: (await import("./screens/Digest")).DigestScreen }),
      },
      {
        path: "inbox",
        lazy: async () => ({ Component: (await import("./screens/Inbox")).InboxScreen }),
      },
      {
        path: "setup",
        lazy: async () => ({ Component: (await import("./screens/Setup")).SetupScreen }),
      },
      {
        path: "activity",
        lazy: async () => ({
          Component: (await import("./screens/ActivityFeed")).ActivityFeedScreen,
        }),
      },
      {
        path: "agents",
        lazy: async () => ({ Component: (await import("./screens/Agents")).AgentsScreen }),
      },
      {
        path: "agents/plugins/:name",
        lazy: async () => ({
          Component: (await import("./screens/PluginDetail")).PluginDetailScreen,
        }),
      },
      {
        path: "agents/:name",
        lazy: async () => ({
          Component: (await import("./screens/AgentDashboard")).AgentDashboardScreen,
        }),
      },
      {
        path: "health",
        lazy: async () => ({ Component: (await import("./screens/Health")).HealthScreen }),
      },
      {
        path: "system-check",
        lazy: async () => ({
          Component: (await import("./screens/SystemCheck")).SystemCheckScreen,
        }),
      },
      {
        path: "overview",
        lazy: async () => ({ Component: (await import("./screens/Overview")).OverviewScreen }),
      },
      {
        path: "learning",
        lazy: async () => ({ Component: (await import("./screens/Learning")).LearningScreen }),
      },
      {
        path: "twin",
        lazy: async () => ({ Component: (await import("./screens/DigitalTwin")).DigitalTwinScreen }),
      },
      {
        path: "routines",
        lazy: async () => ({ Component: (await import("./screens/Routines")).RoutinesScreen }),
      },
      {
        path: "portfolio",
        lazy: async () => ({ Component: (await import("./screens/Portfolio")).PortfolioScreen }),
      },
      {
        path: "heartbeats",
        lazy: async () => ({ Component: (await import("./screens/Heartbeats")).HeartbeatsScreen }),
      },
      {
        path: "documents",
        lazy: async () => ({ Component: (await import("./screens/Rag")).RagScreen }),
      },
      {
        path: "knowledge",
        lazy: async () => ({ Component: (await import("./screens/Knowledge")).KnowledgeScreen }),
      },
      {
        path: "calltrace",
        lazy: async () => ({ Component: (await import("./screens/CallTrace")).CallTraceScreen }),
      },
      {
        path: "calltrace/:traceId",
        lazy: async () => ({ Component: (await import("./screens/CallTrace")).CallTraceScreen }),
      },
      {
        path: "sessions",
        lazy: async () => ({ Component: (await import("./screens/Sessions")).SessionsScreen }),
      },
      {
        path: "governance",
        lazy: async () => ({ Component: (await import("./screens/Governance")).GovernanceScreen }),
      },
      {
        path: "devices",
        lazy: async () => ({ Component: (await import("./screens/Devices")).DevicesScreen }),
      },
      {
        path: "memory",
        lazy: async () => ({ Component: (await import("./screens/Memory")).MemoryScreen }),
      },
      {
        path: "playground",
        lazy: async () => ({ Component: (await import("./screens/Playground")).PlaygroundScreen }),
      },
      {
        path: "settings",
        lazy: async () => ({ Component: (await import("./screens/Settings")).SettingsScreen }),
      },
      { path: "*", element: <Navigate to="/chat" replace /> },
    ],
  },
]);
