---
# IRIS console design system — the single source of truth for tokens.
# Format: google-labs-code/design.md (YAML tokens + canonical prose below).
# Edit tokens HERE; the Tailwind theme + CSS variables mirror these names
# (src/styles/tokens.css). Lint with: npm run tokens:lint  (WCAG-AA contrast).
# Values below are the DARK (default) theme; light overrides are in § Colors.
name: IRIS Console
version: 0.1.0

colors:
  # Surfaces (back to front)
  bg: "#0c0f14"            # app background
  sidebar: "#0a0d12"       # nav rail
  surface: "#141923"       # cards, panels
  surfaceRaised: "#1b2230" # popovers, menus, dialogs
  border: "#232b39"        # hairlines, dividers
  borderStrong: "#33405270" # emphasized borders / inputs

  # Foreground
  fg: "#e9eef5"            # primary text
  fgMuted: "#9aa7b8"       # secondary text
  fgSubtle: "#6b7889"      # tertiary / disabled

  # Brand + interaction
  primary: "#5dcaa5"       # accent (active nav, primary actions)
  primaryFg: "#06241b"     # text/icon on primary fills
  ring: "{colors.primary}" # focus ring

  # Status / semantic
  success: "#5dcaa5"
  warning: "#e0a458"
  danger: "#ed6a8b"
  info: "#6fb4f0"

  # Trace node palette (mirrors the Call Trace node kinds)
  nodePerception: "#6fb4f0" # gateway / api / response_curator
  nodeCognition: "#e0a458"  # intent_router / task_planner
  nodeRuntime: "#7f77dd"    # runtime / agent_executor
  nodeAction: "#5dcaa5"     # memory / agent / tool
  nodeLlm: "#c98bdb"        # llm
  nodeGovernance: "#ed6a8b" # governance

typography:
  fontSans: "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"
  fontMono: "ui-monospace, SFMono-Regular, Menlo, monospace"
  xs: { fontSize: "12px", lineHeight: "16px" }
  sm: { fontSize: "13px", lineHeight: "18px" }
  base: { fontSize: "14px", lineHeight: "20px" }
  lg: { fontSize: "16px", lineHeight: "24px" }
  xl: { fontSize: "20px", lineHeight: "28px" }
  "2xl": { fontSize: "28px", lineHeight: "34px" }

spacing:
  px: "1px"
  "1": "4px"
  "2": "8px"
  "3": "12px"
  "4": "16px"
  "5": "20px"
  "6": "24px"
  "8": "32px"
  "12": "48px"

rounded:
  sm: "6px"
  md: "10px"
  lg: "14px"
  full: "9999px"

elevation:
  e1: "0 1px 2px rgb(0 0 0 / 0.30)"
  e2: "0 4px 12px rgb(0 0 0 / 0.35)"
  e3: "0 12px 32px rgb(0 0 0 / 0.45)"

components:
  button:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.primaryFg}"
    rounded: "{rounded.md}"
    padding: "{spacing.2} {spacing.4}"
    typography: "{typography.sm}"
  buttonGhost:
    backgroundColor: "transparent"
    textColor: "{colors.fgMuted}"
    rounded: "{rounded.md}"
  card:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.fg}"
    border: "{colors.border}"
    rounded: "{rounded.lg}"
    padding: "{spacing.4}"
  input:
    backgroundColor: "{colors.bg}"
    textColor: "{colors.fg}"
    border: "{colors.border}"
    rounded: "{rounded.md}"
    padding: "{spacing.2} {spacing.3}"
  tag:
    backgroundColor: "{colors.surfaceRaised}"
    textColor: "{colors.fgMuted}"
    rounded: "{rounded.full}"
    typography: "{typography.xs}"
---

# IRIS Console — Design System

## Overview

The IRIS console is an **operator surface** for a privacy-first agentic system:
trace visualization, chat, control panels, and (later) RAG + knowledge-graph
views. The aesthetic is a calm, information-dense **dark-first** console — high
signal, low chrome — with a fully supported light theme.

This file is the **single source of truth** for design tokens. Components never
hardcode colors, sizes, or inline CSS; they consume token-named Tailwind classes
(`bg-surface`, `text-fg-muted`, `rounded-lg`, …) that resolve to the CSS variables
in `src/styles/tokens.css`. To change the look, edit the tokens here and keep the
CSS variables in sync (and run `npm run tokens:lint`).

## Colors

Semantic roles, not raw hex, are what components reference. The front-matter holds
the **dark** (default) values; the **light** theme overrides the surface/foreground
roles below, and darkens the status colors and the two node hues a Tag uses as text:

| Role | Dark | Light |
|---|---|---|
| `bg` | `#0c0f14` | `#f6f8fb` |
| `sidebar` | `#0a0d12` | `#eef2f7` |
| `surface` | `#141923` | `#ffffff` |
| `surfaceRaised` | `#1b2230` | `#ffffff` |
| `border` | `#232b39` | `#dde3ec` |
| `fg` | `#e9eef5` | `#1a2230` |
| `fgMuted` | `#9aa7b8` | `#516072` |
| `fgSubtle` | `#6b7889` | `#7a8798` |
| `primary` | `#5dcaa5` | `#1f9e78` (darkened for AA on light) |
| `success` | `#5dcaa5` | `#1e6a52` |
| `warning` | `#e0a458` | `#83531e` |
| `danger` | `#ed6a8b` | `#af2348` |
| `info` | `#6fb4f0` | `#27609f` |
| `nodeCognition` | `#e0a458` | `#825318` |
| `nodeRuntime` | `#7f77dd` | `#5348d1` |

A **Tag** sets its text in a status or node color on a 15% tint of that same color.
On light grounds the dark-theme hues read at 1.8–3.9:1 that way, so each light value
above is the same hue darkened until the Tag reads at WCAG-AA 4.5:1 or better over
`surface`, `bg` and `sidebar`. `npm run tokens:lint` checks only the front-matter (dark)
values; `tests/setup.spec.ts` measures the rendered Tags in both themes. The other
node hues (`nodePerception`, `nodeAction`, `nodeLlm`, `nodeGovernance`) are fills and
accents, never Tag text, and stay shared across themes.

## Typography

One family (`fontSans`, the native OS UI font stack — no webfont to ship, best for
CSP + offline + mobile) and a mono family for code/IDs. The scale is compact — `base` is 14px — because this is a
dense console; `lg`/`xl`/`2xl` are reserved for section and page headings. Never
set font sizes ad hoc; use the scale tokens.

## Layout

Spacing is a 4px scale (`spacing.1`–`spacing.12`). Screens are composed from
reusable layout primitives — `Page`, `Section`, `Stack`, `Grid`, `Toolbar` — never
bespoke fl/grid markup per screen, so density and rhythm stay consistent and new
screens are composition, not copy-paste. The app shell is a fixed nav rail
(collapses to a drawer under `md`) + a scrollable content column.

## Elevation & Depth

Depth is conveyed by surface tiers (`bg` → `surface` → `surfaceRaised`) plus three
shadow tokens (`e1` resting cards, `e2` popovers/menus, `e3` modals). Avoid
borders *and* heavy shadows together; prefer a single hairline `border` on flat
surfaces and reserve shadows for floating layers.

## Shapes

Radii: `sm` (inputs, tags inline), `md` (buttons, inputs, menu items), `lg`
(cards, panels, dialogs), `full` (pills, avatars). Keep one radius per component
family; do not mix.

## Components

Component tokens above define the canonical fills, text, radius, and padding for
the primitive set. Implementations are **shadcn/ui (Radix) primitives** themed to
these tokens — one Button, one Card, one Dialog, etc., reused everywhere. Variants
(ghost, destructive, hover/active) are token-driven, not new components. Trace
graph nodes reuse the node palette via the same token names.

## Do's and Don'ts

- **Do** reference semantic token classes (`bg-surface`, `text-fg`, `border-border`).
- **Do** compose screens from the layout primitives.
- **Do** keep dark + light at WCAG-AA contrast (run `npm run tokens:lint`).
- **Don't** use inline `style={{…}}` for static styling, or raw hex in components
  (ESLint blocks both).
- **Don't** fork a primitive to tweak it — add a token-driven variant instead.
- **Don't** introduce a new color outside this file; add a token here first.
