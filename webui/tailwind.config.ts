import type { Config } from "tailwindcss";
import animate from "tailwindcss-animate";

// Token names map to the CSS variables in src/styles/tokens.css (channel form),
// which mirror webui/design.md. Components reference these names (bg-surface,
// text-fg-muted, rounded-lg, shadow-e2) — never raw hex. The shadcn aliases
// (background/foreground/card/muted/...) point at the SAME tokens so shadcn
// components theme to our system with no duplicate token set.
const c = (v: string) => `rgb(var(${v}) / <alpha-value>)`;

export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // IRIS semantic tokens
        bg: c("--bg"),
        sidebar: c("--sidebar"),
        surface: c("--surface"),
        "surface-raised": c("--surface-raised"),
        "border-strong": c("--border-strong"),
        fg: c("--fg"),
        "fg-muted": c("--fg-muted"),
        "fg-subtle": c("--fg-subtle"),
        "primary-fg": c("--primary-fg"),
        success: c("--success"),
        warning: c("--warning"),
        danger: c("--danger"),
        info: c("--info"),
        overlay: "var(--overlay)",
        node: {
          perception: c("--node-perception"),
          cognition: c("--node-cognition"),
          runtime: c("--node-runtime"),
          action: c("--node-action"),
          llm: c("--node-llm"),
          governance: c("--node-governance"),
        },
        // shadcn/Radix aliases -> our tokens (no duplicate values)
        border: c("--border"),
        input: c("--border"),
        ring: c("--ring"),
        background: c("--bg"),
        foreground: c("--fg"),
        primary: { DEFAULT: c("--primary"), foreground: c("--primary-fg") },
        secondary: { DEFAULT: c("--surface-raised"), foreground: c("--fg") },
        muted: { DEFAULT: c("--surface"), foreground: c("--fg-muted") },
        accent: { DEFAULT: c("--surface-raised"), foreground: c("--fg") },
        destructive: { DEFAULT: c("--danger"), foreground: c("--primary-fg") },
        card: { DEFAULT: c("--surface"), foreground: c("--fg") },
        popover: { DEFAULT: c("--surface-raised"), foreground: c("--fg") },
      },
      fontFamily: {
        sans: [
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      fontSize: {
        xs: ["12px", "16px"],
        sm: ["13px", "18px"],
        base: ["14px", "20px"],
        lg: ["16px", "24px"],
        xl: ["20px", "28px"],
        "2xl": ["28px", "34px"],
      },
      borderRadius: {
        sm: "var(--radius-sm)",
        md: "var(--radius-md)",
        lg: "var(--radius-lg)",
      },
      boxShadow: {
        e1: "var(--shadow-e1)",
        e2: "var(--shadow-e2)",
        e3: "var(--shadow-e3)",
      },
    },
  },
  plugins: [animate],
} satisfies Config;
