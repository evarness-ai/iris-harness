import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";
import react from "eslint-plugin-react";

// Guardrails for the design system (webui/design.md):
//   - no raw hex colors  -> use design tokens (bg-surface, text-fg, ...)
//   - no inline `style`  -> use token utility classes
// React-Flow custom nodes / bars set per-node size/opacity/color at runtime, so
// inline `style` is unavoidable in these — but their COLORS come from design
// tokens (rgb(var(--...))), so the no-raw-hex rule still applies. Only the
// inline-style rule is relaxed here, and only for these files.
const INLINE_STYLE_FILES = [
  "src/components/IRISNode.tsx",
  "src/components/NodeDetail.tsx",
  "src/components/Reasoning.tsx",
  "src/components/Waterfall.tsx",
];

export default tseslint.config(
  { ignores: ["dist", "node_modules"] },
  {
    files: ["src/**/*.{ts,tsx}"],
    languageOptions: {
      parser: tseslint.parser,
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
    plugins: { "react-hooks": reactHooks, react },
    settings: { react: { version: "18" } },
    rules: {
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
      "react/forbid-dom-props": ["error", { forbid: ["style"] }],
      "no-restricted-syntax": [
        "error",
        {
          selector: "Literal[value=/#[0-9a-fA-F]{6}/]",
          message:
            "No raw hex — use a design token (bg-surface, text-fg, border-border, ...). See webui/design.md.",
        },
      ],
    },
  },
  // shadcn vendored primitives — they follow their own conventions (token-themed).
  {
    files: ["src/components/ui/**"],
    rules: { "react/forbid-dom-props": "off", "no-restricted-syntax": "off" },
  },
  // React-Flow nodes/bars: inline `style` is unavoidable (runtime geometry).
  // Colors still come from tokens, so no-raw-hex stays on.
  {
    files: INLINE_STYLE_FILES,
    rules: { "react/forbid-dom-props": "off" },
  },
);
