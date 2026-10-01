import { Moon, Sun } from "lucide-react";
import { toggleTheme, useTheme } from "@/lib/theme";

/** Token-styled light/dark toggle. No inline styles, no raw hex. */
export function ThemeToggle() {
  const theme = useTheme();
  const isDark = theme === "dark";
  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
      title={isDark ? "Light theme" : "Dark theme"}
      className="inline-flex h-11 w-11 items-center justify-center rounded-md border border-border sm:h-8 sm:w-8 text-fg-muted transition-colors hover:border-primary/50 hover:bg-primary/10 hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      {isDark ? <Sun size={16} /> : <Moon size={16} />}
    </button>
  );
}
