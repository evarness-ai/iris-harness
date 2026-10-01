/* Theme manager — toggles the `.dark` class on <html> and persists the choice.
 * Dark is the default (matches the index.html class + the IRIS console identity).
 * Framework-light: a tiny external store consumed via useSyncExternalStore. */
import { useSyncExternalStore } from "react";

export type Theme = "dark" | "light";

const STORAGE_KEY = "iris-theme";
const listeners = new Set<() => void>();

function read(): Theme {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "dark" || saved === "light") return saved;
  } catch {
    /* localStorage unavailable (privacy mode / SSR) — fall back to default */
  }
  return "dark";
}

let current: Theme = read();

function apply(theme: Theme): void {
  const root = document.documentElement;
  root.classList.toggle("dark", theme === "dark");
}

// Reconcile the DOM with the persisted choice on load (index.html defaults to dark).
apply(current);

export function getTheme(): Theme {
  return current;
}

export function setTheme(theme: Theme): void {
  current = theme;
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    /* ignore persistence failure */
  }
  apply(theme);
  listeners.forEach((l) => l());
}

export function toggleTheme(): void {
  setTheme(current === "dark" ? "light" : "dark");
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** React hook returning the current theme; re-renders on change. */
export function useTheme(): Theme {
  return useSyncExternalStore(subscribe, getTheme, getTheme);
}
