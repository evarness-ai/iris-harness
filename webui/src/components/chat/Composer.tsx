import { useEffect, useMemo, useRef, useState } from "react";
import { Send, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { SlashCommandSummary } from "@/lib/client";
import { useIsPhone } from "@/lib/breakpoint";

/** Message composer. Enter sends; Shift+Enter inserts a newline. */
export function Composer({
  value,
  onChange,
  onSend,
  onStop,
  busy,
  slashCommands,
  onRecallHistory,
}: {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  onStop: () => void;
  busy: boolean;
  slashCommands: SlashCommandSummary[];
  onRecallHistory: (direction: "up" | "down") => boolean;
}) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [activeSlashIndex, setActiveSlashIndex] = useState(0);
  const isPhone = useIsPhone();

  const slashMatches = useMemo(() => {
    const trimmed = value.trimStart();
    if (!trimmed.startsWith("/")) return [];
    const query = trimmed.toLowerCase();
    return slashCommands.filter((cmd) => cmd.name.toLowerCase().startsWith(query)).slice(0, 8);
  }, [slashCommands, value]);

  useEffect(() => {
    setActiveSlashIndex(0);
  }, [value]);

  const chooseSlash = (cmd: SlashCommandSummary) => {
    const next = cmd.args ? `${cmd.name} ${cmd.args}` : cmd.name;
    onChange(next);
    requestAnimationFrame(() => {
      const el = textareaRef.current;
      if (!el) return;
      el.focus();
      el.setSelectionRange(next.length, next.length);
    });
  };

  const canRecallHistory = (direction: "up" | "down") => {
    const el = textareaRef.current;
    if (!el) return false;
    const hasSelection = el.selectionStart !== el.selectionEnd;
    if (hasSelection) return false;
    if (value.includes("\n")) return false;
    if (direction === "up") return el.selectionStart === 0 && el.selectionEnd === 0;
    return el.selectionStart === value.length && el.selectionEnd === value.length;
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "ArrowUp" && slashMatches.length > 0 && value.trimStart().startsWith("/")) {
      e.preventDefault();
      setActiveSlashIndex((idx) => (idx <= 0 ? slashMatches.length - 1 : idx - 1));
      return;
    }
    if (e.key === "ArrowDown" && slashMatches.length > 0 && value.trimStart().startsWith("/")) {
      e.preventDefault();
      setActiveSlashIndex((idx) => (idx + 1) % slashMatches.length);
      return;
    }
    if (e.key === "Tab" && slashMatches.length > 0 && value.trimStart().startsWith("/")) {
      e.preventDefault();
      chooseSlash(slashMatches[activeSlashIndex] ?? slashMatches[0]);
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      if (slashMatches.length > 0 && value.trimStart().startsWith("/") && !busy) {
        e.preventDefault();
        chooseSlash(slashMatches[activeSlashIndex] ?? slashMatches[0]);
        return;
      }
      e.preventDefault();
      if (!busy && value.trim()) onSend();
      return;
    }
    if (!busy && (e.key === "ArrowUp" || e.key === "ArrowDown")) {
      const direction = e.key === "ArrowUp" ? "up" : "down";
      if (canRecallHistory(direction) && onRecallHistory(direction)) {
        e.preventDefault();
      }
    }
  };
  return (
    <div className="rounded-lg border border-border bg-surface p-2 focus-within:border-primary/50">
      {slashMatches.length > 0 && value.trimStart().startsWith("/") && (
        <div className="mb-2 rounded-md border border-border bg-bg/80 p-1">
          {slashMatches.map((cmd, index) => {
            const active = index === activeSlashIndex;
            return (
              <button
                key={cmd.name}
                type="button"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => chooseSlash(cmd)}
                className={`flex w-full items-start justify-between gap-3 rounded px-2 py-1.5 text-left ${
                  active ? "bg-primary/10" : "hover:bg-surface"
                }`}
              >
                <div className="min-w-0">
                  <div className="font-mono text-xs text-fg">
                    {cmd.name}
                    {cmd.args ? ` ${cmd.args}` : ""}
                  </div>
                  {cmd.description ? (
                    <div className="truncate text-[11px] text-fg-subtle">{cmd.description}</div>
                  ) : null}
                </div>
              </button>
            );
          })}
          <div className="px-2 pb-1 pt-1 text-[10px] text-fg-subtle">
            Tab or Enter to insert a command. Up/Down moves through commands.
          </div>
        </div>
      )}

      <div className="flex items-end gap-2">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          rows={1}
          // The hint describes a hardware keyboard, so on a phone it is both
          // wrong and too long: it wrapped to a second line the 36px textarea
          // could not show, clipping 6px of it.
          placeholder={
            isPhone ? "Message IRIS…" : "Message IRIS…  (Enter to send · Shift+Enter for a new line)"
          }
          className="max-h-40 min-h-[2.25rem] flex-1 resize-none bg-transparent px-2 py-1.5 text-sm text-fg placeholder:text-fg-subtle focus:outline-none"
        />
        {busy ? (
          <Button type="button" variant="outline" size="icon" onClick={onStop} aria-label="Stop">
            <Square />
          </Button>
        ) : (
          <Button
            type="button"
            size="icon"
            onClick={onSend}
            disabled={!value.trim()}
            aria-label="Send message"
          >
            <Send />
          </Button>
        )}
      </div>
    </div>
  );
}
