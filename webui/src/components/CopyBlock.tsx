import { useState } from "react";

/** A labelled text block whose full content is copyable and expandable (never truncated). */
export function CopyBlock({ label, text, tone }: { label: string; text: string; tone?: string }) {
  const [copied, setCopied] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const copy = () => {
    navigator.clipboard?.writeText(text).then(
      () => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1200);
      },
      () => {},
    );
  };
  return (
    <div className="mt-2">
      <div className="mb-1 flex items-center justify-between">
        <span className="text-[11px] font-semibold text-fg-muted">{label}</span>
        <div className="flex gap-1.5">
          <button
            onClick={() => setExpanded((e) => !e)}
            className="inline-flex min-h-[44px] items-center rounded border border-border px-2.5 py-0.5 text-[10px] text-fg-muted hover:bg-surface-raised sm:min-h-0 sm:px-1.5"
          >
            {expanded ? "collapse" : "expand"}
          </button>
          <button
            onClick={copy}
            className="inline-flex min-h-[44px] items-center rounded border border-border px-2.5 py-0.5 text-[10px] text-fg-muted hover:bg-surface-raised sm:min-h-0 sm:px-1.5"
          >
            {copied ? "copied" : "copy"}
          </button>
        </div>
      </div>
      <pre
        className={`overflow-auto whitespace-pre-wrap break-words rounded bg-bg p-2 font-mono text-[11px] ${
          expanded ? "" : "max-h-52"
        } ${tone ?? "text-fg-muted"}`}
      >
        {text}
      </pre>
    </div>
  );
}
