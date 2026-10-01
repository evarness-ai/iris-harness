// Presentational replay transport bar. State lives in CallTrace.
export function Replay({
  playing,
  step,
  total,
  speed,
  onPlayPause,
  onStep,
  onReset,
  onSpeed,
}: {
  playing: boolean;
  step: number;
  total: number;
  speed: number;
  onPlayPause: () => void;
  onStep: (dir: 1 | -1) => void;
  onReset: () => void;
  onSpeed: (s: number) => void;
}) {
  const atEnd = step >= total;
  const ctrl =
    "rounded-lg border border-border bg-surface-raised px-2.5 py-1 text-xs text-fg-muted hover:bg-border";
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button type="button" onClick={onReset} className={ctrl} title="reset">
        ⏮
      </button>
      <button type="button" onClick={() => onStep(-1)} className={ctrl} title="step back">
        ◀
      </button>
      <button
        type="button"
        onClick={onPlayPause}
        className="rounded-lg bg-primary px-3 py-1 text-xs font-semibold text-primary-fg hover:opacity-90"
      >
        {playing ? "❚❚ Pause" : atEnd ? "↻ Replay" : "▶ Play"}
      </button>
      <button type="button" onClick={() => onStep(1)} className={ctrl} title="step forward">
        ▶
      </button>
      <span className="ml-1 font-mono text-[11px] text-fg-subtle">
        {Math.min(step, total)}/{total}
      </span>
      <div className="ml-3 flex items-center gap-2">
        <span className="text-[11px] text-fg-subtle">speed</span>
        <input
          type="range"
          min={0.5}
          max={4}
          step={0.5}
          value={speed}
          onChange={(e) => onSpeed(Number(e.target.value))}
          aria-label="Replay speed"
          title="Replay speed"
          className="w-24 accent-primary"
        />
        <span className="font-mono text-[11px] text-fg-muted">{speed}×</span>
      </div>
    </div>
  );
}
