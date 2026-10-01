/* An on/off switch (role="switch"), shared by the Heartbeats and Settings screens. */
export function Switch({
  checked,
  disabled,
  label,
  onChange,
}: {
  checked: boolean;
  disabled?: boolean;
  label: string;
  onChange: (next: boolean) => void;
}) {
  // The track is 24px, but the tap target is the 44px the phone needs (Track 2 rule,
  // enforced by the viewport smoke); on a desktop pointer it shrinks to the track.
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className="inline-flex min-h-[44px] min-w-[44px] shrink-0 items-center justify-center disabled:cursor-not-allowed disabled:opacity-45 sm:min-h-0 sm:min-w-0"
    >
      <span
        className={`relative inline-flex h-6 w-10 items-center rounded-full transition-colors ${
          checked ? "bg-primary" : "bg-border"
        }`}
      >
        <span
          className={`inline-block h-[18px] w-[18px] rounded-full bg-white shadow transition-transform ${
            checked ? "translate-x-[19px]" : "translate-x-[3px]"
          }`}
        />
      </span>
    </button>
  );
}
