import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

/** A pending write action awaiting explicit user approval. */
export interface ConfirmState {
  title: string;
  description?: string;
  confirmLabel?: string;
  destructive?: boolean;
  /** Performs the write. Throw to keep the dialog open (e.g. on error). */
  run: () => Promise<void>;
}

/**
 * The approval gate for every UI-driven write. Controlled: a screen holds a
 * `ConfirmState | null` and renders one of these; opening it = asking the user
 * to approve. On confirm it awaits `run()` and closes on success; if `run`
 * throws the dialog stays open so the error toast is visible.
 */
export function ConfirmDialog({
  state,
  onClose,
}: {
  state: ConfirmState | null;
  onClose: () => void;
}) {
  const [pending, setPending] = useState(false);

  const run = async () => {
    if (!state) return;
    setPending(true);
    try {
      await state.run();
      onClose();
    } catch {
      /* keep open; the caller surfaces the error */
    } finally {
      setPending(false);
    }
  };

  return (
    <Dialog
      open={state !== null}
      onOpenChange={(open) => {
        if (!open && !pending) onClose();
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{state?.title}</DialogTitle>
          {state?.description && <DialogDescription>{state.description}</DialogDescription>}
        </DialogHeader>
        <DialogFooter>
          <Button type="button" variant="outline" onClick={onClose} disabled={pending}>
            Cancel
          </Button>
          <Button
            type="button"
            variant={state?.destructive ? "destructive" : "default"}
            onClick={run}
            disabled={pending}
          >
            {pending ? "Working…" : (state?.confirmLabel ?? "Confirm")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
