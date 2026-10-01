/** Removing things from memory (ADR-0119): the confirmation the Map opens, and the
 * Removed tab that lists, restores and permanently deletes.
 *
 * A removal is never only visual. Removing an entity forgets the facts that point at
 * it, removing a session takes it out of recall and the chat list, and removing a
 * summary mention suppresses that name everywhere. So the confirmation always asks the
 * server what will go (`/memory/removed/preview`) rather than guessing from the Map.
 * Every action here has a CLI twin (`iris memory remove / restore / removed / delete`). */
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tag } from "@/components/Tag";
import { Notice, QueryState, fmtDateTime } from "@/components/control/parts";
import { WriteGate } from "@/components/memory/tabs";
import {
  useDeleteRemoved,
  usePreviewRemoval,
  useRemoveFromMemory,
  useRemovedItems,
  useRestoreRemoved,
  useWritesEnabled,
} from "@/lib/queries";
import type { RemovableKind, RemovalTarget, RemovedItem } from "@/lib/control";

const KIND_LABEL: Record<RemovableKind, string> = {
  entity: "entity",
  name: "summary mention",
  session: "conversation",
  fact: "fact",
};

/** One target the owner picked on the Map, with the label they saw. */
export interface PickedTarget extends RemovalTarget {
  label: string;
}

/** Confirm a removal. Open while `targets` is non-empty. */
export function RemoveDialog({
  targets,
  onClose,
  onRemoved,
}: {
  targets: PickedTarget[];
  onClose: () => void;
  onRemoved: () => void;
}) {
  const preview = usePreviewRemoval();
  const remove = useRemoveFromMemory();
  const open = targets.length > 0;
  const key = targets.map((t) => `${t.kind}:${t.id}`).join("|");

  // Ask the server what goes with these, each time the set changes.
  useEffect(() => {
    if (!open) return;
    preview.reset();
    preview.mutate(targets.map(({ kind, id }) => ({ kind, id })));
    // `preview` is a fresh object each render; the key is what matters.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, open]);

  const effects = preview.data?.effects ?? [];
  const title =
    targets.length === 1 ? `Remove “${targets[0].label}”?` : `Remove ${targets.length} items?`;

  const confirm = () =>
    remove.mutate(
      targets.map(({ kind, id }) => ({ kind, id })),
      {
        onSuccess: (r) => {
          toast.success(`Removed ${r.items.length}. Restore it from the Removed tab.`);
          onRemoved();
        },
        onError: (e) => toast.error(`Remove failed: ${(e as Error).message}`),
      },
    );

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (!o && !remove.isPending) onClose();
      }}
    >
      <DialogContent className="max-h-[85vh] max-w-[calc(100vw-2rem)] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="break-words">{title}</DialogTitle>
          <DialogDescription>
            The agent stops using these too. You can restore them from the Removed tab.
          </DialogDescription>
        </DialogHeader>
        {preview.isPending && <p className="text-xs text-fg-subtle">Checking what goes with it…</p>}
        {preview.isError && (
          <Notice tone="danger">Could not preview: {(preview.error as Error).message}</Notice>
        )}
        <ul className="space-y-2 text-xs" data-testid="remove-effects">
          {effects.map((e) => (
            <li key={`${e.target.kind}:${e.target.id}`} className="min-w-0">
              <span className="break-words font-medium text-fg">{e.label}</span>{" "}
              <span className="text-fg-subtle">({KIND_LABEL[e.target.kind]})</span>
              {e.lines.map((line, i) => (
                <p key={i} className="break-words text-fg-subtle">
                  {line}
                </p>
              ))}
            </li>
          ))}
        </ul>
        <DialogFooter className="gap-2">
          <Button type="button" variant="outline" onClick={onClose} disabled={remove.isPending}>
            Cancel
          </Button>
          <Button
            type="button"
            variant="destructive"
            onClick={confirm}
            disabled={remove.isPending || !preview.isSuccess}
          >
            {remove.isPending ? "Removing…" : "Remove"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/** What went with a removal, in one line per thing. */
function wentWith(item: RemovedItem): string[] {
  if (item.cascade.length) return item.cascade.map((c) => c.text);
  if (item.kind === "name") return [`name “${item.label}” suppressed`];
  if (item.kind === "session") return ["its turns and summary, hidden from recall"];
  return [];
}

function DeleteDialog({
  items,
  onClose,
  onDone,
}: {
  items: RemovedItem[];
  onClose: () => void;
  onDone: (refused: { id: string; reason: string }[]) => void;
}) {
  const del = useDeleteRemoved();
  const [typed, setTyped] = useState("");
  const open = items.length > 0;
  useEffect(() => {
    if (open) setTyped("");
  }, [open]);

  const run = () =>
    del.mutate(
      items.map((i) => i.id),
      {
        onSuccess: (r) => {
          if (r.deleted.length) toast.success(`Deleted ${r.deleted.length} permanently.`);
          onDone(r.refused);
        },
        onError: (e) => toast.error(`Delete failed: ${(e as Error).message}`),
      },
    );

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (!o && !del.isPending) onClose();
      }}
    >
      <DialogContent className="max-h-[85vh] max-w-[calc(100vw-2rem)] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Delete {items.length} permanently?</DialogTitle>
          <DialogDescription>
            This cannot be undone. A deleted entity or name stays suppressed, so a summary
            cannot bring it back.
          </DialogDescription>
        </DialogHeader>
        <ul className="list-disc space-y-1 pl-4 text-xs text-fg">
          {items.map((i) => (
            <li key={i.id} className="break-words">
              {i.label} <span className="text-fg-subtle">({KIND_LABEL[i.kind]})</span>
            </li>
          ))}
        </ul>
        <label className="space-y-1 text-xs text-fg-subtle">
          <span>
            Type <b className="text-fg">delete</b> to confirm
          </span>
          <input
            aria-label="Type delete to confirm"
            className="w-full rounded-lg border border-border bg-surface px-2 py-2 text-sm text-fg"
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            autoComplete="off"
          />
        </label>
        <DialogFooter className="gap-2">
          <Button type="button" variant="outline" onClick={onClose} disabled={del.isPending}>
            Cancel
          </Button>
          <Button
            type="button"
            variant="destructive"
            onClick={run}
            disabled={del.isPending || typed.trim().toLowerCase() !== "delete"}
          >
            {del.isPending ? "Deleting…" : "Delete permanently"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/** The Removed tab: every removal, restorable until it is deleted for good. */
export function RemovedTab() {
  const canWrite = useWritesEnabled();
  const { data, isLoading, isError } = useRemovedItems();
  const restore = useRestoreRemoved();
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState<RemovedItem[]>([]);
  const [refused, setRefused] = useState<{ id: string; reason: string }[]>([]);
  const items = data?.items ?? [];
  const byId = new Map(items.map((i) => [i.id, i]));
  const chosen = [...picked].map((id) => byId.get(id)).filter((i): i is RemovedItem => !!i);

  const toggle = (id: string) =>
    setPicked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const restoreMany = async (rows: RemovedItem[]) => {
    for (const r of rows) {
      try {
        await restore.mutateAsync(r.id);
      } catch (e) {
        toast.error(`Restore of “${r.label}” failed: ${(e as Error).message}`);
        return;
      }
    }
    toast.success(`Restored ${rows.length}.`);
    setPicked(new Set());
  };

  return (
    <div className="space-y-3">
      <div>
        <h2 className="text-sm font-semibold text-fg">Removed</h2>
        <p className="text-[11px] text-fg-subtle">
          Everything removed from memory, and what went with it. Restore brings it all
          back; deleting permanently cannot be undone.
        </p>
      </div>

      {canWrite && (
        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            disabled={!chosen.length || restore.isPending}
            onClick={() => void restoreMany(chosen)}
          >
            Restore selected
          </Button>
          <Button
            size="sm"
            variant="destructive"
            disabled={!chosen.length}
            onClick={() => setDeleting(chosen)}
          >
            Delete permanently…
          </Button>
          <span className="text-[11px] text-fg-subtle">{items.length} removed</span>
        </div>
      )}
      <WriteGate canWrite={canWrite} />

      {refused.length > 0 && (
        <Notice tone="danger">
          {refused.map((r) => (
            <span key={r.id} className="block break-words">
              Not deleted — {byId.get(r.id)?.label ?? r.id}: {r.reason}
            </span>
          ))}
        </Notice>
      )}

      <QueryState
        loading={isLoading}
        error={isError}
        empty={items.length === 0}
        emptyText="Nothing removed. Remove a node from the Map and it is listed here."
      >
        <ul className="space-y-1.5" data-testid="removed-list">
          {items.map((item) => (
            <li
              key={item.id}
              className="flex items-start gap-2 rounded-lg border border-border bg-surface p-2 text-xs"
            >
              {canWrite && (
                <input
                  type="checkbox"
                  className="mt-0.5 h-4 w-4 shrink-0"
                  aria-label={`Select ${item.label}`}
                  disabled={item.permanent}
                  checked={picked.has(item.id)}
                  onChange={() => toggle(item.id)}
                />
              )}
              <div className="min-w-0 flex-1 space-y-0.5">
                <p className="flex flex-wrap items-center gap-1">
                  <Tag kind="info">{KIND_LABEL[item.kind]}</Tag>
                  <span className="break-words font-medium text-fg">{item.label}</span>
                  {item.permanent && <Tag kind="bad">deleted · stays suppressed</Tag>}
                </p>
                {wentWith(item).map((line, i) => (
                  <p key={i} className="break-words text-fg-subtle">
                    {line}
                  </p>
                ))}
                <p className="text-[10px] text-fg-subtle">{fmtDateTime(item.removed_at)}</p>
              </div>
              {canWrite && !item.permanent && (
                <Button
                  size="sm"
                  variant="outline"
                  disabled={restore.isPending}
                  onClick={() => void restoreMany([item])}
                >
                  Restore
                </Button>
              )}
            </li>
          ))}
        </ul>
      </QueryState>

      <DeleteDialog
        items={deleting}
        onClose={() => setDeleting([])}
        onDone={(r) => {
          setRefused(r);
          setDeleting([]);
          setPicked(new Set());
        }}
      />
    </div>
  );
}
