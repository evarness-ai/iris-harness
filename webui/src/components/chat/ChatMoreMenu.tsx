/* Phone-only "..." menu for Chat: New chat, the version/context status strip,
 * and Add task -- all desktop shows inline, with no room to spare on a phone.
 * A compact anchored dropdown (the shadcn/radix DropdownMenu already in
 * components/ui, never used elsewhere yet), not a bottom sheet -- that read as
 * a modal takeover rather than the "..." overflow menu it is.
 *
 * Outstanding items (dues/tasks/reminders) are deliberately NOT duplicated
 * here: AttentionBell's own sheet already surfaces them globally on a phone,
 * from the same useOutstandingItems() data. Add task stays, since that bell
 * is read-only and this is the one place on a phone that creates one. */
import { useState, type FormEvent } from "react";
import { MoreVertical, RotateCcw } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import type { ChatStatus } from "@/lib/client";

interface ChatMoreMenuProps {
  session: string;
  chatStatus: ChatStatus | undefined;
  onNewChat: () => void;
  newChatDisabled: boolean;
  addingTask: boolean;
  onToggleAddingTask: () => void;
  newTaskTitle: string;
  onNewTaskTitleChange: (value: string) => void;
  onSubmitTask: (title: string) => void;
  creatingTask: boolean;
}

export function ChatMoreMenu({
  session,
  chatStatus,
  onNewChat,
  newChatDisabled,
  addingTask,
  onToggleAddingTask,
  newTaskTitle,
  onNewTaskTitleChange,
  onSubmitTask,
  creatingTask,
}: ChatMoreMenuProps) {
  // Mirrors Radix's own uncontrolled open state so Escape/outside-click (its
  // defaults) keep working; only needed so "Add task" can reopen cleanly if a
  // stray outside click ever closed the menu mid-form.
  const [open, setOpen] = useState(false);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const title = newTaskTitle.trim();
    if (!title) return;
    onSubmitTask(title);
  };

  return (
    <div className="md:hidden">
      <DropdownMenu open={open} onOpenChange={setOpen}>
        <DropdownMenuTrigger
          aria-label="Chat options"
          className="inline-flex h-11 w-11 items-center justify-center rounded-md border border-border-strong text-fg-muted outline-none hover:bg-surface-raised focus-visible:ring-1 focus-visible:ring-ring"
        >
          <MoreVertical size={16} aria-hidden />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-72">
          <DropdownMenuItem
            onSelect={() => {
              onNewChat();
            }}
            disabled={newChatDisabled}
          >
            <RotateCcw size={15} /> New chat
          </DropdownMenuItem>

          <DropdownMenuSeparator />

          <div className="px-2 py-1.5 font-mono text-[10.5px] leading-relaxed text-fg-subtle">
            <div>
              IRIS v{chatStatus?.iris_version ?? "-"} · session {session}
            </div>
            <div>
              {chatStatus?.provider ?? "-"} · {chatStatus?.model ?? "-"}
            </div>
            <div>
              context {chatStatus?.window?.current_tokens ?? 0}/
              {chatStatus?.window?.budget_tokens ?? 0} (
              {Math.round(chatStatus?.window?.fill_pct ?? 0)}%)
            </div>
          </div>

          <DropdownMenuSeparator />

          {/* preventDefault: this item toggles the form in place, it doesn't
              pick an option and close like "New chat" above. */}
          <DropdownMenuItem
            onSelect={(e) => {
              e.preventDefault();
              onToggleAddingTask();
            }}
          >
            {addingTask ? "Cancel" : "Add task"}
          </DropdownMenuItem>
          {addingTask && (
            <form className="flex flex-col gap-2 px-2 pb-1.5 pt-1" onSubmit={submit}>
              <input
                autoFocus
                value={newTaskTitle}
                onChange={(e) => onNewTaskTitleChange(e.target.value)}
                placeholder="What needs doing?"
                maxLength={500}
                aria-label="New task"
                className="min-h-[44px] w-full rounded-md border border-border-strong bg-bg px-3 text-sm text-fg placeholder:text-fg-subtle focus:outline-none focus:ring-1 focus:ring-ring"
              />
              <Button type="submit" size="sm" disabled={!newTaskTitle.trim() || creatingTask}>
                {creatingTask ? "Adding…" : "Add"}
              </Button>
            </form>
          )}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}
