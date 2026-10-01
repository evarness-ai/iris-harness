/* Email setup (OSS plan R4 + R17): the steps `iris email setup` walks, on the web.
 *
 * A thin renderer over /api/v1/email/onboarding (lib/onboarding.ts). The server decides
 * every step, what it waits for and what it did; this screen shows that and sends the
 * owner's answers: Continue (run until setup waits), the master-key decision, which
 * proposed categories to accept, and step 6 -- the label preview with explicit Approve /
 * Decline. Nothing is approved without that click, and the same approval sits in the
 * Action Center, where answering it counts too.
 *
 * Connecting a mailbox is its provider plugin's login at a terminal; the screen shows
 * that command with a copy button and never asks for a password itself.
 *
 * The long steps (fetch, discover, the judge) run inside the POST, for minutes at a
 * time, so each account has one request in flight: its buttons stay disabled until the
 * server answers, and a dropped connection re-reads the state the server kept. */
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Tag } from "@/components/Tag";
import { ApiUnavailable } from "@/components/control/parts";
import { isUnreachable } from "@/lib/http";
import { ConfirmDialog, type ConfirmState } from "@/components/ConfirmDialog";
import { useWritesEnabled } from "@/lib/queries";
import {
  advance,
  useLabelPreview,
  useSetupBusy,
  useSetupOverview,
  useSetupWrite,
  type AdvanceBody,
  type Proposal,
  type SetupOverview,
  type SetupState,
  type SetupWrite,
} from "@/lib/onboarding";

const BUTTON =
  "inline-flex min-h-11 items-center justify-center rounded-lg border border-border bg-bg px-4 text-sm text-fg hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50";
const PRIMARY =
  "inline-flex min-h-11 items-center justify-center rounded-lg border border-primary bg-primary px-4 text-sm font-semibold text-bg hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50";
const PANEL = "rounded-xl border border-border bg-surface p-3";

type TagKind = "res" | "opp" | "ok" | "bad" | "warn" | "info";

function statusTag(s: SetupState): { kind: TagKind; text: string } {
  if (s.status === "done") return { kind: "ok", text: "done" };
  if (s.waiting_kind === "decision") return { kind: "warn", text: "needs your decision" };
  if (s.waiting_kind === "blocked") return { kind: "bad", text: "waiting" };
  return { kind: "info", text: "in progress" };
}

function titleOf(s: SetupState, step: string): string {
  return s.steps.find((x) => x.step === step)?.title ?? step;
}

/** A command to run at a terminal, with Copy. The clipboard can refuse (an older app
 * view, no permission); then the text is selected so the owner's own copy works. */
export function CopyCommand({ command, label }: { command: string; label?: string }) {
  const ref = useRef<HTMLElement>(null);
  const [copied, setCopied] = useState<"" | "copied" | "selected">("");
  const select = () => {
    const el = ref.current;
    const sel = window.getSelection();
    if (!el || !sel) return;
    const range = document.createRange();
    range.selectNodeContents(el);
    sel.removeAllRanges();
    sel.addRange(range);
    setCopied("selected");
  };
  const copy = () => {
    const write = navigator.clipboard?.writeText(command);
    if (!write) return select();
    write.then(
      () => {
        setCopied("copied");
        window.setTimeout(() => setCopied(""), 1500);
      },
      select,
    );
  };
  return (
    <div className="mt-2 flex flex-col gap-2 sm:flex-row sm:items-center">
      <code
        ref={ref}
        data-testid="connect-command"
        aria-label={label ?? "Command"}
        className="min-w-0 flex-1 select-all break-all rounded-lg bg-bg px-3 py-2 font-mono text-xs text-fg"
      >
        {command}
      </code>
      <button type="button" className={BUTTON} onClick={copy}>
        {copied === "copied" ? "Copied" : "Copy"}
      </button>
      {copied === "selected" && (
        <span role="status" className="text-xs text-fg-muted">
          Selected: copy it with your keyboard or menu.
        </span>
      )}
    </div>
  );
}

function SweepLine({ s }: { s: SetupState }) {
  return (
    <p className="text-xs text-fg-muted" data-testid="sweep">
      Scheduled sweep: {s.sweep.swept ? "on" : "waiting for setup"}
      {s.sweep.reason ? ` (${s.sweep.reason})` : ""}
    </p>
  );
}

function StepList({ s }: { s: SetupState }) {
  return (
    <ol className="flex flex-col gap-2" aria-label="Setup steps">
      {s.steps.map((step, i) => {
        const text = s.rendered[step.step];
        const state = step.done ? "done" : step.current ? "current" : "ahead";
        return (
          <li
            key={step.step}
            data-testid={`step-${step.step}`}
            data-state={state}
            aria-current={step.current ? "step" : undefined}
            className={`rounded-lg border px-3 py-2 ${
              step.current ? "border-primary bg-primary/5" : "border-border"
            }`}
          >
            <div className="flex items-center justify-between gap-2">
              <span className={`text-sm ${step.current ? "font-semibold text-fg" : "text-fg-muted"}`}>
                {i + 1}. {step.title}
              </span>
              {step.done ? (
                <Tag kind="ok">done</Tag>
              ) : step.current ? (
                <Tag kind="info">now</Tag>
              ) : null}
            </div>
            {text && step.step !== "summary" && (
              <pre className="mt-1 whitespace-pre-wrap break-words font-sans text-xs text-fg-muted">
                {text}
              </pre>
            )}
          </li>
        );
      })}
    </ol>
  );
}

function CategoryReview({
  s,
  busy,
  onSubmit,
}: {
  s: SetupState;
  busy: boolean;
  onSubmit: (body: AdvanceBody) => void;
}) {
  const proposals = ((s.results.discover?.proposals as Proposal[] | undefined) ?? []).filter(
    Boolean,
  );
  const acceptable = proposals.filter((p) => p.acceptable).map((p) => p.cluster_id);
  const [picked, setPicked] = useState<Set<number>>(() => new Set(acceptable));
  const toggle = (id: number) =>
    setPicked((old) => {
      const next = new Set(old);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  return (
    <div data-testid="category-review">
      <ul className="mt-2 flex flex-col gap-2">
        {proposals.map((p) => (
          <li key={p.cluster_id} className="rounded-lg border border-border p-2">
            <label className="flex min-h-11 items-start gap-3">
              <input
                type="checkbox"
                className="mt-1 h-5 w-5 shrink-0"
                disabled={!p.acceptable || busy}
                checked={picked.has(p.cluster_id)}
                onChange={() => toggle(p.cluster_id)}
                aria-label={p.path ?? `Category ${p.cluster_id}`}
              />
              <span className="min-w-0 flex-1">
                <span className="block break-words text-sm font-semibold text-fg">
                  {p.path ?? `Category ${p.cluster_id}`}
                </span>
                <span className="block text-xs text-fg-subtle">
                  {p.size} email(s){p.top_domain ? ` · mostly ${p.top_domain}` : ""}
                </span>
                {!p.acceptable && p.why_not && (
                  <span className="block break-words text-xs text-danger">{p.why_not}</span>
                )}
                {p.samples.length > 0 && (
                  <span className="block break-words text-xs text-fg-muted">
                    e.g. {p.samples.join("; ")}
                  </span>
                )}
              </span>
            </label>
          </li>
        ))}
      </ul>
      <button
        type="button"
        className={`${PRIMARY} mt-3`}
        disabled={busy}
        onClick={() => onSubmit({ accept_categories: [...picked].sort((a, b) => a - b) })}
      >
        {picked.size === 0 ? "Accept none" : `Accept ${picked.size} selected`}
      </button>
    </div>
  );
}

function LabelApproval({
  s,
  busy,
  canWrite,
  onAnswer,
}: {
  s: SetupState;
  busy: boolean;
  canWrite: boolean;
  onAnswer: (approve: boolean) => void;
}) {
  const preview = useLabelPreview(s.account_id, true);
  const p = preview.data;
  return (
    <div data-testid="label-approval">
      {preview.isLoading ? (
        <p className="mt-2 text-sm text-fg-muted">Loading the label preview…</p>
      ) : preview.error ? (
        <p className="mt-2 text-sm text-danger">
          Couldn't load the label preview
          {preview.error instanceof Error ? `: ${preview.error.message}` : ""}.
        </p>
      ) : p ? (
        <div className="mt-2">
          <p className="text-sm font-semibold text-fg" data-testid="preview-total">
            {p.total} label(s) would be written
          </p>
          {p.status && <p className="text-sm text-fg-muted">{p.status}</p>}
          <ul className="mt-2 flex flex-col gap-2">
            {(p.groups ?? []).map((g) => (
              <li key={g.bucket} className="rounded-lg border border-border p-2" data-testid="preview-group">
                <p className="text-sm text-fg">
                  <span className="font-semibold">{g.label}</span>: {g.count} email(s)
                </p>
                {g.samples.length > 0 && (
                  <ul className="mt-1 list-disc pl-5 text-xs text-fg-muted">
                    {g.samples.map((subject, i) => (
                      <li key={i} className="break-words">
                        {subject}
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
          {p.removals > 0 && (
            <p className="mt-2 text-xs text-fg-muted">
              IRIS labels removed from {p.removals} email(s) marked promo.
            </p>
          )}
          {(p.notes ?? []).map((line) => (
            <p key={line} className="mt-1 break-words text-xs text-fg-muted">
              {line}
            </p>
          ))}
        </div>
      ) : null}
      <p className="mt-3 text-xs text-fg-muted" data-testid="action-center-note">
        This approval{s.approval_id ? ` (${s.approval_id})` : ""} is also waiting in the Action
        Center; answering it there counts too.
      </p>
      <Link to="/actions" className={`${BUTTON} mt-2`}>
        Open the Action Center
      </Link>
      {canWrite ? (
        <div className="mt-3 flex flex-wrap gap-2">
          <button
            type="button"
            className={PRIMARY}
            disabled={busy || !p}
            onClick={() => onAnswer(true)}
          >
            Approve: let IRIS label mail
          </button>
          <button type="button" className={BUTTON} disabled={busy} onClick={() => onAnswer(false)}>
            Decline: keep it read-only
          </button>
        </div>
      ) : null}
    </div>
  );
}

function CurrentStep({
  s,
  busy,
  canWrite,
  send,
}: {
  s: SetupState;
  busy: boolean;
  canWrite: boolean;
  send: (w: SetupWrite) => void;
}) {
  if (s.status === "done") return null;
  const title = titleOf(s, s.step);
  const again = (label: string) =>
    canWrite ? (
      <button
        type="button"
        className={`${PRIMARY} mt-3`}
        disabled={busy}
        onClick={() => send({ kind: "advance" })}
      >
        {label}
      </button>
    ) : null;

  if (s.status === "in_progress") {
    return (
      <div className={PANEL} data-testid="current-step">
        <p className="text-sm text-fg">
          Next: <span className="font-semibold">{title}</span>
        </p>
        {again("Continue")}
      </div>
    );
  }
  const tag = statusTag(s);
  return (
    <div className={PANEL} data-testid="current-step" data-waiting={s.waiting_kind}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-semibold text-fg">{title}</span>
        <Tag kind={tag.kind}>{tag.text}</Tag>
      </div>
      <p className="mt-1 break-words text-sm text-fg-muted" data-testid="waiting-for">
        {s.waiting_for}
      </p>
      {s.step === "connect" && s.waiting_kind === "blocked" && s.connect_command && (
        <>
          <p className="mt-2 text-xs text-fg-muted">
            Run this at a terminal on the machine IRIS runs on, then check again:
          </p>
          <CopyCommand command={s.connect_command} label="Connect command" />
        </>
      )}
      {s.step === "connect" && s.waiting_kind === "decision" && canWrite && (
        <button
          type="button"
          className={`${PRIMARY} mt-3`}
          disabled={busy}
          onClick={() => send({ kind: "advance", body: { create_master_key: true } })}
        >
          Create the vault master key
        </button>
      )}
      {s.step === "review_categories" && s.waiting_kind === "decision" && canWrite && (
        <CategoryReview
          s={s}
          busy={busy}
          onSubmit={(body) => send({ kind: "advance", body })}
        />
      )}
      {s.step === "label_approval" && s.waiting_kind === "decision" && (
        <LabelApproval
          s={s}
          busy={busy}
          canWrite={canWrite}
          onAnswer={(approve) => send({ kind: "approve", approve })}
        />
      )}
      {s.waiting_kind === "blocked" && again("Check again")}
    </div>
  );
}

/** Seconds since `busy` turned on, for the pending line. */
function useElapsed(busy: boolean): number {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!busy) return;
    setSeconds(0);
    const started = Date.now();
    const id = window.setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(id);
  }, [busy]);
  return seconds;
}

function AccountSetup({ s, canWrite }: { s: SetupState; canWrite: boolean }) {
  const write = useSetupWrite(s.account_id);
  const busy = useSetupBusy(s.account_id);
  const elapsed = useElapsed(busy);
  // A second click lands before React re-renders the disabled button: the ref holds.
  const inFlight = useRef(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [confirm, setConfirm] = useState<ConfirmState | null>(null);

  const send = (w: SetupWrite) => {
    if (inFlight.current || busy) return;
    inFlight.current = true;
    setError("");
    write.mutate(w, {
      onSuccess: (data) => {
        if ("notice" in data && data.notice) setNotice(data.notice);
      },
      onError: (err) => setError(err instanceof Error ? err.message : String(err)),
      onSettled: () => {
        inFlight.current = false;
      },
    });
  };

  const askRestart = () =>
    setConfirm({
      title: `Start ${s.account_id}'s setup over?`,
      description:
        "This forgets where setup stands for this account. Fetched mail, judgments, " +
        "categories, any mailbox-write approval and the sweep stay as they are.",
      confirmLabel: "Start over",
      destructive: true,
      run: async () => {
        if (inFlight.current) throw new Error("busy");
        inFlight.current = true;
        setError("");
        try {
          await write.mutateAsync({ kind: "restart" });
        } catch (err) {
          setError(err instanceof Error ? err.message : String(err));
          throw err;
        } finally {
          inFlight.current = false;
        }
      },
    });

  const summary = s.rendered.summary;
  const tag = statusTag(s);
  return (
    <section aria-label={`Setup for ${s.account_id}`} className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="min-w-0 break-all text-base font-semibold text-fg">{s.account_id}</h2>
        <Tag kind={tag.kind}>{tag.text}</Tag>
      </div>
      <SweepLine s={s} />

      {busy && (
        <div role="status" className={`${PANEL} border-info/40`} data-testid="setup-busy">
          <p className="text-sm text-fg">
            Working on <span className="font-semibold">{titleOf(s, s.step)}</span>… {elapsed}s
          </p>
          <p className="mt-1 text-xs text-fg-muted">
            Fetching, finding categories and judging run inside this request and can take
            minutes. Keep this page open; the buttons come back when IRIS answers.
          </p>
        </div>
      )}
      {error && (
        <div role="alert" className={`${PANEL} border-danger/40`}>
          <p className="break-words text-sm text-danger">{error}</p>
          <p className="mt-1 text-xs text-fg-muted">
            If the connection dropped, IRIS may have kept going: what is shown here is
            re-read from the server.
          </p>
        </div>
      )}
      {notice && (
        <div className={`${PANEL} border-warning/40`} data-testid="setup-notice">
          <p className="text-sm text-fg">
            Shown once and not stored: add this to your shell profile and IRIS's environment.
          </p>
          <CopyCommand command={notice} label="Notice" />
        </div>
      )}

      {summary && (
        <div className={PANEL} data-testid="setup-summary">
          <h3 className="text-sm font-semibold text-fg">{titleOf(s, "summary")}</h3>
          <pre className="mt-1 whitespace-pre-wrap break-words font-sans text-sm text-fg-muted">
            {summary}
          </pre>
        </div>
      )}

      {!canWrite && s.status !== "done" && (
        <p className="text-xs text-fg-muted" data-testid="setup-readonly">
          Setup's steps are writes. Pair this browser with control access, or set
          IRIS_WEBUI_ALLOW_WRITES=1, to continue here; or run `iris email setup`.
        </p>
      )}

      <CurrentStep s={s} busy={busy} canWrite={canWrite} send={send} />
      <StepList s={s} />

      {canWrite && (
        <div>
          <button type="button" className={BUTTON} disabled={busy} onClick={askRestart}>
            Start over…
          </button>
        </div>
      )}
      <ConfirmDialog state={confirm} onClose={() => setConfirm(null)} />
    </section>
  );
}

/** Starting setup for an account with none: one advance, run until it waits. */
function useStartSetup(onStarted: (accountId: string) => void) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (accountId: string) => advance(accountId),
    onSuccess: (state) => onStarted(state.account_id),
    onSettled: () => qc.invalidateQueries({ queryKey: ["email-onboarding"] }),
  });
}

function StartPanel({
  data,
  canWrite,
  onStarted,
}: {
  data: SetupOverview;
  canWrite: boolean;
  onStarted: (accountId: string) => void;
}) {
  const start = useStartSetup(onStarted);
  const [typed, setTyped] = useState("");
  const known = new Set(data.setups.map((s) => s.account_id));
  const fresh = data.accounts.filter((a) => !known.has(a));
  const nothing = data.setups.length === 0 && data.accounts.length === 0;
  const go = (id: string) => {
    if (!start.isPending && id.trim()) start.mutate(id.trim());
  };
  return (
    <div className={`${PANEL} flex flex-col gap-3`} data-testid="start-panel">
      {nothing && (
        <div>
          <p className="text-sm text-fg">
            No mailbox is connected yet. Connect one at a terminal on the machine IRIS runs
            on, then start its setup here:
          </p>
          {data.connect_hints.map((h) => (
            <div key={h.provider} className="mt-2">
              <p className="text-xs font-semibold text-fg-muted">{h.provider}</p>
              <CopyCommand command={h.command} label={`${h.provider} connect command`} />
            </div>
          ))}
        </div>
      )}
      {canWrite && fresh.length > 0 && (
        <ul className="flex flex-col gap-2" aria-label="Connected accounts">
          {fresh.map((a) => (
            <li key={a} className="flex flex-wrap items-center justify-between gap-2">
              <span className="min-w-0 break-all text-sm text-fg">{a}</span>
              <button
                type="button"
                className={PRIMARY}
                disabled={start.isPending}
                onClick={() => go(a)}
              >
                Start setup
              </button>
            </li>
          ))}
        </ul>
      )}
      {canWrite && data.demo_account && !known.has(data.demo_account) && (
        <button
          type="button"
          className={BUTTON}
          disabled={start.isPending}
          onClick={() => go(data.demo_account!)}
        >
          Set up the demo mailbox
        </button>
      )}
      {canWrite && (
        <form
          className="flex flex-col gap-2 sm:flex-row"
          onSubmit={(e) => {
            e.preventDefault();
            go(typed);
          }}
        >
          <label className="sr-only" htmlFor="setup-account">
            Account to set up
          </label>
          <input
            id="setup-account"
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            placeholder="gmail:you@example.com"
            autoCapitalize="none"
            autoCorrect="off"
            spellCheck={false}
            className="min-h-11 min-w-0 flex-1 rounded-lg border border-border bg-bg px-3 text-sm text-fg"
          />
          <button type="submit" className={BUTTON} disabled={start.isPending || !typed.trim()}>
            Set up this account
          </button>
        </form>
      )}
      {start.isPending && (
        <p role="status" className="text-xs text-fg-muted">
          Starting setup… the first steps can take a while.
        </p>
      )}
      {start.error && (
        <p role="alert" className="break-words text-sm text-danger">
          {start.error instanceof Error ? start.error.message : String(start.error)}
        </p>
      )}
    </div>
  );
}

export function SetupScreen() {
  const { data, isLoading, error } = useSetupOverview();
  const canWrite = useWritesEnabled();
  const [picked, setPicked] = useState<string | null>(null);

  const setups = data?.setups ?? [];
  const fallback = setups.find((s) => s.status !== "done") ?? setups[0];
  const current = setups.find((s) => s.account_id === picked) ?? fallback;

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-4">
      <p className="text-sm text-fg-muted">
        The same steps as <code className="font-mono text-xs">iris email setup</code>: connect,
        fetch, find categories, classify, preview the labels, then decide. Nothing is written
        to your mailbox before you approve the label preview.
      </p>
      {isLoading ? (
        <p className="text-sm text-fg-muted">Loading…</p>
      ) : isUnreachable(error) ? (
        <ApiUnavailable />
      ) : error ? (
        <p className="text-sm text-danger">
          Couldn't load email setup{error instanceof Error ? `: ${error.message}` : ""}.
        </p>
      ) : data ? (
        <>
          {setups.length > 1 && (
            <div className="flex flex-wrap gap-2" aria-label="Accounts">
              {setups.map((s) => {
                const on = s.account_id === current?.account_id;
                return (
                  <button
                    key={s.account_id}
                    type="button"
                    aria-pressed={on}
                    onClick={() => setPicked(s.account_id)}
                    className={`min-h-11 max-w-full break-all rounded-full border px-4 text-left text-sm ${
                      on ? "border-primary bg-primary text-bg" : "border-border bg-bg text-fg hover:bg-accent"
                    }`}
                  >
                    {s.account_id}
                  </button>
                );
              })}
            </div>
          )}
          {current && <AccountSetup key={current.account_id} s={current} canWrite={canWrite} />}
          <StartPanel data={data} canWrite={canWrite} onStarted={setPicked} />
        </>
      ) : null}
    </div>
  );
}
