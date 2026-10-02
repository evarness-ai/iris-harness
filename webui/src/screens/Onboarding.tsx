/* Onboarding screen: `iris setup`'s own progress (GET /health/setup), read-only.
 * Lives OUTSIDE the app shell on purpose (own header, no sidebar) -- it's the
 * first thing a new install shows, not a screen inside the console you've
 * already set up. The wizard itself is a terminal flow (preflight needs a TTY,
 * Telegram pairing needs a bot token typed in, email delegates to a subprocess):
 * this page is a status mirror of $IRIS_HOME/setup.json, not a second place to
 * run it. For the one step that already has its own web flow (email), it links
 * to that screen, which is inside the shell. */
import { Link } from "react-router-dom";
import { Section } from "@/components/layout";
import { Tag } from "@/components/Tag";
import { Button } from "@/components/ui/button";
import { Notice, QueryState } from "@/components/control/parts";
import { ThemeToggle } from "@/components/ThemeToggle";
import { useSetupProgress } from "@/lib/queries";
import type { SetupProgress, SetupStepName, SetupStepRecord } from "@/lib/control";

const STEP_LABEL: Record<SetupStepName, string> = {
  preflight: "Preflight",
  home_secret: "Home & secret",
  services: "Services",
  telegram: "Telegram",
  email: "Email",
};

const STEP_DESCRIPTION: Record<SetupStepName, string> = {
  preflight: "Can IRIS run here — the same checks as `iris doctor`.",
  home_secret: "$IRIS_HOME seeded, and IRIS_AUTH_SECRET generated if it wasn't set.",
  services: "The background services (web UI, API, Telegram) — optional.",
  telegram: "Pair a Telegram bot so you can chat with IRIS from your phone — optional.",
  email: "Connect a mailbox — optional.",
};

function StepTag({ record }: { record: SetupStepRecord | undefined }) {
  if (!record) return <Tag kind="warn">Not started</Tag>;
  if (record.status === "done") return <Tag kind="ok">Done</Tag>;
  if (record.status === "failed") return <Tag kind="bad">Failed</Tag>;
  return <Tag kind="info">Skipped</Tag>;
}

function StepRow({ name, record }: { name: SetupStepName; record: SetupStepRecord | undefined }) {
  return (
    <div
      className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-lg border border-border bg-surface px-3 py-2"
      data-testid="setup-step"
    >
      <StepTag record={record} />
      <span className="text-xs font-medium text-fg">{STEP_LABEL[name]}</span>
      <span className="text-xs text-fg-muted">{STEP_DESCRIPTION[name]}</span>
      {record?.detail && (
        <span className="w-full font-mono text-[11px] text-fg-subtle sm:ml-auto sm:w-auto">
          {record.detail}
        </span>
      )}
      {name === "email" && (
        <Button
          asChild
          variant="link"
          size="sm"
          className="w-full justify-start px-2 text-[11px] font-semibold sm:ml-auto sm:w-auto"
        >
          <Link to="/setup">Open email setup →</Link>
        </Button>
      )}
    </div>
  );
}

function NextStep({ data }: { data: SetupProgress }) {
  if (!data.next_step) {
    return <Notice tone="muted">Every step has run or been skipped.</Notice>;
  }
  return (
    <Notice tone="muted">
      Next up: <span className="font-medium text-fg">{STEP_LABEL[data.next_step]}</span>. Continue
      from a terminal: <code className="font-mono text-[11px]">iris setup</code>.
    </Notice>
  );
}

export function OnboardingScreen() {
  const { data, isLoading, isError } = useSetupProgress();

  return (
    <div className="flex min-h-screen flex-col bg-bg text-fg">
      <header className="flex items-center justify-between px-4 py-3">
        <div>
          <div className="text-base font-bold tracking-tight text-primary">IRIS</div>
          <div className="text-[10px] uppercase tracking-widest text-fg-subtle">Onboarding</div>
        </div>
        <div className="flex items-center gap-1">
          <Button asChild variant="link" size="sm" className="px-2 text-xs text-fg-muted hover:text-fg">
            <Link to="/chat">Back to IRIS →</Link>
          </Button>
          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto w-full max-w-2xl flex-1 space-y-6 px-4 pb-10 pt-2">
        <div>
          <h1 className="text-lg font-semibold text-fg">Progress through `iris setup`</h1>
          <p className="mt-1 text-sm text-fg-muted">
            Preflight, secret, services, Telegram, email — read-only.
          </p>
        </div>

        <Section title="Progress">
          <QueryState loading={isLoading} error={isError} empty={!data} emptyText="No setup report.">
            {data && <NextStep data={data} />}
          </QueryState>
        </Section>

        {data && data.order && (
          <Section title="Steps">
            <div className="space-y-1.5">
              {data.order.map((name) => (
                <StepRow key={name} name={name} record={data.steps?.[name]} />
              ))}
            </div>
          </Section>
        )}

        <Notice tone="muted">
          This screen mirrors <code className="font-mono text-[11px]">$IRIS_HOME/setup.json</code>{" "}
          read-only. The wizard itself runs in a terminal (preflight needs a TTY, Telegram pairing
          asks for a bot token, email hands off to{" "}
          <code className="font-mono text-[11px]">iris email setup</code>):{" "}
          <code className="font-mono text-[11px]">iris setup --status</code> shows the same thing
          from the CLI, and <code className="font-mono text-[11px]">iris setup --reset</code>{" "}
          starts over.
        </Notice>
      </main>
    </div>
  );
}
