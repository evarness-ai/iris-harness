/* Playground — the harness test bench. List YAML scenario suites, run one
 * against the live runtime, and see per-scenario pass/fail + which intercept
 * answered. The Drift panel surfaces config-vs-runtime gaps (declared in YAML
 * but not registered, or vice-versa) — observability for the framework itself.
 *
 * Running a suite drives chat, so it is write-gated: the Run buttons are
 * disabled unless IRIS_WEBUI_ALLOW_WRITES=1 (same posture as other actions). */
import { useState } from "react";
import { Tag } from "@/components/Tag";
import { Button } from "@/components/ui/button";
import { Section } from "@/components/layout";
import { Notice, QueryState } from "@/components/control/parts";
import {
  usePlaygroundDrift,
  usePlaygroundSuites,
  useRunPlaygroundSuite,
  useWritesEnabled,
} from "@/lib/queries";
import type {
  DriftSurface,
  PlaygroundRunResult,
  PlaygroundScenarioResult,
  PlaygroundSuiteMeta,
} from "@/lib/control";

function DriftPanel() {
  const { data, isLoading, isError } = usePlaygroundDrift();
  const surfaces = data?.surfaces ?? [];
  return (
    <Section
      title="Config ↔ runtime drift"
      actions={
        data ? (
          <Tag kind={data.ok ? "ok" : "warn"}>{data.ok ? "in sync" : "drift"}</Tag>
        ) : undefined
      }
    >
      <QueryState
        loading={isLoading}
        error={isError}
        empty={surfaces.length === 0}
        emptyText="No surfaces reported."
      >
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {surfaces.map((s) => (
            <DriftCard key={s.surface} surface={s} />
          ))}
        </div>
      </QueryState>
    </Section>
  );
}

function DriftCard({ surface }: { surface: DriftSurface }) {
  return (
    <div className="rounded-lg border border-border p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="font-mono text-xs font-semibold">{surface.surface}</span>
        <Tag kind={surface.ok ? "ok" : "warn"}>
          {surface.ok ? "ok" : `${surface.declared_only.length + surface.registered_only.length}`}
        </Tag>
      </div>
      {surface.ok ? (
        <p className="text-[11px] text-fg-subtle">{surface.in_sync.length} in sync</p>
      ) : (
        <div className="space-y-1 text-[11px]">
          {surface.declared_only.length > 0 && (
            <p className="text-warning">
              declared only: {surface.declared_only.join(", ")}
            </p>
          )}
          {surface.registered_only.length > 0 && (
            <p className="text-info">registered only: {surface.registered_only.join(", ")}</p>
          )}
        </div>
      )}
    </div>
  );
}

function ScenarioRow({ r }: { r: PlaygroundScenarioResult }) {
  return (
    <div className="rounded-md border border-border px-3 py-2">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-xs">{r.name}</span>
        <div className="flex items-center gap-2">
          <span className="text-[10.5px] text-fg-subtle">
            intent={r.intent ?? "—"} · handler={r.handler ?? "agent-loop"} ·{" "}
            {r.duration_ms.toFixed(0)}ms
          </span>
          <Tag kind={r.passed ? "ok" : "bad"}>{r.passed ? "pass" : "fail"}</Tag>
        </div>
      </div>
      {r.error && <p className="mt-1 text-[11px] text-danger">{r.error}</p>}
      {r.failed_assertions.length > 0 && (
        <ul className="mt-1 space-y-0.5">
          {r.failed_assertions.map((a, i) => (
            <li key={i} className="font-mono text-[10.5px] text-danger">
              ✗ {a.field}: want {JSON.stringify(a.expected)} got {JSON.stringify(a.actual)}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function SuiteCard({
  suite,
  result,
  running,
  canRun,
  onRun,
}: {
  suite: PlaygroundSuiteMeta;
  result?: PlaygroundRunResult;
  running: boolean;
  canRun: boolean;
  onRun: () => void;
}) {
  return (
    <div className="mb-4 rounded-lg border border-border p-4">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold">{suite.name}</span>
            {result && (
              <Tag kind={result.ok ? "ok" : "bad"}>
                {result.passed}/{result.total}
              </Tag>
            )}
          </div>
          {suite.description && (
            <p className="text-[11px] text-fg-subtle">{suite.description}</p>
          )}
        </div>
        <Button size="sm" disabled={!canRun || running} onClick={onRun}>
          {running ? "Running…" : "Run"}
        </Button>
      </div>
      {suite.error ? (
        <Notice tone="danger">{suite.error}</Notice>
      ) : result ? (
        <div className="space-y-2">
          {result.results.map((r) => (
            <ScenarioRow key={r.name} r={r} />
          ))}
        </div>
      ) : (
        <ul className="space-y-1">
          {(suite.scenarios ?? []).map((s) => (
            <li key={s.name} className="flex items-center gap-2 text-[11px] text-fg-subtle">
              <span className="font-mono">{s.name}</span>
              {s.tags.map((t) => (
                <Tag key={t} kind="info">
                  {t}
                </Tag>
              ))}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function PlaygroundScreen() {
  const { data, isLoading, isError } = usePlaygroundSuites();
  const run = useRunPlaygroundSuite();
  const writesEnabled = useWritesEnabled();
  const [results, setResults] = useState<Record<string, PlaygroundRunResult>>({});
  const [runningSuite, setRunningSuite] = useState<string | null>(null);

  const suites = data?.suites ?? [];

  const onRun = async (name: string) => {
    setRunningSuite(name);
    try {
      const res = await run.mutateAsync(name);
      setResults((prev) => ({ ...prev, [name]: res }));
    } catch (e) {
      setResults((prev) => ({
        ...prev,
        [name]: {
          suite: name,
          ok: false,
          passed: 0,
          total: 0,
          results: [],
        },
      }));
      // Surface the error inline on the suite card via a synthetic failure.
      console.error("playground run failed", e);
    } finally {
      setRunningSuite(null);
    }
  };

  return (
    <div>
      {!writesEnabled && (
        <div className="mb-4">
          <Notice>
            Running a suite drives chat, so it is write-gated. Set IRIS_WEBUI_ALLOW_WRITES=1 to
            enable the Run buttons. Listing suites and drift stay read-only.
          </Notice>
        </div>
      )}
      <DriftPanel />
      <Section title="Scenario suites">
        <QueryState
          loading={isLoading}
          error={isError}
          empty={suites.length === 0}
          emptyText="No suites under config/playground."
        >
          {suites.map((suite) => (
            <SuiteCard
              key={suite.name}
              suite={suite}
              result={results[suite.name]}
              running={runningSuite === suite.name}
              canRun={writesEnabled}
              onRun={() => onRun(suite.name)}
            />
          ))}
        </QueryState>
      </Section>
    </div>
  );
}
