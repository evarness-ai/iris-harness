import { Link } from "react-router-dom";
import { toast } from "sonner";
import { Card, Kpi } from "@/components/Card";
import { Tag } from "@/components/Tag";
import { Button } from "@/components/ui/button";
import { Grid } from "@/components/layout";
import { Notice } from "@/components/control/parts";
import { fmtMs, fmtTokens } from "@/lib/nodeMeta";
import { isMetricsDisabled } from "@/lib/control";
import {
  useHeartbeats,
  useLearningMetrics,
  useLlmMode,
  usePinLlmMode,
  useRoutines,
  useUnpinLlmMode,
  useWritesEnabled,
} from "@/lib/queries";

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 py-1 text-xs">
      <span className="text-fg-subtle">{k}</span>
      <span className="text-right font-mono text-fg">{v}</span>
    </div>
  );
}

const PINNABLE = ["active", "idle"];

function LlmModeCard() {
  const { data, isLoading, isError } = useLlmMode();
  const pin = usePinLlmMode();
  const unpin = useUnpinLlmMode();
  const canWrite = useWritesEnabled();
  const busy = pin.isPending || unpin.isPending;

  const doPin = async (mode: string) => {
    try {
      await pin.mutateAsync(mode);
      toast.success(`LLM mode pinned to ${mode}`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "pin failed");
    }
  };
  const doUnpin = async () => {
    try {
      await unpin.mutateAsync();
      toast.success("LLM mode unpinned (adaptive)");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "unpin failed");
    }
  };

  return (
    <Card title="LLM mode">
      {isLoading ? (
        <Notice>Loading…</Notice>
      ) : isError || !data ? (
        <Notice tone="danger">API unavailable</Notice>
      ) : (
        <>
          <div className="mb-2 font-mono text-xl font-semibold text-primary">{data.mode}</div>
          <Row k="auto mode" v={data.auto_mode} />
          <Row k="pin" v={data.pin ?? "none"} />
          <Row k="adaptive" v={data.adaptive ? "on" : "off"} />
          {canWrite && (
          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            {PINNABLE.map((m) => (
              <Button
                key={m}
                type="button"
                size="sm"
                variant={data.pin === m ? "default" : "outline"}
                disabled={busy}
                onClick={() => void doPin(m)}
              >
                pin {m}
              </Button>
            ))}
            <Button
              type="button"
              size="sm"
              variant="ghost"
              disabled={busy || !data.pin}
              onClick={() => void doUnpin()}
            >
              unpin
            </Button>
          </div>
          )}
          {data.snapshot && (
            <div className="mt-2 border-t border-border pt-2">
              <Row k="cpu" v={`${Math.round(data.snapshot.cpu_percent)}%`} />
              <Row k="ram free" v={`${data.snapshot.ram_free_gb.toFixed(1)} GB`} />
              <div className="mt-1.5">
                {data.snapshot.thermal_throttled ? (
                  <Tag kind="warn">thermal-throttled</Tag>
                ) : (
                  <Tag kind="ok">nominal</Tag>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </Card>
  );
}

function LearningCard() {
  const { data, isLoading, isError } = useLearningMetrics();
  if (isLoading) return <Card title="Self-learning"><Notice>Loading…</Notice></Card>;
  if (isError || !data) return <Card title="Self-learning"><Notice tone="danger">API unavailable</Notice></Card>;
  if (isMetricsDisabled(data)) {
    return (
      <Card title="Self-learning">
        <Notice>
          Metrics disabled. Set <span className="font-mono">IRIS_OBSERVABILITY_METRICS_ENABLED=1</span>{" "}
          to surface LLM + learning health.
        </Notice>
      </Card>
    );
  }
  const s = data.summary;
  return (
    <Card title="Self-learning">
      <div className="grid grid-cols-2 gap-2">
        <Kpi value={`${s.llm_call_count}`} label="LLM calls" />
        <Kpi value={fmtTokens(s.total_tokens)} label="tokens" />
        <Kpi value={`${s.llm_error_count}`} label="errors" tone={s.llm_error_count > 0 ? "bad" : undefined} />
        <Kpi value={`${s.sessions_scanned}`} label="sessions" />
      </div>
      <Row k="backend" v={data.backend.healthy ? "healthy" : data.backend.enabled ? "degraded" : "off"} />
      <Row k="avg latency" v={s.llm_call_count ? fmtMs(s.total_duration_ms / s.llm_call_count) : "—"} />
    </Card>
  );
}

function RoutinesCard() {
  const { data, isLoading, isError } = useRoutines();
  const routines = data ?? [];
  const active = routines.filter((r) => r.approval_status === "scheduled" || r.approval_status === "approved").length;
  const pending = routines.filter((r) =>
    ["draft", "clarify", "template"].includes(r.approval_status),
  ).length;
  return (
    <Card title="Routines">
      {isLoading ? (
        <Notice>Loading…</Notice>
      ) : isError ? (
        <Notice tone="danger">API unavailable</Notice>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-2">
            <Kpi value={`${routines.length}`} label="total" />
            <Kpi value={`${active}`} label="active" tone="ok" />
            <Kpi value={`${pending}`} label="proposals" />
          </div>
          <Link
            to="/routines"
            className="mt-3 inline-flex min-h-[44px] items-center text-xs font-medium text-primary hover:underline sm:min-h-0"
          >
            manage routines →
          </Link>
        </>
      )}
    </Card>
  );
}

function HeartbeatsCard() {
  const { data, isLoading, isError } = useHeartbeats();
  const beats = data?.heartbeats ?? [];
  const enabled = beats.filter((b) => b.enabled).length;
  return (
    <Card title="Heartbeats">
      {isLoading ? (
        <Notice>Loading…</Notice>
      ) : isError ? (
        <Notice tone="danger">API unavailable</Notice>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-2">
            <Kpi value={`${enabled}`} label="enabled" tone="ok" />
            <Kpi value={`${beats.length}`} label="defined" />
          </div>
          <Link
            to="/heartbeats"
            className="mt-3 inline-flex min-h-[44px] items-center text-xs font-medium text-primary hover:underline sm:min-h-0"
          >
            view heartbeats →
          </Link>
        </>
      )}
    </Card>
  );
}

export function OverviewScreen() {
  return (
    <Grid cols={4}>
      <LlmModeCard />
      <LearningCard />
      <RoutinesCard />
      <HeartbeatsCard />
    </Grid>
  );
}
