import { useState, type ReactNode } from "react";
import { Tag } from "@/components/Tag";
import { CopyBlock } from "@/components/CopyBlock";
import { Section } from "@/components/layout";
import {
  ClassificationTag,
  DecisionTag,
  FlagGrid,
  LocalityTag,
  Notice,
  QueryState,
  fmtDateTime,
} from "@/components/control/parts";
import type { AuditEntry, ProofInvariant } from "@/lib/control";
import {
  useGovernanceAudit,
  useGovernanceState,
  usePiiShadow,
  useProofBundleCheck,
} from "@/lib/queries";

const DECISIONS = ["all", "allow", "deny", "transform", "require_approval"] as const;
const WINDOW_DAYS = 7;
const MCP_NAMESPACE = "mcp:";

function StateCard() {
  const { data, isLoading, isError } = useGovernanceState();
  return (
    <Section title="Posture">
      <QueryState loading={isLoading} error={isError} empty={!data} emptyText="No state.">
        {data && (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <Tag kind={data.enabled ? "ok" : "warn"}>
                kernel {data.enabled ? "enabled" : "disabled"}
              </Tag>
              <span className="font-mono text-[11px] text-fg-subtle">
                {data.audit_count} audit entries
              </span>
              <span className="ml-auto truncate font-mono text-[11px] text-fg-subtle">
                {data.audit_db}
              </span>
            </div>
            {data.write_health && !data.write_health.ok && (
              <div className="flex flex-wrap items-center gap-2">
                <Tag kind="warn">
                  {data.write_health.writes_lost > 0 ? "audit rows lost" : "audit spool pending"}
                </Tag>
                <span className="font-mono text-[11px] text-fg-subtle">
                  {data.write_health.spool_pending} in spool, {data.write_health.spool_rejected}{" "}
                  malformed, {data.write_health.writes_lost} lost
                </span>
              </div>
            )}
            <FlagGrid flags={data.flags} />
            <p className="text-[11px] text-fg-subtle">
              Read-only. Enforce flips are an operator action (env vars / the enforce runbook), never
              a UI write.
            </p>
          </div>
        )}
      </QueryState>
    </Section>
  );
}

/* ---- Proof bundle (R14): the three invariants over the last week ---------------- */

const EVIDENCE_LABEL: Record<string, string> = {
  model_call_rows: "model-call rows checked",
  approval_rows: "approval rows",
  writes_observed: "mailbox writes observed",
  model_calls_observed: "model calls observed",
  answers_observed: "answers observed",
};

const PROOF_COMMANDS = [
  `iris governance proof-bundle check --days ${WINDOW_DAYS}`,
  "iris governance proof-bundle export --since <ISO time> --out proof-bundle.json \\",
  "    [--observations mailbox-writes.json]",
  "iris governance proof-bundle verify proof-bundle.json",
].join("\n");

function InvariantRow({ item }: { item: ProofInvariant }) {
  const evidence = Object.entries(item.evidence ?? {});
  // An invariant over nothing holds trivially; say so instead of a bare "ok".
  const vacuous = item.ok && evidence.every(([, n]) => n === 0);
  return (
    <div className="rounded-lg border border-border bg-surface p-3" data-testid="proof-invariant">
      <div className="flex flex-wrap items-center gap-2">
        <Tag kind={item.ok ? (vacuous ? "info" : "ok") : "bad"}>
          {item.ok ? (vacuous ? "holds (nothing observed)" : "holds") : "violated"}
        </Tag>
        <span className="break-all font-mono text-xs text-fg">{item.id}</span>
      </div>
      <p className="mt-1 text-xs text-fg-muted">{item.statement}</p>
      {evidence.length > 0 && (
        <p className="mt-1 font-mono text-[11px] text-fg-subtle">
          {evidence.map(([k, n]) => `${n} ${EVIDENCE_LABEL[k] ?? k}`).join(" · ")}
        </p>
      )}
      {item.violation_count > 0 && (
        <ul className="mt-2 space-y-1">
          {item.violations.map((v) => (
            <li key={v} className="break-words font-mono text-[11px] text-danger">
              {v}
            </li>
          ))}
          {item.violation_count > item.violations.length && (
            <li className="text-[11px] text-fg-subtle">
              and {item.violation_count - item.violations.length} more
            </li>
          )}
        </ul>
      )}
    </div>
  );
}

function ProofBundleCard() {
  const { data, isLoading, isError } = useProofBundleCheck(WINDOW_DAYS);
  const invariants = data?.invariants ?? [];
  return (
    <Section
      title={`Proof bundle · last ${WINDOW_DAYS} days`}
      actions={
        data && invariants.length > 0 ? (
          <Tag kind={data.ok ? "ok" : "bad"}>{data.ok ? "verified" : "violations"}</Tag>
        ) : undefined
      }
    >
      <QueryState
        loading={isLoading}
        error={isError}
        empty={invariants.length === 0}
        emptyText="No proof bundle check available."
      >
        {data && (
          <div className="space-y-2" data-testid="proof-bundle">
            <p className="text-[11px] text-fg-subtle">
              The ledger window ({data.ledger_rows} rows) and the session logs, exported in memory
              and verified offline. Ids, decisions, labels, tiers and digests only.
            </p>
            {data.ledger_rows === 0 && (
              <Notice>
                Nothing recorded in this window yet, so each invariant holds over nothing. Ask
                IRIS something in Chat and the check runs over what that turn wrote.
              </Notice>
            )}
            {(data.integrity ?? []).map((d) => (
              <Notice key={d} tone="danger">
                {d}
              </Notice>
            ))}
            {invariants.map((item) => (
              <InvariantRow key={item.id} item={item} />
            ))}
            <p className="text-[11px] text-fg-subtle">
              Mailbox writes are counted by the email plugin; export passes them with
              --observations.
            </p>
            <CopyBlock label="Export and verify it yourself" text={PROOF_COMMANDS} />
          </div>
        )}
      </QueryState>
    </Section>
  );
}

/* ---- The ledger: who called, how it was answered, where it ran ------------------ */

function CallFacts({ e }: { e: AuditEntry }) {
  const facts: { key: string; node: ReactNode }[] = [];
  if (e.caller) {
    facts.push({
      key: "caller",
      node: (
        <span className="break-all font-mono text-[11px] text-fg" data-testid="audit-caller">
          caller {e.caller}
        </span>
      ),
    });
  }
  if (e.deterministic) {
    facts.push({
      key: "det",
      node: <Tag kind="opp">deterministic{e.handler ? ` · ${e.handler}` : ""}</Tag>,
    });
  }
  const called = e.tool_name ?? (e.capability ? `${e.capability}.${e.method ?? "?"}` : undefined);
  if (called) {
    facts.push({
      key: "tool",
      node: <span className="break-all font-mono text-[11px] text-fg-muted">{called}</span>,
    });
  }
  if (e.digest_alg) {
    facts.push({
      key: "digest",
      node: (
        <span className="break-all font-mono text-[11px] text-fg-subtle" title={e.digest_alg}>
          keyed digest
        </span>
      ),
    });
  }
  if (facts.length === 0) return null;
  return (
    <div className="mt-1 flex flex-wrap items-center gap-2">
      {facts.map((f) => (
        <span key={f.key}>{f.node}</span>
      ))}
    </div>
  );
}

function AuditTable() {
  const [decision, setDecision] = useState<string>("all");
  const [caller, setCaller] = useState<string>("");
  const { data, isLoading, isError } = useGovernanceAudit(
    decision === "all" ? undefined : decision,
    caller || undefined,
  );
  const entries = data?.entries ?? [];
  const callers = data?.callers ?? [];
  const hasMcp = callers.some((c) => c.startsWith(MCP_NAMESPACE));
  return (
    <Section
      title={`Recent decisions${data ? ` (${data.count} of ${data.total})` : ""}`}
      actions={
        <div className="inline-flex overflow-hidden rounded-lg border border-border">
          {DECISIONS.map((d) => (
            <button
              key={d}
              type="button"
              onClick={() => setDecision(d)}
              className={`min-h-[44px] px-2.5 py-1 text-xs font-medium transition-colors sm:min-h-0 ${
                d === decision
                  ? "bg-primary/15 text-primary"
                  : "bg-bg text-fg-muted hover:bg-surface hover:text-fg"
              }`}
            >
              {d}
            </button>
          ))}
        </div>
      }
    >
      {(callers.length > 0 || caller) && (
        <label className="mb-3 flex flex-wrap items-center gap-2 text-[11px] text-fg-subtle">
          caller
          <select
            value={caller}
            onChange={(ev) => setCaller(ev.target.value)}
            aria-label="Filter by caller"
            className="min-h-[44px] max-w-full rounded-lg border border-border bg-surface px-2 text-xs text-fg sm:min-h-0 sm:py-1"
          >
            <option value="">every caller</option>
            {(hasMcp || caller === MCP_NAMESPACE) && (
              <option value={MCP_NAMESPACE}>every MCP client (mcp:*)</option>
            )}
            {callers.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
      )}
      <QueryState
        loading={isLoading}
        error={isError}
        empty={entries.length === 0}
        emptyText={
          data && data.total > 0
            ? "No decisions match this filter."
            : "No governance decisions yet. Every chat turn writes its decisions here: ask IRIS something in Chat, then come back."
        }
      >
        <div className="space-y-2">
          {entries.map((e) => (
            <div
              key={e.id}
              className="rounded-lg border border-border bg-surface p-3"
              data-testid="audit-entry"
            >
              <div className="flex flex-wrap items-center gap-2">
                <DecisionTag decision={e.decision} />
                <span className="font-mono text-xs text-fg">{e.plugin}</span>
                <span className="font-mono text-[11px] text-fg-subtle">{e.hook_point}</span>
                {e.classification && <ClassificationTag value={e.classification} />}
                <LocalityTag locality={e.locality} tier={e.tier} />
                <span className="ml-auto font-mono text-[11px] text-fg-subtle">
                  {fmtDateTime(e.ts)}
                </span>
              </div>
              <CallFacts e={e} />
              {e.reason && <p className="mt-1 break-words text-xs text-fg-muted">{e.reason}</p>}
            </div>
          ))}
        </div>
      </QueryState>
    </Section>
  );
}

/* ---- Owner-PII shadow (ADR-0125): what the guards would have done -------------- */

function PiiShadowCard() {
  const { data, isLoading, isError } = usePiiShadow(WINDOW_DAYS);
  const cells = data?.cells ?? [];
  const checked = Object.entries(data?.checked ?? {});
  const unobserved = Object.entries(data?.unobserved ?? {});
  return (
    <Section
      title={`Owner-PII guards · shadow · last ${WINDOW_DAYS} days`}
      actions={data?.mode ? <Tag kind={data.mode === "off" ? "warn" : "info"}>{data.mode}</Tag> : undefined}
    >
      <QueryState loading={isLoading} error={isError} empty={!data} emptyText="No summary.">
        {data && (
          <div className="space-y-2" data-testid="pii-shadow">
            {data.mode === "off" && (
              <Notice>
                IRIS_GOVERNANCE_OWNER_PII is off here, so nothing is recorded: an empty table would
                not mean a quiet week.
              </Notice>
            )}
            <p className="text-[11px] text-fg-subtle">
              Counts only: the shadow rows hold no literal. {data.rows ?? 0} rows read.
            </p>
            {checked.length > 0 && (
              <p className="font-mono text-[11px] text-fg-subtle">
                calls read per guard: {checked.map(([g, n]) => `${g} ${n}`).join(", ")}
              </p>
            )}
            {unobserved.length > 0 && (
              <p className="font-mono text-[11px] text-warning">
                calls not observed: {unobserved.map(([k, n]) => `${k} ${n}`).join(", ")}
              </p>
            )}
            {cells.length === 0 ? (
              data.mode !== "off" && <Notice>No owner PII observed in the window.</Notice>
            ) : (
              <div className="overflow-x-auto rounded-lg border border-border">
                <table className="w-full min-w-[560px] text-left text-xs">
                  <thead className="bg-surface text-[11px] text-fg-subtle">
                    <tr>
                      {["hook point", "guard", "kind", "would", "occurrences", "calls", "distinct"].map(
                        (h) => (
                          <th key={h} className="px-2 py-1.5 font-medium">
                            {h}
                          </th>
                        ),
                      )}
                    </tr>
                  </thead>
                  <tbody className="font-mono text-[11px] text-fg">
                    {cells.map((c) => (
                      <tr
                        key={`${c.hook_point}|${c.guard}|${c.kind}|${c.action}|${c.first_name_alone}|${c.log_only_destination}`}
                        className="border-t border-border"
                      >
                        <td className="px-2 py-1.5">{c.hook_point}</td>
                        <td className="px-2 py-1.5">{c.guard}</td>
                        <td className="px-2 py-1.5">
                          {c.kind}
                          {c.first_name_alone ? " (first name alone)" : ""}
                        </td>
                        <td className="px-2 py-1.5">
                          {c.action}
                          {c.log_only_destination ? " (log-only destination)" : ""}
                        </td>
                        <td className="px-2 py-1.5">{c.occurrences}</td>
                        <td className="px-2 py-1.5">{c.calls}</td>
                        <td className="px-2 py-1.5">{c.distinct ?? "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </QueryState>
    </Section>
  );
}

export function GovernanceScreen() {
  return (
    <div className="space-y-6">
      <StateCard />
      <ProofBundleCard />
      <AuditTable />
      <PiiShadowCard />
    </div>
  );
}
