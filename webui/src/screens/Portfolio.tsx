/* Portfolio screen — investment holdings joined with live market prices.
 *
 * A thin renderer over GET /portfolio. Per-position valuation (live price, P&L,
 * asset class) is computed in iris_personal.finance.portfolio; this screen only filters
 * (asset type / currency) and aggregates the displayed rows into KPI cards, an
 * allocation donut, and a value-share column. */
import { useMemo, useState } from "react";
import { Section } from "@/components/layout";
import { QueryState } from "@/components/control/parts";
import { useMarketIndexes, usePortfolio } from "@/lib/queries";
import type { AssetType, MarketIndex, PortfolioPosition } from "@/lib/control";

const ASSET_LABELS: Record<AssetType | "all", string> = {
  all: "All assets",
  stock: "Stocks",
  etf: "ETFs",
  mutual_fund: "Mutual Funds",
  other: "Other",
};

// Categorical token colors for the allocation donut. Tokens are stored in
// channel form ("224 164 88"), so wrap in rgb() to get a valid CSS color.
const ASSET_COLORS: Record<AssetType, string> = {
  stock: "rgb(var(--node-cognition))",
  mutual_fund: "rgb(var(--node-runtime))",
  etf: "rgb(var(--node-action))",
  other: "rgb(var(--node-governance))",
};

const ASSET_ORDER: AssetType[] = ["stock", "mutual_fund", "etf", "other"];

function fmtAmount(value: string | number | null): string {
  if (value === null) return "—";
  const n = Number(value);
  if (Number.isNaN(n)) return String(value);
  return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function pnlClass(value: number | string | null): string {
  if (value === null) return "text-fg-muted";
  const n = Number(value);
  if (Number.isNaN(n) || n === 0) return "text-fg-muted";
  return n > 0 ? "text-success" : "text-danger";
}

function pct(value: number | null): string {
  if (value === null) return "";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function Dropdown<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-xs text-fg-muted">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="rounded-md border border-border bg-surface px-2 py-1 text-xs text-fg outline-none focus:border-fg-muted"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

interface Kpi {
  currency: string;
  value: number;
  invested: number;
  pnl: number;
  hasCost: boolean;
  count: number;
  priced: number;
  byType: Map<AssetType, number>; // value by asset type, for allocation
}

function computeKpis(positions: PortfolioPosition[]): Kpi[] {
  const byCcy = new Map<string, Kpi>();
  for (const p of positions) {
    const k =
      byCcy.get(p.currency) ??
      ({
        currency: p.currency,
        value: 0,
        invested: 0,
        pnl: 0,
        hasCost: false,
        count: 0,
        priced: 0,
        byType: new Map<AssetType, number>(),
      } satisfies Kpi);
    const mv = Number(p.market_value ?? 0);
    k.value += mv;
    k.count += 1;
    if (p.priced) k.priced += 1;
    k.byType.set(p.asset_type, (k.byType.get(p.asset_type) ?? 0) + mv);
    if (p.cost_basis !== null) {
      k.invested += Number(p.cost_basis);
      k.pnl += Number(p.pnl ?? 0);
      k.hasCost = true;
    }
    byCcy.set(p.currency, k);
  }
  return [...byCcy.values()].sort((a, b) => b.value - a.value);
}

function Donut({ k }: { k: Kpi }) {
  const R = 42;
  const SW = 14;
  const segments = ASSET_ORDER.map((type) => ({ type, value: k.byType.get(type) ?? 0 })).filter(
    (s) => s.value > 0,
  );
  let cum = 0;
  return (
    <div className="flex items-center gap-4">
      <svg viewBox="0 0 120 120" className="h-28 w-28 -rotate-90">
        <circle cx="60" cy="60" r={R} fill="none" stroke="rgb(var(--border))" strokeWidth={SW} />
        {k.value > 0 &&
          segments.map((s) => {
            const p = (s.value / k.value) * 100;
            const el = (
              <circle
                key={s.type}
                cx="60"
                cy="60"
                r={R}
                fill="none"
                pathLength={100}
                strokeWidth={SW}
                strokeDasharray={`${p} ${100 - p}`}
                strokeDashoffset={-cum}
                // eslint-disable-next-line react/forbid-dom-props -- per-segment token color set at runtime
                style={{ stroke: ASSET_COLORS[s.type] }}
              />
            );
            cum += p;
            return el;
          })}
      </svg>
      <ul className="min-w-[10rem] space-y-1 text-xs">
        {segments.map((s) => (
          <li key={s.type} className="flex items-center gap-2">
            <span
              className="inline-block h-2.5 w-2.5 rounded-sm"
              // eslint-disable-next-line react/forbid-dom-props -- per-segment token color set at runtime
              style={{ background: ASSET_COLORS[s.type] }}
            />
            <span className="text-fg-muted">{ASSET_LABELS[s.type]}</span>
            <span className="ml-auto font-mono text-fg-subtle">
              {((s.value / k.value) * 100).toFixed(1)}%
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="rounded-lg border border-border bg-surface px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-fg-subtle">{label}</div>
      <div className={`mt-0.5 text-sm font-semibold ${tone ?? "text-fg"}`}>{value}</div>
    </div>
  );
}

function CurrencyBlock({ k }: { k: Kpi }) {
  const pnlPct = k.hasCost && k.invested !== 0 ? (k.pnl / k.invested) * 100 : null;
  return (
    <div className="space-y-3">
      <div className="text-xs font-medium text-fg-muted">{k.currency}</div>
      <div className="flex flex-wrap items-center gap-x-8 gap-y-4">
        <Donut k={k} />
        <div className="grid flex-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
          <Stat label="Value" value={fmtAmount(k.value)} />
          <Stat label="Invested" value={k.hasCost ? fmtAmount(k.invested) : "—"} />
          <Stat
            label="Unrealized P&L"
            value={
              k.hasCost ? `${fmtAmount(k.pnl)}${pnlPct !== null ? ` (${pct(pnlPct)})` : ""}` : "—"
            }
            tone={k.hasCost ? pnlClass(k.pnl) : undefined}
          />
          <Stat label="Holdings" value={`${k.count} · ${k.priced} priced`} />
        </div>
      </div>
    </div>
  );
}

function PositionRow({ p, share }: { p: PortfolioPosition; share: number | null }) {
  return (
    <tr className="border-b border-border/60 last:border-0">
      <td className="py-1.5 pl-3 pr-3">
        <div className="text-xs font-medium text-fg">{p.symbol ?? p.name}</div>
        <div className="max-w-[24ch] truncate text-[11px] text-fg-subtle" title={p.name}>
          {p.symbol ? p.name : (p.isin ?? "")}
        </div>
      </td>
      <td className="py-1.5 pr-3 text-[11px] text-fg-subtle">{ASSET_LABELS[p.asset_type]}</td>
      <td className="py-1.5 pr-3 text-right font-mono text-xs text-fg-muted">
        {p.quantity ?? "—"}
      </td>
      <td className="py-1.5 pr-3 text-right font-mono text-xs text-fg">
        {p.live_price !== null ? p.live_price.toLocaleString() : "—"}
      </td>
      <td className={`py-1.5 pr-3 text-right font-mono text-xs ${pnlClass(p.day_change_pct)}`}>
        {p.day_change_pct !== null ? pct(p.day_change_pct) : "—"}
      </td>
      <td className="py-1.5 pr-3 text-right font-mono text-xs text-fg">
        {fmtAmount(p.market_value)}
      </td>
      <td className="py-1.5 pr-3 text-right font-mono text-xs text-fg-muted">
        {share !== null ? `${share.toFixed(1)}%` : "—"}
      </td>
      <td className={`py-1.5 pr-3 text-right font-mono text-xs ${pnlClass(p.pnl)}`}>
        {p.pnl !== null ? (
          <>
            {fmtAmount(p.pnl)} <span className="text-[11px]">({pct(p.pnl_pct)})</span>
          </>
        ) : (
          "—"
        )}
      </td>
    </tr>
  );
}

function IndexChip({ idx }: { idx: MarketIndex }) {
  return (
    <div className="flex items-baseline gap-1.5 whitespace-nowrap">
      <span className="text-[11px] text-fg-muted">{idx.name}</span>
      <span className="font-mono text-xs text-fg">{idx.price.toLocaleString()}</span>
      <span className={`font-mono text-[11px] ${pnlClass(idx.change_pct)}`}>
        {pct(idx.change_pct)}
      </span>
    </div>
  );
}

function MarketStrip() {
  const { data } = useMarketIndexes();
  const indexes = data?.indexes ?? [];
  if (indexes.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 rounded-lg border border-border bg-surface px-3 py-2">
      <span className="text-[11px] uppercase tracking-wide text-fg-subtle">Markets</span>
      {indexes.map((idx) => (
        <IndexChip key={idx.symbol} idx={idx} />
      ))}
    </div>
  );
}

export function PortfolioScreen() {
  const { data, isLoading, isError } = usePortfolio();
  const [asset, setAsset] = useState<AssetType | "all">("all");
  const [currency, setCurrency] = useState<string>("all");

  const all = data?.positions ?? [];
  const currencies = useMemo(() => [...new Set(all.map((p) => p.currency))].sort(), [all]);

  const filtered = useMemo(
    () =>
      all
        .filter((p) => asset === "all" || p.asset_type === asset)
        .filter((p) => currency === "all" || p.currency === currency)
        .sort((a, b) => Number(b.market_value ?? 0) - Number(a.market_value ?? 0)),
    [all, asset, currency],
  );

  const kpis = useMemo(() => computeKpis(filtered), [filtered]);
  // Value-share denominator: total value of each currency in the filtered set.
  const ccyTotals = useMemo(
    () => new Map(kpis.map((k) => [k.currency, k.value])),
    [kpis],
  );

  return (
    <div className="space-y-6">
      <MarketStrip />
      <Section title="Summary">
        <QueryState
          loading={isLoading}
          error={isError}
          empty={!data || data.count === 0}
          emptyText="No holdings yet — ingest a CAS/broker statement, then import broker cost basis."
        >
          {data && (
            <div className="space-y-5">
              <div className="flex flex-wrap items-center gap-4">
                <Dropdown
                  label="Type"
                  value={asset}
                  onChange={setAsset}
                  options={(["all", "stock", "etf", "mutual_fund", "other"] as const).map((v) => ({
                    value: v,
                    label: ASSET_LABELS[v],
                  }))}
                />
                <Dropdown
                  label="Currency"
                  value={currency}
                  onChange={setCurrency}
                  options={[
                    { value: "all", label: "All currencies" },
                    ...currencies.map((c) => ({ value: c, label: c })),
                  ]}
                />
                <span className="ml-auto text-[11px] text-fg-subtle">
                  {filtered.length}/{data.count} holdings
                  {data.as_of_dates.length > 0 && <> · as of {data.as_of_dates.join(", ")}</>}
                </span>
              </div>
              {kpis.length === 0 ? (
                <div className="text-xs text-fg-muted">No holdings match the filter.</div>
              ) : (
                kpis.map((k) => <CurrencyBlock key={k.currency} k={k} />)
              )}
            </div>
          )}
        </QueryState>
      </Section>

      {filtered.length > 0 && (
        <Section title="Holdings">
          <div className="overflow-x-auto rounded-lg border border-border bg-surface">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-border text-[11px] uppercase tracking-wide text-fg-subtle">
                  <th className="py-2 pl-3 pr-3 font-medium">Holding</th>
                  <th className="py-2 pr-3 font-medium">Type</th>
                  <th className="py-2 pr-3 text-right font-medium">Qty</th>
                  <th className="py-2 pr-3 text-right font-medium">Live</th>
                  <th className="py-2 pr-3 text-right font-medium">Day</th>
                  <th className="py-2 pr-3 text-right font-medium">Value</th>
                  <th className="py-2 pr-3 text-right font-medium">Share</th>
                  <th className="py-2 pr-3 text-right font-medium">P&amp;L</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((p, i) => {
                  const tot = ccyTotals.get(p.currency) ?? 0;
                  const mv = Number(p.market_value ?? 0);
                  const share = tot > 0 ? (mv / tot) * 100 : null;
                  return (
                    <PositionRow key={`${p.isin ?? p.symbol ?? p.name}:${i}`} p={p} share={share} />
                  );
                })}
              </tbody>
            </table>
          </div>
        </Section>
      )}
    </div>
  );
}
