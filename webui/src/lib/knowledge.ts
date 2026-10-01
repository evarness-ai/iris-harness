/* Unified knowledge-graph client (Phase 5).
 *
 * GET /knowledge/graph returns one node/edge graph unifying semantic memory
 * (wiki pages, facts, signals) and the RAG document corpus. Tags bridge the two
 * corpora; [[wikilinks]] resolve across both by title. Read-only. */
import { apiFetch } from "./http";

export type KGKind = "wiki" | "document" | "tag" | "fact" | "signal";
export type KGEdgeKind = "link" | "tag";

export interface KGNode {
  id: string;
  kind: KGKind;
  label: string;
  degree: number;
  meta: Record<string, unknown>;
}

export interface KGEdge {
  id: string;
  source: string;
  target: string;
  kind: KGEdgeKind;
}

export interface KGStats {
  wiki: number;
  documents: number;
  tags: number;
  facts: number;
  signals: number;
  edges: number;
  facts_total: number;
  facts_unlinked: number;
  signals_total: number;
  signals_unlinked: number;
}

export interface KnowledgeGraph {
  nodes: KGNode[];
  edges: KGEdge[];
  stats: KGStats;
}

export const KG_KINDS: KGKind[] = ["wiki", "document", "tag", "fact", "signal"];

export const KG_KIND_LABEL: Record<KGKind, string> = {
  wiki: "Wiki",
  document: "Documents",
  tag: "Tags",
  fact: "Facts",
  signal: "Signals",
};

export async function getKnowledgeGraph(): Promise<KnowledgeGraph> {
  const r = await apiFetch("/knowledge/graph", { headers: { accept: "application/json" } });
  if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText}`.trim());
  return (await r.json()) as KnowledgeGraph;
}
