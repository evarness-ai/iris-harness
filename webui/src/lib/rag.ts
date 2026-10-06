/* RAG document client (Phase 4).
 *
 * Wraps the new IRIS API endpoints: POST /rag/upload (multipart → ingest +
 * register in the FileManager catalog, ADR-0066), GET /rag/documents (catalog
 * listing, owner_module=rag), POST /rag/search (vector search with keyword
 * fallback). Read + a single create (upload); no delete in this phase. */
import { apiFetch } from "./http";

export interface RagDocument {
  file_id: string;
  filename: string;
  kind: string;
  classification: string;
  byte_size: number;
  location_tier: string;
  storage_path: string;
  created_at: string;
  updated_at: string;
}

export interface RagDocumentsResponse {
  count: number;
  index_ready: boolean;
  documents: RagDocument[];
}

export interface RagUploadResult {
  filename: string;
  index_ready: boolean;
  sources_added: number;
  sources_updated: number;
  sources_skipped: number;
  /** Files that classified secret: refused, not indexed (any earlier copy removed). */
  sources_denied: number;
  chunks_indexed: number;
  summary: string;
  document: RagDocument | null;
}

export interface RagSearchHit {
  text: string;
  score: number;
  citation: string;
  page: number | null;
}

async function detail(r: Response): Promise<string> {
  try {
    const j = (await r.json()) as { detail?: string };
    if (j?.detail) return j.detail;
  } catch {
    /* non-JSON error body */
  }
  return `HTTP ${r.status} ${r.statusText}`.trim();
}

export async function getRagDocuments(): Promise<RagDocumentsResponse> {
  const r = await apiFetch("/rag/documents", { headers: { accept: "application/json" } });
  if (!r.ok) throw new Error(await detail(r));
  return (await r.json()) as RagDocumentsResponse;
}

export async function uploadRagDocument(file: File): Promise<RagUploadResult> {
  const form = new FormData();
  form.append("file", file);
  const r = await apiFetch("/rag/upload", { method: "POST", body: form });
  if (!r.ok) throw new Error(await detail(r));
  return (await r.json()) as RagUploadResult;
}

export async function searchRag(query: string, limit = 8): Promise<RagSearchHit[]> {
  const r = await apiFetch("/rag/search", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query, limit }),
  });
  if (!r.ok) throw new Error(await detail(r));
  const data = (await r.json()) as { results?: RagSearchHit[] };
  return data.results ?? [];
}

export interface RagDeleteResult {
  deleted: boolean;
  file_id: string;
  de_indexed: boolean;
  file_removed: boolean;
}

/** De-index a document; only deletes on-disk bytes IRIS itself created (ADR-0067). */
export async function deleteRagDocument(fileId: string): Promise<RagDeleteResult> {
  const r = await apiFetch(`/rag/documents/${encodeURIComponent(fileId)}`, { method: "DELETE" });
  if (!r.ok) throw new Error(await detail(r));
  return (await r.json()) as RagDeleteResult;
}

/** Extensions the backend accepts (mirrors RAG_SUPPORTED_SUFFIXES in iris_api). */
export const RAG_ACCEPT = ".md,.markdown,.txt,.text,.pdf,.png,.jpg,.jpeg,.tiff,.tif,.bmp,.webp,.docx";
