/* The email judge's API (loop-proof PR 5): what IRIS sorted each email into, and the
 * owner's correction.
 *
 *   GET  /api/v1/email/judgments?bucket=&limit=   rows newest first + the buckets
 *   POST /api/v1/email/judgments/<id>/bucket      {"bucket": "..."} -> the row
 *
 * Every correction — this list, the Action Center card, chat, a Gmail relabel — goes
 * through the same server path, which moves the Gmail label and teaches the judge. */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "./http";

export interface JudgedEmail {
  message_id: string;
  account_id: string;
  sender: string;
  from_address: string;
  subject: string;
  snippet: string;
  received_at: string | null;
  /** The effective bucket: the owner's when corrected, else the judge's. */
  bucket: string | null;
  bucket_name: string;
  judge_bucket: string | null;
  owner_bucket: string | null;
  owner_source: string | null;
  confidence: number | null;
  figures: Record<string, unknown>;
  judged_at: string | null;
  corrected_at: string | null;
}

export interface BucketOption {
  key: string;
  name: string;
}

export interface JudgmentsResponse {
  judgments: JudgedEmail[];
  buckets: BucketOption[];
}

export interface BucketChange {
  judgment: JudgedEmail;
  previous: string | null;
  changed: boolean;
}

const BASE = "/api/v1/email/judgments";

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await apiFetch(url, { headers: { accept: "application/json" }, ...init });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = (await r.json()) as { detail?: unknown };
      if (typeof j?.detail === "string") msg = j.detail;
    } catch {
      /* non-JSON */
    }
    throw new Error(msg);
  }
  return (await r.json()) as T;
}

export function getJudgments(bucket?: string | null, limit = 100): Promise<JudgmentsResponse> {
  const q = new URLSearchParams({ limit: String(limit) });
  if (bucket) q.set("bucket", bucket);
  return call<JudgmentsResponse>(`${BASE}?${q.toString()}`);
}

export function setJudgmentBucket(messageId: string, bucket: string): Promise<BucketChange> {
  return call<BucketChange>(`${BASE}/${encodeURIComponent(messageId)}/bucket`, {
    method: "POST",
    headers: { accept: "application/json", "content-type": "application/json" },
    body: JSON.stringify({ bucket }),
  });
}

const KEY = ["email-judgments"] as const;

export function useJudgments(bucket: string | null) {
  return useQuery({ queryKey: [...KEY, bucket ?? "all"], queryFn: () => getJudgments(bucket) });
}

interface Change {
  messageId: string;
  bucket: string;
  /** The bucket's display name, for the optimistic row. */
  name: string;
}

/** Move one email; the rows change at once and roll back if the server refuses. */
export function useSetJudgmentBucket() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ messageId, bucket }: Change) => setJudgmentBucket(messageId, bucket),
    onMutate: async ({ messageId, bucket, name }: Change) => {
      await qc.cancelQueries({ queryKey: KEY });
      const before = qc.getQueriesData<JudgmentsResponse>({ queryKey: KEY });
      qc.setQueriesData<JudgmentsResponse>({ queryKey: KEY }, (data) =>
        data
          ? {
              ...data,
              judgments: data.judgments.map((j) =>
                j.message_id === messageId ? { ...j, bucket, bucket_name: name } : j,
              ),
            }
          : data,
      );
      return { before };
    },
    onError: (_err, _vars, ctx) => {
      for (const [key, data] of ctx?.before ?? []) qc.setQueryData(key, data);
    },
    onSettled: () => qc.invalidateQueries({ queryKey: KEY }),
  });
}
