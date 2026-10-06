import { useRef, useState } from "react";
import { toast } from "sonner";
import { Search, Trash2, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/Card";
import { Tag } from "@/components/Tag";
import { Section } from "@/components/layout";
import { ConfirmDialog, type ConfirmState } from "@/components/ConfirmDialog";
import {
  ClassificationTag,
  Notice,
  QueryState,
  fmtBytes,
  fmtDateTime,
} from "@/components/control/parts";
import {
  useDeleteDocument,
  useRagDocuments,
  useUploadDocument,
  useWritesEnabled,
} from "@/lib/queries";
import { RAG_ACCEPT, searchRag, type RagDocument, type RagSearchHit } from "@/lib/rag";
import { cn } from "@/lib/utils";

function UploadZone() {
  const upload = useUploadDocument();
  const inputRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);

  const handle = async (files: FileList | null) => {
    if (!files?.length) return;
    for (const f of Array.from(files)) {
      try {
        const res = await upload.mutateAsync(f);
        if (res.sources_denied) {
          toast.error(`${f.name}: classified secret, not indexed (use the vault)`);
        } else if (res.sources_skipped && !res.sources_added && !res.sources_updated) {
          toast.info(`${f.name}: already indexed (unchanged)`);
        } else {
          toast.success(`${f.name}: ${res.chunks_indexed} chunks indexed`);
        }
      } catch (e) {
        toast.error(`${f.name}: ${e instanceof Error ? e.message : "upload failed"}`);
      }
    }
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        void handle(e.dataTransfer.files);
      }}
      className={cn(
        "flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-8 text-center transition-colors",
        drag ? "border-primary bg-primary/5" : "border-border",
      )}
    >
      <Upload className="text-fg-subtle" />
      <div className="text-sm text-fg">Drop files here, or</div>
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={() => inputRef.current?.click()}
        disabled={upload.isPending}
      >
        {upload.isPending ? "Uploading…" : "Choose files"}
      </Button>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={RAG_ACCEPT}
        aria-label="Upload documents"
        className="hidden"
        onChange={(e) => void handle(e.target.files)}
      />
      <div className="text-[11px] text-fg-subtle">
        md · txt · pdf · docx · images — max 25 MB. Indexed locally; secret/personal stay on device.
      </div>
    </div>
  );
}

function DocumentsList() {
  const { data, isLoading, isError } = useRagDocuments();
  const del = useDeleteDocument();
  const canWrite = useWritesEnabled();
  const [confirm, setConfirm] = useState<ConfirmState | null>(null);
  const docs = data?.documents ?? [];

  const onDelete = (d: RagDocument) =>
    setConfirm({
      title: "Delete document?",
      description: `"${d.filename}" will be de-indexed. IRIS only removes the file from disk if it created it (uploads); files indexed in place are left untouched.`,
      confirmLabel: "Delete",
      destructive: true,
      run: async () => {
        try {
          const res = await del.mutateAsync(d.file_id);
          toast.success(
            res.file_removed ? `${d.filename}: deleted` : `${d.filename}: de-indexed (file kept)`,
          );
        } catch (e) {
          toast.error(e instanceof Error ? e.message : "delete failed");
          throw e;
        }
      },
    });

  return (
    <Section
      title={`Indexed documents (${docs.length})`}
      actions={
        data && !data.index_ready ? (
          <Tag kind="warn">vector index off — keyword fallback</Tag>
        ) : undefined
      }
    >
      <QueryState
        loading={isLoading}
        error={isError}
        empty={docs.length === 0}
        emptyText="No documents indexed yet. Upload one above."
      >
        <div className="space-y-2">
          {docs.map((d) => (
            <div
              key={d.file_id}
              className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border border-border bg-surface p-3"
            >
              <span className="min-w-0 flex-1 truncate text-sm font-medium text-fg">
                {d.filename}
              </span>
              <Tag kind="info">{d.kind}</Tag>
              <ClassificationTag value={d.classification} />
              <span className="font-mono text-[11px] text-fg-subtle">{fmtBytes(d.byte_size)}</span>
              <span className="font-mono text-[11px] text-fg-subtle">
                {fmtDateTime(d.created_at)}
              </span>
              {canWrite && (
                <Button
                  type="button"
                  size="icon"
                  variant="ghost"
                  aria-label={`Delete ${d.filename}`}
                  onClick={() => onDelete(d)}
                  className="h-7 w-7 text-fg-subtle hover:text-danger"
                >
                  <Trash2 size={15} />
                </Button>
              )}
            </div>
          ))}
        </div>
      </QueryState>
      <ConfirmDialog state={confirm} onClose={() => setConfirm(null)} />
    </Section>
  );
}

function SearchPanel() {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<RagSearchHit[] | null>(null);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    const q = query.trim();
    if (!q) return;
    setBusy(true);
    try {
      setHits(await searchRag(q, 8));
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "search failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Section title="Search your knowledge base">
      <div className="flex items-center gap-2 rounded-lg border border-border bg-surface p-2 focus-within:border-primary/50">
        <Search size={16} className="ml-1 text-fg-subtle" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void run();
          }}
          placeholder="Ask across your indexed documents…"
          aria-label="Search documents"
          className="flex-1 bg-transparent px-1 py-1.5 text-sm text-fg placeholder:text-fg-subtle focus:outline-none"
        />
        <Button type="button" size="sm" onClick={() => void run()} disabled={busy || !query.trim()}>
          {busy ? "Searching…" : "Search"}
        </Button>
      </div>

      {hits !== null && (
        <div className="mt-3 space-y-2">
          {hits.length === 0 ? (
            <Notice>No matches.</Notice>
          ) : (
            hits.map((h, i) => (
              <Card key={`${h.citation}-${i}`}>
                <div className="mb-1 flex items-center justify-between gap-2">
                  <span className="font-mono text-[11px] text-primary">{h.citation}</span>
                  <span className="font-mono text-[11px] text-fg-subtle">
                    {(h.score * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="line-clamp-4 text-xs leading-relaxed text-fg-muted">{h.text}</p>
              </Card>
            ))
          )}
        </div>
      )}
    </Section>
  );
}

export function RagScreen() {
  return (
    <div className="space-y-6">
      <UploadZone />
      <SearchPanel />
      <DocumentsList />
    </div>
  );
}
