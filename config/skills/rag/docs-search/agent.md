# docs-search skill (Document RAG R0)

One tool, **`search_documents`**, over `iris_harness.services.rag.search_documents`: it searches
the user's **indexed documents** (markdown notes, an Obsidian vault, PDFs
later) and returns matching passages **with citations** back to the source
file + section. Use it to ground and attribute answers in the user's own
knowledge base.

IRIS **indexes, doesn't own** — the user points it at an existing file/folder
(`iris docs add <path>`), the files stay the source of truth, and IRIS
re-syncs on change (`iris docs sync`). Retrieval is vector-based when the
local embedding index is available, with a deterministic keyword fallback
otherwise.

Distinct from the auto-synthesised knowledge **wiki** (entity pages from
email): these are **user-provided source documents**.
