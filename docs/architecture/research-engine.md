# Research engine

**Status:** Phase 1 shipped 2026-06-26 — written from code (`src/iris_harness/plugins_builtin/research/`).
Replaces the flaky `web_search` (langchain DuckDuckGo wrapper) and complements the
hardcoded `fetch_web_content` skill. Engineering doc.

The research engine is IRIS's web-knowledge tool: **one query in → a ranked,
citation-ready, content-extracted result set out**. The agent calls one tool
(`research`) and never has to chain searches or pick a source — the engine handles
provider selection, crawling, reranking, and caching behind a single seam.

## Pipeline

```
ResearchInput(query, search_type, max_results, fetch_content, rerank, freshness)
        │
   ResearchEngine.research()
        │
   1. cache lookup ──hit──▶ ResearchResult(cached=True)
        │ miss
   2. provider chain (first non-empty wins; config/search_providers.yaml):
        SearXNG → Tavily → Exa → Brave → plugin providers → DuckDuckGo
        │
   3. dedupe by normalized URL
        │
   4. score + rerank  (0.45 semantic · 0.40 trust · 0.15 freshness) → top max_results
        │
   5. fetch_content? ──▶ async Trafilatura crawl → clean markdown into each result
        │
   6. cache.put → ResearchResult
```

Every stage is best-effort: a provider that raises is skipped, ranking/extraction
failures are swallowed, and total failure yields an empty `ResearchResult` with `error`
set — the LLM-facing tool always gets a serializable string.

## Modules (`src/iris_harness/plugins_builtin/research/`)

| File | Role |
|---|---|
| `models.py` | Typed contract: `ResearchInput`/`ResearchResult` and the engine's own mutable `SearchResult` (built from a provider's `SearchHit` by `SearchResult.from_hit`, the one conversion); re-exports the SDK's `SearchHit`, `SearchType`/`Freshness`. |
| `providers/base.py` | The built-ins' base: the SDK's `SearchProvider` protocol plus the `name` each registers under. |
| `providers/{searxng,tavily,exa,brave,duckduckgo}.py` | Concrete backends. |
| `providers/__init__.py` | `register_builtin_providers(api)` (called by `setup`) and `select_providers()` — the chain a call tries now. |
| `extract.py` | Trafilatura crawl + markdown extraction, async via thread executor; 6k-char cap. |
| `rank.py` | `trust_for(url, search_type=...)` domain authority + `score_results()` blended rank. |
| `news_policy.py` | Loads `news_sources.yaml` into the news lens's `domain -> 0..10` map. Cached; a missing or malformed file logs and yields an empty map. |
| `news_sources.yaml` | The news source policy: tiers and their scores, and the domains in each. Adding a source is one line. |
| `cache.py` | `ResearchCache` — SQLite TTL cache (WAL), keyed by query+type+N+fetch. |
| `engine.py` | `ResearchEngine` — orchestrates the pipeline. |
| `tool.py` | `run_research()` (ReAct callable), `ResearchTool`/`build_langchain_tool()` (skills), `web_search_compat()` (back-compat shim). |

## Provider priority & configuration

Providers are tried in order; the first to return hits wins. Selection is by
availability, so unconfigured backends are simply skipped.

**One chain, one seam.** The chain is a core registry (`iris_harness/services/research/
providers.py`, published as `iris_harness.sdk.research`). Every provider joins it through
`PluginAPI.register_search_provider(name, provider, priority=None)` -- the research
plugin's five built-ins in its `setup`, and any other plugin's the same way -- so a
plugin's provider gets the same guards (finance refusal, owner-identifier stripping:
both run in the tool, before the chain), cache, rerank and audit. The order is
`config/search_providers.yaml` (priority, lower first; `enabled: false` drops one); a
provider it does not name runs at its registered `priority`, else `default_priority`
(500: after the keyed built-ins, before DuckDuckGo at 1000). The file is read per call.
A provider leaves the chain when its plugin is no longer mounted. A plugin declares
each provider it registers under `search_providers:` in its manifest (the research
plugin lists its five); an undeclared name is refused and the plugin degraded.

**What a provider returns.** A list of the SDK's frozen `SearchHit` (url, title,
snippet, `published` datetime, `source`, `extra` labels) -- what the service said and
nothing the engine computes. The engine copies each hit into its own `SearchResult`
and fills score, trust and content there, so every provider's hits are scored the
same way (Exa's and Tavily's own scores were always overwritten by the ranker and are
no longer carried). Anything else returned is refused on the call (`TypeError`, charged
to the plugin) and the chain moves on.

| Provider | Enabled by | Notes |
|---|---|---|
| SearXNG | `IRIS_SEARXNG_URL` | **Primary.** Privacy-first, self-hosted meta-search; no key. |
| Tavily | `TAVILY_API_KEY` | LLM-optimized; keyed. |
| Brave | `BRAVE_API_KEY` | Independent index; keyed. |
| DuckDuckGo | always | Keyless floor (the `ddgs` library). |

DuckDuckGo is always present and last, so the engine works out-of-the-box; setting
`IRIS_SEARXNG_URL` (recommended) lifts quality without any code change.

## Ranking

`score_results` blends, per result:
- **semantic** (0.45) — cosine of the query vs `title + snippet` using the same MiniLM
  embeddings as memory recall (`SemanticIndex.embed`), or a lexical Jaccard fallback when
  no embedder is wired;
- **trust** (0.40) — `trust_for(url, search_type=...)`: a 0–10 domain-authority table
  (official docs/GitHub high; SEO/aggregator spam low; unknown 4.0);
- **freshness** (0.15) — decays from `published`.

### The news lens has its own trust table

`search_type="news"` reads `news_sources.yaml` instead of `TRUST_SCORES`, and does **not**
fall through to it. The web table is not merely incomplete for news — it is inverted. It
names no newsroom at all, so reuters.com and apnews.com fall to `DEFAULT_TRUST` (4.0),
while its `".org": 9.0` suffix rule hands any `.org` a 9.0. With trust at 0.40 of the
blend, a news query ranked a random `.org` above the wire service that `.org` was quoting.

The policy is YAML, inside the plugin, so adding a source or re-scoring a tier is a config
edit and never a code change. Tiers run from `wire` (10.0) and `official` (9.5, `.gov` by
suffix) down through `newsroom`, `beat` and `community` to `syndicator` (3.0). Syndicators
— msn.com, yahoo.com, aol.com, news.google.com — sit **below** an unknown domain
deliberately: they republish other outlets' copy, so the domain says nothing about who
reported it, and counting three copies of one wire story as three sources would break any
later corroboration count. `IRIS_RESEARCH_NEWS_SOURCES` overrides the path.

A broken policy file degrades rather than failing the turn: an empty map scores every news
domain at `DEFAULT_TRUST`, which is flat, not inverted.

### The lens is derived, not remembered

`_coerce_input` (`tool.py`) sets `search_type="news"` — and a `day` or `week` window — when
the caller named no lens and the query reads as a news ask. Deterministic and model-free,
like every other pre-call decision in the harness.

It exists because `search_type` defaults to `web`, which is the provider's plain keyword
endpoint. Asked for "breaking news today" on 2026-09-16, that returned a hockey analysis
piece, an Emmys recap and a how-to on weed-eater line — every one a literal match on
"breaking". Nothing in the tool description told the model a news question needs
`search_type="news"`, so it never passed it.

The derivation only ever **adds** the key: a caller that named the lens keeps it, and
`fetch_web_content` builds its `ResearchInput` directly, so its explicit
`search_type="news"` never passes through here.

## Wiring

`builtin_react_tools` (`runtime/react_tools.py`) registers two tools:
- `research` — the full engine (`run_research`, with `SemanticIndex.embed` for rerank);
- `web_search` — now a thin shim over the engine (`web_search_compat`), snippets only,
  so existing callers and prompts keep working while gaining the new provider chain.

## Phase 2 — opt-in enhancers (shipped)

All off by default; each lazily imports its dependency and **degrades gracefully** to the
Phase-1 default if the dep is missing or the backend is unreachable — nothing crashes.

| Enhancer | Enable | Falls back to | Extra dep |
|---|---|---|---|
| **Exa** provider | `EXA_API_KEY` | next provider in chain | none (urllib) |
| **Redis** cache | `IRIS_RESEARCH_CACHE=redis` (+ `IRIS_REDIS_URL`) | SQLite cache | `pip install redis` |
| **BGE** cross-encoder rerank | `IRIS_RESEARCH_RERANKER=bge` (+ `IRIS_RESEARCH_RERANKER_MODEL`) | embedding/lexical rerank | `sentence-transformers` (already a dep; model downloads on first use) |
| **Crawl4AI** crawler | `IRIS_RESEARCH_CRAWLER=crawl4ai` | Trafilatura | `pip install crawl4ai` (+ `playwright install`) |

- **Provider chain with Exa:** SearXNG → Tavily → **Exa** → Brave → DuckDuckGo.
- **BGE rerank** runs as a *second stage* after the blended score, re-scoring only the top
  candidates (`_RERANK_TOP_K=10`) with `0.7·sigmoid(cross_score) + 0.3·prior_score` so trust
  and freshness still count. A missing/failed model is a silent no-op.
- **Crawl4AI** and **Redis** are intentionally *not* in `pyproject` (Crawl4AI pulls a
  headless browser; Redis is a service) — install them only if you opt in. Selection is via
  `build_cache()` / `build_reranker()` / `_select_extractor()`.
- The Redis cache connects to *your own* Redis (`IRIS_REDIS_URL`) over plain TCP. That is your
  infrastructure, outside the governed-egress model (`docs/architecture/plugin-egress.md`): the
  plugin's `egress: {open_web: true}` declaration does not cover it and `api.http` cannot carry it.

## Phase 3 (shipped)

The legacy `web_search` ReAct tool — a thin shim over this engine — is **retired**.
`research` is now the single web-access tool: for a quick fact, callers pass
`fetch_content=false` (snippets only, no page crawl); omit it for full crawl + extract.
The general (rollout-off) handler's `research` function binding dispatches to
`run_research(..., fetch_content=False)`. Removed: the tool, its general-handler
binding/dispatch, and the `web_search_compat` shim function; `web_search` is gone from
the core/reserved/retrieval tool-name sets, the network-egress allowlist, and the
`analyst` persona policy (all now reference `research`).

**Not migrated (by design):** `fetch_web_content`'s structured fetchers stay — the
news categories already route through this engine internally, but **GitHub-trending**
(star counts) and **Yahoo stocks/indexes** (live quotes) have no `research` equivalent,
so the `web-fetch` skill remains for those brief slots.

## Configuration summary

| Env | Effect | Default |
|---|---|---|
| `IRIS_SEARXNG_URL` | SearXNG base URL (primary provider) | unset → DDG |
| `TAVILY_API_KEY` / `BRAVE_API_KEY` / `EXA_API_KEY` | enable keyed providers | unset |
| `IRIS_RESEARCH_CACHE` | `sqlite` (default) / `redis` / `off` | `sqlite` |
| `IRIS_REDIS_URL` | Redis connection (when cache=redis) | `redis://localhost:6379/0` |
| `IRIS_RESEARCH_RERANKER` | `none` (default) / `bge` | `none` |
| `IRIS_RESEARCH_RERANKER_MODEL` | cross-encoder model id | `BAAI/bge-reranker-base` |
| `IRIS_RESEARCH_CRAWLER` | `trafilatura` (default) / `crawl4ai` | `trafilatura` |
| `IRIS_DATA_DIR` | location of `research_cache.db` | `data/` |
