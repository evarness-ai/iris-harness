# email-triage skill

Classifies locally-stored emails against the user's accepted category
taxonomy. Hybrid kNN-then-LLM per ADR-0021. Used by:

- The runtime's `email.new_arrived` subscriber (Track 1G+) —
  auto-classifies new mail as the sweep heartbeat lands it.
- The harness on demand when the user asks "what category is this?" or
  "triage my inbox."
- The `iris email triage` CLI for manual triggering and validation.

## Prerequisites

1. `iris auth gmail login --user <address>` — Gmail account registered.
2. `iris email bootstrap-categories --account <slug>` — produces the
   per-account `proposals.jsonl` the classifier reads for centroids.
3. `iris email accept-categories --account <slug>` — persists the
   accepted (root, branch, leaf) paths into `data/iris.db.categories`.
4. Tier-3-local llama-server running on `localhost:8090` —
   `bash scripts/serve_tier3_local.sh`. Tracks 1H retires this
   dependency by wiring `tier_router.get_llm_config("classify")`.

## Behavior

- **Centroid construction (per-process)**: on first triage call for an
  account, re-embeds the cluster representatives stored in the
  per-account `proposals.jsonl` and computes one L2-normalized
  centroid per accepted category. ~1s on M4 Max; paid once per
  process lifetime, cached on the classifier instance.
- **Per-email pipeline**: embed envelope → cosine-sim against every
  centroid → top-3 → LLM picks one → write
  `classified_category`/`_confidence`/`_at` + emit
  `email.classified`.
- **Confidence** is `chosen.cohesion × 0.9` if the LLM picked from the
  top-3, else `0.5` fallback to kNN top-1. ADR-0021 §6 heuristic;
  Track 1J replaces with a learned model.
- **Soft-fail** per ADR-0021 §7: any error (LLM down, embedder broken,
  JSONL missing) leaves the row's classified fields NULL. The error
  is logged; the next triage run will retry.

## What's NOT here

- Re-classifying already-classified emails. Once written, the
  `classified_category` row sticks. A future `iris email
  recategorize` CLI handles user corrections (Track 1J).
- Wiki ingestion of classified mail. Lives in Track 1K via the
  `email.classified` event.
- Confidence calibration. The current heuristic is intentionally
  simple — Track 1J replaces with a learned model.

## Related

- ADR-0017 — dynamic hierarchical categories.
- ADR-0019 — categories store schema.
- ADR-0020 — drift correction (Path A for wrong root/branch/leaf;
  Path D for duplicate-leaf collisions).
- ADR-0021 — this skill's implementation shape.
- Canonical doc §3.1 — Email skill split + sweep model.
