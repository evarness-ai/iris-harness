#!/usr/bin/env bash
# The accuracy receipt for the embedder switch: run the existing kNN-gate holdout
# measurement twice on the Mac, once per backend, and diff the writeups.
#
#   bash scripts/embed_backend_holdout.sh gmail:you@gmail.com
#
# Both runs embed the same holdout with the same centroids; only the model runtime
# differs (torch vs the ONNX export of all-MiniLM-L6-v2). The synthetic corpus in
# scripts/embed_bench.py measured mean cosine 1.0000 between them; this is the same
# check on real, labelled mail. Nothing is written to the email store.

set -euo pipefail
ACCOUNT="${1:?usage: $0 <account-id, e.g. gmail:you@gmail.com>}"
# The writeups name the account and describe real mail, so they default to a temp
# directory, never the repo.
OUT="${2:-${TMPDIR:-/tmp}/iris-embed-backend-holdout}"
mkdir -p "$OUT"

for backend in sentence-transformers onnx; do
  echo "== $backend"
  IRIS_EMBED_BACKEND="$backend" poetry run iris email knn-gate \
    --account "$ACCOUNT" --writeup-to "$OUT/$backend.md" | tail -n 20
done

echo
echo "== diff (empty means the two backends classified the holdout identically)"
diff "$OUT/sentence-transformers.md" "$OUT/onnx.md" || true
