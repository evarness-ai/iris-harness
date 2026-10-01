#!/usr/bin/env bash
# Benchmark Flash Attention + quantized V cache vs the current Tier-3-local
# config (scripts/serve_tier3_local.sh) for Qwen3-30B-A3B on Apple Silicon.
#
# Why this exists: the original Phase 1 spike brief chased a quantized V cache
# ('turbo3'), which on llama.cpp requires Flash Attention. As of b9290 on an
# M4 Max, FA is a SEVERE regression for this sparse MoE on Metal (pp@4k ~19x
# slower, tg ~halved) — see serve_tier3_local.sh's header for the numbers, and
# feedback memory "FA-on-Metal regresses Qwen3 MoE". Re-run this on a future
# llama.cpp build to check whether the Metal MoE FA kernel has improved; only
# flip serve_tier3_local.sh's default if FA beats the baseline below.
#
# Env overrides: MODEL, THREADS, REPS (llama-bench repetitions).
set -uo pipefail

MODEL="${MODEL:-$HOME/models/qwen3-30b-a3b-q4.gguf}"
THREADS="${THREADS:-$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)}"
REPS="${REPS:-3}"

if [[ ! -f "$MODEL" ]]; then
  echo "error: model not found at $MODEL" >&2
  exit 1
fi
if ! command -v llama-bench >/dev/null 2>&1; then
  echo "error: llama-bench not found (install llama.cpp)" >&2
  exit 1
fi

# pp512 + tg128 at depth 0 and 4096 (KV-cache quantization matters at depth).
bench() {
  echo "### $1"; shift
  llama-bench -m "$MODEL" -ngl 999 -t "$THREADS" --mmap 0 \
    -p 512 -n 128 -d 0,4096 -r "$REPS" "$@"
  echo ""
}

echo "# Tier-3-local FA benchmark — $(basename "$MODEL")"
echo "# threads=$THREADS reps=$REPS  $(llama-bench --version 2>/dev/null | head -1)"
echo ""
bench "Baseline (current serve script): -fa 0 -ctk q8_0 -ctv f16" -fa 0 -ctk q8_0 -ctv f16
bench "FA only:                         -fa 1 -ctk q8_0 -ctv f16" -fa 1 -ctk q8_0 -ctv f16
bench "FA + quantized V (target):       -fa 1 -ctk q8_0 -ctv q4_0" -fa 1 -ctk q8_0 -ctv q4_0
