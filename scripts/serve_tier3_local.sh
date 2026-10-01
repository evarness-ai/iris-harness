#!/usr/bin/env bash
# Launch llama-server tuned for Qwen3-30B-A3B (MoE, 3B active) on Apple Silicon.
#
# Targets the Phase 1 spike configuration (see docs/architecture/spikes/
# phase1_email_classification.md). Use as the local Tier 3 endpoint while
# the spike runs.
#
# Environment overrides:
#   MODEL    — path to GGUF (default: ~/models/qwen3-30b-a3b-q4.gguf)
#   PORT     — HTTP port (default: 8090)
#   CTX_SIZE — context window (default: 32768; well below the model's
#              256K ceiling but plenty for email-classification prompts)
#
# Notes on flag choices:
#   --n-gpu-layers 999    offload all layers to the Metal GPU. On an
#                         M4 Max with 28GB Metal VRAM the 17GB model
#                         fits comfortably; full-GPU residency beats
#                         CPU-MoE-offload for inference speed here.
#                         (The original brief used `--n-cpu-moe 41` as
#                         a workaround for VRAM-limited boxes; dropped
#                         on this hardware. Re-add if running on <24GB
#                         VRAM. Note: --n-cpu-moe forces Flash Attention
#                         off, which breaks quantized V cache below.)
#   --no-mmap             load model fully into RAM up front. Slightly
#                         slower startup, faster steady-state inference
#                         on Apple Silicon's unified memory.
#   --cache-type-k q8_0   8-bit K cache (halves memory vs f16, near-
#                         lossless for inference). K-cache quantization
#                         does NOT require Flash Attention.
#   (V cache stays at default f16. A quantized V cache (--cache-type-v q4_0)
#    REQUIRES Flash Attention — and FA is a severe regression here. See the
#    "Flash Attention: MEASURED, keep OFF" note below.)
#   --threads <p-cores>   pin to performance cores
#
# Flash Attention: MEASURED, keep OFF (2026-06-30).
#   FA *is* present for this Qwen3-30B-A3B MoE on Metal in b9290 (the older
#   "not supported yet" note was wrong) — but the Metal FA kernel hits a slow
#   path for this sparse MoE and regresses hard. llama-bench on an M4 Max,
#   Qwen3-30B-A3B-Q4, b9290 (tok/s, higher = better):
#     config (ctk q8_0)          pp512   tg128   pp512@4k   tg128@4k
#     baseline  -fa 0  ctv f16    1141    85.6      703       45.2   <- current
#     FA only   -fa 1  ctv f16     516    37.6       78       16.1
#     FA + q4_0 -fa 1  ctv q4_0    442    51.9       37       14.3
#   Prompt processing at 4K context collapses ~19x under FA; generation ~halves.
#   So the quantized V cache the original brief chased ('turbo3') is a net loss
#   here because it can only run with FA. Re-run `scripts/bench_tier3_fa.sh`
#   (or set FLASH_ATTN=1) on a future llama.cpp build to check if Metal MoE FA
#   has improved; flip the default only if FA beats the baseline.
#
# The original Phase 1 spike brief showed "turbo4"/"turbo3" cache types;
# those aren't in llama.cpp b9290 (allowed types: f32, f16, bf16, q8_0,
# q4_0, q4_1, iq4_nl, q5_0, q5_1). q8_0/q4_0 is the standard equivalent
# for substantial memory savings.
set -euo pipefail

MODEL="${MODEL:-$HOME/models/qwen3-30b-a3b-q4.gguf}"
PORT="${PORT:-8090}"
CTX_SIZE="${CTX_SIZE:-32768}"
# Opt-in FA experiment toggle (default off — measured slower, see note above).
# FLASH_ATTN=1 also enables a q4_0 V cache (CACHE_TYPE_V override) since the
# two only make sense together.
FLASH_ATTN="${FLASH_ATTN:-0}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-}"

if [[ ! -f "$MODEL" ]]; then
  echo "error: model not found at $MODEL" >&2
  echo "download via:" >&2
  echo "  curl -L -o '$MODEL' \\" >&2
  echo "    'https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf'" >&2
  exit 1
fi

THREADS="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)"

# Assemble cache/FA flags. FA defaults off (measured slower — see header note).
# Turning FA on auto-selects a q4_0 V cache unless CACHE_TYPE_V is set explicitly.
FA_FLAGS=()
if [[ "$FLASH_ATTN" == "1" ]]; then
  FA_FLAGS+=(--flash-attn on)
  : "${CACHE_TYPE_V:=q4_0}"
fi
[[ -n "$CACHE_TYPE_V" ]] && FA_FLAGS+=(--cache-type-v "$CACHE_TYPE_V")

echo "starting llama-server on port $PORT"
echo "  model       : $MODEL"
echo "  ctx-size    : $CTX_SIZE"
echo "  threads     : $THREADS (performance cores)"
echo "  flash-attn  : $FLASH_ATTN (default 0 — measured slower on Metal MoE, see script header)"
echo "  cache-type-k: $CACHE_TYPE_K"
echo "  cache-type-v: ${CACHE_TYPE_V:-f16 (default)}"
echo ""

exec llama-server \
  --model "$MODEL" \
  --port "$PORT" \
  --host 127.0.0.1 \
  --n-gpu-layers 999 \
  --no-mmap \
  --cache-type-k "$CACHE_TYPE_K" \
  "${FA_FLAGS[@]}" \
  --ctx-size "$CTX_SIZE" \
  --threads "$THREADS"
