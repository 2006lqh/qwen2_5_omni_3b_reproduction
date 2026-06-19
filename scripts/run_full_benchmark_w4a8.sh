#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL="${MODEL_DIR:-$ROOT/../models/Qwen2.5-Omni-3B}"
MAS_ROOT="$ROOT/third_party/EfficientAI/masquant"
RUN_ID=full_reproduction_w4a8_20260524
OUT="$ROOT/outputs/$RUN_ID"
RESULT="$ROOT/results/$RUN_ID"
BENCH_OUT="$OUT/benchmark_sqnr"
CACHE_DIR="$OUT/cache"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${MASQUANT_CONDA_ENV:-llm_quant}"
if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

mkdir -p "$BENCH_OUT/logs" "$RESULT"
cd "$MAS_ROOT"

MAS_PARAMS=$(find "$OUT/mas_w4a8_split_scales" -name mas_parameters.pth -type f | sort | tail -n 1)
if [ -z "${MAS_PARAMS:-}" ]; then
  echo "missing MAS parameters under $OUT/mas_w4a8_split_scales" >&2
  exit 2
fi

INPUT_FILE="$MAS_ROOT/data/jsonls/omnibench_128.jsonl"
CUDA_VISIBLE_DEVICES=0 python infer_mas.py \
  --model "$MODEL" \
  --mode infer \
  --LR \
  --quantize \
  --wbits 4 \
  --abits 8 \
  --net qwen2.5-omni-3b \
  --output_dir "$BENCH_OUT/logs" \
  --n_cali_samples 128 \
  --rank 0.05 \
  --scales_path "$MAS_PARAMS" \
  --quant_cmc 0 \
  --save_white_matrix_path "$BENCH_OUT/white_matrix_text_audio_vision_rank005.pt" \
  --save_low_rank_adapters "$BENCH_OUT/low_rank_adapters_text_audio_vision_rank005.pt" \
  --cali_data_type text-audio-vision \
  --cache_dir "$CACHE_DIR" \
  --input_file "$INPUT_FILE" \
  --output_file "$BENCH_OUT/local_omnibench128_w4a8_cmc_outputs.jsonl" \
  --sqnr_result "$RESULT/local_omnibench128_w4a8_cmc_sqnr.csv" \
  --eval_sqnr \
  --batch_size 1
