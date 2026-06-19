#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL="${MODEL_DIR:?MODEL_DIR must point to the local Qwen2.5-Omni-3B checkpoint}"
MAS_PARAMS="${MAS_PARAMS:?MAS_PARAMS must point to mas_parameters.pth}"
LOW_RANK_ADAPTERS="${LOW_RANK_ADAPTERS:?LOW_RANK_ADAPTERS must point to the CMC low-rank adapter tensor}"
MAS_ROOT="$ROOT/third_party/EfficientAI/masquant"
ARTIFACT_ROOT="${LOCAL_ARTIFACT_ROOT:-$ROOT/../local_artifacts}"
RUN_ROOT="${BENCHMARK_OUTPUT_DIR:-$ARTIFACT_ROOT/raw_results/qwen2_5_omni_3b_w4a8/full_benchmark/openslr_librispeech_other_full2939}"
CACHE_DIR="${BENCHMARK_CACHE_DIR:-$RUN_ROOT/cache}"
LOG_DIR="${BENCHMARK_LOG_DIR:-$RUN_ROOT/logs}"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${MASQUANT_CONDA_ENV:-llm_quant}"
if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

mkdir -p "$CACHE_DIR" "$LOG_DIR"
cd "$MAS_ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python infer_mas.py \
  --model "$MODEL" \
  --mode train \
  --LR \
  --quantize \
  --wbits 4 \
  --abits 8 \
  --net qwen2.5-omni-3b \
  --output_dir "$LOG_DIR" \
  --cache_dir "$CACHE_DIR" \
  --batch_size 1 \
  --n_cali_samples 1 \
  --cali_data_type text-audio-vision \
  --rank 0.05 \
  --scales_path "$MAS_PARAMS" \
  --quant_cmc 1 \
  --save_low_rank_adapters "$LOW_RANK_ADAPTERS" \
  --tasks_multimodal openslr_librispeech_other \
  --limit_multimodal -1 \
  --num_fewshot 0 \
  --attn_implementation sdpa
