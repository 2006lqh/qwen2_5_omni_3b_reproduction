#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL="${MODEL_DIR:-$ROOT/../models/Qwen2.5-Omni-3B}"
MAS_ROOT="$ROOT/third_party/EfficientAI/masquant"
RUN_ID=full_reproduction_w4a8_20260524
OUT="$ROOT/outputs/$RUN_ID"
RESULT="$ROOT/results/$RUN_ID"
ACT_DIR="$OUT/activation_scales"
CACHE_DIR="$OUT/cache"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${MASQUANT_CONDA_ENV:-llm_quant}"
if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

mkdir -p "$OUT" "$RESULT" "$ACT_DIR" "$CACHE_DIR"
python "$ROOT/scripts/full_repro_config_check.py"

cd "$MAS_ROOT"

CUDA_VISIBLE_DEVICES=0 python generate_act_scale_shift.py \
  --model "$MODEL" \
  --dataset-type text-audio-vision \
  --nsamples 128 \
  --batch_size 1 \
  --cache_dir "$CACHE_DIR" \
  --scales-output-path "$ACT_DIR"

ACT_SCALES="$ACT_DIR/Qwen2.5-Omni-3B-text-audio-vision-128.pt"

export inference_mode=merged_scales
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model "$MODEL" \
  --mode train \
  --epochs 0 \
  --wbits 4 \
  --abits 8 \
  --let \
  --loss_multi_modal_mae_alpha \
  --dataset-type text-audio-vision \
  --nsamples 128 \
  --batch_size 1 \
  --cache_dir "$CACHE_DIR" \
  --act-scales "$ACT_SCALES" \
  --output_dir "$OUT/smoothquant_w4a8_baseline" \
  --symmetric \
  --group_size 0

export inference_mode=split_scales
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model "$MODEL" \
  --mode train \
  --epochs 2 \
  --wbits 4 \
  --abits 8 \
  --let \
  --loss_multi_modal_mae_alpha \
  --dataset-type text-audio-vision \
  --nsamples 128 \
  --batch_size 1 \
  --cache_dir "$CACHE_DIR" \
  --act-scales "$ACT_SCALES" \
  --output_dir "$OUT/mas_w4a8_split_scales" \
  --symmetric \
  --group_size 0

MAS_PARAMS=$(find "$OUT/mas_w4a8_split_scales" -name mas_parameters.pth -type f | sort | tail -n 1)
python "$ROOT/scripts/run_exp3_weight_proxy.py" \
  --model-dir "$MODEL" \
  --act-scales "$ACT_SCALES" \
  --mas-parameters "$MAS_PARAMS" \
  --out-dir "$OUT/full_weight_proxy_eval" \
  --max-layers 36 \
  --rank 128 \
  --bits 4

python "$ROOT/scripts/full_repro_collect_summary.py" \
  --run-dir "$OUT" \
  --result-dir "$RESULT" \
  --mas-parameters "$MAS_PARAMS" \
  --act-scales "$ACT_SCALES"
