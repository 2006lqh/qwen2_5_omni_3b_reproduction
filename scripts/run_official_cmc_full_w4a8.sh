#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL="${MODEL_DIR:-$ROOT/../models/Qwen2.5-Omni-3B}"
MAS_ROOT="$ROOT/third_party/EfficientAI/masquant"
RUN_ID=full_reproduction_w4a8_20260524
OUT="$ROOT/outputs/$RUN_ID"
RESULT="$ROOT/results/$RUN_ID"
BENCH_OUT="$OUT/official_cmc_full_w4a8"
CACHE_DIR="$OUT/cache"
STATUS_JSON="$RESULT/official_cmc_full_w4a8_status.json"

HF_CACHE_ROOT="${HF_CACHE_ROOT:-$ROOT/../hf_cache}"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_CACHE_ROOT/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export inference_mode=split_scales

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${MASQUANT_CONDA_ENV:-llm_quant}"
if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

mkdir -p "$BENCH_OUT/logs" "$RESULT" "$CACHE_DIR"
cd "$MAS_ROOT"

MAS_PARAMS=$(find "$OUT/mas_w4a8_split_scales" -name mas_parameters.pth -type f | sort | tail -n 1)
if [ -z "${MAS_PARAMS:-}" ]; then
  echo "missing MAS parameters under $OUT/mas_w4a8_split_scales" >&2
  exit 2
fi

INPUT_FILE="$MAS_ROOT/data/jsonls/omnibench_128.jsonl"
START_TS=$(date -Is)
set +e
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
  --output_file "$BENCH_OUT/local_omnibench128_w4a8_original_cmc_outputs.jsonl" \
  --sqnr_result "$RESULT/local_omnibench128_w4a8_original_cmc_sqnr.csv" \
  --eval_sqnr \
  --batch_size 1
STATUS=$?
set -e
END_TS=$(date -Is)

python - "$STATUS_JSON" "$STATUS" "$START_TS" "$END_TS" "$BENCH_OUT" "$RESULT/local_omnibench128_w4a8_original_cmc_sqnr.csv" <<'PY'
import json
import sys
from pathlib import Path

status_json, status, start, end, out_dir, sqnr = sys.argv[1:]
out = Path(out_dir)
payload = {
    "status_code": int(status),
    "started_at": start,
    "ended_at": end,
    "output_dir": str(out),
    "white_matrix_exists": (out / "white_matrix_text_audio_vision_rank005.pt").exists(),
    "low_rank_adapters_exists": (out / "low_rank_adapters_text_audio_vision_rank005.pt").exists(),
    "sqnr_csv": sqnr,
    "sqnr_csv_exists": Path(sqnr).exists(),
    "files": sorted(str(p) for p in out.rglob("*") if p.is_file()),
}
Path(status_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"status_json={status_json}")
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY
exit "$STATUS"
