#!/usr/bin/env bash
set -u

EXPECTED_ENV="${MASQUANT_CONDA_ENV:-llm_quant}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/../models/Qwen2.5-Omni-3B}"

echo "== Run local Qwen2.5-Omni-3B sanity check =="
if [ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]; then
  echo "[warn] expected conda env ${EXPECTED_ENV}, got ${CONDA_DEFAULT_ENV:-<unset>}"
  echo "Run first:"
  echo "  source ~/miniconda3/etc/profile.d/conda.sh"
  echo "  conda activate ${EXPECTED_ENV}"
fi

cd "$PROJECT_ROOT" || {
  echo "[fail] project root not found: $PROJECT_ROOT"
  exit 1
}

if [ ! -f sanity_check.py ]; then
  echo "[fail] sanity_check.py not found in $PROJECT_ROOT"
  exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
  echo "[fail] local model directory not found: $MODEL_DIR"
  exit 1
fi

python sanity_check.py --model-id "$MODEL_DIR" --local-files-only
status=$?
if [ "$status" -eq 0 ]; then
  echo "[ok] sanity check completed"
else
  echo "[fail] sanity check failed with exit code $status"
fi
exit "$status"
