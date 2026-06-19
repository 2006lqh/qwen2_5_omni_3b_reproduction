#!/usr/bin/env bash
set -u

EXPECTED_ENV="${MASQUANT_CONDA_ENV:-llm_quant}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/../models/Qwen2.5-Omni-3B}"
SOURCE_REPO="${PROJECT_ROOT}/third_party/EfficientAI"
SOURCE_DIR="${SOURCE_REPO}/masquant"
LOCAL_CONFIG="${PROJECT_ROOT}/configs/qwen2_5_omni_3b_local.yaml"

warnings=0
failures=0

ok() { printf '[ok] %s\n' "$1"; }
warn() { printf '[warn] %s\n' "$1"; warnings=$((warnings + 1)); }
fail() { printf '[fail] %s\n' "$1"; failures=$((failures + 1)); }

echo "== MASQuant / Qwen2.5-Omni-3B paper reproduction readiness =="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo

if [ "${CONDA_DEFAULT_ENV:-}" = "$EXPECTED_ENV" ]; then
  ok "conda env is ${EXPECTED_ENV}"
else
  warn "expected conda env ${EXPECTED_ENV}, got ${CONDA_DEFAULT_ENV:-<unset>}"
  echo "Run:"
  echo "  source ~/miniconda3/etc/profile.d/conda.sh"
  echo "  conda activate ${EXPECTED_ENV}"
fi

if [ -d "$MODEL_DIR" ]; then
  ok "model path exists: $MODEL_DIR"
else
  fail "model path missing: $MODEL_DIR"
  echo "Keep model weights outside this repository and set MODEL_DIR to their local path."
fi

if [ -d "$SOURCE_REPO" ]; then
  ok "source repo exists: $SOURCE_REPO"
else
  fail "source repo missing: $SOURCE_REPO"
  echo "Run:"
  echo "  cd \"$PROJECT_ROOT/third_party\""
  echo "  git clone https://github.com/alibaba/EfficientAI.git"
fi

if [ -d "$SOURCE_DIR" ]; then
  ok "MASQuant source exists: $SOURCE_DIR"
else
  fail "MASQuant source missing: $SOURCE_DIR"
fi

if [ -f "$LOCAL_CONFIG" ]; then
  ok "local config exists: $LOCAL_CONFIG"
else
  fail "local config missing: $LOCAL_CONFIG"
fi

for dir in outputs results; do
  if [ -d "$PROJECT_ROOT/$dir" ]; then
    ok "$dir/ exists"
  else
    fail "$dir/ missing"
  fi
done

echo
echo "== Runtime package spot check =="
if command -v python >/dev/null 2>&1; then
python - <<'PY'
import importlib

for name in ["torch", "transformers", "accelerate", "safetensors"]:
    try:
        mod = importlib.import_module(name)
        print(f"[ok] {name}: {getattr(mod, '__version__', '<version unknown>')}")
    except Exception as exc:
        print(f"[warn] {name}: {exc!r}")

try:
    import torch
    print("torch cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch cuda device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("[warn] torch CUDA check failed:", repr(exc))
PY
else
  warn "python not found"
fi

echo
echo "== Summary =="
echo "failures: $failures"
echo "warnings: $warnings"
if [ "$failures" -eq 0 ]; then
  ok "project structure is ready for the next reproduction step"
else
  fail "readiness check found blocking issues"
fi

exit 0
