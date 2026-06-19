#!/usr/bin/env bash
set -u

EXPECTED_ENV="llm_quant"

echo "== MASQuant / Qwen2.5-Omni-3B environment check =="
echo "Tip:"
echo "  source ~/miniconda3/etc/profile.d/conda.sh"
echo "  conda activate ${EXPECTED_ENV}"
echo

echo "== Conda =="
echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-<unset>}"
if [ "${CONDA_DEFAULT_ENV:-}" = "$EXPECTED_ENV" ]; then
  echo "[ok] active conda env is ${EXPECTED_ENV}"
else
  echo "[warn] expected conda env ${EXPECTED_ENV}"
fi
if command -v conda >/dev/null 2>&1; then
  conda env list
else
  echo "[warn] conda not found on PATH"
fi
echo

echo "== Python =="
if command -v python >/dev/null 2>&1; then
  echo "python path: $(command -v python)"
  python --version
else
  echo "[fail] python not found"
fi
echo

echo "== Git =="
if command -v git >/dev/null 2>&1; then
  git --version
else
  echo "[fail] git not found"
fi
echo

echo "== CUDA tools =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || nvidia-smi
else
  echo "[warn] nvidia-smi not found"
fi
echo

echo "== Python package checks =="
if command -v python >/dev/null 2>&1; then
python - <<'PY'
import importlib

def check(name, import_name=None):
    module = import_name or name
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "<version unknown>")
        print(f"[ok] {name}: {version}")
        return mod
    except Exception as exc:
        print(f"[fail] {name}: {exc!r}")
        return None

torch = check("torch")
if torch is not None:
    try:
        print("torch cuda available:", torch.cuda.is_available())
        print("torch cuda runtime:", torch.version.cuda)
        if torch.cuda.is_available():
            print("torch cuda device:", torch.cuda.get_device_name(0))
        else:
            print("[warn] CUDA is not available through torch")
    except Exception as exc:
        print("[warn] torch CUDA check failed:", repr(exc))

check("transformers")
check("accelerate")
check("safetensors")
PY
fi
