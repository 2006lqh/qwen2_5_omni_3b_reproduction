#!/usr/bin/env bash
set -uo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL="${MODEL_DIR:-$ROOT/../models/Qwen2.5-Omni-3B}"
MAS_ROOT="$ROOT/third_party/EfficientAI/masquant"
RUN_ID=full_reproduction_w4a8_20260524
OUT="$ROOT/outputs/$RUN_ID"
RESULT="$ROOT/results/paper_benchmark_review_20260524"
CACHE_DIR="$OUT/cache"
TASK_DIR="$ROOT/benchmarks/lmms_paper_tasks"
MODE="${1:-smoke}"

# Keep the user's default HF token location, but store downloaded data under the project.
unset HF_HOME
HF_CACHE_ROOT="${HF_CACHE_ROOT:-$ROOT/../hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_CACHE_ROOT/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_CACHE_ROOT/hub}"
export HF_HUB_DISABLE_XET=1
export MASQUANT_LMMS_INCLUDE_PATH="$TASK_DIR"
export inference_mode=split_scales
export MASQUANT_FORCE_CPU=1

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${MASQUANT_CONDA_ENV:-llm_quant}"
if [ -f "$CONDA_SH" ]; then
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

mkdir -p "$RESULT" "$OUT/paper_benchmarks_w4a8" "$CACHE_DIR"
cd "$MAS_ROOT" || exit 2

MAS_PARAMS=$(find "$OUT/mas_w4a8_split_scales" -name mas_parameters.pth -type f | sort | tail -n 1)
ACT_SCALES=$(find "$OUT/activation_scales" -name "*.pt" -type f | sort | tail -n 1)
if [ -z "${MAS_PARAMS:-}" ]; then
  echo "missing MAS parameters under $OUT/mas_w4a8_split_scales" >&2
  exit 2
fi
if [ -z "${ACT_SCALES:-}" ]; then
  echo "missing activation scales under $OUT/activation_scales" >&2
  exit 2
fi

TASKS="${TASKS:-librispeech_test_other,mmmu_val_qwen,omni_bench}"
if [ "${INCLUDE_WENET:-0}" = "1" ]; then
  TASKS="${TASKS},wenet_speech_test_net"
fi

if [ "$MODE" = "full" ]; then
  LIMIT=-1
  DEFAULT_TIMEOUT_SECONDS=86400
else
  LIMIT=2
  DEFAULT_TIMEOUT_SECONDS=7200
fi
LIMIT="${LIMIT_MULTIMODAL:-$LIMIT}"
TIMEOUT_SECONDS="${BENCHMARK_TIMEOUT_SECONDS:-$DEFAULT_TIMEOUT_SECONDS}"
TASK_LABEL=$(printf "%s" "$TASKS" | tr ', /' '___' | tr -cd 'A-Za-z0-9_.-')

export MASQUANT_LMMS_RESULT_JSON="$RESULT/mas_w4a8_${MODE}_${TASK_LABEL}_lmms_results.json"
LOG_PATH="$RESULT/benchmark_run_${MODE}_${TASK_LABEL}.log"
STATUS_JSON="$RESULT/benchmark_run_status_${MODE}_${TASK_LABEL}.json"

export STATUS_JSON LOG_PATH TASKS MODE LIMIT TIMEOUT_SECONDS MAS_PARAMS ACT_SCALES MODEL RESULT

python - <<'PY'
import ast
import json
import os
import re
from datetime import datetime
from pathlib import Path

Path(os.environ["STATUS_JSON"]).write_text(
    json.dumps(
        {
            "task": os.environ["TASKS"],
            "mode": os.environ["MODE"],
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "limit": os.environ["LIMIT"],
            "timeout_seconds": int(os.environ["TIMEOUT_SECONDS"]),
            "result_json": os.environ["MASQUANT_LMMS_RESULT_JSON"],
            "log_path": os.environ["LOG_PATH"],
            "model": os.environ["MODEL"],
            "mas_parameters": os.environ["MAS_PARAMS"],
            "activation_scales": os.environ["ACT_SCALES"],
        },
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)
PY

CMD=(
  python main.py
  --model "$MODEL"
  --mode train
  --epochs 0
  --wbits 4
  --abits 8
  --let
  --dataset-type text-audio-vision
  --nsamples 128
  --batch_size 1
  --cache_dir "$CACHE_DIR"
  --output_dir "$OUT/paper_benchmarks_w4a8"
  --resume "$MAS_PARAMS"
  --act-scales "$ACT_SCALES"
  --symmetric
  --group_size 0
  --tasks_multimodal "$TASKS"
  --limit_multimodal "$LIMIT"
)

START_EPOCH=$(date +%s)
timeout --kill-after=60s "${TIMEOUT_SECONDS}s" "${CMD[@]}" 2>&1 | tee "$LOG_PATH"
EXIT_CODE=$?
END_EPOCH=$(date +%s)
export EXIT_CODE START_EPOCH END_EPOCH

python - <<'PY'
import ast
import json
import os
import re
from datetime import datetime
from pathlib import Path

status_path = Path(os.environ["STATUS_JSON"])
result_path = Path(os.environ["MASQUANT_LMMS_RESULT_JSON"])
log_path = Path(os.environ["LOG_PATH"])
exit_code = int(os.environ["EXIT_CODE"])
duration = int(os.environ["END_EPOCH"]) - int(os.environ["START_EPOCH"])
log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""

if exit_code == 0 and result_path.exists():
    status = "completed"
elif exit_code == 124:
    status = "resource_limited_timeout"
elif "out of memory" in log_text.lower() or "cuda error: out of memory" in log_text.lower():
    status = "oom"
elif "chunkedencodingerror" in log_text.lower() or "incompleteread" in log_text.lower() or "datasetgenerationerror" in log_text.lower():
    status = "failed_dataset_download"
else:
    status = "failed"

metrics = {}
if result_path.exists():
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        for task_name, values in payload.get("results", {}).items():
            if isinstance(values, dict):
                metrics[task_name] = {
                    key: value
                    for key, value in values.items()
                    if isinstance(value, (int, float, str)) and not key.endswith("_stderr")
                }
    except Exception as exc:
        metrics = {"parse_error": f"{type(exc).__name__}: {exc}"}
else:
    for match in re.finditer(r"INFO (\{.*?\})", log_text):
        try:
            candidate = ast.literal_eval(match.group(1))
        except Exception:
            continue
        if isinstance(candidate, dict) and os.environ["TASKS"].split(",")[0] in candidate:
            metrics = {
                task_name: {
                    key: value
                    for key, value in values.items()
                    if isinstance(value, (int, float, str)) and not key.endswith("_stderr")
                }
                for task_name, values in candidate.items()
                if isinstance(values, dict)
            }
            result_path.write_text(
                json.dumps({"results": candidate, "recovered_from_log": True}, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
            break

tail = "\n".join(log_text.splitlines()[-80:])
if status == "failed" and metrics:
    status = "completed_with_recovered_metrics"
payload = {
    "task": os.environ["TASKS"],
    "mode": os.environ["MODE"],
    "status": status,
    "exit_code": exit_code,
    "started_epoch": int(os.environ["START_EPOCH"]),
    "ended_at": datetime.now().isoformat(timespec="seconds"),
    "duration_seconds": duration,
    "limit": os.environ["LIMIT"],
    "timeout_seconds": int(os.environ["TIMEOUT_SECONDS"]),
    "result_json": str(result_path),
    "result_json_exists": result_path.exists(),
    "log_path": str(log_path),
    "metrics": metrics,
    "log_tail": tail,
}
status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"benchmark_status={status_path}")
print(f"benchmark_result={result_path}")
print(f"benchmark_log={log_path}")
print(f"benchmark_exit_code={exit_code}")
print(f"benchmark_status_value={status}")
PY

exit "$EXIT_CODE"
