import csv
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT.parent / "models/Qwen2.5-Omni-3B"))
MAS_ROOT = ROOT / "third_party/EfficientAI/masquant"
RUN_ID = "full_reproduction_w4a8_20260524"
OUT_DIR = ROOT / "outputs" / RUN_ID
RESULT_DIR = ROOT / "results" / RUN_ID


PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
    "lmms_eval": "lmms-eval",
    "jiwer": "jiwer",
    "termcolor": "termcolor",
    "qwen_omni_utils": "qwen-omni-utils",
    "datasets": "datasets",
    "evaluate": "evaluate",
    "matplotlib": "matplotlib",
}

BENCHMARK_ENV = [
    "LIBRISPEECH_DIR",
    "WENETSPEECH_DIR",
    "MMMU_DIR",
    "OCRBENCH_DIR",
    "TEXTVQA_DIR",
    "VIZWIZ_DIR",
    "SCIENCEQA_DIR",
    "OMNIBENCH_DIR",
]


def package_status(import_name, distribution_name):
    try:
        importlib.import_module(import_name)
        try:
            version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            version = ""
        return {"status": "ok", "version": version}
    except Exception as exc:
        return {"status": "missing", "error": f"{type(exc).__name__}: {exc}"}


def count_lines(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def missing_media_refs(path, limit=20):
    missing = []
    if not path.exists():
        return missing
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            obj = json.loads(line)
            stack = [obj]
            while stack:
                value = stack.pop()
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key in {"image", "audio", "video"} and isinstance(item, str):
                            candidate = item.replace("file://", "")
                            if candidate.startswith("/") and not Path(candidate).exists():
                                missing.append({"line": line_no, "path": candidate})
                                if len(missing) >= limit:
                                    return missing
                        else:
                            stack.append(item)
                elif isinstance(value, list):
                    stack.extend(value)
    return missing


def command_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    packages = {name: package_status(name, dist) for name, dist in PACKAGES.items()}
    benchmark_paths = {}
    for env_name in BENCHMARK_ENV:
        value = os.environ.get(env_name, "")
        benchmark_paths[env_name] = {
            "value": value,
            "exists": bool(value) and Path(value).exists(),
        }

    jsonl = MAS_ROOT / "data/jsonls/omnibench.jsonl"
    report = {
        "run_id": RUN_ID,
        "root": str(ROOT),
        "model_dir": {"path": str(MODEL_DIR), "exists": MODEL_DIR.exists()},
        "mas_root": {"path": str(MAS_ROOT), "exists": MAS_ROOT.exists()},
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python": shutil.which("python"),
        "cuda": command_output(["python", "-c", "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no_cuda')"]),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"]),
        "packages": packages,
        "benchmark_paths": benchmark_paths,
        "local_calibration_jsonl": {
            "path": str(jsonl),
            "exists": jsonl.exists(),
            "lines": count_lines(jsonl),
            "missing_media_refs": missing_media_refs(jsonl),
        },
        "required_outputs": {
            "output_dir": str(OUT_DIR),
            "result_dir": str(RESULT_DIR),
        },
    }

    report_path = RESULT_DIR / "full_repro_preflight_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = RESULT_DIR / "full_repro_dependency_status.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item", "status", "detail"])
        writer.writeheader()
        for name, info in packages.items():
            writer.writerow({
                "item": name,
                "status": info["status"],
                "detail": info.get("version", info.get("error", "")),
            })
        for env_name, info in benchmark_paths.items():
            writer.writerow({
                "item": env_name,
                "status": "ok" if info["exists"] else "missing",
                "detail": info["value"],
            })

    missing_packages = [k for k, v in packages.items() if v["status"] != "ok"]
    missing_benchmarks = [k for k, v in benchmark_paths.items() if not v["exists"]]
    missing_media = report["local_calibration_jsonl"]["missing_media_refs"]
    print(f"report={report_path}")
    print(f"dependency_csv={csv_path}")
    print(f"missing_packages={','.join(missing_packages) if missing_packages else 'none'}")
    print(f"missing_benchmarks={','.join(missing_benchmarks) if missing_benchmarks else 'none'}")
    print(f"missing_media_refs={len(missing_media)}")
    print(f"calibration_lines={report['local_calibration_jsonl']['lines']}")


if __name__ == "__main__":
    main()
