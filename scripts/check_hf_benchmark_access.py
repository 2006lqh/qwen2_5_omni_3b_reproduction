import csv
import json
import os
import sys
from pathlib import Path


ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
HF_ROOT = Path(os.environ.get("HF_CACHE_ROOT", ROOT.parent / "hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_ROOT / "datasets"))
os.environ.setdefault("HF_HUB_CACHE", str(HF_ROOT / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402


OUT_DIR = ROOT / "results/paper_benchmark_review_20260524"

DATASETS = [
    {
        "benchmark": "LibriSpeech test-other",
        "dataset_path": "lmms-lab/librispeech",
        "dataset_name": "librispeech_test_other",
        "split": "librispeech_test_other",
        "metric": "WER",
    },
    {
        "benchmark": "WenetSpeech test-net public mirror",
        "dataset_path": "lmms-lab/WenetSpeech",
        "dataset_name": None,
        "split": "test_net",
        "metric": "MER/WER",
    },
    {
        "benchmark": "WenetSpeech gated source",
        "dataset_path": "wenet-e2e/wenetspeech",
        "dataset_name": None,
        "split": "test_net",
        "metric": "MER/WER",
    },
    {
        "benchmark": "MMMU validation",
        "dataset_path": "lmms-lab/MMMU",
        "dataset_name": None,
        "split": "validation",
        "metric": "Accuracy",
    },
    {
        "benchmark": "OmniBench",
        "dataset_path": "lmms-lab/Omni_Bench_fix",
        "dataset_name": None,
        "split": "train",
        "metric": "Accuracy",
    },
]


def auth_status():
    try:
        HfApi().whoami()
        return {"logged_in": True}
    except Exception as exc:
        return {"logged_in": False, "error": f"{type(exc).__name__}: {exc}"}


def load_first(item):
    kwargs = {"split": item["split"], "streaming": True}
    if item["dataset_name"]:
        ds = load_dataset(item["dataset_path"], item["dataset_name"], **kwargs)
    else:
        ds = load_dataset(item["dataset_path"], **kwargs)
    first = next(iter(ds))
    return sorted(first.keys())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    auth = auth_status()
    rows = []
    for item in DATASETS:
        row = dict(item)
        try:
            row["first_keys"] = load_first(item)
            row["status"] = "ok"
            row["error"] = ""
        except Exception as exc:
            row["first_keys"] = []
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(f"{row['benchmark']}: {row['status']}")

    payload = {
        "hf_cache": {
            "HF_HOME": os.environ.get("HF_HOME", "default_token_location"),
            "HF_DATASETS_CACHE": os.environ["HF_DATASETS_CACHE"],
            "HF_HUB_CACHE": os.environ["HF_HUB_CACHE"],
            "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", ""),
        },
        "auth": auth,
        "datasets": rows,
    }
    json_path = OUT_DIR / "hf_benchmark_access_status.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = OUT_DIR / "hf_benchmark_access_status.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["benchmark", "metric", "dataset_path", "dataset_name", "split", "status", "error", "first_keys"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else (v or "") for k, v in row.items() if k in fieldnames})
    print(f"auth_logged_in={auth['logged_in']}")
    print(f"status_json={json_path}")
    print(f"status_csv={csv_path}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
