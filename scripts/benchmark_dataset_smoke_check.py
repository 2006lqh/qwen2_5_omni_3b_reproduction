import json
import os
from pathlib import Path

from datasets import load_dataset


ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "results/paper_benchmark_review_20260524/hf_benchmark_dataset_smoke_status.json"

DATASETS = [
    {
        "benchmark": "LibriSpeech test-other",
        "dataset_path": "lmms-lab/librispeech",
        "dataset_name": "librispeech_test_other",
        "split": "librispeech_test_other",
    },
    {
        "benchmark": "WenetSpeech test-net",
        "dataset_path": "lmms-lab/WenetSpeech",
        "dataset_name": None,
        "split": "test_net",
    },
    {
        "benchmark": "MMMU validation",
        "dataset_path": "lmms-lab/MMMU",
        "dataset_name": None,
        "split": "validation",
    },
    {
        "benchmark": "OmniBench",
        "dataset_path": "lmms-lab/Omni_Bench_fix",
        "dataset_name": None,
        "split": "train",
    },
    {
        "benchmark": "OCRBench",
        "dataset_path": "echo840/OCRBench",
        "dataset_name": None,
        "split": "test",
    },
    {
        "benchmark": "TextVQA val lite",
        "dataset_path": "lmms-lab/LMMs-Eval-Lite",
        "dataset_name": "textvqa_val",
        "split": "lite",
    },
    {
        "benchmark": "VizWiz val lite",
        "dataset_path": "lmms-lab/LMMs-Eval-Lite",
        "dataset_name": "vizwiz_vqa_val",
        "split": "lite",
    },
    {
        "benchmark": "ScienceQA image",
        "dataset_path": "lmms-lab/ScienceQA",
        "dataset_name": "ScienceQA-IMG",
        "split": "test",
    },
]


def try_load(item):
    kwargs = {"split": item["split"], "streaming": True}
    if item["dataset_name"]:
        ds = load_dataset(item["dataset_path"], item["dataset_name"], **kwargs)
    else:
        ds = load_dataset(item["dataset_path"], **kwargs)
    first = next(iter(ds))
    return {
        "status": "ok",
        "first_keys": sorted(first.keys()),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in DATASETS:
        row = dict(item)
        try:
            row.update(try_load(item))
        except Exception as exc:
            row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
        print(f"{row['benchmark']}: {row['status']}")
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"status_json={OUT}")


if __name__ == "__main__":
    main()
