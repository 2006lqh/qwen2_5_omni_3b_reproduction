import csv
import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
RESULT = ROOT / "results/paper_benchmark_review_20260524"
SMOKE = RESULT / "hf_benchmark_dataset_smoke_status.json"
OUT = RESULT / "benchmark_path_and_performance_status.csv"

METRICS = {
    "LibriSpeech test-other": "WER",
    "WenetSpeech test-net": "MER/WER",
    "MMMU validation": "Accuracy",
    "OmniBench": "Accuracy",
    "OCRBench": "Accuracy",
    "TextVQA val lite": "Exact match",
    "VizWiz val lite": "Exact match",
    "ScienceQA image": "Exact match",
}


def main():
    rows = []
    smoke_rows = json.loads(SMOKE.read_text(encoding="utf-8"))
    for item in smoke_rows:
        benchmark = item["benchmark"]
        rows.append(
            {
                "benchmark": benchmark,
                "metric": METRICS.get(benchmark, ""),
                "local_path_status": "not_found",
                "hf_dataset_path": item["dataset_path"],
                "hf_dataset_name": item.get("dataset_name") or "",
                "split": item["split"],
                "dataset_smoke_status": item["status"],
                "performance_status": "not_full_benchmark",
                "note": item.get("error", "dataset first sample readable"),
            }
        )

    rows.append(
        {
            "benchmark": "W4A8 local proxy",
            "metric": "mean_error / SQNR",
            "local_path_status": "available",
            "hf_dataset_path": "",
            "hf_dataset_name": "",
            "split": "local synthetic calibration",
            "dataset_smoke_status": "ok",
            "performance_status": "completed",
            "note": "SmoothQuant 9.027 dB; MAS 14.131 dB; simplified CMC 10.985 dB",
        }
    )
    rows.append(
        {
            "benchmark": "Official original CMC",
            "metric": "SQNR",
            "local_path_status": "available_local_jsonl",
            "hf_dataset_path": "",
            "hf_dataset_name": "",
            "split": "omnibench_128 synthetic",
            "dataset_smoke_status": "ok",
            "performance_status": "attempted_no_final_csv",
            "note": "infer_mas.py ran for more than two hours on 6GB GPU and ended without white matrix/adapters/SQNR CSV",
        }
    )

    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={OUT}")


if __name__ == "__main__":
    main()
