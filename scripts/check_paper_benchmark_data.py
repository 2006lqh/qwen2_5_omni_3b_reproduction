import csv
import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
RESULT_DIR = ROOT / "results/paper_benchmark_review_20260524"

BENCHMARKS = [
    ("omni_table_2", "LibriSpeech", "test-other", "audio-text", "WER", "lower", "LIBRISPEECH_DIR"),
    ("omni_table_2", "WenetSpeech", "test-net", "audio-text", "WER", "lower", "WENETSPEECH_DIR"),
    ("omni_table_2", "MMMU", "default", "vision-text", "accuracy", "higher", "MMMU_DIR"),
    ("omni_table_2", "OmniBench", "default", "vision-audio-text", "accuracy", "higher", "OMNIBENCH_DIR"),
    ("vl_table_1", "MMMU", "default", "vision-text", "accuracy", "higher", "MMMU_DIR"),
    ("vl_table_1", "OCRBench", "default", "vision-text", "accuracy", "higher", "OCRBENCH_DIR"),
    ("vl_table_1", "VizWiz", "default", "vision-text", "accuracy", "higher", "VIZWIZ_DIR"),
    ("vl_table_1", "ScienceQA", "default", "vision-text", "accuracy", "higher", "SCIENCEQA_DIR"),
    ("vl_table_1", "TextVQA", "default", "vision-text", "accuracy", "higher", "TEXTVQA_DIR"),
]


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for table, name, split, modality, metric, direction, env_name in BENCHMARKS:
        value = os.environ.get(env_name, "")
        exists = bool(value) and Path(value).exists()
        rows.append(
            {
                "paper_table": table,
                "benchmark": name,
                "split": split,
                "modality": modality,
                "metric": metric,
                "direction": direction,
                "env": env_name,
                "path": value,
                "path_exists": str(exists).lower(),
            }
        )

    csv_path = RESULT_DIR / "paper_benchmark_data_status.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = RESULT_DIR / "paper_benchmark_data_status.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    missing = [row["env"] for row in rows if row["path_exists"] != "true"]
    print(f"status_csv={csv_path}")
    print(f"status_json={json_path}")
    print(f"missing_envs={','.join(dict.fromkeys(missing)) if missing else 'none'}")


if __name__ == "__main__":
    main()
