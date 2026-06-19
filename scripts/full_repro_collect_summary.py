import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--mas-parameters", required=True)
    parser.add_argument("--act-scales", required=True)
    return parser.parse_args()


def newest_file(root, pattern):
    matches = sorted(Path(root).glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    proxy_summary = run_dir / "full_weight_proxy_eval" / "w4a8_three_method_summary.csv"
    rows = []
    if proxy_summary.exists():
        with proxy_summary.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    summary_csv = result_dir / "full_repro_w4a8_method_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["method", "mean_error", "mean_sqnr_db", "description"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "run_dir": str(run_dir),
        "result_dir": str(result_dir),
        "act_scales": args.act_scales,
        "mas_parameters": args.mas_parameters,
        "smoothquant_log": str(newest_file(run_dir / "smoothquant_w4a8_baseline", "**/*.txt")),
        "mas_log": str(newest_file(run_dir / "mas_w4a8_split_scales", "**/*.txt")),
        "proxy_summary": str(proxy_summary),
        "summary_csv": str(summary_csv),
    }
    manifest_path = result_dir / "full_repro_w4a8_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"summary_csv={summary_csv}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
