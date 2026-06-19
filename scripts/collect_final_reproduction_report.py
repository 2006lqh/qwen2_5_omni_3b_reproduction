import csv
import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
RESULT = ROOT / "results"
FULL = RESULT / "full_reproduction_w4a8_20260524"
PAPER = RESULT / "paper_benchmark_review_20260524"


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_many(pattern):
    rows = []
    for path in sorted(PAPER.glob(pattern)):
        payload = read_json(path)
        if payload is not None:
            rows.append({"path": str(path), "payload": payload})
    return rows


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    benchmark_statuses = read_many("benchmark_run_status_*.json")
    benchmark_results = read_many("mas_w4a8_*_lmms_results.json")
    latest_attempt = read_json(PAPER / "latest_reproduction_attempt_status.json")
    payload = {
        "local_w4a8_proxy": read_csv(FULL / "full_repro_w4a8_method_summary.csv"),
        "hf_access": read_json(PAPER / "hf_benchmark_access_status.json"),
        "benchmark_run_statuses": benchmark_statuses,
        "benchmark_result_jsons": benchmark_results,
        "latest_attempt_status": latest_attempt,
        "official_cmc": read_json(FULL / "official_cmc_full_w4a8_status.json"),
    }
    out_json = PAPER / "final_reproduction_status_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# MASQuant final reproduction status", ""]
    lines.append("## Local W4A8 proxy")
    if payload["local_w4a8_proxy"]:
        for row in payload["local_w4a8_proxy"]:
            lines.append(f"- {row.get('method')}: mean_error={row.get('mean_error')}, mean_sqnr_db={row.get('mean_sqnr_db')}")
    else:
        lines.append("- Missing local proxy summary.")

    lines.append("")
    lines.append("## Benchmark data access")
    hf = payload["hf_access"] or {}
    lines.append(f"- HF logged in: {hf.get('auth', {}).get('logged_in')}")
    lines.append("- HF cache paths are local runtime details and are omitted from public summaries.")
    for row in hf.get("datasets", []):
        note = row.get("error") or "first sample readable"
        lines.append(f"- {row.get('benchmark')}: {row.get('status')} ({note})")

    lines.append("")
    lines.append("## MAS LMMS benchmark runs")
    if benchmark_statuses:
        for item in benchmark_statuses:
            run = item["payload"]
            metrics = run.get("metrics") or {}
            metric_note = json.dumps(metrics, ensure_ascii=False) if metrics else "no metric JSON"
            lines.append(
                f"- {run.get('mode')} {run.get('task')}: {run.get('status')} "
                f"(exit={run.get('exit_code')}, seconds={run.get('duration_seconds')}, {metric_note})"
            )
    else:
        lines.append("- No benchmark run status JSON found.")

    lines.append("")
    lines.append("## Official CMC")
    cmc = payload["official_cmc"] or {}
    if cmc:
        lines.append(f"- status_code={cmc.get('status_code')}, sqnr_csv_exists={cmc.get('sqnr_csv_exists')}, low_rank_adapters_exists={cmc.get('low_rank_adapters_exists')}")
    else:
        lines.append("- Official CMC status JSON missing.")

    out_md = PAPER / "final_reproduction_status_summary.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary_json={out_json}")
    print(f"summary_md={out_md}")


if __name__ == "__main__":
    main()
