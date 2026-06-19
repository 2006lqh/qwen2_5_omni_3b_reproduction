import csv
import json
import os
from datetime import datetime
from pathlib import Path


ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
RESULT = ROOT / "results/paper_benchmark_review_20260524"
OUT = ROOT / "outputs/full_reproduction_w4a8_20260524"

METRIC_BY_BENCHMARK = {
    "LibriSpeech test-other": "WER",
    "WenetSpeech test-net public mirror": "MER/WER",
    "WenetSpeech gated source": "MER/WER",
    "MMMU validation": "Accuracy",
    "OmniBench": "Accuracy",
}


def read_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def row(item, status, metric="", samples="", result_path="", note=""):
    return {
        "item": item,
        "status": status,
        "metric": metric,
        "samples": samples,
        "result_path": result_path,
        "note": note,
    }


def main():
    RESULT.mkdir(parents=True, exist_ok=True)
    rows = []

    hf = read_json(RESULT / "hf_benchmark_access_status.json", {})
    auth = hf.get("auth", {})
    if auth:
        login_status = "ok" if auth.get("logged_in") else "blocked"
        rows.append(
            row(
                "HF login",
                login_status,
                result_path=str(RESULT / "hf_benchmark_access_status.json"),
                note="HF authentication was available; local token and cache paths are intentionally omitted",
            )
        )
    else:
        rows.append(row("HF login", "not_checked", note="hf_benchmark_access_status.json not found"))

    for item in hf.get("datasets", []):
        benchmark = item.get("benchmark", "")
        status = item.get("status", "unknown")
        note = item.get("error") or f"first_keys={item.get('first_keys', [])}"
        rows.append(
            row(
                f"{benchmark} data",
                status,
                metric=METRIC_BY_BENCHMARK.get(benchmark, item.get("metric", "")),
                samples="first sample readable" if status == "ok" else "",
                result_path=str(RESULT / "hf_benchmark_access_status.json"),
                note=note,
            )
        )

    for path in sorted(RESULT.glob("benchmark_run_status_*.json")):
        run = read_json(path, {})
        task = run.get("task", path.stem)
        mode = run.get("mode", "")
        metrics = run.get("metrics") or {}
        if metrics:
            note = f"metrics={json.dumps(metrics, ensure_ascii=False)}"
        else:
            tail = run.get("log_tail", "")
            note = tail.splitlines()[-1] if tail else "no metric JSON produced"
        rows.append(
            row(
                f"MAS W4A8 {mode} {task}",
                run.get("status", "unknown"),
                samples=f"limit={run.get('limit', '')}",
                result_path=str(path),
                note=f"exit={run.get('exit_code')}; duration={run.get('duration_seconds')}s; {note}",
            )
        )

    cmc = read_json(OUT / "official_cmc_full_w4a8_status.json", {})
    if cmc:
        status = "completed" if cmc.get("status_code") == 0 and cmc.get("sqnr_csv_exists") else "resource_limited_not_completed"
        rows.append(
            row(
                "Official original CMC",
                status,
                metric="SQNR",
                samples="n_cali_samples=128 target",
                result_path=str(OUT / "official_cmc_full_w4a8_status.json"),
                note=f"status_code={cmc.get('status_code')}; sqnr_csv_exists={cmc.get('sqnr_csv_exists')}; low_rank_adapters_exists={cmc.get('low_rank_adapters_exists')}",
            )
        )
    else:
        rows.append(row("Official original CMC", "not_run_or_missing_status", metric="SQNR", result_path=str(OUT / "official_cmc_full_w4a8_status.json")))

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "policy": "No sudo/chmod/chown used; no file permission changes.",
        "rows": rows,
    }
    json_path = RESULT / "latest_reproduction_attempt_status.json"
    csv_path = RESULT / "latest_reproduction_attempt_status.csv"
    md_path = RESULT / "latest_reproduction_attempt_status.md"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Latest MASQuant reproduction attempt status", ""]
    for item in rows:
        lines.append(f"- {item['item']}: {item['status']} | {item['note']}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"json={json_path}")
    print(f"csv={csv_path}")
    print(f"md={md_path}")


if __name__ == "__main__":
    main()
