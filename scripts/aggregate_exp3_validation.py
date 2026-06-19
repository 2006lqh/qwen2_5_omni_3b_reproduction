import argparse
import csv
import json
from pathlib import Path


METHOD_ORDER = ["smoothquant_unified", "mas_split_scales", "simplified_cmc"]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate three MASQuant validation runs.")
    parser.add_argument("--run", action="append", nargs=2, metavar=("RUN_NAME", "SUMMARY_CSV"), required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_summary(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return {row["method"]: row for row in csv.DictReader(f)}


def write_svg(rows, out_path):
    width, height = 980, 540
    left, right, top, bottom = 90, 40, 70, 100
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_value = max(float(row["mean_error"]) for row in rows) * 1.15
    colors = {
        "smoothquant_unified": "#4C78A8",
        "mas_split_scales": "#59A14F",
        "simplified_cmc": "#F28E2B",
    }

    groups = sorted({row["run"] for row in rows})
    bar_w = chart_w / (len(groups) * 5)

    def y_for(value):
        return top + chart_h - (value / max_value) * chart_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="490" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="700">Three Validation Runs: Mean Error</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{width - right}" y2="{top + chart_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#333"/>',
        f'<text x="24" y="{top + chart_h / 2}" transform="rotate(-90 24 {top + chart_h / 2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Mean error</text>',
    ]
    for tick in range(6):
        value = max_value * tick / 5
        y = y_for(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12">{value:.3f}</text>')

    for gi, run_name in enumerate(groups):
        base_x = left + gi * 5 * bar_w + bar_w
        for mi, method in enumerate(METHOD_ORDER):
            row = next(r for r in rows if r["run"] == run_name and r["method"] == method)
            value = float(row["mean_error"])
            x = base_x + mi * bar_w
            y = y_for(value)
            h = top + chart_h - y
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w * 0.82:.2f}" height="{h:.2f}" fill="{colors[method]}"/>')
            parts.append(f'<text x="{x + bar_w * 0.41:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10">{value:.3f}</text>')
        parts.append(f'<text x="{base_x + bar_w:.2f}" y="{top + chart_h + 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">{run_name}</text>')

    legend_x = left
    legend_y = height - 42
    for i, method in enumerate(METHOD_ORDER):
        x = legend_x + i * 240
        parts.append(f'<rect x="{x}" y="{legend_y - 12}" width="16" height="16" fill="{colors[method]}"/>')
        parts.append(f'<text x="{x + 22}" y="{legend_y + 1}" font-family="Arial, sans-serif" font-size="12">{method}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_name, summary_path in args.run:
        summary = read_summary(summary_path)
        for method in METHOD_ORDER:
            row = summary[method]
            rows.append(
                {
                    "run": run_name,
                    "method": method,
                    "mean_error": row["mean_error"],
                    "mean_sqnr_db": row["mean_sqnr_db"],
                    "summary_csv": summary_path,
                }
            )

    comparison_csv = out_dir / "masquant_w4a8_three_run_comparison.csv"
    with open(comparison_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run", "method", "mean_error", "mean_sqnr_db", "summary_csv"]
        )
        writer.writeheader()
        writer.writerows(rows)

    means = {}
    for method in METHOD_ORDER:
        vals = [float(row["mean_error"]) for row in rows if row["method"] == method]
        sqnrs = [float(row["mean_sqnr_db"]) for row in rows if row["method"] == method]
        means[method] = {
            "mean_error_avg": sum(vals) / len(vals),
            "mean_error_min": min(vals),
            "mean_error_max": max(vals),
            "mean_sqnr_db_avg": sum(sqnrs) / len(sqnrs),
        }

    stats_json = out_dir / "masquant_w4a8_three_run_statistics.json"
    stats_json.write_text(json.dumps(means, indent=2, ensure_ascii=False), encoding="utf-8")

    svg_path = out_dir / "masquant_w4a8_three_run_mean_error.svg"
    write_svg(rows, svg_path)

    md_path = out_dir / "masquant_w4a8_three_run_report.md"
    lines = [
        "# Three Validation Runs",
        "",
        "Metric: deployed-weight reconstruction proxy on selected thinker layers. Lower mean_error is better; higher SQNR is better.",
        "",
        "| Run | SmoothQuant error | MAS error | CMC error |",
        "| --- | ---: | ---: | ---: |",
    ]
    by_run = {}
    for row in rows:
        by_run.setdefault(row["run"], {})[row["method"]] = row
    for run_name in sorted(by_run):
        lines.append(
            f"| {run_name} | "
            f"{float(by_run[run_name]['smoothquant_unified']['mean_error']):.6f} | "
            f"{float(by_run[run_name]['mas_split_scales']['mean_error']):.6f} | "
            f"{float(by_run[run_name]['simplified_cmc']['mean_error']):.6f} |"
        )
    lines.extend(["", "## Averages", "", "| Method | Avg mean_error | Avg SQNR dB |", "| --- | ---: | ---: |"])
    for method in METHOD_ORDER:
        lines.append(
            f"| {method} | {means[method]['mean_error_avg']:.6f} | {means[method]['mean_sqnr_db_avg']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"comparison_csv={comparison_csv}")
    print(f"stats_json={stats_json}")
    print(f"report_md={md_path}")
    print(f"svg={svg_path}")


if __name__ == "__main__":
    main()
