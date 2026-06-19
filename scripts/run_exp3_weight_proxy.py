import argparse
import csv
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


DEFAULT_MODULES = [
    "self_attn.q_proj",
    "self_attn.o_proj",
    "mlp.up_proj",
]
MODALITIES = ["text", "vision", "audio"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Small MASQuant reproduction evaluator: compares unified SmoothQuant, "
            "modality-aware scales, and simplified CMC low-rank compensation on "
            "selected Qwen2.5-Omni thinker weights."
        )
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--act-scales", required=True)
    parser.add_argument("--mas-parameters", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--modules", nargs="+", default=DEFAULT_MODULES)
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()


def load_index(model_dir):
    index_path = Path(model_dir) / "model.safetensors.index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def load_weight(model_dir, weight_map, tensor_name):
    shard = weight_map[tensor_name]
    shard_path = Path(model_dir) / shard
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_name).float()


def fake_quant_symmetric_with_zero_point(weight, bits):
    # Match the MASQuant W4 symmetric per-row fake-quant path used for linear weights.
    qmin = 0
    qmax = 2**bits - 1
    zero_point = 2 ** (bits - 1) - 1
    abs_max = weight.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max / zero_point).clamp(min=1e-5, max=1e4)
    q = torch.round(weight / scale).add(zero_point).clamp(qmin, qmax)
    return q.sub(zero_point).mul(scale)


def mse(a, b):
    return torch.mean((a - b) ** 2).item()


def rel_fro_error(target, estimate):
    denominator = torch.linalg.norm(target).clamp(min=1e-12)
    return (torch.linalg.norm(target - estimate) / denominator).item()


def sqnr_db(target, estimate):
    signal = torch.mean(target ** 2).clamp(min=1e-30)
    noise = torch.mean((target - estimate) ** 2).clamp(min=1e-30)
    return (10 * torch.log10(signal / noise)).item()


def low_rank_delta(delta_t, rank):
    full_rank = min(delta_t.shape)
    rank = min(rank, full_rank)
    q = min(full_rank, max(rank + 8, rank))
    try:
        u, s, v = torch.pca_lowrank(delta_t, q=q, center=False, niter=2)
        left = u[:, :rank].contiguous()
        right = (torch.diag(s[:rank]) @ v[:, :rank].T).contiguous()
    except RuntimeError:
        u, s, vh = torch.linalg.svd(delta_t, full_matrices=False)
        left = u[:, :rank].contiguous()
        right = (torch.diag(s[:rank]) @ vh[:rank, :]).contiguous()
    return left, right, left @ right


def smooth_scale(mas_params, layer, module, kind):
    # MAS saves only scale vectors, not three full modality-specific model copies.
    key = f"{module}.{kind}_smooth_scale"
    return mas_params[layer][key].float().view(1, -1).clamp(min=1e-5)


def mean_metric(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) not in ("", None)]
    return sum(values) / len(values) if values else float("nan")


def metric_rows(rows, method):
    return [
        row
        for row in rows
        if row["method"] == method and row["metric_kind"] == "deployed_weight_to_modality_float"
    ]


def save_plot(summary_rows, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_dir / "plot_skipped.txt").write_text(
            f"matplotlib import failed, wrote SVG fallback: {exc}\n", encoding="utf-8"
        )
        return save_svg_plot(summary_rows, out_dir)

    labels = [row["method"] for row in summary_rows]
    values = [float(row["mean_error"]) for row in summary_rows]
    colors = ["#4C78A8", "#59A14F", "#F28E2B"]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    bars = ax.bar(labels, values, color=colors[: len(labels)])
    ax.set_ylabel("Mean error (lower is better)")
    ax.set_title("MASQuant small reproduction: weight / CMC reconstruction proxy")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    plot_path = out_dir / "w4a8_three_method_error_comparison.png"
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def save_svg_plot(summary_rows, out_dir):
    labels = [row["method"] for row in summary_rows]
    values = [float(row["mean_error"]) for row in summary_rows]
    colors = ["#4C78A8", "#59A14F", "#F28E2B"]
    width, height = 900, 520
    left, right, top, bottom = 90, 40, 70, 100
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_value = max(values) if values else 1.0
    max_value = max_value * 1.2 if max_value > 0 else 1.0
    bar_w = chart_w / (len(values) * 2 + 1)

    def x_for(i):
        return left + bar_w * (1 + i * 2)

    def y_for(value):
        return top + chart_h - (value / max_value) * chart_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="450" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="700">MASQuant Small Reproduction: Error Comparison</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{width - right}" y2="{top + chart_h}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#333" stroke-width="1"/>',
        f'<text x="24" y="{top + chart_h / 2}" transform="rotate(-90 24 {top + chart_h / 2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">Mean error (lower is better)</text>',
    ]

    for tick in range(6):
        value = max_value * tick / 5
        y = y_for(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12">{value:.3f}</text>'
        )

    for i, (label, value) in enumerate(zip(labels, values)):
        x = x_for(i)
        y = y_for(value)
        h = top + chart_h - y
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{colors[i % len(colors)]}"/>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{value:.4f}</text>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{top + chart_h + 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">{label}</text>'
        )

    parts.append("</svg>")
    plot_path = out_dir / "w4a8_three_method_error_comparison.svg"
    plot_path.write_text("\n".join(parts), encoding="utf-8")
    return plot_path


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_map = load_index(model_dir)
    mas_params = torch.load(args.mas_parameters, map_location="cpu")
    act_scales = torch.load(args.act_scales, map_location="cpu")

    rows = []
    cmc_adapters = {"vision": {}, "audio": {}}

    for layer in range(args.max_layers):
        for module in args.modules:
            tensor_name = f"thinker.model.layers.{layer}.{module}.weight"
            if tensor_name not in weight_map:
                print(f"skip missing weight: {tensor_name}")
                continue

            weight = load_weight(model_dir, weight_map, tensor_name)
            base_name = f"model.layers.{layer}.{module}"
            if f"{base_name}.all_in_one_scale" not in act_scales:
                print(f"skip missing activation scale: {base_name}")
                continue

            all_scale = smooth_scale(mas_params, layer, module, "all_in_one")
            text_scale = smooth_scale(mas_params, layer, module, "text")
            baseline_target = weight * all_scale
            baseline_quant = fake_quant_symmetric_with_zero_point(baseline_target, args.bits)

            text_target = weight * text_scale
            text_quant = fake_quant_symmetric_with_zero_point(text_target, args.bits)

            for modality in MODALITIES:
                modality_scale = smooth_scale(mas_params, layer, module, modality)
                target = weight * modality_scale
                quant = fake_quant_symmetric_with_zero_point(target, args.bits)

                rows.append(
                    {
                        "method": "smoothquant_unified",
                        "layer": layer,
                        "module": module,
                        "modality": modality,
                        "metric_kind": "deployed_weight_to_modality_float",
                        "mse": mse(target, baseline_quant),
                        "sqnr_db": sqnr_db(target, baseline_quant),
                        "relative_fro_error": rel_fro_error(target, baseline_quant),
                        "rank": "",
                        "weight_shape": "x".join(str(x) for x in weight.shape),
                    }
                )
                rows.append(
                    {
                        "method": "mas_split_scales",
                        "layer": layer,
                        "module": module,
                        "modality": modality,
                        "metric_kind": "deployed_weight_to_modality_float",
                        "mse": mse(target, quant),
                        "sqnr_db": sqnr_db(target, quant),
                        "relative_fro_error": rel_fro_error(target, quant),
                        "rank": "",
                        "weight_shape": "x".join(str(x) for x in weight.shape),
                    }
                )

                if modality == "text":
                    # Text is the CMC base branch; extra adapters are only needed for vision/audio.
                    rows.append(
                        {
                            "method": "simplified_cmc",
                            "layer": layer,
                            "module": module,
                            "modality": modality,
                            "metric_kind": "deployed_weight_to_modality_float",
                            "mse": mse(target, text_quant),
                            "sqnr_db": sqnr_db(target, text_quant),
                            "relative_fro_error": rel_fro_error(target, text_quant),
                            "rank": 0,
                            "weight_shape": "x".join(str(x) for x in weight.shape),
                        }
                    )
                    continue

                delta_t = (quant - text_quant).T.contiguous()
                # Simplified CMC: approximate the modality delta with a rank-k adapter.
                left, right, delta_hat_t = low_rank_delta(delta_t, args.rank)
                cmc_adapters[modality][base_name] = {
                    "L": left.to(torch.float32).cpu(),
                    "R": right.to(torch.float32).cpu(),
                    "rank": args.rank,
                }
                reconstructed = text_quant.T + delta_hat_t
                target_t = target.T
                rows.append(
                    {
                        "method": "simplified_cmc",
                        "layer": layer,
                        "module": module,
                        "modality": modality,
                        "metric_kind": "deployed_weight_to_modality_float",
                        "mse": mse(target_t, reconstructed),
                        "sqnr_db": sqnr_db(target_t, reconstructed),
                        "relative_fro_error": rel_fro_error(target_t, reconstructed),
                        "rank": args.rank,
                        "weight_shape": "x".join(str(x) for x in weight.shape),
                    }
                )
                rows.append(
                    {
                        "method": "simplified_cmc_delta_residual",
                        "layer": layer,
                        "module": module,
                        "modality": modality,
                        "metric_kind": "low_rank_delta_reconstruction",
                        "mse": mse(quant.T, text_quant.T + delta_hat_t),
                        "sqnr_db": sqnr_db(quant.T, text_quant.T + delta_hat_t),
                        "relative_fro_error": rel_fro_error(quant.T, text_quant.T + delta_hat_t),
                        "rank": args.rank,
                        "weight_shape": "x".join(str(x) for x in weight.shape),
                    }
                )

            del weight

    detail_csv = out_dir / "w4a8_layer_module_weight_proxy_metrics.csv"
    fieldnames = [
        "method",
        "layer",
        "module",
        "modality",
        "metric_kind",
        "mse",
        "sqnr_db",
        "relative_fro_error",
        "rank",
        "weight_shape",
    ]
    with open(detail_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = [
        {
            "method": "smoothquant_unified",
            "mean_error": mean_metric(metric_rows(rows, "smoothquant_unified"), "relative_fro_error"),
            "mean_sqnr_db": mean_metric(metric_rows(rows, "smoothquant_unified"), "sqnr_db"),
            "description": "one shared smoothing scale for text/audio/vision",
        },
        {
            "method": "mas_split_scales",
            "mean_error": mean_metric(metric_rows(rows, "mas_split_scales"), "relative_fro_error"),
            "mean_sqnr_db": mean_metric(metric_rows(rows, "mas_split_scales"), "sqnr_db"),
            "description": "modality-specific text/vision/audio smoothing scales",
        },
        {
            "method": "simplified_cmc",
            "mean_error": mean_metric(metric_rows(rows, "simplified_cmc"), "relative_fro_error"),
            "mean_sqnr_db": mean_metric(metric_rows(rows, "simplified_cmc"), "sqnr_db"),
            "description": "text base quantized weight plus rank-k low-rank modality delta",
        },
    ]
    summary_csv = out_dir / "w4a8_three_method_summary.csv"
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "mean_error", "mean_sqnr_db", "description"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    adapter_path = out_dir / f"simplified_cmc_rank{args.rank}_adapters.pt"
    torch.save(cmc_adapters, adapter_path)

    manifest = {
        "model_dir": str(model_dir),
        "act_scales": args.act_scales,
        "mas_parameters": args.mas_parameters,
        "max_layers": args.max_layers,
        "modules": args.modules,
        "rank": args.rank,
        "bits": args.bits,
        "outputs": {
            "detail_csv": str(detail_csv),
            "summary_csv": str(summary_csv),
            "adapter_path": str(adapter_path),
        },
        "note": (
            "This is a small low-memory reproduction proxy. It compares weight "
            "quantization and CMC low-rank delta reconstruction on selected thinker "
            "layers, not the full OmniBench task accuracy."
        ),
    }
    manifest_path = out_dir / "w4a8_experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_path = save_plot(summary_rows, out_dir)
    if plot_path is not None:
        manifest["outputs"]["plot"] = str(plot_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"detail_csv={detail_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"adapter_path={adapter_path}")
    print(f"manifest={manifest_path}")
    if plot_path is not None:
        print(f"plot={plot_path}")
    print("summary")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
