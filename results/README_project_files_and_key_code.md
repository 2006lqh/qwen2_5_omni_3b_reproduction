# Project Files and Key Code

## Directory Roles

| Path | Role |
| --- | --- |
| `configs/` | Reproduction configuration templates. |
| `scripts/` | Environment checks, model checks, proxy experiments, benchmark runs, and result collection. |
| `benchmarks/lmms_paper_tasks/` | Local `lmms_eval` task wrappers used for paper benchmark smoke runs. |
| `third_party/EfficientAI/masquant/` | Upstream MASQuant source after applying the local compatibility patch. |
| `patches/` | Patch file required to reproduce the local MASQuant runtime changes. |
| `outputs/` | Local generated tensors, parameters, logs, and benchmark outputs. This directory is not tracked by Git. |
| `results/` | Curated public CSV/JSON/Markdown summaries. |

Model weights are expected outside the repository and should be supplied through `MODEL_DIR`.

## MAS Code Locations

| File | Focus |
| --- | --- |
| `third_party/EfficientAI/masquant/generate_act_scale_shift.py` | PyTorch forward hooks collect modality-specific activation ranges. |
| `third_party/EfficientAI/masquant/quantize/masquant.py` | Computes unified and modality-specific smoothing scales from activation and weight ranges. |
| `third_party/EfficientAI/masquant/quantize/int_linear.py` | Implements `merged_scales` for unified SmoothQuant and `split_scales` for MAS. |

## CMC Code Locations

| File | Focus |
| --- | --- |
| `third_party/EfficientAI/masquant/infer_mas.py` | Official MAS/CMC inference entry point. |
| `third_party/EfficientAI/masquant/quantize/svd_utils.py` | Computes `qsmW - qstW` and performs low-rank decomposition. |
| `third_party/EfficientAI/masquant/quantize/infer_quant.py` | Loads low-rank adapters during inference. |
| `scripts/run_exp3_weight_proxy.py` | Local simplified CMC proxy used for 6GB-GPU validation. |

## Validation Summary

The three-run proxy validation is stored in:

- `results/exp3_validation/masquant_w4a8_three_run_comparison.csv`
- `results/exp3_validation/masquant_w4a8_three_run_report.md`
- `results/exp3_validation/masquant_w4a8_three_run_mean_error.svg`

The three runs were identical under the tested setting. MAS split-scale smoothing had the lowest reconstruction error; simplified CMC improved over unified SmoothQuant but did not match MAS-only in the proxy metric.
