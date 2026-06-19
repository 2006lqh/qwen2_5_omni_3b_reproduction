# Project Files and Key Code

## Directory Roles

| Path | Role |
| --- | --- |
| `configs/` | Reproduction configuration templates. |
| `scripts/` | Environment checks and benchmark entry points. |
| `benchmarks/lmms_eval_tasks/` | Local `lmms_eval` task wrappers. |
| `third_party/EfficientAI/masquant/` | Upstream MASQuant source after applying the local compatibility patch. |
| `patches/` | Patch file required to reproduce the local MASQuant runtime changes. |
| `outputs/` | Local generated tensors, parameters, logs, and benchmark outputs. This directory is not tracked by Git. |
| `results/full_benchmark/` | Curated full-benchmark summaries retained for public reporting. |

Model weights and generated MASQuant tensors are expected outside the repository and should be supplied through environment variables.

## MAS Code Locations

| File | Focus |
| --- | --- |
| `third_party/EfficientAI/masquant/generate_act_scale_shift.py` | PyTorch forward hooks collect modality-specific activation ranges. |
| `third_party/EfficientAI/masquant/quantize/masquant.py` | Computes unified and modality-specific smoothing scales from activation and weight ranges. |
| `third_party/EfficientAI/masquant/quantize/int_linear.py` | Implements `merged_scales` for unified SmoothQuant and `split_scales` for MAS. |

## CMC Code Locations

| File | Focus |
| --- | --- |
| `third_party/EfficientAI/masquant/infer_mas.py` | MAS/CMC inference entry point. |
| `third_party/EfficientAI/masquant/quantize/svd_utils.py` | Computes low-rank CMC adapter factors. |
| `third_party/EfficientAI/masquant/quantize/infer_quant.py` | Loads low-rank adapters during inference. |
| `scripts/run_openslr_librispeech_other_full2939_w4a8.sh` | Full OpenSLR LibriSpeech benchmark entry point. |

## Retained Result

The retained public benchmark result is stored in:

- `results/full_benchmark/openslr_librispeech_other_full2939_summary.md`
- `results/full_benchmark/openslr_librispeech_other_full2939_metrics.json`
- `results/full_benchmark/openslr_librispeech_other_full2939_metrics.csv`

The completed OpenSLR LibriSpeech `test-other` run evaluated 2939 / 2939 samples and produced WER `2.878264524387215`.
