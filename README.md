# Qwen2.5-Omni-3B MASQuant Reproduction

This repository contains a focused reproduction package for the `Qwen2.5-Omni-3B` W4A8 path from `MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models`.

The retained public result is the completed OpenSLR LibriSpeech `test-other` benchmark. Model weights, Hugging Face caches, calibration tensors, low-rank adapter tensors, and raw runtime logs are intentionally excluded from Git.

## Scope

- Model: `Qwen2.5-Omni-3B`
- Method: MASQuant with CMC adapters
- Quantization: W4A8
- Retained benchmark: OpenSLR LibriSpeech `test-other`
- Metric: word error rate (WER), lower is better
- Upstream code: EfficientAI MASQuant, tracked as a submodule with a local compatibility patch

## Repository Layout

```text
benchmarks/lmms_paper_tasks/     Local lmms-eval task wrappers
configs/                         Reproduction configuration templates
patches/                         Patch applied to the upstream EfficientAI checkout
scripts/                         Environment checks and reproduction entry points
third_party/EfficientAI/         Upstream EfficientAI repository, tracked as a submodule
results/full_benchmark/          Curated public full-benchmark result summaries
requirements*.txt                Python dependency notes
sanity_check.py                  Minimal local model loading check
PAPER_REPRODUCTION.md            Reproduction status and result details
```

## Setup

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
cd qwen2_5_omni_3b_reproduction
```

If the submodule was not initialized during clone:

```bash
git submodule update --init --recursive
```

Apply the local MASQuant compatibility patch:

```bash
cd third_party/EfficientAI
git apply ../../patches/efficientai_masquant_qwen2_5_omni_w4a8.patch
cd ../..
```

Prepare the Python environment with the package versions listed in `requirements.txt` and `requirements-paper.txt`. The scripts assume local model and MASQuant artifact paths supplied outside this repository:

```bash
export PROJECT_ROOT="$(pwd)"
export MODEL_DIR="/path/to/Qwen2.5-Omni-3B"
export MAS_PARAMS="/path/to/mas_parameters.pth"
export LOW_RANK_ADAPTERS="/path/to/low_rank_adapters.pt"
export MASQUANT_CONDA_ENV="llm_quant"
```

Model files and generated tensors are not redistributed in this repository.

## Reproduction Commands

Run structure and dependency checks:

```bash
bash scripts/check_paper_repro_ready.sh
python scripts/full_repro_config_check.py
```

Run the retained full benchmark:

```bash
bash scripts/run_openslr_librispeech_full_benchmark_w4a8.sh
```

The benchmark script expects `MODEL_DIR`, `MAS_PARAMS`, and `LOW_RANK_ADAPTERS` to point to local artifacts generated outside Git.

## Current Result

| Benchmark | Samples | Quantization | Method | WER |
| --- | ---: | --- | --- | ---: |
| OpenSLR LibriSpeech `test-other` | 2939 / 2939 | W4A8 | MASQuant with CMC adapters | 2.878264524387215 |

Result files:

- `results/full_benchmark/openslr_librispeech_other_full2939_summary.md`
- `results/full_benchmark/openslr_librispeech_other_full2939_metrics.json`
- `results/full_benchmark/openslr_librispeech_other_full2939_metrics.csv`

## Reproducibility Notes

The retained metric is a sanitized summary extracted from the completed benchmark logs. Raw logs are excluded because they contain machine-local paths, and binary artifacts are excluded because they are large generated tensors. The public files in `results/full_benchmark/` are the only tracked result artifacts.
