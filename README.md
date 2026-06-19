# Qwen2.5-Omni-3B MASQuant Reproduction

This repository records a focused reproduction of the MASQuant workflow for `Qwen2.5-Omni-3B`.

Paper: `MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models`

## Scope

The project reproduces the Qwen2.5-Omni-3B W4A8 path:

- multimodal calibration data preparation
- activation scale collection
- unified SmoothQuant proxy baseline
- modality-aware smoothing (MAS)
- simplified CMC weight-reconstruction proxy
- official CMC loading attempt
- paper benchmark smoke checks for LibriSpeech, WenetSpeech, MMMU, and OmniBench

The full model weights, Hugging Face caches, intermediate tensors, `.pt/.pth` artifacts, and raw logs are intentionally excluded from Git.

## Repository Layout

```text
benchmarks/lmms_paper_tasks/     Local lmms-eval task wrappers
configs/                         Reproduction configuration templates
patches/                         Patch applied to the upstream EfficientAI checkout
scripts/                         Environment checks, reproduction runs, benchmark runs, and summaries
third_party/EfficientAI/         Upstream EfficientAI repository, tracked as a submodule
results/                         Curated public result summaries only
requirements*.txt                Python dependency notes
sanity_check.py                  Minimal local model loading check
PAPER_REPRODUCTION.md            Reproduction status and limitations
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

Prepare the Python environment with the package versions listed in `requirements.txt` and `requirements-paper.txt`. The scripts assume a local `Qwen2.5-Omni-3B` checkpoint supplied outside this repository:

```bash
export PROJECT_ROOT="$(pwd)"
export MODEL_DIR="/path/to/Qwen2.5-Omni-3B"
export MASQUANT_CONDA_ENV="llm_quant"
```

Model files are not redistributed in this repository.

## Reproduction Commands

Run structure and dependency checks:

```bash
bash scripts/check_paper_repro_ready.sh
python scripts/full_repro_config_check.py
```

Run the W4A8 proxy reproduction:

```bash
bash scripts/run_full_reproduction_w4a8.sh
```

Run paper benchmark smoke checks:

```bash
bash scripts/run_full_paper_benchmarks_w4a8.sh smoke
```

Run the official CMC loading path:

```bash
bash scripts/run_official_cmc_full_w4a8.sh
```

## Current Results

W4A8 proxy metrics:

| Method | mean_error | mean_sqnr_db | Description |
| --- | ---: | ---: | --- |
| unified SmoothQuant | 0.370504 | 9.027116 | one shared smoothing scale for text/audio/vision |
| MAS split scales | 0.198716 | 14.131121 | modality-specific smoothing scales |
| simplified CMC proxy | 0.300873 | 10.984752 | text-base quantized weight plus low-rank modality delta |

Benchmark status:

- LibriSpeech `test-other` smoke (`limit=1`) produced a real WER of `133.33333333333331`.
- Full LibriSpeech was not run because a single sample took about 25 minutes in this setup.
- MMMU and OmniBench smoke runs reached dataset download but were interrupted by network download errors.
- WenetSpeech `test_net` remains blocked by public mirror split availability and gated-source access.
- Official full CMC did not complete on a 6GB RTX 4050 Laptop GPU; loading failed before white matrix or low-rank adapter generation.

See `PAPER_REPRODUCTION.md` and `results/paper_benchmark_review_20260524/public_reproduction_status.md` for the detailed status.

## Reproducibility Notes

The current results demonstrate the local reproduction workflow and the MAS trend under proxy metrics. They should not be presented as complete paper-level benchmark reproduction. The main limitations are local GPU memory, unavailable/gated benchmark data, and interrupted large dataset downloads.
