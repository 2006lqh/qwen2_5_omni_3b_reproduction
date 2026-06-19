# MASQuant Qwen2.5-Omni-3B W4A8 Reproduction Report

Paper: `MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models`

This report documents the reproduced `Qwen2.5-Omni-3B` W4A8 audio-text benchmark path. The retained public result is the completed OpenSLR LibriSpeech `test-other` evaluation with MASQuant and CMC adapters.

## 1. 方法原理

MASQuant targets multimodal large language models whose activation distributions differ across text, audio, image, and video inputs. A single smoothing scale can overfit the dominant modality and hurt another modality. MASQuant therefore uses modality-aware smoothing: it keeps the quantization target unified at inference time while estimating smoothing behavior with modality-specific activation statistics.

The reproduced setting is W4A8:

| Component | Setting |
| --- | --- |
| Weight quantization | 4-bit per-channel |
| Activation quantization | 8-bit per-token |
| Target module | Qwen2.5-Omni thinker module |
| Calibration data type | text-audio-vision |
| Inference mode | MAS split scales with CMC low-rank adapters |

The high-level computation is:

1. Collect activation ranges on multimodal calibration samples.
2. Combine activation ranges and weight ranges to compute smoothing scales.
3. Use MAS split scales to reduce modality imbalance during quantized inference.
4. Use CMC low-rank adapters to compensate quantization error introduced by using a text-base reference and modality-specific deltas.

CMC is retained as a low-rank correction. In this reproduction, the public benchmark run uses precomputed MAS parameters and CMC low-rank adapters stored outside Git because they are binary tensor artifacts.

## 2. 实验流程

The reproduction follows a source-first workflow so that the public repository remains runnable while large assets stay local.

| Step | Action | Repository entry point |
| --- | --- | --- |
| 1 | Clone repository with submodules | `git clone --recurse-submodules` |
| 2 | Apply EfficientAI compatibility patch | `patches/efficientai_masquant_qwen2_5_omni_w4a8.patch` |
| 3 | Prepare local model and tensor artifacts | `local_artifacts/` layout described in `docs/local_artifact_layout.md` |
| 4 | Check environment and paths | `scripts/check_paper_repro_ready.sh`, `scripts/check_full_benchmark_config.py` |
| 5 | Run full LibriSpeech benchmark | `scripts/run_openslr_librispeech_other_full2939_w4a8.sh` |
| 6 | Extract public metric summaries | `results/full_benchmark/` |

The full retained benchmark used:

| Field | Value |
| --- | --- |
| Model | `Qwen2.5-Omni-3B` |
| Benchmark task | `openslr_librispeech_other` |
| Dataset split | OpenSLR LibriSpeech `test-other` |
| Samples | 2939 / 2939 |
| Weight bits | 4 |
| Activation bits | 8 |
| Low-rank ratio | 0.05 |
| CMC adapters | enabled |
| Attention implementation | `sdpa` |
| Decoding | deterministic generation settings from the lmms-eval task wrapper |

## 3. 代码细节

The repository keeps the runnable public code separate from private local artifacts.

`scripts/run_openslr_librispeech_other_full2939_w4a8.sh` is the main benchmark entry point. It reads `MODEL_DIR`, `MAS_PARAMS`, and `LOW_RANK_ADAPTERS`, creates a local run directory under `local_artifacts/raw_results/`, and invokes `infer_mas.py` with W4A8, MAS parameters, CMC adapters, and `--limit_multimodal -1` for the full split.

`benchmarks/lmms_eval_tasks/` contains local lmms-eval task wrappers. The LibriSpeech wrapper maps dataset samples to audio prompts, uses deterministic generation, and aggregates WER through the lmms-eval LibriSpeech utilities. The task wrapper makes the benchmark invocation explicit and reproducible from the repository.

`patches/efficientai_masquant_qwen2_5_omni_w4a8.patch` records the required EfficientAI runtime changes. The patch switches unsupported `flash_attention_2` paths to `sdpa`, avoids unsafe forced CUDA transfers when device maps are already assigned, supports local lmms-eval task loading through `TaskManager`, writes multimodal result JSON when requested, and adds Qwen2.5-Omni audio/device compatibility fixes.

`configs/qwen2_5_omni_3b_w4a8_full_benchmark.yaml` documents the model, quantization, artifact, and benchmark settings. `results/full_benchmark/openslr_librispeech_other_full2939_metrics.json` is the machine-readable public metric file.

## 4. 结果对比

The reproduced result is compared with the paper's Table 2 numbers for `Qwen2.5-Omni-3B`, W4A8, LibriSpeech WER. Lower is better.

| Method | Paper WER | Reproduced WER | Difference vs paper |
| --- | ---: | ---: | ---: |
| Dense FP16 | 3.9 | N/A | N/A |
| RTN W4A8 | 109.7 | N/A | N/A |
| SmoothQuant W4A8 | 77.4 | N/A | N/A |
| MBQ W4A8 | 9.5 | N/A | N/A |
| MASQuant W4A8 | 3.6 | 2.878264524387215 | -0.721735475612785 |

The reproduced run completed all 2939 samples and produced WER `2.878264524387215`. This is consistent with the paper's main conclusion: modality-aware smoothing avoids the severe WER degradation observed for RTN and SmoothQuant under W4A8 and keeps the quantized model near the dense baseline on LibriSpeech.

## 5. 误差分析

The reproduced WER is lower than the paper's reported MASQuant W4A8 LibriSpeech WER by about `0.7217`. This difference should be interpreted as a reproduction variance rather than a claim of a strictly better method.

Likely sources of difference:

- Dataset and task wrapper version: the run uses the available OpenSLR/lmms-eval task path and may not match the exact preprocessing snapshot used by the paper.
- Runtime library versions: Transformers, lmms-eval, dataset loading, audio resampling, and Qwen2.5-Omni processor behavior can change across versions.
- Decoding behavior: deterministic generation is used, but tokenizer, processor, and generation defaults may still differ from the paper environment.
- Patch-level compatibility changes: the reproduction uses `sdpa` and local device movement fixes to make Qwen2.5-Omni run reliably in the available environment.
- Artifact handling: raw logs and tensor files are retained locally but not uploaded; only sanitized public metrics are tracked in Git.

The result is therefore treated as a successful benchmark-level reproduction of the MASQuant W4A8 trend and workflow, not as an exact bitwise recreation of the original paper environment.

## 6. 可复现性与限制

The public repository contains:

- runnable source wrappers and scripts;
- configuration templates;
- the EfficientAI compatibility patch;
- sanitized metric summaries;
- documentation for the local artifact layout.

The public repository does not contain:

- model checkpoints;
- Hugging Face caches;
- `.pt` or `.pth` tensor artifacts;
- raw benchmark logs;
- machine-local absolute paths.

This keeps the submission shareable and professional while preserving the local artifacts needed to audit the completed run.

## 7. 公开结果文件

```text
results/full_benchmark/openslr_librispeech_other_full2939_summary.md
results/full_benchmark/openslr_librispeech_other_full2939_metrics.json
results/full_benchmark/openslr_librispeech_other_full2939_metrics.csv
```
