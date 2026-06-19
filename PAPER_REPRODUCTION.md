# MASQuant Reproduction Status

Paper: `MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models`

This project packages the `Qwen2.5-Omni-3B` W4A8 reproduction workflow and publishes the completed OpenSLR LibriSpeech `test-other` benchmark result.

## Target

- Model: `Qwen2.5-Omni-3B`
- Quantization: W4A8
- Method: MASQuant with CMC adapters
- Benchmark task: `openslr_librispeech_other`
- Dataset split: OpenSLR LibriSpeech `test-other`
- Metric: WER, lower is better

## Completed Work

1. Prepared the local MASQuant runtime through the EfficientAI submodule and compatibility patch.
2. Prepared local multimodal calibration and MAS parameter artifacts outside Git.
3. Ran the full OpenSLR LibriSpeech `test-other` benchmark with MASQuant W4A8 and CMC adapters.
4. Completed 2939 / 2939 benchmark samples.
5. Extracted a sanitized public result summary from the completed benchmark logs.

## Full Benchmark Result

| Field | Value |
| --- | --- |
| Benchmark | OpenSLR LibriSpeech `test-other` |
| Task name | `openslr_librispeech_other` |
| Model | `Qwen2.5-Omni-3B` |
| Method | MASQuant with CMC adapters |
| Quantization | W4A8 |
| Completed samples | 2939 / 2939 |
| WER | `2.878264524387215` |

## Run Metadata

| Field | Value |
| --- | --- |
| Started | 2026-06-11 21:38:43 |
| Finished | 2026-06-17 15:40:47 |
| Generation elapsed | 138:01:31 |
| Postprocessing elapsed | 00:19 |
| Weight bits | 4 |
| Activation bits | 8 |
| Low-rank ratio | 0.05 |
| CMC adapters | enabled |
| Attention implementation | `sdpa` |

## Public Result Files

```text
results/full_benchmark/openslr_librispeech_other_full2939_summary.md
results/full_benchmark/openslr_librispeech_other_full2939_metrics.json
results/full_benchmark/openslr_librispeech_other_full2939_metrics.csv
```

## Artifact Policy

The repository tracks source, configuration, compatibility patches, and curated public result summaries. It does not track model checkpoints, `.pt` or `.pth` tensors, low-rank adapter files, Hugging Face caches, or raw logs.
