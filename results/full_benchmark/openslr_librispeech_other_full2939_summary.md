# OpenSLR LibriSpeech Other Full Benchmark

This is the retained public result from the completed full benchmark run.

## Result

| Field | Value |
| --- | --- |
| Model | `Qwen2.5-Omni-3B` |
| Method | MASQuant with CMC adapters |
| Quantization | W4A8 |
| Benchmark task | `openslr_librispeech_other` |
| Dataset split | OpenSLR LibriSpeech test-other |
| Completed samples | 2939 / 2939 |
| Metric | WER, lower is better |
| WER | `2.878264524387215` |

## Run Configuration

| Field | Value |
| --- | --- |
| Weight bits | 4 |
| Activation bits | 8 |
| Low-rank ratio | 0.05 |
| CMC adapters | enabled |
| Calibration samples for inference command | 1 |
| Attention implementation | `sdpa` |
| Limit | full split |

## Runtime

| Field | Value |
| --- | --- |
| Started | 2026-06-11 21:38:43 |
| Finished | 2026-06-17 15:40:47 |
| Generation progress | 2939 / 2939 |
| Generation elapsed | 138:01:31 |
| Postprocessing progress | 2939 / 2939 |
| Postprocessing elapsed | 00:19 |

Raw logs and binary tensors are intentionally not tracked because they contain machine-local paths and large model artifacts. The machine-readable metric is stored in `openslr_librispeech_other_full2939_metrics.json`.
