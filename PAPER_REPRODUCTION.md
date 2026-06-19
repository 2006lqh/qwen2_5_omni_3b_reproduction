# MASQuant Reproduction Status

Paper: `MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models`

This project reproduces the `Qwen2.5-Omni-3B` subset of MASQuant under a local W4A8 setting.

## Target

- Model: `Qwen2.5-Omni-3B`
- Quantization: W4A8
- Modalities: text, audio, vision, vision-audio-text
- Methods: unified SmoothQuant baseline, MAS split-scale smoothing, CMC
- Benchmarks: LibriSpeech, WenetSpeech, MMMU, OmniBench

## Completed Work

1. Environment and model loading checks were prepared.
2. Local multimodal calibration data was prepared.
3. Activation scales were generated.
4. Unified SmoothQuant proxy evaluation was completed.
5. MAS split-scale proxy evaluation was completed.
6. Simplified CMC proxy evaluation was completed.
7. Three-run proxy validation was completed.
8. Local lmms-eval wrappers were added for the paper benchmark targets.
9. Benchmark data access was checked.
10. The official CMC loading path was tested and its failure mode was recorded.

## Main Proxy Results

| Method | mean_error | mean_sqnr_db | Notes |
| --- | ---: | ---: | --- |
| unified SmoothQuant | 0.370504 | 9.027116 | shared smoothing scale across modalities |
| MAS split scales | 0.198716 | 14.131121 | modality-specific smoothing scales |
| simplified CMC proxy | 0.300873 | 10.984752 | low-rank compensation proxy |

Under the proxy metric, MAS reduces reconstruction error and improves SQNR compared with unified SmoothQuant. The simplified CMC proxy is only a local weight-reconstruction proxy and is not equivalent to the full official CMC algorithm from the paper.

## Benchmark Status

| Benchmark | Status | Notes |
| --- | --- | --- |
| LibriSpeech test-other | Partially completed | `limit=1` produced WER `133.33333333333331` |
| LibriSpeech full | Not run | single-sample runtime was about 25 minutes |
| MMMU validation | No final score | first sample was readable; smoke run was interrupted during parquet download |
| OmniBench | No final score | first sample was readable; smoke run was interrupted during parquet download |
| WenetSpeech test-net | Blocked | public mirror does not expose the required `test_net` split; gated source access is required |

## Official Full CMC Status

The official full CMC path did not load successfully on the available 6GB GPU. Both the `--eval_sqnr` and non-`--eval_sqnr` runs failed at model transfer:

```python
llm.model.to("cuda")
```

The error was:

```text
RuntimeError: CUDA driver error: device not ready
```

The failure occurred before white matrix and low-rank adapter computation, so no official CMC artifacts were produced.

## Public Result Files

```text
results/full_reproduction_w4a8_20260524/full_repro_w4a8_method_summary.csv
results/exp3_validation/masquant_w4a8_three_run_report.md
results/full_reproduction_w4a8_20260524/official_cmc_gpu_load_attempt_20260525.md
results/paper_benchmark_review_20260524/public_reproduction_status.md
```

## Conclusion

The local workflow reproduces the MASQuant pipeline and confirms the MAS trend on proxy metrics, but it is not a complete paper-level reproduction. Remaining blockers are GPU memory for official CMC and benchmark data availability/download reliability.
