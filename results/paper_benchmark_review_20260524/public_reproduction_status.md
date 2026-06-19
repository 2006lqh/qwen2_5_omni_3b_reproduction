# Public Reproduction Status

## W4A8 Proxy Metrics

| Method | mean_error | mean_sqnr_db |
| --- | ---: | ---: |
| unified SmoothQuant | 0.370504 | 9.027116 |
| MAS split scales | 0.198716 | 14.131121 |
| simplified CMC proxy | 0.300873 | 10.984752 |

MAS split-scale smoothing reduced reconstruction error and improved SQNR compared with unified SmoothQuant in the local proxy evaluation.

## Benchmark Smoke Status

| Benchmark | Status | Metric |
| --- | --- | --- |
| LibriSpeech test-other | completed for `limit=1` | WER `133.33333333333331` |
| LibriSpeech test-other full | not run | full split estimated too slow for the local setup |
| MMMU validation | interrupted | dataset parquet download failed before inference |
| OmniBench | interrupted | dataset parquet download failed before inference |
| WenetSpeech test-net | blocked | public mirror split unavailable; gated source access required |

## Official CMC Status

The official full CMC path did not complete on the available 6GB GPU. The run failed while moving the full model to CUDA, before white matrix and low-rank adapter generation. No official CMC SQNR CSV was produced.

## Interpretation

These results support the local reproduction workflow and proxy-level MAS trend. They should not be reported as complete paper benchmark reproduction results.
