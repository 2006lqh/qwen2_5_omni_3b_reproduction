# Local Artifact Layout

Local artifacts are stored beside the repository, not inside Git.

```text
local_artifacts/
  models/
  hf_cache/
  model_artifacts/
    qwen2_5_omni_3b_w4a8/
      mas_parameters/
      cmc_adapters/
  raw_results/
    qwen2_5_omni_3b_w4a8/
      full_benchmark/
        openslr_librispeech_other_full2939/
          logs/
          summaries/
  archived_local_runs/
    small_validation_proxy_smoke_20260518_20260611/
    repo_outputs_pre_full_benchmark_20260518_20260525/
```

The public metric files in `results/full_benchmark/` are sanitized summaries extracted from these local artifacts. Raw logs are retained locally for auditability but are not uploaded because they include machine-local paths and verbose runtime output.

## Expected Environment Variables

```bash
export MODEL_DIR="$PROJECT_ROOT/../local_artifacts/models/Qwen2.5-Omni-3B"
export MAS_PARAMS="$PROJECT_ROOT/../local_artifacts/model_artifacts/qwen2_5_omni_3b_w4a8/mas_parameters/mas_parameters.pth"
export LOW_RANK_ADAPTERS="$PROJECT_ROOT/../local_artifacts/model_artifacts/qwen2_5_omni_3b_w4a8/cmc_adapters/cmc_full128_low_rank_adapters.pt"
```
