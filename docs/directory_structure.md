# Directory Structure

This repository separates public reproduction materials from local runtime artifacts.

```text
qwen2_5_omni_3b_reproduction/
  README.md
  CITATION.cff
  LICENSE
  docs/
    directory_structure.md
    local_artifact_layout.md
  benchmarks/
    lmms_eval_tasks/
  configs/
    benchmark_targets.yaml
    local_environment.template.yaml
    masquant_qwen2_5_omni_3b.yaml
    qwen2_5_omni_3b_w4a8_full_benchmark.yaml
  patches/
  results/
    README.md
    full_benchmark/
  scripts/
  third_party/
```

Tracked files are limited to source code, configuration templates, compatibility patches, directory documentation, and sanitized result summaries. Generated logs, model weights, caches, tensor artifacts, and submission-package documents stay outside Git.

## Naming

- Directories and files use `snake_case`.
- Benchmark result names include task name, sample count, and metric format.
- Local artifact paths mirror the model and benchmark names used in public summaries.
