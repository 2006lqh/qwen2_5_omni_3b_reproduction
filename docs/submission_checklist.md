# Submission Checklist

This checklist maps the repository to the project requirements.

## 完整复现报告

| Requirement | Status | Location |
| --- | --- | --- |
| 方法原理 | Done | `docs/reproduction_report.md` section 1 |
| 实验流程 | Done | `docs/reproduction_report.md` section 2 |
| 代码细节 | Done | `docs/reproduction_report.md` section 3 |
| 结果对比 | Done | `docs/reproduction_report.md` section 4 |
| 误差分析 | Done | `docs/reproduction_report.md` section 5 |

## 可运行源码

| Requirement | Status | Location |
| --- | --- | --- |
| Source code and patch | Done | `scripts/`, `benchmarks/lmms_eval_tasks/`, `patches/` |
| Environment and path checks | Done | `scripts/check_paper_repro_ready.sh`, `scripts/check_full_benchmark_config.py` |
| Full benchmark entry point | Done | `scripts/run_openslr_librispeech_other_full2939_w4a8.sh` |
| Result files | Done | `results/full_benchmark/` |
| Local artifact policy | Done | `docs/local_artifact_layout.md` |

Model checkpoints, caches, tensor artifacts, and raw logs are intentionally excluded from Git.
