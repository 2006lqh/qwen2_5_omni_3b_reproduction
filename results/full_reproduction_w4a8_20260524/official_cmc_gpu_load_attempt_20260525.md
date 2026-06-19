# Official CMC GPU Load Attempt - 2026-05-25

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB total.
- Small PyTorch CUDA allocation succeeded, so CUDA is visible.
- Official CMC with `--LR --quantize --rank 0.05 --n_cali_samples 1 --eval_sqnr` failed at `infer_mas.py:154 llm.model.to("cuda")`.
- Official CMC with the same settings but without `--eval_sqnr` failed at the same line.
- No official white matrix or official low-rank adapter was produced.
- Therefore the original full CMC has not loaded successfully on this 6GB GPU.

Logs:
- `outputs/full_reproduction_w4a8_20260524/official_cmc_full_load_smoke_20260525/official_cmc_load_smoke.log`
- `outputs/full_reproduction_w4a8_20260524/official_cmc_full_load_only_20260525/official_cmc_load_only.log`
