# Three Validation Runs

Metric: deployed-weight reconstruction proxy on selected thinker layers. Lower mean_error is better; higher SQNR is better.

| Run | SmoothQuant error | MAS error | CMC error |
| --- | ---: | ---: | ---: |
| run1_20260518_existing | 0.425107 | 0.233768 | 0.375369 |
| run2_20260519_rerun | 0.425107 | 0.233768 | 0.375369 |
| run3_20260519_rerun | 0.425107 | 0.233768 | 0.375369 |

## Averages

| Method | Avg mean_error | Avg SQNR dB |
| --- | ---: | ---: |
| smoothquant_unified | 0.425107 | 8.4412 |
| mas_split_scales | 0.233768 | 12.8597 |
| simplified_cmc | 0.375369 | 9.6746 |
