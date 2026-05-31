# DS-Fall-RD Model Selection

Selection priority: direction macro F1, fall F1, per-dataset consistency, then parameter/inference cost.

Best ablation by implemented ranking: `A5WCEFW`
- Direction macro F1: 0.6507053856451447
- Fall F1: 0.8776371308016878
- Params: 65959
- Estimated INT8 KB: 64.4130859375

## Ranked Results

| id | model | feature_set | direction_loss_type | fall_loss_weighted | augment | fall_f1 | direction_macro_f1 | direction_accuracy | params | estimated_int8_kb | inference_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A5WCEFW | ds_fall_rd | tilt12 | weighted_ce | True | False | 0.8776 | 0.6507 | 0.6549 | 65959 | 64.4131 | 82.6274 |

