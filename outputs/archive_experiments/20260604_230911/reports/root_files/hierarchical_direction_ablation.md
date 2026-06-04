# Hierarchical Direction Ablation

Selection priority: final direction macro F1, fall F1, per-dataset consistency, then params/latency.

Best by implemented ranking: `B0`
- Final direction macro F1: 0.6507
- Fall F1: 0.8776
- Per-dataset final direction F1 std: 0.1316
- Params: 65959

## Ranked Results

| id | model | use_summary_branch | fall_f1 | plane_macro_f1 | sagittal_macro_f1 | final_direction_macro_f1 | final_direction_accuracy | per_dataset_final_direction_macro_f1_std | bits_final_direction_macro_f1 | hifd_final_direction_macro_f1 | umafall_final_direction_macro_f1 | weda_final_direction_macro_f1 | params | estimated_int8_kb | inference_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | A5WCEFW | False | 0.8776 |  |  | 0.6507 | 0.6549 | 0.1316 | 0.8173 | 0.7313 | 0.5699 | 0.4820 | 65959 | 64.4131 | 82.6274 |
| B2 | A11_HIER_DIR_SUMMARY | True | 0.8198 | 0.7350 | 0.7786 | 0.6444 | 0.6460 | 0.0725 | 0.7213 | 0.6667 | 0.5231 | 0.6458 | 75033 | 73.2744 | 71.1471 |
| B1 | A10_HIER_DIR | False | 0.8172 | 0.7402 | 0.7256 | 0.5972 | 0.6018 | 0.0875 | 0.7037 | 0.4825 | 0.5047 | 0.5994 | 73801 | 72.0713 | 64.2455 |

