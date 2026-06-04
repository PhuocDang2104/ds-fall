# Best Model Selection

Hard filters:

- E3 Direction Macro F1 >= 0.86
- E7 Direction Macro F1 >= 0.80
- E6 BITS Fall F1 >= 0.90

## Selection Result

| role | run_id | seed | weda_fall_f1 | delta_weda_fall_vs_ref | e3_direction_macro_f1 | delta_e3_dir_vs_ref | e7_direction_macro_f1 | bits_fall_f1 | params | passes_hard_filter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Strict selected | REF_CURRENT_E3_ARTIFACT | 42 | 0.6857 | 0.0000 | 0.8967 | 0.0000 | 0.8390 | 0.9451 | 65959 | True |
| Best WEDA fall ablation | L3_FALL_WCE | 42 | 0.8148 | 0.1291 | 0.8298 | -0.0668 | 0.8390 | 0.9091 | 65959 | False |
| Best direction-preserving ablation | S3_DATASET_BALANCED | 42 | 0.7097 | 0.0240 | 0.8540 | -0.0427 | 0.8409 | 0.9451 | 65959 | False |

Interpretation:

- Main paper model: `REF_CURRENT_E3_ARTIFACT`. It is the only model in this targeted suite that passes all hard filters.
- Best fall-focused ablation: `L3_FALL_WCE` seed `42`. It raises WEDA Fall F1 from 0.6857 to 0.8148, but E3 Direction Macro F1 drops from 0.8967 to 0.8298.
- Best direction-preserving ablation: `S3_DATASET_BALANCED` seed `42`. It keeps E7 Direction Macro F1 at 0.8409 and E6 Fall F1 at 0.9451, but WEDA Fall F1 only reaches 0.7097.

Therefore no trained targeted variant should replace the current main model yet. Use L3 and S3 as ablation evidence, not as the main model.
