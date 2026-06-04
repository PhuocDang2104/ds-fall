# Next 12 Experiments Summary

Benchmark: BITS/WEDA only, event-centered 2s, 25 Hz, global validation-tuned fall threshold.

## All Runs

| run_id | block | alpha_fall | lambda_dir | sampler | adapter_mode | train_mode | beta_kd | fall_threshold | fall_precision | fall_recall | fall_f1 | direction_macro_f1 | direction_accuracy | direction_n_supervised | bits_fall_f1 | bits_direction_macro_f1 | bits_direction_n_supervised | weda_fall_f1 | weda_direction_macro_f1 | weda_direction_n_supervised | params | selection_score | passes_hard_filter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | lightweight_branch_separation | 1.0000 | 1.5000 | current | both16 | standard | 0.0000 | 0.7600 | 0.9231 | 0.8219 | 0.8696 | 0.6280 | 0.6721 | 61 | 0.8837 | 0.6852 | 36 | 0.8462 | 0.5107 | 25 | 54823 | 0.7325 | 0.0000 |
| D1 | direction_preserving_finetune | 1.0000 | 1.5000 | current | none | fall_head_finetune | 0.0000 | 0.8400 | 0.8873 | 0.8630 | 0.8750 | 0.7025 | 0.7377 | 61 | 0.9213 | 0.8450 | 36 | 0.8000 | 0.5070 | 25 | 65959 | 0.7427 | 0.0000 |
| B3 | loss_sampler_near_reference | 1.2500 | 1.2500 | current | none | standard | 0.0000 | 0.4100 | 0.8375 | 0.9178 | 0.8758 | 0.7093 | 0.7377 | 61 | 0.9333 | 0.8354 | 36 | 0.7937 | 0.5460 | 25 | 65959 | 0.7522 | 0.0000 |
| C1 | loss_sampler_near_reference | 1.0000 | 1.5000 | dataset_balanced | none | standard | 0.0000 | 0.6800 | 0.7831 | 0.8904 | 0.8333 | 0.8221 | 0.8361 | 61 | 0.8913 | 0.8354 | 36 | 0.7500 | 0.7898 | 25 | 65959 | 0.8055 | 0.0000 |
| C3 | loss_sampler_near_reference | 1.5000 | 1.5000 | dataset_balanced | none | standard | 0.0000 | 0.8900 | 0.8182 | 0.8630 | 0.8400 | 0.8423 | 0.8525 | 61 | 0.9011 | 0.8354 | 36 | 0.7458 | 0.8380 | 25 | 65959 | 0.8211 | 0.0000 |
| D4 | direction_preserving_finetune | 2.0000 | 1.0000 | current | none | kd_finetune | 2.0000 | 0.6700 | 0.8312 | 0.8767 | 0.8533 | 0.8274 | 0.8361 | 61 | 0.9231 | 0.8867 | 36 | 0.7458 | 0.7533 | 25 | 65959 | 0.8050 | 0.0000 |
| D3 | direction_preserving_finetune | 2.0000 | 1.0000 | current | none | kd_finetune | 1.0000 | 0.9000 | 0.8400 | 0.8630 | 0.8514 | 0.7837 | 0.7869 | 61 | 0.9213 | 0.8867 | 36 | 0.7458 | 0.6133 | 25 | 65959 | 0.7657 | 0.0000 |
| D2 | direction_preserving_finetune | 1.0000 | 1.5000 | current | fall16 | fall_head_finetune | 0.0000 | 0.6800 | 0.8077 | 0.8630 | 0.8344 | 0.7025 | 0.7377 | 61 | 0.9011 | 0.8450 | 36 | 0.7333 | 0.5070 | 25 | 60391 | 0.7156 | 0.0000 |
| B1 | loss_sampler_near_reference | 1.2500 | 1.5000 | current | none | standard | 0.0000 | 0.8200 | 0.7927 | 0.8904 | 0.8387 | 0.8504 | 0.8525 | 61 | 0.9213 | 0.9771 | 36 | 0.7273 | 0.6839 | 25 | 65959 | 0.7896 | 0.0000 |
| F1 | lightweight_branch_separation | 1.0000 | 1.5000 | current | none | head_specific_feature_routing | 0.0000 | 0.3900 | 0.8289 | 0.8630 | 0.8456 | 0.7715 | 0.7869 | 61 | 0.9333 | 0.8635 | 36 | 0.7119 | 0.6242 | 25 | 33615 | 0.7558 | 0.0000 |
| B2 | loss_sampler_near_reference | 1.5000 | 1.5000 | current | none | standard | 0.0000 | 0.1700 | 0.8077 | 0.8630 | 0.8344 | 0.7523 | 0.7705 | 61 | 0.9213 | 0.7870 | 36 | 0.7097 | 0.7037 | 25 | 65959 | 0.7634 | 0.0000 |
| C2 | loss_sampler_near_reference | 1.2500 | 1.5000 | dataset_balanced | none | standard | 0.0000 | 0.8000 | 0.8378 | 0.8493 | 0.8435 | 0.7523 | 0.7705 | 61 | 0.9333 | 0.7966 | 36 | 0.7018 | 0.6825 | 25 | 65959 | 0.7595 | 0.0000 |

## Key Findings

- Runs passing hard filters: none.
- Best E7 WEDA Fall F1: `E1` = 0.8462.
- Best balanced trade-off: `C3` with score 0.8211.
- Reference comparison is kept outside `all_runs.csv` so the table contains exactly the requested 12 experiments.
- D1-D4 used an internally retrained serializable reference teacher because the old Lambda-layer artifact is not loadable in the current Keras runtime.
