# Domain Conflict Experiments

Fixed model: B0 / A5WCEFW = DS-Fall-RD tilt12, task-specific attention, weighted CE fall, weighted CE direction.
Diagnostics use train-only feature scaling per experiment; held-out test datasets are not used for scaler or early stopping.
Dataset-specific runs are evaluated at the final epoch. Train-3-test-1 runs use `val_direction_macro_f1 + val_fall_f1` checkpoint selection.

## Dataset-Specific Training

| id | train_datasets | n_train_direction | n_test_direction | fall_f1 | direction_accuracy | direction_macro_f1 | direction_forward_f1 | direction_backward_f1 | direction_lateral_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DS_BITS | bits | 172 | 36 | 0.8660 | 0.7778 | 0.7321 | 0.9143 | 0.6154 | 0.6667 |
| DS_WEDA | weda | 300 | 25 | 0.8519 | 0.7200 | 0.7319 | 0.6957 | 0.6667 | 0.8333 |
| DS_HIFD | hifd | 74 | 15 | 1.0000 | 0.7333 | 0.7249 | 0.5714 | 0.7143 | 0.8889 |
| DS_UMAFALL | umafall | 148 | 37 | 0.0526 | 0.4054 | 0.3406 | 0.2500 | 0.5366 | 0.2353 |

## Train-3-Test-1

| id | train_datasets | test_datasets | n_train_direction | n_test_direction | fall_f1 | direction_accuracy | direction_macro_f1 | direction_forward_f1 | direction_backward_f1 | direction_lateral_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T3_BITS | weda+hifd+umafall | bits | 522 | 244 | 0.6690 | 0.4467 | 0.4280 | 0.5729 | 0.3008 | 0.4103 |
| T3_WEDA | bits+hifd+umafall | weda | 394 | 350 | 0.6895 | 0.4457 | 0.4310 | 0.5806 | 0.3238 | 0.3886 |
| T3_HIFD | bits+weda+umafall | hifd | 620 | 104 | 0.3296 | 0.4615 | 0.3992 | 0.5098 | 0.1081 | 0.5797 |
| T3_UMAFALL | bits+weda+hifd | umafall | 546 | 208 | 0.7108 | 0.3702 | 0.3685 | 0.4755 | 0.3143 | 0.3158 |

## Diagnosis

- UMAFall-only direction macro F1 is 0.3406, not meaningfully higher than mixed-training UMAFall 0.5699 (delta -0.2293).
- Weakest held-out direction generalization: train bits+weda+hifd -> test umafall with direction macro F1 0.3685.
- Strongest held-out direction generalization: train bits+hifd+umafall -> test weda with direction macro F1 0.4310.
- Dataset-specific mean direction macro F1 is 0.6324; train-3-test-1 mean is 0.4067.
- Evidence supports cross-dataset direction semantic/domain conflict.
- UMAFall should stay in the realistic multi-dataset setup, but its direction labels/window semantics still need manual protocol review.
- Best candidates for a clean-direction setup from this run: bits, weda.
