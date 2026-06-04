# DS-Fall-RD Hybrid Summary

## Protocol

- Hybrid final fall decision uses ML Fall Expert only.
- Direction output is copied from DS-Fall-RD A5WCEFW reference and is not trained or modified.
- Deep fall head is not used for final fall in hybrid runs. In option B, A5 is kept as auxiliary/reference direction expert.
- Thresholds are selected on validation split only; test is used only for evaluation.

## E7 WEDA Hybrid Results

| run_id | fall_expert | feature_set | fall_f1 | fall_precision | fall_recall | FP | FN | E7_WEDA_FP_reduction_vs_A5_reference | direction_macro_f1 | E3_Direction_Macro_F1 | E6_BITS_Fall_F1 | E7_E2E_Direction_Macro_F1 | E7_E2E_Direction_Coverage | upper_bound_timing | edge_suitability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HYB_HGB_FullTiming | HGB | FullTiming | 0.9434 | 0.8929 | 1.0000 | 3 | 0 | 18 | 0.8390 | 0.8967 | 0.9375 | 0.8390 | 1.0000 | yes | low |
| HYB_RF_FullTiming | RF | FullTiming | 0.9091 | 0.8333 | 1.0000 | 5 | 0 | 16 | 0.8390 | 0.8967 | 0.9348 | 0.8390 | 1.0000 | yes | medium |
| HYB_HGB_FullNoTiming | HGB | FullNoTiming | 0.8571 | 0.7742 | 0.9600 | 7 | 1 | 14 | 0.8390 | 0.8967 | 0.9333 | 0.8140 | 0.9600 | no | low |
| HYB_GB_FullTiming | GB | FullTiming | 0.9434 | 0.8929 | 1.0000 | 3 | 0 | 18 | 0.8390 | 0.8967 | 0.8958 | 0.8390 | 1.0000 | yes | medium |
| HYB_GB_FullNoTiming | GB | FullNoTiming | 0.8364 | 0.7667 | 0.9200 | 7 | 2 | 14 | 0.8390 | 0.8967 | 0.8966 | 0.7926 | 0.9200 | no | medium |
| HYB_LR_FullNoTiming | LR | FullNoTiming | 0.7826 | 0.8571 | 0.7200 | 3 | 7 | 18 | 0.8390 | 0.8967 | 0.8235 | 0.6401 | 0.7200 | no | high |
| HYB_LR_Axis | LR | Axis | 0.7541 | 0.6389 | 0.9200 | 13 | 2 | 8 | 0.8390 | 0.8967 | 0.7500 | 0.7855 | 0.9200 | no | high |
| HYB_HGB_Lite10 | HGB | Lite10 | 0.7451 | 0.7308 | 0.7600 | 7 | 6 | 14 | 0.8390 | 0.8967 | 0.7674 | 0.6525 | 0.7600 | no | low |
| HYB_HGB_Axis | HGB | Axis | 0.7241 | 0.6364 | 0.8400 | 12 | 4 | 9 | 0.8390 | 0.8967 | 0.8235 | 0.7239 | 0.8400 | no | low |
| HYB_GB_Axis | GB | Axis | 0.7200 | 0.7200 | 0.7200 | 7 | 7 | 14 | 0.8390 | 0.8967 | 0.7727 | 0.6848 | 0.7200 | no | medium |
| HYB_RF_FullNoTiming | RF | FullNoTiming | 0.7042 | 0.5435 | 1.0000 | 21 | 0 | 0 | 0.8390 | 0.8967 | 0.8636 | 0.8390 | 1.0000 | no | medium |
| HYB_LR_Lite10 | LR | Lite10 | 0.6667 | 0.5106 | 0.9600 | 23 | 1 | -2 | 0.8390 | 0.8967 | 0.7816 | 0.8105 | 0.9600 | no | high |
| HYB_RF_Axis | RF | Axis | 0.6575 | 0.5000 | 0.9600 | 24 | 1 | -3 | 0.8390 | 0.8967 | 0.7525 | 0.8140 | 0.9600 | no | medium |
| HYB_RF_Lite10 | RF | Lite10 | 0.6486 | 0.4898 | 0.9600 | 25 | 1 | -4 | 0.8390 | 0.8967 | 0.7273 | 0.8140 | 0.9600 | no | medium |
| HYB_DT_Axis | DT | Axis | 0.6250 | 0.4545 | 1.0000 | 30 | 0 | -9 | 0.8390 | 0.8967 | 0.6964 | 0.8390 | 1.0000 | no | high |
| HYB_DT_Lite10 | DT | Lite10 | 0.6173 | 0.4464 | 1.0000 | 31 | 0 | -10 | 0.8390 | 0.8967 | 0.6916 | 0.8390 | 1.0000 | no | high |
| HYB_GB_Lite10 | GB | Lite10 | 0.6000 | 0.4364 | 0.9600 | 31 | 1 | -10 | 0.8390 | 0.8967 | 0.6964 | 0.8140 | 0.9600 | no | medium |

## Baseline E7 Rows

| run_id | model_type | fall_expert | fall_f1 | fall_precision | fall_recall | FP | FN | direction_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DS_Fall_RD_A5_reference | baseline_deep_temporal | A5_deep_fall_head | 0.6857 | 0.5333 | 0.9600 | 21 | 1 | 0.8390 |
| Best_FV1_Lite10_AND | baseline_fv_guard | LR_AND_verifier | 0.7500 | 0.6154 | 0.9600 | 15 | 1 | 0.8390 |
| Classical_Fall_Only_Best | baseline_classical_fall_only | gradient_boosting | 0.9615 | 0.9259 | 1.0000 |  |  |  |

## Best By Feature Set

| feature_set | run_id | fall_expert | fall_f1 | fall_precision | fall_recall | FP | upper_bound_timing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FullTiming | HYB_HGB_FullTiming | HGB | 0.9434 | 0.8929 | 1.0000 | 3 | yes |
| FullNoTiming | HYB_HGB_FullNoTiming | HGB | 0.8571 | 0.7742 | 0.9600 | 7 | no |
| Axis | HYB_LR_Axis | LR | 0.7541 | 0.6389 | 0.9200 | 13 | no |
| Lite10 | HYB_HGB_Lite10 | HGB | 0.7451 | 0.7308 | 0.7600 | 7 | no |

## Best By Fall Expert

| fall_expert | run_id | feature_set | fall_f1 | fall_precision | fall_recall | FP |
| --- | --- | --- | --- | --- | --- | --- |
| HGB | HYB_HGB_FullTiming | FullTiming | 0.9434 | 0.8929 | 1.0000 | 3 |
| RF | HYB_RF_FullTiming | FullTiming | 0.9091 | 0.8333 | 1.0000 | 5 |
| GB | HYB_GB_FullTiming | FullTiming | 0.9434 | 0.8929 | 1.0000 | 3 |
| LR | HYB_LR_FullNoTiming | FullNoTiming | 0.7826 | 0.8571 | 0.7200 | 3 |
| DT | HYB_DT_Axis | Axis | 0.6250 | 0.4545 | 1.0000 | 30 |

## Thresholds

| run_id | feature_set | fall_expert | threshold | threshold_note | selected_on |
| --- | --- | --- | --- | --- | --- |
| HYB_LR_Lite10 | Lite10 | LR | 0.7944 | best_val_f1_then_recall088_hard_negative_fp_tiebreak | validation |
| HYB_DT_Lite10 | Lite10 | DT | 0.6524 | best_val_f1_then_recall088_hard_negative_fp_tiebreak | validation |
| HYB_RF_Lite10 | Lite10 | RF | 0.6357 | best_val_f1_then_recall088_hard_negative_fp_tiebreak | validation |
| HYB_GB_Lite10 | Lite10 | GB | 0.1931 | best_val_f1_then_recall088_hard_negative_fp_tiebreak | validation |
| HYB_HGB_Lite10 | Lite10 | HGB | 0.7819 | best_val_f1_then_recall088_hard_negative_fp_tiebreak | validation |
| HYB_LR_Axis | Axis | LR | 0.8183 | best_val_f1_then_recall088_hard_negative_fp_tiebreak | validation |
| HYB_DT_Axis | Axis | DT | 0.5146 | best_val_f1_then_recall088_hard_negative_fp_tiebreak | validation |
| HYB_RF_Axis | Axis | RF | 0.6357 | best_val_f1_then_recall088_hard_negative_fp_tiebreak | validation |
| HYB_GB_Axis | Axis | GB | 0.6315 | best_val_f1_then_recall088_hard_negative_fp_tiebreak | validation |
| HYB_HGB_Axis | Axis | HGB | 0.4979 | best_val_f1_then_recall088_hard_negative_fp_tiebreak | validation |
| HYB_LR_FullNoTiming | FullNoTiming | LR | 0.9465 | best_val_f1_then_recall088_hard_negative_fp_tiebreak | validation |
| HYB_RF_FullNoTiming | FullNoTiming | RF | 0.6691 | best_val_f1_then_recall088_hard_negative_fp_tiebreak | validation |
| HYB_GB_FullNoTiming | FullNoTiming | GB | 0.4854 | best_val_f1_then_recall088_hard_negative_fp_tiebreak | validation |
| HYB_HGB_FullNoTiming | FullNoTiming | HGB | 0.2049 | best_val_f1_then_recall088_hard_negative_fp_tiebreak | validation |
| HYB_RF_FullTiming | FullTiming | RF | 0.5522 | best_val_f1_then_recall088_hard_negative_fp_tiebreak | validation |
| HYB_GB_FullTiming | FullTiming | GB | 0.7735 | best_val_f1_then_recall088_hard_negative_fp_tiebreak | validation |
| HYB_HGB_FullTiming | FullTiming | HGB | 0.3225 | best_val_f1_then_recall088_hard_negative_fp_tiebreak | validation |

## Complexity

| run_id | model_type | feature_set | fall_expert | number_features | estimated_params_or_nodes | number_trees | max_depth | edge_suitability | complexity_notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DS_Fall_RD_A5_reference | baseline_deep_temporal | temporal_tilt12 | A5_deep_fall_head | 12 | 65959 | 0 |  | high | A5 reference. |
| Best_FV1_Lite10_AND | baseline_fv_guard | Lite10 | LR_AND_verifier | 10 | 65970 | 0 |  | high | Prior FV1 baseline. |
| Classical_Fall_Only_Best | baseline_classical_fall_only | summary_features | gradient_boosting |  |  |  |  | analysis_only | Imported from ml_feature_analysis comparison. |
| HYB_LR_Lite10 | hybrid_ml_fall_a5_direction | Lite10 | LR | 10 | 11 | 0 | 0.0000 | high | LR 11 weights plus scaler. |
| HYB_DT_Lite10 | hybrid_ml_fall_a5_direction | Lite10 | DT | 10 | 15 | 1 | 3.0000 | high | DecisionTree depth=3, leaves=8. |
| HYB_RF_Lite10 | hybrid_ml_fall_a5_direction | Lite10 | RF | 10 | 2452 | 50 | 5.0000 | medium | RF 50 trees depth<=5, total nodes=2452. |
| HYB_GB_Lite10 | hybrid_ml_fall_a5_direction | Lite10 | GB | 10 | 1436 | 100 | 3.0000 | medium | GradientBoosting trees=100. |
| HYB_HGB_Lite10 | hybrid_ml_fall_a5_direction | Lite10 | HGB | 10 | 100 | 100 |  | low | HistGradientBoosting iterations=100. |
| HYB_LR_Axis | hybrid_ml_fall_a5_direction | Axis | LR | 10 | 11 | 0 | 0.0000 | high | LR 11 weights plus scaler. |
| HYB_DT_Axis | hybrid_ml_fall_a5_direction | Axis | DT | 10 | 15 | 1 | 3.0000 | high | DecisionTree depth=3, leaves=8. |
| HYB_RF_Axis | hybrid_ml_fall_a5_direction | Axis | RF | 10 | 2464 | 50 | 5.0000 | medium | RF 50 trees depth<=5, total nodes=2464. |
| HYB_GB_Axis | hybrid_ml_fall_a5_direction | Axis | GB | 10 | 1472 | 100 | 3.0000 | medium | GradientBoosting trees=100. |
| HYB_HGB_Axis | hybrid_ml_fall_a5_direction | Axis | HGB | 10 | 100 | 100 |  | low | HistGradientBoosting iterations=100. |
| HYB_LR_FullNoTiming | hybrid_ml_fall_a5_direction | FullNoTiming | LR | 132 | 133 | 0 | 0.0000 | high | LR 133 weights plus scaler. |
| HYB_RF_FullNoTiming | hybrid_ml_fall_a5_direction | FullNoTiming | RF | 132 | 2562 | 50 | 5.0000 | medium | RF 50 trees depth<=5, total nodes=2562. |
| HYB_GB_FullNoTiming | hybrid_ml_fall_a5_direction | FullNoTiming | GB | 132 | 1408 | 100 | 3.0000 | medium | GradientBoosting trees=100. |
| HYB_HGB_FullNoTiming | hybrid_ml_fall_a5_direction | FullNoTiming | HGB | 132 | 100 | 100 |  | low | HistGradientBoosting iterations=100. |
| HYB_RF_FullTiming | hybrid_ml_fall_a5_direction | FullTiming | RF | 136 | 2420 | 50 | 5.0000 | medium | RF 50 trees depth<=5, total nodes=2420. |
| HYB_GB_FullTiming | hybrid_ml_fall_a5_direction | FullTiming | GB | 136 | 1432 | 100 | 3.0000 | medium | GradientBoosting trees=100. |
| HYB_HGB_FullTiming | hybrid_ml_fall_a5_direction | FullTiming | HGB | 136 | 100 | 100 |  | low | HistGradientBoosting iterations=100. |

## End-to-end Direction

| run_id | feature_set | fall_expert | end_to_end_direction_macro_f1 | end_to_end_direction_coverage | end_to_end_direction_correct_rate | direction_n_supervised |
| --- | --- | --- | --- | --- | --- | --- |
| HYB_HGB_FullTiming | FullTiming | HGB | 0.8390 | 1.0000 | 0.8400 | 25 |
| HYB_DT_Lite10 | Lite10 | DT | 0.8390 | 1.0000 | 0.8400 | 25 |
| HYB_GB_FullTiming | FullTiming | GB | 0.8390 | 1.0000 | 0.8400 | 25 |
| HYB_RF_FullTiming | FullTiming | RF | 0.8390 | 1.0000 | 0.8400 | 25 |
| HYB_RF_FullNoTiming | FullNoTiming | RF | 0.8390 | 1.0000 | 0.8400 | 25 |
| HYB_DT_Axis | Axis | DT | 0.8390 | 1.0000 | 0.8400 | 25 |
| HYB_RF_Axis | Axis | RF | 0.8140 | 0.9600 | 0.8000 | 25 |
| HYB_HGB_FullNoTiming | FullNoTiming | HGB | 0.8140 | 0.9600 | 0.8000 | 25 |
| DS_Fall_RD_A5_reference | temporal_tilt12 | A5_deep_fall_head | 0.8140 | 0.9600 | 0.8000 | 25 |
| HYB_GB_Lite10 | Lite10 | GB | 0.8140 | 0.9600 | 0.8000 | 25 |
| HYB_RF_Lite10 | Lite10 | RF | 0.8140 | 0.9600 | 0.8000 | 25 |
| HYB_LR_Lite10 | Lite10 | LR | 0.8105 | 0.9600 | 0.8000 | 25 |
| HYB_GB_FullNoTiming | FullNoTiming | GB | 0.7926 | 0.9200 | 0.7600 | 25 |
| HYB_LR_Axis | Axis | LR | 0.7855 | 0.9200 | 0.7600 | 25 |
| HYB_HGB_Axis | Axis | HGB | 0.7239 | 0.8400 | 0.6800 | 25 |
| HYB_GB_Axis | Axis | GB | 0.6848 | 0.7200 | 0.6000 | 25 |
| HYB_HGB_Lite10 | Lite10 | HGB | 0.6525 | 0.7600 | 0.6000 | 25 |
| HYB_LR_FullNoTiming | FullNoTiming | LR | 0.6401 | 0.7200 | 0.5600 | 25 |

## Answers

1. Hybrid improves WEDA Fall F1 over A5 from 0.6857 to 0.9434 for `HYB_HGB_FullTiming`.
2. Best Fall Expert is `HGB` in `HYB_HGB_FullTiming`.
3. Best feature set is `FullTiming` with `HYB_HGB_FullTiming`.
4. FullTiming is higher (0.9434 vs no-timing 0.8571) but should be treated as upper-bound because it uses event-centered timing cues.
5. Hybrid reduces WEDA FP from A5 21 to 3. Prior FV1 FP=15, so hybrid changes FP by 12 relative to FV1.
6. Direction stays unchanged at the direction-head level because all hybrid runs use the same A5 direction probabilities.
7. End-to-end direction is affected by fall recall: `HYB_HGB_FullTiming` keeps coverage=1.0000 and E2E direction macro F1=0.8390.
8. Option B is more paper-appropriate: keep A5 as the auxiliary/reference multitask direction expert, but use the ML Fall Expert for final fall inference. Option A is the same inference rule but undersells why the direction expert exists.
9. Do not replace A5 main architecture with the FullTiming upper-bound as a paper main result. Practical no-timing candidate is `HYB_HGB_FullNoTiming` with WEDA F1=0.8571, precision=0.7742, recall=0.9600.
10. Paper position: FullTiming is upper-bound analysis because it uses event-centered timing cues. Use `HYB_HGB_FullNoTiming` as the practical no-timing hybrid if a deployable pipeline is needed.
