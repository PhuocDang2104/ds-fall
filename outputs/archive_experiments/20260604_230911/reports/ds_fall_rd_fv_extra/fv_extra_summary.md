# DS-Fall-RD-FV Extra Summary

## Protocol

- Exactly six runs were attempted: FV1A, FV1B, FV7, FV8, FV9, FV10.
- Benchmark is fixed to BITS/WEDA, 25 Hz, 2-second event-centered, existing split.
- A5 reference fall probability is Stage-1; direction probability is copied unchanged from reference.
- Verifiers are trained on train split only; thresholds are selected on validation only.

## Main E7 WEDA Results

| run_id | verifier | features | fusion | fall_f1 | fall_precision | fall_recall | FP | FN | weda_fp_reduction_vs_reference | weda_fp_reduction_vs_fv1_lite10_and | direction_macro_f1 | E3_Direction_Macro_F1 | E6_BITS_Fall_F1 | edge_suitability_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FV1A | logistic_regression | Lite10 | and | 0.6857 | 0.5333 | 0.9600 | 21 | 1 | 0 | -6 | 0.8390 | 0.8967 | 0.9451 | Very edge-friendly LR: 11 scalar parameters plus scaler. |
| FV7 | logistic_regression | AxisHard10 | and | 0.6857 | 0.5333 | 0.9600 | 21 | 1 | 0 | -6 | 0.8390 | 0.8967 | 0.9451 | Very edge-friendly LR: 11 scalar parameters plus scaler. |
| FV9 | decision_tree_depth3 | AxisHard10 | and | 0.6857 | 0.5333 | 0.9600 | 21 | 1 | 0 | -6 | 0.8390 | 0.8967 | 0.9451 | Edge-friendly shallow tree: depth=3, leaves=8. |
| FV10 | tiny_random_forest_10x_depth3 | AxisHard10 | and | 0.6857 | 0.5333 | 0.9600 | 21 | 1 | 0 | -6 | 0.8390 | 0.8967 | 0.9451 | TinyRF: 10 trees, max observed depth=3, total leaves=80. |
| FV8 | logistic_regression | AxisHard10 | stacked_logit_fusion | 0.6761 | 0.5217 | 0.9600 | 22 | 1 | -1 | -7 | 0.8390 | 0.8967 | 0.9451 | Very edge-friendly LR: 11 scalar parameters plus scaler. |
| FV1B | logistic_regression | Lite10 | and | 0.6849 | 0.5208 | 1.0000 | 23 | 0 | -2 | -8 | 0.8390 | 0.8967 | 0.9149 | Very edge-friendly LR: 11 scalar parameters plus scaler. |

## Thresholds

| run_id | threshold_rule | selected_rule | threshold_deep | threshold_verifier | threshold_final | fusion | features | verifier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FV1A | max_f1_precision_065 | max_f1_precision_065 | 0.5000 | 0.0070 |  | and | Lite10 | logistic_regression |
| FV1B | min_fp_recall_088 | min_fp_recall_088 | 0.0985 | 0.0086 |  | and | Lite10 | logistic_regression |
| FV7 | max_f1_precision_065 | max_f1_precision_065 | 0.5000 | 0.0074 |  | and | AxisHard10 | logistic_regression |
| FV8 | max_f1_precision_065 | max_f1_precision_065 |  |  | 0.2500 | stacked_logit_fusion | AxisHard10 | logistic_regression |
| FV9 | max_f1_precision_065 | max_f1_precision_065 | 0.5056 | 0.0010 |  | and | AxisHard10 | decision_tree_depth3 |
| FV10 | max_f1_precision_065 | max_f1_precision_065 | 0.5056 | 0.0010 |  | and | AxisHard10 | tiny_random_forest_10x_depth3 |

## WEDA FP Correction

| run_id | reference_fp | extra_fp | corrected_reference_fp |
| --- | --- | --- | --- |
| FV10 | 21.0000 | 21.0000 | 0.0000 |
| FV1A | 21.0000 | 21.0000 | 0.0000 |
| FV1B | 21.0000 | 23.0000 | 0.0000 |
| FV7 | 21.0000 | 21.0000 | 0.0000 |
| FV8 | 21.0000 | 22.0000 | 0.0000 |
| FV9 | 21.0000 | 21.0000 | 0.0000 |

## Answers

1. FV1A/FV1B vs old FV1: old FV1 had FP=15, precision=0.6154, recall=0.9600, F1=0.7500. FV1A: FP=21, precision=0.5333, recall=0.9600, F1=0.6857; FV1B: FP=23, precision=0.5208, recall=1.0000, F1=0.6849.
2. Lite10 is better by the selection rule. Best Lite10=FV1A FP=21, F1=0.6857; best AxisHard10=FV7 FP=21, F1=0.6857.
3. Best verifier is `logistic_regression` via FV1A: FP=21, precision=0.5333, F1=0.6857.
4. Strongest FP reduction with recall >= 0.88 is `FV1A`: WEDA FP=21, recall=0.9600, FP reduction vs reference=0.
5. Best paper-target trade-off is `FV1A`: FP=21, precision=0.5333, recall=0.9600, F1=0.6857, direction unchanged.
6. Runs with FP <= 12 and precision >= 0.65: none.
7. Do not replace old FV1; no extra run improves its FP/F1 trade-off enough.
8. A5 should remain the main model. The extra FV runs are FPGuard/decision-layer extensions because they do not alter the temporal direction-sensitive architecture.
9. Paper-facing conclusion: the earlier FV1 Lite10 AND remains the best FPGuard result observed so far; these six extra constrained runs do not improve it. Report FV/FPGuard as an optional precision-oriented extension, not a replacement for the A5 main model.
