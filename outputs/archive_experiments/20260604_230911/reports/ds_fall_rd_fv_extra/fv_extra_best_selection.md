# DS-Fall-RD-FV Extra Best Selection

Selection: hard constraints first, then minimize WEDA FP, maximize WEDA precision, keep WEDA Fall F1 > 0.75 if possible, and prefer simpler verifier when metrics are close.

## E7 Selection Table

| run_id | verifier | features | fusion | fall_f1 | fall_precision | fall_recall | FP | FN | weda_fp_reduction_vs_reference | weda_fp_reduction_vs_fv1_lite10_and | direction_macro_f1 | E3_Direction_Macro_F1 | E6_BITS_Fall_F1 | passes_hard_constraints | passes_primary_target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FV1A | logistic_regression | Lite10 | and | 0.6857 | 0.5333 | 0.9600 | 21 | 1 | 0 | -6 | 0.8390 | 0.8967 | 0.9451 | yes | no |
| FV7 | logistic_regression | AxisHard10 | and | 0.6857 | 0.5333 | 0.9600 | 21 | 1 | 0 | -6 | 0.8390 | 0.8967 | 0.9451 | yes | no |
| FV9 | decision_tree_depth3 | AxisHard10 | and | 0.6857 | 0.5333 | 0.9600 | 21 | 1 | 0 | -6 | 0.8390 | 0.8967 | 0.9451 | yes | no |
| FV10 | tiny_random_forest_10x_depth3 | AxisHard10 | and | 0.6857 | 0.5333 | 0.9600 | 21 | 1 | 0 | -6 | 0.8390 | 0.8967 | 0.9451 | yes | no |
| FV8 | logistic_regression | AxisHard10 | stacked_logit_fusion | 0.6761 | 0.5217 | 0.9600 | 22 | 1 | -1 | -7 | 0.8390 | 0.8967 | 0.9451 | yes | no |
| FV1B | logistic_regression | Lite10 | and | 0.6849 | 0.5208 | 1.0000 | 23 | 0 | -2 | -8 | 0.8390 | 0.8967 | 0.9149 | yes | no |

## Best Run

Best selected run: `FV1A`.

- WEDA FP/FN: 21/1
- WEDA precision/recall/F1: 0.5333/0.9600/0.6857
- WEDA Direction Macro F1: 0.8390
- E3 Direction Macro F1: 0.8967
- E6 BITS Fall F1: 0.9451
