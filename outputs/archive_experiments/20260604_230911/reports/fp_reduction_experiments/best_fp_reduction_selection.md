# Best FP Reduction Selection

Hard filters: E3 Direction Macro F1 >= 0.86, E7 WEDA Direction Macro F1 >= 0.80, E6 BITS Fall F1 >= 0.90, E7 WEDA Recall >= 0.88, Params <= 90k.
Desired FP target: E7 WEDA FP < 12, precision > 0.65, recall >= 0.88.

## Threshold-only Best

| block | run_id | rule_description | eval | dataset | threshold | TN | E7_WEDA_FP | E7_WEDA_FN | TP | E7_WEDA_Precision | E7_WEDA_Recall | E7_WEDA_Fall_F1 | fall_accuracy | E7_WEDA_Direction_Macro_F1 | direction_accuracy | direction_n_supervised | weda_fp_reduction_vs_ref | weda_precision_gain_vs_ref | weda_recall_delta_vs_ref | E3_Direction_Macro_F1 | E6_BITS_Fall_F1 | E6_BITS_Direction_Macro_F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| threshold_decision | T1 | global threshold maximizing validation Fall F1 | E7 | weda | 0.7616 | 137 | 18 | 1 | 24 | 0.5714 | 0.9600 | 0.7164 | 0.8944 | 0.8390 | 0.8400 | 25 | 3.0000 | 0.0381 | 0.0000 | 0.8967 | 0.9451 | 0.9350 |

## Hard-negative Best

| run_id | method | hard_negative_source | threshold | params | E7_WEDA_FP | E7_WEDA_FN | E7_WEDA_TP | E7_WEDA_TN | E7_WEDA_Precision | E7_WEDA_Recall | E7_WEDA_Fall_F1 | E7_WEDA_Direction_Macro_F1 | E3_Direction_Macro_F1 | E3_Fall_F1 | E6_BITS_Fall_F1 | E6_BITS_Direction_Macro_F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HN1 | hard_negative_weighted_ce | feature_rule | 0.9606 | 65959 | 10 | 4 | 21 | 145 | 0.6774 | 0.8400 | 0.7500 | 0.5499 | 0.7310 | 0.8552 | 0.9213 | 0.8704 |

## Threshold Rules Passing Hard Filters

| run_id | E7_WEDA_FP | E7_WEDA_FN | E7_WEDA_TP | E7_WEDA_TN | E7_WEDA_Precision | E7_WEDA_Recall | E7_WEDA_Fall_F1 | E7_WEDA_Direction_Macro_F1 | threshold | E3_Direction_Macro_F1 | E3_Fall_F1 | E6_BITS_Fall_F1 | E6_BITS_Direction_Macro_F1 | params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | 18 | 1 | 24 | 137 | 0.5714 | 0.9600 | 0.7164 | 0.8390 | 0.7616 | 0.8967 | 0.8481 | 0.9451 | 0.9350 | 65959 |
| T2 | 18 | 1 | 24 | 137 | 0.5714 | 0.9600 | 0.7164 | 0.8390 | 0.7616 | 0.8967 | 0.8481 | 0.9451 | 0.9350 | 65959 |
| T3 | 18 | 1 | 24 | 137 | 0.5714 | 0.9600 | 0.7164 | 0.8390 | 0.7616 | 0.8967 | 0.8481 | 0.9451 | 0.9350 | 65959 |
| T4 | 23 | 0 | 25 | 132 | 0.5208 | 1.0000 | 0.6849 | 0.8390 | 0.1098 | 0.8967 | 0.8144 | 0.9149 | 0.9350 | 65959 |

## Hard-negative Runs Passing Hard Filters

| run_id | method | hard_negative_source | threshold | params | E7_WEDA_FP | E7_WEDA_FN | E7_WEDA_TP | E7_WEDA_TN | E7_WEDA_Precision | E7_WEDA_Recall | E7_WEDA_Fall_F1 | E7_WEDA_Direction_Macro_F1 | E3_Direction_Macro_F1 | E3_Fall_F1 | E6_BITS_Fall_F1 | E6_BITS_Direction_Macro_F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HN3 | two_stage_fall_branch_finetune | prediction_rule | 0.6702 | 65959 | 21 | 1 | 24 | 134 | 0.5333 | 0.9600 | 0.6857 | 0.8390 | 0.8967 | 0.8323 | 0.9451 | 0.9350 |
| HN4 | two_stage_fall_branch_direction_kd | prediction_rule | 0.7796 | 65959 | 19 | 1 | 24 | 136 | 0.5581 | 0.9600 | 0.7059 | 0.8390 | 0.8967 | 0.8428 | 0.9451 | 0.9350 |

## Hard-negative Runs Passing Hard Filters And FP Target

_No rows._

## Recommendation

No hard-negative run satisfies both the hard filters and the desired FP target. Keep `REF_CURRENT_E3_ARTIFACT` as the main model. Use T1 as a decision-threshold ablation, HN1 as aggressive FP-reduction ablation, and HN4 as direction-safe but modest FP-reduction ablation.
