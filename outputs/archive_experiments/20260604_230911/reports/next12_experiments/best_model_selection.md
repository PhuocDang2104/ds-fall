# Best Model Selection

Hard filters: E3 Direction Macro F1 >= 0.86, E7 WEDA Direction Macro F1 >= 0.80, E6 BITS Fall F1 >= 0.90, Params <= 90k.

| role | run_id | E7_WEDA_Fall_F1 | delta_E7_WEDA_Fall_vs_REF | E3_Direction_Macro_F1 | E3_Direction_Accuracy | E3_Direction_N | E7_WEDA_Direction_Macro_F1 | E6_BITS_Fall_F1 | params | selection_score | passes_hard_filter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Best WEDA Fall F1 | E1 | 0.8462 | 0.1604 | 0.6280 | 0.6721 | 61 | 0.5107 | 0.8837 | 54823 | 0.7325 | 0.0000 |
| Best balanced trade-off | C3 | 0.7458 | 0.0600 | 0.8423 | 0.8525 | 61 | 0.8380 | 0.9011 | 65959 | 0.8211 | 0.0000 |

## Recommendation

No next-12 run passed all hard filters. Keep `REF_CURRENT_E3_ARTIFACT` as the paper main model and report the best new run as an ablation/future direction.
The best WEDA Fall F1 run is `E1` (0.8462), but it does not satisfy the full direction/fall stability criteria.

## Fall/Direction Trade-off

Runs that push fall loss or freeze/fine-tune fall-specific parts can raise WEDA Fall F1, but the common failure mode is lower E3 Direction Macro F1 or lower E6 BITS Fall F1. The final choice should not optimize WEDA Fall F1 alone.
