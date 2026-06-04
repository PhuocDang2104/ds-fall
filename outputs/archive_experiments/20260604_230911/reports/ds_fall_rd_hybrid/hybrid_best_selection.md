# DS-Fall-RD Hybrid Best Selection

Selection prioritizes E7 WEDA Fall F1, precision, recall, E3 direction, E7 end-to-end direction, no-timing features, then lighter experts.

## Hybrid Selection Table

| run_id | fall_expert | feature_set | fall_f1 | fall_precision | fall_recall | FP | FN | E7_WEDA_FP_reduction_vs_A5_reference | direction_macro_f1 | E3_Direction_Macro_F1 | E6_BITS_Fall_F1 | E7_E2E_Direction_Macro_F1 | E7_E2E_Direction_Coverage | upper_bound_timing | edge_suitability | passes_hard_target | meets_strong_weda_target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HYB_HGB_FullTiming | HGB | FullTiming | 0.9434 | 0.8929 | 1.0000 | 3 | 0 | 18 | 0.8390 | 0.8967 | 0.9375 | 0.8390 | 1.0000 | yes | low | yes | yes |
| HYB_RF_FullTiming | RF | FullTiming | 0.9091 | 0.8333 | 1.0000 | 5 | 0 | 16 | 0.8390 | 0.8967 | 0.9348 | 0.8390 | 1.0000 | yes | medium | yes | yes |
| HYB_HGB_FullNoTiming | HGB | FullNoTiming | 0.8571 | 0.7742 | 0.9600 | 7 | 1 | 14 | 0.8390 | 0.8967 | 0.9333 | 0.8140 | 0.9600 | no | low | yes | no |
| HYB_GB_FullTiming | GB | FullTiming | 0.9434 | 0.8929 | 1.0000 | 3 | 0 | 18 | 0.8390 | 0.8967 | 0.8958 | 0.8390 | 1.0000 | yes | medium | no | yes |
| HYB_GB_FullNoTiming | GB | FullNoTiming | 0.8364 | 0.7667 | 0.9200 | 7 | 2 | 14 | 0.8390 | 0.8967 | 0.8966 | 0.7926 | 0.9200 | no | medium | no | no |
| HYB_LR_FullNoTiming | LR | FullNoTiming | 0.7826 | 0.8571 | 0.7200 | 3 | 7 | 18 | 0.8390 | 0.8967 | 0.8235 | 0.6401 | 0.7200 | no | high | no | no |
| HYB_LR_Axis | LR | Axis | 0.7541 | 0.6389 | 0.9200 | 13 | 2 | 8 | 0.8390 | 0.8967 | 0.7500 | 0.7855 | 0.9200 | no | high | no | no |
| HYB_HGB_Lite10 | HGB | Lite10 | 0.7451 | 0.7308 | 0.7600 | 7 | 6 | 14 | 0.8390 | 0.8967 | 0.7674 | 0.6525 | 0.7600 | no | low | no | no |
| HYB_HGB_Axis | HGB | Axis | 0.7241 | 0.6364 | 0.8400 | 12 | 4 | 9 | 0.8390 | 0.8967 | 0.8235 | 0.7239 | 0.8400 | no | low | no | no |
| HYB_GB_Axis | GB | Axis | 0.7200 | 0.7200 | 0.7200 | 7 | 7 | 14 | 0.8390 | 0.8967 | 0.7727 | 0.6848 | 0.7200 | no | medium | no | no |
| HYB_RF_FullNoTiming | RF | FullNoTiming | 0.7042 | 0.5435 | 1.0000 | 21 | 0 | 0 | 0.8390 | 0.8967 | 0.8636 | 0.8390 | 1.0000 | no | medium | no | no |
| HYB_LR_Lite10 | LR | Lite10 | 0.6667 | 0.5106 | 0.9600 | 23 | 1 | -2 | 0.8390 | 0.8967 | 0.7816 | 0.8105 | 0.9600 | no | high | no | no |
| HYB_RF_Axis | RF | Axis | 0.6575 | 0.5000 | 0.9600 | 24 | 1 | -3 | 0.8390 | 0.8967 | 0.7525 | 0.8140 | 0.9600 | no | medium | no | no |
| HYB_RF_Lite10 | RF | Lite10 | 0.6486 | 0.4898 | 0.9600 | 25 | 1 | -4 | 0.8390 | 0.8967 | 0.7273 | 0.8140 | 0.9600 | no | medium | no | no |
| HYB_DT_Axis | DT | Axis | 0.6250 | 0.4545 | 1.0000 | 30 | 0 | -9 | 0.8390 | 0.8967 | 0.6964 | 0.8390 | 1.0000 | no | high | no | no |
| HYB_DT_Lite10 | DT | Lite10 | 0.6173 | 0.4464 | 1.0000 | 31 | 0 | -10 | 0.8390 | 0.8967 | 0.6916 | 0.8390 | 1.0000 | no | high | no | no |
| HYB_GB_Lite10 | GB | Lite10 | 0.6000 | 0.4364 | 0.9600 | 31 | 1 | -10 | 0.8390 | 0.8967 | 0.6964 | 0.8140 | 0.9600 | no | medium | no | no |

## Best Run

Best selected run: `HYB_HGB_FullTiming`.

- E7 WEDA Fall F1: 0.9434
- E7 WEDA precision/recall: 0.8929/1.0000
- E7 WEDA FP/FN: 3/0
- E3 Direction Macro F1: 0.8967
- E7 end-to-end Direction Macro F1: 0.8390
- Feature set: FullTiming; expert: HGB; upper-bound timing: True
