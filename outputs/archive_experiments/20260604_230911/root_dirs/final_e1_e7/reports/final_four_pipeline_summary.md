# Final Four Pipeline E1-E7 Summary

## Scope

- BITS/WEDA only.
- 25 Hz, 2-second event-centered windows.
- Main temporal input: 50 x 12 tilt12.
- FullTiming is not used as a main result.
- Test sets are not used for threshold tuning.

## Topline E3/E6/E7

| pipeline | experiment_id | fall_f1 | fall_precision | fall_recall | FP | direction_macro_f1 | direction_accuracy | E2E_Direction_Macro_F1 | coverage | edge_suitability | estimated_complexity_units |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A5_reference | E3_BITS_WEDA_MIXED | 0.8323 | 0.7614 | 0.9178 | 21 | 0.8967 | 0.9016 | 0.8372 | 0.9180 | medium | 65959 |
| A5_reference | E6_E3_MIXED_TEST_BITS | 0.9451 | 1.0000 | 0.8958 | 0 | 0.9350 | 0.9444 | 0.8430 | 0.8889 | medium | 65959 |
| A5_reference | E7_E3_MIXED_TEST_WEDA | 0.6857 | 0.5333 | 0.9600 | 21 | 0.8390 | 0.8400 | 0.8140 | 0.9600 | medium | 65959 |
| Hybrid_NoTiming | E3_BITS_WEDA_MIXED | 0.8732 | 0.8986 | 0.8493 | 7 | 0.8967 | 0.9016 | 0.7998 | 0.8525 | medium | 67367 |
| Hybrid_NoTiming | E6_E3_MIXED_TEST_BITS | 0.8966 | 1.0000 | 0.8125 | 0 | 0.9350 | 0.9444 | 0.7798 | 0.8056 | medium | 67367 |
| Hybrid_NoTiming | E7_E3_MIXED_TEST_WEDA | 0.8364 | 0.7667 | 0.9200 | 7 | 0.8390 | 0.8400 | 0.7926 | 0.9200 | medium | 67367 |
| EdgeLite_Hybrid | E3_BITS_WEDA_MIXED | 0.8182 | 0.7778 | 0.8630 | 18 | 0.8967 | 0.9016 | 0.8071 | 0.8689 | medium-high | 156977 |
| EdgeLite_Hybrid | E6_E3_MIXED_TEST_BITS | 0.8125 | 0.8125 | 0.8125 | 9 | 0.9350 | 0.9444 | 0.7798 | 0.8056 | medium-high | 156977 |
| EdgeLite_Hybrid | E7_E3_MIXED_TEST_WEDA | 0.8276 | 0.7273 | 0.9600 | 9 | 0.8390 | 0.8400 | 0.8140 | 0.9600 | medium-high | 156977 |
| ML_only_separated | E3_BITS_WEDA_MIXED | 0.8732 | 0.8986 | 0.8493 | 7 | 0.8577 | 0.8689 | 0.7567 | 0.8525 | high | 5812 |
| ML_only_separated | E6_E3_MIXED_TEST_BITS | 0.8966 | 1.0000 | 0.8125 | 0 | 0.8321 | 0.8611 | 0.6586 | 0.8056 | high | 5812 |
| ML_only_separated | E7_E3_MIXED_TEST_WEDA | 0.8364 | 0.7667 | 0.9200 | 7 | 0.8851 | 0.8800 | 0.8410 | 0.9200 | high | 5812 |

## Complexity And Edge Suitability

| pipeline | fall_source | direction_source | deep_params | a5_fp32_kb | a5_int8_kb | fall_feature_count | direction_feature_count | fall_tree_count | fall_tree_nodes | fall_tree_leaves | direction_tree_count | direction_tree_nodes | direction_tree_leaves | total_tree_count | total_tree_nodes | estimated_complexity_units | edge_suitability | complexity_notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A5_reference | A5_deep_fall_head | A5_direction_head | 65959 | 257.6523 | 64.4131 | 12 | 12 | 0 | 0 | 0.0000 | 0 | 0 | 0.0000 | 0 | 0 | 65959 | medium | Single compact temporal CNN; requires TensorFlow/TFLite for deployment. |
| Hybrid_NoTiming | GradientBoosting_FallNoTiming_Full | A5_direction_head | 65959 | 257.6523 | 64.4131 | 132 | 12 | 100 | 1408 | 754.0000 | 0 | 0 | 0.0000 | 100 | 1408 | 67367 | medium | GB fall expert plus A5 direction expert; best practical accuracy, moderate edge cost. |
| EdgeLite_Hybrid | RandomForest_FallNoTiming_Core | A5_direction_head | 65959 | 257.6523 | 64.4131 | 16 | 12 | 300 | 91018 | 45659.0000 | 0 | 0 | 0.0000 | 300 | 91018 | 156977 | medium-high | Compact 16-feature RF fall expert; full pipeline still uses A5 for direction. |
| ML_only_separated | GradientBoosting_FallNoTiming_Full | GradientBoosting_DirectionFullNoTiming | 0 | 0.0000 | 0.0000 | 132 | 132 | 100 | 1408 | 754.0000 | 300 | 4404 | 2352.0000 | 400 | 5812 | 5812 | high | Summary-only tree ensembles; no temporal deep model, CPU-friendly but less direction-consistent. |

## Best By Metric

| criterion | pipeline | experiment_id | metric | metric_value | fall_f1 | fall_precision | fall_recall | FP | direction_macro_f1 | E2E_Direction_Macro_F1 | edge_suitability | estimated_complexity_units | avg_fall_f1 | avg_direction_macro_f1 | avg_e2e_direction_macro_f1 | deep_params | total_tree_count | total_tree_nodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_weda_fall_f1 | Hybrid_NoTiming | E7_E3_MIXED_TEST_WEDA | fall_f1 | 0.8364 | 0.8364 | 0.7667 | 0.9200 | 7 | 0.8390 | 0.7926 | medium | 67367 |  |  |  |  |  |  |
| best_weda_direction_macro_f1 | ML_only_separated | E7_E3_MIXED_TEST_WEDA | direction_macro_f1 | 0.8851 | 0.8364 | 0.7667 | 0.9200 | 7 | 0.8851 | 0.8410 | high | 5812 |  |  |  |  |  |  |
| best_weda_e2e_direction_macro_f1 | ML_only_separated | E7_E3_MIXED_TEST_WEDA | E2E_Direction_Macro_F1 | 0.8410 | 0.8364 | 0.7667 | 0.9200 | 7 | 0.8851 | 0.8410 | high | 5812 |  |  |  |  |  |  |
| best_balanced_e3_e6_e7_score | Hybrid_NoTiming |  | balanced_score | 0.8568 |  |  |  |  |  |  |  |  | 0.8687 | 0.8902 | 0.7908 |  |  |  |
| lightest_estimated_complexity | ML_only_separated |  | estimated_complexity_units | 5812.0000 |  |  |  |  |  |  | high |  |  |  |  | 0 | 400 | 5812 |

## Required Answers

1. Best overall practical pipeline: `Hybrid_NoTiming`. It gives the largest WEDA fall improvement while keeping A5 as the stable direction expert.
2. Best WEDA fall pipeline: `Hybrid_NoTiming` with E7 fall F1=0.8364, FP=7.
3. Best direction pipeline on average over E3/E6/E7: `A5_reference` with average direction macro F1=0.8902.
4. Best end-to-end direction pipeline on average over E3/E6/E7: `A5_reference` with average E2E macro F1=0.8314.
5. Lightest pipeline by estimated complexity units: `ML_only_separated`; edge suitability=high.
6. A5 remains the main deep temporal baseline and the direction expert even if Hybrid improves fall.
7. Yes, the A5 fall head can be removed from final fall inference in the practical hybrid; keep it as auxiliary/reference for reporting.
8. Yes, keep the A5 direction head. It is more reliable across the benchmark than ML-only direction.
9. ML-only separated should not replace A5 direction as main unless it wins consistently. Current E3/E6/E7 average direction macro F1: ML-only=0.8583, A5=0.8902.
10. EdgeLite_Hybrid is worth keeping as a compact ablation. It uses only the core fall summary features but still depends on A5 for direction.
11. FullTiming is not used as a main result. It remains upper-bound only.
12. Final recommended practical architecture: DS-Fall-RD-Hybrid-NoTiming = GradientBoosting FallNoTiming_Full Fall Expert + DS-Fall-RD A5 Direction Expert.

## A5 vs Hybrid WEDA Fall

- A5 E7 fall F1=0.6857, precision=0.5333, recall=0.9600, FP=21.
- Hybrid_NoTiming E7 fall F1=0.8364, precision=0.7667, recall=0.9200, FP=7.

## Limitations

- Pure cross-dataset transfer remains weaker than mixed BITS/WEDA training.
- WEDA hard-negative ADL behavior still explains much of the fall gap.
- ML-only direction can be strong on WEDA but is less stable than A5 across all protocols.
- FullTiming results should not be used as the paper main result.
