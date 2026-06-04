# DS-Fall-RD Compact Hybrid Model Card

## Model

- Name: DS-Fall-RD Compact Hybrid
- Fall: Statistical Fall Expert, `GradientBoostingClassifier`, event-window statistical features.
- Direction: LDX1 Wide-DSConv Direction Expert, temporal `tilt12` input `50 x 12`.
- FullTiming/timing-index artifacts: not used.

## Selected Direction Seed

- Selection mode: `paper_safe`.
- Seed: `12`.
- Learning rate: `0.0005`.

## Final Metrics

### Fall
| experiment_id | fall_precision | fall_recall | fall_f1 | fall_accuracy | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| E3_BITS_WEDA_MIXED | 0.8986 | 0.8493 | 0.8732 | 0.9746 | 7 | 11 |
| E6_E3_MIXED_TEST_BITS | 1.0000 | 0.8125 | 0.8966 | 0.9830 | 0 | 9 |
| E7_E3_MIXED_TEST_WEDA | 0.7667 | 0.9200 | 0.8364 | 0.9500 | 7 | 2 |

### Direction
| experiment_id | direction_macro_f1 | direction_accuracy | forward_f1 | backward_f1 | lateral_f1 | direction_n_supervised |
| --- | --- | --- | --- | --- | --- | --- |
| E3_BITS_WEDA_MIXED | 0.8685 | 0.8852 | 0.9310 | 0.7857 | 0.8889 | 61 |
| E6_E3_MIXED_TEST_BITS | 0.8579 | 0.8889 | 0.9474 | 0.7692 | 0.8571 | 36 |
| E7_E3_MIXED_TEST_WEDA | 0.8778 | 0.8800 | 0.9000 | 0.8000 | 0.9333 | 25 |

### End-to-End Direction
| experiment_id | E2E_Direction_Macro_F1 | coverage | correct_rate | direction_n_supervised |
| --- | --- | --- | --- | --- |
| E3_BITS_WEDA_MIXED | 0.7641 | 0.8525 | 0.7377 | 61 |
| E6_E3_MIXED_TEST_BITS | 0.6857 | 0.8056 | 0.6944 | 36 |
| E7_E3_MIXED_TEST_WEDA | 0.8299 | 0.9200 | 0.8000 | 25 |

## Artifacts

- Fall expert: `C:\Users\ADMIN\Desktop\ds-fall\outputs\models\final_compact_hybrid\statistical_fall_expert.pkl`
- Direction expert: `C:\Users\ADMIN\Desktop\ds-fall\outputs\models\final_compact_hybrid\ldx1_wide_dsconv_direction.pt`
- Config: `outputs/models/final_compact_hybrid/final_config.json`

## Limitations

- TensorFlow/TFLite export is unavailable in this environment if TensorFlow import fails.
- If `best_observed` seed selection is used, it is optimistic and should be reported as analysis only.
