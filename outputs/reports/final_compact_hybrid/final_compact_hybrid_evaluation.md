# Final Compact Hybrid Evaluation

This report is regenerated from saved artifacts, not from model retraining.

## Artifact Config

- Model: `DS-Fall-RD Compact Hybrid`
- Selected seed: `12`
- Selected learning rate: `0.0005`
- Direction params: `5699`
- Direction input: `[50, 12]`
- Fall feature count: `132`
- Fall threshold: `0.484604`

## Fall Metrics

| experiment_id | test_dataset | fall_precision | fall_recall | fall_f1 | fall_accuracy | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E3_BITS_WEDA_MIXED | bits+weda | 0.8986 | 0.8493 | 0.8732 | 0.9746 | 628 | 7 | 11 | 62 |
| E6_E3_MIXED_TEST_BITS | bits | 1.0000 | 0.8125 | 0.8966 | 0.9830 | 480 | 0 | 9 | 39 |
| E7_E3_MIXED_TEST_WEDA | weda | 0.7667 | 0.9200 | 0.8364 | 0.9500 | 148 | 7 | 2 | 23 |

## Direction Metrics

| experiment_id | test_dataset | direction_macro_f1 | direction_accuracy | forward_f1 | backward_f1 | lateral_f1 | direction_n_supervised |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E3_BITS_WEDA_MIXED | bits+weda | 0.8685 | 0.8852 | 0.9310 | 0.7857 | 0.8889 | 61 |
| E6_E3_MIXED_TEST_BITS | bits | 0.8579 | 0.8889 | 0.9474 | 0.7692 | 0.8571 | 36 |
| E7_E3_MIXED_TEST_WEDA | weda | 0.8778 | 0.8800 | 0.9000 | 0.8000 | 0.9333 | 25 |

## End-to-End Metrics

| experiment_id | E2E_Direction_Macro_F1 | coverage | correct_rate | direction_n_supervised |
| --- | --- | --- | --- | --- |
| E3_BITS_WEDA_MIXED | 0.7641 | 0.8525 | 0.7377 | 61 |
| E6_E3_MIXED_TEST_BITS | 0.6857 | 0.8056 | 0.6944 | 36 |
| E7_E3_MIXED_TEST_WEDA | 0.8299 | 0.9200 | 0.8000 | 25 |

