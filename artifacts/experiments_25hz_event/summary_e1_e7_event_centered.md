# Event-centered 25 Hz BITS/WEDA E1-E7 Summary

E6 and E7 reuse the E3 checkpoint trained on BITS+WEDA and evaluate the
same test predictions separately for BITS and WEDA. They are not new
training runs.

## Table 1. Main Results

| experiment_id | train_dataset | test_dataset | fall_f1 | direction_macro_f1 | direction_accuracy | direction_n_supervised | model_params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1_BITS_TO_BITS | bits | bits | 0.8889 | 0.8777 | 0.8889 | 36 | 65959 |
| E2_WEDA_TO_WEDA | weda | weda | 0.7755 | 0.6071 | 0.6400 | 25 | 65959 |
| E3_BITS_WEDA_MIXED | bits+weda | bits+weda | 0.8323 | 0.8967 | 0.9016 | 61 | 65959 |
| E4_BITS_TO_WEDA | bits | weda | 0.5000 | 0.3876 | 0.4000 | 25 | 65959 |
| E5_WEDA_TO_BITS | weda | bits | 0.7073 | 0.6271 | 0.6667 | 36 | 65959 |
| E6_E3_MIXED_TEST_BITS | bits+weda | bits | 0.9451 | 0.9350 | 0.9444 | 36 | 65959 |
| E7_E3_MIXED_TEST_WEDA | bits+weda | weda | 0.6857 | 0.8390 | 0.8400 | 25 | 65959 |

## Table 2. Dataset-specific vs E3 Model Per-dataset Test

| dataset | dataset_specific_fall_f1 | e3_model_test_fall_f1 | fall_delta | dataset_specific_direction_f1 | e3_model_test_direction_f1 | direction_delta |
| --- | --- | --- | --- | --- | --- | --- |
| bits | 0.8889 | 0.9451 | 0.0562 | 0.8777 | 0.9350 | 0.0573 |
| weda | 0.7755 | 0.6857 | -0.0898 | 0.6071 | 0.8390 | 0.2319 |

## Table 3. Pure Transfer vs E3 Model Per-dataset Test

| test_dataset | pure_transfer_exp | pure_transfer_fall_f1 | e3_model_test_fall_f1 | fall_delta | pure_transfer_direction_f1 | e3_model_test_direction_f1 | direction_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bits | E5_WEDA_TO_BITS | 0.7073 | 0.9451 | 0.2377 | 0.6271 | 0.9350 | 0.3079 |
| weda | E4_BITS_TO_WEDA | 0.5000 | 0.6857 | 0.1857 | 0.3876 | 0.8390 | 0.4514 |

## Table 4. Direction Per-class F1

| experiment_id | test_dataset | forward_f1 | backward_f1 | lateral_f1 |
| --- | --- | --- | --- | --- |
| E1_BITS_TO_BITS | bits |  |  |  |
| E2_WEDA_TO_WEDA | weda |  |  |  |
| E3_BITS_WEDA_MIXED | bits+weda |  |  |  |
| E4_BITS_TO_WEDA | weda |  |  |  |
| E5_WEDA_TO_BITS | bits |  |  |  |
| E6_E3_MIXED_TEST_BITS | bits | 0.9730 | 0.9231 | 0.9091 |
| E7_E3_MIXED_TEST_WEDA | weda | 0.8421 | 0.8750 | 0.8000 |

## Insights

- The E3 mixed BITS+WEDA model is strong on both test subsets: BITS direction macro F1 = 0.9350, WEDA direction macro F1 = 0.8390. The overall E3 score is therefore not only a BITS effect.
- Mixed training improves direction over dataset-specific training on both datasets: BITS +0.0573, WEDA +0.2319. This supports using BITS+WEDA as the clean-direction benchmark.
- WEDA fall detection drops under the mixed model compared with WEDA-only (0.7755 to 0.6857), while WEDA direction improves strongly. This suggests the fall and direction heads are affected differently by cross-dataset mixing.
- Pure cross-dataset transfer is still weak: BITS->WEDA direction macro F1 = 0.3876, WEDA->BITS = 0.6271. The clean setup works best when both datasets are represented in training.
- Direction metrics are based on small supervised test counts (BITS n=36, WEDA n=25), so report these counts with every paper table.

## Diagnosis

1. Train BITS+WEDA then test each dataset separately generalizes well for direction on both datasets.
2. BITS is easier for fall detection under the mixed model; WEDA remains harder for fall detection even though direction improves.
3. Sampling-rate normalization to 25 Hz is not the main blocker inside the clean BITS/WEDA setup; pure domain transfer remains the larger issue.
4. BITS+WEDA is suitable as a clean-direction benchmark, but leave-one-dataset transfer should be reported separately as a domain-shift stress test.
