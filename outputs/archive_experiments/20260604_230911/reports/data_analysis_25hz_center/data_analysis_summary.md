# Data Analysis: BITS vs WEDA Fall Gap at 25 Hz Event-centered

Scope: BITS/WEDA only, 25 Hz, 2-second event-centered windows, tilt12 features. This is data analysis only; no DS-Fall architecture experiment was run.

## Prediction Source

Loaded saved reference predictions from:
- C:\Users\ADMIN\Desktop\ds-fall\artifacts\experiments_25hz_event\E3_BITS_WEDA_MIXED\predictions.csv
- C:\Users\ADMIN\Desktop\ds-fall\artifacts\experiments_25hz_event\E6_E3_MIXED_TEST_BITS\predictions.csv
- C:\Users\ADMIN\Desktop\ds-fall\artifacts\experiments_25hz_event\E7_E3_MIXED_TEST_WEDA\predictions.csv
Saved predictions cover all test windows. 4747 train/val windows have no saved prediction, which is expected because the reference artifact stores test predictions.

## Reference Fall Metrics Recomputed From Saved Predictions

| dataset | n | TN | FP | FN | TP | precision | recall | fall_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bits | 528 | 480 | 0 | 5 | 43 | 1.0 | 0.8958333333333334 | 0.945054945054945 |
| weda | 180 | 134 | 21 | 1 | 24 | 0.5333333333333333 | 0.96 | 0.6857142857142857 |

## Top Variables

Top fall/non-fall separators on BITS:

| feature | abs_auc | cohens_d | ks_statistic | median_pos | median_neg |
| --- | --- | --- | --- | --- | --- |
| az_range | 0.9806 | 4.4637 | 0.9021 | 36.2374 | 0.6814 |
| az_std | 0.9799 | 5.0212 | 0.8896 | 7.2822 | 0.1764 |
| tilt_delta_std | 0.9690 | 3.6903 | 0.8917 | 4.9495 | 0.1359 |
| tilt_delta_max | 0.9681 | 3.0522 | 0.9000 | 12.3501 | 0.3071 |
| tilt_delta_p95 | 0.9662 | 3.6410 | 0.8958 | 7.6162 | 0.1962 |
| jerk_max | 0.9628 | 3.5798 | 0.8688 | 449.8027 | 4.1393 |
| jerk_std | 0.9619 | 3.8004 | 0.8604 | 152.4145 | 1.6467 |
| impact_distance_from_center | 0.9604 | -1.6694 | 0.9333 | 0.0000 | 13.0000 |
| ax_range | 0.9573 | 3.6041 | 0.8354 | 58.1939 | 0.7623 |
| acc_mag_range | 0.9569 | 3.6050 | 0.8542 | 57.5254 | 0.6465 |

Top fall/non-fall separators on WEDA:

| feature | abs_auc | cohens_d | ks_statistic | median_pos | median_neg |
| --- | --- | --- | --- | --- | --- |
| impact_distance_from_center | 0.9871 | -1.8144 | 0.9742 | 0.0000 | 13.0000 |
| acc_mag_range | 0.9561 | 2.3580 | 0.7974 | 36.7186 | 10.9844 |
| acc_mag_max | 0.9399 | 2.2340 | 0.7794 | 39.2966 | 16.2864 |
| post_acc_mag_std | 0.9383 | 2.2161 | 0.8129 | 6.3173 | 1.1931 |
| gy_min | 0.9334 | -2.4288 | 0.7406 | -6.5231 | -1.5659 |
| pitch_final_initial | 0.9285 | -1.7951 | 0.7716 | -1.1327 | 0.0203 |
| az_range | 0.9254 | 2.1852 | 0.7510 | 41.2737 | 12.1362 |
| ax_range | 0.9226 | 1.9954 | 0.7677 | 37.6526 | 10.6295 |
| acc_mag_std | 0.9169 | 2.0233 | 0.8000 | 6.6695 | 1.9130 |
| post_gyro_mag_std | 0.9164 | 1.9193 | 0.7677 | 2.4532 | 0.6484 |

Largest BITS-fall vs WEDA-fall shifts:

| feature | abs_auc | cohens_d | ks_statistic | median_pos | median_neg |
| --- | --- | --- | --- | --- | --- |
| pitch_final_initial | 0.8708 | 1.5462 | 0.7533 | 0.0276 | -1.1327 |
| gy_min | 0.8525 | 1.4043 | 0.6333 | -2.9800 | -6.5231 |
| post_gyro_mag_std | 0.8283 | -1.3033 | 0.5742 | 1.4527 | 2.4532 |
| gy_range | 0.8100 | -1.2515 | 0.5033 | 6.2162 | 10.3973 |
| pre_impact_energy | 0.8058 | 0.8679 | 0.5258 | 242.9088 | 118.8526 |
| pre_post_energy_ratio | 0.7658 | 0.7740 | 0.4333 | 1.1435 | 0.6563 |
| gyro_mag_range | 0.7617 | -1.0156 | 0.4650 | 7.9181 | 10.6115 |
| tilt_delta_mean | 0.7608 | 0.8614 | 0.5325 | 0.2119 | -0.1378 |
| gyro_mag_max | 0.7358 | -0.9248 | 0.4442 | 8.4567 | 10.7777 |
| pitch_std | 0.7342 | -0.8104 | 0.5242 | 0.4281 | 0.5777 |

Largest BITS-non-fall vs WEDA-non-fall shifts:

| feature | abs_auc | cohens_d | ks_statistic | median_pos | median_neg |
| --- | --- | --- | --- | --- | --- |
| tilt_delta_max | 0.9026 | -1.5350 | 0.6985 | 0.3071 | 7.6781 |
| az_range | 0.8989 | -1.9927 | 0.6571 | 0.6814 | 12.1362 |
| tilt_delta_std | 0.8984 | -1.6987 | 0.6720 | 0.1359 | 2.6434 |
| tilt_delta_p95 | 0.8869 | -1.4748 | 0.6797 | 0.1962 | 3.3147 |
| az_max | 0.8845 | -1.8963 | 0.6341 | 3.0299 | 10.8507 |
| az_std | 0.8840 | -1.8719 | 0.6331 | 0.1764 | 2.5895 |
| gy_max | 0.8727 | -1.6741 | 0.6372 | 0.1608 | 2.1180 |
| jerk_max | 0.8685 | -1.1313 | 0.6278 | 4.1393 | 105.3204 |
| gy_range | 0.8625 | -1.5590 | 0.5964 | 0.2572 | 3.5015 |
| jerk_std | 0.8596 | -1.0781 | 0.5916 | 1.6467 | 31.3085 |

## WEDA Error Analysis

WEDA fall F1 is mainly limited by false positives: FP=21, FN=1, TP=24, TN=134. Recall is high (0.9600), but precision is low (0.5333).

Strongest WEDA FP vs TN feature differences:

| feature | n_pos | n_neg | median_pos | median_neg | median_delta | ks_statistic |
| --- | --- | --- | --- | --- | --- | --- |
| ay_std | 21 | 134 | 7.0736 | 1.5092 | 5.5643 | 0.8703 |
| ay_range | 21 | 134 | 31.1862 | 7.5532 | 23.6330 | 0.8507 |
| jerk_p95 | 21 | 134 | 185.8205 | 29.6866 | 156.1339 | 0.7825 |
| acc_mag_std | 21 | 134 | 6.4709 | 1.7406 | 4.7303 | 0.7825 |
| jerk_std | 21 | 134 | 115.7873 | 26.4610 | 89.3263 | 0.7676 |
| gyro_mag_max | 21 | 134 | 10.4398 | 3.9785 | 6.4613 | 0.7463 |
| gy_range | 21 | 134 | 10.2622 | 3.2640 | 6.9982 | 0.7377 |
| acc_mag_range | 21 | 134 | 30.2020 | 10.1548 | 20.0471 | 0.7228 |
| gyro_mag_range | 21 | 134 | 9.6955 | 3.7298 | 5.9657 | 0.7164 |
| az_std | 21 | 134 | 6.0405 | 2.3641 | 3.6764 | 0.7154 |

WEDA TP vs FN feature differences. Interpret cautiously if FN count is very small:

| feature | n_pos | n_neg | median_pos | median_neg | median_delta | ks_statistic |
| --- | --- | --- | --- | --- | --- | --- |
| gyro_mag_max | 24 | 1 | 10.6738 | 21.0859 | -10.4121 | 1.0000 |
| az_min | 24 | 1 | -16.1305 | -3.7777 | -12.3529 | 1.0000 |
| jerk_max | 24 | 1 | 321.0923 | 180.5391 | 140.5533 | 1.0000 |
| az_range | 24 | 1 | 41.4651 | 14.7348 | 26.7303 | 1.0000 |
| gyro_mag_range | 24 | 1 | 10.5281 | 20.9444 | -10.4163 | 1.0000 |
| gx_range | 24 | 1 | 10.7143 | 18.6494 | -7.9351 | 0.9583 |
| tilt_delta_std | 24 | 1 | 6.3084 | 4.0824 | 2.2260 | 0.9583 |
| ay_max | 24 | 1 | 13.3961 | 4.8250 | 8.5711 | 0.9583 |
| post_acc_mag_std | 24 | 1 | 6.3381 | 3.9802 | 2.3580 | 0.9167 |
| acc_mag_std | 24 | 1 | 6.6724 | 4.9455 | 1.7269 | 0.9167 |

## Probability And Impact Alignment

| group | n | mean_fall_prob | median_fall_prob | p05 | p25 | p75 | p95 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bits_fall | 48 | 0.8914 | 1.0000 | 0.0001 | 0.9972 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| bits_nonfall | 480 | 0.0023 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0006 | 0.0000 | 0.3976 |
| weda_fall | 25 | 0.9609 | 1.0000 | 0.9564 | 0.9979 | 1.0000 | 1.0000 | 0.1304 | 1.0000 |
| weda_nonfall | 155 | 0.1311 | 0.0000 | 0.0000 | 0.0000 | 0.0018 | 0.9991 | 0.0000 | 1.0000 |
| weda_tp | 24 | 0.9955 | 1.0000 | 0.9689 | 0.9987 | 1.0000 | 1.0000 | 0.9541 | 1.0000 |
| weda_fn | 1 | 0.1304 | 0.1304 | 0.1304 | 0.1304 | 0.1304 | 0.1304 | 0.1304 | 0.1304 |
| weda_fp | 21 | 0.9173 | 0.9984 | 0.6406 | 0.8730 | 0.9998 | 1.0000 | 0.5509 | 1.0000 |
| weda_tn | 134 | 0.0079 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0288 | 0.0000 | 0.4558 |

| group | n | impact_index_mean | impact_index_median | impact_index_iqr | distance_from_center_median | distance_from_center_mean | percent_center_20_30 | percent_early_lt20 | percent_late_gt30 | acc_mag_peak_median | acc_mag_peak_iqr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bits_fall | 48 | 25.7083 | 25.0000 | 0.0000 | 0.0000 | 0.7083 | 95.8333 | 0.0000 | 4.1667 | 62.9667 | 51.9419 |
| weda_fall | 25 | 25.0000 | 25.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0000 | 0.0000 | 0.0000 | 39.2966 | 8.7388 |
| weda_tp | 24 | 25.0000 | 25.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0000 | 0.0000 | 0.0000 | 39.5916 | 8.3447 |
| weda_fn | 1 | 25.0000 | 25.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0000 | 0.0000 | 0.0000 | 30.5281 | 0.0000 |
| weda_nonfall | 155 | 26.4065 | 26.0000 | 24.5000 | 13.0000 | 13.0323 | 20.6452 | 34.8387 | 44.5161 | 16.2864 | 13.1094 |
| weda_fp | 21 | 26.0000 | 27.0000 | 21.0000 | 11.0000 | 11.7619 | 23.8095 | 28.5714 | 47.6190 | 33.4114 | 14.0623 |

## Classical Feature-importance Models

Classical models are trained only on summary features using the existing train/val/test split. Thresholds are tuned on validation for F1, then evaluated on test.

| model | feature | importance | permutation_importance_mean | test_auroc | test_f1 | bits_test_f1 | weda_test_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | gyro_mag_mean | 2.4857 | 0.1181 | 0.9829 | 0.8824 | 0.8864 | 0.8750 |
| hist_gradient_boosting | impact_distance_from_center |  | 0.1087 | 0.9951 | 0.9517 | 0.9462 | 0.9615 |
| gradient_boosting | impact_distance_from_center | 0.8416 | 0.0952 | 0.9882 | 0.9241 | 0.9149 | 0.9412 |
| logistic_regression | impact_distance_from_center | 2.0534 | 0.0570 | 0.9829 | 0.8824 | 0.8864 | 0.8750 |
| logistic_regression | acc_mag_p95 | 1.9194 | 0.0503 | 0.9829 | 0.8824 | 0.8864 | 0.8750 |
| logistic_regression | az_std | 1.6811 | 0.0418 | 0.9829 | 0.8824 | 0.8864 | 0.8750 |
| logistic_regression | tilt_delta_max | 1.3207 | 0.0224 | 0.9829 | 0.8824 | 0.8864 | 0.8750 |
| random_forest | impact_distance_from_center | 0.1808 | 0.0168 | 0.9901 | 0.9459 | 0.9474 | 0.9434 |
| logistic_regression | ay_std | 0.9547 | 0.0104 | 0.9829 | 0.8824 | 0.8864 | 0.8750 |
| logistic_regression | tilt_delta_std | 1.8046 | 0.0087 | 0.9829 | 0.8824 | 0.8864 | 0.8750 |
| logistic_regression | gx_std | 1.5522 | 0.0085 | 0.9829 | 0.8824 | 0.8864 | 0.8750 |
| logistic_regression | gz_std | 1.8481 | 0.0066 | 0.9829 | 0.8824 | 0.8864 | 0.8750 |
| logistic_regression | ax_max | 0.6080 | 0.0064 | 0.9829 | 0.8824 | 0.8864 | 0.8750 |
| gradient_boosting | tilt_delta_std | 0.0022 | 0.0064 | 0.9882 | 0.9241 | 0.9149 | 0.9412 |
| logistic_regression | roll_mean | 0.5422 | 0.0058 | 0.9829 | 0.8824 | 0.8864 | 0.8750 |

## Diagnosis

- Top variables affecting fall separation on BITS are: az_range, az_std, tilt_delta_std, tilt_delta_max, tilt_delta_p95.
- Top variables affecting fall separation on WEDA are: impact_distance_from_center, acc_mag_range, acc_mag_max, post_acc_mag_std, gy_min.
- Features with the strongest BITS-fall vs WEDA-fall shift are: pitch_final_initial, gy_min, post_gyro_mag_std, gy_range, pre_impact_energy. This indicates that the fall class itself is not identically distributed across datasets.
- Features with the strongest BITS-non-fall vs WEDA-non-fall shift are: tilt_delta_max, az_range, tilt_delta_std, tilt_delta_p95, az_max. This is evidence that WEDA ADL/non-fall windows have a different motion distribution.
- WEDA Fall F1 is lower mainly because of false positives, not false negatives: FP=21, FN=1.
- WEDA FP differs from WEDA TN most on: ay_std, ay_range, jerk_p95, acc_mag_std, jerk_std. These are likely hard-negative ADL windows that resemble fall-like high-motion segments.
- WEDA FN median fall probability is 0.1304; WEDA FP median fall probability is 0.9984; WEDA TN median fall probability is 0.0000.
- Median fall acc peak is BITS=62.9667, WEDA=39.2966. WEDA FP impact-center percentage is 23.81%, so some non-fall WEDA windows contain centered high-energy peaks that look event-like.
- WEDA direction can remain good while fall is weak because direction uses signed roll/pitch/gyro patterns on supervised fall windows, whereas fall precision is hurt by WEDA non-fall windows whose impact/jerk/energy statistics overlap with falls.

## Paper-facing Conclusion

On the BITS/WEDA 25 Hz event-centered benchmark, the lower WEDA fall F1 is not a direction-label failure. The E3 model keeps high WEDA fall recall (0.9600) but has low precision (0.5333), producing FP=21 and FN=1.
The data-level explanation is that WEDA non-fall windows are closer to fall windows in high-motion summary features, especially impact_distance_from_center, acc_mag_range, acc_mag_max, post_acc_mag_std, and WEDA non-fall distribution differs from BITS on tilt_delta_max, az_range, tilt_delta_std, tilt_delta_p95.
Impact alignment is not the primary issue for true fall windows: centered-impact percentage is BITS=95.83% and WEDA=100.00%. The harder part is WEDA hard-negative ADL windows that create fall-like impact/jerk evidence.
For the paper, report WEDA as a harder fall-precision subset in the clean BITS/WEDA benchmark, while noting that WEDA direction remains strong because signed orientation features remain separable on supervised fall windows.

## Output Files

- `univariate_fall_importance.csv`
- `weda_error_feature_comparison.csv`
- `fall_probability_distribution.csv`
- `impact_alignment_report.csv`
- `classical_fall_feature_importance.csv`
- `summary_feature_matrix_with_predictions.csv`
- figures under `outputs/figures/data_analysis_25hz_center/`
