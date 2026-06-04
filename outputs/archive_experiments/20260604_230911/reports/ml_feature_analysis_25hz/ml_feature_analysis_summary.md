# ML Feature Analysis 25 Hz

## 1. Purpose

This is a machine-learning-based feature analysis for fall detection and direction classification. It is not a new deep architecture experiment.

## 2. Dataset And Protocol

Benchmark: BITS + WEDA only, 25 Hz, 2-second event-centered windows, existing train/val/test split, summary features derived from tilt12/raw6. Direction metrics use only supervised fall windows with labels forward/backward/lateral.

Classical ML feature columns are signal-derived summaries only. Metadata/protocol columns such as dataset, source_hz, start_idx, timestamps, labels, and reference predictions are retained in `summary_feature_matrix.csv` for audit but excluded from ML training.

XGBoost was not installed in the active Python environment and was skipped per prompt.

## 3. Fall Detection Findings

- Best classical fall model: `random_forest` with E3 Fall F1=0.9459, E7 WEDA Fall F1=0.9434, WEDA precision=0.8929, WEDA recall=1.0000.
- Top fall features: impact_distance_from_center, gyro_mag_p95, jerk_std, acc_mag_std, acc_mag_min, pre_ax_mean, acc_mag_max, acc_mag_range, tilt_delta_p95, acc_mag_peak_value.
- WEDA DS-Fall-RD Fall F1 is low mainly because WEDA non-fall contains high-motion hard negatives, which increases false positives.
- Classical fall-only models can reduce/avoid this issue because they directly exploit compact summary features, but they do not solve direction or multitask learning.
- Impact artifact check: all_features E3 Fall F1=0.9459 and no-impact-timing E3 Fall F1=0.8333.

## 4. Direction Classification Findings

- Best classical direction model: `hist_gradient_boosting` with E3 Direction Macro F1=0.8596, E7 WEDA Direction Macro F1=0.8338.
- Top direction features: gx_median, pitch_max, ax_range, ay_std, acc_mag_p95, gy_mean, gy_median, gz_median, gy_final_initial, ax_min.
- Best direction feature group on E3: `all_features_no_magnitude` with Macro F1=0.8654.
- Direction depends more on signed orientation/gyro/axis summaries than on pure magnitude. This matches DS-Fall-RD's rotation-aware design.
- BITS/WEDA direction shift exists; top shifted rows are shown below.

| direction | feature | n_bits | n_weda | bits_median | weda_median | median_delta_bits_minus_weda | ks_statistic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| backward | tilt_delta_std | 41 | 126 | 2.6788 | 6.5737 | -3.8949 | 0.8316 |
| backward | ax_min | 41 | 126 | -3.4994 | -22.9737 | 19.4743 | 0.8072 |
| backward | pitch_range | 41 | 126 | 1.4286 | 2.5043 | -1.0756 | 0.7993 |
| backward | pitch_max | 41 | 126 | 0.3644 | 1.3847 | -1.0203 | 0.7981 |
| backward | jerk_std | 41 | 126 | 43.1864 | 121.4962 | -78.3097 | 0.7737 |
| backward | tilt_delta_p95 | 41 | 126 | 4.1448 | 10.6599 | -6.5151 | 0.7499 |
| backward | tilt_delta_iqr | 41 | 126 | 2.3597 | 4.9825 | -2.6227 | 0.7352 |
| backward | az_range | 41 | 126 | 14.8388 | 42.7150 | -27.8762 | 0.7346 |
| backward | gy_range | 41 | 126 | 4.3178 | 9.5480 | -5.2302 | 0.6949 |
| forward | pitch_final_initial | 121 | 126 | 0.2472 | -1.0055 | 1.2527 | 0.6915 |

## 5. Classical ML vs Deep DS-Fall-RD

| model_name | model_type | input_type | task_supported | params_or_model_size | E3 Fall F1 | E3 Fall Precision | E3 Fall Recall | E3 Direction Macro F1 | E3 Direction Accuracy | E6 BITS Fall F1 | E6 BITS Direction Macro F1 | E7 WEDA Fall F1 | E7 WEDA Precision | E7 WEDA Recall | E7 WEDA Direction Macro F1 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DS-Fall-RD A5WCEFW | deep_temporal | temporal_tilt12 | multitask_fall_direction | 65959 | 0.8322981366459627 | 0.7613636363636364 | 0.9178082191780822 | 0.8966626725247414 | 0.9016393442622951 | 0.945054945054945 | 0.9350469350469351 | 0.6857142857142857 | 0.5333333333333333 | 0.96 | 0.8390350877192981 | reference saved predictions |
| gradient_boosting | classical_feature | summary_features | fall_only |  | 0.9379310344827586 | 0.9444444444444444 | 0.9315068493150684 |  |  | 0.9247311827956989 |  | 0.9615384615384616 | 0.9259259259259259 | 1.0 |  | fall-only summary feature classifier |
| hist_gradient_boosting | classical_feature | summary_features | fall_only |  | 0.9395973154362416 | 0.9210526315789473 | 0.958904109589041 |  |  | 0.9375 |  | 0.9433962264150944 | 0.8928571428571429 | 1.0 |  | fall-only summary feature classifier |
| linear_svm | classical_feature | summary_features | fall_only |  | 0.8571428571428571 | 0.95 | 0.7808219178082192 |  |  | 0.8470588235294118 |  | 0.875 | 0.9130434782608695 | 0.84 |  | fall-only summary feature classifier |
| logistic_regression | classical_feature | summary_features | fall_only |  | 0.8396946564885496 | 0.9482758620689655 | 0.7534246575342466 |  |  | 0.8333333333333334 |  | 0.851063829787234 | 0.9090909090909091 | 0.8 |  | fall-only summary feature classifier |
| random_forest | classical_feature | summary_features | fall_only |  | 0.9459459459459459 | 0.9333333333333333 | 0.958904109589041 |  |  | 0.9473684210526315 |  | 0.9433962264150944 | 0.8928571428571429 | 1.0 |  | fall-only summary feature classifier |
| gradient_boosting | classical_feature | summary_features | direction_only |  |  |  |  | 0.8200145958766648 | 0.8360655737704918 |  | 0.8051408051408052 |  |  |  | 0.8380952380952381 | direction-only summary feature classifier |
| hist_gradient_boosting | classical_feature | summary_features | direction_only |  |  |  |  | 0.8596491228070176 | 0.8688524590163934 |  | 0.8732563732563733 |  |  |  | 0.8337912087912088 | direction-only summary feature classifier |
| linear_svm | classical_feature | summary_features | direction_only |  |  |  |  | 0.7957219251336897 | 0.8032786885245902 |  | 0.7888888888888889 |  |  |  | 0.7619047619047619 | direction-only summary feature classifier |
| logistic_regression | classical_feature | summary_features | direction_only |  |  |  |  | 0.8275515334338864 | 0.8360655737704918 |  | 0.8074161117639379 |  |  |  | 0.8026418026418026 | direction-only summary feature classifier |
| random_forest | classical_feature | summary_features | direction_only |  |  |  |  | 0.809601034138316 | 0.819672131147541 |  | 0.8172932330827067 |  |  |  | 0.7949604743083004 | direction-only summary feature classifier |

Best fall-only comparison:

| model_name | model_type | input_type | task_supported | params_or_model_size | E3 Fall F1 | E3 Fall Precision | E3 Fall Recall | E3 Direction Macro F1 | E3 Direction Accuracy | E6 BITS Fall F1 | E6 BITS Direction Macro F1 | E7 WEDA Fall F1 | E7 WEDA Precision | E7 WEDA Recall | E7 WEDA Direction Macro F1 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DS-Fall-RD A5WCEFW | deep_temporal | temporal_tilt12 | multitask_fall_direction | 65959 | 0.8322981366459627 | 0.7613636363636364 | 0.9178082191780822 | 0.8966626725247414 | 0.9016393442622951 | 0.945054945054945 | 0.9350469350469351 | 0.6857142857142857 | 0.5333333333333333 | 0.96 | 0.8390350877192981 | reference saved predictions |
| random_forest | classical_feature | summary_features | fall_only |  | 0.9459459459459459 | 0.9333333333333333 | 0.958904109589041 |  |  | 0.9473684210526315 |  | 0.9433962264150944 | 0.8928571428571429 | 1.0 |  | fall-only summary feature classifier |
| random_forest | classical_feature | summary_features | direction_only |  |  |  |  | 0.809601034138316 | 0.819672131147541 |  | 0.8172932330827067 |  |  |  | 0.7949604743083004 | direction-only summary feature classifier |

Best direction-only comparison:

| model_name | model_type | input_type | task_supported | params_or_model_size | E3 Fall F1 | E3 Fall Precision | E3 Fall Recall | E3 Direction Macro F1 | E3 Direction Accuracy | E6 BITS Fall F1 | E6 BITS Direction Macro F1 | E7 WEDA Fall F1 | E7 WEDA Precision | E7 WEDA Recall | E7 WEDA Direction Macro F1 | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DS-Fall-RD A5WCEFW | deep_temporal | temporal_tilt12 | multitask_fall_direction | 65959 | 0.8322981366459627 | 0.7613636363636364 | 0.9178082191780822 | 0.8966626725247414 | 0.9016393442622951 | 0.945054945054945 | 0.9350469350469351 | 0.6857142857142857 | 0.5333333333333333 | 0.96 | 0.8390350877192981 | reference saved predictions |
| hist_gradient_boosting | classical_feature | summary_features | fall_only |  | 0.9395973154362416 | 0.9210526315789473 | 0.958904109589041 |  |  | 0.9375 |  | 0.9433962264150944 | 0.8928571428571429 | 1.0 |  | fall-only summary feature classifier |
| hist_gradient_boosting | classical_feature | summary_features | direction_only |  |  |  |  | 0.8596491228070176 | 0.8688524590163934 |  | 0.8732563732563733 |  |  |  | 0.8337912087912088 | direction-only summary feature classifier |

- Classical ML can beat or approach the deep model on fall-only metrics because impact/jerk/timing summaries are highly discriminative.
- DS-Fall-RD remains stronger as a single multitask temporal model because it handles fall and direction jointly from sequence input.
- A classical FPGuard/verifier is plausible: use summary-feature ML to filter WEDA-like false positives while preserving DS-Fall-RD direction output.

## 6. Paper-ready Conclusion

Summary impact, jerk and timing features are very strong for fall detection, and WEDA is difficult because ADL/non-fall windows include hard-negative high-motion patterns. Direction classification is driven by signed orientation and gyroscope evidence such as roll/pitch transitions and signed axis dynamics. DS-Fall-RD remains the main model for direction-sensitive multitask fall detection, while classical ML baselines provide diagnostic evidence and a possible FPGuard verifier for reducing false positives.

## 7. Tables To Include In Paper

- ML vs Deep comparison
- Fall feature importance
- Direction feature importance
- Feature group ablation
- WEDA FP explanation table

## Feature Interpretation Table

| feature_group | examples | physical meaning | paper interpretation |
| --- | --- | --- | --- |
| fall_top_features | impact_distance_from_center, gyro_mag_p95, jerk_std, acc_mag_std, acc_mag_min, pre_ax_mean, acc_mag_max, acc_mag_range, tilt_delta_p95, acc_mag_peak_value | impact strength, abrupt acceleration, event timing, post-impact instability | fall detection is dominated by impact/jerk/timing evidence |
| direction_top_features | gx_median, pitch_max, ax_range, ay_std, acc_mag_p95, gy_mean, gy_median, gz_median, gy_final_initial, ax_min | signed axis motion, roll/pitch transition, gyroscope pattern | direction depends on signed orientation and rotation evidence |
| features_common_to_both | az_min | features useful for both event detection and motion geometry | some summary features bridge fall and direction but do not replace temporal multitask learning |
| fall_specific_features | impact_distance_from_center, gyro_mag_p95, acc_mag_std, acc_mag_min, pre_ax_mean, acc_mag_range, tilt_delta_p95, acc_mag_peak_value, az_std, gyro_mag_range | magnitude, jerk, impact timing | classical fall-only classifiers can exploit compact impact summaries |
| direction_specific_features | gx_median, pitch_max, ay_std, acc_mag_p95, gy_mean, gy_median, gz_median, gy_final_initial, ax_min, post_roll_mean | signed roll/pitch/gyro differences | rotation-aware signed features explain direction robustness |

## Output Files

- `summary_feature_matrix.csv`
- `fall_ml_results.csv`
- `fall_feature_importance.csv`
- `fall_feature_group_ablation.csv`
- `direction_ml_results.csv`
- `direction_feature_importance.csv`
- `direction_univariate_ovr.csv`
- `direction_dataset_shift.csv`
- `direction_feature_group_ablation.csv`
- `ml_vs_deep_comparison.csv`
- `feature_interpretation_table.csv`
