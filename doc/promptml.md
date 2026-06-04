Bạn đang làm việc trong repo:

C:\Users\ADMIN\Desktop\ds-fall

Mục tiêu:
Tạo một phân tích machine-learning-based feature analysis đầy đủ cho 2 bài toán:
1. Fall detection: fall vs non-fall
2. Direction classification: forward vs backward vs lateral

Sau đó so sánh các classical ML feature-based models với deep learning model DS-Fall-RD reference trên cùng benchmark BITS/WEDA 25 Hz event-centered 2s.

Không thêm dataset mới.
Không chạy kiến trúc deep learning mới.
Không thay đổi benchmark.
Không sửa kết quả reference cũ.
Chỉ phân tích feature, train classical ML baselines, và tạo bảng so sánh khoa học.

Benchmark cố định:
- Dataset: BITS + WEDA only
- Sampling: 25 Hz
- Window: 2 seconds, event-centered
- Input temporal model: tilt12, shape 50 × 12
- Direction labels: forward, backward, lateral
- Direction chỉ tính trên fall windows có direction_supervised = True
- Split: dùng existing train/val/test split
- Deep model reference: REF_CURRENT_E3_ARTIFACT / DS-Fall-RD A5WCEFW

Đọc các file/output hiện có nếu tồn tại:
- outputs/reports/data_analysis_25hz_center/data_analysis_summary.md
- outputs/reports/data_analysis_25hz_center/summary_feature_matrix_with_predictions.csv
- outputs/reports/data_analysis_25hz_center/classical_fall_feature_importance.csv
- outputs/reports/data_analysis_25hz_center/univariate_fall_importance.csv
- outputs/reports/direction_feature_analysis_25hz/ nếu đã có
- artifacts/experiments_25hz_event/E3_BITS_WEDA_MIXED/predictions.csv
- artifacts/experiments_25hz_event/E6_E3_MIXED_TEST_BITS/predictions.csv
- artifacts/experiments_25hz_event/E7_E3_MIXED_TEST_WEDA/predictions.csv
- outputs/reports/targeted_experiments/best_model_selection.md nếu có
- outputs/reports/fp_reduction_experiments/summary.md nếu có

Nếu thiếu file direction analysis thì tự chạy lại direction feature analysis.
Nếu thiếu summary feature matrix thì tự sinh lại từ processed tilt12/raw6 data.

==================================================
PART A — SUMMARY FEATURE EXTRACTION
==================================================

Tạo hoặc load summary feature matrix cho từng window.

Feature groups cần có:

1. Raw signed acceleration:
- ax_mean/std/min/max/range/final_initial
- ay_mean/std/min/max/range/final_initial
- az_mean/std/min/max/range/final_initial

2. Raw signed gyroscope:
- gx_mean/std/min/max/range/final_initial
- gy_mean/std/min/max/range/final_initial
- gz_mean/std/min/max/range/final_initial

3. Magnitude and jerk:
- acc_mag_mean/std/min/max/p95/range
- gyro_mag_mean/std/min/max/p95/range
- jerk_mean/std/max/p95

4. Orientation:
- roll_mean/std/min/max/range/final_initial
- pitch_mean/std/min/max/range/final_initial
- tilt_delta_mean/std/max/p95/final

5. Impact/timing:
- impact_index
- impact_distance_from_center
- gyro_peak_index
- gyro_peak_distance_from_center
- acc_mag_peak_value
- gyro_mag_peak_value

6. Pre/post-impact:
- pre_impact_energy
- post_impact_energy
- pre_post_energy_ratio
- post_acc_mag_std
- post_gyro_mag_std
- pre_acc_mag_std
- pre_gyro_mag_std

7. Optional robust stats:
- median, IQR for acc_mag, gyro_mag, jerk, roll, pitch, tilt_delta

Mỗi row phải có metadata:
- dataset: bits/weda
- split: train/val/test
- subject_id nếu có
- fall_label
- direction_label
- direction_supervised
- prediction/probability của DS-Fall-RD reference nếu có:
  - fall_prob
  - fall_pred
  - direction_pred
  - direction_prob_* nếu có

Lưu:
outputs/reports/ml_feature_analysis_25hz/summary_feature_matrix.csv

==================================================
PART B — FALL DETECTION FEATURE ANALYSIS
==================================================

Bài toán:
fall_label: 0/1

Dữ liệu:
dùng tất cả windows có fall_label.

Chạy các classical ML models:
1. Logistic Regression
2. Linear SVM hoặc SVM-RBF nếu chạy được nhanh
3. Random Forest
4. Gradient Boosting
5. HistGradientBoosting
6. XGBoost nếu package có sẵn; nếu không thì bỏ qua và ghi rõ

Training:
- Train trên train split
- Tune threshold trên validation split để maximize Fall F1
- Evaluate trên test split
- Evaluate riêng:
  - E3 mixed test
  - E6 BITS test
  - E7 WEDA test

Metrics bắt buộc:
- AUROC
- Average Precision
- Accuracy
- Precision
- Recall
- Fall F1
- TN, FP, FN, TP
- BITS Fall F1
- WEDA Fall F1
- WEDA precision/recall
- WEDA FP count
- threshold

Feature importance:
- model-native importance nếu có
- permutation importance
- top 15 features cho mỗi model
- nếu Logistic Regression thì báo signed coefficient

Ablation feature groups:
Chạy thêm group ablation cho fall:
- raw_axis_only
- magnitude_jerk_only
- orientation_only
- impact_timing_only
- pre_post_only
- all_features
- all_features_no_impact_timing

Mục tiêu:
biết feature group nào làm fall detection tốt nhất, và kiểm tra impact_distance_from_center có tạo protocol artifact không.

Lưu:
- fall_ml_results.csv
- fall_feature_importance.csv
- fall_feature_group_ablation.csv
- fall_confusion_matrices/
- fall_roc_pr_curves nếu có

==================================================
PART C — DIRECTION CLASSIFICATION FEATURE ANALYSIS
==================================================

Bài toán:
direction_label ∈ {forward, backward, lateral}

Dữ liệu:
chỉ dùng windows:
- fall_label = 1
- direction_supervised = True
- direction_label in {forward, backward, lateral}

Không dùng non-fall, none, other.

Chạy classical ML models:
1. Multinomial Logistic Regression
2. Linear SVM hoặc SVM-RBF nếu chạy được nhanh
3. Random Forest
4. Gradient Boosting / HistGradientBoosting
5. XGBoost nếu package có sẵn; nếu không thì bỏ qua và ghi rõ

Training:
- Dùng train split direction-supervised fall windows
- Tune hyperparameters nhẹ trên validation nếu có đủ mẫu
- Evaluate trên test split
- Evaluate riêng:
  - E3 mixed direction test
  - E6 BITS direction test
  - E7 WEDA direction test

Metrics bắt buộc:
- Direction Macro F1
- Direction Accuracy
- Per-class F1: forward, backward, lateral
- Confusion matrix
- Direction n supervised
- BITS Direction Macro F1
- WEDA Direction Macro F1
- BITS per-class F1
- WEDA per-class F1 nếu đủ mẫu

Feature importance:
- model-native importance nếu có
- permutation importance
- top 15 features cho mỗi model
- nếu Logistic Regression thì báo class-wise coefficients

Univariate direction analysis:
- one-vs-rest:
  - forward vs rest
  - backward vs rest
  - lateral vs rest
- Cho từng feature tính:
  - AUROC
  - Cohen's d
  - KS statistic
  - median_target
  - median_rest
- Xuất top 20 features mỗi class

Dataset-shift direction analysis:
So sánh cùng một direction giữa BITS và WEDA:
- BITS forward vs WEDA forward
- BITS backward vs WEDA backward
- BITS lateral vs WEDA lateral
Tính:
- Cohen's d
- KS statistic
- median differences
- top shifted features

Feature group ablation cho direction:
- raw_axis_only
- gyro_signed_only
- acceleration_signed_only
- magnitude_jerk_only
- orientation_only
- pre_impact_signed_only
- all_features
- all_features_no_magnitude

Mục tiêu:
trả lời direction phụ thuộc nhiều vào signed roll/pitch/gyro hay magnitude/impact.

Lưu:
- direction_ml_results.csv
- direction_feature_importance.csv
- direction_univariate_ovr.csv
- direction_dataset_shift.csv
- direction_feature_group_ablation.csv
- direction_confusion_matrices/

==================================================
PART D — COMPARE CLASSICAL ML VS DS-FALL-RD DEEP MODEL
==================================================

Tạo bảng tổng hợp so sánh các model feature-based và DS-Fall-RD.

Deep reference metrics dùng từ saved predictions:
- REF_CURRENT_E3_ARTIFACT / DS-Fall-RD A5WCEFW
- E3 Fall F1 = 0.8323
- E3 Direction Macro F1 = 0.8967
- E6 BITS Fall F1 = 0.9451
- E6 BITS Direction Macro F1 = 0.9350
- E7 WEDA Fall F1 = 0.6857
- E7 WEDA Direction Macro F1 = 0.8390
- Params = 65959

Nếu có thể, recompute từ predictions để tránh hardcode. Nếu không, dùng số trên và ghi rõ source.

Bảng tổng hợp chính cần có:

1. Main comparison table:
Columns:
- model_name
- model_type: classical_feature / deep_temporal / hybrid_optional
- input_type: summary_features / temporal_tilt12
- task_supported: fall_only / direction_only / multitask_fall_direction
- params_or_model_size
- E3 Fall F1
- E3 Fall Precision
- E3 Fall Recall
- E3 Direction Macro F1
- E3 Direction Accuracy
- E6 BITS Fall F1
- E6 BITS Direction Macro F1
- E7 WEDA Fall F1
- E7 WEDA Precision
- E7 WEDA Recall
- E7 WEDA Direction Macro F1
- notes

2. Best fall-only model table:
Chọn classical model tốt nhất cho fall detection.
So sánh với DS-Fall-RD fall head.
Nhấn mạnh:
- classical model có thể tốt hơn fall detection nhờ summary features
- nhưng không giải quyết direction/multitask

3. Best direction-only model table:
Chọn classical model tốt nhất cho direction.
So sánh với DS-Fall-RD direction head.
Nhấn mạnh:
- nếu classical direction thấp hơn DS-Fall-RD, chứng minh temporal deep representation có ích
- nếu classical direction gần bằng, chỉ ra signed summary features rất mạnh

4. Feature interpretation table:
Rows:
- fall_top_features
- direction_top_features
- features_common_to_both
- fall_specific_features
- direction_specific_features
Columns:
- feature_group
- examples
- physical meaning
- paper interpretation

Expected interpretation:
- Fall detection likely depends on impact/timing/energy features:
  acc_mag_range, acc_mag_max, jerk_p95, post_acc_mag_std, impact_distance_from_center
- Direction likely depends on signed orientation/gyro features:
  pitch_final_initial, roll_final_initial, gy_min/gy_max, gx/gz ranges, pre-impact gyro and roll/pitch deltas
- Magnitude features help fall more than direction
- Signed roll/pitch/gyro features explain why WEDA direction can stay good even when WEDA fall precision is low

==================================================
PART E — FIGURES
==================================================

Create figures under:
outputs/figures/ml_feature_analysis_25hz/

Required figures:
1. Fall top feature importance bar plot
2. Direction top feature importance bar plot
3. Fall confusion matrix for best classical fall model
4. Direction confusion matrix for best classical direction model
5. DS-Fall-RD vs best ML fall model comparison bar chart
6. DS-Fall-RD vs best ML direction model comparison bar chart
7. Feature group ablation chart for fall
8. Feature group ablation chart for direction
9. Optional UMAP/t-SNE of direction-supervised summary features, colored by direction and dataset

==================================================
PART F — FINAL REPORT
==================================================

Write final report:
outputs/reports/ml_feature_analysis_25hz/ml_feature_analysis_summary.md

Report phải có các section:

1. Purpose
Giải thích đây là machine-learning-based feature analysis cho fall và direction, không phải deep model experiment.

2. Dataset and protocol
Nêu BITS/WEDA, 25 Hz, event-centered 2s, existing split.

3. Fall detection findings
Trả lời:
- Model cổ điển nào tốt nhất cho fall?
- Feature nào ảnh hưởng fall mạnh nhất?
- Vì sao WEDA Fall F1 của DS-Fall-RD thấp?
- Classical model có giảm được vấn đề WEDA FP không?
- Có dấu hiệu impact_distance_from_center là protocol artifact không? Báo cả result no-impact-timing.

4. Direction classification findings
Trả lời:
- Model cổ điển nào tốt nhất cho direction?
- Feature nào ảnh hưởng direction mạnh nhất?
- Direction phụ thuộc signed roll/pitch/gyro hay magnitude?
- BITS/WEDA có shift theo từng direction không?
- Vì sao WEDA direction vẫn tốt dù WEDA fall precision thấp?

5. Classical ML vs Deep DS-Fall-RD
Trả lời:
- Classical ML có thể thắng fall-only không?
- Classical ML có thắng direction-only không?
- DS-Fall-RD còn mạnh ở điểm nào?
- Có nên dùng classical model làm FPGuard/verifier không?

6. Paper-ready conclusion
Viết đoạn kết luận có thể đưa vào paper:
- Fall detection: summary impact/jerk/timing features rất mạnh, WEDA khó do hard-negative ADL.
- Direction: signed orientation/gyro features là chìa khóa, giúp direction vẫn ổn.
- DS-Fall-RD: main model tốt cho multitask direction-sensitive fall detection, classical ML hữu ích như diagnostic baseline và possible FPGuard.

7. Tables to include in paper
Liệt kê bảng nên đưa vào paper:
- ML vs Deep comparison
- Fall feature importance
- Direction feature importance
- Feature group ablation
- WEDA FP explanation table

Output files required:
- summary_feature_matrix.csv
- fall_ml_results.csv
- fall_feature_importance.csv
- fall_feature_group_ablation.csv
- direction_ml_results.csv
- direction_feature_importance.csv
- direction_univariate_ovr.csv
- direction_dataset_shift.csv
- direction_feature_group_ablation.csv
- ml_vs_deep_comparison.csv
- feature_interpretation_table.csv
- ml_feature_analysis_summary.md

Không chạy thêm deep architecture.
Không thêm dataset.
Không dùng test set để tune threshold/hyperparameter.
Nếu một model/package không có sẵn, bỏ qua model đó và ghi rõ trong report.