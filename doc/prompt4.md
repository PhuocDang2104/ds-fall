Bạn đang làm việc trong repo:

C:\Users\ADMIN\Desktop\ds-fall

Mục tiêu:
Chạy một bộ thử nghiệm mới để kiểm tra 2 giả thuyết:

1. ML thuần cho fall detection:

   * Dùng ML Fall Expert tốt nhất.
   * Chạy đầy đủ protocol E1–E7.
   * Không dùng FullTiming / không dùng timing artifact.
   * Chỉ dùng các feature phục vụ fall detection.
   * Mục tiêu: xem nếu bỏ deep fall head và dùng ML thuần cho fall thì E1–E7 tốt đến đâu.

2. Direction-only model:

   * Bỏ fall head khỏi mục tiêu chính.
   * Chỉ dùng các feature quan trọng cho direction.
   * Train/evaluate E1–E7 cho direction classification.
   * Mục tiêu: xem nếu tập trung riêng vào direction thì classical ML hoặc direction-focused pipeline có đạt gần/vượt DS-Fall-RD A5 không.

Không thêm dataset.
Không đổi benchmark.
Không dùng FullTiming làm main result.
Không dùng test set để tune threshold/hyperparameter.
Không train deep architecture mới trong prompt này.
Chỉ chạy ML/classical feature-based experiments và so sánh với DS-Fall-RD A5 reference.

Benchmark cố định:

* Dataset: BITS + WEDA only
* Sampling: 25 Hz
* Window: 2-second event-centered
* Direction labels: forward / backward / lateral
* Direction chỉ tính trên fall windows có direction_supervised = True
* Dùng existing train/val/test split và existing E1–E7 protocol
* Reference model: DS-Fall-RD A5WCEFW / REF_CURRENT_E3_ARTIFACT

Đọc kỹ các file/output có sẵn:

* outputs/reports/ml_feature_analysis_25hz/ml_feature_analysis_summary.md
* outputs/reports/ml_feature_analysis_25hz/summary_feature_matrix.csv
* outputs/reports/ml_feature_analysis_25hz/ml_vs_deep_comparison.csv nếu có
* outputs/reports/ds_fall_rd_hybrid/hybrid_summary.md nếu có
* outputs/reports/ds_fall_rd_hybrid/hybrid_results.csv nếu có
* outputs/reports/data_analysis_25hz_center/data_analysis_summary.md
* outputs/reports/direction_feature_analysis_25hz/ nếu có
* artifacts/experiments_25hz_event/E1_BITS_TO_BITS/predictions.csv nếu có
* artifacts/experiments_25hz_event/E2_WEDA_TO_WEDA/predictions.csv nếu có
* artifacts/experiments_25hz_event/E3_BITS_WEDA_MIXED/predictions.csv
* artifacts/experiments_25hz_event/E4_BITS_TO_WEDA/predictions.csv nếu có
* artifacts/experiments_25hz_event/E5_WEDA_TO_BITS/predictions.csv nếu có
* artifacts/experiments_25hz_event/E6_E3_MIXED_TEST_BITS/predictions.csv
* artifacts/experiments_25hz_event/E7_E3_MIXED_TEST_WEDA/predictions.csv

Nếu thiếu summary_feature_matrix.csv thì tự sinh lại từ processed tilt12/raw6 data.
Nếu thiếu một số saved prediction của A5, chỉ dùng cho baseline comparison nếu có; ML E1–E7 vẫn phải tự chạy từ feature matrix.

==================================================
PHẦN 1 — TẠO / KIỂM TRA FEATURE MATRIX
======================================

Tạo hoặc load summary_feature_matrix.csv với metadata đầy đủ:

Required metadata:

* dataset: bits / weda
* split: train / val / test
* subject_id nếu có
* fall_label
* direction_label
* direction_supervised
* experiment membership nếu có thể map E1–E7
* window_id hoặc index ổn định để trace lỗi
* existing A5 prediction/probability nếu có:

  * fall_prob
  * fall_pred
  * direction_pred
  * direction_prob_* nếu có

Không đưa metadata, label, prediction columns vào training features.

==================================================
PHẦN 2 — FEATURE SETS
=====================

A. FallNoTiming_Core
Chỉ dùng các feature fall quan trọng, không dùng timing artifact:

* acc_mag_range
* acc_mag_max
* acc_mag_p95
* acc_mag_std
* gyro_mag_range
* gyro_mag_max
* gyro_mag_p95
* gyro_mag_mean
* jerk_max
* jerk_p95
* jerk_std
* post_acc_mag_std
* post_gyro_mag_std
* pre_post_energy_ratio
* tilt_delta_p95
* tilt_delta_std

B. FallNoTiming_Full
Dùng tất cả signal-derived summary features nhưng loại bỏ timing:

* giữ raw axis stats, magnitude/jerk, roll/pitch/tilt, pre/post energy/stability
* loại bỏ:
  impact_index
  impact_distance_from_center
  gyro_peak_index
  gyro_peak_distance_from_center
  mọi feature có tên chứa distance_from_center, peak_index, impact_index nếu là timing cue

C. DirectionCore
Chỉ dùng feature quan trọng cho direction, ưu tiên signed axis/orientation/gyro:

* gx_mean, gx_median, gx_min, gx_max, gx_range, gx_std, gx_final_initial
* gy_mean, gy_median, gy_min, gy_max, gy_range, gy_std, gy_final_initial
* gz_mean, gz_median, gz_min, gz_max, gz_range, gz_std, gz_final_initial
* ax_mean, ax_min, ax_max, ax_range, ax_std, ax_final_initial
* ay_mean, ay_min, ay_max, ay_range, ay_std, ay_final_initial
* az_mean, az_min, az_max, az_range, az_std, az_final_initial
* roll_mean, roll_min, roll_max, roll_range, roll_std, roll_final_initial
* pitch_mean, pitch_min, pitch_max, pitch_range, pitch_std, pitch_final_initial
* tilt_delta_mean, tilt_delta_std, tilt_delta_max, tilt_delta_p95, tilt_delta_final
* pre_gx_mean, pre_gy_mean, pre_gz_mean nếu có
* pre_roll_delta, pre_pitch_delta nếu có
* pre_ax_mean, pre_ay_mean, pre_az_mean nếu có

D. DirectionNoMagnitude
Giống DirectionCore nhưng loại bỏ pure magnitude/impact features:

* không dùng acc_mag_*
* không dùng gyro_mag_*
* không dùng jerk_*
* không dùng timing features
  Mục tiêu: kiểm tra direction có thật sự dựa vào signed roll/pitch/gyro hơn là magnitude không.

E. DirectionFullNoTiming
Dùng tất cả signal-derived summary features nhưng loại bỏ timing artifact:

* loại bỏ impact_index, impact_distance_from_center, gyro_peak_index, gyro_peak_distance_from_center
* giữ signed axes, orientation, magnitude nếu không phải timing

Normalize:

* fit scaler trên train split của từng experiment only
* apply val/test
* không leak test

==================================================
PHẦN 3 — E1–E7 PROTOCOL
=======================

Chạy các protocol sau cho cả fall-only và direction-only:

E1_BITS_TO_BITS:

* train_dataset = bits
* test_dataset = bits

E2_WEDA_TO_WEDA:

* train_dataset = weda
* test_dataset = weda

E3_BITS_WEDA_MIXED:

* train_dataset = bits + weda
* test_dataset = bits + weda

E4_BITS_TO_WEDA:

* train_dataset = bits
* test_dataset = weda

E5_WEDA_TO_BITS:

* train_dataset = weda
* test_dataset = bits

E6_E3_MIXED_TEST_BITS:

* train_dataset = bits + weda
* test_dataset = bits only
* reuse model trained in E3 if possible

E7_E3_MIXED_TEST_WEDA:

* train_dataset = bits + weda
* test_dataset = weda only
* reuse model trained in E3 if possible

Nếu existing split đã có train/val/test per dataset, giữ đúng split.
Nếu phải tạo split, dùng subject-level split nếu metadata có subject_id. Không để subject leakage nếu có subject info.
Log rõ split strategy.

==================================================
PHẦN 4 — FALL-ONLY ML E1–E7
===========================

Task:
fall_label 0/1

Feature sets:

* FallNoTiming_Core
* FallNoTiming_Full

Models cần chạy:

1. LogisticRegression
2. LinearSVM nếu có probability/calibration; nếu không dùng decision_function + threshold
3. RandomForest
4. GradientBoosting
5. HistGradientBoosting

Không dùng FullTiming trong main result.

Training:

* train trên train split của protocol
* tune threshold trên validation split
* objective chính: maximize Fall F1
* objective phụ: nếu nhiều threshold gần nhau, ưu tiên precision cao hơn nhưng recall >= 0.88
* evaluate trên test split

Metrics bắt buộc:

* Fall Precision
* Fall Recall
* Fall F1
* Accuracy
* AUROC
* Average Precision
* TN, FP, FN, TP
* threshold
* model complexity:
  LR params, tree count/depth/nodes, feature count
* per-dataset breakdown cho E3/E6/E7
* FP reduction vs A5 nếu A5 baseline có sẵn

Output:

* fall_e1_e7_results.csv
* fall_e1_e7_confusion_matrices/
* fall_e1_e7_best_by_protocol.csv
* fall_e1_e7_feature_importance.csv

Câu hỏi cần trả lời:

* ML Fall Expert tốt nhất trên E1–E7 là model nào?
* FallNoTiming_Core có đủ tốt không, hay cần FallNoTiming_Full?
* Pure transfer E4/E5 có còn yếu không?
* ML fall-only có giải quyết WEDA FP tốt hơn A5 không?
* Có model nhẹ nào gần bằng HGB/GB không?

==================================================
PHẦN 5 — DIRECTION-ONLY ML E1–E7
================================

Task:
direction_label ∈ {forward, backward, lateral}

Data:
Chỉ dùng windows:

* fall_label = 1
* direction_supervised = True
* direction_label in {forward, backward, lateral}

Feature sets:

* DirectionCore
* DirectionNoMagnitude
* DirectionFullNoTiming

Models cần chạy:

1. Multinomial LogisticRegression
2. LinearSVM
3. RandomForest
4. GradientBoosting
5. HistGradientBoosting

Training:

* train trên direction-supervised fall windows trong train split
* tune nhẹ hyperparameter trên validation nếu đủ mẫu
* không dùng non-fall
* không dùng fall head
* không dùng timing artifact trong main result

Metrics bắt buộc:

* Direction Macro F1
* Direction Accuracy
* Per-class F1: forward, backward, lateral
* Confusion matrix
* direction_n_supervised
* per-dataset breakdown
* model complexity
* feature importance / permutation importance nếu có

Output:

* direction_e1_e7_results.csv
* direction_e1_e7_confusion_matrices/
* direction_e1_e7_best_by_protocol.csv
* direction_e1_e7_feature_importance.csv
* direction_e1_e7_dataset_shift_notes.csv

Câu hỏi cần trả lời:

* Nếu bỏ fall head và chỉ train direction-only ML thì E1–E7 direction đạt bao nhiêu?
* DirectionCore hay DirectionNoMagnitude có đủ tốt không?
* Magnitude features có giúp direction không?
* ML direction-only có vượt A5 direction head không?
* Pure transfer E4/E5 direction có còn yếu không?
* Forward/backward/lateral class nào khó nhất?
* Direction shift BITS/WEDA nằm ở feature nào?

==================================================
PHẦN 6 — SO SÁNH VỚI DS-FALL-RD A5
==================================

Tạo bảng so sánh:

A. Fall comparison E1–E7:
Columns:

* experiment_id
* model_name
* model_type
* feature_set
* fall_expert
* Fall Precision
* Fall Recall
* Fall F1
* FP
* FN
* AUROC
* AP
* complexity
* notes

Bao gồm:

* A5 reference nếu có saved predictions
* best ML FallNoTiming_Core
* best ML FallNoTiming_Full
* previous Hybrid HGB FullNoTiming nếu có
* previous Hybrid FullTiming chỉ để upper-bound, không làm main

B. Direction comparison E1–E7:
Columns:

* experiment_id
* model_name
* model_type
* feature_set
* direction_model
* Direction Macro F1
* Direction Accuracy
* forward_f1
* backward_f1
* lateral_f1
* direction_n_supervised
* complexity
* notes

Bao gồm:

* A5 reference direction nếu có saved predictions
* best ML DirectionCore
* best ML DirectionNoMagnitude
* best ML DirectionFullNoTiming

C. Final pipeline recommendation:
So sánh:

1. A5 reference:

   * fall = A5 deep fall head
   * direction = A5 direction head

2. ML-only separated:

   * fall = best ML FallNoTiming_Full
   * direction = best ML DirectionFullNoTiming hoặc DirectionCore

3. Hybrid recommended:

   * fall = best ML FallNoTiming_Full
   * direction = A5 direction head

4. Edge-lite:

   * fall = best lightweight ML FallNoTiming_Core
   * direction = A5 direction head

Metrics:

* E3 Fall F1
* E3 Direction F1
* E6 BITS Fall F1
* E6 BITS Direction F1
* E7 WEDA Fall F1
* E7 WEDA Precision/Recall
* E7 WEDA Direction F1
* E7 end-to-end direction F1 if applicable
* complexity
* paper suitability

==================================================
PHẦN 7 — END-TO-END DIRECTION
=============================

Với các pipeline có fall expert + direction model:
Tính end-to-end direction trên direction-supervised fall windows.

Một sample direction được tính là đúng nếu:

* final_fall = fall
* direction_pred = direction_true

Metrics:

* E3 E2E Direction Macro F1
* E6 E2E Direction Macro F1
* E7 E2E Direction Macro F1
* coverage = số supervised fall windows được final_fall giữ lại / tổng supervised fall windows
* correct_rate

Output:

* e2e_direction_results.csv

==================================================
PHẦN 8 — OUTPUT FOLDER
======================

Tạo:
outputs/reports/ml_e1_e7_specialized/
outputs/figures/ml_e1_e7_specialized/

Save:

* outputs/reports/ml_e1_e7_specialized/fall_e1_e7_results.csv
* outputs/reports/ml_e1_e7_specialized/fall_e1_e7_best_by_protocol.csv
* outputs/reports/ml_e1_e7_specialized/fall_e1_e7_feature_importance.csv
* outputs/reports/ml_e1_e7_specialized/direction_e1_e7_results.csv
* outputs/reports/ml_e1_e7_specialized/direction_e1_e7_best_by_protocol.csv
* outputs/reports/ml_e1_e7_specialized/direction_e1_e7_feature_importance.csv
* outputs/reports/ml_e1_e7_specialized/ml_vs_a5_e1_e7_comparison.csv
* outputs/reports/ml_e1_e7_specialized/final_pipeline_recommendation.csv
* outputs/reports/ml_e1_e7_specialized/e2e_direction_results.csv
* outputs/reports/ml_e1_e7_specialized/ml_e1_e7_specialized_summary.md
* outputs/reports/ml_e1_e7_specialized/feature_set_configs.json
* confusion matrices dưới outputs/figures/ml_e1_e7_specialized/

==================================================
PHẦN 9 — FINAL REPORT PHẢI KẾT LUẬN
===================================

Trong ml_e1_e7_specialized_summary.md, trả lời rõ:

1. ML thuần fall-only E1–E7 tốt nhất là model nào khi không dùng FullTiming?
2. FallNoTiming_Core có đủ tốt không hay phải dùng FallNoTiming_Full?
3. Fall ML pure transfer E4/E5 có còn yếu không?
4. ML fall-only có vượt A5 fall head nhất quán không?
5. Nếu bỏ fall head và train direction-only ML, direction E1–E7 đạt bao nhiêu?
6. Direction-only ML có vượt A5 direction head không?
7. DirectionNoMagnitude có giữ được performance không, chứng minh direction phụ thuộc signed features không?
8. Pipeline tốt nhất cuối cùng là gì:

   * A5 full deep multitask
   * ML fall + ML direction
   * ML fall + A5 direction
   * edge-lite variant
9. Có nên bỏ deep fall head khỏi final inference không?
10. Có nên giữ A5 fall head như auxiliary training/reference không?
11. Paper-facing conclusion:

    * Fall detection tốt nhất bởi feature-based no-timing ML Fall Expert.
    * Direction tốt nhất bởi DS-Fall-RD A5 temporal direction head nếu kết quả xác nhận.
    * Hybrid ML Fall + A5 Direction là final practical pipeline nếu nó thắng.
    * FullTiming chỉ là upper-bound, không dùng làm main.

Không silently skip. Nếu thiếu file hoặc model nào, ghi rõ limitation và vẫn chạy phần có thể.
