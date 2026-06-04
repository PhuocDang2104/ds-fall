Bạn đang làm việc trong repo:

C:\Users\ADMIN\Desktop\ds-fall

Mục tiêu:
Chạy thêm đúng 6 run nhỏ để tối ưu DS-Fall-RD-FV / FPGuard sau kết quả FV1 Lite10 AND. Không chạy thêm ngoài scope. Không thêm dataset. Không thay benchmark. Không train kiến trúc deep mới. Giữ DS-Fall-RD A5 reference làm Stage-1 model và giữ direction output từ reference.

Bối cảnh hiện tại:
- Reference A5 E7 WEDA:
  - Fall F1 = 0.6857
  - Precision = 0.5333
  - Recall = 0.9600
  - FP = 21
  - FN = 1
  - Direction Macro F1 = 0.8390
- Best FV hiện tại:
  - FV1 Lite10 AND
  - WEDA Fall F1 = 0.7500
  - Precision = 0.6154
  - Recall = 0.9600
  - FP = 15
  - FN = 1
  - Direction Macro F1 = 0.8390
- Mục tiêu mới:
  - WEDA FP <= 12 nếu có thể
  - WEDA precision >= 0.65
  - WEDA recall >= 0.88
  - WEDA Fall F1 > 0.75
  - E3 Direction Macro F1 >= 0.86
  - E7 WEDA Direction Macro F1 >= 0.80
  - Direction phải giữ nguyên vì không được đụng direction branch.

Đọc kỹ output hiện có:
- outputs/reports/ds_fall_rd_fv/fv_summary.md
- outputs/reports/ds_fall_rd_fv/fv_results.csv
- outputs/reports/ds_fall_rd_fv/fv_best_selection.md
- outputs/reports/ml_feature_analysis_25hz/ml_feature_analysis_summary.md
- outputs/reports/ml_feature_analysis_25hz/summary_feature_matrix.csv
- outputs/reports/data_analysis_25hz_center/data_analysis_summary.md

Nếu script run_ds_fall_rd_fv.py đã có, hãy mở rộng script đó hoặc tạo script mới run_ds_fall_rd_fv_extra.py. Không phá kết quả cũ.

==================================================
Benchmark cố định
==================================================

- Dataset: BITS + WEDA only
- Sampling: 25 Hz
- Window: 2-second event-centered
- Temporal input: tilt12, shape (50, 12)
- Existing train/val/test split
- Reference model: DS-Fall-RD A5WCEFW
- Final direction output lấy nguyên từ reference, không thay đổi.

==================================================
Feature sets
==================================================

1. Lite10, dùng lại đúng như FV1:
- acc_mag_range
- acc_mag_max
- acc_mag_std
- gyro_mag_p95
- gyro_mag_mean
- jerk_p95
- jerk_std
- tilt_delta_p95
- post_acc_mag_std
- post_gyro_mag_std

2. AxisHard10, thiết kế để đánh trực tiếp WEDA hard-negative FP:
- ay_std
- ay_range
- jerk_p95
- jerk_std
- acc_mag_std
- acc_mag_range
- gyro_mag_max
- gyro_mag_range
- tilt_delta_p95
- post_acc_mag_std

Nếu feature name hơi khác trong summary_feature_matrix.csv, tự map tên tương đương. Nếu thiếu feature, sinh lại từ tilt12/raw6 hoặc ghi rõ limitation.

Normalize features:
- Fit scaler trên train split only.
- Apply cho val/test.
- Không leak test.

==================================================
6 runs cần chạy
==================================================

FV1A:
- Verifier: Logistic Regression
- Features: Lite10
- Fusion: AND
- Threshold selection: chọn threshold trên validation để maximize F1 với điều kiện precision >= 0.65.
- Nếu không có threshold nào đạt precision >= 0.65 trên validation, chọn threshold có precision cao nhất trong nhóm recall >= 0.88 và ghi rõ.

FV1B:
- Verifier: Logistic Regression
- Features: Lite10
- Fusion: AND
- Threshold selection: minimize FP trên validation với điều kiện recall >= 0.88.
- Nếu nhiều threshold cùng FP, chọn threshold có F1 cao nhất.

FV7:
- Verifier: Logistic Regression
- Features: AxisHard10
- Fusion: AND
- Threshold selection: maximize F1 với precision >= 0.65 trên validation; fallback như FV1A.

FV8:
- Verifier: Logistic Regression
- Features: AxisHard10
- Fusion: trainable/logit fusion nếu code cũ đã hỗ trợ.
- Nếu logit fusion trong graph khó tái sử dụng nhanh, dùng stacked logistic fusion trên validation features:
  input = [deep_fall_logit_or_prob, lr_verifier_logit_or_prob]
  train logistic fusion on train/val only, tune threshold on validation.
- Threshold selection: maximize F1 với precision >= 0.65.

FV9:
- Verifier: DecisionTreeClassifier(max_depth=3, class_weight/sample_weight nếu phù hợp)
- Features: AxisHard10
- Fusion: AND
- Threshold: nếu tree có predict_proba thì tune như FV7; nếu chỉ hard label, dùng tree label as verifier.
- Tune thresholds on validation only.

FV10:
- Verifier: RandomForestClassifier(n_estimators=10, max_depth=3, random_state fixed, class_weight/sample_weight nếu phù hợp)
- Features: AxisHard10
- Fusion: AND
- Tune verifier threshold on validation như FV7.

Không chạy thêm run nào khác.

==================================================
Hard-negative weighting
==================================================

Dùng hard-negative sample weights khi train verifier:
- normal non-fall: 1.0
- fall: 1.0 hoặc 1.2
- WEDA hard-negative non-fall: 2.0

Hard-negative definition:
dataset = weda, fall_label = 0, high-motion non-fall.

Tạo hard_negative_score bằng rank-normalized average của:
- ay_std
- ay_range
- jerk_p95
- jerk_std
- acc_mag_std
- acc_mag_range
- gyro_mag_max
- gyro_mag_range
- tilt_delta_p95
- post_acc_mag_std

Chọn top 20% WEDA non-fall train/val làm hard negatives.

==================================================
Evaluation
==================================================

Evaluate mỗi run trên:
- E3 mixed test
- E6 BITS test
- E7 WEDA test

Metrics bắt buộc:
- threshold_deep nếu dùng
- threshold_verifier nếu dùng
- threshold_final nếu dùng
- TN, FP, FN, TP
- precision
- recall
- Fall F1
- AUROC nếu có
- Average Precision nếu có
- E3 Direction Macro F1
- E3 Direction Accuracy
- E6 BITS Direction Macro F1
- E7 WEDA Direction Macro F1
- direction_n_supervised
- WEDA FP reduction so với reference
- WEDA FP reduction so với FV1 Lite10 AND
- model/verifier size estimate:
  - LR params
  - tree depth/leaves
  - RF number of trees/depth
- edge suitability note

Direction phải giữ nguyên reference. Nếu direction thay đổi, đó là bug trong evaluation.

==================================================
Outputs
==================================================

Tạo folder:
outputs/reports/ds_fall_rd_fv_extra/
outputs/figures/ds_fall_rd_fv_extra/confusion_matrices/

Lưu:
- outputs/reports/ds_fall_rd_fv_extra/fv_extra_results.csv
- outputs/reports/ds_fall_rd_fv_extra/fv_extra_best_selection.md
- outputs/reports/ds_fall_rd_fv_extra/fv_extra_summary.md
- outputs/reports/ds_fall_rd_fv_extra/fv_extra_thresholds.csv
- outputs/reports/ds_fall_rd_fv_extra/fv_extra_weda_fp_analysis.csv
- confusion matrices cho từng run

Final report phải trả lời:
1. FV1A/FV1B có cải thiện FV1 cũ không?
2. AxisHard10 có tốt hơn Lite10 không?
3. LR, DecisionTree depth=3, TinyRF 10 trees depth=3: verifier nào tốt nhất?
4. Run nào giảm WEDA FP mạnh nhất mà vẫn giữ recall >= 0.88?
5. Run nào có trade-off tốt nhất theo target paper?
6. Có run nào đạt FP <= 12 và precision >= 0.65 không?
7. Có nên thay FV1 bằng run mới không?
8. Có nên giữ DS-Fall-RD A5 là main model và báo FV như FPGuard extension không?
9. Paper-facing conclusion ngắn gọn.

Selection:
- Hard constraints:
  - E7 WEDA recall >= 0.88
  - E7 WEDA Direction Macro F1 >= 0.80
  - E3 Direction Macro F1 >= 0.86
  - E6 BITS Fall F1 >= 0.90
- Primary target:
  - minimize WEDA FP
  - maximize WEDA precision
  - keep WEDA Fall F1 > 0.75 if possible
- Tie-break:
  - simpler and more edge-friendly verifier wins:
    LR > DecisionTree > TinyRF, if metrics are close.

Không silently skip. Nếu run fail, fix hoặc ghi rõ lý do trong summary.