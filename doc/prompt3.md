Bạn đang làm việc trong repo:

C:\Users\ADMIN\Desktop\ds-fall

Mục tiêu:
Chạy thử nghiệm nhanh kiến trúc hybrid tách riêng:

* Fall detection dùng ML Fall Expert trên summary features.
* Direction classification dùng DS-Fall-RD A5WCEFW direction head.
* So sánh 2 lựa chọn:
  A) Bỏ deep fall head khỏi final decision: final_fall = ML Fall Expert.
  B) Giữ deep fall head như auxiliary training/reference, nhưng inference vẫn dùng ML Fall Expert cho final fall và DS-Fall-RD direction head cho direction.

Không thêm dataset.
Không đổi benchmark.
Không train deep architecture mới nếu không cần.
Không thay đổi kết quả reference cũ.
Không chạy ngoài scope các run bên dưới.

Benchmark cố định:

* Dataset: BITS + WEDA only
* Sampling: 25 Hz
* Window: 2-second event-centered
* Temporal input: tilt12, shape (50, 12)
* Existing train/val/test split
* Direction labels: forward / backward / lateral
* Direction metric chỉ tính trên fall windows có direction_supervised = True
* Reference direction expert: DS-Fall-RD A5WCEFW / REF_CURRENT_E3_ARTIFACT

Đọc kỹ các file/output có sẵn:

* outputs/reports/ml_feature_analysis_25hz/ml_feature_analysis_summary.md
* outputs/reports/ml_feature_analysis_25hz/summary_feature_matrix.csv
* outputs/reports/ml_feature_analysis_25hz/ml_vs_deep_comparison.csv nếu có
* outputs/reports/data_analysis_25hz_center/data_analysis_summary.md
* outputs/reports/ds_fall_rd_fv/fv_summary.md nếu có
* outputs/reports/ds_fall_rd_fv_extra/fv_extra_summary.md nếu có
* artifacts/experiments_25hz_event/E3_BITS_WEDA_MIXED/predictions.csv
* artifacts/experiments_25hz_event/E6_E3_MIXED_TEST_BITS/predictions.csv
* artifacts/experiments_25hz_event/E7_E3_MIXED_TEST_WEDA/predictions.csv

Nếu reference artifact cũ có Lambda layer không load được:

* dùng saved predictions cho direction/fall reference nếu đủ;
* hoặc rebuild serializable A5 tương đương và nạp weights như các script trước;
* ghi rõ trong report.

==================================================

1. KIẾN TRÚC CẦN TEST
   ==================================================

Hybrid decision architecture:

Summary features
→ ML Fall Expert
→ final_fall

tilt12 sequence
→ DS-Fall-RD A5 direction head
→ direction_pred / direction_prob

Final decision:
if final_fall == non-fall:
output = non-fall
else:
output = fall + direction from DS-Fall-RD direction head

Quan trọng:

* Deep fall head KHÔNG dùng làm final fall decision trong hybrid runs.
* Direction output luôn lấy từ DS-Fall-RD A5 reference.
* Direction branch không bị train/sửa.
* ML Fall Expert train bằng train split, tune threshold trên validation split, evaluate trên test split.
* Không dùng test set để tune threshold/hyperparameter.

==================================================
2. FEATURE SETS
===============

Dùng summary_feature_matrix.csv nếu có. Nếu thiếu thì sinh lại từ tilt12/raw6.

Chạy 3 bộ feature:

A. Lite10, edge-friendly, không timing artifact:

* acc_mag_range
* acc_mag_max
* acc_mag_std
* gyro_mag_p95
* gyro_mag_mean
* jerk_p95
* jerk_std
* tilt_delta_p95
* post_acc_mag_std
* post_gyro_mag_std

B. Lite10_plus_axis, thêm feature hard-negative WEDA:

* acc_mag_range
* acc_mag_max
* acc_mag_std
* jerk_p95
* jerk_std
* gyro_mag_p95
* gyro_mag_max
* ay_std
* ay_range
* post_acc_mag_std

C. FullSummary_NoTiming:

* tất cả signal-derived summary features
* loại bỏ metadata/label/prediction columns
* loại bỏ impact_distance_from_center, impact_index, gyro_peak_distance_from_center, gyro_peak_index
* không dùng timing artifact

D. FullSummary_WithTiming, upper-bound:

* tất cả signal-derived summary features
* có thể dùng impact_distance_from_center, impact_index, gyro_peak_distance_from_center, gyro_peak_index
* ghi rõ đây là upper-bound vì timing feature có thể phụ thuộc event-centered protocol

Normalize:

* fit scaler trên train split only;
* apply cho val/test;
* save scaler/config.

==================================================
3. ML FALL EXPERTS CẦN CHẠY
===========================

Chạy các Fall Expert sau:

1. Logistic Regression
2. DecisionTree max_depth=3
3. RandomForest n_estimators=50, max_depth=5
4. GradientBoosting
5. HistGradientBoosting

Nếu model nào không khả dụng trong environment thì skip và ghi rõ.

Training:

* Train trên train split.
* Tune threshold trên validation split.
* Primary threshold objective: maximize Fall F1.
* Secondary threshold objective: maximize Fall F1 with recall >= 0.88.
* Nếu nhiều threshold tương đương, chọn threshold có WEDA-like hard-negative FP thấp hơn trên validation nếu có metadata.
* Evaluate trên E3 mixed test, E6 BITS test, E7 WEDA test.

==================================================
4. RUNS CẦN CHẠY
================

Chạy nhanh nhưng đầy đủ các run sau:

A. Hybrid-Lite:
HYB_LR_Lite10
HYB_DT_Lite10
HYB_RF_Lite10
HYB_GB_Lite10
HYB_HGB_Lite10

B. Hybrid-Axis:
HYB_LR_Axis
HYB_DT_Axis
HYB_RF_Axis
HYB_GB_Axis
HYB_HGB_Axis

C. Hybrid-Full-NoTiming:
HYB_LR_FullNoTiming
HYB_RF_FullNoTiming
HYB_GB_FullNoTiming
HYB_HGB_FullNoTiming

D. Hybrid-Full-WithTiming upper-bound:
HYB_RF_FullTiming
HYB_GB_FullTiming
HYB_HGB_FullTiming

Không chạy thêm ngoài danh sách này.

==================================================
5. SO SÁNH LỰA CHỌN A VÀ B
==========================

Lựa chọn A:

* final_fall = ML Fall Expert
* direction = DS-Fall-RD direction expert
* deep fall head không tham gia inference

Lựa chọn B:

* DS-Fall-RD A5 vẫn được ghi nhận là auxiliary multitask training/reference model.
* Inference vẫn:
  final_fall = ML Fall Expert
  direction = DS-Fall-RD direction expert
* B không cần train lại A5 nếu reference đã có.
* Report phải nói rõ deep fall head chỉ có vai trò auxiliary/reference, không dùng final fall decision.

Nếu có thời gian và code đã hỗ trợ an toàn:

* train thêm một direction-focused A5 variant với fall loss giảm nhẹ chỉ để kiểm tra direction có tăng không.
* Nhưng mặc định KHÔNG cần chạy deep retraining. Nếu không chạy, ghi rõ giữ A5 reference là direction expert.

==================================================
6. METRICS BẮT BUỘC
===================

Cho từng hybrid run, báo:

Fall metrics:

* E3 Fall Precision / Recall / F1
* E3 TN/FP/FN/TP
* E6 BITS Fall Precision / Recall / F1
* E6 TN/FP/FN/TP
* E7 WEDA Fall Precision / Recall / F1
* E7 TN/FP/FN/TP
* E7 WEDA FP reduction so với A5 reference
* AUROC, Average Precision nếu có
* threshold

Direction metrics:

* E3 Direction Macro F1
* E3 Direction Accuracy
* E6 BITS Direction Macro F1
* E7 WEDA Direction Macro F1
* per-class direction F1: forward/backward/lateral
* direction_n_supervised

End-to-end direction metrics:
Tính thêm end-to-end direction trên direction-supervised fall windows:

* Một direction sample chỉ tính đúng nếu final_fall phát hiện fall và direction_pred đúng.
* Báo:

  * E3 end-to-end Direction Macro F1
  * E6 end-to-end Direction Macro F1
  * E7 end-to-end Direction Macro F1
  * end-to-end direction coverage/recall: bao nhiêu supervised fall windows được final_fall giữ lại

Complexity:

* model_type
* feature_set
* number_features
* estimated params / number trees / max_depth
* edge suitability: high / medium / low
* notes

==================================================
7. BASELINES CẦN SO SÁNH
========================

Luôn đưa các dòng baseline vào bảng tổng hợp:

1. DS-Fall-RD A5 reference:

* final_fall = deep fall head
* direction = A5 direction head
* Params = 65959
* Dùng metrics từ saved predictions.

2. Best FV1 cũ nếu có:

* DS-Fall-RD + LR Lite10 AND verifier
* WEDA Fall F1 = 0.7500, Precision = 0.6154, Recall = 0.9600, FP = 15 nếu lấy được từ report cũ.

3. Classical fall-only best từ ml_feature_analysis nếu có:

* RF / GB / HGB fall-only
* Direction blank hoặc ghi not_supported.

4. Hybrid runs mới:

* final_fall = ML Fall Expert
* direction = A5 direction head

==================================================
8. SELECTION CRITERIA
=====================

Hard target:

* E7 WEDA Fall F1 >= 0.85 nếu có thể
* E7 WEDA Precision >= 0.80 nếu có thể
* E7 WEDA Recall >= 0.88
* E7 WEDA Direction Macro F1 >= 0.80
* E3 Direction Macro F1 >= 0.86
* E6 BITS Fall F1 >= 0.90

Primary selection:

1. Best balanced hybrid metric:

   * E7 WEDA Fall F1
   * E7 WEDA Precision
   * E7 WEDA Recall
   * E3 Direction Macro F1
   * E7 end-to-end Direction Macro F1
2. Prefer no-timing features over timing features if metrics are close.
3. Prefer lighter model if metrics are close:
   Logistic Regression > DecisionTree > RandomForest > GradientBoosting > HistGradientBoosting
4. FullTiming runs are upper-bound only, not main recommendation unless explicitly justified.

==================================================
9. OUTPUTS
==========

Create folders:

* outputs/reports/ds_fall_rd_hybrid/
* outputs/figures/ds_fall_rd_hybrid/confusion_matrices/

Save:

* outputs/reports/ds_fall_rd_hybrid/hybrid_results.csv
* outputs/reports/ds_fall_rd_hybrid/hybrid_best_selection.md
* outputs/reports/ds_fall_rd_hybrid/hybrid_summary.md
* outputs/reports/ds_fall_rd_hybrid/hybrid_feature_configs.json
* outputs/reports/ds_fall_rd_hybrid/hybrid_thresholds.csv
* outputs/reports/ds_fall_rd_hybrid/hybrid_complexity.csv
* outputs/reports/ds_fall_rd_hybrid/hybrid_end_to_end_direction.csv
* confusion matrix figures for fall and direction where useful

==================================================
10. FINAL REPORT PHẢI TRẢ LỜI
=============================

Trong hybrid_summary.md, trả lời rõ:

1. Hybrid tách fall expert và direction expert có cải thiện Fall F1 không?
2. Fall expert nào tốt nhất?
3. Feature set nào tốt nhất: Lite10, Axis, FullNoTiming, FullTiming?
4. Có cần dùng timing features không, hay NoTiming đã đủ?
5. Hybrid có giảm WEDA FP mạnh không?
6. Direction có giữ nguyên không?
7. End-to-end direction có bị ảnh hưởng bởi fall expert recall không?
8. Lựa chọn A hay B hợp paper hơn?
9. Có nên thay A5 reference bằng Hybrid làm final pipeline không?
10. Nếu dùng paper, nên trình bày Hybrid là main architecture, optional extension, hay upper-bound analysis?
11. Paper-facing conclusion.

Kết luận mong muốn phải rõ:

* Nếu Hybrid-FullNoTiming hoặc Hybrid-Lite đạt Fall tốt và giữ direction, đề xuất làm final practical pipeline.
* Nếu FullTiming tốt hơn nhiều, chỉ xem là upper-bound vì có protocol timing cues.
* Nếu Logistic/Tree đủ tốt, ưu tiên chúng vì edge-friendly.
* Nếu RF/GB/HGB tốt hơn rõ, báo chúng như performance-oriented hybrid, còn LR/Tree là edge-friendly variant.

Không silently skip failed runs. Nếu run nào fail, fix hoặc ghi lý do cụ thể trong summary.
