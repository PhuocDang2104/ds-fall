Bạn đang làm việc trong repo:

C:\Users\ADMIN\Desktop\ds-fall

Mục tiêu: KHÔNG tối ưu kiến trúc lớn, KHÔNG thêm dataset, KHÔNG chạy lan man. Chỉ test nhanh 2 block sau dựa trên output data analysis đã có để giảm WEDA false positive trong benchmark BITS/WEDA 25 Hz event-centered 2s.

Đọc kỹ các output đã có:
- outputs/reports/data_analysis_25hz_center/data_analysis_summary.md
- outputs/reports/data_analysis_25hz_center/fall_probability_distribution.csv
- outputs/reports/data_analysis_25hz_center/weda_error_feature_comparison.csv
- outputs/reports/data_analysis_25hz_center/univariate_fall_importance.csv
- outputs/reports/data_analysis_25hz_center/summary_feature_matrix_with_predictions.csv
- outputs/reports/next12_experiments/summary.md nếu có
- outputs/reports/targeted_experiments/best_model_selection.md nếu có

Bối cảnh:
- Main model hiện vẫn là REF_CURRENT_E3_ARTIFACT / DS-Fall-RD A5WCEFW.
- WEDA Fall F1 thấp chủ yếu do FP, không phải FN.
- WEDA: FP=21, FN=1, precision=0.5333, recall=0.9600, Fall F1=0.6857.
- Vấn đề chính: WEDA non-fall hard-negative có high-motion giống fall.
- Không đổi benchmark: BITS/WEDA only, 25 Hz, event-centered 2s, tilt12.

Chỉ chạy 2 block sau.

==================================================
BLOCK A — THRESHOLD / DECISION ANALYSIS
==================================================

Không retrain model. Dùng saved predictions/probabilities của reference nếu có.

Chạy 4 decision rules:

T1: Global threshold chọn bằng validation để maximize Fall F1 như hiện tại.
T2: Global threshold maximize Fall F1 nhưng yêu cầu Fall Precision >= 0.65 trên validation.
T3: Global threshold maximize Fall F1 nhưng yêu cầu Fall Precision >= 0.70 trên validation.
T4: Global threshold chọn để minimize FP nhưng Fall Recall >= 0.88 trên validation.

Yêu cầu:
- Không dùng dataset-aware threshold làm main result.
- Nhưng có thể báo dataset-specific analysis phụ trong report nếu hữu ích.
- Với mỗi rule, evaluate trên E3 mixed test, E6 BITS test, E7 WEDA test.
- Báo đầy đủ: threshold, TP, FP, FN, TN, precision, recall, Fall F1, Direction Macro F1, Direction n supervised.
- Đặc biệt báo WEDA FP giảm từ 21 xuống bao nhiêu, WEDA precision tăng bao nhiêu, WEDA recall còn bao nhiêu.

==================================================
BLOCK B — HARD-NEGATIVE TRAINING
==================================================

Chỉ chạy 4 experiment nhanh. Không thay kiến trúc lớn.

Hard-negative definition:

HN feature-rule:
WEDA non-fall train/val windows có high-motion, chọn top quantile theo một score từ các feature:
- jerk_p95 hoặc jerk_max
- acc_mag_std hoặc acc_mag_range
- tilt_delta_max hoặc tilt_delta_std
- ay_range hoặc ay_std
- gyro_mag_max hoặc gyro_mag_range

Tạo hard_negative_score bằng cách rank-normalize các feature trên rồi lấy trung bình. Chọn top 20% hoặc top 30% WEDA non-fall làm hard negatives. Nếu code đã có feature matrix thì dùng trực tiếp; nếu chưa có thì sinh lại từ tilt12/raw6.

HN prediction-rule:
Dùng reference predictions nếu có. Chọn WEDA non-fall train/val có fall_prob cao nhất hoặc fall_prob > 0.5. Nếu train/val không có prediction thì dùng model hiện tại chạy inference để lấy fall_prob. Nếu không load được old Lambda artifact, dùng serializable reference/retrained teacher gần nhất và ghi rõ limitation.

Chạy 4 runs:

HN1:
- Method: hard-negative weighted CE
- Hard negative source: feature-rule
- hard_negative_weight = 2.0
- alpha_fall = 1.0
- lambda_dir = 1.5
- sampler = current hoặc giữ pipeline reference

HN2:
- Method: hard-negative weighted CE
- Hard negative source: prediction-rule
- hard_negative_weight = 2.0
- alpha_fall = 1.0
- lambda_dir = 1.5

HN3:
- Method: 2-stage fine-tune
- Stage 1: dùng reference-compatible model hoặc checkpoint tốt nhất có thể load/retrain.
- Stage 2: fine-tune fall attention + fall head only trên batch có hard negatives.
- Hard negative source: saved FP / prediction-rule
- Giữ direction branch frozen nếu code hỗ trợ.
- 10–20 epochs, early stopping.

HN4:
- Method: 2-stage fine-tune fall branch + direction KD
- Hard negative source: saved FP / prediction-rule
- Fine-tune fall branch/last fall layers, giữ direction ổn bằng KD từ reference teacher.
- beta_kd = 1.0 hoặc 2.0, chọn 1.0 nếu chỉ chạy nhanh.
- 10–20 epochs, early stopping.

Yêu cầu chung cho HN runs:
- Không tăng params đáng kể.
- Không phá direction branch.
- Dùng global validation-tuned threshold, sau đó có thể evaluate thêm T2/T3/T4 decision rules nếu nhanh.
- Metrics bắt buộc:
  - E3 Fall precision/recall/F1
  - E3 Direction Macro F1
  - E6 BITS Fall precision/recall/F1
  - E6 BITS Direction Macro F1
  - E7 WEDA Fall precision/recall/F1
  - E7 WEDA FP/FN/TP/TN
  - E7 WEDA Direction Macro F1
  - Direction n supervised
  - params

Selection criteria:
Hard filters:
- E3 Direction Macro F1 >= 0.86
- E7 WEDA Direction Macro F1 >= 0.80
- E6 BITS Fall F1 >= 0.90
- E7 WEDA Recall >= 0.88
- Params <= 90k

Primary target:
- giảm WEDA FP
- tăng WEDA precision
- giữ WEDA recall không tụt mạnh

Target mong muốn:
- WEDA FP: 21 → dưới 12
- WEDA precision: 0.5333 → trên 0.65
- WEDA recall: >= 0.88
- WEDA Fall F1: 0.75–0.82
- E7 WEDA Direction F1: >= 0.80
- E3 Direction F1: >= 0.86

Output folder:
outputs/reports/fp_reduction_experiments/
outputs/figures/fp_reduction_experiments/

Save:
- outputs/reports/fp_reduction_experiments/threshold_decision_results.csv
- outputs/reports/fp_reduction_experiments/hard_negative_runs.csv
- outputs/reports/fp_reduction_experiments/best_fp_reduction_selection.md
- outputs/reports/fp_reduction_experiments/summary.md
- outputs/reports/fp_reduction_experiments/weda_fp_analysis.csv
- confusion matrices vào outputs/figures/fp_reduction_experiments/confusion_matrices/

Final report phải trả lời:
1. Threshold-only có giảm được WEDA FP không?
2. Rule nào tốt nhất trong T1–T4?
3. Hard-negative training có giảm FP tốt hơn threshold-only không?
4. HN1/HN2/HN3/HN4 run nào tốt nhất?
5. Có run nào đủ điều kiện thay reference main model không?
6. Nếu chưa có, nên giữ reference và report hard-negative result như ablation hay future work?
7. Kết luận paper-facing: WEDA thấp do hard-negative ADL, và hướng FP-aware training có cải thiện được precision tới đâu.

Không chạy ngoài 2 block trên. Nếu thiếu file prediction/model, tự tìm file phù hợp trong artifacts/outputs; nếu không thể load artifact cũ vì Lambda layer, ghi limitation rõ ràng và dùng serializable reference gần nhất.