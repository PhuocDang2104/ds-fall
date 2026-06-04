Bạn đang làm việc trong repo:

C:\Users\ADMIN\Desktop\ds-fall

Mục tiêu:
Triển khai và chạy thử nghiệm kiến trúc mới DS-Fall-RD-FV, tức DS-Fall-RD A5WCEFW + Logistic Regression Fall Verifier head, nhằm giảm WEDA false positive nhưng giữ direction performance.

Không thêm dataset mới.
Không chạy Transformer.
Không thay benchmark.
Không thay đổi kết quả reference cũ.
Không chạy lan man ngoài các biến thể FV được yêu cầu.

Benchmark cố định:
- Dataset: BITS + WEDA only
- Sampling: 25 Hz
- Window: 2 seconds event-centered
- Temporal input: tilt12, shape (50, 12)
- Direction labels: forward / backward / lateral
- Direction metric chỉ tính trên fall windows có direction_supervised = True
- Existing train/val/test split
- Reference model: DS-Fall-RD A5WCEFW / REF_CURRENT_E3_ARTIFACT

Bối cảnh kết quả hiện tại:
- Reference E3 Fall F1 = 0.8323
- Reference E3 Direction Macro F1 = 0.8967
- Reference E6 BITS Fall F1 = 0.9451
- Reference E6 BITS Direction Macro F1 = 0.9350
- Reference E7 WEDA Fall F1 = 0.6857
- Reference E7 WEDA Precision = 0.5333
- Reference E7 WEDA Recall = 0.9600
- Reference E7 WEDA FP = 21
- Reference E7 WEDA FN = 1
- Reference E7 WEDA Direction Macro F1 = 0.8390

Đọc kỹ các output có sẵn trước khi code:
- outputs/reports/ml_feature_analysis_25hz/ml_feature_analysis_summary.md
- outputs/reports/ml_feature_analysis_25hz/summary_feature_matrix.csv
- outputs/reports/data_analysis_25hz_center/data_analysis_summary.md
- outputs/reports/fp_reduction_experiments/summary.md nếu có
- artifacts/experiments_25hz_event/E3_BITS_WEDA_MIXED/predictions.csv
- artifacts/experiments_25hz_event/E6_E3_MIXED_TEST_BITS/predictions.csv
- artifacts/experiments_25hz_event/E7_E3_MIXED_TEST_WEDA/predictions.csv

Nếu reference artifact cũ có Lambda layer không load được, dùng lại cách đã làm trước đó:
- rebuild kiến trúc serializable tương đương,
- nạp weights nếu có thể,
- kiểm tra fall_prob trên test khớp saved predictions với max diff nhỏ,
- ghi rõ trong report.

==================================================
1. KIẾN TRÚC CẦN TRIỂN KHAI: DS-Fall-RD-FV
==================================================

Thiết kế:

Input 1:
- temporal_tilt12_input, shape = (50, 12)

Input 2:
- summary_feature_input, shape = (K,)

Temporal branch:
- reuse DS-Fall-RD A5WCEFW exactly as much as possible
- shared encoder
- fall attention -> deep fall logit z_deep
- direction attention -> direction logits z_dir

Summary branch:
- Logistic Regression verifier head:
  z_lr = Dense(1, activation=None, name="lr_fall_verifier_logit")(summary_feature_input)

Fall fusion:
- concatenate [z_deep, z_lr]
- Dense(1, activation=None, name="fall_fusion_logit")
- sigmoid activation -> final fall output

Direction output:
- unchanged from DS-Fall-RD direction head
- summary features must NOT be fed to direction head

Final model outputs:
- final_fall_output
- direction_output

Quan trọng:
- z_deep nên là logit trước sigmoid của deep fall head. Nếu code hiện tại chỉ có sigmoid fall output, chỉnh model để expose logit sạch.
- LR verifier là logistic regression thật sự: chỉ Dense(1), không hidden layer trong biến thể chính.
- Fusion là linear logit fusion: Dense(1) trên [z_deep, z_lr].
- Kiến trúc phải serializable, không dùng Lambda layer không cần thiết.

==================================================
2. SUMMARY FEATURES
==================================================

Tạo summary vector từ existing summary_feature_matrix.csv nếu có. Nếu thiếu thì sinh lại từ tilt12/raw6.

Chạy 2 bộ feature:

FV-Lite10, không dùng timing artifact:
1. acc_mag_range
2. acc_mag_max
3. acc_mag_std
4. gyro_mag_p95
5. gyro_mag_mean
6. jerk_p95
7. jerk_std
8. tilt_delta_p95
9. post_acc_mag_std
10. post_gyro_mag_std

FV-Center12, upper-bound có timing:
Lite10 +
11. impact_distance_from_center
12. gyro_peak_distance_from_center

Nếu feature name trong file hơi khác, tự map tương đương. Nếu thiếu feature nào, sinh lại hoặc ghi rõ limitation.

Normalize summary features:
- fit scaler trên train split only
- apply cho val/test
- save scaler

==================================================
3. HARD NEGATIVE WEIGHTING
==================================================

Tạo hard-negative flag cho train/val:
- dataset = weda
- fall_label = 0
- high-motion non-fall

Hard-negative score:
rank-normalize rồi average các feature:
- jerk_p95
- jerk_std
- acc_mag_std
- acc_mag_range
- tilt_delta_p95
- tilt_delta_max nếu có
- ay_range nếu có
- ay_std nếu có
- gyro_mag_p95 hoặc gyro_mag_max

Chọn top 20% WEDA non-fall train/val làm hard negatives.

Sample weights cho fall final loss:
- normal non-fall: 1.0
- fall: 1.0 hoặc 1.2
- WEDA hard-negative non-fall: 2.0 hoặc 3.0

Chạy theo config từng run bên dưới.

==================================================
4. TRAINING STRATEGY
==================================================

Không train lại toàn bộ model từ đầu nếu không cần.

Stage 1:
- Load hoặc reproduce A5 reference.
- Confirm E3/E6/E7 predictions match saved predictions nếu có thể.

Stage 2:
- Freeze DS-Fall-RD temporal encoder, deep fall head, and direction branch.
- Train only:
  - lr_fall_verifier_logit
  - fall_fusion_logit
- Loss:
  - binary cross entropy on final_fall_output
- Direction branch frozen, direction output should remain unchanged.
- Tune global fall threshold on validation.

Optional Stage 3 only if Stage 2 is too weak:
- Unfreeze fall attention + deep fall head only.
- Keep shared encoder and direction branch frozen.
- Fine-tune 5–15 epochs.
- Use low learning rate.
- Do not run this unless needed for FV4 or explicitly logged.

==================================================
5. RUN EXACTLY THESE VARIANTS
==================================================

FV1_LITE10_AND:
- features = Lite10
- train LR verifier only if needed
- decision fusion outside graph:
  final_fall = deep_fall_prob >= T_deep AND lr_fall_prob >= T_lr
- no trainable logit fusion used for final report, but model components may still be saved
- tune T_deep and T_lr on validation
- sample hard-negative weight = 2.0

FV2_LITE10_LOGIT_FUSION:
- features = Lite10
- fusion = trainable logit fusion
- train LR verifier + fusion layer only
- hard-negative weight = 2.0
- this is the preferred clean architecture

FV3_CENTER12_LOGIT_FUSION:
- features = Center12
- fusion = trainable logit fusion
- train LR verifier + fusion layer only
- hard-negative weight = 2.0
- treat as upper-bound because timing features may be protocol-specific

FV4_LITE10_GATED_PRODUCT:
- features = Lite10
- fusion:
  p_final = p_deep * (alpha_gate + (1 - alpha_gate) * p_lr)
  alpha_gate in {0.2, 0.3}
- tune alpha_gate and thresholds on validation
- no deep encoder update
- hard-negative weight = 2.0

FV5_LITE10_LOGIT_FUSION_HN3:
- same as FV2
- hard-negative weight = 3.0

FV6_LITE10_LOGIT_FUSION_UNFREEZE_FALL:
- start from best between FV2/FV5
- unfreeze fall attention + deep fall head only
- keep shared encoder and direction branch frozen
- fine-tune 5–15 epochs with low LR
- hard-negative weight = 2.0
- only run if implementation is safe; otherwise skip and explain

Không chạy thêm biến thể ngoài FV1–FV6.

==================================================
6. EVALUATION
==================================================

Evaluate each run on:
- E3 mixed test
- E6 BITS test
- E7 WEDA test

Metrics bắt buộc:
Fall:
- threshold
- TN, FP, FN, TP
- precision
- recall
- Fall F1
- AUROC nếu có
- Average Precision nếu có

Direction:
- Direction Macro F1
- Direction Accuracy
- per-class F1: forward, backward, lateral
- direction_n_supervised

Dataset-specific:
- E6 BITS Fall F1
- E6 BITS Direction Macro F1
- E7 WEDA Fall F1
- E7 WEDA Precision
- E7 WEDA Recall
- E7 WEDA FP/FN/TP/TN
- E7 WEDA Direction Macro F1

Model:
- params
- added params over reference
- summary feature count
- whether timing features are used
- whether direction output changed from reference

Selection target:
- WEDA FP < 12
- WEDA precision > 0.65
- WEDA recall >= 0.88
- WEDA Fall F1 > 0.75
- E3 Direction Macro F1 >= 0.86
- E7 WEDA Direction Macro F1 >= 0.80
- E6 BITS Fall F1 >= 0.90
- params <= 70k

Because direction branch is frozen, direction should match reference. If direction changes unexpectedly, debug and report.

==================================================
7. OUTPUTS
==================================================

Create folders:
- outputs/reports/ds_fall_rd_fv/
- outputs/figures/ds_fall_rd_fv/confusion_matrices/

Save:
- outputs/reports/ds_fall_rd_fv/fv_results.csv
- outputs/reports/ds_fall_rd_fv/fv_best_selection.md
- outputs/reports/ds_fall_rd_fv/fv_summary.md
- outputs/reports/ds_fall_rd_fv/fv_feature_configs.json
- outputs/reports/ds_fall_rd_fv/fv_thresholds.csv
- outputs/reports/ds_fall_rd_fv/fv_weda_fp_analysis.csv
- outputs/reports/ds_fall_rd_fv/fv_model_params.csv
- confusion matrices for each run

Final report must answer:
1. Does DS-Fall-RD-FV reduce WEDA FP compared with reference?
2. Which variant is best: FV1/FV2/FV3/FV4/FV5/FV6?
3. Does Lite10 without timing work well enough?
4. Does Center12 give an upper-bound improvement?
5. Does logit fusion beat AND/gated fusion?
6. Does increasing hard-negative weight help?
7. Does unfreezing fall head help or hurt?
8. Does direction remain unchanged?
9. Can DS-Fall-RD-FV replace A5 reference as main paper model, or should it be reported as an optional FPGuard extension?
10. Paper-facing conclusion.

Report conclusion should be careful:
- If FV improves WEDA precision/FP while preserving direction, recommend DS-Fall-RD-FV as an enhanced model or optional verifier.
- If FV only modestly improves, keep A5 as main and report FV as FPGuard analysis.
- If Center12 works much better than Lite10, warn that timing features may be event-centered protocol-specific.

Do not silently skip failed runs. If a run cannot be completed, write the reason in fv_summary.md.