Bạn đang làm việc trong repo:

C:\Users\ADMIN\Desktop\ds-fall

Mục tiêu tổng thể:
Dọn dẹp repo và xây dựng một pipeline paper-ready cuối cùng cho bài DS-Fall-RD, tập trung vào một kiến trúc hybrid gọn, ổn định, có thể submit paper.

Pipeline chính cần build:

DS-Fall-RD Compact Hybrid
= Statistical Fall Expert + LDX1 Wide-DSConv Direction Expert

Trong đó:
- Fall detection dùng GradientBoosting trên event-window statistical features.
- Direction classification dùng LDX1_D1_Wide_DSConv trên temporal tilt12 input 50 x 12.
- Không dùng FullTiming.
- Không dùng các timing-index artifact như impact_index, distance_from_center, peak_index.
- Không gọi model là “NoTiming” trong tên chính thức, vì tên này không chuyên nghiệp cho paper.
- Thay vào đó dùng tên:
  - Statistical Fall Expert
  - Event-Stat Fall Expert
  - Compact Hybrid
  - DS-Fall-RD Compact Hybrid

Nhiệm vụ chính:
1. Test nhiều seed để tìm seed tối ưu nhất cho LDX1 Direction Expert.
2. Build lại final model tốt nhất dựa trên seed tối ưu.
3. Dọn repo/output để chỉ giữ pipeline final và các baseline cần so sánh.
4. So sánh pipeline final với các phương pháp ML baseline trên cùng train/val/test split.
5. Xuất report, bảng, figure, model artifact đủ sạch để đưa vào paper.

============================================================
1. FINAL MODEL NAMING
============================================================

Không dùng tên chính thức kiểu:
- NoTiming
- Hybrid_NoTiming
- Full132
- FallNoTiming

Dùng tên paper-ready như sau:

Main model name:
DS-Fall-RD Compact Hybrid

Internal components:
1. Statistical Fall Expert
   - model: GradientBoostingClassifier
   - input: event-window statistical features

2. LDX1 Wide-DSConv Direction Expert
   - model: lightweight temporal DSConv
   - input: tilt12 temporal sequence, 50 x 12

Tên trong bảng:
- DS-Fall-RD Compact Hybrid
- DS-Fall-RD A5 Hybrid Reference
- Statistical-ML Baseline
- Temporal-CNN Baseline
- Classical ML Baseline

Nếu cần ghi chú:
The Statistical Fall Expert uses event-window statistical features excluding explicit timing-index artifacts.

============================================================
2. FEATURE DEFINITION FOR FALL EXPERT
============================================================

Fall Expert dùng event-window statistical features từ cửa sổ 2 giây.

Không gọi là NoTiming trong tên model, nhưng cần ghi rõ trong report:

The fall feature vector contains statistical descriptors computed from the 2-second IMU window, including axis statistics, magnitude statistics, jerk, tilt, and pre/post-window energy descriptors. It excludes explicit timing-index artifacts such as impact index, peak index, and distance-to-center variables.

Feature group chính:

A. Axis-wise accelerometer statistics:
- ax_mean, ax_std, ax_min, ax_max, ax_range, ax_final_initial, ax_median, ax_iqr, peak_signed_ax, integrated_ax
- ay_mean, ay_std, ay_min, ay_max, ay_range, ay_final_initial, ay_median, ay_iqr, peak_signed_ay, integrated_ay
- az_mean, az_std, az_min, az_max, az_range, az_final_initial, az_median, az_iqr, peak_signed_az, integrated_az

B. Axis-wise gyroscope statistics:
- gx_mean, gx_std, gx_min, gx_max, gx_range, gx_final_initial, gx_median, gx_iqr, peak_signed_gx, integrated_gx
- gy_mean, gy_std, gy_min, gy_max, gy_range, gy_final_initial, gy_median, gy_iqr, peak_signed_gy, integrated_gy
- gz_mean, gz_std, gz_min, gz_max, gz_range, gz_final_initial, gz_median, gz_iqr, peak_signed_gz, integrated_gz

C. Magnitude and jerk statistics:
- acc_mag_mean, acc_mag_std, acc_mag_min, acc_mag_max, acc_mag_p95, acc_mag_range, acc_mag_median, acc_mag_iqr
- gyro_mag_mean, gyro_mag_std, gyro_mag_min, gyro_mag_max, gyro_mag_p95, gyro_mag_range, gyro_mag_median, gyro_mag_iqr
- jerk_mean, jerk_std, jerk_max, jerk_p95, jerk_median, jerk_iqr

D. Orientation / tilt statistics:
- roll_mean, roll_std, roll_min, roll_max, roll_range, roll_final_initial, roll_median, roll_iqr
- pitch_mean, pitch_std, pitch_min, pitch_max, pitch_range, pitch_final_initial, pitch_median, pitch_iqr
- tilt_delta_mean, tilt_delta_std, tilt_delta_max, tilt_delta_p95, tilt_delta_final, tilt_delta_median, tilt_delta_iqr

E. Peak magnitude values:
- acc_mag_peak_value
- gyro_mag_peak_value

F. Pre/post event-window descriptors:
- pre_impact_energy
- post_impact_energy
- pre_post_energy_ratio
- post_acc_mag_std
- post_gyro_mag_std
- pre_acc_mag_std
- pre_gyro_mag_std
- pre_ax_mean, post_ax_mean
- pre_ay_mean, post_ay_mean
- pre_az_mean, post_az_mean
- pre_gx_mean, post_gx_mean
- pre_gy_mean, post_gy_mean
- pre_gz_mean, post_gz_mean
- pre_roll_mean, post_roll_mean
- pre_pitch_mean, post_pitch_mean
- delta_roll_window
- delta_pitch_window

Explicitly exclude:
- impact_index
- impact_distance_from_center
- gyro_peak_index
- gyro_peak_distance_from_center
- any feature containing distance_from_center
- any feature containing peak_index if it represents index/time
- any feature containing impact_index

Important:
- peak magnitude values are allowed because they are signal values, not timing indices.

============================================================
3. FINAL DIRECTION EXPERT: LDX1 WIDE-DSCONV
============================================================

Final compact direction model:

LDX1 Wide-DSConv Direction Expert

Input:
- temporal tilt12
- shape: 50 x 12

Channels:
ax, ay, az,
gx, gy, gz,
acc_mag,
gyro_mag,
jerk,
roll,
pitch,
tilt_delta

Architecture:

SeparableConv1D(24, kernel_size=5, padding="same", activation="relu")
BatchNormalization
SeparableConv1D(32, kernel_size=3, padding="same", activation="relu")
BatchNormalization
SeparableConv1D(48, kernel_size=3, padding="same", activation="relu")
GlobalAveragePooling1D
Dense(48, activation="relu")
Dropout(0.15)
Dense(3, activation="softmax")

Direction labels:
- forward
- backward
- lateral

Train/evaluate direction only on:
- fall_label = 1
- direction_supervised = True
- direction_label in {forward, backward, lateral}

============================================================
4. SEED SEARCH FOR BEST LDX1 MODEL
============================================================

Chạy seed search để tìm seed tối ưu nhất cho LDX1.

Seeds cần chạy:
- 1 đến 100 nếu thời gian cho phép
- nếu quá lâu, chạy ít nhất:
  1, 2, 3, 4, 5,
  10, 11, 12, 13, 14,
  21, 22, 23, 24, 25,
  42, 43, 44, 45, 46,
  50, 60, 70, 80, 90, 100

Learning rates:
- 0.001
- 0.0005

Training:
- Adam
- batch size 16 hoặc 32
- max epochs 120
- EarlyStopping patience 15
- ReduceLROnPlateau patience 6
- class_weight balanced
- model selection by validation direction macro F1
- no test set tuning

Protocol for seed search:
- Primary protocol: E3_BITS_WEDA_MIXED
- Then evaluate the selected E3 model on:
  - E3 mixed test
  - E6 BITS test
  - E7 WEDA test

Primary score for seed selection:

score = 0.40 * E3_direction_macro_f1
      + 0.30 * E6_direction_macro_f1
      + 0.30 * E7_direction_macro_f1

Tie-breakers:
1. higher min(E3, E6, E7)
2. lower std(E3, E6, E7)
3. higher E2E direction macro F1
4. lower validation-test gap
5. lower epoch count if metrics are tied

Important:
- Seed search is allowed only for selecting a reproducible final training seed.
- Do not tune using hidden/unseen external data.
- Clearly report that this is a seed-selection experiment and provide the full table.

Output:
outputs/reports/final_compact_hybrid/ldx1_seed_search_results.csv
outputs/reports/final_compact_hybrid/ldx1_seed_search_top10.csv
outputs/reports/final_compact_hybrid/ldx1_selected_seed.json
outputs/figures/final_compact_hybrid/ldx1_seed_score_distribution.png

============================================================
5. BUILD FINAL MODEL USING BEST SEED
============================================================

After seed search:
- choose best seed and learning rate
- retrain final LDX1 Direction Expert using E3 mixed train/val
- evaluate on E3, E6, E7
- also report E1, E2, E4, E5 if possible

Final model:
DS-Fall-RD Compact Hybrid

Fall:
- GradientBoostingClassifier + event-window statistical features
- threshold selected on validation only

Direction:
- LDX1 Wide-DSConv Direction Expert with best seed

Final inference logic:

if fall_pred == non_fall:
    final_output = non_fall
else:
    final_output = fall + direction_pred

Save final artifacts:

outputs/models/final_compact_hybrid/statistical_fall_expert.pkl
outputs/models/final_compact_hybrid/ldx1_wide_dsconv_direction.pt
outputs/models/final_compact_hybrid/final_config.json
outputs/reports/final_compact_hybrid/final_model_card.md

If TensorFlow works:
- also export .tflite and INT8 .tflite
If TensorFlow fails:
- log clearly and save PyTorch model
- estimated INT8 size is allowed but must be marked as estimate

============================================================
6. REPO CLEANUP / OUTPUT ORGANIZATION
============================================================

Do not delete source code blindly.
Create a clean final output structure.

Keep:
src/
scripts/
notebooks/ if needed
outputs/reports/final_compact_hybrid/
outputs/models/final_compact_hybrid/
outputs/figures/final_compact_hybrid/

Archive old experimental outputs into:
outputs/archive_experiments/

Do not permanently delete unless explicitly safe.
Move old reports/models/figures to archive.

Create final README:

README_FINAL_MODEL.md

It must explain:
- dataset
- preprocessing
- input shapes
- fall features
- direction model
- training protocol
- evaluation protocol
- how to reproduce final results
- how to run final inference

Create final run scripts:

scripts/run_final_compact_hybrid.py
scripts/evaluate_final_compact_hybrid.py
scripts/compare_final_with_baselines.py

============================================================
7. BASELINES TO COMPARE ON SAME SPLITS
============================================================

So sánh final compact hybrid với các baseline trên cùng train/val/test split.

Dataset:
- BITS/WEDA only
- 25 Hz
- 2-second event-centered windows
- same splits

Baselines cần chạy:

A. Classical ML fall-only baselines:
1. LogisticRegression
2. RandomForest
3. GradientBoosting
4. HistGradientBoosting
5. SVM RBF nếu thời gian cho phép
6. XGBoost/LightGBM nếu package có sẵn, nếu không thì skip có log

Input:
- same event-window statistical features
- same fall labels
- threshold tuned on validation only

B. Classical ML direction baselines:
1. LogisticRegression
2. RandomForest
3. GradientBoosting
4. HistGradientBoosting
5. SVM RBF nếu thời gian cho phép

Input:
- statistical features
- only supervised fall direction windows

C. End-to-end ML-only baseline:
- fall = best ML fall baseline
- direction = best ML direction baseline
- same final inference rule

D. Historical reference if available:
- Historical A5 multitask direction predictions
- A5 baseline
- Hybrid A5 reference:
  Fall = Statistical Fall Expert
  Direction = historical A5 direction head

E. Final compact hybrid:
- Fall = Statistical Fall Expert
- Direction = LDX1 Wide-DSConv

Metrics:
Fall:
- precision
- recall
- F1
- accuracy
- AUROC
- AUPRC
- TN, FP, FN, TP

Direction:
- macro F1
- accuracy
- forward F1
- backward F1
- lateral F1
- confusion matrix

End-to-end:
- E2E direction macro F1
- E2E accuracy
- coverage
- final output accuracy if implemented:
  non-fall / forward-fall / backward-fall / lateral-fall

Complexity:
- feature count
- params
- estimated FP32 KB
- estimated INT8 KB
- tree count
- tree nodes
- edge suitability

============================================================
8. FINAL PAPER TABLES
============================================================

Generate paper-ready tables:

Table 1: Dataset and window statistics
- dataset
- sampling rate
- window length
- input shape
- number of windows
- fall/non-fall count
- supervised direction count

Table 2: Fall detection comparison
Columns:
- Method
- Feature/Input
- E3 Fall F1
- E6 Fall F1
- E7 Fall F1
- E7 Precision
- E7 Recall
- E7 FP
- Params/Complexity

Table 3: Direction classification comparison
Columns:
- Method
- Input
- Params
- E3 Direction F1
- E6 Direction F1
- E7 Direction F1
- Avg E3/E6/E7
- INT8 KB

Table 4: End-to-end hybrid comparison
Columns:
- Method
- Fall Expert
- Direction Expert
- E3 E2E F1
- E6 E2E F1
- E7 E2E F1
- Avg E2E F1
- Edge suitability

Table 5: Final compact model vs main reference
Columns:
- Model
- Fall feature type
- Direction model
- Direction params
- Direction param reduction
- Fall E7 F1
- Direction avg F1
- E2E avg F1
- Limitation

Save:
outputs/reports/final_compact_hybrid/paper_table_1_dataset_stats.csv
outputs/reports/final_compact_hybrid/paper_table_2_fall_detection.csv
outputs/reports/final_compact_hybrid/paper_table_3_direction.csv
outputs/reports/final_compact_hybrid/paper_table_4_e2e.csv
outputs/reports/final_compact_hybrid/paper_table_5_final_comparison.csv

Also save markdown:
outputs/reports/final_compact_hybrid/paper_tables.md

============================================================
9. FINAL REPORT QUESTIONS
============================================================

Create:

outputs/reports/final_compact_hybrid/final_paper_readiness_report.md

Answer these questions clearly:

1. What is the final selected seed and learning rate for LDX1?
2. Why was this seed selected?
3. What are the final E3/E6/E7 direction metrics?
4. What are the final E3/E6/E7 E2E metrics?
5. How does DS-Fall-RD Compact Hybrid compare with:
   - Classical ML-only methods
   - Historical A5 reference
   - A5 Direction-Only
   - LDX1 multi-seed mean
6. Does the final model keep high metric while reducing parameters?
7. How many direction parameters are reduced versus historical A5 and A5 Direction-Only?
8. Does the final model remain paper-safe, or is it seed-selected too aggressively?
9. What limitations must be stated?
10. Is the repo ready for paper submission?
11. What files should be included in supplementary materials?

Important:
Do not overclaim.
If best seed is selected using test metrics, mark it as “best observed seed” and not as unbiased model selection.
For paper-safe selection, prefer validation-based seed selection and only then report test results.
If seed selection used E3/E6/E7 test performance, state it clearly and use it as an analysis/upper candidate, not the main unbiased result.

============================================================
10. PAPER-SAFE SEED SELECTION RULE
============================================================

To avoid test leakage, implement two modes:

Mode A: paper_safe
- choose seed using validation macro F1 only
- report test metrics once

Mode B: best_observed
- rank seeds using E3/E6/E7 test metrics
- this is for analysis only
- mark as optimistic

Default final paper model should use:
--selection-mode paper_safe

But also save:
- best_observed_seed analysis

Command examples:

python scripts/run_final_compact_hybrid.py --repo-root . --seed-search --selection-mode paper_safe --seeds 1 2 3 4 5 10 11 12 13 14 21 22 23 24 25 42 43 44 45 46 50 60 70 80 90 100

python scripts/run_final_compact_hybrid.py --repo-root . --build-final --selection-mode paper_safe

python scripts/compare_final_with_baselines.py --repo-root . --run-all

============================================================
11. EXPECTED FINAL CONCLUSION
============================================================

Expected conclusion if results follow previous experiments:

- The strongest main reference remains the historical multitask A5 direction head.
- However, LDX1 Wide-DSConv is the best compact direction expert.
- The final compact hybrid keeps the Statistical Fall Expert and replaces A5 direction with LDX1.
- This greatly reduces direction parameters while preserving competitive E3/E6/E7 direction and E2E performance.
- A5 Direction-Only should not replace historical multitask A5, because removing multitask learning weakens direction representation.
- LDX1 should be reported as compact edge model, not as a fully superior replacement to historical multitask A5 unless final paper-safe seed selection proves otherwise.

============================================================
12. CODE QUALITY REQUIREMENTS
============================================================

- No silent skip.
- Save failed_runs.csv.
- Save environment_info.txt.
- Save all configs.
- All random seeds must be logged.
- Use relative paths from repo root.
- Do not delete old outputs; archive them.
- Every generated table must include model name, feature/input type, protocol, and split.
- Print progress clearly.
- Make the final report readable without opening CSV files.
- If any result cannot be reproduced, explain why.
- If TensorFlow/TFLite fails, do not hide it.
- If package is missing, log it and continue with available baselines.