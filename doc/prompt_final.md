Bạn đang làm việc trong repo:

```text
C:\Users\ADMIN\Desktop\ds-fall
```

Mục tiêu:
Dọn dẹp repo, giữ lại các tài liệu/kết quả chính quan trọng, chuẩn hóa lại 4 pipeline cuối cùng, chạy lại đầy đủ E1 → E7, rồi sinh báo cáo tổng hợp cuối cùng cho đề tài DS-Fall-RD.

Không thêm dataset mới.
Không đổi benchmark.
Không dùng FullTiming làm main result.
Không tune trên test set.
Không giữ lại các experiment tạm/nháp nếu không cần cho kết luận cuối.
Không xóa raw dataset hoặc processed dataset.

============================================================

1. CONTEXT NGHIÊN CỨU HIỆN TẠI
   ============================================================

Đề tài:
DS-Fall-RD: Direction-sensitive fall detection from wrist IMU.

Dataset chính:

* BITS
* WEDA

Benchmark chính:

* Sampling: 25 Hz
* Window: 2 seconds
* Input temporal: 50 × 12
* Feature temporal tilt12 gồm:

  * ax, ay, az
  * gx, gy, gz
  * acc_mag
  * gyro_mag
  * jerk
  * roll
  * pitch
  * tilt_delta

Direction labels:

* forward
* backward
* lateral

Direction metric:

* chỉ tính trên fall windows có `direction_supervised = True`.

A5 reference:

* DS-Fall-RD A5WCEFW
* deep temporal multitask model
* fall head + direction head
* params khoảng 65,959
* hiện tại dùng làm main deep reference và direction expert.

Kết quả/insight hiện tại:

* A5 direction rất tốt và ổn định.
* A5 fall head yếu trên WEDA do false positive cao.
* ML Fall Expert no-timing giảm WEDA FP tốt hơn.
* FullTiming rất mạnh nhưng chỉ xem là upper-bound vì dùng event-centered timing cues.
* Kiến trúc chính nên chốt:

  * DS-Fall-RD-Hybrid-NoTiming
  * Fall = GradientBoosting FallNoTiming_Full
  * Direction = A5 direction head
* EdgeLite Hybrid:

  * Fall = RandomForest FallNoTiming_Core
  * Direction = A5 direction head
* ML-only separated:

  * Fall = best ML FallNoTiming_Full
  * Direction = best ML DirectionFullNoTiming hoặc DirectionCore tùy E1–E7, nhưng phải cố định lựa chọn cuối sau validation/report.
* A5 reference:

  * Fall = A5 deep fall head
  * Direction = A5 direction head

============================================================
2. VIỆC CẦN LÀM TỔNG QUÁT
=========================

Làm 4 nhóm việc:

A. Dọn dẹp repo và tài liệu:

* Giữ lại raw dataset.
* Giữ lại processed dataset.
* Giữ lại model code chính.
* Giữ lại scripts chính để process/train/evaluate.
* Giữ lại A5 artifacts/reference predictions.
* Giữ lại outputs/reports cuối cùng.
* Gom hoặc archive các experiment cũ/tạm vào thư mục `archive/` thay vì xóa cứng.
* Tạo thư mục `docs/final/` chứa các tài liệu chính.
* Tạo một file tổng hợp nghiên cứu từ đầu đến giờ.

B. Chuẩn hóa 4 pipeline cuối:

1. A5_reference
2. Hybrid_NoTiming
3. EdgeLite_Hybrid
4. ML_only_separated

C. Chạy lại 4 pipeline trên toàn bộ E1 → E7:

* E1 BITS→BITS
* E2 WEDA→WEDA
* E3 BITS+WEDA mixed
* E4 BITS→WEDA
* E5 WEDA→BITS
* E6 E3 mixed model test BITS
* E7 E3 mixed model test WEDA

D. Sinh báo cáo:

* So sánh 4 pipeline.
* Kết luận kiến trúc chính.
* Sinh file `.md` giải thích nguyên lý kiến trúc của 4 pipeline ngắn gọn nhất.

============================================================
3. DỌN DẸP REPO
===============

Tạo script:

```text
scripts/cleanup_final_research_state.py
```

Script này phải:

1. Tạo folder:

```text
docs/final/
outputs/final_e1_e7/
outputs/final_e1_e7/reports/
outputs/final_e1_e7/tables/
outputs/final_e1_e7/figures/
outputs/final_e1_e7/predictions/
archive/old_experiments/
archive/old_reports/
```

2. Không xóa:

* `data/raw/`
* `data/processed/`
* `src/`
* `scripts/`
* `notebooks/` nếu còn dùng
* artifacts A5 reference
* các file config chính
* các report quan trọng

3. Archive các output nháp/tạm nếu có:

* các experiment lặp lại không còn dùng cho final conclusion
* các file log tạm
* các report cũ trùng lặp

Không hard-delete nếu không chắc. Chỉ move sang `archive/`.

4. Tạo hoặc cập nhật các file trong `docs/final/`:

```text
docs/final/01_dataset_and_protocol.md
docs/final/02_processed_dataset.md
docs/final/03_model_architecture_a5.md
docs/final/04_hybrid_architecture.md
docs/final/05_final_four_pipelines.md
docs/final/06_research_summary_from_start_to_now.md
docs/final/07_final_e1_e7_results_summary.md
```

Nếu các file này chưa có thì tạo mới.
Nếu đã có thì overwrite bằng bản sạch, có cấu trúc rõ.

============================================================
4. NỘI DUNG DOCS CẦN SINH
=========================

## 01_dataset_and_protocol.md

Nội dung cần có:

* Mục tiêu đề tài.
* Dataset BITS và WEDA.
* Sampling 25 Hz.
* Window 2 giây.
* Event-centered protocol.
* Input temporal shape 50 × 12.
* Direction classes.
* E1–E7 protocol.
* Lưu ý FullTiming chỉ là upper-bound, không dùng main.

## 02_processed_dataset.md

Nội dung cần có:

* Đường dẫn processed dataset.
* Cách tạo window.
* Các cột label:

  * fall_label
  * direction_label
  * direction_supervised
  * dataset
  * split
  * subject_id nếu có
* Feature temporal tilt12.
* Summary feature matrix.
* Phân biệt:

  * FallNoTiming_Core
  * FallNoTiming_Full
  * DirectionCore
  * DirectionFullNoTiming
  * FullTiming upper-bound

## 03_model_architecture_a5.md

Nội dung cần có:

* A5WCEFW là gì.
* Input 50 × 12 tilt12.
* Shared temporal encoder.
* Fall head.
* Direction head.
* Task-specific attention.
* Weighted CE nếu có.
* Vai trò hiện tại:

  * main deep temporal reference
  * direction expert chính
  * fall head chỉ còn baseline/auxiliary reference trong Hybrid.

## 04_hybrid_architecture.md

Nội dung cần có:

* Vì sao tách fall và direction.
* Fall detection phụ thuộc impact/jerk/energy/summary.
* Direction phụ thuộc signed temporal roll/pitch/gyro pattern.
* Hybrid:

  * ML Fall Expert
  * A5 Direction Expert
* Không dùng FullTiming trong main.
* FullTiming chỉ upper-bound.
* Final decision rule.

## 05_final_four_pipelines.md

Mô tả 4 pipeline cuối:

1. A5_reference:

```text
fall = A5 fall head
direction = A5 direction head
```

2. Hybrid_NoTiming:

```text
fall = GradientBoosting + FallNoTiming_Full
direction = A5 direction head
```

3. EdgeLite_Hybrid:

```text
fall = RandomForest + FallNoTiming_Core
direction = A5 direction head
```

4. ML_only_separated:

```text
fall = GradientBoosting + FallNoTiming_Full
direction = GradientBoosting/selected best fixed ML Direction model with no timing
```

Nhớ ghi rõ:

* ML-only separated là analysis pipeline, không phải main paper architecture nếu direction không thắng ổn định.
* Hybrid_NoTiming là practical main pipeline.

## 06_research_summary_from_start_to_now.md

Tổng hợp nghiên cứu từ đầu đến giờ:

* DS-Fall ban đầu.
* A5WCEFW là reference tốt nhất.
* Phân tích WEDA fall gap.
* WEDA FP cao do hard-negative ADL.
* Feature analysis:

  * ML fall mạnh hơn deep fall head.
  * A5 direction mạnh/ổn định hơn hoặc an toàn hơn ML direction trên toàn benchmark.
* FV1 chỉ cải thiện vừa.
* Hybrid tách task là hướng tốt nhất.
* FullTiming upper-bound.
* Chốt architecture.

## 07_final_e1_e7_results_summary.md

Sau khi chạy xong E1–E7, tự động điền:

* bảng tổng hợp 4 pipeline
* best model
* kết luận chính
* limitation
* next experiments

============================================================
5. CHUẨN HÓA 4 PIPELINE CUỐI
============================

Tạo script chính:

```text
scripts/run_final_four_pipelines_e1_e7.py
```

Script này phải chạy 4 pipeline:

---

## PIPELINE 1: A5_reference

Definition:

```text
fall_pred = A5 deep fall head prediction
direction_pred = A5 direction head prediction
```

Yêu cầu:

* Load saved A5 predictions nếu có.
* Nếu thiếu saved predictions cho một protocol, cố gắng load/rebuild A5 artifact.
* Nếu vẫn thiếu, ghi limitation rõ trong report, không fake kết quả.

Input:

* temporal tilt12 50 × 12

Output:

* fall metrics
* direction metrics
* end-to-end direction metrics

---

## PIPELINE 2: Hybrid_NoTiming

Definition:

```text
fall_model = GradientBoosting
fall_features = FallNoTiming_Full
direction_model = A5 direction head
```

FallNoTiming_Full:

* Dùng tất cả signal-derived summary features.
* Loại bỏ timing artifact:

  * impact_index
  * impact_distance_from_center
  * gyro_peak_index
  * gyro_peak_distance_from_center
  * mọi feature chứa distance_from_center
  * mọi feature chứa peak_index nếu là timing index
  * mọi feature chứa impact_index
* Không dùng labels/metadata/predictions làm features.

Training:

* Train fall model trên train split của từng protocol.
* Tune threshold trên validation split.
* Test trên test split.
* E6/E7 reuse E3 mixed fall model.

Direction:

* lấy từ A5 direction predictions.

---

## PIPELINE 3: EdgeLite_Hybrid

Definition:

```text
fall_model = RandomForest
fall_features = FallNoTiming_Core
direction_model = A5 direction head
```

FallNoTiming_Core gồm đúng các feature sau nếu có:

```text
acc_mag_range
acc_mag_max
acc_mag_p95
acc_mag_std
gyro_mag_range
gyro_mag_max
gyro_mag_p95
gyro_mag_mean
jerk_max
jerk_p95
jerk_std
post_acc_mag_std
post_gyro_mag_std
pre_post_energy_ratio
tilt_delta_p95
tilt_delta_std
```

Nếu thiếu feature nào, log rõ và dùng các feature còn lại, nhưng report phải ghi missing features.

Training:

* RandomForest
* nên dùng cố định:

```text
n_estimators=300
max_depth hoặc default theo script cũ nếu đã có
random_state=42
class_weight balanced nếu có lợi, nhưng phải cố định và report
```

E6/E7 reuse E3 mixed model.

Direction:

* A5 direction predictions.

---

## PIPELINE 4: ML_only_separated

Definition:

```text
fall_model = GradientBoosting
fall_features = FallNoTiming_Full

direction_model = ML Direction model no-timing
direction_features = DirectionFullNoTiming hoặc DirectionCore
```

Chọn direction model cố định dựa theo previous specialized result:

* ưu tiên `GradientBoosting + DirectionFullNoTiming`
* nếu validation/E1–E7 summary cho thấy model khác ổn định hơn, chọn model tốt nhất nhưng phải ghi rõ tiêu chí.
* Không chọn model riêng cho từng test set trong final pipeline nếu muốn so sánh công bằng.
* Direction-only chỉ train trên:

```text
fall_label = 1
direction_supervised = True
direction_label in {forward, backward, lateral}
```

DirectionFullNoTiming:

* all signal-derived summary features
* no timing artifact
* no labels/metadata/predictions.

E6/E7 reuse E3 mixed models.

============================================================
6. E1–E7 PROTOCOL
=================

Chạy cho cả 4 pipeline:

E1_BITS_TO_BITS:

```text
train_dataset = bits
val_dataset = bits
test_dataset = bits
```

E2_WEDA_TO_WEDA:

```text
train_dataset = weda
val_dataset = weda
test_dataset = weda
```

E3_BITS_WEDA_MIXED:

```text
train_dataset = bits + weda
val_dataset = bits + weda
test_dataset = bits + weda
```

E4_BITS_TO_WEDA:

```text
train_dataset = bits
val_dataset = bits
test_dataset = weda
```

E5_WEDA_TO_BITS:

```text
train_dataset = weda
val_dataset = weda
test_dataset = bits
```

E6_E3_MIXED_TEST_BITS:

```text
reuse model trained in E3
test_dataset = bits only
```

E7_E3_MIXED_TEST_WEDA:

```text
reuse model trained in E3
test_dataset = weda only
```

Nếu existing split có train/val/test:

* giữ đúng existing split.
* Không random split lại nếu không cần.
* Nếu subject_id có sẵn, đảm bảo không tạo subject leakage.
* Log split summary:

  * number train/val/test windows
  * fall/non-fall counts
  * direction supervised counts.

============================================================
7. METRICS BẮT BUỘC
===================

Cho từng pipeline và từng E1–E7:

## Fall metrics

* Fall Precision
* Fall Recall
* Fall F1
* Fall Accuracy
* AUROC nếu có probability
* Average Precision nếu có probability
* TN, FP, FN, TP
* threshold nếu dùng ML
* FP reduction vs A5 nếu A5 available

## Direction metrics

* Direction Macro F1
* Direction Accuracy
* forward F1
* backward F1
* lateral F1
* confusion matrix
* direction_n_supervised

## End-to-end direction

Tính trên direction-supervised fall windows.

Một sample direction chỉ đúng khi:

```text
final_fall = fall
AND
direction_pred = direction_true
```

Metrics:

* E2E Direction Macro F1
* E2E Direction Accuracy / correct rate
* coverage:

```text
number of supervised fall windows kept by final_fall / total supervised fall windows
```

## Complexity

* model params hoặc number of weights/nodes/trees
* feature count
* edge suitability:

  * high
  * medium
  * low
* complexity notes

============================================================
8. OUTPUT FILES
===============

Script phải lưu:

```text
outputs/final_e1_e7/reports/final_four_pipeline_results.csv
outputs/final_e1_e7/reports/final_four_pipeline_fall_results.csv
outputs/final_e1_e7/reports/final_four_pipeline_direction_results.csv
outputs/final_e1_e7/reports/final_four_pipeline_e2e_direction_results.csv
outputs/final_e1_e7/reports/final_four_pipeline_best_by_metric.csv
outputs/final_e1_e7/reports/final_four_pipeline_complexity.csv
outputs/final_e1_e7/reports/final_four_pipeline_summary.md
outputs/final_e1_e7/reports/final_architecture_principles.md
outputs/final_e1_e7/reports/final_research_summary.md
outputs/final_e1_e7/reports/final_feature_sets.json
outputs/final_e1_e7/reports/final_thresholds.csv
```

Figures:

```text
outputs/final_e1_e7/figures/confusion_matrices/
outputs/final_e1_e7/figures/bar_charts/
```

Predictions:

```text
outputs/final_e1_e7/predictions/
```

Mỗi pipeline/protocol lưu prediction file:

```text
{pipeline}_{experiment_id}_predictions.csv
```

Prediction file cần có:

* dataset
* split
* y_fall_true
* y_fall_pred
* y_fall_prob nếu có
* y_direction_true
* y_direction_pred
* direction_supervised
* window_id nếu có
* subject_id nếu có
* final_output label nếu có

============================================================
9. FINAL SUMMARY REPORT
=======================

File:

```text
outputs/final_e1_e7/reports/final_four_pipeline_summary.md
```

Phải trả lời rõ:

1. Trong 4 pipeline, pipeline nào tốt nhất tổng thể?
2. Pipeline nào tốt nhất cho WEDA fall?
3. Pipeline nào tốt nhất cho direction?
4. Pipeline nào có end-to-end direction tốt nhất?
5. Pipeline nào nhẹ nhất?
6. A5 còn vai trò gì nếu Hybrid tốt hơn fall?
7. Có nên bỏ A5 fall head khỏi final inference không?
8. Có nên giữ A5 direction head không?
9. ML-only separated có đủ tốt để thay A5 direction không?
10. EdgeLite Hybrid có đáng giữ không?
11. FullTiming có được dùng main không? Trả lời: không, chỉ upper-bound nếu có nhắc.
12. Chốt kiến trúc chính cuối cùng.

Kết luận mong muốn nếu kết quả khớp insight hiện tại:

```text
Recommended main practical architecture:
DS-Fall-RD-Hybrid-NoTiming
= GradientBoosting FallNoTiming_Full Fall Expert
+ DS-Fall-RD A5 Direction Expert

A5_reference remains the main deep temporal baseline and direction expert.
A5 fall head is kept as auxiliary/reference only, not final fall inference.
EdgeLite Hybrid is a compact ablation.
ML-only separated is analysis-only unless it beats A5 direction consistently across E1–E7.
```

============================================================
10. FINAL ARCHITECTURE PRINCIPLES MD
====================================

Tạo file:

```text
outputs/final_e1_e7/reports/final_architecture_principles.md
```

Nội dung ngắn gọn, có bảng và sơ đồ text.

Phải mô tả nguyên lý 4 pipeline:

## 1. A5_reference

```text
tilt12 sequence 50 × 12
        ↓
DS-Fall-RD A5
        ├── fall head → fall / non-fall
        └── direction head → forward / backward / lateral
```

Vai trò:

* deep temporal multitask baseline
* direction-sensitive representation
* reference chính

## 2. Hybrid_NoTiming

```text
tilt12/raw window
        ├── FallNoTiming_Full summary features
        │       ↓
        │   GradientBoosting Fall Expert
        │       ↓
        │   final fall
        │
        └── tilt12 sequence
                ↓
            A5 direction head
                ↓
            direction
```

Vai trò:

* main practical architecture
* giảm WEDA false positives
* không dùng timing artifact

## 3. EdgeLite_Hybrid

```text
tilt12/raw window
        ├── FallNoTiming_Core 16 features
        │       ↓
        │   RandomForest Fall Expert
        │       ↓
        │   final fall
        │
        └── A5 direction head
```

Vai trò:

* compact / edge-friendly ablation
* feature count thấp hơn
* so sánh với full hybrid

## 4. ML_only_separated

```text
summary features
        ├── ML Fall Expert
        └── ML Direction Expert
```

Vai trò:

* analysis pipeline
* kiểm tra liệu ML direction có thể thay A5 không
* không chọn làm main nếu pure transfer/direction consistency yếu

Phải có bảng:

| Pipeline | Fall decision | Direction decision | Feature/Input | Vai trò |
| -------- | ------------- | ------------------ | ------------- | ------- |

============================================================
11. FINAL RESEARCH SUMMARY MD
=============================

Tạo file:

```text
outputs/final_e1_e7/reports/final_research_summary.md
```

Nội dung:

* Tổng hợp nghiên cứu từ đầu đến giờ.
* Ngắn gọn nhưng đầy đủ.
* Không văn phong AI.
* Có các mục:

1. Problem definition
2. Dataset and benchmark
3. DS-Fall-RD A5 model
4. WEDA fall gap
5. Feature analysis
6. FV/FPGuard experiments
7. Hybrid experiments
8. Specialized ML E1–E7
9. Final architecture decision
10. Limitations
11. Next work

Kết luận phải nhấn mạnh:

* fall và direction nên tách expert.
* no-timing ML Fall Expert xử lý fall tốt hơn.
* A5 direction head vẫn giữ vai trò chính.
* FullTiming chỉ upper-bound.
* Final practical architecture là Hybrid_NoTiming.

============================================================
12. KIỂM TRA CHẤT LƯỢNG
=======================

Sau khi chạy xong, in ra console:

1. Đường dẫn các file report.
2. Topline result table:

* 4 pipeline × E3/E6/E7
* Fall F1
* Direction Macro F1
* E2E Direction F1
* WEDA FP

3. Recommended architecture.
4. Các limitation nếu có.

Không silently fail.
Nếu thiếu file, tạo warning rõ.
Nếu không load được A5, vẫn chạy ML pipelines nhưng report A5 limitation.
Nếu thiếu một feature trong Core, log và tiếp tục với feature còn lại.
Nếu một model fail, ghi vào failed_runs.csv với lỗi cụ thể.

============================================================
13. YÊU CẦU CODE STYLE
======================

* Code rõ ràng, có hàm:

  * load_feature_matrix
  * build_feature_sets
  * get_protocol_splits
  * train_fall_model
  * train_direction_model
  * evaluate_fall
  * evaluate_direction
  * evaluate_e2e_direction
  * save_reports
  * write_final_docs
* Dùng random_state=42.
* Không hard-code absolute path quá nhiều ngoài repo root.
* Có argparse:

```text
python scripts/run_final_four_pipelines_e1_e7.py --repo-root . --run-all
python scripts/cleanup_final_research_state.py --repo-root .
```

* Nếu chạy lâu, có progress print theo từng pipeline/protocol.
* Save mọi config vào json để tái lập.

Bắt đầu bằng:

1. inspect repo structure
2. locate feature matrix / predictions
3. implement cleanup script
4. implement final pipeline script
5. run cleanup
6. run final E1–E7
7. generate final docs
8. print final recommendation
