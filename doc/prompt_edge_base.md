Bạn đang làm việc trong repo:

C:\Users\ADMIN\Desktop\ds-fall

Mục tiêu:
Chạy một thí nghiệm multi-seed công bằng để so sánh:

1. A5_Direction_Only
2. LDX1_D1_Wide_DSConv

Cả hai đều:
- train direction-only
- cùng input temporal tilt12 50 x 12
- cùng train/val/test split
- cùng direction-supervised fall windows
- cùng seeds
- cùng training protocol E1 -> E7
- cùng metric và report

Không đổi Fall Expert.
Không dùng FullTiming.
Không thêm dataset.
Không tune trên test set.
Không dùng A5 reference saved prediction làm kết quả chính trong thí nghiệm này.
A5 reference saved prediction chỉ dùng làm historical reference row nếu cần.

Benchmark:
- Dataset: BITS + WEDA only
- Sampling: 25 Hz
- Window: 2 seconds
- Input: tilt12 temporal sequence
- Shape: 50 x 12
- Direction labels:
  - forward
  - backward
  - lateral

Direction training/evaluation data:
Chỉ dùng windows thỏa:
- fall_label = 1
- direction_supervised = True
- direction_label in {forward, backward, lateral}

Current historical reference:
- A5 saved reference direction:
  - E3 Direction Macro F1 ≈ 0.8967
  - E6 Direction Macro F1 ≈ 0.9350
  - E7 Direction Macro F1 ≈ 0.8390
  - Avg E3/E6/E7 ≈ 0.8902
  - Params ≈ 65,959

Previous compact candidate:
- LDX1_D1_Wide_DSConv:
  - params ≈ 5,699
  - best single-seed avg E3/E6/E7 ≈ 0.8855
  - but seed-sensitive
  - needs fair multi-seed comparison against A5 direction-only

============================================================
1. CREATE SCRIPT
============================================================

Tạo script:

scripts/run_a5_direction_vs_ldx1_multiseed.py

Output folders:

outputs/reports/a5_direction_vs_ldx1_multiseed/
outputs/figures/a5_direction_vs_ldx1_multiseed/
outputs/models/a5_direction_vs_ldx1_multiseed/

Script phải:
- load processed tilt12 temporal windows
- load labels and metadata
- filter direction-supervised fall windows
- build A5_Direction_Only model
- build LDX1_D1_Wide_DSConv model
- run both models on same seeds, same protocols, same learning rates
- evaluate E1 -> E7
- compute mean ± std across seeds
- compare params/F1/edge suitability
- save reports, models, confusion matrices
- no silent skip

============================================================
2. DATA LOADING
============================================================

Load temporal data from existing processed pipeline.

Required arrays/dataframe:
- X_temporal: shape (N, 50, 12)
- y_fall: 0/1
- y_direction: integer labels mapped consistently:
    forward -> 0
    backward -> 1
    lateral -> 2
- direction_supervised: bool
- dataset: bits / weda
- split: train / val / test
- subject_id if available
- window_id if available

Use the exact same splits as existing final benchmark.
Do not resplit if split metadata exists.

Direction subset:
- fall_label == 1
- direction_supervised == True
- direction_label in {forward, backward, lateral}

Log class counts for every:
- protocol
- train split
- val split
- test split

============================================================
3. MODELS TO COMPARE
============================================================

------------------------------------------------------------
MODEL 1: A5_Direction_Only
------------------------------------------------------------

Goal:
Rebuild A5-style temporal direction model but with only one output:

```text
direction_output = 3-class softmax

No fall head.
No fall loss.
No final fall prediction.
No multitask training.

Input:

50 x 12 tilt12

Architecture requirement:
Reuse existing A5/A5WCEFW/DS-Fall-RD blocks as much as possible.

Important:

If existing A5 code has shared encoder + task-specific attention/head, keep:
temporal encoder
direction attention
direction head
Remove or disable:
fall attention
fall head
fall loss
If code requires fall head internally, allow constructing it but do not train/use it; however report clearly that direction-only loss is used.

Expected params:

should be close to or slightly below A5 original ≈ 65,959
log actual params

Name:

A5_Direction_Only
MODEL 2: LDX1_D1_Wide_DSConv

Input:

50 x 12 tilt12

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

Expected params:

~5,699

Name:

LDX1_D1_Wide_DSConv
============================================================
4. TRAINING SETUP

Use identical training setup for both models unless architecture requires otherwise.

Seeds:
Run at least:

42, 43, 44, 45, 46

If time allows:

42, 43, 44, 45, 46, 47, 48, 49

Learning rates:
Run:

1e-3
5e-4

Batch size:

16 or 32

Use the same batch size for both models.

Epochs:

max 120

Optimizer:

Adam

Loss:

SparseCategoricalCrossentropy

or categorical crossentropy if labels are one-hot.

Class imbalance:

use class_weight balanced based on train direction labels.
log class weights.

Callbacks:

EarlyStopping:
monitor validation direction macro F1 if custom callback is available
patience = 15
restore_best_weights = True
ReduceLROnPlateau:
monitor val_loss
factor = 0.5
patience = 6
Model checkpoint:
save best model by val_macro_f1 if possible
otherwise save best val_loss and report limitation

Important:

Do not use test set for early stopping, model selection, threshold, or hyperparameter choice.
For E6 and E7, reuse the model trained in E3 mixed protocol.
============================================================
5. E1 -> E7 PROTOCOL

Run both models on:

E1_BITS_TO_BITS:

train = BITS train supervised fall windows
val = BITS val supervised fall windows
test = BITS test supervised fall windows

E2_WEDA_TO_WEDA:

train = WEDA train supervised fall windows
val = WEDA val supervised fall windows
test = WEDA test supervised fall windows

E3_BITS_WEDA_MIXED:

train = BITS + WEDA train supervised fall windows
val = BITS + WEDA val supervised fall windows
test = BITS + WEDA test supervised fall windows

E4_BITS_TO_WEDA:

train = BITS train supervised fall windows
val = BITS val supervised fall windows
test = WEDA test supervised fall windows

E5_WEDA_TO_BITS:

train = WEDA train supervised fall windows
val = WEDA val supervised fall windows
test = BITS test supervised fall windows

E6_E3_MIXED_TEST_BITS:

reuse E3 mixed model
test = BITS test supervised fall windows only

E7_E3_MIXED_TEST_WEDA:

reuse E3 mixed model
test = WEDA test supervised fall windows only
============================================================
6. METRICS

For every model, seed, learning rate, protocol:

Direction metrics:

Direction Macro F1
Direction Accuracy
forward F1
backward F1
lateral F1
confusion matrix
direction_n_supervised

Main aggregate metrics:

E3 Direction Macro F1
E6 Direction Macro F1
E7 Direction Macro F1
Avg E3/E6/E7 Direction Macro F1
Std across E3/E6/E7
Mean across seeds
Std across seeds
coefficient of variation if useful

Comparison metrics:

delta_vs_A5_Direction_Only_mean
delta_vs_LDX1_mean
delta_vs_historical_A5_reference if available
params
params_reduction_vs_A5
estimated FP32 KB
estimated INT8 KB
edge suitability

End-to-end direction metrics:
Use fixed final fall branch:

GradientBoosting + FallNoTiming_Full_132

If existing final fall predictions are available, reuse them.
If not, train/reuse same final fall expert from previous final pipeline.

For E2E direction:
A sample is correct only when:

final_fall_pred == fall
and direction_pred == direction_true

Compute:

E2E Direction Macro F1
E2E Accuracy / correct_rate
coverage
E3/E6/E7 E2E average
mean ± std across seeds
============================================================
7. OUTPUT FILES

Save reports:

outputs/reports/a5_direction_vs_ldx1_multiseed/a5_vs_ldx1_direction_results.csv
outputs/reports/a5_direction_vs_ldx1_multiseed/a5_vs_ldx1_seed_summary.csv
outputs/reports/a5_direction_vs_ldx1_multiseed/a5_vs_ldx1_best_by_protocol.csv
outputs/reports/a5_direction_vs_ldx1_multiseed/a5_vs_ldx1_e3_e6_e7_summary.csv
outputs/reports/a5_direction_vs_ldx1_multiseed/a5_vs_ldx1_e2e_results.csv
outputs/reports/a5_direction_vs_ldx1_multiseed/a5_vs_ldx1_complexity.csv
outputs/reports/a5_direction_vs_ldx1_multiseed/a5_vs_ldx1_final_summary.md
outputs/reports/a5_direction_vs_ldx1_multiseed/failed_runs.csv
outputs/reports/a5_direction_vs_ldx1_multiseed/model_configs.json

Save models:

outputs/models/a5_direction_vs_ldx1_multiseed/{model_name}_{experiment_id}_seed{seed}_lr{lr}.keras
or .pt if using PyTorch

If TensorFlow is available:
outputs/models/a5_direction_vs_ldx1_multiseed/{model_name}_{experiment_id}_seed{seed}lr{lr}.tflite
outputs/models/a5_direction_vs_ldx1_multiseed/{model_name}{experiment_id}_seed{seed}_lr{lr}_int8.tflite

Figures:

outputs/figures/a5_direction_vs_ldx1_multiseed/e3_e6_e7_boxplot.png
outputs/figures/a5_direction_vs_ldx1_multiseed/mean_std_comparison.png
outputs/figures/a5_direction_vs_ldx1_multiseed/params_vs_mean_f1.png
outputs/figures/a5_direction_vs_ldx1_multiseed/confusion_matrices/

============================================================
8. FINAL REPORT QUESTIONS

In a5_vs_ldx1_final_summary.md, answer clearly:

Is A5_Direction_Only actually more stable than LDX1 across seeds?
Which model has higher mean E3/E6/E7 Direction Macro F1?
Which model has lower std across seeds?
Which model has better E6 BITS performance?
Which model has better E7 WEDA performance?
Does LDX1 remain close to A5 after fair multi-seed comparison?
How much parameter reduction does LDX1 provide compared with A5_Direction_Only?
Which model has better end-to-end direction when using fixed GB Full132 fall branch?
Should A5_Direction_Only replace historical A5 reference?
Should LDX1 replace A5 direction in the final Hybrid?
Final recommendation:
main paper direction expert
compact edge direction expert
future work

Decision rules:

A. Choose A5_Direction_Only as main if:

mean avg E3/E6/E7 F1 is higher than LDX1 by >= 0.02
or
std across seeds is much lower
or
E6/E7 balance is more stable

B. Choose LDX1 as compact direction main if:

mean avg E3/E6/E7 F1 is within 0.02 of A5_Direction_Only
std across seeds is comparable
params reduction >= 85%
E2E direction does not drop more than 0.02

C. If LDX1 has high seed sensitivity:

keep A5_Direction_Only/A5 reference as main
report LDX1 as compact edge ablation

D. If A5_Direction_Only underperforms historical A5 reference:

explain that removing fall head/multitask training may reduce representation quality
keep historical A5 multitask reference as main direction source
report A5_Direction_Only as analysis only

Expected possible conclusion:

A5_Direction_Only may or may not beat historical A5 multitask direction.
LDX1 may remain best compact candidate.
The key fair comparison is mean ± std across seeds, not best single seed.
============================================================
9. CODE QUALITY

Requirements:

argparse:
python scripts/run_a5_direction_vs_ldx1_multiseed.py --repo-root . --run-all --seeds 42 43 44 45 46
python scripts/run_a5_direction_vs_ldx1_multiseed.py --repo-root . --run-all --seeds 42 43 44 45 46 47 48 49
Optional flags:
--quick
--full
--learning-rates 0.001 0.0005
--epochs 120
--batch-size 16
Print progress:
model / protocol / seed / lr / best val macro F1 / test macro F1
Log:
failed_runs.csv
class_counts.csv
model_configs.json
environment_info.txt
No silent skip.
If A5 architecture cannot be rebuilt exactly, implement the closest A5-direction-only version and document differences clearly.
If TensorFlow import fails, allow PyTorch implementation but document framework difference clearly.
Save random seeds and configs for reproducibility.
Do not hard-code absolute paths except repo root.

Kết luận mong muốn sau prompt này:

```text id="eengdg"
Nếu A5_Direction_Only mean cao và std thấp:
    A5 direction vẫn là main.

Nếu LDX1 mean gần A5 trong ±0.02 và params giảm >85%:
    LDX1 có thể trở thành compact direction main/edge-ready model.

Nếu A5_Direction_Only thua historical A5:
    chứng minh multitask A5 cũ có lợi cho direction representation.