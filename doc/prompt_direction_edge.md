Bạn đang làm việc trong repo:

C:\Users\ADMIN\Desktop\ds-fall

Mục tiêu:
Thử thêm một batch nhỏ các Direction Expert có params tăng nhẹ so với D1 Tiny DSConv nhưng vẫn nhỏ hơn A5 rất nhiều, nhằm kéo Direction Macro F1 sát A5 baseline.

Không đổi Fall Expert.
Không dùng FullTiming.
Không thêm dataset.
Không tune trên test set.

Benchmark:
- BITS/WEDA only
- 25 Hz
- 2-second event-centered windows
- temporal input tilt12
- input shape 50 x 12
- direction labels: forward / backward / lateral
- direction train/evaluate chỉ dùng:
  fall_label = 1
  direction_supervised = True
  direction_label in {forward, backward, lateral}

Reference:
A5 direction:
- params ≈ 65,959
- E3 Direction F1 ≈ 0.8967
- E6 Direction F1 ≈ 0.9350
- E7 Direction F1 ≈ 0.8390
- Avg E3/E6/E7 ≈ 0.8902

Best lightweight previous:
D1_Tiny_DSConv_Direction:
- params = 2,947
- best avg E3/E6/E7 Direction F1 ≈ 0.8267
- E3 ≈ 0.8197
- E6 ≈ 0.7831
- E7 ≈ 0.8774

Current practical fall branch:
- Fall = GradientBoosting + FallNoTiming_Full_132
- Use this fixed fall branch only for E2E direction evaluation.
- Do not retrain or change fall branch in this script.

============================================================
1. CREATE SCRIPT
============================================================

Create:

scripts/run_direction_mid_size_candidates.py

Output folders:

outputs/reports/direction_mid_size_candidates/
outputs/figures/direction_mid_size_candidates/
outputs/models/direction_mid_size_candidates/

The script must:
- load temporal tilt12 windows
- load labels and metadata
- load A5 direction reference predictions if available
- load or reuse GradientBoosting FallNoTiming_Full_132 fall predictions for E2E metrics
- train only direction models
- evaluate E1 -> E7
- save reports and models

============================================================
2. MODELS TO TEST
============================================================

Test exactly these candidate families:

------------------------------------------------------------
LDX1_D1_Wide_DSConv
------------------------------------------------------------

Input: 50 x 12

Architecture:
- SeparableConv1D(24, kernel_size=5, padding="same", activation="relu")
- BatchNormalization
- SeparableConv1D(32, kernel_size=3, padding="same", activation="relu")
- BatchNormalization
- SeparableConv1D(48, kernel_size=3, padding="same", activation="relu")
- GlobalAveragePooling1D
- Dense(48, activation="relu")
- Dropout(0.15)
- Dense(3, activation="softmax")

Goal:
- params around 6k-9k
- improve E3/E6 over D1 Tiny DSConv

------------------------------------------------------------
LDX2_D1_Wide_Attention
------------------------------------------------------------

Input: 50 x 12

Architecture:
- SeparableConv1D(24, kernel_size=5, padding="same", activation="relu")
- BatchNormalization
- SeparableConv1D(32, kernel_size=3, padding="same", activation="relu")
- BatchNormalization
- SeparableConv1D(48, kernel_size=3, padding="same", activation="relu")
- Lightweight temporal attention pooling:
    score_t = Dense(1)(h_t)
    alpha_t = softmax(score_t over time)
    context = sum(alpha_t * h_t)
- Dense(48, activation="relu")
- Dropout(0.15)
- Dense(3, activation="softmax")

Goal:
- params around 7k-10k
- recover temporal focus similar to A5 attention

------------------------------------------------------------
LDX3_A5_Direction_Small_075
------------------------------------------------------------

Implement direction-only A5-like model with width multiplier 0.75.

Requirements:
- no fall head
- only direction output
- reuse A5 blocks if available
- preserve the same temporal direction logic as much as possible
- reduce width/channels to around 75% of A5

Goal:
- params < 30k
- closer to A5 direction than D1/D2

------------------------------------------------------------
LDX4_D1_Wide_Attention_Distill
------------------------------------------------------------

Same architecture as LDX2_D1_Wide_Attention, but train with A5 teacher distillation.

Teacher:
- use saved A5 direction probabilities if available
- if only hard A5 direction labels are available, implement hard-label teacher fallback but mark limitation

Loss:
L = CE(y_true, student_pred) + alpha * KL(soft_teacher, soft_student)

Try:
- temperature = 2
- alpha = 0.5

If time allows, also try:
- temperature = 3
- alpha = 0.5
- temperature = 2
- alpha = 0.7

Goal:
- params < 12k
- improve average E3/E6/E7 direction macro F1 by at least +0.03 over D1 Tiny DSConv
- target avg >= 0.856

============================================================
3. TRAINING SETUP
============================================================

Direction-only training data:
- fall_label = 1
- direction_supervised = True
- direction_label in {forward, backward, lateral}

Loss:
- CE for normal models
- CE + distillation KL for distill model

Optimizer:
- Adam

Learning rates:
- 1e-3
- 5e-4

Batch size:
- 16 or 32

Epochs:
- max 120

Callbacks:
- EarlyStopping monitor val_macro_f1 if implemented
- patience 15
- restore_best_weights=True
- ReduceLROnPlateau monitor val_loss, factor=0.5, patience=6

Class imbalance:
- use balanced class weights for CE
- log class counts

Seeds:
Run:
- seed 42 first
If time allows:
- seed 43
- seed 44

At minimum, all candidate models must run seed 42.

============================================================
4. E1 -> E7 PROTOCOL
============================================================

Run every candidate on:

E1_BITS_TO_BITS:
- train BITS
- val BITS
- test BITS

E2_WEDA_TO_WEDA:
- train WEDA
- val WEDA
- test WEDA

E3_BITS_WEDA_MIXED:
- train BITS + WEDA
- val BITS + WEDA
- test BITS + WEDA

E4_BITS_TO_WEDA:
- train BITS
- val BITS
- test WEDA

E5_WEDA_TO_BITS:
- train WEDA
- val WEDA
- test BITS

E6_E3_MIXED_TEST_BITS:
- reuse E3 mixed model
- test BITS only

E7_E3_MIXED_TEST_WEDA:
- reuse E3 mixed model
- test WEDA only

Do not use test data for model selection.

============================================================
5. METRICS
============================================================

Direction metrics:
- Direction Macro F1
- Direction Accuracy
- forward F1
- backward F1
- lateral F1
- confusion matrix
- direction_n_supervised

End-to-end direction metrics:
- use fixed GB Full132 fall branch
- a sample is correct only if:
    final_fall = fall
    direction_pred = direction_true
- E2E Direction Macro F1
- E2E accuracy / correct rate
- coverage

Complexity:
- params
- estimated FP32 KB
- estimated INT8 KB
- estimated MACs if easy
- TFLite convert success/fail if TensorFlow works
- INT8 TFLite convert success/fail if TensorFlow works
- edge suitability

Comparison:
- compare against A5 direction
- compare against previous D1 Tiny DSConv
- compare against D2 Micro TCN

============================================================
6. OUTPUT FILES
============================================================

Save:

outputs/reports/direction_mid_size_candidates/direction_mid_size_results.csv
outputs/reports/direction_mid_size_candidates/direction_mid_size_best_by_protocol.csv
outputs/reports/direction_mid_size_candidates/direction_mid_size_vs_a5_and_d1.csv
outputs/reports/direction_mid_size_candidates/direction_mid_size_e2e_results.csv
outputs/reports/direction_mid_size_candidates/direction_mid_size_complexity.csv
outputs/reports/direction_mid_size_candidates/direction_mid_size_summary.md
outputs/reports/direction_mid_size_candidates/failed_runs.csv
outputs/reports/direction_mid_size_candidates/model_configs.json

Save models:

outputs/models/direction_mid_size_candidates/{model_name}_{experiment_id}_seed{seed}_lr{lr}.keras
or .pt if using PyTorch

If TensorFlow works:
outputs/models/direction_mid_size_candidates/{model_name}_{experiment_id}_seed{seed}_lr{lr}.tflite
outputs/models/direction_mid_size_candidates/{model_name}_{experiment_id}_seed{seed}_lr{lr}_int8.tflite

Save figures:

outputs/figures/direction_mid_size_candidates/params_vs_avg_f1.png
outputs/figures/direction_mid_size_candidates/e3_e6_e7_direction_bar.png
outputs/figures/direction_mid_size_candidates/confusion_matrices/

============================================================
7. FINAL REPORT QUESTIONS
============================================================

In direction_mid_size_summary.md, answer:

1. Did increasing params above D1 improve average E3/E6/E7 direction F1?
2. Which candidate gives the best params/F1 trade-off?
3. Which candidate is closest to A5 while staying under 30k params?
4. Which candidate is best under 12k params?
5. Did attention pooling help over plain GlobalAveragePooling?
6. Did distillation from A5 help?
7. Which candidate improves E6 BITS the most?
8. Which candidate improves E3 mixed the most?
9. Which candidate keeps E7 WEDA strong?
10. Can any candidate replace A5 direction in the main Hybrid?
11. If not, which should be reported as edge ablation?

Selection criteria:

Main replacement candidate:
- avg E3/E6/E7 Direction Macro F1 >= 0.870
- E3 >= 0.86
- E6 >= 0.88
- E7 >= 0.83
- params <= 30k

Edge ablation candidate:
- avg E3/E6/E7 Direction Macro F1 >= 0.850
- params <= 12k
- E7 >= 0.85

If no candidate satisfies these:
- keep A5 direction as main
- keep best candidate as edge ablation

Expected outcome:
- D1 Wide Attention or Distilled D1 is most likely to improve over D1 Tiny DSConv.
- A5 Small 0.75 may be closest to A5 but could cost more params.
- If distillation works, it may be the best under-12k candidate.

============================================================
8. CODE QUALITY
============================================================

- argparse:
  python scripts/run_direction_mid_size_candidates.py --repo-root . --run-all --seed 42
  python scripts/run_direction_mid_size_candidates.py --repo-root . --run-all --seeds 42 43 44

- progress print for model/protocol/lr/seed
- no silent skip
- save failed_runs.csv
- save all configs
- do not hard-code absolute paths except repo root
- use random_state/seed consistently