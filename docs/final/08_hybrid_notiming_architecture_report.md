# Hybrid_NoTiming Architecture Report

## 1. Final Name

Recommended practical pipeline:

```text
DS-Fall-RD-Hybrid-NoTiming
= GradientBoosting FallNoTiming_Full Fall Expert
+ DS-Fall-RD A5 Direction Expert
```

This is a task-separated hybrid pipeline. It does not replace the A5 temporal
direction head. It replaces only the final fall decision with a no-timing
classical fall expert.

## 2. Benchmark Assumption

The reported Hybrid_NoTiming results are for the final BITS/WEDA benchmark:

- datasets: BITS and WEDA only
- sampling rate: 25 Hz
- window duration: 2 seconds
- temporal window length: 50 timesteps
- temporal feature shape for A5: 50 x 12
- temporal feature set: tilt12
- split: existing train/val/test split
- no FullTiming feature in the main result
- no test-set threshold tuning

## 3. High-Level Principle

Fall detection and direction classification are separated because they use
different evidence.

Fall evidence is dominated by:

- impact magnitude
- jerk peak
- post-impact stillness
- magnitude/range/dispersion summary
- hard-negative ADL separation

Direction evidence is dominated by:

- signed gyro axis pattern
- roll and pitch transition
- signed pre/post posture change
- temporal pattern around the fall event

Therefore, Hybrid_NoTiming uses:

```text
same 2s IMU window
        |
        |-- summary feature branch -> GradientBoosting fall expert -> fall/non-fall
        |
        `-- tilt12 temporal branch -> A5 direction head -> forward/backward/lateral
```

The final output is:

```text
if fall_pred == non_fall:
    final_output = non_fall
else:
    final_output = direction_pred
```

For end-to-end direction scoring, a supervised fall sample is counted as
correct only when:

```text
fall_pred == fall
and direction_pred == direction_true
```

## 4. Input And Feature Flow

### 4.1 Temporal Input For A5 Direction Expert

A5 uses the tilt12 sequence:

```text
shape = 50 x 12
channels =
    ax, ay, az,
    gx, gy, gz,
    acc_mag,
    gyro_mag,
    jerk,
    roll,
    pitch,
    tilt_delta
```

A5 direction is loaded from saved reference predictions in:

```text
artifacts/experiments_25hz_event/<experiment_id>/predictions.csv
```

In the final runner, Hybrid_NoTiming does not retrain A5. It reuses the A5
direction head predictions.

### 4.2 Summary Input For Fall Expert

The fall expert uses `FallNoTiming_Full`.

Source:

```text
outputs/reports/ml_feature_analysis_25hz/summary_feature_matrix.csv
```

Feature count:

```text
132 features
```

Timing-index artifacts are excluded:

```text
impact_index
impact_distance_from_center
gyro_peak_index
gyro_peak_distance_from_center
any feature containing distance_from_center
any feature containing peak_index as timing index
any feature containing impact_index
```

Important detail: peak values such as `acc_mag_peak_value` and
`gyro_mag_peak_value` are kept because they are signal magnitude values, not
event timing indexes.

## 5. FallNoTiming_Full Feature List

```text
ax_mean
ax_std
ax_min
ax_max
ax_range
ax_final_initial
ax_median
ax_iqr
peak_signed_ax
integrated_ax
ay_mean
ay_std
ay_min
ay_max
ay_range
ay_final_initial
ay_median
ay_iqr
peak_signed_ay
integrated_ay
az_mean
az_std
az_min
az_max
az_range
az_final_initial
az_median
az_iqr
peak_signed_az
integrated_az
gx_mean
gx_std
gx_min
gx_max
gx_range
gx_final_initial
gx_median
gx_iqr
peak_signed_gx
integrated_gx
gy_mean
gy_std
gy_min
gy_max
gy_range
gy_final_initial
gy_median
gy_iqr
peak_signed_gy
integrated_gy
gz_mean
gz_std
gz_min
gz_max
gz_range
gz_final_initial
gz_median
gz_iqr
peak_signed_gz
integrated_gz
acc_mag_mean
acc_mag_std
acc_mag_min
acc_mag_max
acc_mag_p95
acc_mag_range
acc_mag_median
acc_mag_iqr
gyro_mag_mean
gyro_mag_std
gyro_mag_min
gyro_mag_max
gyro_mag_p95
gyro_mag_range
gyro_mag_median
gyro_mag_iqr
jerk_mean
jerk_std
jerk_max
jerk_p95
jerk_median
jerk_iqr
roll_mean
roll_std
roll_min
roll_max
roll_range
roll_final_initial
roll_median
roll_iqr
pitch_mean
pitch_std
pitch_min
pitch_max
pitch_range
pitch_final_initial
pitch_median
pitch_iqr
tilt_delta_mean
tilt_delta_std
tilt_delta_max
tilt_delta_p95
tilt_delta_final
tilt_delta_median
tilt_delta_iqr
acc_mag_peak_value
gyro_mag_peak_value
pre_impact_energy
post_impact_energy
pre_post_energy_ratio
post_acc_mag_std
post_gyro_mag_std
pre_acc_mag_std
pre_gyro_mag_std
pre_ax_mean
post_ax_mean
pre_ay_mean
post_ay_mean
pre_az_mean
post_az_mean
pre_gx_mean
post_gx_mean
pre_gy_mean
post_gy_mean
pre_gz_mean
post_gz_mean
pre_roll_mean
post_roll_mean
pre_pitch_mean
post_pitch_mean
delta_roll_window
delta_pitch_window
```

## 6. Fall Expert Model

Implementation in `scripts/run_final_four_pipelines_e1_e7.py`:

```python
make_pipeline(
    StandardScaler(),
    GradientBoostingClassifier(random_state=42)
)
```

The current sklearn default GradientBoostingClassifier configuration is used:

```text
n_estimators = 100
learning_rate = 0.1
max_depth = 3
subsample = 1.0
criterion = friedman_mse
loss = log_loss
```

Training target:

```text
fall_label: 0 = non-fall, 1 = fall
```

Training data:

- E1 trains on BITS train split.
- E2 trains on WEDA train split.
- E3 trains on BITS+WEDA train split.
- E4 trains on BITS train split and tests WEDA.
- E5 trains on WEDA train split and tests BITS.
- E6 reuses E3 model and tests BITS.
- E7 reuses E3 model and tests WEDA.

## 7. Fall Threshold Selection

Hybrid_NoTiming does not use a fixed 0.5 threshold. It selects the fall
threshold on the validation split of the training protocol only.

Selection rule:

```text
maximize validation fall F1
then prefer recall >= 0.88
then prefer higher precision
then prefer fewer false positives
then prefer higher recall
```

No test data is used for threshold tuning.

Hybrid_NoTiming thresholds from the final run:

| Protocol | Train protocol | Fall threshold |
| --- | --- | ---: |
| E1_BITS_TO_BITS | E1_BITS_TO_BITS | 0.623590 |
| E2_WEDA_TO_WEDA | E2_WEDA_TO_WEDA | 0.718773 |
| E3_BITS_WEDA_MIXED | E3_BITS_WEDA_MIXED | 0.484604 |
| E4_BITS_TO_WEDA | E4_BITS_TO_WEDA | 0.623590 |
| E5_WEDA_TO_BITS | E5_WEDA_TO_BITS | 0.718773 |
| E6_E3_MIXED_TEST_BITS | E3_BITS_WEDA_MIXED | 0.484604 |
| E7_E3_MIXED_TEST_WEDA | E3_BITS_WEDA_MIXED | 0.484604 |

## 8. Direction Expert

Direction decision is not produced by the GradientBoosting fall model.

Direction source:

```text
A5 direction head
```

Input:

```text
tilt12 sequence, 50 x 12
```

Output:

```text
forward / backward / lateral
```

Direction metrics are computed only for samples satisfying:

```text
fall_label == 1
direction_supervised == True
direction_label in {forward, backward, lateral}
```

This is why Hybrid_NoTiming keeps the same direction macro F1 as A5 for each
protocol.

## 9. Complexity And Edge Capability

| Component | Value |
| --- | --- |
| A5 deep params used for direction | 65,959 |
| A5 FP32 size estimate | 257.65 KB |
| A5 INT8 size estimate | 64.41 KB |
| Fall feature count | 132 |
| Direction temporal channel count | 12 |
| Fall model trees | 100 |
| Fall tree nodes | 1,408 |
| Fall tree leaves | 754 |
| Direction tree nodes | 0 |
| Estimated complexity units | 67,367 |
| Edge suitability | medium |

Interpretation:

- The fall expert itself is light: 100 shallow boosting trees and 132 scalar
  features.
- The full Hybrid_NoTiming pipeline still needs A5 for direction, so edge
  deployment needs either TensorFlow Lite or another compact inference path for
  the A5 direction head.
- Compared with A5 alone, Hybrid_NoTiming adds only the summary feature
  extractor and 100 boosting trees.
- Compared with ML_only_separated, Hybrid_NoTiming is less CPU-only, but more
  direction-consistent across E1/E6/E7.

## 10. Final E1-E7 Results For Hybrid_NoTiming

| Experiment | Test | Fall F1 | Fall Precision | Fall Recall | FP | FN | Direction Macro F1 | E2E Direction F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1_BITS_TO_BITS | bits | 0.8889 | 0.9524 | 0.8333 | 2 | 8 | 0.8777 | 0.7487 |
| E2_WEDA_TO_WEDA | weda | 0.8800 | 0.8800 | 0.8800 | 3 | 3 | 0.6071 | 0.5844 |
| E3_BITS_WEDA_MIXED | bits+weda | 0.8732 | 0.8986 | 0.8493 | 7 | 11 | 0.8967 | 0.7998 |
| E4_BITS_TO_WEDA | weda | 0.4364 | 0.2824 | 0.9600 | 61 | 1 | 0.3876 | 0.3515 |
| E5_WEDA_TO_BITS | bits | 0.7857 | 0.9167 | 0.6875 | 3 | 15 | 0.6271 | 0.4869 |
| E6_E3_MIXED_TEST_BITS | bits | 0.8966 | 1.0000 | 0.8125 | 0 | 9 | 0.9350 | 0.7798 |
| E7_E3_MIXED_TEST_WEDA | weda | 0.8364 | 0.7667 | 0.9200 | 7 | 2 | 0.8390 | 0.7926 |

## 11. Why Hybrid_NoTiming Improves WEDA Fall

A5 on E7 WEDA:

```text
Fall F1 = 0.6857
Precision = 0.5333
Recall = 0.9600
FP = 21
FN = 1
```

Hybrid_NoTiming on E7 WEDA:

```text
Fall F1 = 0.8364
Precision = 0.7667
Recall = 0.9200
FP = 7
FN = 2
```

Main effect:

```text
WEDA false positives: 21 -> 7
FP reduction: 14
```

This means the hybrid fall expert is better at rejecting WEDA hard-negative
ADL windows. It sacrifices one additional false negative compared with A5, but
the large false-positive reduction improves WEDA Fall F1 substantially.

## 12. What Hybrid_NoTiming Is Not

Hybrid_NoTiming is not:

- a new deep neural architecture
- a replacement for A5 direction
- a FullTiming/event-index model
- a dataset-aware threshold model
- a test-tuned model

It is a clean task-separated practical pipeline:

```text
fall decision: no-timing summary ML
direction decision: A5 temporal direction head
```

## 13. Paper-Facing Statement

Use this wording:

```text
The final practical DS-Fall-RD-Hybrid-NoTiming pipeline separates fall
detection from direction classification. A GradientBoosting fall expert trained
on no-timing signal summaries reduces WEDA hard-negative false positives,
whereas the DS-Fall-RD A5 temporal direction head is retained for stable
direction classification. This preserves the rotation-aware direction
representation while improving fall precision on WEDA without using
event-timing artifacts.
```

## 14. Recommended Final Role

Recommended use:

- `Hybrid_NoTiming` as the main practical pipeline.
- `A5_reference` as the main deep temporal baseline and direction expert.
- `EdgeLite_Hybrid` as compact ablation.
- `ML_only_separated` as analysis-only, because its direction result is not as
  stable across E1/E6/E7 even though it is strong on E7 WEDA.

