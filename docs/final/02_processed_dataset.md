# Processed Dataset

Processed data is expected under `data/processed/` and final feature summaries
are loaded from `outputs/reports/ml_feature_analysis_25hz/summary_feature_matrix.csv`.

## Labels

- `fall_label`: binary non-fall/fall target.
- `direction_label`: forward, backward, lateral, or none.
- `direction_supervised`: whether direction loss/metric is valid.
- `dataset`: bits or weda for the final benchmark.
- `split`: train, val, or test.
- `subject_id`: used when available for audit and reporting.

## Temporal Feature Set

The main temporal input is tilt12:

- ax, ay, az
- gx, gy, gz
- acc_mag
- gyro_mag
- jerk
- roll
- pitch
- tilt_delta

## Summary Feature Sets

- FallNoTiming_Core: compact fall summary features.
- FallNoTiming_Full: all signal-derived summary features without timing indexes.
- DirectionCore: signed gyro/acc/roll/pitch direction features.
- DirectionFullNoTiming: all signal-derived direction features without timing indexes.
- FullTiming: upper-bound only, not a main result.
