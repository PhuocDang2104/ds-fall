# DS-Fall-RD

DS-Fall-RD means Rotation-aware Direction-sensitive DS-Fall. The upgrade keeps
the original lightweight DS-Fall design, but gives the direction head better
signed orientation evidence:

- `acc_mag`, `gyro_mag`, `jerk` for impact and abrupt motion.
- `roll`, `pitch`, `tilt_delta` for signed body/sensor tilt transitions.
- task-specific attention so fall detection and direction classification can
  pool different temporal regions.
- optional class-weighted focal direction loss for the small supervised
  direction subset.

Energy and spectral entropy are intentionally not part of the main model. They
can help ADL/noise separation, but fall direction is more directly tied to
signed axis motion, roll/pitch transition, and pre-impact gyroscope patterns.

## Feature Sets

| ID | Channels | Shape |
| --- | --- | --- |
| `raw6` | `ax ay az gx gy gz` | `100 x 6` |
| `mag_jerk9` | raw6 + `acc_mag gyro_mag jerk` | `100 x 9` |
| `roll_pitch11` | mag_jerk9 + `roll pitch` | `100 x 11` |
| `tilt12` | roll_pitch11 + `tilt_delta` | `100 x 12` |

Feature engineering must run before standardization. If `data/processed/X.npy`
was created by notebook 01, the ablation runner uses `data/processed/scaler.pkl`
to recover raw6 values before generating RD features, then fits a new feature
scaler on the train split only.

## Run

After notebook 01 has created `data/processed`, run:

```bash
python scripts/run_ds_fall_rd_ablation.py --epochs 80 --batch-size 64
```

Run only the RD candidates:

```bash
python scripts/run_ds_fall_rd_ablation.py --ablation A4 A5 A6
```

Outputs:

- `outputs/reports/data_quality_report.md`
- `outputs/reports/ablation_results.csv`
- `outputs/reports/ablation_results.md`
- `outputs/reports/model_selection.md`
- `outputs/figures/confusion_matrices/`
- `outputs/figures/direction_signal_diagnostics/`
- `outputs/runs/<ablation_id>_*/feature_config.json`
- `outputs/runs/<ablation_id>_*/feature_scaler.pkl`
- `outputs/runs/<ablation_id>_*/model_config.json`
- `outputs/runs/<ablation_id>_*/per_subject_metrics.csv`

Model selection should prioritize direction macro F1, then fall F1,
per-dataset consistency, and model size/latency. Do not select by overall
accuracy alone.
