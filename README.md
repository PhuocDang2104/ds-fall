# DS-Fall

Dual-Stream Depthwise-Separable Temporal Network for Direction-Sensitive Fall Detection.

DS-Fall trains a lightweight edge-AI model for wrist-worn 6-axis IMU signals. Every processed sample uses the same representation:

```text
X shape: 100 x 6
sampling: 50 Hz
duration: 2 seconds
channels: [ax, ay, az, gx, gy, gz]
```

## Dataset Placement

Place the raw datasets under:

```text
/content/drive/MyDrive/ds-fall/data/raw/
  Dataset/
  HR_IMU_falldetection_dataset-master/
  WEDA-FALL-main/
```

On Windows this repo can also be developed from:

```text
C:\Users\ADMIN\Desktop\ds-fall
```

Raw data is intentionally ignored by git.

## Google Colab Usage

1. Upload this repository to Google Drive at:

```text
/content/drive/MyDrive/ds-fall
```

2. Put raw datasets in:

```text
/content/drive/MyDrive/ds-fall/data/raw/
```

3. Run:

```text
notebooks/01_process_visualize_data.ipynb
```

4. Then run:

```text
notebooks/02_train_ds_fall_model.ipynb
```

5. Check outputs:

```text
outputs/models/
outputs/figures/
outputs/metrics/
outputs/logs/
```

## Data Processing Logic

WEDA-FALL and HIFD are already 50 Hz. They are windowed directly into 2-second windows of 100 samples.

BITS-2 / `Dataset` is originally 20 Hz for motion sensors. The main pipeline always converts it to 50 Hz before windowing:

```python
t_original = np.arange(N) / 20.0
t_target = np.arange(0.0, t_original[-1] + 1e-9, 1.0 / 50.0)
```

Each channel is interpolated independently. The local BITS timestamp column is not used because it can be rounded, repeated, or unreliable. BITS windows in the processed dataset are therefore also 2 seconds at 50 Hz, never 20 Hz / 5 seconds.

## Model Summary

DS-Fall splits the 6-axis IMU input into two streams:

- accelerometer stream: impact, gravity/posture change, motion intensity
- gyroscope stream: rotation, useful for forward/backward/lateral direction

The streams use Conv1D and depthwise-separable Conv1D encoders, then fuse features with pointwise Conv1D. A dilated depthwise-separable temporal encoder captures pre-fall, impact, and post-fall dynamics. Gated attention pooling focuses on informative timesteps. The model has two heads:

- binary fall detection
- fall direction classification: forward, backward, lateral

Direction loss is masked and applied only to true fall samples with clear direction labels.

## Processed Outputs

Notebook 01 writes:

```text
data/processed/X.npy
data/processed/y_fall.npy
data/processed/y_direction.npy
data/processed/direction_mask.npy
data/processed/metadata.csv
data/processed/scaler.pkl
data/processed/label_mapping.json
data/processed/split_subjects.json
data/processed/preprocessing_summary.json
```

Notebook 02 writes:

```text
outputs/models/ds_fall_best.keras
outputs/models/ds_fall_final.keras
outputs/models/ds_fall_float32.tflite
outputs/models/ds_fall_dynamic_range.tflite
outputs/metrics/test_metrics.json
outputs/figures/training/
outputs/logs/training_log.csv
```

## Subject-Wise Split

Splits are subject-wise, not window-wise. A subject cannot appear in more than one split. This avoids leakage from adjacent windows of the same trial.
