# BITS UMAFall WEDA DS-Fall-RD Check

## Processed Dataset

- Datasets: bits, umafall, weda
- X shape: (6414, 100, 6)
- Windows by dataset: {'bits': 3557, 'weda': 1879, 'umafall': 978}
- Fall labels: {'0': 5531, '1': 883}
- Direction labels: {'none': 5531, 'forward': 318, 'lateral': 244, 'backward': 240, 'other': 81}
- UMAFall preprocessing: WRIST SensorTag accelerometer+gyroscope, timestamp interpolation to 50Hz
- BITS preprocessing: 20Hz row-order uniform interpolation to 50Hz before 2-second windowing

## Data Quality

- Supervised direction samples: 802
- NaN/Inf before features: {'nan': 0, 'inf': 0}
- NaN/Inf after features: {'nan': 0, 'inf': 0}
- Impact boundary summary: {'num_fall_windows': 883, 'early': 20, 'valid': 849, 'late': 14, 'early_pct': 2.2650056625141564, 'valid_pct': 96.14949037372594, 'late_pct': 1.5855039637599093}
- Warnings: none
- Handcrafted direction baseline macro F1: 0.6458

## Ablation Results

| id | direction_loss_type | fall_loss_weighted | fall_f1 | direction_macro_f1 | direction_accuracy | params | estimated_int8_kb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A5 | focal | False | 0.8611 | 0.6974 | 0.7143 | 65959 | 64.4131 |
| A5FW | focal | True | 0.8720 | 0.6268 | 0.6429 | 65959 | 64.4131 |
| A5WCEFW | weighted_ce | True | 0.8382 | 0.5610 | 0.6020 | 65959 | 64.4131 |

## Selected Model

- Selected: A5
- Fall F1: 0.8611
- Direction macro F1: 0.6974
- Params: 65959
- Estimated INT8 size: 64.41 KB
- Best model path: outputs\models\ds_fall_rd_a5_best.keras

## A5 Per Dataset

- bits: fall F1=0.8966, direction macro F1=0.7804, direction supervised=36
- umafall: fall F1=0.8684, direction macro F1=0.5600, direction supervised=37
- weda: fall F1=0.7925, direction macro F1=0.7594, direction supervised=25
