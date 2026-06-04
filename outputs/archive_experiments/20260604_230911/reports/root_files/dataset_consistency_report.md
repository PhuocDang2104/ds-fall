# Dataset Consistency Diagnostics

This is a data audit only. It summarizes all available processed windows and does not fit any DS-Fall model or create train-time scalers.

- Processed windows: 7743
- Window shape: (100, 6)
- Channel order source: scaler.pkl channel_names = [ax, ay, az, gx, gy, gz]
- Input note: Loaded X from data\processed\X.npy. X.npy was converted back to raw6 values using scaler.pkl.

## bits

### Basic Count

- Total windows: 3557
- Split counts: {'train': 2501, 'val': 528, 'test': 528}
- Fall / non-fall: 325 / 3232
- Supervised direction count: 244
- Direction class count: {'forward': 121, 'backward': 41, 'lateral': 82}
- Subject count: 41
- Trial count: not available
- Activity/fall type count: {'adl1': 205, 'adl12': 205, 'adl13': 205, 'adl2': 205, 'adl5': 205, 'adl9': 203, 'adl10': 201, 'adl3': 201, 'adl4': 201, 'adl8': 201, 'adl11': 200, 'adl14': 200, 'adl15': 200, 'adl16': 200, 'adl6': 200, 'adl7': 200, 'fall2': 41, 'fall3': 41, 'fall5': 41, 'fall6': 41, 'fall7': 41, 'fall1': 40, 'fall4': 40, 'fall8': 40}

### Sampling Check

- Original sampling note: BITS raw files are treated as 20 Hz and resampled to 50 Hz by row-order interpolation.
- Sampling rows written to `dataset_sampling_check.csv`: 2

### Gravity Check

- acc_mag mean/median/std: 10.7819 / 9.8397 / 5.5997
- acc_mag p05/p25/p75/p95/p99: 6.3084 / 9.5952 / 10.2092 / 18.4159 / 34.8573
- % in [0.7, 1.3]: 0.19
- % in [7.0, 12.5]: 81.92
- % < 0.4: 0.02
- % < 4.0: 2.16
- Conclusion: likely contains gravity, likely_unit=m/s^2. Median acceleration magnitude is near 9.81 m/s^2.

### Impact/Window Quality

- acc_mag peak mean/median: 59.9596 / 56.4518
- gyro_mag peak mean/median: 8.7322 / 8.0047
- jerk peak mean/median: 1095.3512 / 858.0766
- impact_index mean/median: 51.82 / 50.00
- impact early/valid/late %: 0.62 / 95.69 / 3.69

### Direction Separability

- Best baseline macro F1: 0.7926829268292682
- Rows written to `dataset_direction_baseline.csv`: 2

### Warnings

- BITS raw timestamp column is rounded/duplicated in these CSVs; current preprocessing intentionally uses configured row-order 20 Hz before resampling to 50 Hz.

## hifd

### Basic Count

- Total windows: 1329
- Split counts: {'train': 959, 'val': 195, 'test': 175}
- Fall / non-fall: 104 / 1225
- Supervised direction count: 104
- Direction class count: {'forward': 33, 'backward': 34, 'lateral': 37}
- Subject count: 21
- Trial count: not available
- Activity/fall type count: {'chair': 105, 'clap': 105, 'cloth': 105, 'shoe': 105, 'teeth': 105, 'walk': 105, 'wash': 105, 'zip': 105, 'stair': 95, 'bed': 95, 'hair': 75, 'eat': 65, 'write': 55, 'fall1': 21, 'fall2': 21, 'fall3': 21, 'fall6': 16, 'fall5': 13, 'fall4': 12}

### Sampling Check

- Original sampling note: HIFD processed windows are expected at 50 Hz.
- Sampling rows written to `dataset_sampling_check.csv`: 1

### Gravity Check

- acc_mag mean/median/std: 3.9588 / 2.4614 / 6.6091
- acc_mag p05/p25/p75/p95/p99: 0.1312 / 0.9807 / 5.0518 / 11.6372 / 23.0297
- % in [0.7, 1.3]: 11.36
- % in [7.0, 12.5]: 10.35
- % < 0.4: 13.64
- % < 4.0: 66.90
- Conclusion: likely gravity-removed, likely_unit=m/s^2-like. Median acceleration magnitude is below 4.0.

### Impact/Window Quality

- acc_mag peak mean/median: 130.7096 / 129.6519
- gyro_mag peak mean/median: 14.9162 / 13.4737
- jerk peak mean/median: 5225.8330 / 5262.7217
- impact_index mean/median: 50.00 / 50.00
- impact early/valid/late %: 0.00 / 100.00 / 0.00

### Direction Separability

- Best baseline macro F1: 0.5992028985507246
- Rows written to `dataset_direction_baseline.csv`: 2

### Warnings

- hifd: raw timestamp sampling check not available; using processed window metadata only.

## weda

### Basic Count

- Total windows: 1879
- Split counts: {'train': 1488, 'val': 219, 'test': 172}
- Fall / non-fall: 350 / 1529
- Supervised direction count: 350
- Direction class count: {'forward': 126, 'backward': 126, 'lateral': 98}
- Subject count: 25
- Trial count: not available
- Activity/fall type count: {'D10': 250, 'D09': 239, 'D05': 140, 'D07': 140, 'D08': 135, 'D01': 125, 'D04': 125, 'D11': 125, 'D03': 110, 'D02': 70, 'D06': 70, 'F08': 56, 'F01': 42, 'F02': 42, 'F03': 42, 'F04': 42, 'F05': 42, 'F06': 42, 'F07': 42}

### Sampling Check

- Original sampling note: WEDA processed windows are expected at 50 Hz.
- Sampling rows written to `dataset_sampling_check.csv`: 82

### Gravity Check

- acc_mag mean/median/std: 11.2557 / 9.8853 / 5.7921
- acc_mag p05/p25/p75/p95/p99: 6.5924 / 9.3904 / 11.1086 / 20.4085 / 40.7418
- % in [0.7, 1.3]: 0.04
- % in [7.0, 12.5]: 77.68
- % < 0.4: 0.00
- % < 4.0: 1.20
- Conclusion: likely contains gravity, likely_unit=m/s^2. Median acceleration magnitude is near 9.81 m/s^2.

### Impact/Window Quality

- acc_mag peak mean/median: 53.5828 / 55.6184
- gyro_mag peak mean/median: 13.5259 / 12.1033
- jerk peak mean/median: 1659.5706 / 1639.6805
- impact_index mean/median: 45.22 / 50.00
- impact early/valid/late %: 5.14 / 94.57 / 0.29

### Direction Separability

- Best baseline macro F1: 0.7141078336804637
- Rows written to `dataset_direction_baseline.csv`: 2

### Warnings

- weda raw timestamp median fs=500.00 Hz differs from expected/configured 50.0 Hz.

## umafall

### Basic Count

- Total windows: 978
- Split counts: {'train': 638, 'test': 192, 'val': 148}
- Fall / non-fall: 208 / 770
- Supervised direction count: 208
- Direction class count: {'forward': 71, 'backward': 73, 'lateral': 64}
- Subject count: 19
- Trial count: 18
- Activity/fall type count: {'Sitting_GettingUpOnAChair': 95, 'Walking': 95, 'Bending': 85, 'Hopping': 75, 'LyingDown_OnABed': 75, 'backwardFall': 73, 'forwardFall': 71, 'lateralFall': 64, 'Aplausing': 60, 'HandsUp': 60, 'MakingACall': 60, 'OpeningDoor': 60, 'Jogging': 45, 'GoDownstairs': 30, 'GoUpstairs': 30}

### Sampling Check

- Original sampling note: UMAFall SensorTag wrist streams are treated as about 20 Hz and resampled to 50 Hz by timestamp interpolation.
- Sampling rows written to `dataset_sampling_check.csv`: 82

### Gravity Check

- acc_mag mean/median/std: 11.3469 / 10.0858 / 6.7985
- acc_mag p05/p25/p75/p95/p99: 5.5839 / 9.2214 / 11.2983 / 21.5624 / 39.8765
- % in [0.7, 1.3]: 0.16
- % in [7.0, 12.5]: 74.37
- % < 0.4: 0.01
- % < 4.0: 2.60
- Conclusion: likely contains gravity, likely_unit=m/s^2. Median acceleration magnitude is near 9.81 m/s^2.

### Impact/Window Quality

- acc_mag peak mean/median: 63.9351 / 59.7442
- gyro_mag peak mean/median: 6.6449 / 6.7165
- jerk peak mean/median: 1418.1770 / 1277.5483
- impact_index mean/median: 50.16 / 50.00
- impact early/valid/late %: 0.00 / 99.52 / 0.48

### Direction Separability

- Best baseline macro F1: 0.7461193200957766
- Rows written to `dataset_direction_baseline.csv`: 2

### Warnings

- none

## Unit/Scale Cross-Dataset Warnings

- none

## Cross-Dataset Comparability Table

| dataset | n_windows | n_fall | n_nonfall | n_direction | direction_counts | acc_mag_median | gravity_status | likely_unit | gyro_range_note | impact_valid_pct | direction_baseline_macro_f1 | main_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bits | 3557 | 325 | 3232 | 244 | {"forward": 121, "backward": 41, "lateral": 82} | 9.8397 | likely contains gravity | m/s^2 | gyro scale unclear/intermediate | 95.6923 | 0.7927 | BITS raw timestamp column is rounded/duplicated in these CSVs; current preprocessing intentionally uses configured row-order 20 Hz before resampling to 50 Hz. |
| hifd | 1329 | 104 | 1225 | 104 | {"forward": 33, "backward": 34, "lateral": 37} | 2.4614 | likely gravity-removed | m/s^2-like | gyro scale unclear/intermediate | 100.0000 | 0.5992 | hifd: raw timestamp sampling check not available; using processed window metadata only. |
| weda | 1879 | 350 | 1529 | 350 | {"forward": 126, "backward": 126, "lateral": 98} | 9.8853 | likely contains gravity | m/s^2 | gyro scale unclear/intermediate | 94.5714 | 0.7141 | weda raw timestamp median fs=500.00 Hz differs from expected/configured 50.0 Hz. |
| umafall | 978 | 208 | 770 | 208 | {"forward": 71, "backward": 73, "lateral": 64} | 10.0858 | likely contains gravity | m/s^2 | gyro scale unclear/intermediate | 99.5192 | 0.7461 |  |

## Channel Stats Preview

| dataset | channel | mean | std | median | p01 | p99 | min | max | abs_p99 | range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bits | ax | -4.5594 | 7.8169 | -7.0883 | -24.8509 | 11.2920 | -78.4436 | 78.4269 | 24.8509 | 156.8705 |
| bits | ay | -2.9687 | 5.2465 | -2.5259 | -22.5970 | 8.6014 | -78.4163 | 78.4197 | 22.5970 | 156.8360 |
| bits | az | 2.8641 | 4.6012 | 2.2960 | -9.7425 | 11.5233 | -78.4460 | 78.4264 | 11.5233 | 156.8724 |
| bits | gx | -0.0284 | 1.2108 | -0.0012 | -3.8839 | 3.5638 | -31.0417 | 22.4982 | 3.8839 | 53.5399 |
| bits | gy | 0.0064 | 0.9140 | 0.0005 | -2.9187 | 2.9001 | -17.3168 | 16.1000 | 2.9187 | 33.4168 |
| bits | gz | -0.0065 | 1.2696 | 0.0012 | -4.3371 | 4.1612 | -14.9344 | 16.4359 | 4.3371 | 31.3704 |
| hifd | ax | -0.6805 | 4.1013 | -0.0785 | -12.4447 | 6.4430 | -153.5133 | 148.3158 | 12.4447 | 301.8291 |
| hifd | ay | 0.0338 | 5.0313 | 0.0785 | -12.9056 | 11.1404 | -153.4152 | 164.5556 | 12.9056 | 317.9708 |
| hifd | az | -0.2410 | 4.0860 | -0.0686 | -9.7087 | 8.6200 | -146.8546 | 156.4945 | 9.7087 | 303.3491 |
| hifd | gx | 0.0134 | 1.6885 | 0.0000 | -4.9410 | 5.0754 | -34.8699 | 25.2497 | 5.0754 | 60.1196 |
| hifd | gy | -0.0156 | 1.0821 | 0.0000 | -3.2184 | 3.1521 | -25.1589 | 16.9698 | 3.2184 | 42.1288 |
| hifd | gz | 0.0289 | 1.6112 | 0.0017 | -4.4122 | 5.1069 | -22.2564 | 33.2904 | 5.1069 | 55.5468 |
| weda | ax | -2.8359 | 7.3399 | -3.1000 | -21.4000 | 15.8000 | -39.0000 | 39.2000 | 21.4000 | 78.2000 |
| weda | ay | -3.8035 | 5.7384 | -3.6000 | -24.3000 | 11.2000 | -39.2000 | 39.1000 | 24.3000 | 78.3000 |
| weda | az | 3.3071 | 6.3238 | 3.0000 | -14.9000 | 20.1000 | -39.2000 | 39.2000 | 20.1000 | 78.4000 |
| weda | gx | -0.0966 | 1.9052 | 0.0000 | -6.2000 | 5.4000 | -32.9000 | 31.7000 | 6.2000 | 64.6000 |
| weda | gy | -0.0291 | 1.3262 | 0.0000 | -4.3000 | 4.0000 | -21.1000 | 25.2000 | 4.3000 | 46.3000 |
| weda | gz | -0.0114 | 1.4752 | 0.0000 | -4.8000 | 4.7000 | -25.6000 | 16.9000 | 4.8000 | 42.5000 |
| umafall | ax | -1.1037 | 9.0488 | -1.8166 | -24.2501 | 19.5607 | -77.8379 | 78.4532 | 24.2501 | 156.2911 |
| umafall | ay | -0.2180 | 6.6764 | -0.2544 | -19.0393 | 17.0395 | -78.4532 | 77.8379 | 19.0393 | 156.2911 |
| umafall | az | 3.0351 | 6.1676 | 2.5857 | -12.3443 | 22.2507 | -76.5729 | 78.4532 | 22.2507 | 155.0261 |
| umafall | gx | -0.0575 | 1.3707 | -0.0672 | -4.0762 | 4.2695 | -4.4680 | 4.4437 | 4.2695 | 8.9118 |
| umafall | gy | -0.0320 | 1.2097 | -0.0021 | -4.1891 | 3.8765 | -4.4834 | 4.4330 | 4.1891 | 8.9164 |
| umafall | gz | 0.0029 | 1.2707 | -0.0282 | -4.0306 | 4.3724 | -4.4722 | 4.4445 | 4.3724 | 8.9167 |

## Direction Baseline Table

| dataset | classifier | n_supervised | n_train | n_test | accuracy | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| bits | logistic_regression | 244 | 170 | 74 | 0.6081 | 0.6029 |
| bits | random_forest | 244 | 170 | 74 | 0.8108 | 0.7927 |
| hifd | logistic_regression | 104 | 72 | 32 | 0.5000 | 0.4970 |
| hifd | random_forest | 104 | 72 | 32 | 0.6250 | 0.5992 |
| weda | logistic_regression | 350 | 245 | 105 | 0.6286 | 0.6183 |
| weda | random_forest | 350 | 245 | 105 | 0.7238 | 0.7141 |
| umafall | logistic_regression | 208 | 145 | 63 | 0.6190 | 0.6187 |
| umafall | random_forest | 208 | 145 | 63 | 0.7460 | 0.7461 |

## Diagnosis

1. Which datasets likely contain gravity? ['bits', 'weda', 'umafall'].
2. Which datasets are likely gravity-removed? ['hifd'].
3. Which dataset has unit/scale mismatch? none detected by >5x rule.
4. Which dataset has poor direction separability? none by current thresholds.
5. Which dataset has poor impact alignment? none by current thresholds.
6. Whether UMAFall appears synchronized and comparable with BITS/WEDA/HIFD. UMAFall appears synchronized at processed-window level (impact valid 99.5%). Comparability depends on its gravity/unit result: likely contains gravity, m/s^2.
