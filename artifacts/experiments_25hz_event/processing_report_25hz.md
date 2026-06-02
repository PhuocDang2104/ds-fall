# BITS + WEDA 25 Hz Processing Report

This audit uses all windows only for data inspection. Each training experiment fits its scaler on that experiment's train split only.

## Configuration

- sampling_rate: 25 Hz
- window_seconds: 2.0
- window_size: 50
- stride_seconds: 0.5
- stride_size: 12
- input_shape: (50, 12)
- window_mode: event_centered
- feature_order: ax, ay, az, gx, gy, gz, acc_mag, gyro_mag, jerk, roll, pitch, tilt_delta
- WEDA source: raw dataset/25Hz folder, not 50Hz downsampling
- BITS source: raw 20Hz row-order sequence interpolated to 25Hz per trial

## Dataset Audit

| dataset | source_folder | original_sampling_rate | target_sampling_rate | resampling_method | n_subjects | n_trials | raw_samples | effective_fs_before_mean | effective_fs_after_mean | n_windows | fall_windows | non_fall_windows | direction_supervised_windows | direction_class_distribution | feature_shape | nan_count | inf_count | acc_mag_median | acc_mag_p95 | acc_mag_p99 | gyro_mag_median | gyro_mag_p95 | gyro_mag_p99 | unit_map |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bits | C:\Users\ADMIN\Desktop\ds-fall\data\raw\Dataset | 20.0000 | 25.0000 | ['uniform_20hz_time_interpolation_to_25hz'] | 41 | 975 | 625393.0000 | 20.0000 | 25.0000 | 3560 | 325 | 3235 | 244 | {"forward": 121, "lateral": 82, "backward": 41} | (3560, 50, 12) | 0.0000 | 0.0000 | 9.8372 | 17.3699 | 33.3119 | 0.4645 | 4.2172 | 7.0662 | {"acc_unit_before": "m/s^2", "gyro_unit_before": "rad/s", "acc_unit_after": "m/s^2", "gyro_unit_after": "rad/s", "conversion_factor_acc": 1.0, "conversion_factor_gyro": 1.0} |
| weda | C:\Users\ADMIN\Desktop\ds-fall\data\raw\WEDA-FALL-main\dataset\25Hz | 25.0000 | 25.0000 | ['raw_25hz_time_aligned_to_uniform_25hz'] | 25 | 584 | 304155.0000 | 24.7887 | 25.0000 | 1895 | 350 | 1545 | 350 | {"forward": 126, "backward": 126, "lateral": 98} | (1895, 50, 12) | 0.0000 | 0.0000 | 9.8879 | 19.6797 | 35.4441 | 1.1474 | 5.7253 | 9.2309 | {"acc_unit_before": "m/s^2", "gyro_unit_before": "rad/s", "acc_unit_after": "m/s^2", "gyro_unit_after": "rad/s", "conversion_factor_acc": 1.0, "conversion_factor_gyro": 1.0} |

## Feature Statistics

| feature | mean | std | min | max | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- |
| ax | -3.9898 | 7.5423 | -78.4436 | 78.4269 | 8.6696 | 12.8110 |
| ay | -3.2406 | 5.1257 | -78.4163 | 78.2736 | 3.9394 | 9.0534 |
| az | 3.0431 | 5.0855 | -78.2760 | 78.4264 | 9.6477 | 14.6677 |
| gx | -0.0801 | 1.4407 | -23.6084 | 26.7095 | 1.8608 | 3.9802 |
| gy | -0.0159 | 1.0459 | -18.8412 | 17.7963 | 1.4260 | 3.2175 |
| gz | -0.0235 | 1.3076 | -19.3488 | 16.4359 | 1.8788 | 4.2182 |
| acc_mag | 10.8282 | 5.2395 | 0.1138 | 134.6576 | 18.2763 | 34.1584 |
| gyro_mag | 1.3259 | 1.7687 | 0.0001 | 27.2049 | 4.8373 | 7.9498 |
| jerk | 0.1694 | 59.8731 | -1387.2000 | 1274.3146 | 52.0250 | 201.4208 |
| roll | -0.6749 | 0.9980 | -3.1413 | 3.1416 | 1.4071 | 2.8263 |
| pitch | 0.4787 | 0.7871 | -1.5707 | 1.5643 | 1.3795 | 1.4673 |
| tilt_delta | 0.0444 | 3.1569 | -55.4035 | 48.8777 | 3.8884 | 10.9613 |

## Trial Audit Files

- `bits_trial_audit_25hz.csv`
- `weda_trial_audit_25hz.csv`

## Unit Notes

- Final accelerometer unit: m/s^2.
- Final gyroscope unit: rad/s.
- BITS and WEDA use conversion factor 1.0 from the current dataset unit map.
- Unit statistics are saved to `unit_audit_25hz.csv`.

