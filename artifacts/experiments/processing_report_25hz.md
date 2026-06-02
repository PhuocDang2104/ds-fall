# BITS + WEDA 25 Hz Processing Report

This audit uses all windows only for data inspection. Each training experiment fits its scaler on that experiment's train split only.

## Configuration

- sampling_rate: 25 Hz
- window_seconds: 2.0
- window_size: 50
- stride_seconds: 0.5
- stride_size: 12
- input_shape: (50, 12)
- feature_order: ax, ay, az, gx, gy, gz, acc_mag, gyro_mag, jerk, roll, pitch, tilt_delta
- WEDA source: raw dataset/25Hz folder, not 50Hz downsampling
- BITS source: raw 20Hz row-order sequence interpolated to 25Hz per trial

## Dataset Audit

| dataset | source_folder | original_sampling_rate | target_sampling_rate | resampling_method | n_subjects | n_trials | raw_samples | effective_fs_before_mean | effective_fs_after_mean | n_windows | fall_windows | non_fall_windows | direction_supervised_windows | direction_class_distribution | feature_shape | nan_count | inf_count | acc_mag_median | acc_mag_p95 | acc_mag_p99 | gyro_mag_median | gyro_mag_p95 | gyro_mag_p99 | unit_map |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bits | C:\Users\ADMIN\Desktop\ds-fall\data\raw\Dataset | 20.0000 | 25.0000 | ['uniform_20hz_time_interpolation_to_25hz'] | 41 | 975 | 625393.0000 | 20.0000 | 25.0000 | 61531 | 21775 | 39756 | 15451 | {"forward": 6947, "lateral": 6362, "backward": 2142} | (61531, 50, 12) | 0.0000 | 0.0000 | 9.8512 | 18.8104 | 35.0259 | 0.4148 | 4.4817 | 7.0180 | {"acc_unit_before": "m/s^2", "gyro_unit_before": "rad/s", "acc_unit_after": "m/s^2", "gyro_unit_after": "rad/s", "conversion_factor_acc": 1.0, "conversion_factor_gyro": 1.0} |
| weda | C:\Users\ADMIN\Desktop\ds-fall\data\raw\WEDA-FALL-main\dataset\25Hz | 25.0000 | 25.0000 | ['raw_25hz_time_aligned_to_uniform_25hz'] | 25 | 969 | 304155.0000 | 24.7887 | 25.0000 | 22044 | 2224 | 19820 | 2224 | {"forward": 803, "backward": 795, "lateral": 626} | (22044, 50, 12) | 0.0000 | 0.0000 | 9.8421 | 16.2214 | 31.4091 | 0.9277 | 4.8648 | 7.8624 | {"acc_unit_before": "m/s^2", "gyro_unit_before": "rad/s", "acc_unit_after": "m/s^2", "gyro_unit_after": "rad/s", "conversion_factor_acc": 1.0, "conversion_factor_gyro": 1.0} |

## Feature Statistics

| feature | mean | std | min | max | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- |
| ax | -3.5881 | 6.7832 | -78.4436 | 78.4269 | 7.6184 | 10.5010 |
| ay | -3.0114 | 5.9616 | -78.4163 | 78.2736 | 5.4473 | 9.0415 |
| az | 2.5245 | 5.4027 | -78.2760 | 78.4264 | 9.5117 | 11.7663 |
| gx | -0.0149 | 1.2868 | -31.0417 | 27.8393 | 1.8000 | 3.8791 |
| gy | 0.0035 | 0.9278 | -18.8412 | 26.2916 | 1.3813 | 3.0836 |
| gz | -0.0121 | 1.2640 | -28.2000 | 26.9150 | 1.8253 | 4.3447 |
| acc_mag | 10.7180 | 4.9176 | 0.1048 | 134.6576 | 17.9981 | 34.0768 |
| gyro_mag | 1.1956 | 1.6387 | 0.0001 | 37.5196 | 4.5856 | 7.2209 |
| jerk | 0.0015 | 47.5784 | -2154.0701 | 1906.1859 | 37.0083 | 168.1599 |
| roll | -0.5201 | 1.2543 | -3.1416 | 3.1416 | 2.5154 | 3.0454 |
| pitch | 0.4159 | 0.7229 | -1.5707 | 1.5699 | 1.3583 | 1.4631 |
| tilt_delta | 0.0022 | 2.2074 | -68.0089 | 54.6202 | 2.3341 | 7.2913 |

## Trial Audit Files

- `bits_trial_audit_25hz.csv`
- `weda_trial_audit_25hz.csv`

## Unit Notes

- Final accelerometer unit: m/s^2.
- Final gyroscope unit: rad/s.
- BITS and WEDA use conversion factor 1.0 from the current dataset unit map.
- Unit statistics are saved to `unit_audit_25hz.csv`.

