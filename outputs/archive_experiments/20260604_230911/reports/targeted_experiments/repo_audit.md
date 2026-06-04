# Repository/Data Audit

- Reused `scripts/run_25hz_a5wcefw_bits_weda.py` for BITS/WEDA event-centered preprocessing.
- Direction masks are preserved through `make_sample_weights`; direction loss is masked for non-fall/unlabeled samples.
- Scaler is fit on E3 train split only for every run.
- Test split is never used for scaler, early stopping, or threshold tuning.

## Dataset Audit

| dataset | source_folder | original_sampling_rate | target_sampling_rate | resampling_method | n_subjects | n_trials | raw_samples | effective_fs_before_mean | effective_fs_after_mean | n_windows | fall_windows | non_fall_windows | direction_supervised_windows | direction_class_distribution | feature_shape | nan_count | inf_count | acc_mag_median | acc_mag_p95 | acc_mag_p99 | gyro_mag_median | gyro_mag_p95 | gyro_mag_p99 | unit_map |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bits | C:\Users\ADMIN\Desktop\ds-fall\data\raw\Dataset | 20.0000 | 25.0000 | ['uniform_20hz_time_interpolation_to_25hz'] | 41.0000 | 975.0000 | 625393.0000 | 20.0000 | 25.0000 | 3560.0000 | 325.0000 | 3235.0000 | 244.0000 | {"forward": 121, "lateral": 82, "backward": 41} | (3560, 50, 12) | 0.0000 | 0.0000 | 9.8372 | 17.3699 | 33.3119 | 0.4645 | 4.2172 | 7.0662 | {"acc_unit_before": "m/s^2", "gyro_unit_before": "rad/s", "acc_unit_after": "m/s^2", "gyro_unit_after": "rad/s", "conversion_factor_acc": 1.0, "conversion_factor_gyro": 1.0} |
| weda | C:\Users\ADMIN\Desktop\ds-fall\data\raw\WEDA-FALL-main\dataset\25Hz | 25.0000 | 25.0000 | ['raw_25hz_time_aligned_to_uniform_25hz'] | 25.0000 | 584.0000 | 304155.0000 | 24.7887 | 25.0000 | 1895.0000 | 350.0000 | 1545.0000 | 350.0000 | {"forward": 126, "backward": 126, "lateral": 98} | (1895, 50, 12) | 0.0000 | 0.0000 | 9.8879 | 19.6797 | 35.4441 | 1.1474 | 5.7253 | 9.2309 | {"acc_unit_before": "m/s^2", "gyro_unit_before": "rad/s", "acc_unit_after": "m/s^2", "gyro_unit_after": "rad/s", "conversion_factor_acc": 1.0, "conversion_factor_gyro": 1.0} |
