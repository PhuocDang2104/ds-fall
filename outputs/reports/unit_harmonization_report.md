# Unit Harmonization Report

Raw windows are converted before train-set normalization and before DS-Fall-RD feature engineering.

- Target accelerometer unit: `m/s^2`
- Target gyroscope unit: `rad/s`

## Dataset Unit Map

| dataset | acc_unit_before | gyro_unit_before | acc_unit_after | gyro_unit_after | conversion_factor_acc | conversion_factor_gyro |
| --- | --- | --- | --- | --- | --- | --- |
| bits | m/s^2 | rad/s | m/s^2 | rad/s | 1 | 1 |
| hifd | m/s^2 (loader converted from g) | deg/s | m/s^2 | rad/s | 1 | 0.0174533 |
| umafall | g | deg/s | m/s^2 | rad/s | 9.80665 | 0.0174533 |
| weda | m/s^2 | rad/s | m/s^2 | rad/s | 1 | 1 |

## Metadata Counts

| dataset | acc_unit_before | gyro_unit_before | acc_unit_after | gyro_unit_after | conversion_factor_acc | conversion_factor_gyro |
| --- | --- | --- | --- | --- | --- | --- |
| bits | m/s^2 | rad/s | m/s^2 | rad/s | 1 | 1 |
| hifd | m/s^2 (loader converted from g) | deg/s | m/s^2 | rad/s | 1 | 0.0174533 |
| umafall | g | deg/s | m/s^2 | rad/s | 9.80665 | 0.0174533 |
| weda | m/s^2 | rad/s | m/s^2 | rad/s | 1 | 1 |

## Warnings

- none
