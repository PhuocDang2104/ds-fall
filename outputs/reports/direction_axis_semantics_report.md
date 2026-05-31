# Direction Axis Semantics Report

This audit uses only supervised fall windows with direction labels. It does not modify data and does not train DS-Fall.

- Window shape: (100, 6)
- Channel order source: scaler.pkl channel_names = [ax, ay, az, gx, gy, gz]
- Input note: Loaded X from data\processed\X.npy. X.npy was converted back to harmonized raw6 values using scaler.pkl.

## Direction Counts

| dataset | direction | n |
| --- | --- | --- |
| bits | backward | 41 |
| bits | forward | 121 |
| bits | lateral | 82 |
| hifd | backward | 34 |
| hifd | forward | 33 |
| hifd | lateral | 37 |
| umafall | backward | 73 |
| umafall | forward | 71 |
| umafall | lateral | 64 |
| weda | backward | 126 |
| weda | forward | 126 |
| weda | lateral | 98 |

## Dominant Axis Summary

| dataset | direction | n | dominant_acc_axis | dominant_gyro_axis | dominant_tilt_axis | integrated_gx_mean | integrated_gy_mean | integrated_gz_mean | delta_roll_window_mean | delta_pitch_window_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bits | backward | 41 | +ax | -gy | +roll_window | 0.1829 | -1.1365 | -0.5613 | 0.3411 | -0.2208 |
| bits | forward | 121 | -ax | +gy | +roll_window | 0.2659 | 0.6259 | -0.0111 | 0.6522 | 0.1965 |
| bits | lateral | 82 | -ax | -gz | +roll_window | -0.5182 | 0.0683 | -1.5520 | 1.0771 | -0.3214 |
| hifd | backward | 34 | -ax | +gx | -roll_window | 0.8268 | -0.7756 | 0.3727 | -0.9815 | -0.1263 |
| hifd | forward | 33 | -ay | -gy | +roll_window | 0.6889 | -0.8654 | 0.0971 | 1.6171 | -0.1304 |
| hifd | lateral | 37 | -ay | +gz | +roll_window | -0.3960 | -0.0461 | 0.8320 | 1.9694 | -0.2622 |
| umafall | backward | 73 | +az | +gz | -roll_window | -0.1213 | -0.1444 | 0.3590 | -0.6316 | -0.3852 |
| umafall | forward | 71 | +az | +gz | +roll_window | -0.2514 | -0.2342 | 0.3029 | 0.6632 | -0.4090 |
| umafall | lateral | 64 | +az | +gz | -pitch_window | 0.2756 | -0.5809 | 0.6650 | 0.1053 | -0.4704 |
| weda | backward | 126 | +az | -gy | +roll_window | -0.0174 | -1.1074 | -0.1855 | 2.1815 | -0.8961 |
| weda | forward | 126 | -ax | -gy | -pitch_window | 0.0483 | -0.1662 | -0.0304 | 0.3368 | -0.7080 |
| weda | lateral | 98 | +az | -gy | +roll_window | -0.4359 | -0.5273 | -0.2820 | 0.9771 | -0.5170 |

## UMAFall Axis Permutation Baseline

| transform | classifier | n_supervised | n_train | n_test | accuracy | macro_f1 | warning |
| --- | --- | --- | --- | --- | --- | --- | --- |
| swap_xz | logistic_regression | 208 | 145 | 63 | 0.6349 | 0.6314 |  |
| swap_xz_flip_x | logistic_regression | 208 | 145 | 63 | 0.6349 | 0.6314 |  |
| swap_yz | logistic_regression | 208 | 145 | 63 | 0.5714 | 0.5707 |  |
| swap_yz_flip_y | logistic_regression | 208 | 145 | 63 | 0.5714 | 0.5707 |  |
| swap_xy | logistic_regression | 208 | 145 | 63 | 0.5556 | 0.5586 |  |
| swap_xy_flip_x | logistic_regression | 208 | 145 | 63 | 0.5556 | 0.5586 |  |
| swap_xy_flip_y | logistic_regression | 208 | 145 | 63 | 0.5556 | 0.5586 |  |
| flip_z | logistic_regression | 208 | 145 | 63 | 0.5556 | 0.5535 |  |
| identity | logistic_regression | 208 | 145 | 63 | 0.5397 | 0.5391 |  |
| flip_x | logistic_regression | 208 | 145 | 63 | 0.5397 | 0.5391 |  |
| flip_y | logistic_regression | 208 | 145 | 63 | 0.5397 | 0.5391 |  |
| flip_x | random_forest | 208 | 145 | 63 | 0.8095 | 0.8077 |  |
| swap_yz_flip_y | random_forest | 208 | 145 | 63 | 0.7937 | 0.7912 |  |
| identity | random_forest | 208 | 145 | 63 | 0.7778 | 0.7744 |  |
| swap_xy | random_forest | 208 | 145 | 63 | 0.7619 | 0.7631 |  |
| flip_z | random_forest | 208 | 145 | 63 | 0.7619 | 0.7622 |  |
| swap_yz | random_forest | 208 | 145 | 63 | 0.7619 | 0.7601 |  |
| swap_xy_flip_y | random_forest | 208 | 145 | 63 | 0.7302 | 0.7319 |  |
| swap_xy_flip_x | random_forest | 208 | 145 | 63 | 0.7302 | 0.7300 |  |
| flip_y | random_forest | 208 | 145 | 63 | 0.7302 | 0.7289 |  |
| swap_xz | random_forest | 208 | 145 | 63 | 0.6984 | 0.6997 |  |
| swap_xz_flip_x | random_forest | 208 | 145 | 63 | 0.6667 | 0.6694 |  |

## Diagnosis

- UMAFall does not consistently use the same dominant gyro/tilt axes as BITS/WEDA. forward: gyro gz vs ref gy, tilt roll_window vs ref roll_window; backward: gyro gz vs ref gy, tilt roll_window vs ref roll_window; lateral: gyro gz vs ref gz, tilt pitch_window vs ref roll_window
- Forward/backward sign consistency by dataset: bits: forward/backward gyro +gy/-gy, tilt +roll_window/+roll_window; hifd: forward/backward gyro -gy/+gx, tilt +roll_window/-roll_window; umafall: forward/backward gyro +gz/+gz, tilt +roll_window/-roll_window; weda: forward/backward gyro -gy/-gy, tilt -pitch_window/+roll_window
- Lateral merge diagnostic: bits lateral may contain opposite signs (integrated_gx pos_frac=0.48, integrated_gy pos_frac=0.55, delta_roll_window pos_frac=0.55); hifd lateral may contain opposite signs (integrated_gx pos_frac=0.41, integrated_gy pos_frac=0.49, integrated_gz pos_frac=0.59); umafall lateral may contain opposite signs (integrated_gx pos_frac=0.62, integrated_gy pos_frac=0.38, delta_roll_window pos_frac=0.44); weda lateral may contain opposite signs (integrated_gx pos_frac=0.36, integrated_gz pos_frac=0.46, delta_roll_window pos_frac=0.57)
- No UMAFall axis permutation improves RF macro F1 substantially over identity. Identity RF=0.7744, best=flip_x RF=0.8077, delta=0.0333.
- Most likely current direction bottleneck: label semantics, especially lateral left/right merge.

## Output Files

- `outputs/reports/direction_axis_summary.csv`
- `outputs/reports/umafall_axis_permutation_baseline.csv`
- `outputs/figures/direction_axis_semantics/*.png`
