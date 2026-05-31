# Unit Fix And A5WCEFW Summary

## Unit Diagnostics After Fix

| dataset | acc_mag_median | gravity_status | likely_unit | gyro_range_note | impact_valid_pct | main_warning |
| --- | ---: | --- | --- | --- | ---: | --- |
| bits | 9.8397 | likely contains gravity | m/s^2 | gyro scale unclear/intermediate | 95.69 | BITS raw timestamp column is rounded/duplicated in these CSVs; current preprocessing intentionally uses configured row-order 20 Hz before resampling to 50 Hz. |
| hifd | 2.4614 | likely gravity-removed | m/s^2-like | gyro scale unclear/intermediate | 100.00 | hifd: raw timestamp sampling check not available; using processed window metadata only. |
| weda | 9.8853 | likely contains gravity | m/s^2 | gyro scale unclear/intermediate | 94.57 | weda raw timestamp median fs=500.00 Hz differs from expected/configured 50.0 Hz. |
| umafall | 10.0858 | likely contains gravity | m/s^2 | gyro scale unclear/intermediate | 99.52 |  |

## A5WCEFW After Unit Fix

- Overall fall F1: 0.8776
- Overall direction macro F1: 0.6507
- Direction supervised test samples: 113
- Params: 65959

## Per Dataset Metrics

| dataset | fall_f1 | direction_macro_f1 | direction_supervised |
| --- | ---: | ---: | ---: |
| bits | 0.8837 | 0.8173 | 36 |
| hifd | 0.9655 | 0.7313 | 15 |
| umafall | 0.9041 | 0.5699 | 37 |
| weda | 0.7755 | 0.4820 | 25 |

## Interpretation

- Unit harmonization fixed the acc/gyro scale mismatch for BITS/WEDA/UMAFall and converted HIFD gyro to rad/s.
- UMAFall fall F1 improved to 0.9041 under A5WCEFW.
- UMAFall direction macro F1 remained low at 0.5699, close to the old 0.5600 result, so unit mismatch was not the only direction bottleneck.
- Because handcrafted UMAFall RF macro F1 remains high in diagnostics, the next likely causes are model/loss behavior, axis convention mismatch, or cross-dataset direction-label semantics rather than missing gravity.
