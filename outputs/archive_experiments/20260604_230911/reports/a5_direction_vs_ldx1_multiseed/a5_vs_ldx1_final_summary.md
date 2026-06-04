# A5 Direction Only vs LDX1 Multi-Seed Summary

Benchmark: BITS/WEDA only, 25 Hz, 2-second event-centered windows, temporal `tilt12` input `50 x 12`. Both models are trained direction-only on supervised fall windows. Historical A5 saved predictions are not used as main results.

## Run Status

- Direction rows: `140`.
- Focus rows E3/E6/E7: `20`.
- Failed rows: `0`.
- TensorFlow available: `False`; import error: `ModuleNotFoundError("No module named 'tensorflow.python'")`.
- Framework note: TensorFlow import fails in this environment, so both models are implemented and trained in PyTorch for a fair same-framework comparison.
- A5_Direction_Only note: fall branch is removed; this tests direction-only representation, not the historical multitask A5 artifact.

## Seed Summary

| model_name | learning_rate | params | mean_avg_e3_e6_e7_f1 | std_avg_e3_e6_e7_f1 | mean_e3_f1 | mean_e6_f1 | mean_e7_f1 | mean_delta_vs_historical_A5 | params_reduction_vs_A5_Direction_Only | estimated_int8_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LDX1_D1_Wide_DSConv | 0.0010 | 5699 | 0.8285 | 0.0482 | 0.8303 | 0.8381 | 0.8171 | -0.0618 | 0.9005 | 5.5654 |
| LDX1_D1_Wide_DSConv | 0.0005 | 5699 | 0.8263 | 0.0255 | 0.8275 | 0.8096 | 0.8417 | -0.0640 | 0.9005 | 5.5654 |
| A5_Direction_Only | 0.0005 | 57268 | 0.7716 | 0.0378 | 0.7815 | 0.8159 | 0.7174 | -0.1186 | 0.0000 | 55.9258 |
| A5_Direction_Only | 0.0010 | 57268 | 0.7499 | 0.0932 | 0.7638 | 0.8318 | 0.6541 | -0.1403 | 0.0000 | 55.9258 |

## Per-Seed E3/E6/E7 Summary

| model_name | seed | learning_rate | params | avg_e3_e6_e7_f1 | std_e3_e6_e7_f1 | e3_f1 | e6_f1 | e7_f1 | delta_vs_historical_A5_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A5_Direction_Only | 42 | 0.0010 | 57268 | 0.9006 | 0.0132 | 0.8966 | 0.8898 | 0.9153 | 0.0104 |
| LDX1_D1_Wide_DSConv | 42 | 0.0010 | 5699 | 0.8855 | 0.0041 | 0.8829 | 0.8902 | 0.8835 | -0.0047 |
| LDX1_D1_Wide_DSConv | 43 | 0.0010 | 5699 | 0.8724 | 0.0046 | 0.8717 | 0.8682 | 0.8772 | -0.0179 |
| LDX1_D1_Wide_DSConv | 43 | 0.0005 | 5699 | 0.8441 | 0.0331 | 0.8439 | 0.8111 | 0.8772 | -0.0461 |
| LDX1_D1_Wide_DSConv | 42 | 0.0005 | 5699 | 0.8437 | 0.0059 | 0.8444 | 0.8492 | 0.8375 | -0.0465 |
| LDX1_D1_Wide_DSConv | 44 | 0.0005 | 5699 | 0.8369 | 0.0420 | 0.8344 | 0.7962 | 0.8801 | -0.0533 |
| A5_Direction_Only | 43 | 0.0005 | 57268 | 0.8268 | 0.0266 | 0.8338 | 0.8492 | 0.7974 | -0.0634 |
| LDX1_D1_Wide_DSConv | 46 | 0.0005 | 5699 | 0.8234 | 0.0541 | 0.8310 | 0.8733 | 0.7659 | -0.0669 |
| LDX1_D1_Wide_DSConv | 46 | 0.0010 | 5699 | 0.8109 | 0.0429 | 0.8155 | 0.8513 | 0.7659 | -0.0793 |
| LDX1_D1_Wide_DSConv | 45 | 0.0010 | 5699 | 0.7992 | 0.0084 | 0.8029 | 0.8051 | 0.7897 | -0.0910 |
| A5_Direction_Only | 46 | 0.0005 | 57268 | 0.7916 | 0.0303 | 0.7888 | 0.7627 | 0.8231 | -0.0987 |
| LDX1_D1_Wide_DSConv | 45 | 0.0005 | 5699 | 0.7832 | 0.0648 | 0.7840 | 0.7181 | 0.8477 | -0.1070 |
| LDX1_D1_Wide_DSConv | 44 | 0.0010 | 5699 | 0.7744 | 0.0049 | 0.7786 | 0.7754 | 0.7690 | -0.1159 |
| A5_Direction_Only | 42 | 0.0005 | 57268 | 0.7571 | 0.1706 | 0.7845 | 0.9124 | 0.5745 | -0.1331 |
| A5_Direction_Only | 46 | 0.0010 | 57268 | 0.7568 | 0.0367 | 0.7715 | 0.7839 | 0.7151 | -0.1334 |
| A5_Direction_Only | 45 | 0.0005 | 57268 | 0.7515 | 0.0189 | 0.7553 | 0.7310 | 0.7683 | -0.1387 |
| A5_Direction_Only | 45 | 0.0010 | 57268 | 0.7345 | 0.0548 | 0.7419 | 0.7852 | 0.6763 | -0.1558 |
| A5_Direction_Only | 44 | 0.0005 | 57268 | 0.7311 | 0.1009 | 0.7453 | 0.8241 | 0.6238 | -0.1592 |
| A5_Direction_Only | 43 | 0.0010 | 57268 | 0.7072 | 0.1728 | 0.7344 | 0.8649 | 0.5224 | -0.1830 |
| A5_Direction_Only | 44 | 0.0010 | 57268 | 0.6504 | 0.1982 | 0.6746 | 0.8354 | 0.4412 | -0.2398 |

## Best By Protocol

| experiment_id | model_name | seed | learning_rate | params | direction_macro_f1 | direction_accuracy | forward_f1 | backward_f1 | lateral_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_BITS_TO_BITS | A5_Direction_Only | 43 | 0.0005 | 57268 | 0.9314 | 0.9167 | 0.9143 | 1.0000 | 0.8800 |
| E1_BITS_TO_BITS | LDX1_D1_Wide_DSConv | 45 | 0.0010 | 5699 | 0.9267 | 0.9167 | 0.9231 | 1.0000 | 0.8571 |
| E2_WEDA_TO_WEDA | A5_Direction_Only | 45 | 0.0005 | 57268 | 0.8743 | 0.8800 | 0.8421 | 0.9474 | 0.8333 |
| E2_WEDA_TO_WEDA | LDX1_D1_Wide_DSConv | 42 | 0.0010 | 5699 | 0.9568 | 0.9600 | 0.9474 | 1.0000 | 0.9231 |
| E3_BITS_WEDA_MIXED | A5_Direction_Only | 42 | 0.0010 | 57268 | 0.8966 | 0.9016 | 0.9434 | 0.8966 | 0.8500 |
| E3_BITS_WEDA_MIXED | LDX1_D1_Wide_DSConv | 42 | 0.0010 | 5699 | 0.8829 | 0.8852 | 0.9020 | 0.8750 | 0.8718 |
| E4_BITS_TO_WEDA | A5_Direction_Only | 45 | 0.0010 | 57268 | 0.5010 | 0.5600 | 0.6667 | 0.2000 | 0.6364 |
| E4_BITS_TO_WEDA | LDX1_D1_Wide_DSConv | 46 | 0.0010 | 5699 | 0.5651 | 0.6000 | 0.7000 | 0.3636 | 0.6316 |
| E5_WEDA_TO_BITS | A5_Direction_Only | 44 | 0.0010 | 57268 | 0.7268 | 0.7222 | 0.7778 | 0.8571 | 0.5455 |
| E5_WEDA_TO_BITS | LDX1_D1_Wide_DSConv | 45 | 0.0005 | 5699 | 0.7808 | 0.8333 | 0.9444 | 0.6154 | 0.7826 |
| E6_E3_MIXED_TEST_BITS | A5_Direction_Only | 42 | 0.0005 | 57268 | 0.9124 | 0.9167 | 0.9444 | 0.9231 | 0.8696 |
| E6_E3_MIXED_TEST_BITS | LDX1_D1_Wide_DSConv | 42 | 0.0010 | 5699 | 0.8902 | 0.8889 | 0.9143 | 0.9231 | 0.8333 |
| E7_E3_MIXED_TEST_WEDA | A5_Direction_Only | 42 | 0.0010 | 57268 | 0.9153 | 0.9200 | 1.0000 | 0.8889 | 0.8571 |
| E7_E3_MIXED_TEST_WEDA | LDX1_D1_Wide_DSConv | 42 | 0.0010 | 5699 | 0.8835 | 0.8800 | 0.8750 | 0.8421 | 0.9333 |

## E2E Direction With Fixed GB Full132 Fall

| model_name | learning_rate | mean_avg_E2E_direction_macro_f1 | std_avg_E2E_direction_macro_f1 | mean_e3_E2E | mean_e6_E2E | mean_e7_E2E | mean_coverage | mean_correct_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LDX1_D1_Wide_DSConv | 0.0010 | 0.7451 | 0.0431 | 0.7505 | 0.6963 | 0.7885 | 0.8593 | 0.7135 |
| LDX1_D1_Wide_DSConv | 0.0005 | 0.7443 | 0.0087 | 0.7510 | 0.6699 | 0.8121 | 0.8593 | 0.7151 |
| A5_Direction_Only | 0.0005 | 0.6888 | 0.0342 | 0.7023 | 0.6694 | 0.6946 | 0.8593 | 0.6670 |
| A5_Direction_Only | 0.0010 | 0.6652 | 0.0977 | 0.6810 | 0.7027 | 0.6120 | 0.8593 | 0.6517 |

## Complexity

| model_name | learning_rate | params | estimated_fp32_kb | estimated_int8_kb | params_reduction_vs_A5_Direction_Only | params_reduction_vs_historical_A5 | edge_suitability | tflite_convert | int8_tflite_convert |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A5_Direction_Only | 0.0010 | 57268 | 223.7031 | 55.9258 | 0.0000 | 0.1318 | medium | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed |
| A5_Direction_Only | 0.0005 | 57268 | 223.7031 | 55.9258 | 0.0000 | 0.1318 | medium | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed |
| LDX1_D1_Wide_DSConv | 0.0010 | 5699 | 22.2617 | 5.5654 | 0.9005 | 0.9136 | high | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed |
| LDX1_D1_Wide_DSConv | 0.0005 | 5699 | 22.2617 | 5.5654 | 0.9005 | 0.9136 | high | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed |

## Answers

1. A5_Direction_Only stability vs LDX1: comparing best-mean configs, A5 std `0.0378` and LDX1 std `0.0482`. The lowest-std config overall is `LDX1_D1_Wide_DSConv` lr `0.0005` with std `0.0255`.
2. Higher mean E3/E6/E7 Direction Macro F1: `LDX1`. A5 mean `0.7716`, LDX1 mean `0.8285`, LDX1-A5 delta `0.0569`.
3. Lower std across seeds: `LDX1_D1_Wide_DSConv`. LDX1 lowest std `0.0255` vs A5 lowest std `0.0378`.
4. Better E6 BITS mean: `LDX1`. A5 `0.8159`, LDX1 `0.8381`.
5. Better E7 WEDA mean: `LDX1`. A5 `0.7174`, LDX1 `0.8171`.
6. Does LDX1 remain close to A5_Direction_Only after fair multi-seed comparison? no; LDX1 is better than A5_Direction_Only by 0.0569, so it is not merely close.
7. LDX1 parameter reduction vs A5_Direction_Only: `90.05%`; LDX1 params `5699`, A5 params `57268`.
8. Better E2E direction with fixed GB Full132 fall: `LDX1`. A5 E2E `0.6888`, LDX1 E2E `0.7451`, delta `0.0563`.
9. Should A5_Direction_Only replace historical A5 reference? `no`. Historical multitask A5 avg is about `0.8902`; A5 direction-only mean is `0.7716`.
10. LDX1 can replace A5_Direction_Only for compact direction-only deployment: it has higher mean F1, better E2E, and >85% parameter reduction. It should not be claimed to replace the historical multitask A5 reference unless it also beats that reference.
11. Final recommendation: main paper direction expert = `historical multitask A5 direction reference`.
12. Compact edge direction expert = `LDX1_D1_Wide_DSConv` if its measured mean/std trade-off is acceptable for deployment.
13. Future work: stabilize LDX1 across seeds or restore TensorFlow/TFLite export to validate real INT8 deployment.

## Output Files

- `outputs\reports\a5_direction_vs_ldx1_multiseed\a5_vs_ldx1_direction_results.csv`
- `outputs\reports\a5_direction_vs_ldx1_multiseed\a5_vs_ldx1_seed_summary.csv`
- `outputs\reports\a5_direction_vs_ldx1_multiseed\a5_vs_ldx1_best_by_protocol.csv`
- `outputs\reports\a5_direction_vs_ldx1_multiseed\a5_vs_ldx1_e3_e6_e7_summary.csv`
- `outputs\reports\a5_direction_vs_ldx1_multiseed\a5_vs_ldx1_e2e_results.csv`
- `outputs\reports\a5_direction_vs_ldx1_multiseed\a5_vs_ldx1_complexity.csv`
- `outputs\reports\a5_direction_vs_ldx1_multiseed\class_counts.csv`
- `outputs\reports\a5_direction_vs_ldx1_multiseed\failed_runs.csv`
- `outputs\reports\a5_direction_vs_ldx1_multiseed\model_configs.json`
- `outputs\reports\a5_direction_vs_ldx1_multiseed\environment_info.txt`
