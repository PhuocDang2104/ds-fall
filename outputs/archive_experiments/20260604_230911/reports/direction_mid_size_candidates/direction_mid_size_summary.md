# Direction Mid-Size Candidates Summary

Benchmark: BITS/WEDA only, 25 Hz, 2-second event-centered windows, temporal `tilt12` input `50 x 12`. Fall branch is fixed to `GradientBoosting + FallNoTiming_Full_132`; no FullTiming and no new dataset.

## Run Status

- Direction result rows: `252`.
- Failed rows: `0`.
- TensorFlow available: `False`; import error: `ModuleNotFoundError("No module named 'tensorflow.python'")`.
- TFLite/INT8 export is marked unavailable when TensorFlow import fails; PyTorch `.pt` models are saved.
- LDX4 distillation limitation: train/val A5 teacher probabilities are not available in saved artifacts, so the script falls back to hard true-label teacher for missing rows and logs this in `teacher_note`.

## Best E3/E6/E7 Average

| model_name | seed | learning_rate | params | avg_direction_macro_f1 | e3_direction_macro_f1 | e6_direction_macro_f1 | e7_direction_macro_f1 | avg_delta_vs_a5 | avg_delta_vs_d1 | passes_main_replacement | passes_edge_ablation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LDX1_D1_Wide_DSConv | 42 | 0.0010 | 5699 | 0.8855 | 0.8829 | 0.8902 | 0.8835 | -0.0047 | 0.0564 | yes | yes |
| LDX1_D1_Wide_DSConv | 43 | 0.0010 | 5699 | 0.8724 | 0.8717 | 0.8682 | 0.8772 | -0.0179 | 0.0432 | no | yes |
| LDX1_D1_Wide_DSConv | 43 | 0.0005 | 5699 | 0.8441 | 0.8439 | 0.8111 | 0.8772 | -0.0462 | 0.0149 | no | no |
| LDX1_D1_Wide_DSConv | 42 | 0.0005 | 5699 | 0.8437 | 0.8444 | 0.8492 | 0.8375 | -0.0466 | 0.0146 | no | no |
| LDX1_D1_Wide_DSConv | 44 | 0.0005 | 5699 | 0.8369 | 0.8344 | 0.7962 | 0.8801 | -0.0533 | 0.0078 | no | no |
| LDX3_A5_Direction_Small_075 | 43 | 0.0005 | 19648 | 0.7811 | 0.7829 | 0.8063 | 0.7541 | -0.1091 | -0.0480 | no | no |
| LDX1_D1_Wide_DSConv | 44 | 0.0010 | 5699 | 0.7744 | 0.7786 | 0.7754 | 0.7690 | -0.1159 | -0.0548 | no | no |
| LDX2_D1_Wide_Attention | 44 | 0.0005 | 5748 | 0.7627 | 0.7733 | 0.8063 | 0.7085 | -0.1275 | -0.0664 | no | no |
| LDX3_A5_Direction_Small_075 | 44 | 0.0005 | 19648 | 0.7440 | 0.7437 | 0.7754 | 0.7130 | -0.1462 | -0.0851 | no | no |
| LDX2_D1_Wide_Attention | 44 | 0.0010 | 5748 | 0.7335 | 0.7422 | 0.7769 | 0.6814 | -0.1568 | -0.0957 | no | no |
| LDX3_A5_Direction_Small_075 | 42 | 0.0010 | 19648 | 0.7303 | 0.7362 | 0.7868 | 0.6678 | -0.1600 | -0.0988 | no | no |
| LDX3_A5_Direction_Small_075 | 44 | 0.0010 | 19648 | 0.7027 | 0.7114 | 0.8063 | 0.5905 | -0.1875 | -0.1264 | no | no |
| LDX3_A5_Direction_Small_075 | 43 | 0.0010 | 19648 | 0.7026 | 0.7147 | 0.7927 | 0.6005 | -0.1876 | -0.1265 | no | no |
| LDX3_A5_Direction_Small_075 | 42 | 0.0005 | 19648 | 0.6845 | 0.6981 | 0.7242 | 0.6312 | -0.2057 | -0.1446 | no | no |
| LDX4_D1_Wide_Attention_Distill_T2_A050 | 44 | 0.0005 | 5748 | 0.6679 | 0.6862 | 0.7627 | 0.5547 | -0.2224 | -0.1613 | no | no |
| LDX4_D1_Wide_Attention_Distill_T2_A070 | 44 | 0.0010 | 5748 | 0.6538 | 0.6743 | 0.7542 | 0.5328 | -0.2365 | -0.1754 | no | no |
| LDX2_D1_Wide_Attention | 43 | 0.0010 | 5748 | 0.6502 | 0.6683 | 0.7627 | 0.5196 | -0.2400 | -0.1789 | no | no |
| LDX4_D1_Wide_Attention_Distill_T3_A050 | 43 | 0.0010 | 5748 | 0.6489 | 0.6849 | 0.8173 | 0.4444 | -0.2414 | -0.1802 | no | no |
| LDX4_D1_Wide_Attention_Distill_T2_A050 | 44 | 0.0010 | 5748 | 0.6143 | 0.6270 | 0.7745 | 0.4414 | -0.2759 | -0.2148 | no | no |
| LDX4_D1_Wide_Attention_Distill_T2_A070 | 43 | 0.0005 | 5748 | 0.6028 | 0.6322 | 0.7260 | 0.4502 | -0.2874 | -0.2263 | no | no |
| LDX4_D1_Wide_Attention_Distill_T3_A050 | 42 | 0.0010 | 5748 | 0.5949 | 0.6437 | 0.8173 | 0.3236 | -0.2954 | -0.2343 | no | no |
| LDX4_D1_Wide_Attention_Distill_T2_A070 | 44 | 0.0005 | 5748 | 0.5897 | 0.6124 | 0.6857 | 0.4710 | -0.3005 | -0.2394 | no | no |
| LDX4_D1_Wide_Attention_Distill_T2_A050 | 42 | 0.0010 | 5748 | 0.5847 | 0.6319 | 0.8244 | 0.2978 | -0.3055 | -0.2444 | no | no |
| LDX4_D1_Wide_Attention_Distill_T2_A070 | 42 | 0.0010 | 5748 | 0.5772 | 0.6207 | 0.8033 | 0.3077 | -0.3130 | -0.2519 | no | no |
| LDX4_D1_Wide_Attention_Distill_T2_A050 | 42 | 0.0005 | 5748 | 0.5753 | 0.6037 | 0.7190 | 0.4031 | -0.3150 | -0.2539 | no | no |
| LDX4_D1_Wide_Attention_Distill_T2_A070 | 42 | 0.0005 | 5748 | 0.5753 | 0.6037 | 0.7190 | 0.4031 | -0.3150 | -0.2539 | no | no |
| LDX4_D1_Wide_Attention_Distill_T3_A050 | 44 | 0.0005 | 5748 | 0.5720 | 0.6017 | 0.6857 | 0.4286 | -0.3182 | -0.2571 | no | no |
| LDX2_D1_Wide_Attention | 42 | 0.0005 | 5748 | 0.5713 | 0.6145 | 0.7756 | 0.3236 | -0.3190 | -0.2579 | no | no |
| LDX4_D1_Wide_Attention_Distill_T3_A050 | 44 | 0.0010 | 5748 | 0.5690 | 0.5973 | 0.7425 | 0.3674 | -0.3212 | -0.2601 | no | no |
| LDX4_D1_Wide_Attention_Distill_T2_A070 | 43 | 0.0010 | 5748 | 0.5630 | 0.5924 | 0.6410 | 0.4556 | -0.3273 | -0.2661 | no | no |

## Seed Robustness

| model_name | learning_rate | params | avg_f1_mean | avg_f1_std | e3_mean | e6_mean | e7_mean | pass_main_count | pass_edge_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LDX1_D1_Wide_DSConv | 0.0010 | 5699 | 0.8441 | 0.0607 | 0.8444 | 0.8446 | 0.8432 | 1 | 2 |
| LDX1_D1_Wide_DSConv | 0.0005 | 5699 | 0.8416 | 0.0040 | 0.8409 | 0.8188 | 0.8649 | 0 | 0 |
| LDX3_A5_Direction_Small_075 | 0.0005 | 19648 | 0.7365 | 0.0487 | 0.7416 | 0.7686 | 0.6994 | 0 | 0 |
| LDX3_A5_Direction_Small_075 | 0.0010 | 19648 | 0.7119 | 0.0159 | 0.7208 | 0.7953 | 0.6196 | 0 | 0 |
| LDX2_D1_Wide_Attention | 0.0010 | 5748 | 0.6422 | 0.0955 | 0.6634 | 0.7620 | 0.5013 | 0 | 0 |
| LDX2_D1_Wide_Attention | 0.0005 | 5748 | 0.6049 | 0.1440 | 0.6379 | 0.7451 | 0.4316 | 0 | 0 |
| LDX4_D1_Wide_Attention_Distill_T3_A050 | 0.0010 | 5748 | 0.6043 | 0.0407 | 0.6419 | 0.7923 | 0.3785 | 0 | 0 |
| LDX4_D1_Wide_Attention_Distill_T2_A070 | 0.0010 | 5748 | 0.5980 | 0.0488 | 0.6291 | 0.7328 | 0.4320 | 0 | 0 |
| LDX4_D1_Wide_Attention_Distill_T2_A070 | 0.0005 | 5748 | 0.5893 | 0.0138 | 0.6161 | 0.7102 | 0.4415 | 0 | 0 |
| LDX4_D1_Wide_Attention_Distill_T2_A050 | 0.0010 | 5748 | 0.5669 | 0.0584 | 0.5983 | 0.7615 | 0.3409 | 0 | 0 |
| LDX4_D1_Wide_Attention_Distill_T3_A050 | 0.0005 | 5748 | 0.5542 | 0.0226 | 0.5902 | 0.6936 | 0.3787 | 0 | 0 |
| LDX4_D1_Wide_Attention_Distill_T2_A050 | 0.0005 | 5748 | 0.5506 | 0.1313 | 0.5796 | 0.6908 | 0.3815 | 0 | 0 |

## Best By Protocol

| experiment_id | model_name | seed | learning_rate | params | direction_macro_f1 | direction_accuracy | forward_f1 | backward_f1 | lateral_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_BITS_TO_BITS | LDX4_D1_Wide_Attention_Distill_T2_A050 | 43 | 0.0010 | 5748 | 0.8878 | 0.8611 | 0.8485 | 1.0000 | 0.8148 |
| E2_WEDA_TO_WEDA | LDX1_D1_Wide_DSConv | 42 | 0.0010 | 5699 | 0.9568 | 0.9600 | 0.9474 | 1.0000 | 0.9231 |
| E3_BITS_WEDA_MIXED | LDX1_D1_Wide_DSConv | 42 | 0.0010 | 5699 | 0.8829 | 0.8852 | 0.9020 | 0.8750 | 0.8718 |
| E4_BITS_TO_WEDA | LDX3_A5_Direction_Small_075 | 44 | 0.0005 | 19648 | 0.4603 | 0.5600 | 0.6667 | 0.0000 | 0.7143 |
| E5_WEDA_TO_BITS | LDX1_D1_Wide_DSConv | 42 | 0.0005 | 5699 | 0.7425 | 0.7778 | 0.8718 | 0.8000 | 0.5556 |
| E6_E3_MIXED_TEST_BITS | LDX1_D1_Wide_DSConv | 42 | 0.0010 | 5699 | 0.8902 | 0.8889 | 0.9143 | 0.9231 | 0.8333 |
| E7_E3_MIXED_TEST_WEDA | LDX1_D1_Wide_DSConv | 42 | 0.0010 | 5699 | 0.8835 | 0.8800 | 0.8750 | 0.8421 | 0.9333 |

## End-to-End Direction

| pipeline | train_protocol | avg_E2E_direction_macro_f1 | e3_E2E | e6_E2E | e7_E2E | coverage |
| --- | --- | --- | --- | --- | --- | --- |
| LDX1_D1_Wide_DSConv_E3_BITS_WEDA_MIXED_seed43_lr0.001 | E3_BITS_WEDA_MIXED | 0.7942 | 0.7974 | 0.7131 | 0.8722 | 0.8593 |
| LDX1_D1_Wide_DSConv_E3_BITS_WEDA_MIXED_seed42_lr0.001 | E3_BITS_WEDA_MIXED | 0.7857 | 0.7883 | 0.7317 | 0.8370 | 0.8593 |
| LDX1_D1_Wide_DSConv_E3_BITS_WEDA_MIXED_seed44_lr0.0005 | E3_BITS_WEDA_MIXED | 0.7531 | 0.7560 | 0.6667 | 0.8366 | 0.8593 |
| LDX1_D1_Wide_DSConv_E3_BITS_WEDA_MIXED_seed43_lr0.0005 | E3_BITS_WEDA_MIXED | 0.7531 | 0.7638 | 0.6593 | 0.8361 | 0.8593 |
| LDX1_D1_Wide_DSConv_E3_BITS_WEDA_MIXED_seed42_lr0.0005 | E3_BITS_WEDA_MIXED | 0.7340 | 0.7419 | 0.6690 | 0.7912 | 0.8593 |
| LDX3_A5_Direction_Small_075_E3_BITS_WEDA_MIXED_seed43_lr0.0005 | E3_BITS_WEDA_MIXED | 0.6964 | 0.6965 | 0.6517 | 0.7410 | 0.8593 |
| LDX1_D1_Wide_DSConv_E3_BITS_WEDA_MIXED_seed44_lr0.001 | E3_BITS_WEDA_MIXED | 0.6945 | 0.7034 | 0.6517 | 0.7285 | 0.8593 |
| LDX2_D1_Wide_Attention_E3_BITS_WEDA_MIXED_seed44_lr0.0005 | E3_BITS_WEDA_MIXED | 0.6792 | 0.6956 | 0.6828 | 0.6593 | 0.8593 |
| LDX2_D1_Wide_Attention_E3_BITS_WEDA_MIXED_seed44_lr0.001 | E3_BITS_WEDA_MIXED | 0.6637 | 0.6747 | 0.6897 | 0.6266 | 0.8593 |
| LDX3_A5_Direction_Small_075_E3_BITS_WEDA_MIXED_seed44_lr0.0005 | E3_BITS_WEDA_MIXED | 0.6564 | 0.6569 | 0.6517 | 0.6606 | 0.8593 |
| LDX3_A5_Direction_Small_075_E3_BITS_WEDA_MIXED_seed42_lr0.001 | E3_BITS_WEDA_MIXED | 0.6447 | 0.6504 | 0.6263 | 0.6575 | 0.8593 |
| LDX3_A5_Direction_Small_075_E3_BITS_WEDA_MIXED_seed43_lr0.001 | E3_BITS_WEDA_MIXED | 0.6253 | 0.6340 | 0.6597 | 0.5822 | 0.8593 |
| LDX3_A5_Direction_Small_075_E3_BITS_WEDA_MIXED_seed42_lr0.0005 | E3_BITS_WEDA_MIXED | 0.6187 | 0.6332 | 0.5985 | 0.6243 | 0.8593 |
| LDX3_A5_Direction_Small_075_E3_BITS_WEDA_MIXED_seed44_lr0.001 | E3_BITS_WEDA_MIXED | 0.5959 | 0.6017 | 0.6517 | 0.5343 | 0.8593 |
| LDX4_D1_Wide_Attention_Distill_T2_A050_E3_BITS_WEDA_MIXED_seed44_lr0.0005 | E3_BITS_WEDA_MIXED | 0.5800 | 0.6038 | 0.6405 | 0.4958 | 0.8593 |
| LDX4_D1_Wide_Attention_Distill_T2_A070_E3_BITS_WEDA_MIXED_seed44_lr0.001 | E3_BITS_WEDA_MIXED | 0.5665 | 0.5937 | 0.6263 | 0.4795 | 0.8593 |
| LDX4_D1_Wide_Attention_Distill_T3_A050_E3_BITS_WEDA_MIXED_seed43_lr0.001 | E3_BITS_WEDA_MIXED | 0.5620 | 0.5972 | 0.6590 | 0.4297 | 0.8593 |
| LDX2_D1_Wide_Attention_E3_BITS_WEDA_MIXED_seed43_lr0.001 | E3_BITS_WEDA_MIXED | 0.5594 | 0.5733 | 0.6036 | 0.5013 | 0.8593 |
| LDX4_D1_Wide_Attention_Distill_T2_A050_E3_BITS_WEDA_MIXED_seed44_lr0.001 | E3_BITS_WEDA_MIXED | 0.5405 | 0.5555 | 0.6482 | 0.4180 | 0.8593 |
| LDX4_D1_Wide_Attention_Distill_T2_A070_E3_BITS_WEDA_MIXED_seed43_lr0.0005 | E3_BITS_WEDA_MIXED | 0.5389 | 0.5671 | 0.6116 | 0.4381 | 0.8593 |

## Complexity

| model_name | learning_rate | params | estimated_fp32_kb | estimated_int8_kb | tflite_convert | int8_tflite_convert | edge_suitability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LDX1_D1_Wide_DSConv | 0.0010 | 5699 | 22.2617 | 5.5654 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high |
| LDX1_D1_Wide_DSConv | 0.0005 | 5699 | 22.2617 | 5.5654 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high |
| LDX2_D1_Wide_Attention | 0.0010 | 5748 | 22.4531 | 5.6133 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high |
| LDX2_D1_Wide_Attention | 0.0005 | 5748 | 22.4531 | 5.6133 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high |
| LDX3_A5_Direction_Small_075 | 0.0010 | 19648 | 76.7500 | 19.1875 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | medium-high |
| LDX3_A5_Direction_Small_075 | 0.0005 | 19648 | 76.7500 | 19.1875 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | medium-high |
| LDX4_D1_Wide_Attention_Distill_T2_A050 | 0.0010 | 5748 | 22.4531 | 5.6133 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high |
| LDX4_D1_Wide_Attention_Distill_T2_A050 | 0.0005 | 5748 | 22.4531 | 5.6133 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high |
| LDX4_D1_Wide_Attention_Distill_T3_A050 | 0.0010 | 5748 | 22.4531 | 5.6133 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high |
| LDX4_D1_Wide_Attention_Distill_T3_A050 | 0.0005 | 5748 | 22.4531 | 5.6133 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high |
| LDX4_D1_Wide_Attention_Distill_T2_A070 | 0.0010 | 5748 | 22.4531 | 5.6133 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high |
| LDX4_D1_Wide_Attention_Distill_T2_A070 | 0.0005 | 5748 | 22.4531 | 5.6133 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high |

## Answers

1. Increasing params above D1 improved over the previous D1 only if `avg_delta_vs_d1` is positive. Best run is `LDX1_D1_Wide_DSConv` with avg `0.8855` and delta vs D1 `0.0564`.
   Across seeds, best mean is `LDX1_D1_Wide_DSConv` lr `0.001` with mean avg F1 `0.8441` and mean delta vs D1 `0.0150`.
2. Best params/F1 trade-off: `LDX1_D1_Wide_DSConv` at `5699` params, avg F1 `0.8855`.
3. Closest to A5 under 30k params: `LDX1_D1_Wide_DSConv` with avg F1 `0.8855` and delta vs A5 `-0.0047`.
4. Best under 12k params: `LDX1_D1_Wide_DSConv` with avg F1 `0.8855`.
5. Attention pooling effect: LDX2 attention best avg 0.7627 vs LDX1 GAP best avg 0.8855, delta -0.1228.
6. Distillation effect: best LDX4 distill avg 0.6679 vs LDX2 non-distill avg 0.7627, delta -0.0948; teacher fallback must be considered.
7. Best E6 BITS candidate: `LDX1_D1_Wide_DSConv` with E6 F1 `0.8902`.
8. Best E3 mixed candidate: `LDX1_D1_Wide_DSConv` with E3 F1 `0.8829`.
9. Best E7 WEDA candidate: `LDX1_D1_Wide_DSConv` with E7 F1 `0.8835`.
10. Single-seed main candidate exists (`LDX1_D1_Wide_DSConv`), but no candidate passes the main replacement criteria robustly across all seeds. Keep A5 as the main direction expert unless LDX1 stability is improved or a fixed seed protocol is explicitly justified.
11. Single-seed edge candidate: `LDX1_D1_Wide_DSConv`. Report it as the best mid-size edge ablation with seed-sensitivity caveat.

## Files

- `outputs\reports\direction_mid_size_candidates\direction_mid_size_results.csv`
- `outputs\reports\direction_mid_size_candidates\direction_mid_size_best_by_protocol.csv`
- `outputs\reports\direction_mid_size_candidates\direction_mid_size_vs_a5_and_d1.csv`
- `outputs\reports\direction_mid_size_candidates\direction_mid_size_e2e_results.csv`
- `outputs\reports\direction_mid_size_candidates\direction_mid_size_complexity.csv`
- `outputs\reports\direction_mid_size_candidates\failed_runs.csv`
- `outputs\reports\direction_mid_size_candidates\model_configs.json`
