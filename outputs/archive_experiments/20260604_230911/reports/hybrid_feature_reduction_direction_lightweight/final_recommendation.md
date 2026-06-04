# Hybrid Feature Reduction and Lightweight Direction Report

This report was generated from the completed `prompt_next.md` run. It uses BITS/WEDA only, 25 Hz, 2-second event-centered windows, temporal `tilt12` input shape `50 x 12`, and no FullTiming main result.

## Run Status

- Script: `scripts/run_hybrid_feature_reduction_and_lightweight_direction.py`.
- Full command used: `python scripts/run_hybrid_feature_reduction_and_lightweight_direction.py --repo-root . --run-all --full --epochs 100 --seeds 42`.
- Fall feature-reduction rows: `560`.
- Lightweight direction rows: `84`.
- Figure files: `647`.
- Model/cache files: `462`.
- Failed runs: `0`.
- Missing features: `0`.
- TensorFlow/TFLite note: local TensorFlow import failed with `ModuleNotFoundError("No module named 'tensorflow.python'")`; lightweight neural direction models were therefore trained/evaluated in PyTorch and TFLite/INT8 conversion is reported as unavailable, not silently assumed.

## Fall Feature Reduction

### Best E3/E6/E7 Fall Candidates

| model_name | feature_set | feature_count | avg_fall_f1 | E3_fall_f1 | E6_fall_f1 | E7_fall_f1 | E7_precision | E7_recall | E7_FP | E7_FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoosting | FallNoTiming_Full_132 | 132 | 0.8687 | 0.8732 | 0.8966 | 0.8364 | 0.7667 | 0.9200 | 7 | 2 |
| TinyGB30_d3 | FallNoTiming_Full_132 | 132 | 0.8297 | 0.8296 | 0.8293 | 0.8302 | 0.7857 | 0.8800 | 6 | 3 |
| RandomForest | FallNoTiming_Core_16 | 16 | 0.8194 | 0.8182 | 0.8125 | 0.8276 | 0.7273 | 0.9600 | 9 | 1 |
| HistGradientBoosting | FallNoTiming_Core_16 | 16 | 0.8104 | 0.8079 | 0.7957 | 0.8276 | 0.7273 | 0.9600 | 9 | 1 |
| TinyGB20_d3 | FallNoTiming_Full_132 | 132 | 0.8225 | 0.8235 | 0.8293 | 0.8148 | 0.7586 | 0.8800 | 7 | 3 |
| HistGradientBoosting | FallNoTiming_Full_132 | 132 | 0.8840 | 0.8933 | 0.9451 | 0.8136 | 0.7059 | 0.9600 | 10 | 1 |
| GradientBoosting | FallTop12_WithDirectionRelevant | 12 | 0.7856 | 0.7826 | 0.7677 | 0.8065 | 0.6757 | 1.0000 | 12 | 0 |
| TinyGB30_d3 | FallTop12_SelectedByImportance | 12 | 0.8306 | 0.8345 | 0.8571 | 0.8000 | 0.7333 | 0.8800 | 8 | 3 |
| TinyGB10_d3 | FallTop12_SelectedByImportance | 12 | 0.8156 | 0.8175 | 0.8293 | 0.8000 | 0.7333 | 0.8800 | 8 | 3 |
| TinyGB20_d3 | FallTop12_SelectedByImportance | 12 | 0.8079 | 0.8088 | 0.8148 | 0.8000 | 0.7333 | 0.8800 | 8 | 3 |
| TinyGB30_d2 | FallTop12_SelectedByImportance | 12 | 0.7965 | 0.7970 | 0.8000 | 0.7925 | 0.7500 | 0.8400 | 7 | 4 |
| TinyGB10_d2 | FallNoTiming_Full_132 | 132 | 0.8089 | 0.8116 | 0.8293 | 0.7857 | 0.7097 | 0.8800 | 9 | 3 |
| LogisticRegression | FallNoTiming_Full_132 | 132 | 0.8051 | 0.8092 | 0.8235 | 0.7826 | 0.8571 | 0.7200 | 3 | 7 |
| TinyGB10_d3 | FallNoTiming_Full_132 | 132 | 0.8133 | 0.8169 | 0.8434 | 0.7797 | 0.6765 | 0.9200 | 11 | 2 |
| RandomForest | FallTop12_WithDirectionRelevant | 12 | 0.8040 | 0.8077 | 0.8247 | 0.7797 | 0.6765 | 0.9200 | 11 | 2 |
| TinyGB10_d3 | FallTop10_SelectedByImportance | 10 | 0.8053 | 0.8088 | 0.8293 | 0.7778 | 0.7241 | 0.8400 | 8 | 4 |
| TinyGB10_d2 | FallTop12_SelectedByImportance | 12 | 0.8023 | 0.8058 | 0.8293 | 0.7719 | 0.6875 | 0.8800 | 10 | 3 |
| TinyGB20_d3 | FallTop10_SelectedByImportance | 10 | 0.8322 | 0.8414 | 0.8966 | 0.7586 | 0.6667 | 0.8800 | 11 | 3 |
| TinyGB20_d2 | FallTop12_SelectedByImportance | 12 | 0.7960 | 0.8000 | 0.8293 | 0.7586 | 0.6667 | 0.8800 | 11 | 3 |
| TinyGB30_d2 | FallTop10_SelectedByImportance | 10 | 0.7960 | 0.8000 | 0.8293 | 0.7586 | 0.6667 | 0.8800 | 11 | 3 |

### Best Candidate Per Feature Set

| feature_set | model_name | feature_count | avg_fall_f1 | E3_fall_f1 | E6_fall_f1 | E7_fall_f1 | E7_recall | E7_FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FallNoTiming_Full_132 | GradientBoosting | 132 | 0.8687 | 0.8732 | 0.8966 | 0.8364 | 0.9200 | 7 |
| FallNoTiming_Core_16 | RandomForest | 16 | 0.8194 | 0.8182 | 0.8125 | 0.8276 | 0.9600 | 9 |
| FallTop12_WithDirectionRelevant | GradientBoosting | 12 | 0.7856 | 0.7826 | 0.7677 | 0.8065 | 1.0000 | 12 |
| FallTop12_SelectedByImportance | TinyGB30_d3 | 12 | 0.8306 | 0.8345 | 0.8571 | 0.8000 | 0.8800 | 8 |
| FallTop10_SelectedByImportance | TinyGB10_d3 | 10 | 0.8053 | 0.8088 | 0.8293 | 0.7778 | 0.8400 | 8 |
| FallTop8_ImpactEnergy | RandomForest | 8 | 0.7570 | 0.7568 | 0.7556 | 0.7586 | 0.8800 | 11 |
| FallTop10_ImpactGyroPosture | GradientBoosting | 10 | 0.7008 | 0.6957 | 0.6733 | 0.7333 | 0.8800 | 13 |
| FallTop12_NoTimingCore | RandomForest | 12 | 0.7394 | 0.7403 | 0.7447 | 0.7333 | 0.8800 | 13 |

### Best Strict <12 Feature Candidates

| model_name | feature_set | feature_count | avg_fall_f1 | E3_fall_f1 | E6_fall_f1 | E7_fall_f1 | E7_precision | E7_recall | E7_FP | E7_FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TinyGB10_d3 | FallTop10_SelectedByImportance | 10 | 0.8053 | 0.8088 | 0.8293 | 0.7778 | 0.7241 | 0.8400 | 8 | 4 |
| TinyGB20_d3 | FallTop10_SelectedByImportance | 10 | 0.8322 | 0.8414 | 0.8966 | 0.7586 | 0.6667 | 0.8800 | 11 | 3 |
| TinyGB30_d2 | FallTop10_SelectedByImportance | 10 | 0.7960 | 0.8000 | 0.8293 | 0.7586 | 0.6667 | 0.8800 | 11 | 3 |
| RandomForest | FallTop8_ImpactEnergy | 8 | 0.7570 | 0.7568 | 0.7556 | 0.7586 | 0.6667 | 0.8800 | 11 | 3 |
| TinyGB10_d2 | FallTop10_SelectedByImportance | 10 | 0.7708 | 0.7727 | 0.7848 | 0.7547 | 0.7143 | 0.8000 | 8 | 5 |
| GradientBoosting | FallTop10_SelectedByImportance | 10 | 0.8354 | 0.8472 | 0.9091 | 0.7500 | 0.6774 | 0.8400 | 10 | 4 |
| TinyGB30_d3 | FallTop10_SelectedByImportance | 10 | 0.8286 | 0.8392 | 0.8966 | 0.7500 | 0.6774 | 0.8400 | 10 | 4 |
| HistGradientBoosting | FallTop10_SelectedByImportance | 10 | 0.8422 | 0.8535 | 0.9348 | 0.7385 | 0.6000 | 0.9600 | 16 | 1 |
| GradientBoosting | FallTop8_ImpactEnergy | 8 | 0.7381 | 0.7383 | 0.7391 | 0.7368 | 0.6562 | 0.8400 | 11 | 4 |
| GradientBoosting | FallTop10_ImpactGyroPosture | 10 | 0.7008 | 0.6957 | 0.6733 | 0.7333 | 0.6286 | 0.8800 | 13 | 3 |

### Best <=12 Feature Candidates

| model_name | feature_set | feature_count | avg_fall_f1 | E3_fall_f1 | E6_fall_f1 | E7_fall_f1 | E7_precision | E7_recall | E7_FP | E7_FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradientBoosting | FallTop12_WithDirectionRelevant | 12 | 0.7856 | 0.7826 | 0.7677 | 0.8065 | 0.6757 | 1.0000 | 12 | 0 |
| TinyGB30_d3 | FallTop12_SelectedByImportance | 12 | 0.8306 | 0.8345 | 0.8571 | 0.8000 | 0.7333 | 0.8800 | 8 | 3 |
| TinyGB10_d3 | FallTop12_SelectedByImportance | 12 | 0.8156 | 0.8175 | 0.8293 | 0.8000 | 0.7333 | 0.8800 | 8 | 3 |
| TinyGB20_d3 | FallTop12_SelectedByImportance | 12 | 0.8079 | 0.8088 | 0.8148 | 0.8000 | 0.7333 | 0.8800 | 8 | 3 |
| TinyGB30_d2 | FallTop12_SelectedByImportance | 12 | 0.7965 | 0.7970 | 0.8000 | 0.7925 | 0.7500 | 0.8400 | 7 | 4 |
| RandomForest | FallTop12_WithDirectionRelevant | 12 | 0.8040 | 0.8077 | 0.8247 | 0.7797 | 0.6765 | 0.9200 | 11 | 2 |
| TinyGB10_d3 | FallTop10_SelectedByImportance | 10 | 0.8053 | 0.8088 | 0.8293 | 0.7778 | 0.7241 | 0.8400 | 8 | 4 |
| TinyGB10_d2 | FallTop12_SelectedByImportance | 12 | 0.8023 | 0.8058 | 0.8293 | 0.7719 | 0.6875 | 0.8800 | 10 | 3 |
| TinyGB20_d3 | FallTop10_SelectedByImportance | 10 | 0.8322 | 0.8414 | 0.8966 | 0.7586 | 0.6667 | 0.8800 | 11 | 3 |
| TinyGB20_d2 | FallTop12_SelectedByImportance | 12 | 0.7960 | 0.8000 | 0.8293 | 0.7586 | 0.6667 | 0.8800 | 11 | 3 |

### Repeated SelectedByImportance Features

| feature | times_selected |
| --- | --- |
| az_range | 10 |
| ay_max | 10 |
| az_std | 6 |
| acc_mag_range | 6 |
| acc_mag_std | 6 |
| post_impact_energy | 6 |
| integrated_gy | 6 |
| peak_signed_ay | 4 |
| tilt_delta_std | 4 |
| tilt_delta_p95 | 4 |
| gz_range | 4 |
| tilt_delta_max | 4 |
| az_iqr | 4 |
| gx_std | 4 |
| ay_mean | 4 |
| integrated_ay | 4 |
| ax_range | 4 |
| pitch_final_initial | 4 |
| post_ay_mean | 4 |
| tilt_delta_iqr | 4 |

### Fall Answers

1. Full 132 features are still needed for the best practical Fall Expert. `GradientBoosting + FallNoTiming_Full_132` remains best on E7 WEDA with Fall F1 `0.8364`, recall `0.9200`, FP `7`, and E3/E6/E7 average Fall F1 `0.8687`.
2. Best strict `<12` feature set is `TinyGB10_d3 + FallTop10_SelectedByImportance` with `10` features, E7 Fall F1 `0.7778`, recall `0.8400`, FP `8`. It does not keep WEDA near the Full132 baseline.
3. Best `<=12` feature set is `GradientBoosting + FallTop12_WithDirectionRelevant` with E7 Fall F1 `0.8065`, recall `1.0000`, FP `12`. It is only `0.0299` below Full132 F1, but FP rises by `5`, so it is a compact ablation rather than the main model.
4. Feature count reduction from 132 is: 12 features = 90.9% fewer, 10 features = 92.4% fewer, 8 features = 93.9% fewer, and Core16 = 87.9% fewer.
5. E7 FP increases materially for the best 12-feature high-recall candidate: `12` FP versus `7` FP for Full132. Core16 gives `9` FP, which is a better compact trade-off but uses 16 features.
6. E7 recall can be kept above 0.90 with compact variants, but the price is more FP. `FallTop12_WithDirectionRelevant` reaches recall `1.0000` with FP `12`; `FallNoTiming_Core_16` reaches recall `0.9600` with FP `9`.
7. TinyGB is not a clean replacement for default GradientBoosting. `TinyGB30_d3 + Full132` reaches E7 Fall F1 `0.8302`, close to baseline, but its E3/E6/E7 average `0.8297` is clearly below GradientBoosting Full132 `0.8687`.
8. Final Fall Expert recommendation: best accuracy/practical = `GradientBoosting + FallNoTiming_Full_132`; best compact = `RandomForest` or `HistGradientBoosting + FallNoTiming_Core_16`; best strict <=12 edge ablation = `GradientBoosting + FallTop12_WithDirectionRelevant`, with FP caveat.

## Lightweight Direction

### Best E3/E6/E7 Direction Candidates

| model_name | seed | learning_rate | params | avg_direction_macro_f1 | std_direction_macro_f1 | E3_direction_macro_f1 | E6_direction_macro_f1 | E7_direction_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1_Tiny_DSConv_Direction | 42 | 0.0005 | 2947 | 0.8267 | 0.0476 | 0.8197 | 0.7831 | 0.8774 |
| D1_Tiny_DSConv_Direction | 42 | 0.0010 | 2947 | 0.8163 | 0.0654 | 0.8098 | 0.7544 | 0.8847 |
| D2_Micro_TCN_Direction | 42 | 0.0005 | 3523 | 0.7942 | 0.0077 | 0.8024 | 0.7872 | 0.7930 |
| D3_A5_Direction_Small_050 | 42 | 0.0005 | 6223 | 0.7841 | 0.0232 | 0.7880 | 0.8051 | 0.7593 |
| D2_Micro_TCN_Direction | 42 | 0.0010 | 3523 | 0.7786 | 0.0681 | 0.8018 | 0.8321 | 0.7019 |
| D6_Summary_MLP_DirectionCore | 42 | 0.0010 | 2659 | 0.7462 | 0.0694 | 0.7622 | 0.8063 | 0.6702 |
| D6_Summary_MLP_DirectionCore | 42 | 0.0005 | 2659 | 0.7341 | 0.0586 | 0.7470 | 0.7852 | 0.6702 |
| D5_Tiny_GRU_Direction | 42 | 0.0005 | 1763 | 0.6690 | 0.0762 | 0.6609 | 0.5972 | 0.7489 |
| D3_A5_Direction_Small_050 | 42 | 0.0010 | 6223 | 0.6567 | 0.0597 | 0.6582 | 0.7157 | 0.5963 |
| D5_Tiny_GRU_Direction | 42 | 0.0010 | 1763 | 0.6470 | 0.0538 | 0.6397 | 0.5972 | 0.7041 |
| D4_A5_Direction_Small_025 | 42 | 0.0005 | 1761 | 0.6436 | 0.0101 | 0.6474 | 0.6512 | 0.6321 |
| D4_A5_Direction_Small_025 | 42 | 0.0010 | 1761 | 0.6350 | 0.1073 | 0.6549 | 0.7310 | 0.5191 |

### Best E7 WEDA Direction Candidates

| model_name | seed | learning_rate | params | direction_macro_f1 | direction_accuracy | forward_f1 | backward_f1 | lateral_f1 | delta_vs_A5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1_Tiny_DSConv_Direction | 42 | 0.0010 | 2947 | 0.8847 | 0.8800 | 0.8889 | 0.8421 | 0.9231 | 0.0457 |
| D1_Tiny_DSConv_Direction | 42 | 0.0005 | 2947 | 0.8774 | 0.8800 | 0.9000 | 0.8750 | 0.8571 | 0.0383 |
| D2_Micro_TCN_Direction | 42 | 0.0005 | 3523 | 0.7930 | 0.8000 | 0.8235 | 0.8889 | 0.6667 | -0.0460 |
| D3_A5_Direction_Small_050 | 42 | 0.0005 | 6223 | 0.7593 | 0.7600 | 0.8750 | 0.7778 | 0.6250 | -0.0798 |
| D5_Tiny_GRU_Direction | 42 | 0.0005 | 1763 | 0.7489 | 0.7600 | 0.8182 | 0.7143 | 0.7143 | -0.0901 |
| D5_Tiny_GRU_Direction | 42 | 0.0010 | 1763 | 0.7041 | 0.7200 | 0.7826 | 0.6154 | 0.7143 | -0.1349 |
| D2_Micro_TCN_Direction | 42 | 0.0010 | 3523 | 0.7019 | 0.7200 | 0.6250 | 0.9474 | 0.5333 | -0.1371 |
| D6_Summary_MLP_DirectionCore | 42 | 0.0010 | 2659 | 0.6702 | 0.6800 | 0.5333 | 0.7500 | 0.7273 | -0.1688 |
| D6_Summary_MLP_DirectionCore | 42 | 0.0005 | 2659 | 0.6702 | 0.6800 | 0.5333 | 0.7500 | 0.7273 | -0.1688 |
| D4_A5_Direction_Small_025 | 42 | 0.0005 | 1761 | 0.6321 | 0.6800 | 0.8571 | 0.7059 | 0.3333 | -0.2069 |

### End-to-End Direction With Best Fall Expert

| pipeline | avg_E2E_direction_macro_f1 | E3_E2E_direction_macro_f1 | E6_E2E_direction_macro_f1 | E7_E2E_direction_macro_f1 |
| --- | --- | --- | --- | --- |
| D1_Tiny_DSConv_Direction_E3_BITS_WEDA_MIXED_seed42_lr0.0005 | 0.7153 | 0.7149 | 0.5980 | 0.8331 |
| D1_Tiny_DSConv_Direction_E3_BITS_WEDA_MIXED_seed42_lr0.001 | 0.7117 | 0.7115 | 0.5822 | 0.8415 |
| D2_Micro_TCN_Direction_E3_BITS_WEDA_MIXED_seed42_lr0.001 | 0.7089 | 0.7397 | 0.6918 | 0.6951 |
| D2_Micro_TCN_Direction_E3_BITS_WEDA_MIXED_seed42_lr0.0005 | 0.7061 | 0.7229 | 0.6487 | 0.7467 |
| D3_A5_Direction_Small_050_E3_BITS_WEDA_MIXED_seed42_lr0.0005 | 0.6754 | 0.6859 | 0.6301 | 0.7103 |
| D6_Summary_MLP_DirectionCore_E3_BITS_WEDA_MIXED_seed42_lr0.0005 | 0.6657 | 0.6828 | 0.6517 | 0.6626 |
| D6_Summary_MLP_DirectionCore_E3_BITS_WEDA_MIXED_seed42_lr0.001 | 0.6657 | 0.6828 | 0.6517 | 0.6626 |
| D3_A5_Direction_Small_050_E3_BITS_WEDA_MIXED_seed42_lr0.001 | 0.5723 | 0.5731 | 0.6175 | 0.5265 |
| D5_Tiny_GRU_Direction_E3_BITS_WEDA_MIXED_seed42_lr0.0005 | 0.5680 | 0.5619 | 0.4449 | 0.6972 |
| D5_Tiny_GRU_Direction_E3_BITS_WEDA_MIXED_seed42_lr0.001 | 0.5423 | 0.5349 | 0.4449 | 0.6472 |
| D4_A5_Direction_Small_025_E3_BITS_WEDA_MIXED_seed42_lr0.001 | 0.5419 | 0.5627 | 0.5630 | 0.5000 |
| D4_A5_Direction_Small_025_E3_BITS_WEDA_MIXED_seed42_lr0.0005 | 0.5381 | 0.5413 | 0.4869 | 0.5861 |

### Complexity

| model_name | learning_rate | params | estimated_fp32_kb | estimated_int8_kb | tflite_convert | int8_tflite_convert | edge_suitability | complexity_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1_Tiny_DSConv_Direction | 0.0005 | 2947 | 11.5117 | 2.8779 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high | PyTorch temporal; params=2947; TFLite unavailable because TensorFlow import status: False |
| D1_Tiny_DSConv_Direction | 0.0010 | 2947 | 11.5117 | 2.8779 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high | PyTorch temporal; params=2947; TFLite unavailable because TensorFlow import status: False |
| D2_Micro_TCN_Direction | 0.0005 | 3523 | 13.7617 | 3.4404 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high | PyTorch temporal; params=3523; TFLite unavailable because TensorFlow import status: False |
| D2_Micro_TCN_Direction | 0.0010 | 3523 | 13.7617 | 3.4404 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high | PyTorch temporal; params=3523; TFLite unavailable because TensorFlow import status: False |
| D3_A5_Direction_Small_050 | 0.0005 | 6223 | 24.3086 | 6.0771 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high | PyTorch temporal; params=6223; TFLite unavailable because TensorFlow import status: False |
| D3_A5_Direction_Small_050 | 0.0010 | 6223 | 24.3086 | 6.0771 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high | PyTorch temporal; params=6223; TFLite unavailable because TensorFlow import status: False |
| D4_A5_Direction_Small_025 | 0.0005 | 1761 | 6.8789 | 1.7197 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high | PyTorch temporal; params=1761; TFLite unavailable because TensorFlow import status: False |
| D4_A5_Direction_Small_025 | 0.0010 | 1761 | 6.8789 | 1.7197 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high | PyTorch temporal; params=1761; TFLite unavailable because TensorFlow import status: False |
| D5_Tiny_GRU_Direction | 0.0005 | 1763 | 6.8867 | 1.7217 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high | PyTorch temporal; params=1763; TFLite unavailable because TensorFlow import status: False |
| D5_Tiny_GRU_Direction | 0.0010 | 1763 | 6.8867 | 1.7217 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high | PyTorch temporal; params=1763; TFLite unavailable because TensorFlow import status: False |
| D6_Summary_MLP_DirectionCore | 0.0005 | 2659 | 10.3867 | 2.5967 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high | PyTorch summary; params=2659; TFLite unavailable because TensorFlow import status: False |
| D6_Summary_MLP_DirectionCore | 0.0010 | 2659 | 10.3867 | 2.5967 | not_available_tensorflow_import_failed | not_available_tensorflow_import_failed | high | PyTorch summary; params=2659; TFLite unavailable because TensorFlow import status: False |

### Direction Answers

1. Best lightweight direction model overall is `D1_Tiny_DSConv_Direction`, seed `42`, lr `0.0005`, params `2947`, with average E3/E6/E7 Direction Macro F1 `0.8267`.
2. Best params/F1 trade-off is also `D1_Tiny_DSConv_Direction`: it has only `2,947` params and is the only lightweight model that stays near the 0.82 average macro-F1 range.
3. No model under 30k params reaches E3 Direction Macro F1 >= 0.86. Count passing this condition: `0`.
4. No model under 10k params reaches E3 Direction Macro F1 >= 0.84. Count passing this condition: `0`.
5. Best on E7 WEDA is `D1_Tiny_DSConv_Direction`, lr `0.001`, params `2947`, E7 Direction Macro F1 `0.8847`, delta vs A5 `0.0457`.
6. Stability across E3/E6/E7 is still below A5. The best lightweight average drops by about `0.0635` macro F1 versus A5 reference average `0.8902`.
7. Lightweight direction should not replace A5 as the paper main model. It is a useful edge ablation, especially for WEDA, but it fails the E3 consistency target.
8. Best lightweight E2E direction average is `0.7153`; it remains lower than the A5-based Hybrid_NoTiming direction path, so replacing A5 reduces end-to-end consistency even when E7 improves.
9. TFLite/INT8 conversion did not run because TensorFlow is broken in this environment. Complexity is therefore reported using param-based FP32/INT8 estimates only.

## Final Architecture Recommendation

### Best Practical

- Keep `Hybrid_NoTiming_Full`: Fall = `GradientBoosting + FallNoTiming_Full_132`, Direction = A5 direction head.
- Reason: it has the strongest WEDA fall result among tested non-timing variants and keeps the established A5 direction stability.
- Do not use FullTiming as a main result; it remains excluded because timing-index artifacts are not acceptable for the main benchmark.

### Best Compact

- Use `FallNoTiming_Core_16` as the compact fall ablation: `RandomForest` or `HistGradientBoosting` reaches E7 Fall F1 `0.8276`, recall `0.9600`, FP `9`.
- Keep A5 direction head if direction quality is important. Use `D1_Tiny_DSConv_Direction` only when edge model size is prioritized over E3/E6/E7 direction consistency.

### Paper-Facing Conclusion

The experiments support a conservative paper claim: removing timing artifacts is feasible, but the fall expert still benefits from a broad no-timing summary representation. The 8/10/12-feature fall experts are attractive ablations but do not reliably preserve WEDA Fall F1 without increasing false positives. For direction, a tiny DSConv head can improve WEDA-only direction and cuts parameters sharply, but A5 remains the main direction expert because cross-protocol consistency is better. The final model for the paper should therefore remain `Hybrid_NoTiming_Full + A5 direction`, with `Core16` and `Tiny_DSConv_Direction` reported as compact/edge ablations.

## Output Files

- `outputs\reports\hybrid_feature_reduction_direction_lightweight\fall_feature_reduction_results.csv`
- `outputs\reports\hybrid_feature_reduction_direction_lightweight\fall_feature_reduction_best_by_protocol.csv`
- `outputs\reports\hybrid_feature_reduction_direction_lightweight\fall_feature_reduction_feature_importance.csv`
- `outputs\reports\hybrid_feature_reduction_direction_lightweight\fall_selected_features_by_protocol.csv`
- `outputs\reports\hybrid_feature_reduction_direction_lightweight\fall_feature_reduction_e2e_direction.csv`
- `outputs\reports\hybrid_feature_reduction_direction_lightweight\lightweight_direction_results.csv`
- `outputs\reports\hybrid_feature_reduction_direction_lightweight\lightweight_direction_best_by_protocol.csv`
- `outputs\reports\hybrid_feature_reduction_direction_lightweight\lightweight_direction_vs_a5.csv`
- `outputs\reports\hybrid_feature_reduction_direction_lightweight\lightweight_direction_complexity.csv`
- `outputs\reports\hybrid_feature_reduction_direction_lightweight\lightweight_direction_e2e_results.csv`
- `outputs\reports\hybrid_feature_reduction_direction_lightweight\final_hybrid_variants_comparison.csv`
- `outputs\reports\hybrid_feature_reduction_direction_lightweight\feature_set_configs.json`
- `outputs\reports\hybrid_feature_reduction_direction_lightweight\failed_runs.csv`
- `outputs\reports\hybrid_feature_reduction_direction_lightweight\missing_features.csv`
- figures: `outputs\figures\hybrid_feature_reduction_direction_lightweight`
- models: `outputs\models\hybrid_feature_reduction_direction_lightweight`

