# DS-Fall-RD Targeted Experiments Summary

Scope: BITS/WEDA only, event-centered 2-second windows at 25 Hz, input `(50, 12)` unless a feature ablation changes the channel count. No HIFD/UMAFall, no Transformer, no heavy module.

## Current Reference

The current main model remains `REF_CURRENT_E3_ARTIFACT`, copied from the existing E3/E6/E7 event-centered artifact:

| metric | value |
| --- | ---: |
| E3 Fall F1 | 0.8323 |
| E3 Direction Macro F1 | 0.8967 |
| E6 BITS Fall F1 | 0.9451 |
| E6 BITS Direction Macro F1 | 0.9350 |
| E7 WEDA Fall F1 | 0.6857 |
| E7 WEDA Direction Macro F1 | 0.8390 |
| Params | 65959 |

## Main Ranking

| run_id | seed | group_id | config_id | feature_set | sampler | fall_loss | alpha_fall | lambda_dir | fall_threshold | fall_f1 | direction_macro_f1 | bits_fall_f1 | bits_direction_macro_f1 | weda_fall_f1 | weda_direction_macro_f1 | params | selection_score | passes_hard_filter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| REF_CURRENT_E3_ARTIFACT | 42 | reference | REF | tilt12 | current | weighted_ce | 1.0000 | 1.5000 | 0.5000 | 0.8323 | 0.8967 | 0.9451 | 0.9350 | 0.6857 | 0.8390 | 65959 | 0.8227 | True |
| L3_FALL_WCE | 42 | loss | L3 | tilt12 | current | weighted_ce | 2.0000 | 1.0000 | 0.8300 | 0.8732 | 0.8298 | 0.9091 | 0.8036 | 0.8148 | 0.8390 | 65959 | 0.8434 | False |
| FA8_TILT12 | 42 | feature_ablation | FA8 | tilt12 | current | weighted_ce | 2.0000 | 1.0000 | 0.9000 | 0.8784 | 0.7618 | 0.9213 | 0.7840 | 0.8136 | 0.7243 | 65959 | 0.8054 | False |
| W100_WIDTH | 42 | compactness | W100 | tilt12 | current | weighted_ce | 2.0000 | 1.0000 | 0.9000 | 0.8784 | 0.7618 | 0.9213 | 0.7840 | 0.8136 | 0.7243 | 65959 | 0.8054 | False |
| FA0_RAW6 | 42 | feature_ablation | FA0 | raw6 | current | weighted_ce | 2.0000 | 1.0000 | 0.7200 | 0.8369 | 0.7203 | 0.8706 | 0.8181 | 0.7857 | 0.6019 | 65479 | 0.7504 | False |
| D32_16_HEAD | 42 | compactness | D32_16 | tilt12 | current | weighted_ce | 2.0000 | 1.0000 | 0.4100 | 0.8667 | 0.7278 | 0.9231 | 0.8071 | 0.7797 | 0.6197 | 66935 | 0.7642 | False |
| L2_FALL_WCE | 42 | loss | L2 | tilt12 | current | weighted_ce | 1.5000 | 1.0000 | 0.8000 | 0.8435 | 0.7349 | 0.8966 | 0.8019 | 0.7667 | 0.6353 | 65959 | 0.7596 | False |
| S5_DATASET_BALANCED | 123 | sampler | S5 | tilt12 | dataset_balanced | focal | 1.0000 | 1.0000 | 0.7200 | 0.7947 | 0.6855 | 0.8132 | 0.7731 | 0.7667 | 0.5444 | 65959 | 0.7117 | False |
| FA7_FA7_NO_GYRO_MAG11 | 42 | feature_ablation | FA7 | fa7_no_gyro_mag11 | current | weighted_ce | 2.0000 | 1.0000 | 0.8900 | 0.8333 | 0.6883 | 0.8837 | 0.7504 | 0.7586 | 0.5985 | 65879 | 0.7353 | False |
| B0_BASE_A5WCEFW | 42 | baseline | B0 | tilt12 | current | weighted_ce | 1.0000 | 1.5000 | 0.9000 | 0.8435 | 0.7025 | 0.8989 | 0.8450 | 0.7586 | 0.5070 | 65959 | 0.7238 | False |
| FA6_FA6_NO_JERK11 | 42 | feature_ablation | FA6 | fa6_no_jerk11 | current | weighted_ce | 2.0000 | 1.0000 | 0.8900 | 0.8387 | 0.7507 | 0.8936 | 0.7668 | 0.7541 | 0.7148 | 65879 | 0.7748 | False |
| S3_DATASET_BALANCED | 123 | sampler | S3 | tilt12 | dataset_balanced | weighted_ce | 2.0000 | 1.0000 | 0.8000 | 0.8553 | 0.7115 | 0.9231 | 0.7348 | 0.7541 | 0.6774 | 65959 | 0.7636 | False |
| L6_FALL2_DIRW15 | 42 | loss | L6 | tilt12 | current | weighted_ce | 2.0000 | 1.5000 | 0.8000 | 0.8344 | 0.7042 | 0.8889 | 0.8106 | 0.7541 | 0.5569 | 65959 | 0.7304 | False |
| FL2_FALL_FOCAL | 42 | loss | FL2 | tilt12 | current | focal | 1.0000 | 1.0000 | 0.7700 | 0.8322 | 0.6646 | 0.8864 | 0.7568 | 0.7541 | 0.5437 | 65959 | 0.7173 | False |
| S4_DATASET_BALANCED | 42 | sampler | S4 | tilt12 | dataset_balanced | focal | 1.0000 | 1.0000 | 0.7300 | 0.8129 | 0.8006 | 0.8571 | 0.7668 | 0.7500 | 0.8380 | 65959 | 0.8026 | False |
| W125_WIDTH | 42 | compactness | W125 | tilt12 | current | weighted_ce | 2.0000 | 1.0000 | 0.7000 | 0.8378 | 0.6293 | 0.8913 | 0.7531 | 0.7500 | 0.4702 | 99111 | 0.6687 | False |
| FA1_FA1_ACC_MAG7 | 42 | feature_ablation | FA1 | fa1_acc_mag7 | current | weighted_ce | 2.0000 | 1.0000 | 0.9000 | 0.8378 | 0.6597 | 0.8989 | 0.6185 | 0.7458 | 0.5908 | 65559 | 0.7254 | False |
| L1_FALL_WCE | 42 | loss | L1 | tilt12 | current | weighted_ce | 1.0000 | 1.0000 | 0.5800 | 0.8079 | 0.7508 | 0.8539 | 0.8281 | 0.7419 | 0.6389 | 65959 | 0.7470 | False |

## Loss Tuning

| run_id | seed | group_id | config_id | feature_set | sampler | fall_loss | alpha_fall | lambda_dir | fall_threshold | fall_f1 | direction_macro_f1 | bits_fall_f1 | bits_direction_macro_f1 | weda_fall_f1 | weda_direction_macro_f1 | params | selection_score | passes_hard_filter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L3_FALL_WCE | 42 | loss | L3 | tilt12 | current | weighted_ce | 2.0000 | 1.0000 | 0.8300 | 0.8732 | 0.8298 | 0.9091 | 0.8036 | 0.8148 | 0.8390 | 65959 | 0.8434 | False |
| L2_FALL_WCE | 42 | loss | L2 | tilt12 | current | weighted_ce | 1.5000 | 1.0000 | 0.8000 | 0.8435 | 0.7349 | 0.8966 | 0.8019 | 0.7667 | 0.6353 | 65959 | 0.7596 | False |
| FL2_FALL_FOCAL | 42 | loss | FL2 | tilt12 | current | focal | 1.0000 | 1.0000 | 0.7700 | 0.8322 | 0.6646 | 0.8864 | 0.7568 | 0.7541 | 0.5437 | 65959 | 0.7173 | False |
| L6_FALL2_DIRW15 | 42 | loss | L6 | tilt12 | current | weighted_ce | 2.0000 | 1.5000 | 0.8000 | 0.8344 | 0.7042 | 0.8889 | 0.8106 | 0.7541 | 0.5569 | 65959 | 0.7304 | False |
| L1_FALL_WCE | 42 | loss | L1 | tilt12 | current | weighted_ce | 1.0000 | 1.0000 | 0.5800 | 0.8079 | 0.7508 | 0.8539 | 0.8281 | 0.7419 | 0.6389 | 65959 | 0.7470 | False |
| FL1_FALL_FOCAL | 42 | loss | FL1 | tilt12 | current | focal | 1.0000 | 1.0000 | 0.5500 | 0.8169 | 0.6949 | 0.8706 | 0.8174 | 0.7368 | 0.5231 | 65959 | 0.7117 | False |
| FL3_FALL_FOCAL | 42 | loss | FL3 | tilt12 | current | focal | 1.0000 | 1.0000 | 0.5900 | 0.8493 | 0.7151 | 0.9213 | 0.8394 | 0.7368 | 0.5571 | 65959 | 0.7344 | False |
| L7_FALL2_DIRW20 | 42 | loss | L7 | tilt12 | current | weighted_ce | 2.0000 | 2.0000 | 0.7300 | 0.8414 | 0.8003 | 0.9091 | 0.8513 | 0.7368 | 0.7206 | 65959 | 0.7858 | False |
| L5_FALL_WCE | 42 | loss | L5 | tilt12 | current | weighted_ce | 2.0000 | 0.7500 | 0.6700 | 0.8421 | 0.6303 | 0.9130 | 0.6263 | 0.7333 | 0.6381 | 65959 | 0.7264 | False |
| FL4_FALL_FOCAL | 42 | loss | FL4 | tilt12 | current | focal | 1.0000 | 1.0000 | 0.5300 | 0.8235 | 0.7091 | 0.8889 | 0.8513 | 0.7302 | 0.5292 | 65959 | 0.7178 | False |
| L4_FALL_WCE | 42 | loss | L4 | tilt12 | current | weighted_ce | 1.5000 | 0.7500 | 0.8000 | 0.8289 | 0.7680 | 0.9091 | 0.8063 | 0.7188 | 0.7100 | 65959 | 0.7689 | False |
| L4_FALL_WCE | 123 | loss | L4 | tilt12 | current | weighted_ce | 1.5000 | 0.7500 | 0.6100 | 0.8072 | 0.7257 | 0.8776 | 0.7870 | 0.7059 | 0.5500 | 65959 | 0.7155 | False |

Main observation: L3 (`alpha_fall=2.0`, `lambda_dir=1.0`, direction CE) gives the largest WEDA Fall F1 gain, but it drops E3 Direction Macro F1 below the hard filter.

## Dataset-balanced Sampler

| run_id | seed | group_id | config_id | feature_set | sampler | fall_loss | alpha_fall | lambda_dir | fall_threshold | fall_f1 | direction_macro_f1 | bits_fall_f1 | bits_direction_macro_f1 | weda_fall_f1 | weda_direction_macro_f1 | params | selection_score | passes_hard_filter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S3_DATASET_BALANCED | 42 | sampler | S3 | tilt12 | dataset_balanced | weighted_ce | 2.0000 | 1.0000 | 0.5400 | 0.8497 | 0.8540 | 0.9451 | 0.8535 | 0.7097 | 0.8409 | 65959 | 0.8213 | False |
| S1_DATASET_BALANCED | 42 | sampler | S1 | tilt12 | dataset_balanced | weighted_ce | 1.0000 | 1.0000 | 0.9000 | 0.8333 | 0.8492 | 0.9247 | 0.8577 | 0.6984 | 0.8309 | 65959 | 0.8101 | False |
| S4_DATASET_BALANCED | 42 | sampler | S4 | tilt12 | dataset_balanced | focal | 1.0000 | 1.0000 | 0.7300 | 0.8129 | 0.8006 | 0.8571 | 0.7668 | 0.7500 | 0.8380 | 65959 | 0.8026 | False |
| S5_DATASET_BALANCED | 7 | sampler | S5 | tilt12 | dataset_balanced | focal | 1.0000 | 1.0000 | 0.9000 | 0.7692 | 0.8827 | 0.8966 | 0.8649 | 0.6087 | 0.9214 | 65959 | 0.7990 | False |
| S5_DATASET_BALANCED | 42 | sampler | S5 | tilt12 | dataset_balanced | focal | 1.0000 | 1.0000 | 0.9000 | 0.8356 | 0.8018 | 0.9213 | 0.8063 | 0.7018 | 0.7963 | 65959 | 0.7920 | False |
| S3_DATASET_BALANCED | 123 | sampler | S3 | tilt12 | dataset_balanced | weighted_ce | 2.0000 | 1.0000 | 0.8000 | 0.8553 | 0.7115 | 0.9231 | 0.7348 | 0.7541 | 0.6774 | 65959 | 0.7636 | False |
| S2_DATASET_BALANCED | 42 | sampler | S2 | tilt12 | dataset_balanced | weighted_ce | 1.5000 | 1.0000 | 0.8700 | 0.8456 | 0.7565 | 0.9333 | 0.8354 | 0.7119 | 0.6263 | 65959 | 0.7525 | False |
| S3_DATASET_BALANCED | 7 | sampler | S3 | tilt12 | dataset_balanced | weighted_ce | 2.0000 | 1.0000 | 0.8100 | 0.8188 | 0.7318 | 0.8864 | 0.7993 | 0.7213 | 0.6331 | 65959 | 0.7408 | False |
| S5_DATASET_BALANCED | 123 | sampler | S5 | tilt12 | dataset_balanced | focal | 1.0000 | 1.0000 | 0.7200 | 0.7947 | 0.6855 | 0.8132 | 0.7731 | 0.7667 | 0.5444 | 65959 | 0.7117 | False |

Main observation: S3 keeps BITS Fall F1 and E7 Direction strong, but WEDA Fall F1 improves only modestly. It is the best direction-preserving training ablation, not the best main model.

## Feature Ablation

| run_id | feature_set | input_shape | weda_fall_f1 | weda_direction_macro_f1 | direction_macro_f1 | bits_fall_f1 | bits_direction_macro_f1 | params | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FA8_TILT12 | tilt12 | (50, 12) | 0.8136 | 0.7243 | 0.7618 | 0.9213 | 0.7840 | 65959 | 0.8054 |
| FA0_RAW6 | raw6 | (50, 6) | 0.7857 | 0.6019 | 0.7203 | 0.8706 | 0.8181 | 65479 | 0.7504 |
| FA7_FA7_NO_GYRO_MAG11 | fa7_no_gyro_mag11 | (50, 11) | 0.7586 | 0.5985 | 0.6883 | 0.8837 | 0.7504 | 65879 | 0.7353 |
| FA6_FA6_NO_JERK11 | fa6_no_jerk11 | (50, 11) | 0.7541 | 0.7148 | 0.7507 | 0.8936 | 0.7668 | 65879 | 0.7748 |
| FA1_FA1_ACC_MAG7 | fa1_acc_mag7 | (50, 7) | 0.7458 | 0.5908 | 0.6597 | 0.8989 | 0.6185 | 65559 | 0.7254 |
| FA3_MAG_JERK9 | mag_jerk9 | (50, 9) | 0.7368 | 0.6263 | 0.6557 | 0.8736 | 0.6484 | 65719 | 0.7232 |
| FA4_FA4_ROLL_PITCH8 | fa4_roll_pitch8 | (50, 8) | 0.7119 | 0.5597 | 0.6033 | 0.8837 | 0.6318 | 65639 | 0.6903 |
| FA2_FA2_MAG8 | fa2_mag8 | (50, 8) | 0.7077 | 0.5943 | 0.6829 | 0.9111 | 0.6848 | 65639 | 0.7211 |
| FA5_FA5_TILT9 | fa5_tilt9 | (50, 9) | 0.6866 | 0.6238 | 0.7188 | 0.8367 | 0.8291 | 65719 | 0.7135 |

Main observation: `tilt12` remains the safest feature set. Raw6 can raise WEDA Fall F1 in one run but hurts direction. Removing jerk or gyro magnitude reduces stability.

## Compactness

| run_id | width_multiplier | shared_dense | head_dense | weda_fall_f1 | weda_direction_macro_f1 | direction_macro_f1 | bits_fall_f1 | params | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W100_WIDTH | 1.0000 | 0 | 32 | 0.8136 | 0.7243 | 0.7618 | 0.9213 | 65959 | 0.8054 |
| D32_16_HEAD | 1.0000 | 32 | 16 | 0.7797 | 0.6197 | 0.7278 | 0.9231 | 66935 | 0.7642 |
| W125_WIDTH | 1.2500 | 0 | 32 | 0.7500 | 0.4702 | 0.6293 | 0.8913 | 99111 | 0.6687 |
| W075_WIDTH | 0.7500 | 0 | 32 | 0.7188 | 0.7012 | 0.6821 | 0.9231 | 39495 | 0.7487 |
| W050_WIDTH | 0.5000 | 0 | 32 | 0.6786 | 0.6633 | 0.6621 | 0.8602 | 19719 | 0.7100 |
| D48_24_HEAD | 1.0000 | 48 | 24 | 0.6667 | 0.6349 | 0.7091 | 0.8750 | 71375 | 0.7109 |

Main observation: width 0.75 and 0.50 reduce capacity too much for this split. W100 remains the best compactness point among the tested settings.

## Notes

- Global validation-tuned thresholds were used for trained targeted runs. Dataset-aware thresholds were not used as main results.
- Existing E3 checkpoint threshold tuning could not be rerun directly because the old Keras `.keras` artifact contains Lambda layers that fail deserialization in the current Keras runtime. The reference row therefore uses the already saved E3/E6/E7 metrics at threshold 0.5.
- No newly trained targeted run passed all hard filters. The reference model is the only strict-pass model in this run set.
