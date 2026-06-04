# DS-Fall-RD-FV Summary

## Protocol

- Dataset: BITS + WEDA only.
- Sampling/window: 25 Hz, 2-second event-centered, tilt12 temporal reference input.
- Reference: DS-Fall-RD A5WCEFW / REF_CURRENT_E3_ARTIFACT.
- FV training: frozen deep reference fall probability plus Logistic Regression verifier over summary features. Direction probabilities are copied from the reference and remain unchanged.
- Thresholds are global and tuned on validation only.

## E7 WEDA Results

| run_id | features | fusion | hard_negative_weight | fall_f1 | fall_precision | fall_recall | FP | FN | direction_macro_f1 | E3_Direction_Macro_F1 | E6_BITS_Fall_F1 | params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FV1_LITE10_AND | FV-Lite10 | and | 2.0000 | 0.7500 | 0.6154 | 0.9600 | 15 | 1 | 0.8390 | 0.8967 | 0.9451 | 65970 |
| FV2_LITE10_LOGIT_FUSION | FV-Lite10 | logit_fusion | 2.0000 | 0.7385 | 0.6000 | 0.9600 | 16 | 1 | 0.8390 | 0.8967 | 0.9451 | 65973 |
| FV5_LITE10_LOGIT_FUSION_HN3 | FV-Lite10 | logit_fusion | 3.0000 | 0.7385 | 0.6000 | 0.9600 | 16 | 1 | 0.8390 | 0.8967 | 0.9451 | 65973 |
| REF_CURRENT_E3_ARTIFACT | temporal_tilt12 | reference_softmax |  | 0.6857 | 0.5333 | 0.9600 | 21 | 1 | 0.8390 | 0.8967 | 0.9451 | 65959 |
| FV3_CENTER12_LOGIT_FUSION | FV-Center12 | logit_fusion | 2.0000 | 0.6857 | 0.5333 | 0.9600 | 21 | 1 | 0.8390 | 0.8967 | 0.9451 | 65975 |
| FV4_LITE10_GATED_PRODUCT | FV-Lite10 | gated_product | 2.0000 | 0.6761 | 0.5217 | 0.9600 | 22 | 1 | 0.8390 | 0.8967 | 0.9451 | 65970 |

## Thresholds

| run_id | features | fusion | hard_negative_weight | status | threshold | t_deep | t_lr | alpha_gate | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FV1_LITE10_AND | FV-Lite10 | and | 2.0000 | completed |  | 0.8900 | 0.0052 |  |  |
| FV2_LITE10_LOGIT_FUSION | FV-Lite10 | logit_fusion | 2.0000 | completed | 0.9108 |  |  |  |  |
| FV3_CENTER12_LOGIT_FUSION | FV-Center12 | logit_fusion | 2.0000 | completed | 0.2674 |  |  |  |  |
| FV4_LITE10_GATED_PRODUCT | FV-Lite10 | gated_product | 2.0000 | completed | 0.1931 |  |  | 0.3000 |  |
| FV5_LITE10_LOGIT_FUSION_HN3 | FV-Lite10 | logit_fusion | 3.0000 | completed | 0.9108 |  |  |  |  |
| FV6_LITE10_LOGIT_FUSION_UNFREEZE_FALL | FV-Lite10 | stage3_unfreeze_fall | 2.0000 | not_run |  |  |  |  | Skipped because safe fall-head unfreezing would require rebuilding a trainable temporal FV graph with explicit fall-logit exposure and verifying unchanged direction outputs. This frozen-probability FV script completed FV1-FV5 and records FV6 as not run instead of applying an unsafe partial unfreeze. |

## Model Params

| run_id | params | added_params_over_reference | summary_feature_count | timing_features_used | direction_changed | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| REF_CURRENT_E3_ARTIFACT | 65959 | 0 | 0 | 0.0000 | 0.0000 | reference | Reference DS-Fall-RD A5WCEFW. |
| FV1_LITE10_AND | 65970 | 11 | 10 | 0.0000 | 0.0000 | completed | AND fusion with Lite10 verifier. |
| FV2_LITE10_LOGIT_FUSION | 65973 | 14 | 10 | 0.0000 | 0.0000 | completed | Preferred clean FV architecture. |
| FV3_CENTER12_LOGIT_FUSION | 65975 | 16 | 12 | 1.0000 | 0.0000 | completed | Upper-bound with center timing features. |
| FV4_LITE10_GATED_PRODUCT | 65970 | 11 | 10 | 0.0000 | 0.0000 | completed | Gated-product verifier. |
| FV5_LITE10_LOGIT_FUSION_HN3 | 65973 | 14 | 10 | 0.0000 | 0.0000 | completed | Same as FV2 with hard-negative weight 3. |
| FV6_LITE10_LOGIT_FUSION_UNFREEZE_FALL |  |  | 10 | 0.0000 | 0.0000 | not_run | Skipped because safe fall-head unfreezing would require rebuilding a trainable temporal FV graph with explicit fall-logit exposure and verifying unchanged direction outputs. This frozen-probability FV script completed FV1-FV5 and records FV6 as not run instead of applying an unsafe partial unfreeze. |

## Answers

1. DS-Fall-RD-FV reduces WEDA FP from 21 to 15 for the best run `FV1_LITE10_AND`. WEDA precision changes from 0.5333 to 0.6154.
2. Best variant by WEDA Fall F1/precision trade-off is `FV1_LITE10_AND` with WEDA Fall F1=0.7500.
3. Lite10 without timing works only modestly; best Lite10 run is `FV1_LITE10_AND` with WEDA Fall F1=0.7500.
4. Center12 upper-bound run `FV3_CENTER12_LOGIT_FUSION` gets WEDA Fall F1=0.6857; because it uses impact-center timing, treat it as protocol-specific.
5. Logit fusion best WEDA Fall F1=0.7385; AND best=0.7500; gated best=0.6761.
6. Hard-negative weight 3 does not clearly help versus weight 2: FV2 WEDA Fall F1=0.7385, FV5 WEDA Fall F1=0.7385.
7. FV6 fall-head unfreezing status: Skipped because safe fall-head unfreezing would require rebuilding a trainable temporal FV graph with explicit fall-logit exposure and verifying unchanged direction outputs. This frozen-probability FV script completed FV1-FV5 and records FV6 as not run instead of applying an unsafe partial unfreeze..
8. Direction remains unchanged from reference for completed FV runs because direction probabilities are copied from the frozen reference output.
9. DS-Fall-RD-FV should be reported as an optional FPGuard/verifier extension unless the paper wants to prioritize WEDA fall precision over a pure end-to-end temporal model. A5 remains the cleaner main architecture; FV is the enhanced false-positive guard.
10. Paper-facing conclusion: WEDA Fall F1 weakness is a false-positive hard-negative problem. A lightweight logistic verifier using signal summary features can substantially reduce WEDA false positives while preserving DS-Fall-RD direction behavior.

## Not Run / Failed Runs

| run_id | status | reason |
| --- | --- | --- |
| FV6_LITE10_LOGIT_FUSION_UNFREEZE_FALL | not_run | Skipped because safe fall-head unfreezing would require rebuilding a trainable temporal FV graph with explicit fall-logit exposure and verifying unchanged direction outputs. This frozen-probability FV script completed FV1-FV5 and records FV6 as not run instead of applying an unsafe partial unfreeze. |

## WEDA FP Correction Count

| run_id | reference_fp | fv_fp | corrected_reference_fp |
| --- | --- | --- | --- |
| FV1_LITE10_AND | 21.0000 | 15.0000 | 6.0000 |
| FV2_LITE10_LOGIT_FUSION | 21.0000 | 16.0000 | 5.0000 |
| FV3_CENTER12_LOGIT_FUSION | 21.0000 | 21.0000 | 0.0000 |
| FV4_LITE10_GATED_PRODUCT | 21.0000 | 22.0000 | 0.0000 |
| FV5_LITE10_LOGIT_FUSION_HN3 | 21.0000 | 16.0000 | 5.0000 |
