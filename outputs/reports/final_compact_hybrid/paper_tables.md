# Final Paper Readiness Report

## Final Model

- Name: DS-Fall-RD Compact Hybrid.
- Fall: Statistical Fall Expert, GradientBoostingClassifier, event-window statistical features excluding timing-index artifacts.
- Direction: LDX1 Wide-DSConv Direction Expert, temporal tilt12 input `50 x 12`.

## Questions

1. Selected seed/lr: seed `12`, learning rate `0.0005`, mode `paper_safe`.
2. Reason: highest validation direction macro F1, with val loss/gap/epoch tie-breakers.
3. Final direction metrics are in Table 3 below and `final_compact_hybrid_direction_metrics.csv`.
4. Final E2E metrics are in Table 4 below and `final_compact_hybrid_e2e_metrics.csv`.
5. Comparisons include classical ML-only baselines, historical A5 hybrid reference, and final compact hybrid.
6. Direction params: `5699`, estimated INT8 `5.57 KB`.
7. Direction parameter reduction vs historical A5: `91.36%`.
8. Paper safety: if selection mode is `paper_safe`, seed is validation-selected. If `best_observed`, mark as optimistic analysis only.
9. Limitations: BITS/WEDA only, TensorFlow/TFLite unavailable in current environment if logged, and seed sensitivity should be disclosed.
10. Repo readiness: final scripts, model card, tables, configs, and artifacts are saved under `final_compact_hybrid`; inspect failed runs before submission.
11. Supplementary files: seed search table, selected seed JSON, final config, model card, paper tables, class counts, and confusion matrices.
12. A5 Direction-Only and LDX1 multi-seed context is included below when prior reports are available.

## Baseline Selection

| best_fall_baseline | best_direction_baseline | selection_note |
| --- | --- | --- |
| HistGradientBoosting | HistGradientBoosting | selected by E3 test for compact paper table analysis |

## Table 1: Dataset and Window Statistics

| dataset | sampling_rate_hz | window_seconds | input_shape | n_windows | n_fall | n_nonfall | n_supervised_direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bits | 25 | 2 | 50 x 12 | 3560 | 325 | 3235 | 244 |
| weda | 25 | 2 | 50 x 12 | 1895 | 350 | 1545 | 350 |

## Table 2: Fall Detection Comparison

| Method | Feature/Input | E3 Fall F1 | E6 Fall F1 | E7 Fall F1 | E7 Precision | E7 Recall | E7_FP | Params/Complexity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical ML Fall - GradientBoosting | event-window statistical features | 0.8732 | 0.8966 | 0.8364 | 0.7667 | 0.9200 | 7 | trees=100, nodes=1408 |
| Classical ML Fall - HistGradientBoosting | event-window statistical features | 0.8933 | 0.9451 | 0.8136 | 0.7059 | 0.9600 | 10 | hist_gb_predictors=140 |
| Classical ML Fall - LogisticRegression | event-window statistical features | 0.8092 | 0.8235 | 0.7826 | 0.8571 | 0.7200 | 3 | LogisticRegression |
| Classical ML Fall - RandomForest | event-window statistical features | 0.8366 | 0.9111 | 0.7302 | 0.6053 | 0.9200 | 15 | trees=300, nodes=63290 |
| DS-Fall-RD Compact Hybrid | event-window statistical features | 0.8732 | 0.8966 | 0.8364 | 0.7667 | 0.9200 | 7 | trees=100, nodes=1408 |

## Table 3: Direction Classification Comparison

| Method | Input | Params | E3 Direction F1 | E6 Direction F1 | E7 Direction F1 | Avg_E3_E6_E7 | INT8 KB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Classical ML Direction - GradientBoosting | event-window statistical features | 0 | 0.8577 | 0.8321 | 0.8851 | 0.8583 | 0.0000 |
| Classical ML Direction - HistGradientBoosting | event-window statistical features | 0 | 0.8797 | 0.8733 | 0.8772 | 0.8767 | 0.0000 |
| Classical ML Direction - LogisticRegression | event-window statistical features | 0 | 0.7962 | 0.7666 | 0.8026 | 0.7885 | 0.0000 |
| Classical ML Direction - RandomForest | event-window statistical features | 0 | 0.8096 | 0.8460 | 0.7417 | 0.7991 | 0.0000 |
| DS-Fall-RD A5 Hybrid Reference | historical A5 direction head | 65959 | 0.8967 | 0.9350 | 0.8390 | 0.8902 | 64.4131 |
| DS-Fall-RD Compact Hybrid | temporal tilt12 sequence 50x12 | 5699 | 0.8685 | 0.8579 | 0.8778 | 0.8681 | 5.5654 |

## Table 4: End-to-End Hybrid Comparison

| Method | Fall Expert | Direction Expert | E3 E2E F1 | E6 E2E F1 | E7 E2E F1 | Avg E2E F1 | Edge suitability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DS-Fall-RD A5 Hybrid Reference | Statistical Fall Expert | historical multitask A5 direction head | 0.7998 | 0.7798 | 0.7926 | 0.7908 | medium |
| DS-Fall-RD Compact Hybrid | Statistical Fall Expert | LDX1 Wide-DSConv | 0.7641 | 0.6857 | 0.8299 | 0.7599 | high |
| Statistical-ML Baseline | Best classical ML fall baseline | Best classical ML direction baseline | 0.8294 | 0.8045 | 0.8552 | 0.8297 | medium |

## Table 5: Final Compact Model vs Main Reference

| Model | Fall feature type | Direction model | Direction params | Direction param reduction | Fall E7 F1 | Direction avg F1 | E2E avg F1 | Limitation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DS-Fall-RD Compact Hybrid | event-window statistical features | LDX1 Wide-DSConv | 5699 | 0.9136 | 0.8364 | 0.8681 | 0.7599 | paper-safe seed selected by validation; compact direction is below historical A5 if so observed |
| DS-Fall-RD A5 Hybrid Reference | event-window statistical features | historical multitask A5 direction head | 65959 | 0.0000 | 0.8364 | 0.8902 | 0.7908 | larger temporal direction expert |

## LDX1 Multi-Seed Summary

Full final seed-search summary for the compact direction expert:

| learning_rate | n_runs | mean_E3_direction_f1 | mean_E6_direction_f1 | mean_E7_direction_f1 | mean_score | std_score | best_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0010 | 26 | 0.8361 | 0.8406 | 0.8204 | 0.8328 | 0.0515 | 0.9202 |
| 0.0005 | 26 | 0.8360 | 0.8359 | 0.8245 | 0.8325 | 0.0347 | 0.9009 |

Best-observed seed is analysis-only because it is ranked with E3/E6/E7 test metrics:

| seed | learning_rate | best_val_macro_f1 | E3_direction_macro_f1 | E6_direction_macro_f1 | E7_direction_macro_f1 | score_best_observed | params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 0.0010 | 0.9313 | 0.9153 | 0.8902 | 0.9568 | 0.9202 | 5699 |

## Historical Direction-Only Context

Prior A5 Direction-Only and LDX1 multi-seed runs are kept as context, not as the final selected paper-safe model:

| model_name | learning_rate | params | mean_avg_e3_e6_e7_f1 | std_avg_e3_e6_e7_f1 | mean_e3_f1 | mean_e6_f1 | mean_e7_f1 | estimated_int8_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LDX1_D1_Wide_DSConv | 0.0010 | 5699 | 0.8285 | 0.0482 | 0.8303 | 0.8381 | 0.8171 | 5.5654 |
| LDX1_D1_Wide_DSConv | 0.0005 | 5699 | 0.8263 | 0.0255 | 0.8275 | 0.8096 | 0.8417 | 5.5654 |
| A5_Direction_Only | 0.0005 | 57268 | 0.7716 | 0.0378 | 0.7815 | 0.8159 | 0.7174 | 55.9258 |
| A5_Direction_Only | 0.0010 | 57268 | 0.7499 | 0.0932 | 0.7638 | 0.8318 | 0.6541 | 55.9258 |
