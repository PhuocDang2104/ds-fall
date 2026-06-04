# DS-Fall-RD Targeted Experiments Summary

Scope: BITS/WEDA only, event-centered 2s windows, 25 Hz, no new datasets and no heavy architecture.

## Best Observed Run

- run_id: `B0_BASE_A5WCEFW`
- E7 WEDA Fall F1: 0.5376
- E3 Direction Macro F1: 0.1446
- E7 Direction Macro F1: 0.1876
- E6 BITS Fall F1: 0.4815
- params: 65959

## Main Results

| run_id | group_id | config_id | feature_set | sampler | fall_loss | alpha_fall | lambda_dir | fall_threshold | fall_f1 | direction_macro_f1 | bits_fall_f1 | bits_direction_macro_f1 | weda_fall_f1 | weda_direction_macro_f1 | params | selection_score | passes_hard_filter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0_BASE_A5WCEFW | baseline | B0 | tilt12 | current | weighted_ce | 1.0000 | 1.5000 | 0.3200 | 0.5020 | 0.1446 | 0.4815 | 0.1131 | 0.5376 | 0.1876 | 65959 | 0.3574 | 0.0000 |

## Key Diagnostics

- Selection prioritizes E7 WEDA Fall F1, then direction stability and E6 BITS Fall F1.
- Dataset-aware thresholds are saved as analysis only; the selected rows use a global validation-tuned threshold.
- Direction metrics always use supervised direction samples only.
