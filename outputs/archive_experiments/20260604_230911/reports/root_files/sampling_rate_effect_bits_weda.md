# Sampling Rate Effect: BITS and WEDA

Fixed model: A5WCEFW / DS-Fall-RD with tilt12, task-specific attention, weighted CE fall, weighted CE direction.
HIFD and UMAFall are not used. Jerk is computed with the experiment sampling rate.

## Table 1: Main Results

| experiment_id | dataset | sampling_rate | window_length | fall_f1 | fall_precision | fall_recall | direction_macro_f1 | direction_accuracy | direction_n | params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BITS-50 | bits | 50 | 100 | 0.8602 | 0.8889 | 0.8333 | 0.7822 | 0.8056 | 36 | 65959 |
| BITS-20 | bits | 20 | 40 | 0.9091 | 1.0000 | 0.8333 | 0.8169 | 0.8333 | 36 | 65959 |
| WEDA-50 | weda | 50 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.1765 | 0.3600 | 25 | 65959 |
| WEDA-20 | weda | 20 | 40 | 0.0000 | 0.0000 | 0.0000 | 0.1765 | 0.3600 | 25 | 65959 |

## Table 2: Sampling Effect

| dataset | high_rate_config | low_rate_config | fall_f1_delta | direction_macro_f1_delta | direction_accuracy_delta |
| --- | --- | --- | --- | --- | --- |
| bits | BITS-50 | BITS-20 | -0.0489 | -0.0347 | -0.0278 |
| weda | WEDA-50 | WEDA-20 | 0.0000 | 0.0000 | 0.0000 |

## Table 3: Direction Per-Class F1

| experiment_id | forward_f1 | backward_f1 | lateral_f1 |
| --- | --- | --- | --- |
| BITS-50 | 0.8824 | 0.7143 | 0.7500 |
| BITS-20 | 0.8889 | 0.8000 | 0.7619 |
| WEDA-50 | 0.5294 | 0.0000 | 0.0000 |
| WEDA-20 | 0.5294 | 0.0000 | 0.0000 |

## Diagnosis

1. BITS 50 Hz vs 20 Hz direction macro F1: 0.7822 vs 0.8169 (high-low delta -0.0347).
2. BITS native 20 Hz is still usable for direction at macro F1 0.8169.
3. WEDA 50 Hz vs 20 Hz direction macro F1: 0.1765 vs 0.1765 (high-low delta 0.0000), but the WEDA runs are degenerate: the selected checkpoint predicts no fall samples and only one direction class. Treat this WEDA sampling comparison as inconclusive.
4. Common 20 Hz feasibility: not strong enough with minimum direction macro F1 0.1765.
5. Sampling rate is not proven to be the main cause of BITS/WEDA differences; WEDA needs a non-degenerate training run before this can be isolated cleanly.
6. For the paper, report BITS 50 Hz vs native 20 Hz as a valid sampling ablation, and rerun/fix WEDA checkpointing before making a 50 Hz vs 20 Hz claim for WEDA.
