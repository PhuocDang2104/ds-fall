# Train-2-Test-1: BITS, WEDA, HIFD

Fixed model: B0 / A5WCEFW = DS-Fall-RD, tilt12, task-specific attention, weighted CE fall, weighted CE direction.
UMAFall is excluded before split construction. Held-out test datasets are not used for scaler fitting, early stopping, or threshold tuning.

## Table 1: Train 2 Test 1 Results

| experiment_id | train_datasets | test_dataset | fall_f1 | direction_macro_f1 | direction_accuracy | direction_n | params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | bits+weda | hifd | 0.5014 | 0.3225 | 0.3942 | 104 | 65959 |
| E2 | bits+hifd | weda | 0.5882 | 0.3452 | 0.3914 | 350 | 65959 |
| E3 | weda+hifd | bits | 0.5609 | 0.5561 | 0.5820 | 244 | 65959 |

## Table 2: Compare with Dataset-Specific

| test_dataset | dataset_specific_direction_f1 | train2_test1_direction_f1 | drop |
| --- | --- | --- | --- |
| hifd | 0.7249 | 0.3225 | 0.4023 |
| weda | 0.7319 | 0.3452 | 0.3867 |
| bits | 0.7321 | 0.5561 | 0.1760 |

## Table 3: Direction Per-class F1

| experiment_id | test_dataset | forward_f1 | backward_f1 | lateral_f1 |
| --- | --- | --- | --- | --- |
| E1 | hifd | 0.4054 | 0.0571 | 0.5051 |
| E2 | weda | 0.5349 | 0.1618 | 0.3388 |
| E3 | bits | 0.6389 | 0.4037 | 0.6258 |

## Mixed 3-Dataset Baseline

No mixed BITS+WEDA+HIFD-only baseline was found in existing reports, and this task did not request running an extra mixed baseline.

## Diagnosis

1. Best unseen direction generalization: bits (train weda+hifd), direction macro F1 0.5561.
2. Hardest unseen dataset: hifd (train bits+weda), direction macro F1 0.3225.
3. Direction drop versus dataset-specific baseline: hifd: 0.4023, weda: 0.3867, bits: 0.1760; mean drop 0.3217.
4. Fall detection generalizes better than direction: yes (mean fall F1 0.5502 vs mean direction macro F1 0.4079).
5. BITS+WEDA is a plausible clean-direction setup, while HIFD should be treated as an external stress-test.
6. There is still evidence of direction label/domain conflict among BITS, WEDA, and HIFD.
