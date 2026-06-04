# Clean BITS-WEDA Experiments

Fixed model: B0 / A5WCEFW = DS-Fall-RD, tilt12, task-specific attention, weighted CE fall, weighted CE direction.
Only BITS and WEDA are retained before split construction. HIFD and UMAFall are not used.

## Table 1: Clean BITS-WEDA Results

| experiment_id | train_datasets | test_datasets | fall_f1 | direction_macro_f1 | direction_accuracy | direction_n | params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | bits | weda | 0.5631 | 0.3425 | 0.4229 | 350 | 65959 |
| E2 | weda | bits | 0.0000 | 0.2210 | 0.4959 | 244 | 65959 |
| E3 | bits+weda | bits+weda | 0.8511 | 0.6999 | 0.7213 | 61 | 65959 |

## Table 2: Per-dataset Results for E3

| dataset | fall_f1 | direction_macro_f1 | direction_accuracy | direction_n |
| --- | --- | --- | --- | --- |
| bits | 0.8571 | 0.7529 | 0.7778 | 36 |
| weda | 0.8400 | 0.6349 | 0.6400 | 25 |

## Table 3: Direction Per-class F1

| experiment_id | test_dataset | forward_f1 | backward_f1 | lateral_f1 |
| --- | --- | --- | --- | --- |
| E1 | weda | 0.5886 | 0.0303 | 0.4085 |
| E2 | bits | 0.6630 | 0.0000 | 0.0000 |
| E3 | bits+weda | 0.7931 | 0.6400 | 0.6667 |

## Table 4: Cross-dataset Drop

| comparison | dataset_specific_direction_f1 | cross_dataset_direction_f1 | drop |
| --- | --- | --- | --- |
| bits-only vs train weda -> test bits | 0.7321 | 0.2210 | 0.5111 |
| weda-only vs train bits -> test weda | 0.7319 | 0.3425 | 0.3894 |

## Diagnosis

1. Train bits -> test weda: direction macro F1 0.3425; does not generalize well.
2. Train weda -> test bits: direction macro F1 0.2210; does not generalize well.
3. Train bits+weda then test both gives direction macro F1 0.6999; it is better than either cross-dataset direction run.
4. Easier unseen dataset in this setup: weda.
5. Direction drop when test is unseen: bits-only vs train weda -> test bits: 0.5111, weda-only vs train bits -> test weda: 0.3894; mean drop 0.4503.
6. BITS+WEDA is strong enough as a clean-direction benchmark for this model/config.
7. There is still evidence of label/domain conflict between BITS and WEDA.
