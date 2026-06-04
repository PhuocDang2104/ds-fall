# FP Reduction Experiments Summary

Scope: BITS/WEDA only, 25 Hz, 2-second event-centered windows, tilt12. No new dataset and no large architecture change.

## Block A: Threshold / Decision Analysis

| run_id | eval | dataset | threshold | TP | FP | FN | TN | fall_precision | fall_recall | fall_f1 | direction_macro_f1 | direction_n_supervised | weda_fp_reduction_vs_ref | weda_precision_gain_vs_ref | weda_recall_delta_vs_ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | E3 | bits+weda | 0.7616 | 67 | 18 | 6 | 617 | 0.7882 | 0.9178 | 0.8481 | 0.8967 | 61 |  |  |  |
| T1 | E6 | bits | 0.7616 | 43 | 0 | 5 | 480 | 1.0000 | 0.8958 | 0.9451 | 0.9350 | 36 |  |  |  |
| T1 | E7 | weda | 0.7616 | 24 | 18 | 1 | 137 | 0.5714 | 0.9600 | 0.7164 | 0.8390 | 25 | 3.0000 | 0.0381 | 0.0000 |
| T2 | E3 | bits+weda | 0.7616 | 67 | 18 | 6 | 617 | 0.7882 | 0.9178 | 0.8481 | 0.8967 | 61 |  |  |  |
| T2 | E6 | bits | 0.7616 | 43 | 0 | 5 | 480 | 1.0000 | 0.8958 | 0.9451 | 0.9350 | 36 |  |  |  |
| T2 | E7 | weda | 0.7616 | 24 | 18 | 1 | 137 | 0.5714 | 0.9600 | 0.7164 | 0.8390 | 25 | 3.0000 | 0.0381 | 0.0000 |
| T3 | E3 | bits+weda | 0.7616 | 67 | 18 | 6 | 617 | 0.7882 | 0.9178 | 0.8481 | 0.8967 | 61 |  |  |  |
| T3 | E6 | bits | 0.7616 | 43 | 0 | 5 | 480 | 1.0000 | 0.8958 | 0.9451 | 0.9350 | 36 |  |  |  |
| T3 | E7 | weda | 0.7616 | 24 | 18 | 1 | 137 | 0.5714 | 0.9600 | 0.7164 | 0.8390 | 25 | 3.0000 | 0.0381 | 0.0000 |
| T4 | E3 | bits+weda | 0.1098 | 68 | 26 | 5 | 609 | 0.7234 | 0.9315 | 0.8144 | 0.8967 | 61 |  |  |  |
| T4 | E6 | bits | 0.1098 | 43 | 3 | 5 | 477 | 0.9348 | 0.8958 | 0.9149 | 0.9350 | 36 |  |  |  |
| T4 | E7 | weda | 0.1098 | 25 | 23 | 0 | 132 | 0.5208 | 1.0000 | 0.6849 | 0.8390 | 25 | -2.0000 | -0.0125 | 0.0400 |

## Block B: Hard-negative Training

| run_id | eval | dataset | threshold | TP | FP | FN | TN | fall_precision | fall_recall | fall_f1 | direction_macro_f1 | direction_n_supervised | params | weda_fp_reduction_vs_ref | weda_precision_gain_vs_ref | weda_recall_delta_vs_ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HN1 | E3 | bits+weda | 0.9606 | 62 | 10 | 11 | 625 | 0.8611 | 0.8493 | 0.8552 | 0.7310 | 61 | 65959 |  |  |  |
| HN1 | E6 | bits | 0.9606 | 41 | 0 | 7 | 480 | 1.0000 | 0.8542 | 0.9213 | 0.8704 | 36 | 65959 |  |  |  |
| HN1 | E7 | weda | 0.9606 | 21 | 10 | 4 | 145 | 0.6774 | 0.8400 | 0.7500 | 0.5499 | 25 | 65959 | 11.0000 | 0.1441 | -0.1200 |
| HN2 | E3 | bits+weda | 0.8157 | 63 | 14 | 10 | 621 | 0.8182 | 0.8630 | 0.8400 | 0.6698 | 61 | 65959 |  |  |  |
| HN2 | E6 | bits | 0.8157 | 41 | 0 | 7 | 480 | 1.0000 | 0.8542 | 0.9213 | 0.9112 | 36 | 65959 |  |  |  |
| HN2 | E7 | weda | 0.8157 | 22 | 14 | 3 | 141 | 0.6111 | 0.8800 | 0.7213 | 0.3430 | 25 | 65959 | 7.0000 | 0.0778 | -0.0800 |
| HN3 | E3 | bits+weda | 0.6702 | 67 | 21 | 6 | 614 | 0.7614 | 0.9178 | 0.8323 | 0.8967 | 61 | 65959 |  |  |  |
| HN3 | E6 | bits | 0.6702 | 43 | 0 | 5 | 480 | 1.0000 | 0.8958 | 0.9451 | 0.9350 | 36 | 65959 |  |  |  |
| HN3 | E7 | weda | 0.6702 | 24 | 21 | 1 | 134 | 0.5333 | 0.9600 | 0.6857 | 0.8390 | 25 | 65959 | 0.0000 | 0.0000 | 0.0000 |
| HN4 | E3 | bits+weda | 0.7796 | 67 | 19 | 6 | 616 | 0.7791 | 0.9178 | 0.8428 | 0.8967 | 61 | 65959 |  |  |  |
| HN4 | E6 | bits | 0.7796 | 43 | 0 | 5 | 480 | 1.0000 | 0.8958 | 0.9451 | 0.9350 | 36 | 65959 |  |  |  |
| HN4 | E7 | weda | 0.7796 | 24 | 19 | 1 | 136 | 0.5581 | 0.9600 | 0.7059 | 0.8390 | 25 | 65959 | 2.0000 | 0.0248 | 0.0000 |

## Per-class Direction F1

| run_id | eval | dataset | forward_f1 | backward_f1 | lateral_f1 |
| --- | --- | --- | --- | --- | --- |
| HN1 | E3 | bits+weda | 0.8214 | 0.6400 | 0.7317 |
| HN1 | E6 | bits | 0.9444 | 0.8333 | 0.8333 |
| HN1 | E7 | weda | 0.6000 | 0.4615 | 0.5882 |
| HN2 | E3 | bits+weda | 0.7451 | 0.5833 | 0.6809 |
| HN2 | E6 | bits | 0.9444 | 0.9091 | 0.8800 |
| HN2 | E7 | weda | 0.2667 | 0.3077 | 0.4545 |
| HN3 | E3 | bits+weda | 0.9286 | 0.8966 | 0.8649 |
| HN3 | E6 | bits | 0.9730 | 0.9231 | 0.9091 |
| HN3 | E7 | weda | 0.8421 | 0.8750 | 0.8000 |
| HN4 | E3 | bits+weda | 0.9286 | 0.8966 | 0.8649 |
| HN4 | E6 | bits | 0.9730 | 0.9231 | 0.9091 |
| HN4 | E7 | weda | 0.8421 | 0.8750 | 0.8000 |

## Answers

1. Threshold-only can reduce WEDA FP to 18 with precision 0.5714, recall 0.9600, Fall F1 0.7164. Best rule by FP is T1.
2. The best T1-T4 rule is the row selected above by lowest WEDA FP, with Fall F1 as tie-breaker. Threshold rules passing hard filters: T1, T2, T3, T4.
3. Best hard-negative run by WEDA FP is HN1 (hard_negative_weighted_ce, feature_rule): FP=10, precision=0.6774, recall=0.8400, Fall F1=0.7500. It reduces FP more than threshold-only, but check recall/direction hard filters before treating it as a replacement.
4. HN1-HN4 comparison is in `hard_negative_runs.csv`; the best row above is selected by WEDA FP first, then WEDA Fall F1.
5. HN runs passing hard filters: HN3, HN4. No HN run reaches both the hard filters and the desired FP target. Keep `REF_CURRENT_E3_ARTIFACT` as the paper main model; report T1/HN1/HN4 as FP-aware ablations.
6. If no run passes hard filters, keep `REF_CURRENT_E3_ARTIFACT` and describe hard-negative training as an FP-aware ablation.
7. Paper-facing: WEDA fall weakness is a hard-negative ADL precision problem; FP-aware thresholding/training quantifies how far precision can be improved before recall/direction trade-offs appear.
