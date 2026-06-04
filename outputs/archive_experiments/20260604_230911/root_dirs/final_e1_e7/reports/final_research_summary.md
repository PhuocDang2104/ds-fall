# Final Research Summary

## 1. Problem Definition

The goal is direction-sensitive fall detection from wrist-worn IMU windows.
The system must decide fall/non-fall and, for supervised fall windows, classify
direction as forward, backward, or lateral.

## 2. Dataset And Benchmark

The final benchmark uses BITS and WEDA only. The protocol is fixed at 25 Hz,
2-second event-centered windows, and 50 x 12 tilt12 temporal input.

## 3. DS-Fall-RD A5 Model

A5WCEFW is the main deep temporal reference. It uses rotation-aware tilt12
features, task-specific attention, weighted fall loss, and weighted direction
loss. It remains the strongest direction expert.

## 4. WEDA Fall Gap

WEDA fall F1 is lower for the deep fall head because WEDA contains harder ADL
negatives and produces more false positives. Direction on WEDA remains good,
so the issue is mainly fall discrimination rather than direction separability.

## 5. Feature Analysis

No-timing summary features separate WEDA fall/non-fall better than the A5 fall
head in the final benchmark. Fall depends heavily on impact, jerk, post-impact
stability, and magnitude summaries. Direction depends more on signed gyro and
roll/pitch transitions.

## 6. FV/FPGuard Experiments

Previous fall-focused variants improved only part of the WEDA false-positive
problem. They did not justify replacing the stable A5 direction head.

## 7. Hybrid Experiments

Task separation is the most defensible final direction: use a no-timing ML fall
expert for fall decision and A5 direction head for direction.

## 8. Specialized ML E1-E7

The final E1-E7 ML analysis evaluates fall-only and direction-only classical
experts with fixed protocols. FullTiming is excluded from the main result.

## 9. Final Architecture Decision

Recommended practical architecture:

```text
DS-Fall-RD-Hybrid-NoTiming
= GradientBoosting FallNoTiming_Full Fall Expert
+ DS-Fall-RD A5 Direction Expert
```

A5 remains the main deep baseline and direction expert. A5 fall head is kept
as auxiliary/reference only.

## 10. Limitations

- BITS/WEDA pure cross-dataset transfer remains weaker than mixed training.
- WEDA hard negatives still need careful discussion.
- ML-only direction is useful analysis but not stable enough to replace A5.
- FullTiming is upper-bound only.

## 11. Next Work

- Validate on external datasets after axis and label harmonization.
- Study hard-negative WEDA ADL cases.
- Explore compact deployment of the A5 direction expert.
