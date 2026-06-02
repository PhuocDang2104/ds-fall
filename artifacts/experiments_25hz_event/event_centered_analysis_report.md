# Event-Centered 25 Hz A5WCEFW Analysis

## Why The Previous Full-Trial 25 Hz Run Was Lower

The full-trial 25 Hz runner labeled every sliding window inside a BITS fall trial as fall/direction. That created many weakly labeled fall windows that do not necessarily contain the impact or the direction-discriminative rotation segment.

| setup | bits windows | bits fall windows | bits supervised direction | bits test direction n | bits fall F1 | bits direction macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full-trial 25 Hz | 61531 | 21775 | 15451 | 1635 | 0.7214 | 0.6154 |
| event-centered 25 Hz | 3560 | 325 | 244 | 36 | 0.8889 | 0.8777 |

The event-centered setup matches the earlier good BITS sampling-rate experiment: one fall window centered around impact plus capped ADL windows. This makes fall and direction labels much cleaner.

## Event-Centered Main Results

Model: DS-Fall-RD / A5WCEFW, input `(50, 12)`, sampling rate 25 Hz, feature set `tilt12`, weighted CE fall, weighted CE direction, checkpoint monitor `val_domain_score = val_fall_f1 + val_direction_macro_f1`.

| experiment | train | test | fall F1 | direction macro F1 | direction accuracy | direction n |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| E1_BITS_TO_BITS | bits | bits | 0.8889 | 0.8777 | 0.8889 | 36 |
| E2_WEDA_TO_WEDA | weda | weda | 0.7755 | 0.6071 | 0.6400 | 25 |
| E3_BITS_WEDA_MIXED | bits+weda | bits+weda | 0.8323 | 0.8967 | 0.9016 | 61 |
| E4_BITS_TO_WEDA | bits | weda | 0.5000 | 0.3876 | 0.4000 | 25 |
| E5_WEDA_TO_BITS | weda | bits | 0.7073 | 0.6271 | 0.6667 | 36 |

## Mixed Test Per Dataset

| dataset | fall F1 | direction macro F1 | direction accuracy | direction n |
| --- | ---: | ---: | ---: | ---: |
| bits | 0.9451 | 0.9350 | 0.9444 | 36 |
| weda | 0.6857 | 0.8390 | 0.8400 | 25 |

## Diagnosis

1. BITS is not inherently weak at 25 Hz. With clean event-centered windows, BITS direction macro F1 is 0.8777, better than the earlier BITS-20 result of 0.8169.
2. The previous low BITS full-trial result was mainly a window-label noise problem, not a model limitation.
3. WEDA-only improves compared with the degenerate sampling-rate run, but direction still has only 25 supervised test samples, so macro F1 is unstable.
4. Mixed BITS+WEDA performs best overall in this event-centered setup. It gives strong direction macro F1 overall and strong per-dataset direction results.
5. Cross-dataset transfer remains weak, especially BITS -> WEDA direction. This still suggests domain/label/device/protocol mismatch.
6. For the paper, use event-centered 25 Hz as the clean benchmark and report full-trial 25 Hz as an ablation showing that loose fall-trial labels inject noise.

## Output Files

- `artifacts/experiments_25hz_event/summary_25hz_a5wcefw.csv`
- `artifacts/experiments_25hz_event/summary_25hz_a5wcefw.md`
- `artifacts/experiments_25hz_event/processing_report_25hz.md`
- `artifacts/experiments_25hz_event/dataset_audit_25hz.csv`
- `artifacts/experiments_25hz_event/E*/metrics.json`
- `artifacts/experiments_25hz_event/E*/predictions.csv`
