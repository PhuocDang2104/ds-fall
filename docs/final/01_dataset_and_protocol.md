# Dataset And Protocol

## Objective

DS-Fall-RD studies direction-sensitive fall detection from wrist IMU signals.
The final benchmark is restricted to BITS and WEDA.

## Benchmark

- Datasets: BITS, WEDA.
- Sampling rate: 25 Hz.
- Window: 2 seconds, event-centered.
- Temporal input: 50 x 12 tilt12.
- Direction classes: forward, backward, lateral.
- Direction metrics are computed only on supervised fall windows.

## E1-E7 Protocol

| ID | Train | Validation | Test |
| --- | --- | --- | --- |
| E1_BITS_TO_BITS | BITS train | BITS val | BITS test |
| E2_WEDA_TO_WEDA | WEDA train | WEDA val | WEDA test |
| E3_BITS_WEDA_MIXED | BITS+WEDA train | BITS+WEDA val | BITS+WEDA test |
| E4_BITS_TO_WEDA | BITS train | BITS val | WEDA test |
| E5_WEDA_TO_BITS | WEDA train | WEDA val | BITS test |
| E6_E3_MIXED_TEST_BITS | reuse E3 | reuse E3 | BITS test |
| E7_E3_MIXED_TEST_WEDA | reuse E3 | reuse E3 | WEDA test |

FullTiming features are treated only as an upper bound. They are not used as
the main result because they can encode event-centered timing cues.
