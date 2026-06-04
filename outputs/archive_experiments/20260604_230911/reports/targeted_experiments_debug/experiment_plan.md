# Targeted Experiment Plan

- Scope: BITS and WEDA only.
- Windowing: event-centered 2 seconds, 25 Hz, 50 timesteps.
- Main model: DS-Fall-RD A5WCEFW.
- Priority: threshold tuning, fall loss tuning, dataset-balanced sampler, feature ablation, compactness.
- Seeds: [42]
- Epochs: 1, patience: 1, suite: core
- Optional hard-negative mining and impact-position sweeps are documented, not forced into the main model unless safe and useful.
