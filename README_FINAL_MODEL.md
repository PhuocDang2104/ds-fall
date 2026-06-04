# DS-Fall-RD Compact Hybrid

Final paper-ready pipeline for BITS/WEDA 25 Hz event-centered 2-second wrist IMU windows.

## Inputs

- Fall expert: event-window statistical feature vector from the 2-second IMU window.
- Direction expert: temporal `tilt12` sequence with shape `50 x 12`.

The fall feature vector excludes explicit timing-index artifacts such as `impact_index`, `peak_index`, and distance-to-center variables. Peak magnitude values are allowed because they are signal values.

## Model

- Statistical Fall Expert: `GradientBoostingClassifier`.
- LDX1 Wide-DSConv Direction Expert: lightweight temporal depthwise-separable CNN.
- Inference: if the fall expert predicts non-fall, output non-fall; otherwise output fall with direction predicted by LDX1.

## Reproduce

```bash
python scripts/run_final_compact_hybrid.py --repo-root . --seed-search --build-final --selection-mode paper_safe
python scripts/compare_final_with_baselines.py --repo-root . --run-all
python scripts/evaluate_final_compact_hybrid.py --repo-root .
```

Outputs are written to `outputs/reports/final_compact_hybrid`, `outputs/models/final_compact_hybrid`, and `outputs/figures/final_compact_hybrid`.
