# Hybrid Architecture

Fall and direction are separated because they use different evidence.

- Fall: impact magnitude, jerk, post-impact stillness, and summary statistics.
- Direction: signed temporal gyro pattern and roll/pitch transition.

Final practical pipeline:

```text
Fall = GradientBoosting FallNoTiming_Full
Direction = DS-Fall-RD A5 direction head
```

No FullTiming features are used in the main pipeline. FullTiming is upper-bound
only.
