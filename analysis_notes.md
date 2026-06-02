# Analysis Notes

<!-- AUTO_CROSS_DATASET_ANALYSIS_START -->
## Automatic Cross-Dataset Analysis

This section is generated from cross-dataset result CSV files. It is a hypothesis list for debugging, not a confirmed causal analysis.

Low-performance rule: `direction_macro_f1 < 0.60` or `fall_f1 < 0.70` on cross-dataset rows.

Status: low cross-dataset performance detected.

Low rows:

| experiment_id | experiment_type | train | test | fall_f1 | direction_macro_f1 | direction_accuracy | direction_n | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T3_BITS | train3_test1 | weda+hifd+umafall | bits | 0.6690 | 0.4280 | 0.4467 | 244.0000 | outputs/reports/domain_conflict_experiments.csv |
| T3_WEDA | train3_test1 | bits+hifd+umafall | weda | 0.6895 | 0.4310 | 0.4457 | 350.0000 | outputs/reports/domain_conflict_experiments.csv |
| T3_HIFD | train3_test1 | bits+weda+umafall | hifd | 0.3296 | 0.3992 | 0.4615 | 104.0000 | outputs/reports/domain_conflict_experiments.csv |
| T3_UMAFALL | train3_test1 | bits+weda+hifd | umafall | 0.7108 | 0.3685 | 0.3702 | 208.0000 | outputs/reports/domain_conflict_experiments.csv |
| E1 | train2_test1_bits_weda_hifd | bits+weda | hifd | 0.5014 | 0.3225 | 0.3942 | 104.0000 | outputs/reports/train2_test1_bits_weda_hifd.csv |
| E2 | train2_test1_bits_weda_hifd | bits+hifd | weda | 0.5882 | 0.3452 | 0.3914 | 350.0000 | outputs/reports/train2_test1_bits_weda_hifd.csv |
| E3 | train2_test1_bits_weda_hifd | weda+hifd | bits | 0.5609 | 0.5561 | 0.5820 | 244.0000 | outputs/reports/train2_test1_bits_weda_hifd.csv |
| E1 | clean_bits_weda | bits | weda | 0.5631 | 0.3425 | 0.4229 | 350.0000 | outputs/reports/clean_bits_weda_experiments.csv |
| E2 | clean_bits_weda | weda | bits | 0.0000 | 0.2210 | 0.4959 | 244.0000 | outputs/reports/clean_bits_weda_experiments.csv |

Possible causes:

1. Sampling protocol mismatch.
2. Sensor placement / device mismatch.
3. Activity definition mismatch.
4. Direction label mismatch.
5. Subject movement style mismatch.
6. Fall impact interval annotation mismatch.
7. BITS 20 Hz interpolation to 25 Hz may smooth impact peaks.
8. WEDA 50 Hz downsampling to 25 Hz may reduce high-frequency impact detail.

Recommended checks:

- Verify resampling is done per trial and preserves acc/gyro alignment.
- Compare per-dataset axis semantics using signed gyro, roll, pitch, and impact-centered curves.
- Recheck direction label definitions and whether lateral merges left/right patterns.
- Inspect impact index and fall interval annotation consistency before changing the model.
<!-- AUTO_CROSS_DATASET_ANALYSIS_END -->
