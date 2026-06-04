# WEDA Error Analysis

Hard-negative mining was kept as analysis unless explicitly run. The targeted runner saves WEDA predictions for every run, so false negatives and false positives can be inspected from each run folder.

Best current WEDA prediction file: `C:\Users\ADMIN\Desktop\ds-fall\outputs\reports\targeted_experiments_debug\runs\B0_BASE_A5WCEFW_seed42\E7_predictions.csv`

If a second-stage hard-negative run is needed, use WEDA false negatives and high-probability false positives from this file. It was not selected automatically unless it passes E3/E6 direction stability filters.
