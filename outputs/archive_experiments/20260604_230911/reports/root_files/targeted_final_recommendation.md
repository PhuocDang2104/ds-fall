# Final Recommendation

Recommended paper main model: `REF_CURRENT_E3_ARTIFACT` / DS-Fall-RD A5WCEFW, `tilt12`, task-specific attention, weighted CE fall, weighted CE direction.

## Direct Answers

- Did WEDA Fall F1 improve? Yes in ablations, but not under the hard-filtered main-model rule. L3 raises E7 WEDA Fall F1 to 0.8148 versus reference 0.6857, but direction drops too much.
- Did E3 Direction Macro F1 remain stable? Only the reference remains clearly stable at 0.8967. S3 reaches 0.8540, just below the 0.86 hard filter. L3 reaches 0.8298.
- Did E7 Direction Macro F1 remain above 0.80? Reference, L3 seed 42, S1/S3/S4 seed 42 do. Several fall-focused/compact runs do not.
- Which loss setting worked best? L3 is best for WEDA Fall F1, but it is a fall-focused ablation rather than the final model.
- Did dataset-balanced sampler help? It helped direction stability more than WEDA Fall F1. S3 is the best sampler trade-off.
- Is threshold tuning enough? For newly trained runs, threshold tuning can lift WEDA Fall F1, but it does not solve the direction trade-off. Existing checkpoint threshold tuning could not be evaluated because the old Lambda-based Keras artifact is not loadable in the current runtime.
- Which feature set is most compact without hurting performance? None beat `tilt12` under the selection criteria. Keep `tilt12`.
- Can width 0.75 replace width 1.0? No. W075 loses too much Fall/Direction performance.
- Did smaller dense/head layers reduce overfitting? No clear benefit; D32_16 does not preserve direction.
- Should adapters or hard negative mining be kept? Adapters were not run in the core suite. Hard-negative mining was not selected because no stable pre-HN candidate passed the main filters; keep it as future analysis only.
- Impact alignment? Not changed in the main suite. The current event-centered 50% impact-centered window is preserved to avoid changing the benchmark while tuning loss/sampler/features.

## Final Paper Recommendation

Use the current reference DS-Fall-RD as the main model. Report L3 as a fall-prioritized loss ablation showing that WEDA Fall F1 can reach 0.8148, but with a direction trade-off. Report S3 as the best direction-preserving sampler ablation.

## Tables/Figures To Use

- Main table: `outputs/reports/targeted_experiments/all_runs.csv`
- Selection: `outputs/reports/targeted_experiments/best_model_selection.md`
- Feature ablation: `outputs/reports/targeted_experiments/feature_ablation.csv`
- Compactness: `outputs/reports/targeted_experiments/compactness_results.csv`
- Confusion matrices: `outputs/figures/targeted_experiments/confusion_matrices/`
