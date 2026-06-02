# UMAFall-Only Failure Audit

Scope: processed UMAFall windows after unit harmonization. This audit does not modify preprocessing or model code.

## Basic Counts

- Total UMAFall windows: 978
- Fall windows: 208
- Non-fall windows: 770
- Supervised direction windows: 208

### Split / Label Counts

| section | split | label | count |
| --- | --- | --- | --- |
| total_windows | test | all | 192 |
| fall_label | test | non_fall | 155 |
| fall_label | test | fall | 37 |
| direction_label | test | backward | 12 |
| direction_label | test | forward | 13 |
| direction_label | test | lateral | 12 |
| direction_label | test | unsupervised | 155 |
| unique_count | test | subject_id | 3 |
| unique_count | test | trial_id | 6 |
| unique_count | test | activity_name | 15 |
| unique_count | test | activity_id | 15 |
| activity_count | test | applausing | 15 |
| activity_count | test | backward_fall | 12 |
| activity_count | test | bending | 15 |
| activity_count | test | forward_fall | 13 |
| activity_count | test | go_downstairs | 5 |
| activity_count | test | go_upstairs | 5 |
| activity_count | test | hands_up | 15 |
| activity_count | test | hopping | 15 |
| activity_count | test | jogging | 10 |
| activity_count | test | lateral_fall | 12 |
| activity_count | test | lying_down_on_a_bed | 15 |
| activity_count | test | making_a_call | 15 |
| activity_count | test | opening_door | 15 |
| activity_count | test | sitting_getting_up_on_a_chair | 15 |
| activity_count | test | walking | 15 |
| total_windows | train | all | 638 |
| fall_label | train | non_fall | 490 |
| fall_label | train | fall | 148 |
| direction_label | train | backward | 49 |
| direction_label | train | forward | 53 |
| direction_label | train | lateral | 46 |
| direction_label | train | unsupervised | 490 |
| unique_count | train | subject_id | 13 |
| unique_count | train | trial_id | 18 |
| unique_count | train | activity_name | 15 |
| unique_count | train | activity_id | 15 |
| activity_count | train | applausing | 30 |
| activity_count | train | backward_fall | 49 |
| activity_count | train | bending | 60 |
| activity_count | train | forward_fall | 53 |
| activity_count | train | go_downstairs | 25 |
| activity_count | train | go_upstairs | 25 |
| activity_count | train | hands_up | 30 |
| activity_count | train | hopping | 50 |
| activity_count | train | jogging | 30 |
| activity_count | train | lateral_fall | 46 |
| activity_count | train | lying_down_on_a_bed | 50 |
| activity_count | train | making_a_call | 30 |
| activity_count | train | opening_door | 30 |
| activity_count | train | sitting_getting_up_on_a_chair | 65 |
| activity_count | train | walking | 65 |
| total_windows | val | all | 148 |
| fall_label | val | non_fall | 125 |
| fall_label | val | fall | 23 |
| direction_label | val | backward | 12 |
| direction_label | val | forward | 5 |
| direction_label | val | lateral | 6 |
| direction_label | val | unsupervised | 125 |
| unique_count | val | subject_id | 3 |
| unique_count | val | trial_id | 6 |
| unique_count | val | activity_name | 13 |
| unique_count | val | activity_id | 13 |
| activity_count | val | applausing | 15 |
| activity_count | val | backward_fall | 12 |
| activity_count | val | bending | 10 |
| activity_count | val | forward_fall | 5 |
| activity_count | val | hands_up | 15 |
| activity_count | val | hopping | 10 |
| activity_count | val | jogging | 5 |
| activity_count | val | lateral_fall | 6 |
| activity_count | val | lying_down_on_a_bed | 10 |
| activity_count | val | making_a_call | 15 |
| activity_count | val | opening_door | 15 |
| activity_count | val | sitting_getting_up_on_a_chair | 15 |
| activity_count | val | walking | 15 |

## Label Mapping

| activity_name | activity_id | n_windows | fall_0_count | fall_1_count | direction_forward_count | direction_backward_count | direction_lateral_count | direction_unsupervised_count | split_train_count | split_val_count | split_test_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| applausing | Aplausing | 60 | 60 | 0 | 0 | 0 | 0 | 60 | 30 | 15.0000 | 15 |
| backward_fall | backwardFall | 73 | 0 | 73 | 0 | 73 | 0 | 0 | 49 | 12.0000 | 12 |
| bending | Bending | 85 | 85 | 0 | 0 | 0 | 0 | 85 | 60 | 10.0000 | 15 |
| forward_fall | forwardFall | 71 | 0 | 71 | 71 | 0 | 0 | 0 | 53 | 5.0000 | 13 |
| go_downstairs | GoDownstairs | 30 | 30 | 0 | 0 | 0 | 0 | 30 | 25 |  | 5 |
| go_upstairs | GoUpstairs | 30 | 30 | 0 | 0 | 0 | 0 | 30 | 25 |  | 5 |
| hands_up | HandsUp | 60 | 60 | 0 | 0 | 0 | 0 | 60 | 30 | 15.0000 | 15 |
| hopping | Hopping | 75 | 75 | 0 | 0 | 0 | 0 | 75 | 50 | 10.0000 | 15 |
| jogging | Jogging | 45 | 45 | 0 | 0 | 0 | 0 | 45 | 30 | 5.0000 | 10 |
| lateral_fall | lateralFall | 64 | 0 | 64 | 0 | 0 | 64 | 0 | 46 | 6.0000 | 12 |
| lying_down_on_a_bed | LyingDown_OnABed | 75 | 75 | 0 | 0 | 0 | 0 | 75 | 50 | 10.0000 | 15 |
| making_a_call | MakingACall | 60 | 60 | 0 | 0 | 0 | 0 | 60 | 30 | 15.0000 | 15 |
| opening_door | OpeningDoor | 60 | 60 | 0 | 0 | 0 | 0 | 60 | 30 | 15.0000 | 15 |
| sitting_getting_up_on_a_chair | Sitting_GettingUpOnAChair | 95 | 95 | 0 | 0 | 0 | 0 | 95 | 65 | 15.0000 | 15 |
| walking | Walking | 95 | 95 | 0 | 0 | 0 | 0 | 95 | 65 | 15.0000 | 15 |

### Label Mapping Warnings

- No obvious fall/ADL label mapping warnings found.

## Trainability Baseline

- Summary features: max_acc_mag, max_gyro_mag, max_jerk, post_acc_std, post_gyro_std

| model | split | accuracy | precision | recall | fall_f1 | confusion_matrix |
| --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | train | 0.9467 | 0.8314 | 0.9662 | 0.8938 | [[461, 29], [5, 143]] |
| logistic_regression | val | 0.9054 | 0.6452 | 0.8696 | 0.7407 | [[114, 11], [3, 20]] |
| logistic_regression | test | 0.9010 | 0.6667 | 0.9730 | 0.7912 | [[137, 18], [1, 36]] |
| random_forest | train | 0.9890 | 0.9548 | 1.0000 | 0.9769 | [[483, 7], [0, 148]] |
| random_forest | val | 0.9392 | 0.7692 | 0.8696 | 0.8163 | [[119, 6], [3, 20]] |
| random_forest | test | 0.9375 | 0.7778 | 0.9459 | 0.8537 | [[145, 10], [2, 35]] |

## DS-Fall UMAFall-Only Training Log

- Run dir: `outputs\runs\DOMAIN_DS_UMAFALL_A5WCEFW`
- Epochs logged: 80
- best_by_val_domain_score: `epoch=0.0000, loss=1.1364, fall_output_accuracy=0.7069, fall_output_loss=0.6055, direction_output_accuracy=0.0846, direction_output_loss=0.3532, val_loss=0.8692, val_fall_output_accuracy=0.8446, val_fall_output_loss=0.6758, val_direction_output_accuracy=0.7095, val_direction_output_loss=0.2416, val_fall_f1=0.0000, val_direction_macro_f1=0.3532, val_domain_score=0.3532, learning_rate=0.0010`
- best_by_val_direction_macro_f1: `epoch=0.0000, loss=1.1364, fall_output_accuracy=0.7069, fall_output_loss=0.6055, direction_output_accuracy=0.0846, direction_output_loss=0.3532, val_loss=0.8692, val_fall_output_accuracy=0.8446, val_fall_output_loss=0.6758, val_direction_output_accuracy=0.7095, val_direction_output_loss=0.2416, val_fall_f1=0.0000, val_direction_macro_f1=0.3532, val_domain_score=0.3532, learning_rate=0.0010`
- best_by_val_fall_f1: `epoch=0.0000, loss=1.1364, fall_output_accuracy=0.7069, fall_output_loss=0.6055, direction_output_accuracy=0.0846, direction_output_loss=0.3532, val_loss=0.8692, val_fall_output_accuracy=0.8446, val_fall_output_loss=0.6758, val_direction_output_accuracy=0.7095, val_direction_output_loss=0.2416, val_fall_f1=0.0000, val_direction_macro_f1=0.3532, val_domain_score=0.3532, learning_rate=0.0010`
- best_by_val_loss: `epoch=4.0000, loss=0.5224, fall_output_accuracy=0.9326, fall_output_loss=0.1593, direction_output_accuracy=0.4655, direction_output_loss=0.2421, val_loss=0.8429, val_fall_output_accuracy=0.8446, val_fall_output_loss=0.7012, val_direction_output_accuracy=0.8784, val_direction_output_loss=0.2414, val_fall_f1=0.0000, val_direction_macro_f1=0.1190, val_domain_score=0.1190, learning_rate=0.0010`
- last_epoch: `epoch=79.0000, loss=0.0105, fall_output_accuracy=1.0000, fall_output_loss=0.0011, direction_output_accuracy=0.5063, direction_output_loss=0.0062, val_loss=2.9443, val_fall_output_accuracy=0.8446, val_fall_output_loss=3.7789, val_direction_output_accuracy=0.2230, val_direction_output_loss=0.1905, val_fall_f1=0.0000, val_direction_macro_f1=0.2828, val_domain_score=0.2828, learning_rate=0.0000`

### Class Weights

```json
{
  "fall": [
    0.6510204076766968,
    2.1554055213928223
  ],
  "direction": [
    0.9308176040649414,
    1.0068026781082153,
    1.0724637508392334
  ]
}
```

## DS-Fall UMAFall-Only Fall Confusion

- TN: 155
- FP: 0
- FN: 36
- TP: 1
- Predicted fall windows: 1 / 192 (0.0052)
- Precision: 1.0000
- Recall: 0.0270
- F1: 0.0526

## Diagnosis

- UMAFall-only split has fall samples train/val/test = 148/23/37; enough for a coarse audit.
- Fall label mapping looks internally consistent from activity names: fall activities are fall_label=1 and ADL are non-fall.
- DS-Fall UMAFall-only is effectively predicting almost all windows as non-fall.
- RF fall baseline is high while DS-Fall is low: likely training config/checkpointing/model calibration issue, not split/window/label difficulty.
- Training log never reaches nonzero validation fall F1; fall head is not learning UMAFall-only validation under this setup.
- Best logged val_direction_macro_f1: 0.3532; best logged val_fall_f1: 0.0000.
- Most likely cause from this audit: not unit harmonization; prioritize training/checkpoint behavior and UMAFall-only fall separability, then manually verify UMAFall activity/window labels.
