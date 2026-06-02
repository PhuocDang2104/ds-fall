# Summary 25 Hz A5WCEFW

Model: DS-Fall-RD / A5WCEFW, input shape `(50, 12)`, feature set `tilt12`, task-specific attention, weighted CE fall, weighted CE direction.

## Results

| experiment_id | train_dataset | test_dataset | sampling_rate | input_shape | n_train | n_val | n_test | fall_f1 | fall_precision | fall_recall | fall_accuracy | direction_macro_f1 | direction_accuracy | direction_n_supervised | best_epoch | best_val_loss | model_params | model_size_float32_kb | model_size_int8_kb_if_available | inference_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_BITS_TO_BITS | bits | bits | 25.0000 | (50, 12) | 2504 | 528 | 528 | 0.8889 | 0.9524 | 0.8333 | 0.9811 | 0.8777 | 0.8889 | 36.0000 | 16 | 0.4114 | 65959 | 257.6523 | 64.4131 | 96.7971 |
| E2_WEDA_TO_WEDA | weda | weda | 25.0000 | (50, 12) | 1495 | 220 | 180 | 0.7755 | 0.7917 | 0.7600 | 0.9389 | 0.6071 | 0.6400 | 25.0000 | 23 | 0.2885 | 65959 | 257.6523 | 64.4131 | 97.7580 |
| E3_BITS_WEDA_MIXED | bits+weda | bits+weda | 25.0000 | (50, 12) | 3999 | 748 | 708 | 0.8323 | 0.7614 | 0.9178 | 0.9619 | 0.8967 | 0.9016 | 61.0000 | 41 | 0.5924 | 65959 | 257.6523 | 64.4131 | 96.2534 |
| E4_BITS_TO_WEDA | bits | weda | 25.0000 | (50, 12) | 2504 | 528 | 180 | 0.5000 | 0.3433 | 0.9200 | 0.7444 | 0.3876 | 0.4000 | 25.0000 | 16 | 0.4114 | 65959 | 257.6523 | 64.4131 | 99.3935 |
| E5_WEDA_TO_BITS | weda | bits | 25.0000 | (50, 12) | 1495 | 220 | 528 | 0.7073 | 0.8529 | 0.6042 | 0.9545 | 0.6271 | 0.6667 | 36.0000 | 23 | 0.2885 | 65959 | 257.6523 | 64.4131 | 98.4613 |

## Notes

- Direction metrics use only samples with `true_direction != -1`.
- Scalers are fit on each experiment's train portion only.
- Cross-dataset tests never use the test dataset for scaler fitting or early stopping.
