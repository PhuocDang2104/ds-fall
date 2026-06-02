# Summary 25 Hz A5WCEFW

Model: DS-Fall-RD / A5WCEFW, input shape `(50, 12)`, feature set `tilt12`, task-specific attention, weighted CE fall, weighted CE direction.

## Results

| experiment_id | train_dataset | test_dataset | sampling_rate | input_shape | n_train | n_val | n_test | fall_f1 | fall_precision | fall_recall | fall_accuracy | direction_macro_f1 | direction_accuracy | direction_n_supervised | best_epoch | best_val_loss | model_params | model_size_float32_kb | model_size_int8_kb_if_available | inference_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1_BITS_TO_BITS | bits | bits | 25.0000 | (50, 12) | 45398 | 8427 | 7706 | 0.7214 | 0.6197 | 0.8630 | 0.7885 | 0.6154 | 0.6165 | 1635.0000 | 1 | 1.2431 | 65959 | 257.6523 | 64.4131 | 76.1857 |
| E2_WEDA_TO_WEDA | weda | weda | 25.0000 | (50, 12) | 18016 | 2441 | 1587 | 0.7143 | 0.6081 | 0.8654 | 0.9319 | 0.7269 | 0.7372 | 156.0000 | 13 | 0.2259 | 65959 | 257.6523 | 64.4131 | 91.1920 |
| E3_BITS_WEDA_MIXED | bits+weda | bits+weda | 25.0000 | (50, 12) | 63414 | 10868 | 9293 | 0.7113 | 0.6080 | 0.8570 | 0.8053 | 0.6215 | 0.6214 | 1791.0000 | 1 | 1.0682 | 65959 | 257.6523 | 64.4131 | 75.1564 |
| E4_BITS_TO_WEDA | bits | weda | 25.0000 | (50, 12) | 45398 | 8427 | 1587 | 0.3289 | 0.2027 | 0.8718 | 0.6503 | 0.4868 | 0.5256 | 156.0000 | 1 | 1.2431 | 65959 | 257.6523 | 64.4131 | 98.8504 |
| E5_WEDA_TO_BITS | weda | bits | 25.0000 | (50, 12) | 18016 | 2441 | 7706 | 0.1519 | 0.2026 | 0.1215 | 0.5696 | 0.5901 | 0.6508 | 1635.0000 | 13 | 0.2259 | 65959 | 257.6523 | 64.4131 | 100.8265 |

## Notes

- Direction metrics use only samples with `true_direction != -1`.
- Scalers are fit on each experiment's train portion only.
- Cross-dataset tests never use the test dataset for scaler fitting or early stopping.
