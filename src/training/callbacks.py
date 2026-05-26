from __future__ import annotations

from pathlib import Path

from src.utils.io import ensure_dir


def build_callbacks(
    models_dir: str | Path,
    logs_dir: str | Path,
    monitor: str = "val_fall_output_accuracy",
    patience: int = 10,
):
    import tensorflow as tf

    models_dir = ensure_dir(models_dir)
    logs_dir = ensure_dir(logs_dir)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(models_dir / "ds_fall_best.keras"),
            monitor=monitor,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            patience=5,
            factor=0.5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(logs_dir / "training_log.csv")),
    ]
