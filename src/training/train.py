from __future__ import annotations

import numpy as np

from src.models.losses import make_direction_loss, make_fall_loss


def compile_ds_fall_model(
    model,
    learning_rate: float = 1e-3,
    direction_loss_type: str = "ce",
    lambda_fall: float = 1.0,
    lambda_direction: float = 0.5,
    focal_gamma: float = 2.0,
    direction_class_weights: np.ndarray | list[float] | None = None,
    fall_class_weights: np.ndarray | list[float] | None = None,
):
    import tensorflow as tf

    fall_loss = make_fall_loss(class_weights=fall_class_weights)
    direction_loss = make_direction_loss(
        loss_type=direction_loss_type,
        class_weights=direction_class_weights,
        focal_gamma=focal_gamma,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "fall_output": fall_loss,
            "direction_output": direction_loss,
        },
        loss_weights={
            "fall_output": float(lambda_fall),
            "direction_output": float(lambda_direction),
        },
        metrics={
            "fall_output": ["accuracy"],
            "direction_output": ["accuracy"],
        },
    )
    return model


def train_model(model, train_ds, val_ds, callbacks, epochs: int = 80):
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
