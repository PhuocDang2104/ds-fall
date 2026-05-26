from __future__ import annotations


def compile_ds_fall_model(model, learning_rate: float = 1e-3):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "fall_output": "sparse_categorical_crossentropy",
            "direction_output": "sparse_categorical_crossentropy",
        },
        loss_weights={
            "fall_output": 1.0,
            "direction_output": 0.5,
        },
        metrics={
            "fall_output": ["accuracy"],
            "direction_output": ["accuracy"],
        },
    )
    return model


def train_model(model, train_ds, val_ds, callbacks, epochs: int = 80):
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
