from __future__ import annotations

import numpy as np

from src.models.losses import make_sample_weights, prepare_direction_targets


def make_keras_arrays(
    X: np.ndarray,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    y_dict = {
        "fall_output": np.asarray(y_fall, dtype=np.int64),
        "direction_output": prepare_direction_targets(y_direction),
    }
    sw_dict = make_sample_weights(y_fall, direction_mask)
    return np.asarray(X, dtype=np.float32), y_dict, sw_dict


def make_tf_dataset(
    X: np.ndarray,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int = 42,
):
    import tensorflow as tf

    X, y_dict, sw_dict = make_keras_arrays(X, y_fall, y_direction, direction_mask)
    ds = tf.data.Dataset.from_tensor_slices((X, y_dict, sw_dict))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
