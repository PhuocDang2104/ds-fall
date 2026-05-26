from __future__ import annotations

import numpy as np


def prepare_direction_targets(y_direction: np.ndarray) -> np.ndarray:
    """Replace -1 unsupervised labels with 0 before passing to Keras."""
    y = np.asarray(y_direction, dtype=np.int64).copy()
    y[y < 0] = 0
    return y


def make_sample_weights(y_fall: np.ndarray, direction_mask: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "fall_output": np.ones_like(y_fall, dtype=np.float32),
        "direction_output": np.asarray(direction_mask, dtype=np.float32),
    }
