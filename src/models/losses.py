from __future__ import annotations

import numpy as np

_WEIGHTED_SPARSE_CE_CLASS = None
_SPARSE_FOCAL_CLASS = None


def prepare_direction_targets(y_direction: np.ndarray) -> np.ndarray:
    """Replace -1 unsupervised labels with 0 before passing to Keras."""
    y = np.asarray(y_direction, dtype=np.int64).copy()
    y[y < 0] = 0
    return y


def make_sample_weights(y_fall: np.ndarray, direction_mask: np.ndarray) -> dict[str, np.ndarray]:
    supervised_direction = np.asarray(direction_mask, dtype=np.float32) * (np.asarray(y_fall, dtype=np.int64) == 1)
    return {
        "fall_output": np.ones_like(y_fall, dtype=np.float32),
        "direction_output": supervised_direction.astype(np.float32),
    }


def compute_class_weights(
    y: np.ndarray,
    num_classes: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    labels = np.asarray(y, dtype=np.int64)
    valid = labels >= 0
    if mask is not None:
        valid &= np.asarray(mask).astype(bool)
    labels = labels[valid]

    weights = np.ones(num_classes, dtype=np.float32)
    if len(labels) == 0:
        return weights

    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    nonzero = counts > 0
    if nonzero.any():
        weights[nonzero] = counts[nonzero].sum() / (float(nonzero.sum()) * counts[nonzero])
        weights[~nonzero] = 0.0
    return weights.astype(np.float32)


def make_sparse_ce_loss(class_weights: np.ndarray | None = None, name: str = "weighted_sparse_ce"):
    import tensorflow as tf

    if class_weights is None:
        return tf.keras.losses.SparseCategoricalCrossentropy(name=name)
    loss_cls = _get_weighted_sparse_ce_class()
    return loss_cls(class_weights=class_weights, name=name)


def make_direction_loss(
    loss_type: str = "ce",
    class_weights: np.ndarray | None = None,
    focal_gamma: float = 2.0,
):
    loss_type = loss_type.lower()
    if loss_type == "ce":
        return make_sparse_ce_loss(name="direction_sparse_ce")
    if loss_type == "weighted_ce":
        return make_sparse_ce_loss(class_weights=class_weights, name="direction_weighted_ce")
    if loss_type == "focal":
        loss_cls = _get_sparse_focal_class()
        return loss_cls(
            class_weights=class_weights,
            gamma=focal_gamma,
            name="direction_focal_loss",
        )
    raise ValueError("direction_loss_type must be one of: ce, weighted_ce, focal")


def make_fall_loss(class_weights: np.ndarray | None = None):
    if class_weights is None:
        return make_sparse_ce_loss(name="fall_sparse_ce")
    return make_sparse_ce_loss(class_weights=class_weights, name="fall_weighted_ce")


def _as_float_list(values: np.ndarray | list[float] | None) -> list[float] | None:
    if values is None:
        return None
    return [float(v) for v in np.asarray(values, dtype=np.float32).tolist()]


def _gather_class_weights(y_true, class_weights):
    import tensorflow as tf

    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    return tf.gather(tf.cast(class_weights, tf.float32), y_true)


def _get_weighted_sparse_ce_class():
    global _WEIGHTED_SPARSE_CE_CLASS
    if _WEIGHTED_SPARSE_CE_CLASS is not None:
        return _WEIGHTED_SPARSE_CE_CLASS

    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="DSFall")
    class WeightedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
        def __init__(
            self,
            class_weights: np.ndarray | list[float],
            from_logits: bool = False,
            name: str = "weighted_sparse_ce",
            **kwargs,
        ):
            super().__init__(name=name, **kwargs)
            self.class_weights = _as_float_list(class_weights)
            self.from_logits = bool(from_logits)

        def call(self, y_true, y_pred):
            y_true_flat = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            losses = tf.keras.losses.sparse_categorical_crossentropy(
                y_true_flat,
                y_pred,
                from_logits=self.from_logits,
            )
            weights = _gather_class_weights(y_true_flat, self.class_weights)
            return losses * weights

        def get_config(self):
            config = super().get_config()
            config.update({"class_weights": self.class_weights, "from_logits": self.from_logits})
            return config

    _WEIGHTED_SPARSE_CE_CLASS = WeightedSparseCategoricalCrossentropy
    return _WEIGHTED_SPARSE_CE_CLASS


def _get_sparse_focal_class():
    global _SPARSE_FOCAL_CLASS
    if _SPARSE_FOCAL_CLASS is not None:
        return _SPARSE_FOCAL_CLASS

    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="DSFall")
    class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
        def __init__(
            self,
            class_weights: np.ndarray | list[float] | None = None,
            gamma: float = 2.0,
            from_logits: bool = False,
            name: str = "sparse_categorical_focal_loss",
            **kwargs,
        ):
            super().__init__(name=name, **kwargs)
            self.class_weights = _as_float_list(class_weights)
            self.gamma = float(gamma)
            self.from_logits = bool(from_logits)

        def call(self, y_true, y_pred):
            y_true_flat = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            if self.from_logits:
                y_prob = tf.nn.softmax(y_pred, axis=-1)
            else:
                y_prob = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())

            ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_flat, y_prob, from_logits=False)
            one_hot = tf.one_hot(y_true_flat, depth=tf.shape(y_prob)[-1], dtype=y_prob.dtype)
            pt = tf.reduce_sum(one_hot * y_prob, axis=-1)
            focal = tf.pow(1.0 - pt, self.gamma)
            loss = focal * ce
            if self.class_weights is not None:
                loss = loss * _gather_class_weights(y_true_flat, self.class_weights)
            return loss

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "class_weights": self.class_weights,
                    "gamma": self.gamma,
                    "from_logits": self.from_logits,
                }
            )
            return config

    _SPARSE_FOCAL_CLASS = SparseCategoricalFocalLoss
    return _SPARSE_FOCAL_CLASS
