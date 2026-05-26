from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers


def dsconv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    name: str,
    dilation_rate: int = 1,
    dropout: float = 0.0,
) -> tf.Tensor:
    x = layers.SeparableConv1D(
        filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        name=f"{name}_sepconv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name=f"{name}_dropout")(x)
    return x


def _sensor_encoder(x: tf.Tensor, prefix: str) -> tf.Tensor:
    x = layers.Conv1D(16, kernel_size=5, padding="same", name=f"{prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{prefix}_conv1_bn")(x)
    x = layers.Activation("relu", name=f"{prefix}_conv1_relu")(x)
    x = dsconv_block(x, 24, 5, name=f"{prefix}_dsconv1")
    x = dsconv_block(x, 32, 3, name=f"{prefix}_dsconv2")
    return x


def dstcn_block(
    x: tf.Tensor,
    channels: int,
    dilation: int,
    name: str,
    dropout: float = 0.1,
) -> tf.Tensor:
    residual = x
    y = layers.SeparableConv1D(
        channels,
        kernel_size=3,
        dilation_rate=dilation,
        padding="same",
        name=f"{name}_sepconv1",
    )(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.Activation("relu", name=f"{name}_relu1")(y)
    y = layers.Dropout(dropout, name=f"{name}_dropout")(y)
    y = layers.SeparableConv1D(
        channels,
        kernel_size=3,
        dilation_rate=dilation,
        padding="same",
        name=f"{name}_sepconv2",
    )(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)

    if residual.shape[-1] != channels:
        residual = layers.Conv1D(channels, kernel_size=1, padding="same", name=f"{name}_projection")(residual)

    y = layers.Add(name=f"{name}_add")([residual, y])
    return layers.Activation("relu", name=f"{name}_out_relu")(y)


def gated_attention_pooling(x: tf.Tensor, name: str = "gated_attention") -> tf.Tensor:
    channels = int(x.shape[-1])
    hidden_units = max(channels // 2, 1)
    h = layers.Dense(hidden_units, activation="tanh", name=f"{name}_hidden")(x)
    score = layers.Dense(1, name=f"{name}_score")(h)
    alpha = layers.Softmax(axis=1, name=f"{name}_alpha")(score)
    context = layers.Lambda(lambda tensors: tf.reduce_sum(tensors[0] * tensors[1], axis=1), name=f"{name}_pool")(
        [x, alpha]
    )
    return context


def build_ds_fall_model(
    input_shape: tuple[int, int] = (100, 6),
    num_direction_classes: int = 3,
    show_summary: bool = True,
) -> Model:
    inputs = layers.Input(shape=input_shape, name="imu_input")

    acc = layers.Lambda(lambda t: t[:, :, 0:3], name="acc_input")(inputs)
    gyro = layers.Lambda(lambda t: t[:, :, 3:6], name="gyro_input")(inputs)

    acc_features = _sensor_encoder(acc, "acc")
    gyro_features = _sensor_encoder(gyro, "gyro")

    x = layers.Concatenate(name="sensor_concat")([acc_features, gyro_features])
    x = layers.Conv1D(64, kernel_size=1, padding="same", name="fusion_pointwise_conv")(x)
    x = layers.BatchNormalization(name="fusion_bn")(x)
    x = layers.Activation("relu", name="fusion_relu")(x)

    x = dstcn_block(x, channels=64, dilation=1, name="dstcn_d1")
    x = dstcn_block(x, channels=64, dilation=2, name="dstcn_d2")
    x = dstcn_block(x, channels=96, dilation=4, name="dstcn_d4")

    z = gated_attention_pooling(x, name="gated_attention")

    z = layers.Dense(64, name="shared_dense")(z)
    z = layers.BatchNormalization(name="shared_bn")(z)
    z = layers.Activation("relu", name="shared_relu")(z)
    z = layers.Dropout(0.2, name="shared_dropout")(z)

    fall = layers.Dense(32, activation="relu", name="fall_dense")(z)
    fall = layers.Dropout(0.1, name="fall_dropout")(fall)
    fall_output = layers.Dense(2, activation="softmax", name="fall_output")(fall)

    direction = layers.Dense(32, activation="relu", name="direction_dense")(z)
    direction = layers.Dropout(0.1, name="direction_dropout")(direction)
    direction_output = layers.Dense(num_direction_classes, activation="softmax", name="direction_output")(direction)

    model = Model(inputs=inputs, outputs={"fall_output": fall_output, "direction_output": direction_output}, name="DS-Fall")

    if show_summary:
        model.summary()
        print(f"Total parameters: {model.count_params():,}")
    return model
