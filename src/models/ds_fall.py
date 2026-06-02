from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers

from src.data.features import feature_channel_names, stream_channel_indices, validate_feature_set


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


def _slice_channels(inputs: tf.Tensor, indices: list[int], name: str) -> tf.Tensor:
    return layers.Lambda(lambda t, idx=indices: tf.gather(t, idx, axis=-1), name=name)(inputs)


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
    feature_set: str = "raw6",
    show_summary: bool = True,
) -> Model:
    feature_set = validate_feature_set(feature_set)
    expected_channels = len(feature_channel_names(feature_set))
    if input_shape[-1] != expected_channels:
        raise ValueError(
            f"feature_set={feature_set!r} expects {expected_channels} channels, got input_shape={input_shape}"
        )
    acc_indices, gyro_indices = stream_channel_indices(feature_set)

    inputs = layers.Input(shape=input_shape, name="imu_input")

    acc = _slice_channels(inputs, acc_indices, name="acc_input")
    gyro = _slice_channels(inputs, gyro_indices, name="gyro_input")

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


def build_ds_fall_rd_model(
    input_shape: tuple[int, int] = (100, 12),
    num_direction_classes: int = 3,
    feature_set: str = "tilt12",
    show_summary: bool = True,
) -> Model:
    feature_set = validate_feature_set(feature_set)
    expected_channels = len(feature_channel_names(feature_set))
    if input_shape[-1] != expected_channels:
        raise ValueError(
            f"feature_set={feature_set!r} expects {expected_channels} channels, got input_shape={input_shape}"
        )
    acc_indices, gyro_indices = stream_channel_indices(feature_set)

    inputs = layers.Input(shape=input_shape, name="imu_input")

    acc = _slice_channels(inputs, acc_indices, name="acc_input")
    gyro = _slice_channels(inputs, gyro_indices, name="gyro_input")

    acc_features = _sensor_encoder(acc, "acc")
    gyro_features = _sensor_encoder(gyro, "gyro")

    x = layers.Concatenate(name="sensor_concat")([acc_features, gyro_features])
    x = layers.Conv1D(64, kernel_size=1, padding="same", name="fusion_pointwise_conv")(x)
    x = layers.BatchNormalization(name="fusion_bn")(x)
    x = layers.Activation("relu", name="fusion_relu")(x)

    x = dstcn_block(x, channels=64, dilation=1, name="dstcn_d1")
    x = dstcn_block(x, channels=64, dilation=2, name="dstcn_d2")
    x = dstcn_block(x, channels=96, dilation=4, name="dstcn_d4")

    fall_context = gated_attention_pooling(x, name="fall_attention_pooling")
    direction_context = gated_attention_pooling(x, name="direction_attention_pooling")

    fall = layers.Dense(32, activation="relu", name="fall_dense")(fall_context)
    fall = layers.Dropout(0.2, name="fall_dropout")(fall)
    fall_output = layers.Dense(2, activation="softmax", name="fall_output")(fall)

    direction = layers.Dense(32, activation="relu", name="direction_dense")(direction_context)
    direction = layers.Dropout(0.2, name="direction_dropout")(direction)
    direction_output = layers.Dense(num_direction_classes, activation="softmax", name="direction_output")(direction)

    model = Model(
        inputs=inputs,
        outputs={"fall_output": fall_output, "direction_output": direction_output},
        name="DS-Fall-RD",
    )

    if show_summary:
        model.summary()
        print(f"Total parameters: {model.count_params():,}")
    return model


def build_ds_fall_hier_dir_model(
    input_shape: tuple[int, int] = (100, 12),
    feature_set: str = "tilt12",
    use_summary_branch: bool = False,
    summary_shape: tuple[int, ...] = (12,),
    show_summary: bool = True,
) -> Model:
    feature_set = validate_feature_set(feature_set)
    expected_channels = len(feature_channel_names(feature_set))
    if input_shape[-1] != expected_channels:
        raise ValueError(
            f"feature_set={feature_set!r} expects {expected_channels} channels, got input_shape={input_shape}"
        )
    acc_indices, gyro_indices = stream_channel_indices(feature_set)

    imu_input = layers.Input(shape=input_shape, name="imu_input")

    acc = _slice_channels(imu_input, acc_indices, name="acc_input")
    gyro = _slice_channels(imu_input, gyro_indices, name="gyro_input")

    acc_features = _sensor_encoder(acc, "acc")
    gyro_features = _sensor_encoder(gyro, "gyro")

    x = layers.Concatenate(name="sensor_concat")([acc_features, gyro_features])
    x = layers.Conv1D(64, kernel_size=1, padding="same", name="fusion_pointwise_conv")(x)
    x = layers.BatchNormalization(name="fusion_bn")(x)
    x = layers.Activation("relu", name="fusion_relu")(x)

    x = dstcn_block(x, channels=64, dilation=1, name="dstcn_d1")
    x = dstcn_block(x, channels=64, dilation=2, name="dstcn_d2")
    x = dstcn_block(x, channels=96, dilation=4, name="dstcn_d4")

    fall_context = gated_attention_pooling(x, name="fall_attention_pooling")
    plane_context = gated_attention_pooling(x, name="plane_attention_pooling")
    sagittal_context = gated_attention_pooling(x, name="sagittal_attention_pooling")

    model_inputs: tf.Tensor | dict[str, tf.Tensor] = imu_input
    if use_summary_branch:
        summary_input = layers.Input(shape=summary_shape, name="summary_input")
        summary_embedding = layers.Dense(16, activation="relu", name="direction_summary_dense")(summary_input)
        plane_context = layers.Concatenate(name="plane_summary_concat")([plane_context, summary_embedding])
        sagittal_context = layers.Concatenate(name="sagittal_summary_concat")([sagittal_context, summary_embedding])
        model_inputs = {"imu_input": imu_input, "summary_input": summary_input}

    fall = layers.Dense(32, activation="relu", name="fall_dense")(fall_context)
    fall = layers.Dropout(0.2, name="fall_dropout")(fall)
    fall_output = layers.Dense(2, activation="softmax", name="fall_output")(fall)

    plane = layers.Dense(32, activation="relu", name="plane_dense")(plane_context)
    plane = layers.Dropout(0.2, name="plane_dropout")(plane)
    plane_output = layers.Dense(2, activation="softmax", name="plane_output")(plane)

    sagittal = layers.Dense(32, activation="relu", name="sagittal_dense")(sagittal_context)
    sagittal = layers.Dropout(0.2, name="sagittal_dropout")(sagittal)
    sagittal_output = layers.Dense(2, activation="softmax", name="sagittal_output")(sagittal)

    model = Model(
        inputs=model_inputs,
        outputs={
            "fall_output": fall_output,
            "plane_output": plane_output,
            "sagittal_output": sagittal_output,
        },
        name="DS-Fall-A10-Hier-Dir" if not use_summary_branch else "DS-Fall-A11-Hier-Dir-Summary",
    )

    if show_summary:
        model.summary()
        print(f"Total parameters: {model.count_params():,}")
    return model


def build_model(
    model_name: str,
    input_shape: tuple[int, int],
    feature_set: str,
    num_direction_classes: int = 3,
    show_summary: bool = True,
) -> Model:
    normalized = model_name.lower().replace("-", "_")
    if normalized in {"ds_fall", "baseline"}:
        return build_ds_fall_model(
            input_shape=input_shape,
            num_direction_classes=num_direction_classes,
            feature_set=feature_set,
            show_summary=show_summary,
        )
    if normalized in {"ds_fall_rd", "rd"}:
        return build_ds_fall_rd_model(
            input_shape=input_shape,
            num_direction_classes=num_direction_classes,
            feature_set=feature_set,
            show_summary=show_summary,
        )
    if normalized in {"ds_fall_hier_dir", "a10_hier_dir", "hier_dir"}:
        return build_ds_fall_hier_dir_model(
            input_shape=input_shape,
            feature_set=feature_set,
            use_summary_branch=False,
            show_summary=show_summary,
        )
    if normalized in {"ds_fall_hier_dir_summary", "a11_hier_dir_summary", "hier_dir_summary"}:
        return build_ds_fall_hier_dir_model(
            input_shape=input_shape,
            feature_set=feature_set,
            use_summary_branch=True,
            show_summary=show_summary,
        )
    raise ValueError(f"Unknown model_name={model_name!r}")
