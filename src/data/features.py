from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


RAW6_CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]

FEATURE_CHANNELS: dict[str, list[str]] = {
    "raw6": RAW6_CHANNELS,
    "mag_jerk9": RAW6_CHANNELS + ["acc_mag", "gyro_mag", "jerk"],
    "roll_pitch11": RAW6_CHANNELS + ["acc_mag", "gyro_mag", "jerk", "roll", "pitch"],
    "tilt12": RAW6_CHANNELS + ["acc_mag", "gyro_mag", "jerk", "roll", "pitch", "tilt_delta"],
}

ACC_STREAM_CHANNELS: dict[str, list[str]] = {
    "raw6": ["ax", "ay", "az"],
    "mag_jerk9": ["ax", "ay", "az", "acc_mag", "jerk"],
    "roll_pitch11": ["ax", "ay", "az", "acc_mag", "jerk", "roll", "pitch"],
    "tilt12": ["ax", "ay", "az", "acc_mag", "jerk", "roll", "pitch", "tilt_delta"],
}

GYRO_STREAM_CHANNELS: dict[str, list[str]] = {
    "raw6": ["gx", "gy", "gz"],
    "mag_jerk9": ["gx", "gy", "gz", "gyro_mag"],
    "roll_pitch11": ["gx", "gy", "gz", "gyro_mag"],
    "tilt12": ["gx", "gy", "gz", "gyro_mag"],
}


def validate_feature_set(feature_set: str) -> str:
    if feature_set not in FEATURE_CHANNELS:
        valid = ", ".join(sorted(FEATURE_CHANNELS))
        raise ValueError(f"Unknown feature_set={feature_set!r}. Expected one of: {valid}")
    return feature_set


def feature_channel_names(feature_set: str) -> list[str]:
    return list(FEATURE_CHANNELS[validate_feature_set(feature_set)])


def stream_channel_indices(feature_set: str) -> tuple[list[int], list[int]]:
    names = feature_channel_names(feature_set)
    by_name = {name: idx for idx, name in enumerate(names)}
    acc_idx = [by_name[name] for name in ACC_STREAM_CHANNELS[feature_set]]
    gyro_idx = [by_name[name] for name in GYRO_STREAM_CHANNELS[feature_set]]
    return acc_idx, gyro_idx


def compute_imu_features(
    X_raw6: np.ndarray,
    feature_set: str = "tilt12",
    fs: float = 50.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Build DS-Fall feature sets from unnormalized 6-axis IMU windows.

    The first six output channels always remain ax, ay, az, gx, gy, gz.
    Additional features are sequence-level, physically meaningful channels for
    impact, rotation magnitude, jerk, and signed tilt geometry.
    """
    feature_set = validate_feature_set(feature_set)
    X = np.asarray(X_raw6, dtype=np.float32)
    if X.ndim != 3 or X.shape[-1] != 6:
        raise ValueError(f"Expected X_raw6 with shape (N, T, 6), got {X.shape}")

    if feature_set == "raw6":
        return _finite(X.astype(np.float32))

    ax = X[:, :, 0]
    ay = X[:, :, 1]
    az = X[:, :, 2]
    gx = X[:, :, 3]
    gy = X[:, :, 4]
    gz = X[:, :, 5]

    acc_mag = np.sqrt(np.maximum(ax * ax + ay * ay + az * az, 0.0) + eps)
    gyro_mag = np.sqrt(np.maximum(gx * gx + gy * gy + gz * gz, 0.0) + eps)

    dt = 1.0 / float(fs)
    jerk = np.diff(acc_mag, axis=1, prepend=acc_mag[:, :1]) / dt

    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(np.maximum(ay * ay + az * az, 0.0) + eps))

    features = [X]
    if feature_set in {"mag_jerk9", "roll_pitch11", "tilt12"}:
        features.extend([acc_mag[:, :, None], gyro_mag[:, :, None], jerk[:, :, None]])
    if feature_set in {"roll_pitch11", "tilt12"}:
        features.extend([roll[:, :, None], pitch[:, :, None]])
    if feature_set == "tilt12":
        roll_unwrapped = np.unwrap(roll, axis=1)
        pitch_unwrapped = np.unwrap(pitch, axis=1)
        d_roll = np.diff(roll_unwrapped, axis=1, prepend=roll_unwrapped[:, :1])
        d_pitch = np.diff(pitch_unwrapped, axis=1, prepend=pitch_unwrapped[:, :1])
        tilt_delta = np.sqrt(np.maximum(d_roll * d_roll + d_pitch * d_pitch, 0.0))
        features.append(tilt_delta[:, :, None])

    out = np.concatenate(features, axis=-1).astype(np.float32)
    expected_channels = len(FEATURE_CHANNELS[feature_set])
    if out.shape[-1] != expected_channels:
        raise RuntimeError(f"Feature shape mismatch for {feature_set}: got {out.shape[-1]} channels")
    return _finite(out)


def fit_feature_scaler(
    X_features: np.ndarray,
    metadata: pd.DataFrame,
    feature_set: str,
    split_col: str = "split",
    train_split: str = "train",
    eps: float = 1e-6,
) -> dict[str, Any]:
    feature_set = validate_feature_set(feature_set)
    X = np.asarray(X_features, dtype=np.float32)
    if len(X) != len(metadata):
        raise ValueError("X_features and metadata length mismatch")
    if split_col not in metadata:
        raise ValueError(f"metadata must contain {split_col!r} to fit train-only scaler")

    train_mask = metadata[split_col].to_numpy() == train_split
    if not train_mask.any():
        raise ValueError(f"No samples found for {split_col}={train_split!r}")

    train_x = X[train_mask]
    mean = train_x.mean(axis=(0, 1)).astype(np.float32)
    std = train_x.std(axis=(0, 1)).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    return {
        "feature_set": feature_set,
        "channel_names": feature_channel_names(feature_set),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "fit_split": train_split,
        "eps": float(eps),
    }


def transform_with_feature_scaler(X_features: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    X = np.asarray(X_features, dtype=np.float32)
    mean = np.asarray(scaler["mean"], dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(scaler["std"], dtype=np.float32).reshape(1, 1, -1)
    if X.shape[-1] != mean.shape[-1]:
        raise ValueError(f"Scaler expects {mean.shape[-1]} channels, got {X.shape[-1]}")
    return _finite(((X - mean) / std).astype(np.float32))


def fit_transform_feature_set(
    X_raw6: np.ndarray,
    metadata: pd.DataFrame,
    feature_set: str,
    fs: float = 50.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    features = compute_imu_features(X_raw6, feature_set=feature_set, fs=fs)
    scaler = fit_feature_scaler(features, metadata, feature_set=feature_set)
    return transform_with_feature_scaler(features, scaler), scaler


def inverse_standardize_raw6(X_processed: np.ndarray, raw6_scaler: dict[str, Any] | None) -> np.ndarray:
    """
    Recover physical-ish raw6 values from notebook-01 normalized X.npy.

    If no compatible scaler is provided, the input is returned as float32. This
    keeps the helper usable for future pipelines that may save raw windows.
    """
    X = np.asarray(X_processed, dtype=np.float32)
    if raw6_scaler is None:
        return _finite(X)

    channels = raw6_scaler.get("channel_names")
    mean = raw6_scaler.get("mean")
    std = raw6_scaler.get("std")
    if channels != RAW6_CHANNELS or mean is None or std is None or len(mean) != 6 or len(std) != 6:
        return _finite(X)

    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 6)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 6)
    return _finite((X * std_arr + mean_arr).astype(np.float32))


def handcrafted_direction_features(
    X_raw6: np.ndarray,
    fs: float = 50.0,
    pre_steps: int = 20,
    post_steps: int = 20,
) -> tuple[np.ndarray, list[str]]:
    features = compute_imu_features(X_raw6, feature_set="tilt12", fs=fs)
    names = feature_channel_names("tilt12")
    idx = {name: i for i, name in enumerate(names)}

    acc_mag = features[:, :, idx["acc_mag"]]
    gyro_mag = features[:, :, idx["gyro_mag"]]
    jerk = features[:, :, idx["jerk"]]
    roll = features[:, :, idx["roll"]]
    pitch = features[:, :, idx["pitch"]]

    pre_slice = slice(0, min(pre_steps, roll.shape[1]))
    post_slice = slice(max(0, roll.shape[1] - post_steps), roll.shape[1])
    pre_roll = roll[:, pre_slice].mean(axis=1)
    post_roll = roll[:, post_slice].mean(axis=1)
    pre_pitch = pitch[:, pre_slice].mean(axis=1)
    post_pitch = pitch[:, post_slice].mean(axis=1)

    out = np.column_stack(
        [
            acc_mag.max(axis=1),
            gyro_mag.max(axis=1),
            np.abs(jerk).max(axis=1),
            pre_roll,
            post_roll,
            pre_pitch,
            post_pitch,
            post_roll - pre_roll,
            post_pitch - pre_pitch,
        ]
    ).astype(np.float32)
    feature_names = [
        "max_acc_mag",
        "max_gyro_mag",
        "max_jerk",
        "pre_roll",
        "post_roll",
        "pre_pitch",
        "post_pitch",
        "delta_roll_window",
        "delta_pitch_window",
    ]
    return _finite(out), feature_names


def _finite(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
