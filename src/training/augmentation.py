from __future__ import annotations

import numpy as np


def augment_raw_imu_windows(
    X_raw6: np.ndarray,
    seed: int = 42,
    strength: str = "light",
) -> np.ndarray:
    """
    Apply train-only safe augmentation on raw 6-axis IMU windows.

    The transform preserves axis identity: no flips, no permutations, no mixup.
    A small common 3D rotation is applied to accelerometer and gyroscope axes so
    derived roll/pitch features can be recomputed consistently afterwards.
    """
    X = np.asarray(X_raw6, dtype=np.float32)
    if X.ndim != 3 or X.shape[-1] != 6:
        raise ValueError(f"Expected X_raw6 with shape (N, T, 6), got {X.shape}")
    if strength != "light":
        raise ValueError("Only augment_strength='light' is currently implemented")

    rng = np.random.default_rng(seed)
    out = X.copy()
    n = len(out)
    if n == 0:
        return out

    out = _time_shift_edge(out, rng.integers(-5, 6, size=n))

    acc_scale = rng.uniform(0.9, 1.1, size=(n, 1, 1)).astype(np.float32)
    gyro_scale = rng.uniform(0.9, 1.1, size=(n, 1, 1)).astype(np.float32)
    out[:, :, :3] *= acc_scale
    out[:, :, 3:6] *= gyro_scale

    channel_std = np.std(X, axis=(0, 1), keepdims=True).astype(np.float32)
    channel_std = np.where(channel_std < 1e-6, 1.0, channel_std)
    out += rng.normal(0.0, 0.01, size=out.shape).astype(np.float32) * channel_std

    max_angle_rad = np.deg2rad(3.0)
    for i in range(n):
        angles = rng.uniform(-max_angle_rad, max_angle_rad, size=3)
        rotation = _rotation_matrix_xyz(float(angles[0]), float(angles[1]), float(angles[2]))
        out[i, :, :3] = out[i, :, :3] @ rotation.T
        out[i, :, 3:6] = out[i, :, 3:6] @ rotation.T

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _time_shift_edge(X: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    out = np.empty_like(X)
    for i, shift in enumerate(shifts.astype(int)):
        if shift == 0:
            out[i] = X[i]
        elif shift > 0:
            out[i, :shift] = X[i, :1]
            out[i, shift:] = X[i, :-shift]
        else:
            k = abs(shift)
            out[i, :-k] = X[i, k:]
            out[i, -k:] = X[i, -1:]
    return out


def _rotation_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (rot_z @ rot_y @ rot_x).astype(np.float32)
