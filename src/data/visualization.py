from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import CHANNEL_NAMES
from src.utils.io import ensure_dir

from .common import compute_acc_mag


def _save_or_show(fig: plt.Figure, save_path: str | Path | None = None) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_imu_window(
    window: np.ndarray,
    title: str = "IMU window",
    save_path: str | Path | None = None,
) -> None:
    t = np.arange(len(window)) / 50.0
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for i, name in enumerate(CHANNEL_NAMES[:3]):
        axes[0].plot(t, window[:, i], label=name)
    axes[0].set_title(f"{title} - accelerometer")
    axes[0].set_ylabel("acc")
    axes[0].legend(ncol=3)
    for i, name in enumerate(CHANNEL_NAMES[3:], start=3):
        axes[1].plot(t, window[:, i], label=name)
    axes[1].set_title("gyroscope")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("gyro")
    axes[1].legend(ncol=3)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_bits_resampling(
    original_seq: np.ndarray,
    resampled_seq: np.ndarray,
    original_fs: float = 20.0,
    target_fs: float = 50.0,
    save_path: str | Path | None = None,
) -> None:
    orig_t = np.arange(len(original_seq)) / original_fs
    res_t = np.arange(len(resampled_seq)) / target_fs
    orig_mag = compute_acc_mag(original_seq)
    res_mag = compute_acc_mag(resampled_seq)
    orig_peak = int(np.argmax(orig_mag))
    res_peak = int(np.argmax(res_mag))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    axes[0].plot(orig_t, orig_mag, label="20 Hz original")
    axes[0].axvline(orig_t[orig_peak], color="r", linestyle="--", label="original peak")
    axes[0].set_title("BITS original acceleration magnitude")
    axes[0].set_xlabel("time (s)")
    axes[0].legend()

    axes[1].plot(res_t, res_mag, label="50 Hz resampled")
    axes[1].axvline(res_t[res_peak], color="r", linestyle="--", label="resampled peak")
    axes[1].set_title("BITS resampled acceleration magnitude")
    axes[1].set_xlabel("time (s)")
    axes[1].legend()
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_class_distribution(
    metadata: pd.DataFrame,
    column: str,
    title: str,
    save_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    order = metadata[column].value_counts().index
    sns.countplot(data=metadata, x=column, order=order, ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_dataset_distribution(metadata: pd.DataFrame, save_path: str | Path | None = None) -> None:
    plot_class_distribution(metadata, "dataset", "Windows by dataset", save_path)


def plot_channel_stats(
    X: np.ndarray,
    title: str,
    save_path: str | Path | None = None,
) -> None:
    means = X.mean(axis=(0, 1))
    stds = X.std(axis=(0, 1))
    x = np.arange(len(CHANNEL_NAMES))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.18, means, width=0.36, label="mean")
    ax.bar(x + 0.18, stds, width=0.36, label="std")
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNEL_NAMES)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_training_curves(history: dict, keys: Iterable[str], save_path: str | Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in keys:
        if key in history:
            ax.plot(history[key], label=key)
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_path)
