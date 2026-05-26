from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    path = Path(path)
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", **kwargs)


def compute_acc_mag(seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq)
    return np.sqrt(np.sum(np.square(seq[:, :3]), axis=1))


def has_nan_inf(seq: np.ndarray) -> bool:
    arr = np.asarray(seq)
    return bool(np.isnan(arr).any() or np.isinf(arr).any())


def pad_or_crop_window(seq: np.ndarray, center_idx: int, window_size: int = 100) -> tuple[np.ndarray, int, int]:
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError(f"Expected 2D sequence, got shape {seq.shape}")
    if len(seq) == 0:
        raise ValueError("Cannot create a window from an empty sequence")

    half = window_size // 2
    start = int(center_idx) - half
    end = start + window_size

    if start < 0:
        end -= start
        start = 0
    if end > len(seq):
        start -= end - len(seq)
        end = len(seq)
    start = max(start, 0)
    end = min(end, len(seq))

    window = seq[start:end]
    if len(window) < window_size:
        pad_before = 0
        pad_after = window_size - len(window)
        if start == 0 and center_idx < half:
            pad_before = min(half - int(center_idx), pad_after)
            pad_after -= pad_before
        window = np.pad(window, ((pad_before, pad_after), (0, 0)), mode="edge")

    return window.astype(np.float32), int(start), int(end)


def sliding_windows(seq: np.ndarray, window_size: int = 100, stride: int = 50) -> list[tuple[np.ndarray, int, int]]:
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) < window_size:
        return []

    out: list[tuple[np.ndarray, int, int]] = []
    for start in range(0, len(seq) - window_size + 1, stride):
        end = start + window_size
        out.append((seq[start:end].astype(np.float32), start, end))
    return out


def resample_sequence_uniform(
    seq: np.ndarray,
    original_fs: float,
    target_fs: float,
) -> np.ndarray | None:
    """
    Resample a uniformly sampled sequence using index-derived time.

    This function intentionally ignores raw timestamp columns. It is used for
    BITS-2 local CSV files where timestamps can be rounded/repeated.
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2 or len(seq) < 2:
        return None

    t_original = np.arange(len(seq), dtype=np.float64) / float(original_fs)
    duration = t_original[-1]
    t_target = np.arange(0.0, duration + 1e-9, 1.0 / float(target_fs))

    out = np.zeros((len(t_target), seq.shape[1]), dtype=np.float32)
    for c in range(seq.shape[1]):
        out[:, c] = np.interp(t_target, t_original, seq[:, c]).astype(np.float32)
    return out


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(_jsonable(k)): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
