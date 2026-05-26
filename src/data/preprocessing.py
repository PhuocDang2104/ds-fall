from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import CHANNEL_NAMES, DIRECTION_LABEL_MAPPING, FALL_LABEL_MAPPING, ProjectConfig
from src.utils.io import ensure_dir, save_json, save_pickle

from .bits_loader import create_bits_windows
from .common import has_nan_inf
from .hifd_loader import create_hifd_windows
from .weda_loader import create_weda_windows


def _cfg(config: ProjectConfig | Any, name: str, default: Any = None) -> Any:
    return getattr(config, name, default)


def build_all_windows(config: ProjectConfig) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    all_records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"datasets": {}, "warnings": []}

    weda_dir = Path(_cfg(config, "weda_raw_dir"))
    if weda_dir.exists():
        records = create_weda_windows(
            weda_dir,
            window_size=config.window_size,
            stride=config.stride,
            adl_cap_per_subject_activity=config.weda_adl_cap,
        )
        all_records.extend(records)
        summary["datasets"]["weda"] = {"windows": len(records), "raw_dir": str(weda_dir)}
        print(f"WEDA windows: {len(records)}")
    else:
        msg = f"WARNING: missing WEDA raw folder: {weda_dir}; skipping."
        print(msg)
        summary["warnings"].append(msg)

    hifd_dir = Path(_cfg(config, "hifd_raw_dir"))
    if hifd_dir.exists():
        records = create_hifd_windows(
            hifd_dir,
            window_size=config.window_size,
            stride=config.stride,
            nonfall_cap_per_subject_activity=config.hifd_nonfall_cap,
            convert_g_to_ms2=config.convert_g_to_ms2,
        )
        all_records.extend(records)
        summary["datasets"]["hifd"] = {"windows": len(records), "raw_dir": str(hifd_dir)}
        print(f"HIFD windows: {len(records)}")
    else:
        msg = f"WARNING: missing HIFD raw folder: {hifd_dir}; skipping."
        print(msg)
        summary["warnings"].append(msg)

    bits_dir = Path(_cfg(config, "bits_raw_dir"))
    if bits_dir.exists():
        records = create_bits_windows(
            bits_dir,
            window_size=config.window_size,
            stride=config.stride,
            adl_cap_per_subject_activity=config.bits_adl_cap,
            accel_source=config.bits_accel_source,
            original_fs=config.bits_original_fs,
            target_fs=config.bits_target_fs,
        )
        all_records.extend(records)
        summary["datasets"]["bits"] = {"windows": len(records), "raw_dir": str(bits_dir)}
        print(f"BITS windows: {len(records)}")
    else:
        msg = f"WARNING: missing BITS raw folder: {bits_dir}; skipping."
        print(msg)
        summary["warnings"].append(msg)

    if not all_records:
        empty = np.empty((0, config.window_size, 6), dtype=np.float32)
        return empty, pd.DataFrame(), summary

    all_records = _filter_valid_records(all_records, config.window_size)
    X = np.stack([rec.pop("X") for rec in all_records]).astype(np.float32)
    metadata = create_metadata_dataframe(all_records)
    summary.update(build_preprocessing_summary(X, metadata))
    return X, metadata, summary


def _filter_valid_records(records: list[dict[str, Any]], window_size: int) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for rec in records:
        x = rec.get("X")
        if not isinstance(x, np.ndarray):
            continue
        if x.shape != (window_size, 6):
            continue
        if has_nan_inf(x):
            continue
        valid.append(rec)
    return valid


def create_metadata_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    metadata = pd.DataFrame(records)
    metadata.insert(0, "sample_id", [f"sample_{i:07d}" for i in range(len(metadata))])
    metadata["direction_supervised"] = metadata["direction_supervised"].astype(bool)
    metadata["fall_label"] = metadata["fall_label"].astype(int)
    return metadata


def labels_from_metadata(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_fall = metadata["fall_label"].to_numpy(dtype=np.int64)
    y_direction = metadata["direction_label"].map(DIRECTION_LABEL_MAPPING).fillna(-1).to_numpy(dtype=np.int64)
    direction_mask = metadata["direction_supervised"].astype(np.float32).to_numpy()
    return y_fall, y_direction, direction_mask


def normalize_with_train_scaler(
    X: np.ndarray,
    metadata: pd.DataFrame,
    eps: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    if len(X) != len(metadata):
        raise ValueError("X and metadata length mismatch")
    train_mask = metadata["split"].to_numpy() == "train"
    if not train_mask.any():
        raise ValueError("No train samples found for scaler fitting")

    before_stats = channel_stats(X)
    train_x = X[train_mask]
    mean = train_x.mean(axis=(0, 1)).astype(np.float32)
    std = train_x.std(axis=(0, 1)).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    X_norm = ((X - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)
    after_stats = channel_stats(X_norm)

    scaler = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "channel_names": CHANNEL_NAMES,
        "fit_split": "train",
    }
    stats = {"before": before_stats, "after": after_stats}
    return X_norm, scaler, stats


def channel_stats(X: np.ndarray) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(CHANNEL_NAMES):
        arr = X[:, :, idx]
        stats[name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return stats


def build_preprocessing_summary(X: np.ndarray, metadata: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_samples": int(len(X)),
        "x_shape": list(X.shape),
        "nan_count": int(np.isnan(X).sum()),
        "inf_count": int(np.isinf(X).sum()),
    }
    if not metadata.empty:
        summary["windows_by_dataset"] = metadata["dataset"].value_counts().to_dict()
        summary["windows_by_fall_label"] = metadata["fall_label"].value_counts().to_dict()
        summary["windows_by_direction_label"] = metadata["direction_label"].value_counts().to_dict()
        summary["subjects_by_dataset"] = metadata.groupby("dataset")["subject_id"].nunique().to_dict()
        if "split" in metadata:
            summary["windows_by_split"] = metadata["split"].value_counts().to_dict()
        bits_meta = metadata[metadata["dataset"] == "bits"]
        if not bits_meta.empty:
            summary["bits_original_length"] = bits_meta["original_length"].describe().to_dict()
            summary["bits_resampled_length"] = bits_meta["resampled_length"].describe().to_dict()
    return summary


def save_processed_dataset(
    processed_dir: str | Path,
    X: np.ndarray,
    metadata: pd.DataFrame,
    scaler: dict[str, Any],
    split_subjects: dict[str, Any],
    preprocessing_summary: dict[str, Any],
) -> None:
    processed_dir = ensure_dir(processed_dir)
    y_fall, y_direction, direction_mask = labels_from_metadata(metadata)

    np.save(processed_dir / "X.npy", X.astype(np.float32))
    np.save(processed_dir / "y_fall.npy", y_fall.astype(np.int64))
    np.save(processed_dir / "y_direction.npy", y_direction.astype(np.int64))
    np.save(processed_dir / "direction_mask.npy", direction_mask.astype(np.float32))
    metadata.to_csv(processed_dir / "metadata.csv", index=False)
    save_pickle(processed_dir / "scaler.pkl", scaler)
    save_json(
        processed_dir / "label_mapping.json",
        {"fall": FALL_LABEL_MAPPING, "direction": DIRECTION_LABEL_MAPPING},
    )
    save_json(processed_dir / "split_subjects.json", split_subjects)
    save_json(processed_dir / "preprocessing_summary.json", preprocessing_summary)
