from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DIRECTION_LABEL_MAPPING, FALL_LABEL_MAPPING, make_config
from src.data.bits_loader import (
    BITS_ADL_ACTIVITY_NAMES,
    BITS_FALL_ACTIVITY_NAMES,
    BITS_FALL_DIRECTIONS,
    find_bits_trials,
)
from src.data.common import compute_acc_mag, has_nan_inf, pad_or_crop_window, sliding_windows
from src.data.units import harmonize_record_units
from src.data.weda_loader import (
    WEDA_ACTIVITY_NAMES,
    WEDA_FALL_DIRECTIONS,
    WEDA_HARD_NEGATIVES,
    load_weda_fall_timestamps,
    parse_weda_subject_trial,
)
from src.models.losses import compute_class_weights
from src.training.dataset import make_tf_dataset
from src.training.train import compile_ds_fall_model
from src.utils.io import ensure_dir, load_json, save_json, save_pickle
from src.utils.seed import set_seed


SAMPLING_RATE = 25.0
WINDOW_SECONDS = 2.0
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_SECONDS)
STRIDE_SECONDS = 0.5
STRIDE_SIZE = 12
FEATURE_SET = "tilt12"
FEATURE_ORDER = [
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "acc_mag",
    "gyro_mag",
    "jerk",
    "roll",
    "pitch",
    "tilt_delta",
]
DIRECTION_NAMES = {0: "forward", 1: "backward", 2: "lateral", -1: "none"}
FALL_NAMES = {0: "non_fall", 1: "fall"}


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    train_dataset: str
    test_dataset: str
    train_datasets: tuple[str, ...]
    val_datasets: tuple[str, ...]
    test_datasets: tuple[str, ...]


EXPERIMENTS = [
    ExperimentSpec("E1_BITS_TO_BITS", "bits", "bits", ("bits",), ("bits",), ("bits",)),
    ExperimentSpec("E2_WEDA_TO_WEDA", "weda", "weda", ("weda",), ("weda",), ("weda",)),
    ExperimentSpec("E3_BITS_WEDA_MIXED", "bits+weda", "bits+weda", ("bits", "weda"), ("bits", "weda"), ("bits", "weda")),
    ExperimentSpec("E4_BITS_TO_WEDA", "bits", "weda", ("bits",), ("bits",), ("weda",)),
    ExperimentSpec("E5_WEDA_TO_BITS", "weda", "bits", ("weda",), ("weda",), ("bits",)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the required 25 Hz A5WCEFW experiments on raw BITS and raw WEDA-FALL 25Hz."
    )
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--window-mode",
        choices=["full_trial", "event_centered"],
        default="full_trial",
        help="full_trial matches prompt mode; event_centered uses one impact/fall-interval fall window plus capped ADL windows.",
    )
    parser.add_argument(
        "--artifact-subdir",
        type=str,
        default=None,
        help="Subdirectory under artifacts/ for experiment outputs.",
    )
    parser.add_argument(
        "--monitor",
        choices=["val_loss", "val_domain_score"],
        default="val_loss",
        help="Checkpoint/early-stopping monitor.",
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--audit-only", action="store_true", help="Build 25 Hz windows and reports, but do not train.")
    parser.add_argument("--force", action="store_true", help="Retrain experiments even if metrics.json already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = make_config(args.project_root)
    set_seed(config.seed)
    artifact_subdir = args.artifact_subdir or ("experiments" if args.window_mode == "full_trial" else "experiments_25hz_event")
    artifacts_dir = ensure_dir(config.project_root / "artifacts" / artifact_subdir)

    print("=== DS-Fall-RD A5WCEFW 25 Hz pipeline ===")
    print(f"project_root={config.project_root}")
    print(f"sampling_rate={SAMPLING_RATE:g} Hz, window_size={WINDOW_SIZE}, input_shape=({WINDOW_SIZE}, {len(FEATURE_ORDER)})")

    print(f"window_mode={args.window_mode}, artifact_dir={artifacts_dir}")
    data = build_25hz_dataset(config, window_mode=args.window_mode)
    save_25hz_dataset_artifacts(data, artifacts_dir)
    print_dataset_audit(data["audit"])
    validate_25hz_dataset(data)

    if args.audit_only:
        print("Audit-only mode: preprocessing artifacts were saved; training skipped.")
        return

    rows: list[dict[str, Any]] = []
    for spec in EXPERIMENTS:
        row = run_or_load_experiment(
            config=config,
            artifacts_dir=artifacts_dir,
            spec=spec,
            data=data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            monitor=args.monitor,
            patience=args.patience,
            force=args.force,
        )
        rows.append(row)
        save_summary(rows, artifacts_dir)

    save_summary(rows, artifacts_dir)
    print("\n=== Summary 25 Hz A5WCEFW ===")
    print(pd.DataFrame(rows)[summary_columns()].to_string(index=False))


def build_25hz_dataset(config, window_mode: str = "full_trial") -> dict[str, Any]:
    print("\n[1/3] Building BITS 20 Hz -> 25 Hz windows...")
    bits_records, bits_trial_audit = create_bits_25hz_records(config, window_mode=window_mode)
    print(f"BITS records: {len(bits_records)}")

    print("\n[2/3] Building WEDA-FALL raw 25Hz windows...")
    weda_records, weda_trial_audit = create_weda_25hz_records(config, window_mode=window_mode)
    print(f"WEDA records: {len(weda_records)}")

    records = bits_records + weda_records
    records, unit_summary = harmonize_record_units(records, config.dataset_unit_map)
    X_raw6, metadata = records_to_arrays(records)
    metadata = apply_saved_subject_split(metadata, load_json(config.processed_dir / "split_subjects.json"))
    y_fall, y_direction, direction_mask = labels_from_metadata(metadata)

    X_features = compute_features_25hz(X_raw6)
    feature_stats = feature_stats_dataframe(X_features)
    channel_stats = raw6_channel_stats(metadata, X_raw6, X_features)
    audit = build_audit_report(
        metadata=metadata,
        X_raw6=X_raw6,
        X_features=X_features,
        trial_audits={"bits": bits_trial_audit, "weda": weda_trial_audit},
        unit_summary=unit_summary,
    )
    return {
        "X_raw6": X_raw6,
        "X_features": X_features,
        "metadata": metadata,
        "y_fall": y_fall,
        "y_direction": y_direction,
        "direction_mask": direction_mask,
        "feature_stats": feature_stats,
        "channel_stats": channel_stats,
        "audit": audit,
        "trial_audits": {"bits": bits_trial_audit, "weda": weda_trial_audit},
        "unit_summary": unit_summary,
        "window_mode": window_mode,
    }


def create_bits_25hz_records(config, window_mode: str = "full_trial") -> tuple[list[dict[str, Any]], pd.DataFrame]:
    bits_dir = Path(config.bits_raw_dir)
    if not bits_dir.exists():
        raise FileNotFoundError(f"Missing BITS raw folder: {bits_dir}")

    records: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for trial in find_bits_trials(bits_dir):
        seq20, sensor_info = load_bits_raw6_uniform20(trial["csv_path"], accel_source=config.bits_accel_source)
        if seq20 is None or len(seq20) < 2 or has_nan_inf(seq20):
            continue

        seq25, time25 = resample_uniform_time(seq20, source_hz=20.0, target_hz=SAMPLING_RATE)
        effective_after = effective_fs_from_time(time25)
        trial_rows.append(
            {
                "dataset": "bits",
                "source_path": str(trial["csv_path"]),
                "subject_id": trial["subject_id"],
                "trial_id": Path(trial["csv_path"]).stem,
                "activity_id": trial["activity_id"],
                "class_dir": trial["class_dir"],
                "source_hz": 20.0,
                "target_hz": SAMPLING_RATE,
                "raw_samples": int(len(seq20)),
                "processed_samples": int(len(seq25)),
                "duration_seconds": float(time25[-1] - time25[0]) if len(time25) > 1 else 0.0,
                "effective_fs_before": 20.0,
                "effective_fs_after": effective_after,
                "resampling_method": "uniform_20hz_time_interpolation_to_25hz",
                "timestamp_mode": sensor_info.get("timestamp_mode"),
                "acc_rows": sensor_info.get("acc_rows"),
                "gyro_rows": sensor_info.get("gyro_rows"),
            }
        )
        if len(seq25) < WINDOW_SIZE:
            continue

        fall_label = int(trial["class_dir"] == "fall")
        if fall_label:
            direction_label, direction_supervised = BITS_FALL_DIRECTIONS.get(trial["activity_id"], ("other", False))
        else:
            direction_label, direction_supervised = "none", False

        if window_mode == "event_centered" and fall_label:
            center = int(np.argmax(compute_acc_mag(seq25)))
            windows = [pad_or_crop_window(seq25, center, WINDOW_SIZE)]
            window_rule = "impact_peak_centered_label_by_activity"
        elif window_mode == "event_centered":
            windows = sliding_windows(seq25, window_size=WINDOW_SIZE, stride=STRIDE_SIZE)[: config.bits_adl_cap]
            window_rule = "adl_sliding_capped"
        else:
            windows = sliding_windows(seq25, window_size=WINDOW_SIZE, stride=STRIDE_SIZE)
            window_rule = "full_trial_sliding_label_by_activity"

        for window, start, end in windows:
            if window.shape != (WINDOW_SIZE, 6) or has_nan_inf(window):
                continue
            records.append(
                build_record(
                    dataset="bits",
                    window=window,
                    source_path=str(trial["csv_path"]),
                    subject_id=trial["subject_id"],
                    trial_id=Path(trial["csv_path"]).stem,
                    activity_id=trial["activity_id"],
                    activity_name=bits_activity_name(trial["activity_id"]),
                    fall_label=fall_label,
                    direction_label=direction_label if fall_label else "none",
                    direction_supervised=bool(direction_supervised and fall_label),
                    start_idx=start,
                    end_idx=end,
                    source_folder=str(bits_dir),
                    source_hz=20.0,
                    target_hz=SAMPLING_RATE,
                    resampling_method="uniform_20hz_time_interpolation_to_25hz",
                    window_rule=window_rule,
                    original_length=len(seq20),
                    processed_length=len(seq25),
                    fall_interval_start=np.nan,
                    fall_interval_end=np.nan,
                    unit_conversion="dataset_unit_map",
                )
            )
    if not records:
        raise ValueError("BITS 25 Hz preprocessing produced no windows.")
    return records, pd.DataFrame(trial_rows)


def load_bits_raw6_uniform20(csv_path: str | Path, accel_source: str = "acg") -> tuple[np.ndarray | None, dict[str, Any]]:
    acc_t, acc = read_bits_sensor_rows(csv_path, accel_source)
    gyro_t, gyro = read_bits_sensor_rows(csv_path, "gyro")
    n = min(len(acc), len(gyro))
    info = {
        "timestamp_mode": "uniform_20hz_fallback",
        "acc_rows": int(len(acc)),
        "gyro_rows": int(len(gyro)),
        "acc_duplicate_timestamps": count_duplicate_timestamps(acc_t),
        "gyro_duplicate_timestamps": count_duplicate_timestamps(gyro_t),
    }
    if n == 0:
        return None, info
    seq = np.concatenate([acc[:n], gyro[:n]], axis=1).astype(np.float32)
    if seq.shape[1] != 6 or has_nan_inf(seq):
        return None, info
    return seq, info


def read_bits_sensor_rows(csv_path: str | Path, sensor_label: str) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    rows: list[list[float]] = []
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().lower() == "t" or len(row) < 6:
                continue
            if row[-1].strip() != sensor_label:
                continue
            try:
                timestamps.append(float(row[0]))
                rows.append([float(row[1]), float(row[2]), float(row[3])])
            except Exception:
                continue
    return np.asarray(timestamps, dtype=np.float64), np.asarray(rows, dtype=np.float32)


def create_weda_25hz_records(config, window_mode: str = "full_trial") -> tuple[list[dict[str, Any]], pd.DataFrame]:
    weda_dir = Path(config.weda_raw_dir)
    root25 = weda_dir / "dataset" / "25Hz"
    if not root25.exists():
        raise FileNotFoundError(f"Cannot locate WEDA-FALL raw 25Hz folder: {root25}")
    print(f"WEDA raw 25Hz folder: {root25}")

    timestamps = load_weda_fall_timestamps(weda_dir)
    records: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    adl_counts: dict[tuple[str, str], int] = {}

    for trial in find_weda_trials_25hz(weda_dir):
        seq25, time25, trial_info = load_weda_trial_25hz_time_aligned(trial["accel_path"], trial["gyro_path"])
        if seq25 is None or len(seq25) < WINDOW_SIZE or has_nan_inf(seq25):
            continue

        activity_id = trial["activity_id"]
        base_id = trial["base_id"]
        is_fall_trial = activity_id.startswith("F")
        fall_interval = timestamps.get(base_id)
        direction_for_activity = WEDA_FALL_DIRECTIONS.get(activity_id, "other")

        trial_rows.append(
            {
                "dataset": "weda",
                "source_path": str(trial["accel_path"]),
                "subject_id": trial["subject_id"],
                "trial_id": trial["trial_id"],
                "activity_id": activity_id,
                "source_hz": 25.0,
                "target_hz": SAMPLING_RATE,
                "raw_samples": int(trial_info["raw_samples"]),
                "processed_samples": int(len(seq25)),
                "duration_seconds": float(time25[-1] - time25[0]) if len(time25) > 1 else 0.0,
                "effective_fs_before": trial_info["effective_fs_before"],
                "effective_fs_after": effective_fs_from_time(time25),
                "resampling_method": trial_info["resampling_method"],
                "timestamp_irregular": trial_info["timestamp_irregular"],
                "duplicate_timestamps": trial_info["duplicate_timestamps"],
                "fall_interval_start": fall_interval[0] if fall_interval else np.nan,
                "fall_interval_end": fall_interval[1] if fall_interval else np.nan,
            }
        )

        if window_mode == "event_centered" and is_fall_trial:
            if fall_interval is not None:
                interval_mask = (time25 >= fall_interval[0]) & (time25 <= fall_interval[1])
                if interval_mask.any():
                    interval_indices = np.flatnonzero(interval_mask)
                    local = int(np.argmax(compute_acc_mag(seq25[interval_indices])))
                    center_idx = int(interval_indices[local])
                    event_source = "fall_interval_peak_acc_centered"
                else:
                    center_idx = int(np.argmax(compute_acc_mag(seq25)))
                    event_source = "peak_acc_centered_interval_missing_in_time"
            else:
                center_idx = int(np.argmax(compute_acc_mag(seq25)))
                event_source = "peak_acc_centered_no_interval"
            window_specs = [pad_or_crop_window(seq25, center_idx, WINDOW_SIZE)]
            window_rule_for_trial = event_source
        elif window_mode == "event_centered":
            cap = 10 if activity_id in WEDA_HARD_NEGATIVES else config.weda_adl_cap
            cap_key = (trial["subject_id"], activity_id)
            remaining = max(0, cap - adl_counts.get(cap_key, 0))
            if remaining == 0:
                continue
            window_specs = sliding_windows(seq25, window_size=WINDOW_SIZE, stride=STRIDE_SIZE)[:remaining]
            adl_counts[cap_key] = adl_counts.get(cap_key, 0) + len(window_specs)
            window_rule_for_trial = "adl_sliding_capped"
        else:
            window_specs = sliding_windows(seq25, window_size=WINDOW_SIZE, stride=STRIDE_SIZE)
            window_rule_for_trial = "center_time_in_fall_interval" if is_fall_trial else "adl_full_trial_sliding"

        for window, start, end in window_specs:
            if window.shape != (WINDOW_SIZE, 6) or has_nan_inf(window):
                continue
            center_idx = min(len(time25) - 1, start + WINDOW_SIZE // 2)
            center_time = float(time25[center_idx])
            start_time = float(time25[start])
            end_time = float(time25[min(len(time25) - 1, end - 1)])

            if window_mode == "event_centered" and is_fall_trial:
                fall_label = 1
                window_rule = window_rule_for_trial
            elif is_fall_trial:
                if fall_interval is not None:
                    fall_label = int(fall_interval[0] <= center_time <= fall_interval[1])
                    window_rule = "center_time_in_fall_interval"
                else:
                    fall_label = 1
                    window_rule = "trial_label_fallback_no_fall_interval"
            else:
                fall_label = 0
                window_rule = window_rule_for_trial

            direction_label = direction_for_activity if fall_label else "none"
            direction_supervised = bool(fall_label and direction_label in {"forward", "backward", "lateral"})
            records.append(
                build_record(
                    dataset="weda",
                    window=window,
                    source_path=str(trial["accel_path"]),
                    subject_id=trial["subject_id"],
                    trial_id=f"{activity_id}_{trial['trial_id']}",
                    activity_id=activity_id,
                    activity_name=WEDA_ACTIVITY_NAMES.get(activity_id, activity_id),
                    fall_label=fall_label,
                    direction_label=direction_label,
                    direction_supervised=direction_supervised,
                    start_idx=start,
                    end_idx=end,
                    source_folder=str(root25),
                    source_hz=25.0,
                    target_hz=SAMPLING_RATE,
                    resampling_method=trial_info["resampling_method"],
                    window_rule=window_rule,
                    original_length=int(trial_info["raw_samples"]),
                    processed_length=len(seq25),
                    fall_interval_start=fall_interval[0] if fall_interval else np.nan,
                    fall_interval_end=fall_interval[1] if fall_interval else np.nan,
                    unit_conversion="dataset_unit_map",
                    start_time=start_time,
                    end_time=end_time,
                )
            )
    if not records:
        raise ValueError("WEDA 25 Hz preprocessing produced no windows.")
    return records, pd.DataFrame(trial_rows)


def find_weda_trials_25hz(weda_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(weda_dir) / "dataset" / "25Hz"
    trials: list[dict[str, Any]] = []
    for accel_path in sorted(root.glob("*/*_accel.csv")):
        activity_id = accel_path.parent.name
        base = accel_path.name.replace("_accel.csv", "")
        gyro_path = accel_path.with_name(f"{base}_gyro.csv")
        if not gyro_path.exists():
            continue
        subject_id, trial_id = parse_weda_subject_trial(accel_path.name)
        trials.append(
            {
                "dataset": "weda",
                "activity_id": activity_id,
                "subject_id": subject_id,
                "trial_id": trial_id,
                "base_id": f"{activity_id}/{base}",
                "accel_path": accel_path,
                "gyro_path": gyro_path,
            }
        )
    return trials


def load_weda_trial_25hz_time_aligned(
    accel_path: str | Path,
    gyro_path: str | Path,
) -> tuple[np.ndarray | None, np.ndarray, dict[str, Any]]:
    acc_t, acc = read_time_xyz(accel_path, prefix="accel")
    gyro_t, gyro = read_time_xyz(gyro_path, prefix="gyro")
    n_raw = min(len(acc), len(gyro))
    info = {
        "raw_samples": int(n_raw),
        "effective_fs_before": float(np.nanmedian([effective_fs_from_time(acc_t), effective_fs_from_time(gyro_t)])),
        "duplicate_timestamps": int(count_duplicate_timestamps(acc_t) + count_duplicate_timestamps(gyro_t)),
        "timestamp_irregular": bool(is_irregular_timestamp(acc_t) or is_irregular_timestamp(gyro_t)),
        "resampling_method": "raw_25hz_time_aligned_to_uniform_25hz",
    }
    if n_raw == 0:
        return None, np.empty((0,), dtype=np.float64), info

    acc_t, acc = unique_sorted_time_series(acc_t, acc)
    gyro_t, gyro = unique_sorted_time_series(gyro_t, gyro)
    start = max(float(acc_t[0]), float(gyro_t[0]))
    end = min(float(acc_t[-1]), float(gyro_t[-1]))
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        n = min(len(acc), len(gyro))
        time = np.arange(n, dtype=np.float64) / SAMPLING_RATE
        seq = np.concatenate([acc[:n], gyro[:n]], axis=1).astype(np.float32)
        info["resampling_method"] = "raw_25hz_row_order_fallback"
        return np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), time, info

    time = np.arange(start, end + 1e-9, 1.0 / SAMPLING_RATE, dtype=np.float64)
    if len(time) < 2:
        return None, time, info
    acc_interp = np.column_stack([np.interp(time, acc_t, acc[:, i]) for i in range(3)])
    gyro_interp = np.column_stack([np.interp(time, gyro_t, gyro[:, i]) for i in range(3)])
    seq = np.concatenate([acc_interp, gyro_interp], axis=1).astype(np.float32)
    return np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), time, info


def read_time_xyz(path: str | Path, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    time_col = f"{prefix}_time_list"
    cols = [f"{prefix}_x_list", f"{prefix}_y_list", f"{prefix}_z_list"]
    if time_col not in df.columns or not set(cols).issubset(df.columns):
        raise ValueError(f"Missing required {prefix} columns in {path}")
    clean = df[[time_col] + cols].apply(pd.to_numeric, errors="coerce").dropna()
    return clean[time_col].to_numpy(dtype=np.float64), clean[cols].to_numpy(dtype=np.float32)


def unique_sorted_time_series(t: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(t) == 0:
        return t, values
    order = np.argsort(t)
    t_sorted = np.asarray(t[order], dtype=np.float64)
    v_sorted = np.asarray(values[order], dtype=np.float32)
    df = pd.DataFrame(v_sorted, columns=["x", "y", "z"])
    df.insert(0, "t", t_sorted)
    grouped = df.groupby("t", sort=True, as_index=False)[["x", "y", "z"]].mean()
    return grouped["t"].to_numpy(dtype=np.float64), grouped[["x", "y", "z"]].to_numpy(dtype=np.float32)


def resample_uniform_time(seq: np.ndarray, source_hz: float, target_hz: float) -> tuple[np.ndarray, np.ndarray]:
    seq = np.asarray(seq, dtype=np.float32)
    t_old = np.arange(len(seq), dtype=np.float64) / float(source_hz)
    t_new = np.arange(0.0, t_old[-1] + 1e-9, 1.0 / float(target_hz), dtype=np.float64)
    out = np.column_stack([np.interp(t_new, t_old, seq[:, i]) for i in range(seq.shape[1])]).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), t_new


def build_record(
    dataset: str,
    window: np.ndarray,
    source_path: str,
    subject_id: str,
    trial_id: str,
    activity_id: str,
    activity_name: str,
    fall_label: int,
    direction_label: str,
    direction_supervised: bool,
    start_idx: int,
    end_idx: int,
    source_folder: str,
    source_hz: float,
    target_hz: float,
    resampling_method: str,
    window_rule: str,
    original_length: int,
    processed_length: int,
    fall_interval_start: float,
    fall_interval_end: float,
    unit_conversion: str,
    start_time: float | None = None,
    end_time: float | None = None,
) -> dict[str, Any]:
    return {
        "X": np.asarray(window, dtype=np.float32),
        "dataset": dataset,
        "source_path": source_path,
        "source_folder": source_folder,
        "subject_id": subject_id,
        "trial_id": trial_id,
        "activity_id": activity_id,
        "activity_name": activity_name,
        "activity": activity_name,
        "fall_label": int(fall_label),
        "direction_label": direction_label,
        "direction_supervised": bool(direction_supervised),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "start_time": float(start_idx / target_hz) if start_time is None else float(start_time),
        "end_time": float(end_idx / target_hz) if end_time is None else float(end_time),
        "sampling_rate": float(target_hz),
        "window_size": int(WINDOW_SIZE),
        "stride_size": int(STRIDE_SIZE),
        "feature_order": ",".join(FEATURE_ORDER),
        "source_hz": float(source_hz),
        "target_hz": float(target_hz),
        "resampling_method": resampling_method,
        "window_rule": window_rule,
        "original_length": int(original_length),
        "processed_length": int(processed_length),
        "fall_interval_start": fall_interval_start,
        "fall_interval_end": fall_interval_end,
        "unit_conversion": unit_conversion,
        "sensor_position": "wrist",
    }


def records_to_arrays(records: list[dict[str, Any]]) -> tuple[np.ndarray, pd.DataFrame]:
    windows: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for rec in records:
        x = np.asarray(rec.get("X"), dtype=np.float32)
        if x.shape != (WINDOW_SIZE, 6) or has_nan_inf(x):
            continue
        row = dict(rec)
        windows.append(row.pop("X"))
        rows.append(row)
    if not windows:
        raise ValueError("No valid 25 Hz windows after filtering.")
    metadata = pd.DataFrame(rows)
    metadata.insert(0, "window_id", [f"w25_{i:08d}" for i in range(len(metadata))])
    metadata["direction_supervised"] = metadata["direction_supervised"].astype(bool)
    metadata["fall_label"] = metadata["fall_label"].astype(int)
    return np.stack(windows).astype(np.float32), metadata


def apply_saved_subject_split(metadata: pd.DataFrame, split_subjects: dict[str, Any]) -> pd.DataFrame:
    out = metadata.copy()
    subject_to_split: dict[tuple[str, str], str] = {}
    for dataset in ["bits", "weda"]:
        if dataset not in split_subjects:
            raise ValueError(f"{dataset!r} not found in split_subjects.json")
        for split_name, subjects in split_subjects[dataset].items():
            for subject in subjects:
                subject_to_split[(dataset, str(subject))] = str(split_name)
    out["split"] = [
        subject_to_split.get((str(dataset).lower(), str(subject)))
        for dataset, subject in zip(out["dataset"], out["subject_id"])
    ]
    if out["split"].isna().any():
        missing = out.loc[out["split"].isna(), ["dataset", "subject_id"]].drop_duplicates()
        raise ValueError(f"Subjects missing from split_subjects.json:\n{missing.to_string(index=False)}")
    return out


def labels_from_metadata(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_fall = metadata["fall_label"].to_numpy(dtype=np.int64)
    y_direction = metadata["direction_label"].map(DIRECTION_LABEL_MAPPING).fillna(-1).to_numpy(dtype=np.int64)
    direction_mask = metadata["direction_supervised"].astype(np.float32).to_numpy()
    return y_fall, y_direction, direction_mask


def compute_features_25hz(X_raw6: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X_raw6, dtype=np.float32)
    if X.ndim != 3 or X.shape[1:] != (WINDOW_SIZE, 6):
        raise ValueError(f"Expected raw6 shape (N, {WINDOW_SIZE}, 6), got {X.shape}")

    ax = X[:, :, 0]
    ay = X[:, :, 1]
    az = X[:, :, 2]
    gx = X[:, :, 3]
    gy = X[:, :, 4]
    gz = X[:, :, 5]

    dt = 1.0 / SAMPLING_RATE
    acc_mag = np.sqrt(np.maximum(ax * ax + ay * ay + az * az, 0.0) + eps)
    gyro_mag = np.sqrt(np.maximum(gx * gx + gy * gy + gz * gz, 0.0) + eps)
    jerk = np.gradient(acc_mag, dt, axis=1)
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(np.maximum(ay * ay + az * az, 0.0) + eps))
    tilt_arg = np.clip(az / np.maximum(acc_mag, eps), -1.0, 1.0)
    tilt = np.arccos(tilt_arg)
    tilt_delta = np.gradient(tilt, dt, axis=1)

    features = np.concatenate(
        [
            X,
            acc_mag[:, :, None],
            gyro_mag[:, :, None],
            jerk[:, :, None],
            roll[:, :, None],
            pitch[:, :, None],
            tilt_delta[:, :, None],
        ],
        axis=-1,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_scaler(X_train: np.ndarray, eps: float = 1e-6) -> dict[str, Any]:
    mean = X_train.mean(axis=(0, 1)).astype(np.float32)
    std = X_train.std(axis=(0, 1)).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    return {
        "feature_set": FEATURE_SET,
        "feature_order": FEATURE_ORDER,
        "fit_scope": "experiment_train_only",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "eps": float(eps),
    }


def transform_scaler(X: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(scaler["std"], dtype=np.float32).reshape(1, 1, -1)
    out = (np.asarray(X, dtype=np.float32) - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def run_or_load_experiment(
    config,
    artifacts_dir: Path,
    spec: ExperimentSpec,
    data: dict[str, Any],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    monitor: str,
    patience: int,
    force: bool,
) -> dict[str, Any]:
    run_dir = ensure_dir(artifacts_dir / spec.experiment_id)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force:
        print(f"\n=== {spec.experiment_id}: reusing existing metrics ({metrics_path}) ===")
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        row_path = run_dir / "metrics.csv"
        if row_path.exists():
            return pd.read_csv(row_path).iloc[0].to_dict()
        return summary_row_from_metrics(spec, metrics, {}, pd.DataFrame(), None)
    return run_experiment(
        config=config,
        run_dir=run_dir,
        spec=spec,
        data=data,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        monitor=monitor,
        patience=patience,
    )


def run_experiment(
    config,
    run_dir: Path,
    spec: ExperimentSpec,
    data: dict[str, Any],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    monitor: str,
    patience: int,
) -> dict[str, Any]:
    import tensorflow as tf
    from src.models.ds_fall import build_model

    set_seed(config.seed)
    print(f"\n=== {spec.experiment_id}: train={spec.train_dataset}, test={spec.test_dataset} ===")
    metadata = data["metadata"].copy().reset_index(drop=True)
    X_features = data["X_features"]
    y_fall = data["y_fall"]
    y_direction = data["y_direction"]
    direction_mask = data["direction_mask"]

    train_mask, val_mask, test_mask = experiment_masks(metadata, spec)
    assert_no_leakage(metadata, train_mask, val_mask, test_mask, spec)

    scaler = fit_scaler(X_features[train_mask])
    X_scaled = transform_scaler(X_features, scaler)
    X_train, yf_train, yd_train, dm_train = split_arrays(X_scaled, y_fall, y_direction, direction_mask, train_mask)
    X_val, yf_val, yd_val, dm_val = split_arrays(X_scaled, y_fall, y_direction, direction_mask, val_mask)
    X_test, yf_test, yd_test, dm_test = split_arrays(X_scaled, y_fall, y_direction, direction_mask, test_mask)
    meta_train = metadata.loc[train_mask].reset_index(drop=True)
    meta_val = metadata.loc[val_mask].reset_index(drop=True)
    meta_test = metadata.loc[test_mask].reset_index(drop=True)

    direction_class_weights = compute_class_weights(yd_train, num_classes=3, mask=dm_train * (yf_train == 1))
    fall_class_weights = compute_class_weights(yf_train, num_classes=2)

    model = build_model(
        "ds_fall_rd",
        input_shape=(WINDOW_SIZE, len(FEATURE_ORDER)),
        feature_set=FEATURE_SET,
        num_direction_classes=3,
        show_summary=False,
    )
    compile_ds_fall_model(
        model,
        learning_rate=learning_rate,
        direction_loss_type="weighted_ce",
        lambda_fall=1.0,
        lambda_direction=1.5,
        focal_gamma=2.0,
        direction_class_weights=direction_class_weights,
        fall_class_weights=fall_class_weights,
    )

    train_ds = make_tf_dataset(X_train, yf_train, yd_train, dm_train, batch_size=batch_size, shuffle=True, seed=config.seed)
    val_ds = make_tf_dataset(X_val, yf_val, yd_val, dm_val, batch_size=batch_size, shuffle=False, seed=config.seed)
    monitor_mode = "max" if monitor == "val_domain_score" else "min"
    callbacks = []
    if monitor == "val_domain_score":
        callbacks.append(make_validation_metric_callback(X_val, yf_val, yd_val, dm_val))
    callbacks.extend([
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best_model.keras"),
            monitor=monitor,
            mode=monitor_mode,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=monitor_mode,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=max(3, patience // 2),
            min_lr=1e-6,
            verbose=1,
        ),
        make_learning_rate_logger(),
        tf.keras.callbacks.CSVLogger(str(run_dir / "training_history.csv")),
    ])
    start_time = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)
    train_seconds = time.perf_counter() - start_time

    history_df = pd.DataFrame(history.history)
    if "epoch" not in history_df.columns:
        history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(run_dir / "training_history.csv", index=False)
    best_epoch, best_val_loss, best_monitor_value = best_epoch_from_history(history_df, monitor=monitor, mode=monitor_mode)

    model.save(run_dir / "model_final.keras")
    save_json(run_dir / "history.json", history.history)
    save_json(
        run_dir / "model_config.json",
        {
            "model": "A5WCEFW",
            "model_name": "ds_fall_rd",
            "feature_set": FEATURE_SET,
            "input_shape": [WINDOW_SIZE, len(FEATURE_ORDER)],
            "sampling_rate": SAMPLING_RATE,
            "window_seconds": WINDOW_SECONDS,
            "window_size": WINDOW_SIZE,
            "stride_seconds": STRIDE_SECONDS,
            "stride_size": STRIDE_SIZE,
            "direction_loss_type": "weighted_ce",
            "lambda_fall": 1.0,
            "lambda_direction": 1.5,
            "fall_loss_weighted": True,
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_epochs": epochs,
            "early_stopping_patience": patience,
            "reduce_lr_patience": max(3, patience // 2),
            "monitor": monitor,
            "seed": config.seed,
        },
    )
    save_json(
        run_dir / "feature_config.json",
        {
            "feature_set": FEATURE_SET,
            "feature_order": FEATURE_ORDER,
            "sampling_rate": SAMPLING_RATE,
            "dt": 1.0 / SAMPLING_RATE,
            "formulas": {
                "acc_mag": "sqrt(ax**2 + ay**2 + az**2)",
                "gyro_mag": "sqrt(gx**2 + gy**2 + gz**2)",
                "jerk": "gradient(acc_mag, dt)",
                "roll": "atan2(ay, az)",
                "pitch": "atan2(-ax, sqrt(ay**2 + az**2))",
                "tilt_delta": "gradient(arccos(clip(az / max(acc_mag, eps), -1, 1)), dt)",
            },
        },
    )
    save_json(run_dir / "feature_scaler.json", scaler)
    save_pickle(run_dir / "scaler.pkl", scaler)
    save_json(run_dir / "label_mapping.json", {"fall": FALL_LABEL_MAPPING, "direction": DIRECTION_LABEL_MAPPING})
    save_json(
        run_dir / "class_weights.json",
        {"fall": fall_class_weights.tolist(), "direction": direction_class_weights.tolist()},
    )
    save_split_manifest(run_dir, metadata, train_mask, val_mask, test_mask)

    metrics = evaluate_and_save(
        model=model,
        X_test=X_test,
        y_fall=yf_test,
        y_direction=yd_test,
        direction_mask=dm_test,
        meta_test=meta_test,
        run_dir=run_dir,
        experiment_id=spec.experiment_id,
    )
    latency_ms = estimate_latency(model, X_test)
    metrics.update(
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_monitor": monitor,
            "best_monitor_value": best_monitor_value,
            "train_seconds": train_seconds,
            "inference_latency_ms": latency_ms,
            "class_weights": {
                "fall": fall_class_weights.tolist(),
                "direction": direction_class_weights.tolist(),
            },
            "split_counts": {
                "train": int(train_mask.sum()),
                "val": int(val_mask.sum()),
                "test": int(test_mask.sum()),
                "train_direction_supervised": int(((dm_train > 0) & (yd_train >= 0)).sum()),
                "val_direction_supervised": int(((dm_val > 0) & (yd_val >= 0)).sum()),
                "test_direction_supervised": int(((dm_test > 0) & (yd_test >= 0)).sum()),
            },
        }
    )
    save_json(run_dir / "metrics.json", metrics)

    row = summary_row_from_metrics(spec, metrics, {
        "train": int(train_mask.sum()),
        "val": int(val_mask.sum()),
        "test": int(test_mask.sum()),
    }, history_df, model)
    pd.DataFrame([row]).to_csv(run_dir / "metrics.csv", index=False)
    print(
        f"{spec.experiment_id} result: fall_f1={row['fall_f1']:.4f}, "
        f"direction_macro_f1={row['direction_macro_f1']:.4f}, direction_n={row['direction_n_supervised']}"
    )
    return row


def experiment_masks(metadata: pd.DataFrame, spec: ExperimentSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = metadata["dataset"].astype(str).str.lower()
    split = metadata["split"].astype(str).str.lower()
    train_mask = dataset.isin(spec.train_datasets).to_numpy() & split.eq("train").to_numpy()
    val_mask = dataset.isin(spec.val_datasets).to_numpy() & split.eq("val").to_numpy()
    test_mask = dataset.isin(spec.test_datasets).to_numpy() & split.eq("test").to_numpy()
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError(
            f"{spec.experiment_id} empty split: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}"
        )
    return train_mask, val_mask, test_mask


def assert_no_leakage(
    metadata: pd.DataFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    spec: ExperimentSpec,
) -> None:
    train_subjects = set(zip(metadata.loc[train_mask, "dataset"], metadata.loc[train_mask, "subject_id"].astype(str)))
    val_subjects = set(zip(metadata.loc[val_mask, "dataset"], metadata.loc[val_mask, "subject_id"].astype(str)))
    test_subjects = set(zip(metadata.loc[test_mask, "dataset"], metadata.loc[test_mask, "subject_id"].astype(str)))
    if train_subjects & test_subjects:
        raise ValueError(f"{spec.experiment_id} subject leakage train/test: {sorted(train_subjects & test_subjects)[:5]}")
    if train_subjects & val_subjects:
        raise ValueError(f"{spec.experiment_id} subject leakage train/val: {sorted(train_subjects & val_subjects)[:5]}")

    train_trials = set(zip(metadata.loc[train_mask, "dataset"], metadata.loc[train_mask, "subject_id"], metadata.loc[train_mask, "trial_id"]))
    test_trials = set(zip(metadata.loc[test_mask, "dataset"], metadata.loc[test_mask, "subject_id"], metadata.loc[test_mask, "trial_id"]))
    if train_trials & test_trials:
        raise ValueError(f"{spec.experiment_id} trial leakage train/test.")


def split_arrays(
    X: np.ndarray,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        X[mask].astype(np.float32),
        y_fall[mask].astype(np.int64),
        y_direction[mask].astype(np.int64),
        direction_mask[mask].astype(np.float32),
    )


def evaluate_and_save(
    model,
    X_test: np.ndarray,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    meta_test: pd.DataFrame,
    run_dir: Path,
    experiment_id: str,
) -> dict[str, Any]:
    preds = model.predict(X_test, verbose=0)
    if isinstance(preds, dict):
        fall_probs = preds["fall_output"]
        direction_probs = preds["direction_output"]
    else:
        fall_probs, direction_probs = preds
    pred_fall = np.argmax(fall_probs, axis=1)
    pred_direction = np.argmax(direction_probs, axis=1)
    supervised = (direction_mask > 0) & (y_direction >= 0)

    fall_metrics = classification_metrics_binary(y_fall, pred_fall)
    direction_metrics = classification_metrics_direction(y_direction, pred_direction, supervised)
    model_size = model_size_summary(model)

    save_classification_artifacts(run_dir, y_fall, pred_fall, y_direction, pred_direction, supervised)
    save_predictions(run_dir, meta_test, y_fall, pred_fall, fall_probs, y_direction, pred_direction, direction_probs, experiment_id)
    per_dataset_df = per_dataset_metrics(meta_test, y_fall, pred_fall, y_direction, pred_direction, supervised)
    per_dataset_df.to_csv(run_dir / "per_dataset_metrics.csv", index=False)

    metrics = {
        "fall": fall_metrics,
        "direction": direction_metrics,
        "per_dataset": per_dataset_df.to_dict(orient="records"),
        "model_size": model_size,
    }
    return metrics


def classification_metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["non_fall", "fall"],
            zero_division=0,
            output_dict=True,
        ),
    }


def classification_metrics_direction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    supervised: np.ndarray,
) -> dict[str, Any]:
    n = int(supervised.sum())
    if n == 0:
        return {"num_supervised": 0, "accuracy": np.nan, "macro_f1": np.nan}
    yt = y_true[supervised]
    yp = y_pred[supervised]
    return {
        "num_supervised": n,
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(yt, yp, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            yt,
            yp,
            labels=[0, 1, 2],
            target_names=["forward", "backward", "lateral"],
            zero_division=0,
            output_dict=True,
        ),
    }


def save_classification_artifacts(
    run_dir: Path,
    y_fall: np.ndarray,
    pred_fall: np.ndarray,
    y_direction: np.ndarray,
    pred_direction: np.ndarray,
    supervised: np.ndarray,
) -> None:
    fall_report = classification_report(
        y_fall,
        pred_fall,
        labels=[0, 1],
        target_names=["non_fall", "fall"],
        zero_division=0,
    )
    (run_dir / "classification_report_fall.txt").write_text(fall_report, encoding="utf-8")
    pd.DataFrame(
        confusion_matrix(y_fall, pred_fall, labels=[0, 1]),
        index=["true_non_fall", "true_fall"],
        columns=["pred_non_fall", "pred_fall"],
    ).to_csv(run_dir / "confusion_matrix_fall.csv")

    if supervised.sum() > 0:
        direction_report = classification_report(
            y_direction[supervised],
            pred_direction[supervised],
            labels=[0, 1, 2],
            target_names=["forward", "backward", "lateral"],
            zero_division=0,
        )
        direction_cm = confusion_matrix(y_direction[supervised], pred_direction[supervised], labels=[0, 1, 2])
    else:
        direction_report = "No supervised direction samples.\n"
        direction_cm = np.zeros((3, 3), dtype=int)
    (run_dir / "classification_report_direction.txt").write_text(direction_report, encoding="utf-8")
    pd.DataFrame(
        direction_cm,
        index=["true_forward", "true_backward", "true_lateral"],
        columns=["pred_forward", "pred_backward", "pred_lateral"],
    ).to_csv(run_dir / "confusion_matrix_direction.csv")


def save_predictions(
    run_dir: Path,
    meta_test: pd.DataFrame,
    y_fall: np.ndarray,
    pred_fall: np.ndarray,
    fall_probs: np.ndarray,
    y_direction: np.ndarray,
    pred_direction: np.ndarray,
    direction_probs: np.ndarray,
    experiment_id: str,
) -> None:
    out = meta_test.copy().reset_index(drop=True)
    out["true_fall"] = [FALL_NAMES[int(v)] for v in y_fall]
    out["pred_fall"] = [FALL_NAMES[int(v)] for v in pred_fall]
    out["fall_prob"] = fall_probs[:, 1].astype(float)
    out["true_direction"] = [DIRECTION_NAMES[int(v)] for v in y_direction]
    out["pred_direction"] = [DIRECTION_NAMES[int(v)] for v in pred_direction]
    out["direction_confidence"] = direction_probs.max(axis=1).astype(float)
    out["experiment_id"] = experiment_id
    columns = [
        "window_id",
        "dataset",
        "subject_id",
        "trial_id",
        "activity",
        "true_fall",
        "pred_fall",
        "fall_prob",
        "true_direction",
        "pred_direction",
        "direction_confidence",
        "split",
        "experiment_id",
    ]
    out[columns].to_csv(run_dir / "predictions.csv", index=False)


def per_dataset_metrics(
    meta_test: pd.DataFrame,
    y_fall: np.ndarray,
    pred_fall: np.ndarray,
    y_direction: np.ndarray,
    pred_direction: np.ndarray,
    supervised: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, group in meta_test.reset_index().groupby("dataset", sort=True):
        idx = group["index"].to_numpy()
        fall = classification_metrics_binary(y_fall[idx], pred_fall[idx])
        sup = supervised[idx]
        direction = classification_metrics_direction(y_direction[idx], pred_direction[idx], sup)
        rows.append(
            {
                "dataset": dataset,
                "n_test": int(len(idx)),
                "fall_accuracy": fall["accuracy"],
                "fall_precision": fall["precision"],
                "fall_recall": fall["recall"],
                "fall_f1": fall["f1"],
                "direction_n_supervised": direction.get("num_supervised", 0),
                "direction_accuracy": direction.get("accuracy"),
                "direction_macro_f1": direction.get("macro_f1"),
            }
        )
    return pd.DataFrame(rows)


def model_size_summary(model) -> dict[str, Any]:
    params = int(model.count_params())
    return {
        "params": params,
        "model_size_float32_kb": params * 4 / 1024.0,
        "model_size_int8_kb_if_available": params / 1024.0,
    }


def estimate_latency(model, X: np.ndarray, repeats: int = 30) -> float | None:
    if len(X) == 0:
        return None
    sample = X[:1].astype(np.float32)
    for _ in range(3):
        model.predict(sample, verbose=0)
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(sample, verbose=0)
    return float((time.perf_counter() - start) * 1000.0 / repeats)


def make_learning_rate_logger():
    import tensorflow as tf

    class LearningRateLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            lr = self.model.optimizer.learning_rate
            try:
                lr_value = float(tf.keras.backend.get_value(lr))
            except Exception:
                lr_value = float(lr)
            logs["learning_rate"] = lr_value
            print(f" - learning_rate: {lr_value:.6g}")

    return LearningRateLogger()


def make_validation_metric_callback(
    X_val: np.ndarray,
    y_fall_val: np.ndarray,
    y_direction_val: np.ndarray,
    direction_mask_val: np.ndarray,
):
    import tensorflow as tf

    class ValidationMetricCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            preds = self.model.predict(X_val, verbose=0)
            if isinstance(preds, dict):
                fall_probs = preds["fall_output"]
                direction_probs = preds["direction_output"]
            else:
                fall_probs, direction_probs = preds

            fall_pred = np.argmax(fall_probs, axis=1)
            logs["val_fall_f1"] = float(f1_score(y_fall_val, fall_pred, average="binary", zero_division=0))

            supervised = direction_mask_val.astype(bool) & (y_direction_val >= 0)
            if supervised.any():
                direction_pred = np.argmax(direction_probs[supervised], axis=1)
                logs["val_direction_macro_f1"] = float(
                    f1_score(y_direction_val[supervised], direction_pred, average="macro", zero_division=0)
                )
            else:
                logs["val_direction_macro_f1"] = 0.0
            logs["val_domain_score"] = logs["val_fall_f1"] + logs["val_direction_macro_f1"]
            print(
                f" - val_fall_f1: {logs['val_fall_f1']:.4f}"
                f" - val_direction_macro_f1: {logs['val_direction_macro_f1']:.4f}"
                f" - val_domain_score: {logs['val_domain_score']:.4f}"
            )

    return ValidationMetricCallback()


def best_epoch_from_history(
    history_df: pd.DataFrame,
    monitor: str = "val_loss",
    mode: str = "min",
) -> tuple[int | None, float | None, float | None]:
    if history_df.empty:
        return None, None, None
    monitor_col = monitor if monitor in history_df else "val_loss"
    if monitor_col not in history_df:
        return None, None, None
    values = history_df[monitor_col].astype(float)
    idx = int(values.idxmax() if mode == "max" else values.idxmin())
    best_val_loss = float(history_df.loc[idx, "val_loss"]) if "val_loss" in history_df else None
    return int(history_df.loc[idx, "epoch"]), best_val_loss, float(history_df.loc[idx, monitor_col])


def summary_row_from_metrics(
    spec: ExperimentSpec,
    metrics: dict[str, Any],
    counts: dict[str, int],
    history_df: pd.DataFrame,
    model,
) -> dict[str, Any]:
    fall = metrics.get("fall", {})
    direction = metrics.get("direction", {})
    size = metrics.get("model_size", {})
    if model is not None and not size:
        size = model_size_summary(model)
    best_epoch = metrics.get("best_epoch")
    best_val_loss = metrics.get("best_val_loss")
    if best_epoch is None and not history_df.empty:
        best_epoch, best_val_loss, _ = best_epoch_from_history(history_df)
    return {
        "experiment_id": spec.experiment_id,
        "train_dataset": spec.train_dataset,
        "test_dataset": spec.test_dataset,
        "sampling_rate": SAMPLING_RATE,
        "input_shape": f"({WINDOW_SIZE}, {len(FEATURE_ORDER)})",
        "n_train": counts.get("train"),
        "n_val": counts.get("val"),
        "n_test": counts.get("test"),
        "fall_f1": fall.get("f1"),
        "fall_precision": fall.get("precision"),
        "fall_recall": fall.get("recall"),
        "fall_accuracy": fall.get("accuracy"),
        "direction_macro_f1": direction.get("macro_f1"),
        "direction_accuracy": direction.get("accuracy"),
        "direction_n_supervised": direction.get("num_supervised", 0),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "model_params": size.get("params"),
        "model_size_float32_kb": size.get("model_size_float32_kb"),
        "model_size_int8_kb_if_available": size.get("model_size_int8_kb_if_available"),
        "inference_latency_ms": metrics.get("inference_latency_ms"),
    }


def save_split_manifest(run_dir: Path, metadata: pd.DataFrame, train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray) -> None:
    manifest = metadata.copy().reset_index(drop=True)
    experiment_split = np.full(len(manifest), "unused", dtype=object)
    experiment_split[train_mask] = "train"
    experiment_split[val_mask] = "val"
    experiment_split[test_mask] = "test"
    manifest["experiment_split"] = experiment_split
    cols = [
        "window_id",
        "dataset",
        "subject_id",
        "trial_id",
        "activity",
        "fall_label",
        "direction_label",
        "split",
        "experiment_split",
    ]
    manifest.loc[train_mask | val_mask | test_mask, cols].to_csv(run_dir / "split_manifest.csv", index=False)


def save_25hz_dataset_artifacts(data: dict[str, Any], artifacts_dir: Path) -> None:
    metadata = data["metadata"]
    metadata.to_csv(artifacts_dir / "dataset_25hz_metadata.csv", index=False)
    split_cols = ["window_id", "dataset", "subject_id", "trial_id", "activity", "fall_label", "direction_label", "split"]
    metadata[split_cols].to_csv(artifacts_dir / "split_manifest_25hz.csv", index=False)
    data["feature_stats"].to_csv(artifacts_dir / "feature_stats_25hz.csv", index=False)
    data["channel_stats"].to_csv(artifacts_dir / "unit_audit_25hz.csv", index=False)
    data["audit"].to_csv(artifacts_dir / "dataset_audit_25hz.csv", index=False)
    for dataset, trial_df in data.get("trial_audits", {}).items():
        trial_df.to_csv(artifacts_dir / f"{dataset}_trial_audit_25hz.csv", index=False)
    (artifacts_dir / "processing_report_25hz.md").write_text(build_processing_report(data), encoding="utf-8")
    save_json(
        artifacts_dir / "feature_config_25hz.json",
        {
            "feature_set": FEATURE_SET,
            "feature_order": FEATURE_ORDER,
            "sampling_rate": SAMPLING_RATE,
            "window_seconds": WINDOW_SECONDS,
            "window_size": WINDOW_SIZE,
            "stride_seconds": STRIDE_SECONDS,
            "stride_size": STRIDE_SIZE,
        },
    )


def build_audit_report(
    metadata: pd.DataFrame,
    X_raw6: np.ndarray,
    X_features: np.ndarray,
    trial_audits: dict[str, pd.DataFrame],
    unit_summary: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset in ["bits", "weda"]:
        meta = metadata[metadata["dataset"] == dataset]
        trial_df = trial_audits.get(dataset, pd.DataFrame())
        idx = meta.index.to_numpy()
        Xd = X_features[idx]
        direction_counts = meta.loc[meta["direction_label"].isin(["forward", "backward", "lateral"]), "direction_label"].value_counts().to_dict()
        acc_mag = X_raw6[idx, :, :3]
        gyro = X_raw6[idx, :, 3:6]
        acc_mag_flat = np.sqrt(np.sum(acc_mag * acc_mag, axis=-1)).reshape(-1)
        gyro_mag_flat = np.sqrt(np.sum(gyro * gyro, axis=-1)).reshape(-1)
        rows.append(
            {
                "dataset": dataset,
                "source_folder": str(meta["source_folder"].iloc[0]) if not meta.empty else "",
                "original_sampling_rate": float(trial_df["source_hz"].median()) if "source_hz" in trial_df and not trial_df.empty else np.nan,
                "target_sampling_rate": SAMPLING_RATE,
                "resampling_method": sorted(meta["resampling_method"].astype(str).unique().tolist()) if not meta.empty else [],
                "n_subjects": int(meta["subject_id"].nunique()),
                "n_trials": int(meta[["subject_id", "trial_id", "activity_id"]].drop_duplicates().shape[0]),
                "raw_samples": int(trial_df["raw_samples"].sum()) if "raw_samples" in trial_df and not trial_df.empty else np.nan,
                "effective_fs_before_mean": float(trial_df["effective_fs_before"].mean()) if "effective_fs_before" in trial_df and not trial_df.empty else np.nan,
                "effective_fs_after_mean": float(trial_df["effective_fs_after"].mean()) if "effective_fs_after" in trial_df and not trial_df.empty else np.nan,
                "n_windows": int(len(meta)),
                "fall_windows": int((meta["fall_label"] == 1).sum()),
                "non_fall_windows": int((meta["fall_label"] == 0).sum()),
                "direction_supervised_windows": int(meta["direction_supervised"].sum()),
                "direction_class_distribution": json.dumps(direction_counts, ensure_ascii=False),
                "feature_shape": f"{tuple(Xd.shape)}",
                "nan_count": int(np.isnan(Xd).sum()),
                "inf_count": int(np.isinf(Xd).sum()),
                "acc_mag_median": float(np.median(acc_mag_flat)) if len(acc_mag_flat) else np.nan,
                "acc_mag_p95": float(np.percentile(acc_mag_flat, 95)) if len(acc_mag_flat) else np.nan,
                "acc_mag_p99": float(np.percentile(acc_mag_flat, 99)) if len(acc_mag_flat) else np.nan,
                "gyro_mag_median": float(np.median(gyro_mag_flat)) if len(gyro_mag_flat) else np.nan,
                "gyro_mag_p95": float(np.percentile(gyro_mag_flat, 95)) if len(gyro_mag_flat) else np.nan,
                "gyro_mag_p99": float(np.percentile(gyro_mag_flat, 99)) if len(gyro_mag_flat) else np.nan,
                "unit_map": json.dumps(unit_summary.get("dataset_unit_map", {}).get(dataset, {}), ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def validate_25hz_dataset(data: dict[str, Any]) -> None:
    X = data["X_features"]
    meta = data["metadata"]
    if X.ndim != 3 or X.shape[1:] != (WINDOW_SIZE, len(FEATURE_ORDER)):
        raise ValueError(f"Shape mismatch: expected (N, {WINDOW_SIZE}, {len(FEATURE_ORDER)}), got {X.shape}")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("NaN/Inf detected after feature engineering.")
    for dataset in ["bits", "weda"]:
        subset = meta[meta["dataset"] == dataset]
        if subset.empty:
            raise ValueError(f"No windows for dataset={dataset}.")
        if (subset["fall_label"] == 1).sum() == 0:
            raise ValueError(f"No fall windows for dataset={dataset}.")
        if subset["direction_supervised"].sum() == 0:
            raise ValueError(f"No direction-supervised windows for dataset={dataset}.")


def print_dataset_audit(audit: pd.DataFrame) -> None:
    print("\n[3/3] Mandatory dataset audit before training")
    cols = [
        "dataset",
        "source_folder",
        "original_sampling_rate",
        "target_sampling_rate",
        "n_subjects",
        "n_trials",
        "n_windows",
        "fall_windows",
        "non_fall_windows",
        "direction_supervised_windows",
        "acc_mag_median",
        "gyro_mag_p99",
        "feature_shape",
        "nan_count",
        "inf_count",
    ]
    print(audit[cols].to_string(index=False))


def feature_stats_dataframe(X_features: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(FEATURE_ORDER):
        values = X_features[:, :, i].reshape(-1)
        rows.append(
            {
                "feature": name,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
            }
        )
    return pd.DataFrame(rows)


def raw6_channel_stats(metadata: pd.DataFrame, X_raw6: np.ndarray, X_features: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    names = FEATURE_ORDER
    for dataset in ["bits", "weda"]:
        mask = metadata["dataset"].eq(dataset).to_numpy()
        for i, name in enumerate(names):
            values = X_features[mask, :, i].reshape(-1)
            rows.append(
                {
                    "dataset": dataset,
                    "channel": name,
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "median": float(np.median(values)),
                    "p01": float(np.percentile(values, 1)),
                    "p95": float(np.percentile(values, 95)),
                    "p99": float(np.percentile(values, 99)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def build_processing_report(data: dict[str, Any]) -> str:
    lines = [
        "# BITS + WEDA 25 Hz Processing Report",
        "",
        "This audit uses all windows only for data inspection. Each training experiment fits its scaler on that experiment's train split only.",
        "",
        "## Configuration",
        "",
        f"- sampling_rate: {SAMPLING_RATE:g} Hz",
        f"- window_seconds: {WINDOW_SECONDS}",
        f"- window_size: {WINDOW_SIZE}",
        f"- stride_seconds: {STRIDE_SECONDS}",
        f"- stride_size: {STRIDE_SIZE}",
        f"- input_shape: ({WINDOW_SIZE}, {len(FEATURE_ORDER)})",
        f"- window_mode: {data.get('window_mode', 'full_trial')}",
        f"- feature_order: {', '.join(FEATURE_ORDER)}",
        "- WEDA source: raw dataset/25Hz folder, not 50Hz downsampling",
        "- BITS source: raw 20Hz row-order sequence interpolated to 25Hz per trial",
        "",
        "## Dataset Audit",
        "",
        markdown_table(format_float_df(data["audit"])),
        "",
        "## Feature Statistics",
        "",
        markdown_table(format_float_df(data["feature_stats"])),
        "",
        "## Trial Audit Files",
        "",
        "- `bits_trial_audit_25hz.csv`",
        "- `weda_trial_audit_25hz.csv`",
        "",
        "## Unit Notes",
        "",
        "- Final accelerometer unit: m/s^2.",
        "- Final gyroscope unit: rad/s.",
        "- BITS and WEDA use conversion factor 1.0 from the current dataset unit map.",
        "- Unit statistics are saved to `unit_audit_25hz.csv`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def save_summary(rows: list[dict[str, Any]], artifacts_dir: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    cols = summary_columns()
    df[cols].to_csv(artifacts_dir / "summary_25hz_a5wcefw.csv", index=False)
    (artifacts_dir / "summary_25hz_a5wcefw.md").write_text(build_summary_markdown(df[cols]), encoding="utf-8")


def build_summary_markdown(df: pd.DataFrame) -> str:
    lines = [
        "# Summary 25 Hz A5WCEFW",
        "",
        "Model: DS-Fall-RD / A5WCEFW, input shape `(50, 12)`, feature set `tilt12`, task-specific attention, weighted CE fall, weighted CE direction.",
        "",
        "## Results",
        "",
        markdown_table(format_float_df(df)),
        "",
        "## Notes",
        "",
        "- Direction metrics use only samples with `true_direction != -1`.",
        "- Scalers are fit on each experiment's train portion only.",
        "- Cross-dataset tests never use the test dataset for scaler fitting or early stopping.",
        "",
    ]
    return "\n".join(lines)


def summary_columns() -> list[str]:
    return [
        "experiment_id",
        "train_dataset",
        "test_dataset",
        "sampling_rate",
        "input_shape",
        "n_train",
        "n_val",
        "n_test",
        "fall_f1",
        "fall_precision",
        "fall_recall",
        "fall_accuracy",
        "direction_macro_f1",
        "direction_accuracy",
        "direction_n_supervised",
        "best_epoch",
        "best_val_loss",
        "model_params",
        "model_size_float32_kb",
        "model_size_int8_kb_if_available",
        "inference_latency_ms",
    ]


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    compact = df.astype(object).where(pd.notna(df), "")
    header = "| " + " | ".join(map(str, compact.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(compact.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in compact.to_numpy()]
    return "\n".join([header, sep] + rows)


def format_float_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            if col.startswith("n_") or col.endswith("_windows") or col in {"best_epoch", "model_params", "window_size"}:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(int(v)))
            else:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    return out


def count_duplicate_timestamps(t: np.ndarray) -> int:
    if len(t) == 0:
        return 0
    return int(len(t) - len(np.unique(t)))


def effective_fs_from_time(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=np.float64)
    if len(t) < 2 or not np.isfinite(t[0]) or not np.isfinite(t[-1]) or t[-1] <= t[0]:
        return float("nan")
    return float((len(t) - 1) / (t[-1] - t[0]))


def is_irregular_timestamp(t: np.ndarray, expected_dt: float = 1.0 / SAMPLING_RATE) -> bool:
    t = np.asarray(t, dtype=np.float64)
    if len(t) < 3:
        return False
    dt = np.diff(np.sort(t))
    dt = dt[np.isfinite(dt)]
    if len(dt) == 0:
        return False
    return bool(np.nanstd(dt) > expected_dt * 0.25 or np.nanmax(dt) > expected_dt * 2.0)


def bits_activity_name(activity_id: str) -> str:
    return BITS_ADL_ACTIVITY_NAMES.get(activity_id) or BITS_FALL_ACTIVITY_NAMES.get(activity_id, activity_id)


if __name__ == "__main__":
    main()
