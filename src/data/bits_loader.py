from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import numpy as np

from .common import compute_acc_mag, has_nan_inf, pad_or_crop_window, resample_sequence_uniform, sliding_windows


BITS_ADL_ACTIVITY_NAMES = {
    "adl1": "walking_slowly",
    "adl2": "walking_quickly",
    "adl3": "jogging",
    "adl4": "jumping",
    "adl5": "climbing_up_slowly",
    "adl6": "climbing_down_slowly",
    "adl7": "climbing_up_normally",
    "adl8": "climbing_down_normally",
    "adl9": "slowly_sitting_on_chair",
    "adl10": "rapidly_sitting_on_chair",
    "adl11": "nearly_sitting_and_getting_up",
    "adl12": "swinging_hands",
    "adl13": "lying_on_bed",
    "adl14": "lying_on_back_getting_up_slowly",
    "adl15": "lying_on_back_getting_up_normally",
    "adl16": "lying_sideways_to_back",
}

BITS_FALL_ACTIVITY_NAMES = {
    "fall1": "forward_fall_landing_on_knee",
    "fall2": "right_fall",
    "fall3": "left_fall",
    "fall4": "forward_fall",
    "fall5": "seated_on_bed_falling_on_ground",
    "fall6": "forward_fall_body_weight_on_hand",
    "fall7": "backward_fall_from_seated",
    "fall8": "grabbing_while_falling",
}

BITS_FALL_DIRECTIONS = {
    "fall1": ("forward", True),
    "fall2": ("lateral", True),
    "fall3": ("lateral", True),
    "fall4": ("forward", True),
    "fall5": ("other", False),
    "fall6": ("forward", True),
    "fall7": ("backward", True),
    "fall8": ("other", False),
}


def parse_bits_subject_activity(path: str | Path) -> tuple[str, str, str]:
    path = Path(path)
    subject_id = path.parent.name
    class_dir = path.parent.parent.name
    match = re.search(r"(adl|fall)\d+", path.name)
    if not match:
        # Handles irregular names such as user31_.fall1.csv.
        match = re.search(r"(adl|fall)\D*(\d+)", path.name)
        if not match:
            raise ValueError(f"Cannot parse BITS activity from {path}")
        activity_id = f"{match.group(1)}{match.group(2)}"
    else:
        activity_id = match.group(0)
    return subject_id, class_dir, activity_id


def find_bits_trials(bits_dir: str | Path) -> list[dict[str, Any]]:
    bits_dir = Path(bits_dir)
    if not bits_dir.exists():
        return []
    trials: list[dict[str, Any]] = []
    for csv_path in sorted(bits_dir.glob("*/*/*.csv")):
        try:
            subject_id, class_dir, activity_id = parse_bits_subject_activity(csv_path)
        except ValueError:
            continue
        trials.append(
            {
                "dataset": "bits",
                "subject_id": subject_id,
                "class_dir": class_dir,
                "activity_id": activity_id,
                "csv_path": csv_path,
            }
        )
    return trials


def _try_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _read_sensor_rows(csv_path: Path, sensor_label: str) -> np.ndarray:
    rows: list[list[float]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().lower() == "t":
                continue
            if len(row) < 6:
                continue
            label = row[-1].strip()
            if label != sensor_label:
                continue
            xyz = [_try_float(row[1]), _try_float(row[2]), _try_float(row[3])]
            if any(v is None for v in xyz):
                continue
            rows.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
    return np.asarray(rows, dtype=np.float32)


def parse_bits_csv(csv_path: str | Path, accel_source: str = "acg") -> np.ndarray | None:
    if accel_source not in {"acg", "acc"}:
        raise ValueError("BITS accel_source must be 'acg' or 'acc'")

    csv_path = Path(csv_path)
    acc = _read_sensor_rows(csv_path, accel_source)
    gyro = _read_sensor_rows(csv_path, "gyro")
    n = min(len(acc), len(gyro))
    if n == 0:
        return None
    seq = np.concatenate([acc[:n], gyro[:n]], axis=1).astype(np.float32)
    if seq.shape[1] != 6 or has_nan_inf(seq):
        return None
    return seq


def resample_bits_sequence_to_50hz(
    seq: np.ndarray,
    original_fs: float = 20.0,
    target_fs: float = 50.0,
) -> np.ndarray | None:
    return resample_sequence_uniform(seq, original_fs=original_fs, target_fs=target_fs)


def _activity_name(activity_id: str) -> str:
    return BITS_ADL_ACTIVITY_NAMES.get(activity_id) or BITS_FALL_ACTIVITY_NAMES.get(activity_id, activity_id)


def _record(
    window: np.ndarray,
    trial: dict[str, Any],
    start: int,
    end: int,
    event_source: str,
    direction_label: str,
    direction_supervised: bool,
    original_length: int,
    resampled_length: int,
) -> dict[str, Any]:
    fall_label = int(trial["class_dir"] == "fall")
    return {
        "X": window.astype(np.float32),
        "dataset": "bits",
        "source_path": str(trial["csv_path"]),
        "subject_id": trial["subject_id"],
        "activity_id": trial["activity_id"],
        "activity_name": _activity_name(trial["activity_id"]),
        "fall_label": fall_label,
        "direction_label": direction_label,
        "direction_supervised": direction_supervised,
        "sampling_rate_original": 20.0,
        "sampling_rate_model": 50.0,
        "sensor_position": "left_wrist",
        "window_start_idx": start,
        "window_end_idx": end,
        "event_source": event_source,
        "resampled": True,
        "resample_method": "linear_index_time",
        "original_length": original_length,
        "resampled_length": resampled_length,
    }


def create_bits_windows(
    bits_dir: str | Path,
    window_size: int = 100,
    stride: int = 50,
    adl_cap_per_subject_activity: int = 5,
    accel_source: str = "acg",
    original_fs: float = 20.0,
    target_fs: float = 50.0,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trial in find_bits_trials(bits_dir):
        seq_20hz = parse_bits_csv(trial["csv_path"], accel_source=accel_source)
        if seq_20hz is None or len(seq_20hz) < 2:
            continue
        seq = resample_bits_sequence_to_50hz(seq_20hz, original_fs=original_fs, target_fs=target_fs)
        if seq is None or len(seq) < window_size or has_nan_inf(seq):
            continue

        if trial["class_dir"] == "fall":
            center = int(np.argmax(compute_acc_mag(seq)))
            window, start, end = pad_or_crop_window(seq, center, window_size)
            direction, supervised = BITS_FALL_DIRECTIONS.get(trial["activity_id"], ("other", False))
            if window.shape == (window_size, 6) and not has_nan_inf(window):
                records.append(
                    _record(
                        window,
                        trial,
                        start,
                        end,
                        "peak_acc_after_resample",
                        direction,
                        supervised,
                        len(seq_20hz),
                        len(seq),
                    )
                )
        else:
            windows = sliding_windows(seq, window_size=window_size, stride=stride)
            for window, start, end in windows[:adl_cap_per_subject_activity]:
                if window.shape == (window_size, 6) and not has_nan_inf(window):
                    records.append(
                        _record(
                            window,
                            trial,
                            start,
                            end,
                            "sliding_after_resample",
                            "none",
                            False,
                            len(seq_20hz),
                            len(seq),
                        )
                    )
    return records
