from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import compute_acc_mag, has_nan_inf, pad_or_crop_window, safe_read_csv, sliding_windows


WEDA_FALL_DIRECTIONS = {
    "F01": "forward",
    "F02": "lateral",
    "F03": "backward",
    "F04": "forward",
    "F05": "backward",
    "F06": "forward",
    "F07": "backward",
    "F08": "lateral",
}

WEDA_ACTIVITY_NAMES = {
    "F01": "fall_forward_walking_slip",
    "F02": "lateral_fall_walking_slip",
    "F03": "fall_backward_walking_slip",
    "F04": "fall_forward_walking_trip",
    "F05": "fall_backward_trying_to_sit",
    "F06": "fall_forward_sitting",
    "F07": "fall_backward_sitting",
    "F08": "lateral_fall_sitting",
    "D01": "walking",
    "D02": "jogging",
    "D03": "stairs",
    "D04": "sit_wait_stand",
    "D05": "collapse_into_chair",
    "D06": "crouch_tie_shoes_stand",
    "D07": "stumble_walking",
    "D08": "gentle_jump",
    "D09": "hit_table_with_hand",
    "D10": "clapping_hands",
    "D11": "open_close_door",
}

WEDA_HARD_NEGATIVES = {"D05", "D07", "D08", "D09", "D10"}


def parse_weda_subject_trial(filename: str | Path) -> tuple[str, str]:
    name = Path(filename).name
    match = re.match(r"(?P<subject>U\d+)_R(?P<trial>\d+)_", name)
    if not match:
        raise ValueError(f"Cannot parse WEDA subject/trial from {filename}")
    return match.group("subject"), f"R{match.group('trial')}"


def find_weda_trials(weda_dir: str | Path) -> list[dict[str, Any]]:
    weda_dir = Path(weda_dir)
    root = weda_dir / "dataset" / "50Hz"
    if not root.exists():
        return []

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


def _numeric_columns(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    out = df[cols].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=np.float32)
    return out


def load_weda_trial(accel_path: str | Path, gyro_path: str | Path) -> np.ndarray | None:
    accel_df = safe_read_csv(accel_path)
    gyro_df = safe_read_csv(gyro_path)

    acc_cols = ["accel_x_list", "accel_y_list", "accel_z_list"]
    gyr_cols = ["gyro_x_list", "gyro_y_list", "gyro_z_list"]
    if not set(acc_cols).issubset(accel_df.columns) or not set(gyr_cols).issubset(gyro_df.columns):
        return None

    acc = _numeric_columns(accel_df, acc_cols)
    gyr = _numeric_columns(gyro_df, gyr_cols)
    n = min(len(acc), len(gyr))
    if n == 0:
        return None
    seq = np.concatenate([acc[:n], gyr[:n]], axis=1).astype(np.float32)
    if seq.shape[1] != 6 or has_nan_inf(seq):
        return None
    return seq


def load_weda_fall_timestamps(weda_dir: str | Path) -> dict[str, tuple[float, float]]:
    ts_path = Path(weda_dir) / "dataset" / "fall_timestamps.csv"
    if not ts_path.exists():
        return {}
    df = safe_read_csv(ts_path)
    timestamps: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        try:
            timestamps[str(row["filename"])] = (float(row["start_time"]), float(row["end_time"]))
        except Exception:
            continue
    return timestamps


def _record(
    window: np.ndarray,
    trial: dict[str, Any],
    start: int,
    end: int,
    event_source: str,
    direction_label: str,
    direction_supervised: bool,
    original_length: int,
) -> dict[str, Any]:
    fall_label = int(trial["activity_id"].startswith("F"))
    return {
        "X": window.astype(np.float32),
        "dataset": "weda",
        "source_path": str(trial["accel_path"]),
        "subject_id": trial["subject_id"],
        "activity_id": trial["activity_id"],
        "activity_name": WEDA_ACTIVITY_NAMES.get(trial["activity_id"], trial["activity_id"]),
        "fall_label": fall_label,
        "direction_label": direction_label,
        "direction_supervised": direction_supervised,
        "sampling_rate_original": 50.0,
        "sampling_rate_model": 50.0,
        "sensor_position": "wrist",
        "window_start_idx": start,
        "window_end_idx": end,
        "event_source": event_source,
        "resampled": False,
        "resample_method": "none",
        "original_length": original_length,
        "resampled_length": original_length,
    }


def create_weda_windows(
    weda_dir: str | Path,
    window_size: int = 100,
    stride: int = 50,
    adl_cap_per_subject_activity: int = 5,
    hard_negative_cap: int = 10,
) -> list[dict[str, Any]]:
    trials = find_weda_trials(weda_dir)
    timestamps = load_weda_fall_timestamps(weda_dir)
    records: list[dict[str, Any]] = []
    adl_counts: dict[tuple[str, str], int] = {}

    for trial in trials:
        seq = load_weda_trial(trial["accel_path"], trial["gyro_path"])
        if seq is None or len(seq) < 2:
            continue

        activity_id = trial["activity_id"]
        if activity_id.startswith("F"):
            if trial["base_id"] in timestamps:
                start_time, end_time = timestamps[trial["base_id"]]
                lo = max(0, int(np.floor(start_time * 50.0)))
                hi = min(len(seq), int(np.ceil(end_time * 50.0)))
                if hi <= lo:
                    lo, hi = 0, len(seq)
                local_peak = int(np.argmax(compute_acc_mag(seq[lo:hi])))
                center = lo + local_peak
                event_source = "timestamp_peak_acc"
            else:
                center = int(np.argmax(compute_acc_mag(seq)))
                event_source = "peak_acc"

            window, start, end = pad_or_crop_window(seq, center, window_size)
            if window.shape == (window_size, 6) and not has_nan_inf(window):
                direction = WEDA_FALL_DIRECTIONS.get(activity_id, "other")
                records.append(
                    _record(
                        window,
                        trial,
                        start,
                        end,
                        event_source,
                        direction,
                        direction in {"forward", "backward", "lateral"},
                        len(seq),
                    )
                )
        else:
            windows = sliding_windows(seq, window_size=window_size, stride=stride)
            cap = hard_negative_cap if activity_id in WEDA_HARD_NEGATIVES else adl_cap_per_subject_activity
            cap_key = (trial["subject_id"], activity_id)
            remaining = max(0, cap - adl_counts.get(cap_key, 0))
            if remaining == 0:
                continue
            for window, start, end in windows[:remaining]:
                if window.shape == (window_size, 6) and not has_nan_inf(window):
                    records.append(
                        _record(
                            window,
                            trial,
                            start,
                            end,
                            "sliding",
                            "none",
                            False,
                            len(seq),
                        )
                    )
                    adl_counts[cap_key] = adl_counts.get(cap_key, 0) + 1
    return records
