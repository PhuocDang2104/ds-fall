from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import compute_acc_mag, has_nan_inf, pad_or_crop_window, sliding_windows


UMAFALL_FALL_DIRECTIONS = {
    "forwardFall": "forward",
    "backwardFall": "backward",
    "lateralFall": "lateral",
}


UMAFALL_ACTIVITY_NAMES = {
    "Aplausing": "applausing",
    "Bending": "bending",
    "GoDownstairs": "go_downstairs",
    "GoUpstairs": "go_upstairs",
    "HandsUp": "hands_up",
    "Hopping": "hopping",
    "Jogging": "jogging",
    "LyingDown_OnABed": "lying_down_on_a_bed",
    "MakingACall": "making_a_call",
    "OpeningDoor": "opening_door",
    "Sitting_GettingUpOnAChair": "sitting_getting_up_on_a_chair",
    "Walking": "walking",
    "backwardFall": "backward_fall",
    "forwardFall": "forward_fall",
    "lateralFall": "lateral_fall",
}


def parse_umafall_filename(path: str | Path) -> dict[str, Any]:
    name = Path(path).name
    match = re.match(
        r"^UMAFall_Subject_(?P<subject>\d+)_(?P<class_dir>ADL|Fall)_(?P<activity>.+)_"
        r"(?P<trial>\d+)_(?P<datetime>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv$",
        name,
    )
    if not match:
        raise ValueError(f"Cannot parse UMAFall filename: {path}")
    parts = match.groupdict()
    return {
        "dataset": "umafall",
        "subject_id": f"subject_{parts['subject']}",
        "class_dir": parts["class_dir"],
        "activity_id": parts["activity"],
        "trial_id": parts["trial"],
        "datetime": parts["datetime"],
        "csv_path": Path(path),
    }


def find_umafall_trials(umafall_dir: str | Path) -> list[dict[str, Any]]:
    umafall_dir = Path(umafall_dir)
    if not umafall_dir.exists():
        return []
    trials: list[dict[str, Any]] = []
    for csv_path in sorted(umafall_dir.glob("UMAFall_Subject_*.csv")):
        try:
            trials.append(parse_umafall_filename(csv_path))
        except ValueError:
            continue
    return trials


def load_umafall_trial(
    csv_path: str | Path,
    sensor_positions: tuple[str, ...] = ("WRIST",),
    target_fs: float = 50.0,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    csv_path = Path(csv_path)
    allowed_positions = {pos.upper() for pos in sensor_positions}
    sensor_positions_by_id: dict[int, str] = {}
    rows: list[tuple[float, float, float, float, int, int]] = []

    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("%"):
                _parse_sensor_position_header(line, sensor_positions_by_id)
                continue
            parts = [part.strip() for part in line.split(";")]
            if len(parts) < 7:
                continue
            try:
                rows.append(
                    (
                        float(parts[0]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                        int(parts[5]),
                        int(parts[6]),
                    )
                )
            except ValueError:
                continue

    selected_ids = [
        sensor_id for sensor_id, position in sensor_positions_by_id.items() if position.upper() in allowed_positions
    ]
    if not selected_ids:
        return None, {"reason": "missing_requested_sensor_position", "sensor_positions": sensor_positions_by_id}

    df = pd.DataFrame(rows, columns=["timestamp_ms", "x", "y", "z", "sensor_type", "sensor_id"])
    if df.empty:
        return None, {"reason": "no_numeric_rows", "sensor_positions": sensor_positions_by_id}

    for sensor_id in selected_ids:
        seq = _resample_sensor_id_to_50hz(df, sensor_id=sensor_id, target_fs=target_fs)
        if seq is not None and len(seq) >= 2 and not has_nan_inf(seq):
            return seq.astype(np.float32), {
                "sensor_id": int(sensor_id),
                "sensor_position": sensor_positions_by_id[sensor_id].lower(),
                "sensor_positions": sensor_positions_by_id,
            }
    return None, {"reason": "missing_acc_or_gyro_for_requested_sensor", "sensor_positions": sensor_positions_by_id}


def _parse_sensor_position_header(line: str, out: dict[int, str]) -> None:
    parts = [part.strip() for part in line[1:].split(";")]
    if len(parts) < 4 or not parts[1].isdigit():
        return
    out[int(parts[1])] = parts[2].upper()


def _resample_sensor_id_to_50hz(df: pd.DataFrame, sensor_id: int, target_fs: float) -> np.ndarray | None:
    acc = _sensor_type_frame(df, sensor_id=sensor_id, sensor_type=0)
    gyro = _sensor_type_frame(df, sensor_id=sensor_id, sensor_type=1)
    if len(acc) < 2 or len(gyro) < 2:
        return None

    start_ms = max(float(acc["timestamp_ms"].iloc[0]), float(gyro["timestamp_ms"].iloc[0]))
    end_ms = min(float(acc["timestamp_ms"].iloc[-1]), float(gyro["timestamp_ms"].iloc[-1]))
    if end_ms <= start_ms:
        return None

    step_ms = 1000.0 / float(target_fs)
    target_t = np.arange(start_ms, end_ms + 1e-6, step_ms, dtype=np.float64)
    if len(target_t) < 2:
        return None

    acc_interp = _interp_xyz(acc, target_t)
    gyro_interp = _interp_xyz(gyro, target_t)
    return np.concatenate([acc_interp, gyro_interp], axis=1).astype(np.float32)


def _sensor_type_frame(df: pd.DataFrame, sensor_id: int, sensor_type: int) -> pd.DataFrame:
    sub = df[(df["sensor_id"] == sensor_id) & (df["sensor_type"] == sensor_type)]
    if sub.empty:
        return sub
    sub = sub[["timestamp_ms", "x", "y", "z"]].sort_values("timestamp_ms")
    # Some UMAFall files contain repeated timestamps; average them before interpolation.
    return sub.groupby("timestamp_ms", as_index=False)[["x", "y", "z"]].mean()


def _interp_xyz(frame: pd.DataFrame, target_t: np.ndarray) -> np.ndarray:
    values = np.zeros((len(target_t), 3), dtype=np.float32)
    source_t = frame["timestamp_ms"].to_numpy(dtype=np.float64)
    for idx, axis in enumerate(["x", "y", "z"]):
        values[:, idx] = np.interp(target_t, source_t, frame[axis].to_numpy(dtype=np.float64)).astype(np.float32)
    return values


def _record(
    window: np.ndarray,
    trial: dict[str, Any],
    sensor_info: dict[str, Any],
    start: int,
    end: int,
    event_source: str,
    direction_label: str,
    direction_supervised: bool,
    original_length: int,
    resampled_length: int,
) -> dict[str, Any]:
    fall_label = int(trial["class_dir"] == "Fall")
    return {
        "X": window.astype(np.float32),
        "dataset": "umafall",
        "source_path": str(trial["csv_path"]),
        "subject_id": trial["subject_id"],
        "activity_id": trial["activity_id"],
        "activity_name": UMAFALL_ACTIVITY_NAMES.get(trial["activity_id"], trial["activity_id"]),
        "fall_label": fall_label,
        "direction_label": direction_label,
        "direction_supervised": direction_supervised,
        "sampling_rate_original": 20.0,
        "sampling_rate_model": 50.0,
        "sensor_position": sensor_info.get("sensor_position", "wrist"),
        "sensor_id": sensor_info.get("sensor_id"),
        "trial_id": trial["trial_id"],
        "window_start_idx": start,
        "window_end_idx": end,
        "event_source": event_source,
        "resampled": True,
        "resample_method": "linear_timestamp_ms",
        "original_length": original_length,
        "resampled_length": resampled_length,
    }


def create_umafall_windows(
    umafall_dir: str | Path,
    window_size: int = 100,
    stride: int = 50,
    adl_cap_per_subject_activity: int = 5,
    sensor_positions: tuple[str, ...] = ("WRIST",),
    target_fs: float = 50.0,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    adl_counts: dict[tuple[str, str], int] = {}

    for trial in find_umafall_trials(umafall_dir):
        seq, sensor_info = load_umafall_trial(
            trial["csv_path"],
            sensor_positions=sensor_positions,
            target_fs=target_fs,
        )
        if seq is None or len(seq) < window_size or has_nan_inf(seq):
            continue

        if trial["class_dir"] == "Fall":
            center = int(np.argmax(compute_acc_mag(seq)))
            window, start, end = pad_or_crop_window(seq, center, window_size)
            direction = UMAFALL_FALL_DIRECTIONS.get(trial["activity_id"], "other")
            if window.shape == (window_size, 6) and not has_nan_inf(window):
                records.append(
                    _record(
                        window,
                        trial,
                        sensor_info,
                        start,
                        end,
                        "peak_acc_after_timestamp_resample",
                        direction,
                        direction in {"forward", "backward", "lateral"},
                        _original_wrist_length(seq, target_fs=target_fs),
                        len(seq),
                    )
                )
        else:
            windows = sliding_windows(seq, window_size=window_size, stride=stride)
            cap_key = (trial["subject_id"], trial["activity_id"])
            remaining = max(0, adl_cap_per_subject_activity - adl_counts.get(cap_key, 0))
            if remaining == 0:
                continue
            for window, start, end in windows[:remaining]:
                if window.shape == (window_size, 6) and not has_nan_inf(window):
                    records.append(
                        _record(
                            window,
                            trial,
                            sensor_info,
                            start,
                            end,
                            "sliding_after_timestamp_resample",
                            "none",
                            False,
                            _original_wrist_length(seq, target_fs=target_fs),
                            len(seq),
                        )
                    )
                    adl_counts[cap_key] = adl_counts.get(cap_key, 0) + 1
    return records


def _original_wrist_length(seq: np.ndarray, target_fs: float) -> int:
    return int(round(len(seq) * 20.0 / float(target_fs)))
