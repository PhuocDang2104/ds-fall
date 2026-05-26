from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from .common import compute_acc_mag, has_nan_inf, pad_or_crop_window, sliding_windows


HIFD_FALL_DIRECTIONS = {
    "fall1": "forward",
    "fall2": "backward",
    "fall3": "lateral",
    "fall4": "forward",
    "fall5": "backward",
    "fall6": "lateral",
}

HIFD_ACTIVITY_NAMES = {
    "fall1": "clockwise_forward_fall",
    "fall2": "clockwise_backward_fall",
    "fall3": "right_to_left_lateral_fall",
    "fall4": "counterclockwise_forward_fall",
    "fall5": "counterclockwise_backward_fall",
    "fall6": "left_to_right_lateral_fall",
    "bed": "lying_down_up_bed",
    "chair": "sitting_down_up",
    "clap": "hitting_sensor",
    "cloth": "wearing_cloth",
    "eat": "eating",
    "hair": "brushing_hair",
    "shoe": "tying_shoelace",
    "stair": "stairs",
    "teeth": "brushing_teeth",
    "walk": "walking",
    "wash": "washing",
    "write": "writing",
    "zip": "zipping",
}


def parse_hifd_subject(path: str | Path) -> str:
    for part in Path(path).parts:
        if part.startswith("subject_"):
            return part
    raise ValueError(f"Cannot parse HIFD subject from {path}")


def find_hifd_trials(hifd_dir: str | Path) -> list[dict[str, Any]]:
    hifd_dir = Path(hifd_dir)
    if not hifd_dir.exists():
        return []
    trials: list[dict[str, Any]] = []
    for mat_path in sorted(hifd_dir.glob("subject_*/*/*.mat")):
        split = mat_path.parent.name
        activity_id = mat_path.stem
        trials.append(
            {
                "dataset": "hifd",
                "subject_id": parse_hifd_subject(mat_path),
                "activity_id": activity_id,
                "class_dir": split,
                "mat_path": mat_path,
            }
        )
    return trials


def _col(data: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(data[key]).reshape(-1).astype(np.float32)


def load_hifd_mat(
    mat_path: str | Path,
    convert_g_to_ms2: bool = True,
) -> np.ndarray | None:
    data = loadmat(mat_path)
    required = ["ax", "ay", "az", "droll", "dpitch", "dyaw"]
    if any(key not in data for key in required):
        return None

    acc = np.stack([_col(data, "ax"), _col(data, "ay"), _col(data, "az")], axis=1)
    if convert_g_to_ms2:
        acc = acc * np.float32(9.80665)
    gyr = np.stack([_col(data, "droll"), _col(data, "dpitch"), _col(data, "dyaw")], axis=1)
    n = min(len(acc), len(gyr))
    if n == 0:
        return None
    seq = np.concatenate([acc[:n], gyr[:n]], axis=1).astype(np.float32)
    if seq.shape[1] != 6 or has_nan_inf(seq):
        return None
    return seq


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
    fall_label = int(trial["class_dir"] == "fall")
    return {
        "X": window.astype(np.float32),
        "dataset": "hifd",
        "source_path": str(trial["mat_path"]),
        "subject_id": trial["subject_id"],
        "activity_id": trial["activity_id"],
        "activity_name": HIFD_ACTIVITY_NAMES.get(trial["activity_id"], trial["activity_id"]),
        "fall_label": fall_label,
        "direction_label": direction_label,
        "direction_supervised": direction_supervised,
        "sampling_rate_original": 50.0,
        "sampling_rate_model": 50.0,
        "sensor_position": "left_wrist",
        "window_start_idx": start,
        "window_end_idx": end,
        "event_source": event_source,
        "resampled": False,
        "resample_method": "none",
        "original_length": original_length,
        "resampled_length": original_length,
    }


def create_hifd_windows(
    hifd_dir: str | Path,
    window_size: int = 100,
    stride: int = 50,
    nonfall_cap_per_subject_activity: int = 5,
    convert_g_to_ms2: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trial in find_hifd_trials(hifd_dir):
        seq = load_hifd_mat(trial["mat_path"], convert_g_to_ms2=convert_g_to_ms2)
        if seq is None or len(seq) < 2:
            continue

        if trial["class_dir"] == "fall":
            center = int(np.argmax(compute_acc_mag(seq)))
            window, start, end = pad_or_crop_window(seq, center, window_size)
            if window.shape == (window_size, 6) and not has_nan_inf(window):
                direction = HIFD_FALL_DIRECTIONS.get(trial["activity_id"], "other")
                records.append(
                    _record(
                        window,
                        trial,
                        start,
                        end,
                        "peak_acc",
                        direction,
                        direction in {"forward", "backward", "lateral"},
                        len(seq),
                    )
                )
        else:
            windows = sliding_windows(seq, window_size=window_size, stride=stride)
            for window, start, end in windows[:nonfall_cap_per_subject_activity]:
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
    return records
