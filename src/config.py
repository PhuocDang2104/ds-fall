from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


CHANNEL_NAMES = ["ax", "ay", "az", "gx", "gy", "gz"]

FALL_LABEL_MAPPING = {
    "non_fall": 0,
    "fall": 1,
}

DIRECTION_LABEL_MAPPING = {
    "forward": 0,
    "backward": 1,
    "lateral": 2,
    "none": -1,
    "other": -1,
}


def default_dataset_unit_map() -> dict[str, dict[str, float | str]]:
    return {
        "bits": {
            "acc_unit_before": "m/s^2",
            "gyro_unit_before": "rad/s",
            "acc_unit_after": "m/s^2",
            "gyro_unit_after": "rad/s",
            "conversion_factor_acc": 1.0,
            "conversion_factor_gyro": 1.0,
        },
        "weda": {
            "acc_unit_before": "m/s^2",
            "gyro_unit_before": "rad/s",
            "acc_unit_after": "m/s^2",
            "gyro_unit_after": "rad/s",
            "conversion_factor_acc": 1.0,
            "conversion_factor_gyro": 1.0,
        },
        "umafall": {
            "acc_unit_before": "g",
            "gyro_unit_before": "deg/s",
            "acc_unit_after": "m/s^2",
            "gyro_unit_after": "rad/s",
            "conversion_factor_acc": 9.80665,
            "conversion_factor_gyro": 0.017453292519943295,
        },
        "hifd": {
            "acc_unit_before": "m/s^2 (loader converted from g)",
            "gyro_unit_before": "deg/s",
            "acc_unit_after": "m/s^2",
            "gyro_unit_after": "rad/s",
            "conversion_factor_acc": 1.0,
            "conversion_factor_gyro": 0.017453292519943295,
        },
    }


@dataclass
class ProjectConfig:
    project_root: Path
    raw_dir: Path
    processed_dir: Path
    output_dir: Path
    bits_raw_dir: Path
    hifd_raw_dir: Path
    weda_raw_dir: Path
    umafall_raw_dir: Path
    figures_dir: Path
    models_dir: Path
    logs_dir: Path
    metrics_dir: Path

    window_size: int = 100
    stride: int = 50
    model_fs: float = 50.0
    window_seconds: float = 2.0
    seed: int = 42

    weda_adl_cap: int = 5
    hifd_nonfall_cap: int = 5
    bits_adl_cap: int = 5
    umafall_adl_cap: int = 5
    enabled_datasets: tuple[str, ...] = ("weda", "bits", "hifd", "umafall")
    dataset_unit_map: dict[str, dict[str, float | str]] = field(default_factory=default_dataset_unit_map)

    hifd_accel_unit: str = "g"
    convert_g_to_ms2: bool = True

    bits_original_fs: float = 20.0
    bits_target_fs: float = 50.0
    bits_resample_method: str = "linear_index_time"
    bits_accel_source: str = "acg"

    umafall_original_fs: float = 20.0
    umafall_target_fs: float = 50.0
    umafall_resample_method: str = "linear_timestamp_ms"
    umafall_sensor_positions: tuple[str, ...] = ("WRIST",)


def default_project_root() -> Path:
    env_root = os.environ.get("DS_FALL_PROJECT_ROOT")
    if env_root:
        return Path(env_root)

    colab_root = Path("/content/drive/MyDrive/ds-fall")
    if Path("/content").exists():
        return colab_root

    return Path.cwd()


def make_config(project_root: str | Path | None = None) -> ProjectConfig:
    root = Path(project_root) if project_root is not None else default_project_root()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    output_dir = root / "outputs"

    return ProjectConfig(
        project_root=root,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        output_dir=output_dir,
        bits_raw_dir=raw_dir / "Dataset",
        hifd_raw_dir=raw_dir / "HR_IMU_falldetection_dataset-master",
        weda_raw_dir=raw_dir / "WEDA-FALL-main",
        umafall_raw_dir=raw_dir / "UMAFall",
        figures_dir=output_dir / "figures",
        models_dir=output_dir / "models",
        logs_dir=output_dir / "logs",
        metrics_dir=output_dir / "metrics",
    )
