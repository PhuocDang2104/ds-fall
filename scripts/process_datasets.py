from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import make_config
from src.data.preprocessing import (
    build_all_windows,
    normalize_with_train_scaler,
    save_processed_dataset,
)
from src.data.splitting import subject_wise_split
from src.data.units import unit_harmonization_markdown
from src.utils.io import ensure_dir, save_json
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process DS-Fall datasets into data/processed.")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["weda", "bits", "hifd", "umafall"],
        choices=["weda", "bits", "umafall", "hifd"],
        help="Datasets to include. Default uses WEDA, BITS, HIFD, UMAFall.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = make_config(args.project_root)
    config.enabled_datasets = tuple(args.datasets)
    set_seed(config.seed)

    X_raw, metadata, summary = build_all_windows(config)
    if len(X_raw) == 0:
        raise RuntimeError("No windows were created. Check raw dataset paths.")

    metadata, split_subjects = subject_wise_split(
        metadata,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=config.seed,
    )
    X_norm, scaler, norm_stats = normalize_with_train_scaler(X_raw, metadata)
    summary["normalization"] = norm_stats
    summary["enabled_datasets"] = list(config.enabled_datasets)
    summary["umafall_preprocessing"] = "WRIST SensorTag accelerometer+gyroscope, timestamp interpolation to 50Hz"
    summary["bits_preprocessing"] = "20Hz row-order uniform interpolation to 50Hz before 2-second windowing"
    summary["unit_harmonization_note"] = "Dataset-specific unit conversion is applied before train-only normalization."

    save_processed_dataset(
        config.processed_dir,
        X_norm,
        metadata,
        scaler,
        split_subjects,
        summary,
    )
    save_json(config.processed_dir / "process_config.json", {"enabled_datasets": list(config.enabled_datasets)})
    reports_dir = ensure_dir(config.output_dir / "reports")
    (reports_dir / "unit_harmonization_report.md").write_text(
        unit_harmonization_markdown(metadata, summary.get("unit_harmonization", {})),
        encoding="utf-8",
    )
    save_json(reports_dir / "unit_harmonization_report.json", summary.get("unit_harmonization", {}))

    print("Saved processed dataset:", config.processed_dir)
    print("X shape:", X_norm.shape)
    print("Datasets:", metadata["dataset"].value_counts().to_dict())
    print("Splits:", metadata["split"].value_counts().to_dict())
    print("Fall labels:", metadata["fall_label"].value_counts().to_dict())
    print("Direction labels:", metadata["direction_label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
