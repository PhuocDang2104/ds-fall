from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RAW6_CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
DIRECTION_ORDER = ["forward", "backward", "lateral"]
DIRECTION_TO_ID = {"forward": 0, "backward": 1, "lateral": 2}
ORIGINAL_FS_HINTS = {
    "bits": "BITS raw files are treated as 20 Hz and resampled to 50 Hz by row-order interpolation.",
    "umafall": "UMAFall SensorTag wrist streams are treated as about 20 Hz and resampled to 50 Hz by timestamp interpolation.",
    "hifd": "HIFD processed windows are expected at 50 Hz.",
    "weda": "WEDA processed windows are expected at 50 Hz.",
}


@dataclass
class ProcessedData:
    X_raw6: np.ndarray
    metadata: pd.DataFrame
    y_fall: np.ndarray | None
    y_direction: np.ndarray | None
    direction_mask: np.ndarray | None
    channel_order_source: str
    input_note: str


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    reports_dir = ensure_dir(output_dir / "reports")
    figures_dir = ensure_dir(output_dir / "figures" / "dataset_diagnostics")

    data = load_processed_data(processed_dir)
    datasets = list(args.datasets)
    present = sorted(data.metadata["dataset"].dropna().astype(str).unique())
    print("Loaded processed data:", processed_dir)
    print("Present datasets:", present)
    print("Requested datasets:", datasets)
    print("Channel order:", data.channel_order_source)
    print(data.input_note)

    audit_note = (
        "This is a data audit only. It summarizes all available processed windows and does not fit "
        "any DS-Fall model or create train-time scalers."
    )

    all_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    warnings_by_dataset: dict[str, list[str]] = {}

    metrics_by_dataset: dict[str, dict[str, Any]] = {}
    summary_features = compute_summary_features(data.X_raw6)

    for dataset in datasets:
        mask = data.metadata["dataset"].astype(str).eq(dataset).to_numpy()
        ds_meta = data.metadata.loc[mask].reset_index(drop=True)
        ds_X = data.X_raw6[mask]
        ds_features = summary_features.loc[mask].reset_index(drop=True)
        warnings: list[str] = []
        warnings_by_dataset[dataset] = warnings

        if len(ds_meta) == 0:
            warnings.append(f"Dataset {dataset!r} not found in processed metadata.")
            metrics_by_dataset[dataset] = {"missing": True, "warnings": warnings}
            all_rows.append(missing_summary_row(dataset, warnings))
            continue

        basic = basic_counts(dataset, ds_meta)
        gravity = gravity_check(ds_X)
        impact = impact_quality(ds_X, ds_meta)
        sampling = sampling_check(dataset, ds_meta)
        sampling_rows.extend(sampling["rows"])
        channel = channel_stats(dataset, ds_X)
        channel_rows.extend(channel)
        direction_baseline = direction_separability_baseline(
            dataset=dataset,
            features=ds_features,
            metadata=ds_meta,
            seed=args.seed,
        )
        baseline_rows.extend(direction_baseline["rows"])

        warnings.extend(gravity["warnings"])
        warnings.extend(impact["warnings"])
        warnings.extend(sampling["warnings"])
        warnings.extend(direction_baseline["warnings"])

        metrics_by_dataset[dataset] = {
            "missing": False,
            "basic": basic,
            "gravity": gravity,
            "impact": impact,
            "sampling": sampling,
            "direction_baseline": direction_baseline,
            "warnings": warnings,
        }

        all_rows.append(
            {
                "dataset": dataset,
                "n_windows": basic["total_windows"],
                "n_fall": basic["fall_count"],
                "n_nonfall": basic["nonfall_count"],
                "n_direction": basic["supervised_direction_count"],
                "direction_counts": json.dumps(basic["direction_counts"], ensure_ascii=False),
                "acc_mag_median": gravity["acc_mag_median"],
                "gravity_status": gravity["gravity_status"],
                "likely_unit": gravity["likely_unit"],
                "gyro_range_note": gyro_range_note(channel),
                "impact_valid_pct": impact["valid_pct"],
                "direction_baseline_macro_f1": direction_baseline.get("best_macro_f1"),
                "main_warning": "; ".join(warnings[:3]) if warnings else "",
            }
        )

    scale_warnings = cross_dataset_scale_warnings(channel_rows, datasets)
    for warning in scale_warnings:
        dataset = warning.get("dataset")
        if dataset in warnings_by_dataset:
            warnings_by_dataset[dataset].append(warning["warning"])

    summary_df = pd.DataFrame(all_rows)
    if not summary_df.empty and "dataset" in summary_df:
        warning_lookup = {
            dataset: "; ".join(warnings[:3])
            for dataset, warnings in warnings_by_dataset.items()
            if warnings
        }
        summary_df["main_warning"] = summary_df.apply(
            lambda row: warning_lookup.get(str(row["dataset"]), row.get("main_warning", "")),
            axis=1,
        )
    channel_df = pd.DataFrame(channel_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    sampling_df = pd.DataFrame(sampling_rows)

    save_csv(summary_df, reports_dir / "dataset_consistency_summary.csv")
    save_csv(channel_df, reports_dir / "dataset_channel_stats.csv")
    save_csv(baseline_df, reports_dir / "dataset_direction_baseline.csv")
    save_csv(sampling_df, reports_dir / "dataset_sampling_check.csv")

    plot_all_figures(
        X=data.X_raw6,
        metadata=data.metadata,
        datasets=datasets,
        figures_dir=figures_dir,
    )

    report = build_markdown_report(
        data=data,
        datasets=datasets,
        metrics_by_dataset=metrics_by_dataset,
        summary_df=summary_df,
        channel_df=channel_df,
        baseline_df=baseline_df,
        sampling_df=sampling_df,
        scale_warnings=scale_warnings,
        audit_note=audit_note,
    )
    report_path = reports_dir / "dataset_consistency_report.md"
    report_path.write_text(report, encoding="utf-8")

    print("Saved:", report_path)
    print("Saved:", reports_dir / "dataset_consistency_summary.csv")
    print("Saved:", reports_dir / "dataset_channel_stats.csv")
    print("Saved:", reports_dir / "dataset_direction_baseline.csv")
    print("Saved figures under:", figures_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit DS-Fall dataset consistency without training a model.")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--datasets", nargs="+", default=["bits", "hifd", "weda", "umafall"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_processed_data(processed_dir: Path) -> ProcessedData:
    if not processed_dir.exists():
        raise FileNotFoundError(f"processed_dir does not exist: {processed_dir}")

    metadata_path = find_first(processed_dir, ["metadata.csv", "metadata.parquet"])
    if metadata_path is None:
        raise FileNotFoundError(f"Cannot find metadata.csv or metadata.parquet in {processed_dir}")
    metadata = load_metadata(metadata_path)

    if "dataset" not in metadata.columns:
        raise ValueError("metadata must contain a 'dataset' column.")

    X, input_note = load_X(processed_dir)
    if X.ndim != 3 or X.shape[-1] < 6:
        raise ValueError(f"Expected processed X with shape (N, T, >=6), got {X.shape}")
    if len(X) != len(metadata):
        raise ValueError(f"X length {len(X)} does not match metadata length {len(metadata)}")

    scaler = load_scaler(processed_dir / "scaler.pkl")
    channel_order_source = validate_channel_order(processed_dir, scaler, X)
    X_raw6 = X[:, :, :6].astype(np.float32)
    if scaler is not None and scaler.get("channel_names") == RAW6_CHANNELS:
        mean = np.asarray(scaler["mean"], dtype=np.float32).reshape(1, 1, 6)
        std = np.asarray(scaler["std"], dtype=np.float32).reshape(1, 1, 6)
        X_raw6 = X_raw6 * std + mean
        input_note += " X.npy was converted back to raw6 values using scaler.pkl."
    else:
        input_note += " X.npy is treated as raw6 values because no compatible scaler.pkl was found."
    X_raw6 = np.nan_to_num(X_raw6, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    y_fall = load_optional_npy(processed_dir / "y_fall.npy")
    y_direction = load_optional_npy(processed_dir / "y_direction.npy")
    direction_mask = load_optional_npy(processed_dir / "direction_mask.npy")
    return ProcessedData(
        X_raw6=X_raw6,
        metadata=metadata,
        y_fall=y_fall,
        y_direction=y_direction,
        direction_mask=direction_mask,
        channel_order_source=channel_order_source,
        input_note=input_note,
    )


def load_metadata(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_X(processed_dir: Path) -> tuple[np.ndarray, str]:
    npy = processed_dir / "X.npy"
    if npy.exists():
        return np.load(npy).astype(np.float32), f"Loaded X from {npy}."

    for npz_path in sorted(processed_dir.glob("*.npz")):
        npz = np.load(npz_path)
        for key in ["X", "x", "windows", "X_raw", "X_processed"]:
            if key in npz:
                return npz[key].astype(np.float32), f"Loaded X key {key!r} from {npz_path}."
        if len(npz.files) == 1:
            key = npz.files[0]
            return npz[key].astype(np.float32), f"Loaded X key {key!r} from {npz_path}."

    raise FileNotFoundError(f"Cannot find X.npy or a usable .npz file in {processed_dir}")


def validate_channel_order(processed_dir: Path, scaler: dict[str, Any] | None, X: np.ndarray) -> str:
    if scaler is not None and scaler.get("channel_names") == RAW6_CHANNELS:
        return "scaler.pkl channel_names = [ax, ay, az, gx, gy, gz]"
    summary_path = processed_dir / "preprocessing_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            channel_order = summary.get("channel_order")
            if channel_order == RAW6_CHANNELS:
                return "preprocessing_summary.json channel_order = [ax, ay, az, gx, gy, gz]"
        except Exception:
            pass
    if X.shape[-1] == 6 and (processed_dir / "metadata.csv").exists():
        raise ValueError(
            "Channel order is not explicitly available. Refusing to audit because wrong channel order "
            "would invalidate gravity/unit diagnostics. Expected scaler.pkl with channel_names."
        )
    raise ValueError("Channel order is unclear; expected first six channels ax, ay, az, gx, gy, gz.")


def load_scaler(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def load_optional_npy(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def find_first(root: Path, names: list[str]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def basic_counts(dataset: str, metadata: pd.DataFrame) -> dict[str, Any]:
    fall = metadata["fall_label"].astype(int) if "fall_label" in metadata else pd.Series([], dtype=int)
    direction_counts = {}
    if "direction_label" in metadata:
        supervised = supervised_direction_mask(metadata)
        direction_counts = (
            metadata.loc[supervised, "direction_label"]
            .astype(str)
            .value_counts()
            .reindex(DIRECTION_ORDER, fill_value=0)
            .astype(int)
            .to_dict()
        )
    activity_count = "not available"
    for col in ["activity_id", "activity_name", "fall_type"]:
        if col in metadata:
            activity_count = metadata[col].astype(str).value_counts().head(30).to_dict()
            break
    return {
        "dataset": dataset,
        "total_windows": int(len(metadata)),
        "split_counts": metadata["split"].astype(str).value_counts().to_dict() if "split" in metadata else "not available",
        "fall_count": int((fall == 1).sum()) if len(fall) else "not available",
        "nonfall_count": int((fall == 0).sum()) if len(fall) else "not available",
        "supervised_direction_count": int(supervised_direction_mask(metadata).sum()),
        "direction_counts": direction_counts,
        "subject_count": count_unique_or_unavailable(metadata, "subject_id"),
        "trial_count": count_unique_or_unavailable(metadata, "trial_id"),
        "activity_count": activity_count,
    }


def count_unique_or_unavailable(metadata: pd.DataFrame, column: str) -> int | str:
    if column not in metadata:
        return "not available"
    values = metadata[column].dropna()
    if values.empty:
        return "not available"
    return int(values.nunique())


def supervised_direction_mask(metadata: pd.DataFrame) -> np.ndarray:
    if "direction_supervised" in metadata:
        mask = metadata["direction_supervised"].astype(bool).to_numpy()
    elif "direction_label" in metadata:
        mask = metadata["direction_label"].astype(str).isin(DIRECTION_ORDER).to_numpy()
    else:
        mask = np.zeros(len(metadata), dtype=bool)
    if "fall_label" in metadata:
        mask = mask & metadata["fall_label"].astype(int).eq(1).to_numpy()
    if "direction_label" in metadata:
        mask = mask & metadata["direction_label"].astype(str).isin(DIRECTION_ORDER).to_numpy()
    return mask


def sampling_check(dataset: str, metadata: pd.DataFrame) -> dict[str, Any]:
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    target_fs = metadata["sampling_rate_model"].dropna().median() if "sampling_rate_model" in metadata else 50.0
    row = {
        "dataset": dataset,
        "mode": "processed_window",
        "window_length": "not available",
        "target_sampling_rate_assumed": float(target_fs) if pd.notna(target_fs) else 50.0,
        "original_sampling_rate_hint": ORIGINAL_FS_HINTS.get(dataset, "not available"),
        "recordings_checked": 0,
        "fs_mean": np.nan,
        "fs_median": np.nan,
        "fs_std": np.nan,
        "dt_min": np.nan,
        "dt_max": np.nan,
        "duplicate_timestamps": "not available",
        "irregular_timestamps": "not available",
    }
    if {"window_start_idx", "window_end_idx"}.issubset(metadata.columns):
        lengths = metadata["window_end_idx"] - metadata["window_start_idx"]
        row["window_length"] = int(lengths.mode().iloc[0]) if len(lengths.dropna()) else "not available"
    rows.append(row)

    if "source_path" in metadata:
        raw_rows = estimate_raw_sampling_from_sources(dataset, metadata)
        rows.extend(raw_rows)
        if raw_rows:
            seen = set()
            for row_item in raw_rows:
                warning = str(row_item.get("warning") or "")
                if warning and warning not in seen:
                    warnings.append(warning)
                    seen.add(warning)
    raw_sampling_modes = {"raw_timestamp", "raw_timestamp_summary", "raw_timestamp_unreliable"}
    if not any(r["mode"] in raw_sampling_modes for r in rows):
        warnings.append(f"{dataset}: raw timestamp sampling check not available; using processed window metadata only.")
    return {"rows": rows, "warnings": warnings}


def estimate_raw_sampling_from_sources(dataset: str, metadata: pd.DataFrame, max_files: int = 80) -> list[dict[str, Any]]:
    paths = [Path(p) for p in metadata["source_path"].dropna().astype(str).unique()]
    rows = []
    if dataset == "bits":
        return [
            {
                "dataset": dataset,
                "mode": "raw_timestamp_unreliable",
                "source_path": "multiple",
                "window_length": "not applicable",
                "target_sampling_rate_assumed": 50.0,
                "original_sampling_rate_hint": ORIGINAL_FS_HINTS.get(dataset, "not available"),
                "recordings_checked": len(paths),
                "fs_mean": np.nan,
                "fs_median": np.nan,
                "fs_std": np.nan,
                "dt_min": np.nan,
                "dt_max": np.nan,
                "duplicate_timestamps": "not audited",
                "irregular_timestamps": "not audited",
                "warning": (
                    "BITS raw timestamp column is rounded/duplicated in these CSVs; "
                    "current preprocessing intentionally uses configured row-order 20 Hz before resampling to 50 Hz."
                ),
            }
        ]
    if dataset == "umafall":
        for path in paths[:max_files]:
            if path.exists():
                rows.append(umafall_timestamp_stats(path))
    else:
        for path in paths[:max_files]:
            if path.exists() and path.suffix.lower() == ".csv":
                rows.append(generic_csv_timestamp_stats(dataset, path))
    rows = [row for row in rows if row is not None]
    if not rows:
        return []
    valid = [row for row in rows if np.isfinite(row.get("fs_median", np.nan))]
    if valid:
        summary_fs = float(np.nanmedian([r["fs_median"] for r in valid]))
        rows.append(
            {
                "dataset": dataset,
                "mode": "raw_timestamp_summary",
                "source_path": "multiple",
                "window_length": "not applicable",
                "target_sampling_rate_assumed": 50.0,
                "original_sampling_rate_hint": ORIGINAL_FS_HINTS.get(dataset, "not available"),
                "recordings_checked": len(valid),
                "fs_mean": float(np.nanmean([r["fs_mean"] for r in valid])),
                "fs_median": summary_fs,
                "fs_std": float(np.nanstd([r["fs_median"] for r in valid])),
                "dt_min": float(np.nanmin([r["dt_min"] for r in valid])),
                "dt_max": float(np.nanmax([r["dt_max"] for r in valid])),
                "duplicate_timestamps": int(np.nansum([r["duplicate_timestamps"] for r in valid])),
                "irregular_timestamps": int(np.nansum([r["irregular_timestamps"] for r in valid])),
                "warning": sampling_rate_warning(dataset, summary_fs),
            }
        )
    return rows


def umafall_timestamp_stats(path: Path) -> dict[str, Any] | None:
    rows = []
    sensor_positions: dict[int, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("%"):
                parts = [p.strip() for p in line[1:].split(";")]
                if len(parts) >= 3 and parts[1].isdigit():
                    sensor_positions[int(parts[1])] = parts[2].upper()
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 7:
                continue
            try:
                rows.append((float(parts[0]), int(parts[5]), int(parts[6])))
            except ValueError:
                continue
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["timestamp_ms", "sensor_type", "sensor_id"])
    wrist_ids = [sid for sid, pos in sensor_positions.items() if pos == "WRIST"]
    if not wrist_ids:
        return None
    sub = df[(df["sensor_id"].isin(wrist_ids)) & (df["sensor_type"] == 0)]
    return timestamp_stats_from_series("umafall", path, sub["timestamp_ms"].to_numpy(dtype=float), time_unit="ms")


def bits_timestamp_stats(path: Path) -> dict[str, Any] | None:
    times = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().lower() == "t":
                continue
            if len(row) < 6:
                continue
            if row[-1].strip() not in {"acg", "acc"}:
                continue
            try:
                times.append(float(row[0]))
            except ValueError:
                continue
    if not times:
        return None
    # BITS CSV files store large Android-like timestamps. Treat the deltas as
    # nanoseconds for diagnostic purposes, while the processing pipeline still
    # uses row-order 20 Hz interpolation.
    return timestamp_stats_from_series("bits", path, np.asarray(times, dtype=float), time_unit="ns")


def generic_csv_timestamp_stats(dataset: str, path: Path) -> dict[str, Any] | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    candidates = [c for c in df.columns if any(k in str(c).lower() for k in ["time", "timestamp", "millis"])]
    if not candidates:
        return None
    values = pd.to_numeric(df[candidates[0]], errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 3:
        return None
    return timestamp_stats_from_series(dataset, path, values, time_unit="auto")


def timestamp_stats_from_series(dataset: str, path: Path, values: np.ndarray, time_unit: str) -> dict[str, Any] | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return None
    values = np.sort(values)
    duplicate_count = int(pd.Series(values).duplicated().sum())
    unique = np.unique(values)
    dt = np.diff(unique)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return None
    unit = infer_time_unit(dt, time_unit)
    dt_seconds = dt * unit
    median_dt = float(np.median(dt_seconds))
    fs = 1.0 / median_dt if median_dt > 0 else np.nan
    irregular = int(np.sum(np.abs(dt_seconds - median_dt) > max(0.005, 0.25 * median_dt)))
    return {
        "dataset": dataset,
        "mode": "raw_timestamp",
        "source_path": str(path),
        "window_length": "not applicable",
        "target_sampling_rate_assumed": 50.0,
        "original_sampling_rate_hint": ORIGINAL_FS_HINTS.get(dataset, "not available"),
        "recordings_checked": 1,
        "fs_mean": float(1.0 / np.mean(dt_seconds)) if np.mean(dt_seconds) > 0 else np.nan,
        "fs_median": float(fs),
        "fs_std": float(np.std(1.0 / dt_seconds)) if len(dt_seconds) > 1 else 0.0,
        "dt_min": float(np.min(dt_seconds)),
        "dt_max": float(np.max(dt_seconds)),
        "duplicate_timestamps": duplicate_count,
        "irregular_timestamps": irregular,
        "warning": "",
    }


def sampling_rate_warning(dataset: str, fs_median: float) -> str:
    if not np.isfinite(fs_median):
        return ""
    expected = {"umafall": 20.0, "weda": 50.0, "hifd": 50.0}.get(dataset)
    if expected is None:
        return ""
    tolerance = 0.25 * expected
    if abs(fs_median - expected) > tolerance:
        return f"{dataset} raw timestamp median fs={fs_median:.2f} Hz differs from expected/configured {expected:.1f} Hz."
    return ""


def infer_time_unit(dt: np.ndarray, time_unit: str) -> float:
    if time_unit == "ns":
        return 1e-9
    if time_unit == "us":
        return 1e-6
    if time_unit == "ms":
        return 0.001
    median_raw = float(np.median(dt))
    if median_raw > 10.0:
        return 0.001
    if median_raw > 0.5:
        return 1.0
    return 1.0


def gravity_check(X: np.ndarray) -> dict[str, Any]:
    acc_mag = np.linalg.norm(X[:, :, :3], axis=2).reshape(-1)
    stats = quantile_stats(acc_mag, prefix="acc_mag")
    median = stats["acc_mag_median"]
    pct_g = percentage_between(acc_mag, 0.7, 1.3)
    pct_ms2 = percentage_between(acc_mag, 7.0, 12.5)
    pct_lt_04 = percentage_lt(acc_mag, 0.4)
    pct_lt_4 = percentage_lt(acc_mag, 4.0)

    gravity_status = "unclear"
    likely_unit = "unclear"
    confidence_note = "acc_mag median does not match common gravity ranges."
    if 0.7 <= median <= 1.3:
        gravity_status = "likely contains gravity"
        likely_unit = "g"
        confidence_note = "Median acceleration magnitude is near 1 g."
    elif 7.0 <= median <= 12.5:
        gravity_status = "likely contains gravity"
        likely_unit = "m/s^2"
        confidence_note = "Median acceleration magnitude is near 9.81 m/s^2."
    elif median < 0.4:
        gravity_status = "likely gravity-removed"
        likely_unit = "g-like"
        confidence_note = "Median acceleration magnitude is below 0.4."
    elif median < 4.0:
        gravity_status = "likely gravity-removed"
        likely_unit = "m/s^2-like"
        confidence_note = "Median acceleration magnitude is below 4.0."

    warnings = []
    if gravity_status == "unclear":
        warnings.append(f"Gravity status unclear from acc_mag median={median:.4f}.")
    return {
        **stats,
        "pct_acc_mag_0p7_1p3": pct_g,
        "pct_acc_mag_7p0_12p5": pct_ms2,
        "pct_acc_mag_lt_0p4": pct_lt_04,
        "pct_acc_mag_lt_4p0": pct_lt_4,
        "gravity_status": gravity_status,
        "likely_unit": likely_unit,
        "confidence_note": confidence_note,
        "warnings": warnings,
    }


def quantile_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {f"{prefix}_{name}": np.nan for name in ["mean", "median", "std", "p05", "p25", "p75", "p95", "p99"]}
    qs = np.percentile(values, [5, 25, 75, 95, 99])
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p05": float(qs[0]),
        f"{prefix}_p25": float(qs[1]),
        f"{prefix}_p75": float(qs[2]),
        f"{prefix}_p95": float(qs[3]),
        f"{prefix}_p99": float(qs[4]),
    }


def percentage_between(values: np.ndarray, lo: float, hi: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(100.0 * np.mean((values >= lo) & (values <= hi)))


def percentage_lt(values: np.ndarray, threshold: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(100.0 * np.mean(values < threshold))


def channel_stats(dataset: str, X: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, channel in enumerate(RAW6_CHANNELS):
        values = X[:, :, idx].reshape(-1)
        values = values[np.isfinite(values)]
        q = np.percentile(values, [1, 99]) if len(values) else [np.nan, np.nan]
        rows.append(
            {
                "dataset": dataset,
                "channel": channel,
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "std": float(np.std(values)) if len(values) else np.nan,
                "median": float(np.median(values)) if len(values) else np.nan,
                "p01": float(q[0]),
                "p99": float(q[1]),
                "min": float(np.min(values)) if len(values) else np.nan,
                "max": float(np.max(values)) if len(values) else np.nan,
                "abs_p99": float(max(abs(q[0]), abs(q[1]))) if len(values) else np.nan,
                "range": float(np.max(values) - np.min(values)) if len(values) else np.nan,
            }
        )
    return rows


def gyro_range_note(channel_rows: list[dict[str, Any]]) -> str:
    gyro_rows = [r for r in channel_rows if r["channel"] in {"gx", "gy", "gz"}]
    if not gyro_rows:
        return "not available"
    gyro_abs_p99 = max(float(r.get("abs_p99", np.nan)) for r in gyro_rows)
    if gyro_abs_p99 > 20.0:
        return "gyro scale looks like deg/s or high-range arbitrary units"
    if gyro_abs_p99 < 3.5:
        return "gyro scale could be rad/s"
    return "gyro scale unclear/intermediate"


def cross_dataset_scale_warnings(channel_rows: list[dict[str, Any]], datasets: list[str]) -> list[dict[str, str]]:
    df = pd.DataFrame(channel_rows)
    warnings = []
    if df.empty:
        return warnings
    for family, channels in {"acc": ["ax", "ay", "az"], "gyro": ["gx", "gy", "gz"]}.items():
        scale = (
            df[df["channel"].isin(channels)]
            .groupby("dataset")["abs_p99"]
            .median()
            .dropna()
        )
        if len(scale) < 2:
            continue
        min_scale = float(scale.min())
        max_scale = float(scale.max())
        if min_scale <= 0:
            continue
        if max_scale / min_scale > 5.0:
            for dataset, value in scale.items():
                participates = (value / min_scale > 5.0) or (max_scale / value > 5.0)
                if participates:
                    warnings.append(
                        {
                            "dataset": str(dataset),
                            "warning": (
                                f"{family} scale differs by >5x across datasets; "
                                f"{dataset} median abs_p99={value:.4f}, cross-dataset min={min_scale:.4f}, max={max_scale:.4f}"
                            ),
                        }
                    )
        if family == "gyro":
            rad_like = scale[scale < 3.5].index.tolist()
            deg_like = scale[scale > 20.0].index.tolist()
            if rad_like and deg_like:
                for dataset in rad_like + deg_like:
                    warnings.append(
                        {
                            "dataset": str(dataset),
                            "warning": f"gyro unit mismatch suspected: rad/s-like datasets={rad_like}, deg/s-like datasets={deg_like}",
                        }
                    )
    return warnings


def compute_summary_features(X: np.ndarray, fs: float = 50.0) -> pd.DataFrame:
    acc = X[:, :, :3]
    gyro = X[:, :, 3:6]
    acc_mag = np.linalg.norm(acc, axis=2)
    gyro_mag = np.linalg.norm(gyro, axis=2)
    jerk = np.diff(acc_mag, axis=1, prepend=acc_mag[:, :1]) * float(fs)
    roll = np.unwrap(np.arctan2(acc[:, :, 1], acc[:, :, 2]), axis=1)
    pitch = np.unwrap(
        np.arctan2(-acc[:, :, 0], np.sqrt(acc[:, :, 1] ** 2 + acc[:, :, 2] ** 2 + 1e-8)),
        axis=1,
    )
    pre = slice(0, min(20, X.shape[1]))
    post = slice(max(0, X.shape[1] - 20), X.shape[1])

    rows = {
        "max_acc_mag": acc_mag.max(axis=1),
        "max_gyro_mag": gyro_mag.max(axis=1),
        "max_jerk": np.abs(jerk).max(axis=1),
        "mean_ax_pre": acc[:, pre, 0].mean(axis=1),
        "mean_ay_pre": acc[:, pre, 1].mean(axis=1),
        "mean_az_pre": acc[:, pre, 2].mean(axis=1),
        "mean_ax_post": acc[:, post, 0].mean(axis=1),
        "mean_ay_post": acc[:, post, 1].mean(axis=1),
        "mean_az_post": acc[:, post, 2].mean(axis=1),
        "mean_gx_pre": gyro[:, pre, 0].mean(axis=1),
        "mean_gy_pre": gyro[:, pre, 1].mean(axis=1),
        "mean_gz_pre": gyro[:, pre, 2].mean(axis=1),
        "mean_gx_post": gyro[:, post, 0].mean(axis=1),
        "mean_gy_post": gyro[:, post, 1].mean(axis=1),
        "mean_gz_post": gyro[:, post, 2].mean(axis=1),
        "roll_mean_pre": roll[:, pre].mean(axis=1),
        "roll_mean_post": roll[:, post].mean(axis=1),
        "pitch_mean_pre": pitch[:, pre].mean(axis=1),
        "pitch_mean_post": pitch[:, post].mean(axis=1),
        "delta_roll_window": roll[:, post].mean(axis=1) - roll[:, pre].mean(axis=1),
        "delta_pitch_window": pitch[:, post].mean(axis=1) - pitch[:, pre].mean(axis=1),
        "peak_signed_gx": signed_peak(gyro[:, :, 0]),
        "peak_signed_gy": signed_peak(gyro[:, :, 1]),
        "peak_signed_gz": signed_peak(gyro[:, :, 2]),
        "acc_mag_median_window": np.median(acc_mag, axis=1),
        "acc_mag_peak": acc_mag.max(axis=1),
        "gyro_mag_peak": gyro_mag.max(axis=1),
        "impact_index": np.argmax(acc_mag, axis=1),
    }
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def signed_peak(values: np.ndarray) -> np.ndarray:
    idx = np.argmax(np.abs(values), axis=1)
    return values[np.arange(values.shape[0]), idx]


def impact_quality(X: np.ndarray, metadata: pd.DataFrame) -> dict[str, Any]:
    fall_mask = metadata["fall_label"].astype(int).eq(1).to_numpy() if "fall_label" in metadata else np.zeros(len(metadata), dtype=bool)
    if not fall_mask.any():
        return {
            "num_fall": 0,
            "acc_mag_peak_mean": np.nan,
            "gyro_mag_peak_mean": np.nan,
            "jerk_peak_mean": np.nan,
            "impact_index_mean": np.nan,
            "impact_index_median": np.nan,
            "early_pct": np.nan,
            "valid_pct": np.nan,
            "late_pct": np.nan,
            "warnings": ["No fall windows available for impact check."],
        }
    Xf = X[fall_mask]
    acc_mag = np.linalg.norm(Xf[:, :, :3], axis=2)
    gyro_mag = np.linalg.norm(Xf[:, :, 3:6], axis=2)
    jerk = np.diff(acc_mag, axis=1, prepend=acc_mag[:, :1]) * 50.0
    impact_idx = np.argmax(acc_mag, axis=1)
    warnings = []
    early_pct = float(100.0 * np.mean(impact_idx < 20))
    valid_pct = float(100.0 * np.mean((impact_idx >= 20) & (impact_idx <= 80)))
    late_pct = float(100.0 * np.mean(impact_idx > 80))
    if valid_pct < 80.0:
        warnings.append(f"Poor impact alignment: only {valid_pct:.1f}% impact_index in [20, 80].")
    return {
        "num_fall": int(fall_mask.sum()),
        "acc_mag_peak_mean": float(acc_mag.max(axis=1).mean()),
        "acc_mag_peak_median": float(np.median(acc_mag.max(axis=1))),
        "gyro_mag_peak_mean": float(gyro_mag.max(axis=1).mean()),
        "gyro_mag_peak_median": float(np.median(gyro_mag.max(axis=1))),
        "jerk_peak_mean": float(np.abs(jerk).max(axis=1).mean()),
        "jerk_peak_median": float(np.median(np.abs(jerk).max(axis=1))),
        "impact_index_mean": float(np.mean(impact_idx)),
        "impact_index_median": float(np.median(impact_idx)),
        "early_pct": early_pct,
        "valid_pct": valid_pct,
        "late_pct": late_pct,
        "warnings": warnings,
    }


def direction_separability_baseline(
    dataset: str,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    mask = supervised_direction_mask(metadata)
    n = int(mask.sum())
    warnings = []
    rows = []
    result: dict[str, Any] = {"rows": rows, "warnings": warnings, "best_macro_f1": np.nan}
    if n < 30:
        warnings.append(f"{dataset}: direction supervised < 30 ({n}); baseline is unstable or skipped.")
        return result
    y = metadata.loc[mask, "direction_label"].astype(str).map(DIRECTION_TO_ID).to_numpy()
    X = features.loc[mask].drop(columns=["acc_mag_median_window", "acc_mag_peak", "gyro_mag_peak", "impact_index"], errors="ignore").to_numpy(dtype=np.float32)
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2 or counts.min() < 2:
        warnings.append(f"{dataset}: not enough per-class direction samples for stratified split.")
        return result
    test_size = 0.30
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
    except ValueError as exc:
        warnings.append(f"{dataset}: stratified split failed: {exc}")
        return result

    classifiers = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        ),
    }
    best = -1.0
    for name, clf in classifiers.items():
        try:
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            acc = float(accuracy_score(y_test, pred))
            macro = float(f1_score(y_test, pred, average="macro", zero_division=0))
            best = max(best, macro)
            rows.append(
                {
                    "dataset": dataset,
                    "classifier": name,
                    "n_supervised": n,
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    "accuracy": acc,
                    "macro_f1": macro,
                }
            )
        except Exception as exc:
            warnings.append(f"{dataset}: {name} baseline failed: {exc}")
    if best >= 0:
        result["best_macro_f1"] = best
        if best < 0.45:
            warnings.append(f"{dataset}: poor direction separability, best baseline macro F1={best:.3f}.")
    return result


def plot_all_figures(X: np.ndarray, metadata: pd.DataFrame, datasets: list[str], figures_dir: Path) -> None:
    ensure_dir(figures_dir)
    features = compute_summary_features(X)
    plot_acc_mag_median_hist(features, metadata, datasets, figures_dir / "hist_acc_mag_median_by_dataset.png")
    plot_boxplot_feature(features, metadata, datasets, "acc_mag_peak", figures_dir / "boxplot_acc_mag_peak_by_dataset.png")
    plot_boxplot_feature(features, metadata, datasets, "gyro_mag_peak", figures_dir / "boxplot_gyro_mag_peak_by_dataset.png")
    plot_hist_feature(features, metadata, datasets, "impact_index", figures_dir / "hist_impact_index_by_dataset.png", fall_only=True)

    for dataset in datasets:
        mask = metadata["dataset"].astype(str).eq(dataset).to_numpy()
        if not mask.any():
            continue
        plot_mean_std_by_direction(
            X[mask],
            metadata.loc[mask].reset_index(drop=True),
            dataset,
            "acc_mag",
            figures_dir / f"{dataset}_mean_std_acc_mag_by_direction.png",
        )
        plot_mean_std_by_direction(
            X[mask],
            metadata.loc[mask].reset_index(drop=True),
            dataset,
            "gyro_mag",
            figures_dir / f"{dataset}_mean_std_gyro_mag_by_direction.png",
        )
        plot_roll_pitch_by_direction(
            X[mask],
            metadata.loc[mask].reset_index(drop=True),
            dataset,
            figures_dir / f"{dataset}_mean_roll_pitch_by_direction.png",
        )
        plot_gyro_axes_by_direction(
            X[mask],
            metadata.loc[mask].reset_index(drop=True),
            dataset,
            figures_dir / f"{dataset}_mean_gx_gy_gz_by_direction.png",
        )


def direction_labels(metadata: pd.DataFrame) -> pd.Series:
    if "direction_label" not in metadata:
        return pd.Series(["none"] * len(metadata))
    labels = metadata["direction_label"].astype(str).copy()
    labels = labels.where(supervised_direction_mask(metadata), "none")
    return labels


def plot_mean_std_by_direction(X: np.ndarray, metadata: pd.DataFrame, dataset: str, quantity: str, path: Path) -> None:
    labels = direction_labels(metadata)
    if not labels.isin(DIRECTION_ORDER).any():
        return
    if quantity == "acc_mag":
        values = np.linalg.norm(X[:, :, :3], axis=2)
        ylabel = "acc_mag"
    else:
        values = np.linalg.norm(X[:, :, 3:6], axis=2)
        ylabel = "gyro_mag"
    t = np.arange(X.shape[1]) / 50.0
    fig, ax = plt.subplots(figsize=(10, 4))
    for direction in DIRECTION_ORDER:
        mask = labels.eq(direction).to_numpy()
        if not mask.any():
            continue
        mean = values[mask].mean(axis=0)
        std = values[mask].std(axis=0)
        ax.plot(t, mean, label=direction)
        ax.fill_between(t, mean - std, mean + std, alpha=0.15)
    ax.set_title(f"{dataset} mean +/- std {quantity} by direction")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roll_pitch_by_direction(X: np.ndarray, metadata: pd.DataFrame, dataset: str, path: Path) -> None:
    labels = direction_labels(metadata)
    if not labels.isin(DIRECTION_ORDER).any():
        return
    acc = X[:, :, :3]
    roll = np.unwrap(np.arctan2(acc[:, :, 1], acc[:, :, 2]), axis=1)
    pitch = np.unwrap(np.arctan2(-acc[:, :, 0], np.sqrt(acc[:, :, 1] ** 2 + acc[:, :, 2] ** 2 + 1e-8)), axis=1)
    t = np.arange(X.shape[1]) / 50.0
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, values, name in [(axes[0], roll, "roll"), (axes[1], pitch, "pitch")]:
        for direction in DIRECTION_ORDER:
            mask = labels.eq(direction).to_numpy()
            if not mask.any():
                continue
            ax.plot(t, values[mask].mean(axis=0), label=direction)
        ax.set_title(f"{dataset} mean {name} by direction")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gyro_axes_by_direction(X: np.ndarray, metadata: pd.DataFrame, dataset: str, path: Path) -> None:
    labels = direction_labels(metadata)
    if not labels.isin(DIRECTION_ORDER).any():
        return
    t = np.arange(X.shape[1]) / 50.0
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis_idx, channel in enumerate(["gx", "gy", "gz"]):
        values = X[:, :, 3 + axis_idx]
        ax = axes[axis_idx]
        for direction in DIRECTION_ORDER:
            mask = labels.eq(direction).to_numpy()
            if not mask.any():
                continue
            ax.plot(t, values[mask].mean(axis=0), label=direction)
        ax.set_title(f"{dataset} mean {channel} by direction")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_acc_mag_median_hist(features: pd.DataFrame, metadata: pd.DataFrame, datasets: list[str], path: Path) -> None:
    df = features[["acc_mag_median_window"]].copy()
    df["dataset"] = metadata["dataset"].astype(str).to_numpy()
    df = df[df["dataset"].isin(datasets)]
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x="acc_mag_median_window", hue="dataset", bins=40, element="step", stat="density", common_norm=False, ax=ax)
    ax.set_title("Histogram of per-window median acc_mag by dataset")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot_feature(features: pd.DataFrame, metadata: pd.DataFrame, datasets: list[str], feature: str, path: Path) -> None:
    df = features[[feature]].copy()
    df["dataset"] = metadata["dataset"].astype(str).to_numpy()
    df = df[df["dataset"].isin(datasets)]
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="dataset", y=feature, ax=ax)
    ax.set_title(f"Boxplot {feature} by dataset")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hist_feature(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    datasets: list[str],
    feature: str,
    path: Path,
    fall_only: bool = False,
) -> None:
    df = features[[feature]].copy()
    df["dataset"] = metadata["dataset"].astype(str).to_numpy()
    if fall_only and "fall_label" in metadata:
        df = df[metadata["fall_label"].astype(int).eq(1).to_numpy()]
    df = df[df["dataset"].isin(datasets)]
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x=feature, hue="dataset", bins=30, element="step", stat="count", common_norm=False, ax=ax)
    ax.set_title(f"Histogram {feature} by dataset")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_markdown_report(
    data: ProcessedData,
    datasets: list[str],
    metrics_by_dataset: dict[str, dict[str, Any]],
    summary_df: pd.DataFrame,
    channel_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    sampling_df: pd.DataFrame,
    scale_warnings: list[dict[str, str]],
    audit_note: str,
) -> str:
    lines = [
        "# Dataset Consistency Diagnostics",
        "",
        audit_note,
        "",
        f"- Processed windows: {len(data.X_raw6)}",
        f"- Window shape: {tuple(data.X_raw6.shape[1:])}",
        f"- Channel order source: {data.channel_order_source}",
        f"- Input note: {data.input_note}",
        "",
    ]

    for dataset in datasets:
        item = metrics_by_dataset.get(dataset, {"missing": True, "warnings": ["not processed"]})
        lines.extend([f"## {dataset}", ""])
        if item.get("missing"):
            lines.extend([f"- Missing from processed metadata.", f"- Warnings: {item.get('warnings')}", ""])
            continue
        basic = item["basic"]
        gravity = item["gravity"]
        impact = item["impact"]
        sampling = item["sampling"]
        baseline = item["direction_baseline"]
        lines.extend(
            [
                "### Basic Count",
                "",
                f"- Total windows: {basic['total_windows']}",
                f"- Split counts: {basic['split_counts']}",
                f"- Fall / non-fall: {basic['fall_count']} / {basic['nonfall_count']}",
                f"- Supervised direction count: {basic['supervised_direction_count']}",
                f"- Direction class count: {basic['direction_counts']}",
                f"- Subject count: {basic['subject_count']}",
                f"- Trial count: {basic['trial_count']}",
                f"- Activity/fall type count: {basic['activity_count']}",
                "",
                "### Sampling Check",
                "",
                f"- Original sampling note: {ORIGINAL_FS_HINTS.get(dataset, 'not available')}",
                f"- Sampling rows written to `dataset_sampling_check.csv`: {len(sampling['rows'])}",
                "",
                "### Gravity Check",
                "",
                f"- acc_mag mean/median/std: {gravity['acc_mag_mean']:.4f} / {gravity['acc_mag_median']:.4f} / {gravity['acc_mag_std']:.4f}",
                f"- acc_mag p05/p25/p75/p95/p99: {gravity['acc_mag_p05']:.4f} / {gravity['acc_mag_p25']:.4f} / {gravity['acc_mag_p75']:.4f} / {gravity['acc_mag_p95']:.4f} / {gravity['acc_mag_p99']:.4f}",
                f"- % in [0.7, 1.3]: {gravity['pct_acc_mag_0p7_1p3']:.2f}",
                f"- % in [7.0, 12.5]: {gravity['pct_acc_mag_7p0_12p5']:.2f}",
                f"- % < 0.4: {gravity['pct_acc_mag_lt_0p4']:.2f}",
                f"- % < 4.0: {gravity['pct_acc_mag_lt_4p0']:.2f}",
                f"- Conclusion: {gravity['gravity_status']}, likely_unit={gravity['likely_unit']}. {gravity['confidence_note']}",
                "",
                "### Impact/Window Quality",
                "",
                f"- acc_mag peak mean/median: {impact['acc_mag_peak_mean']:.4f} / {impact['acc_mag_peak_median']:.4f}",
                f"- gyro_mag peak mean/median: {impact['gyro_mag_peak_mean']:.4f} / {impact['gyro_mag_peak_median']:.4f}",
                f"- jerk peak mean/median: {impact['jerk_peak_mean']:.4f} / {impact['jerk_peak_median']:.4f}",
                f"- impact_index mean/median: {impact['impact_index_mean']:.2f} / {impact['impact_index_median']:.2f}",
                f"- impact early/valid/late %: {impact['early_pct']:.2f} / {impact['valid_pct']:.2f} / {impact['late_pct']:.2f}",
                "",
                "### Direction Separability",
                "",
                f"- Best baseline macro F1: {baseline.get('best_macro_f1')}",
                f"- Rows written to `dataset_direction_baseline.csv`: {len(baseline.get('rows', []))}",
                "",
                "### Warnings",
                "",
            ]
        )
        warnings = item.get("warnings") or []
        if warnings:
            lines.extend([f"- {warning}" for warning in warnings])
        else:
            lines.append("- none")
        lines.append("")

    lines.extend(
        [
            "## Unit/Scale Cross-Dataset Warnings",
            "",
        ]
    )
    if scale_warnings:
        lines.extend([f"- {warning['warning']}" for warning in scale_warnings])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Cross-Dataset Comparability Table",
            "",
            dataframe_to_markdown(summary_df),
            "",
            "## Channel Stats Preview",
            "",
            dataframe_to_markdown(channel_df.head(30)),
            "",
            "## Direction Baseline Table",
            "",
            dataframe_to_markdown(baseline_df),
            "",
            "## Diagnosis",
            "",
        ]
    )
    lines.extend(build_diagnosis(metrics_by_dataset, summary_df, scale_warnings))
    return "\n".join(lines) + "\n"


def build_diagnosis(
    metrics_by_dataset: dict[str, dict[str, Any]],
    summary_df: pd.DataFrame,
    scale_warnings: list[dict[str, str]],
) -> list[str]:
    present = {k: v for k, v in metrics_by_dataset.items() if not v.get("missing")}
    gravity = [
        dataset
        for dataset, item in present.items()
        if item["gravity"]["gravity_status"] == "likely contains gravity"
    ]
    removed = [
        dataset
        for dataset, item in present.items()
        if item["gravity"]["gravity_status"] == "likely gravity-removed"
    ]
    unit_mismatch = sorted({w["dataset"] for w in scale_warnings}) if scale_warnings else []
    poor_direction = []
    for dataset, item in present.items():
        f1 = item["direction_baseline"].get("best_macro_f1")
        n = item["basic"]["supervised_direction_count"]
        if (isinstance(f1, float) and np.isfinite(f1) and f1 < 0.45) or n < 30:
            poor_direction.append(dataset)
    poor_impact = [
        dataset
        for dataset, item in present.items()
        if isinstance(item["impact"].get("valid_pct"), float) and item["impact"]["valid_pct"] < 80.0
    ]
    umafall = present.get("umafall")
    if umafall:
        umafall_note = (
            "UMAFall appears synchronized at processed-window level "
            f"(impact valid {umafall['impact']['valid_pct']:.1f}%). "
            f"Comparability depends on its gravity/unit result: {umafall['gravity']['gravity_status']}, "
            f"{umafall['gravity']['likely_unit']}."
        )
    else:
        umafall_note = "UMAFall is not present in processed metadata, so synchronization/comparability cannot be audited."
    return [
        f"1. Which datasets likely contain gravity? {gravity or 'none/unclear'}.",
        f"2. Which datasets are likely gravity-removed? {removed or 'none/unclear'}.",
        f"3. Which dataset has unit/scale mismatch? {unit_mismatch or 'none detected by >5x rule'}.",
        f"4. Which dataset has poor direction separability? {poor_direction or 'none by current thresholds'}.",
        f"5. Which dataset has poor impact alignment? {poor_impact or 'none by current thresholds'}.",
        f"6. Whether UMAFall appears synchronized and comparable with BITS/WEDA/HIFD. {umafall_note}",
    ]


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    compact = df.copy()
    for col in compact.columns:
        if pd.api.types.is_float_dtype(compact[col]):
            compact[col] = compact[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    header = "| " + " | ".join(map(str, compact.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(compact.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in compact.to_numpy()]
    return "\n".join([header, sep] + rows)


def missing_summary_row(dataset: str, warnings: list[str]) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "n_windows": 0,
        "n_fall": 0,
        "n_nonfall": 0,
        "n_direction": 0,
        "direction_counts": "{}",
        "acc_mag_median": np.nan,
        "gravity_status": "not available",
        "likely_unit": "not available",
        "gyro_range_note": "not available",
        "impact_valid_pct": np.nan,
        "direction_baseline_macro_f1": np.nan,
        "main_warning": "; ".join(warnings),
    }


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
