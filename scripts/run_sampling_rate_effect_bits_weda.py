from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DIRECTION_LABEL_MAPPING, FALL_LABEL_MAPPING, make_config
from src.data.bits_loader import (
    BITS_ADL_ACTIVITY_NAMES,
    BITS_FALL_ACTIVITY_NAMES,
    BITS_FALL_DIRECTIONS,
    find_bits_trials,
    parse_bits_csv,
)
from src.data.common import compute_acc_mag, has_nan_inf, pad_or_crop_window, resample_sequence_uniform, sliding_windows
from src.data.features import compute_imu_features, feature_channel_names, fit_feature_scaler, transform_with_feature_scaler
from src.data.preprocessing import labels_from_metadata
from src.data.units import harmonize_record_units
from src.data.weda_loader import (
    WEDA_ACTIVITY_NAMES,
    WEDA_FALL_DIRECTIONS,
    WEDA_HARD_NEGATIVES,
    find_weda_trials,
    load_weda_fall_timestamps,
    load_weda_trial,
)
from src.experiments.ablation_rd import load_processed_training_data
from src.experiments.domain_conflict import build_domain_callbacks
from src.models.losses import compute_class_weights
from src.training.dataset import make_tf_dataset
from src.training.evaluate import estimate_inference_latency_ms, evaluate_all, plot_confusion_matrix, save_metrics_json
from src.training.train import compile_ds_fall_model
from src.utils.io import ensure_dir, load_json, save_json, save_pickle
from src.utils.seed import set_seed


@dataclass(frozen=True)
class SamplingSpec:
    id: str
    dataset: str
    sampling_rate: float
    window_length: int
    source: str
    note: str


SPECS = [
    SamplingSpec("BITS-50", "bits", 50.0, 100, "processed50", "Current BITS pipeline: 20 Hz raw upsampled to 50 Hz."),
    SamplingSpec("BITS-20", "bits", 20.0, 40, "bits_native20", "BITS native 20 Hz windows; no upsampling."),
    SamplingSpec("WEDA-50", "weda", 50.0, 100, "processed50", "Current WEDA native 50 Hz pipeline."),
    SamplingSpec("WEDA-20", "weda", 20.0, 40, "weda_downsample20", "WEDA downsampled from 50 Hz to 20 Hz before windowing."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sampling-rate effects for BITS and WEDA with A5WCEFW.")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--processed-x-is-raw", action="store_true", help="Use if data/processed/X.npy is raw6, not normalized.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = make_config(args.project_root)
    set_seed(config.seed)

    processed_data = load_processed_training_data(
        config.processed_dir,
        processed_x_is_normalized=not args.processed_x_is_raw,
    )
    split_subjects = load_json(config.processed_dir / "split_subjects.json")

    rows: list[dict[str, Any]] = []
    for spec in SPECS:
        print(
            f"\n=== {spec.id}: dataset={spec.dataset}, fs={spec.sampling_rate:g} Hz, "
            f"window={spec.window_length}, source={spec.source} ==="
        )
        row = load_existing_row(config.output_dir, spec)
        if row is not None:
            print(f"Reusing existing metrics for {spec.id}.")
        else:
            dataset = load_sampling_dataset(config, processed_data, split_subjects, spec)
            row = run_sampling_experiment(
                config=config,
                spec=spec,
                X_raw6=dataset["X_raw6"],
                metadata=dataset["metadata"],
                y_fall=dataset["y_fall"],
                y_direction=dataset["y_direction"],
                direction_mask=dataset["direction_mask"],
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
        rows.append(row)
        save_outputs(rows, config.output_dir)

    save_outputs(rows, config.output_dir)


def load_sampling_dataset(config, processed_data: dict[str, Any], split_subjects: dict[str, Any], spec: SamplingSpec) -> dict[str, Any]:
    if spec.source == "processed50":
        metadata = processed_data["metadata"].copy()
        metadata["dataset"] = metadata["dataset"].astype(str).str.lower()
        mask = metadata["dataset"].eq(spec.dataset).to_numpy()
        if not mask.any():
            raise ValueError(f"No processed rows found for dataset={spec.dataset!r}")
        return {
            "X_raw6": processed_data["X_raw6"][mask].astype(np.float32),
            "metadata": metadata.loc[mask].reset_index(drop=True),
            "y_fall": processed_data["y_fall"][mask].astype(np.int64),
            "y_direction": processed_data["y_direction"][mask].astype(np.int64),
            "direction_mask": processed_data["direction_mask"][mask].astype(np.float32),
        }

    if spec.source == "bits_native20":
        records = create_bits_native20_records(config)
    elif spec.source == "weda_downsample20":
        records = create_weda_downsample20_records(config)
    else:
        raise ValueError(f"Unsupported source={spec.source!r}")

    records, _ = harmonize_record_units(records, config.dataset_unit_map)
    X_raw6, metadata = records_to_arrays(records)
    metadata = apply_saved_subject_split(metadata, split_subjects, spec.dataset)
    y_fall, y_direction, direction_mask = labels_from_metadata(metadata)
    return {
        "X_raw6": X_raw6,
        "metadata": metadata,
        "y_fall": y_fall,
        "y_direction": y_direction,
        "direction_mask": direction_mask,
    }


def create_bits_native20_records(config) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    window_size = 40
    stride = 20
    for trial in find_bits_trials(config.bits_raw_dir):
        seq = parse_bits_csv(trial["csv_path"], accel_source=config.bits_accel_source)
        if seq is None or len(seq) < window_size or has_nan_inf(seq):
            continue
        if trial["class_dir"] == "fall":
            center = int(np.argmax(compute_acc_mag(seq)))
            window, start, end = pad_or_crop_window(seq, center, window_size)
            direction, supervised = BITS_FALL_DIRECTIONS.get(trial["activity_id"], ("other", False))
            if window.shape == (window_size, 6) and not has_nan_inf(window):
                records.append(bits_record(window, trial, start, end, "peak_acc_native20", direction, supervised, len(seq)))
        else:
            for window, start, end in sliding_windows(seq, window_size=window_size, stride=stride)[: config.bits_adl_cap]:
                if window.shape == (window_size, 6) and not has_nan_inf(window):
                    records.append(bits_record(window, trial, start, end, "sliding_native20", "none", False, len(seq)))
    if not records:
        raise ValueError("BITS-20 produced no records")
    return records


def bits_record(
    window: np.ndarray,
    trial: dict[str, Any],
    start: int,
    end: int,
    event_source: str,
    direction_label: str,
    direction_supervised: bool,
    original_length: int,
) -> dict[str, Any]:
    activity_id = trial["activity_id"]
    activity_name = BITS_ADL_ACTIVITY_NAMES.get(activity_id) or BITS_FALL_ACTIVITY_NAMES.get(activity_id, activity_id)
    return {
        "X": window.astype(np.float32),
        "dataset": "bits",
        "source_path": str(trial["csv_path"]),
        "subject_id": trial["subject_id"],
        "activity_id": activity_id,
        "activity_name": activity_name,
        "fall_label": int(trial["class_dir"] == "fall"),
        "direction_label": direction_label,
        "direction_supervised": direction_supervised,
        "sampling_rate_original": 20.0,
        "sampling_rate_model": 20.0,
        "sensor_position": "left_wrist",
        "window_start_idx": start,
        "window_end_idx": end,
        "event_source": event_source,
        "resampled": False,
        "resample_method": "none_native20",
        "original_length": original_length,
        "resampled_length": original_length,
    }


def create_weda_downsample20_records(config) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    window_size = 40
    stride = 20
    timestamps = load_weda_fall_timestamps(config.weda_raw_dir)
    adl_counts: dict[tuple[str, str], int] = {}
    for trial in find_weda_trials(config.weda_raw_dir):
        seq50 = load_weda_trial(trial["accel_path"], trial["gyro_path"])
        if seq50 is None or len(seq50) < 2 or has_nan_inf(seq50):
            continue
        seq20 = resample_sequence_uniform(seq50, original_fs=50.0, target_fs=20.0)
        if seq20 is None or len(seq20) < window_size or has_nan_inf(seq20):
            continue

        activity_id = trial["activity_id"]
        if activity_id.startswith("F"):
            if trial["base_id"] in timestamps:
                start_time, end_time = timestamps[trial["base_id"]]
                lo = max(0, int(np.floor(start_time * 20.0)))
                hi = min(len(seq20), int(np.ceil(end_time * 20.0)))
                if hi <= lo:
                    lo, hi = 0, len(seq20)
                center = lo + int(np.argmax(compute_acc_mag(seq20[lo:hi])))
                event_source = "timestamp_peak_acc_downsample20"
            else:
                center = int(np.argmax(compute_acc_mag(seq20)))
                event_source = "peak_acc_downsample20"
            window, start, end = pad_or_crop_window(seq20, center, window_size)
            direction = WEDA_FALL_DIRECTIONS.get(activity_id, "other")
            if window.shape == (window_size, 6) and not has_nan_inf(window):
                records.append(weda_record(window, trial, start, end, event_source, direction, direction in {"forward", "backward", "lateral"}, len(seq50), len(seq20)))
        else:
            cap = 10 if activity_id in WEDA_HARD_NEGATIVES else config.weda_adl_cap
            cap_key = (trial["subject_id"], activity_id)
            remaining = max(0, cap - adl_counts.get(cap_key, 0))
            if remaining == 0:
                continue
            for window, start, end in sliding_windows(seq20, window_size=window_size, stride=stride)[:remaining]:
                if window.shape == (window_size, 6) and not has_nan_inf(window):
                    records.append(weda_record(window, trial, start, end, "sliding_downsample20", "none", False, len(seq50), len(seq20)))
                    adl_counts[cap_key] = adl_counts.get(cap_key, 0) + 1
    if not records:
        raise ValueError("WEDA-20 produced no records")
    return records


def weda_record(
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
    return {
        "X": window.astype(np.float32),
        "dataset": "weda",
        "source_path": str(trial["accel_path"]),
        "subject_id": trial["subject_id"],
        "activity_id": trial["activity_id"],
        "activity_name": WEDA_ACTIVITY_NAMES.get(trial["activity_id"], trial["activity_id"]),
        "fall_label": int(trial["activity_id"].startswith("F")),
        "direction_label": direction_label,
        "direction_supervised": direction_supervised,
        "sampling_rate_original": 50.0,
        "sampling_rate_model": 20.0,
        "sensor_position": "wrist",
        "window_start_idx": start,
        "window_end_idx": end,
        "event_source": event_source,
        "resampled": True,
        "resample_method": "linear_index_time_50_to_20",
        "original_length": original_length,
        "resampled_length": resampled_length,
    }


def records_to_arrays(records: list[dict[str, Any]]) -> tuple[np.ndarray, pd.DataFrame]:
    cleaned: list[dict[str, Any]] = []
    windows: list[np.ndarray] = []
    for rec in records:
        x = rec.get("X")
        if not isinstance(x, np.ndarray) or has_nan_inf(x):
            continue
        out = dict(rec)
        windows.append(out.pop("X").astype(np.float32))
        cleaned.append(out)
    if not windows:
        raise ValueError("No valid windows after record filtering")
    metadata = pd.DataFrame(cleaned)
    metadata.insert(0, "sample_id", [f"sample_{i:07d}" for i in range(len(metadata))])
    metadata["direction_supervised"] = metadata["direction_supervised"].astype(bool)
    metadata["fall_label"] = metadata["fall_label"].astype(int)
    return np.stack(windows).astype(np.float32), metadata


def apply_saved_subject_split(metadata: pd.DataFrame, split_subjects: dict[str, Any], dataset: str) -> pd.DataFrame:
    if dataset not in split_subjects:
        raise ValueError(f"Dataset {dataset!r} not found in split_subjects.json")
    out = metadata.copy()
    subject_to_split: dict[str, str] = {}
    for split_name, subjects in split_subjects[dataset].items():
        for subject in subjects:
            subject_to_split[str(subject)] = split_name
    out["split"] = out["subject_id"].astype(str).map(subject_to_split)
    if out["split"].isna().any():
        missing = sorted(out.loc[out["split"].isna(), "subject_id"].astype(str).unique().tolist())
        raise ValueError(f"Some {dataset} subjects are missing from split_subjects.json: {missing}")
    return out


def run_sampling_experiment(
    config,
    spec: SamplingSpec,
    X_raw6: np.ndarray,
    metadata: pd.DataFrame,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    from src.models.ds_fall import build_model

    set_seed(config.seed)
    run_id = spec.id.replace("-", "_")
    run_dir = ensure_dir(config.output_dir / "runs" / f"SR_{run_id}_A5WCEFW")
    metadata = metadata.copy().reset_index(drop=True)
    metadata["dataset"] = metadata["dataset"].astype(str).str.lower()
    split_values = metadata["split"].astype(str).str.lower().to_numpy()
    masks = {name: split_values == name for name in ["train", "val", "test"]}
    for split_name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"{spec.id} has no {split_name} samples")

    features = compute_imu_features(X_raw6, feature_set="tilt12", fs=spec.sampling_rate)
    scaler = fit_feature_scaler(features, metadata, feature_set="tilt12")
    X_scaled = transform_with_feature_scaler(features, scaler)

    X_train, yf_train, yd_train, dm_train = split_arrays(X_scaled, y_fall, y_direction, direction_mask, masks["train"])
    X_val, yf_val, yd_val, dm_val = split_arrays(X_scaled, y_fall, y_direction, direction_mask, masks["val"])
    X_test, yf_test, yd_test, dm_test = split_arrays(X_scaled, y_fall, y_direction, direction_mask, masks["test"])
    meta_test = metadata.loc[masks["test"]].reset_index(drop=True)

    direction_class_weights = compute_class_weights(yd_train, num_classes=3, mask=dm_train * (yf_train == 1))
    fall_class_weights = compute_class_weights(yf_train, num_classes=2)

    model = build_model(
        "ds_fall_rd",
        input_shape=(X_train.shape[1], X_train.shape[2]),
        feature_set="tilt12",
        num_direction_classes=3,
        show_summary=False,
    )
    compile_ds_fall_model(
        model,
        learning_rate=learning_rate,
        direction_loss_type="weighted_ce",
        lambda_fall=1.0,
        lambda_direction=1.5,
        direction_class_weights=direction_class_weights,
        fall_class_weights=fall_class_weights,
    )

    train_ds = make_tf_dataset(X_train, yf_train, yd_train, dm_train, batch_size=batch_size, shuffle=True, seed=config.seed)
    val_ds = make_tf_dataset(X_val, yf_val, yd_val, dm_val, batch_size=batch_size, shuffle=False, seed=config.seed)
    callbacks = build_domain_callbacks(
        run_dir,
        X_val=X_val,
        y_fall_val=yf_val,
        y_direction_val=yd_val,
        direction_mask_val=dm_val,
        patience=20,
        use_early_stopping=True,
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)

    model.save(run_dir / "model_final.keras")
    save_json(run_dir / "history.json", history.history)
    save_json(
        run_dir / "model_config.json",
        asdict(spec)
        | {
            "model": "A5WCEFW",
            "model_name": "ds_fall_rd",
            "feature_set": "tilt12",
            "direction_loss_type": "weighted_ce",
            "lambda_fall": 1.0,
            "lambda_direction": 1.5,
            "fall_loss_weighted": True,
            "augment": False,
            "input_shape": [int(X_train.shape[1]), int(X_train.shape[2])],
            "learning_rate": learning_rate,
        },
    )
    save_json(run_dir / "feature_config.json", {"feature_set": "tilt12", "channel_names": feature_channel_names("tilt12"), "fs": spec.sampling_rate})
    save_json(run_dir / "feature_scaler.json", scaler)
    save_pickle(run_dir / "feature_scaler.pkl", scaler)
    save_json(run_dir / "label_mapping.json", {"fall": FALL_LABEL_MAPPING, "direction": DIRECTION_LABEL_MAPPING})
    save_json(run_dir / "class_weights.json", {"fall": fall_class_weights.tolist(), "direction": direction_class_weights.tolist()})
    save_json(
        run_dir / "split_counts.json",
        {
            "train": int(masks["train"].sum()),
            "val": int(masks["val"].sum()),
            "test": int(masks["test"].sum()),
            "train_direction": int(((direction_mask > 0) & (y_direction >= 0) & masks["train"]).sum()),
            "val_direction": int(((direction_mask > 0) & (y_direction >= 0) & masks["val"]).sum()),
            "test_direction": int(((direction_mask > 0) & (y_direction >= 0) & masks["test"]).sum()),
        },
    )

    metrics = evaluate_all(model, X_test, yf_test, yd_test, dm_test, metadata=meta_test)
    latency_ms = estimate_inference_latency_ms(model, X_test)
    metrics["inference_latency_ms"] = latency_ms
    save_metrics_json(run_dir / "metrics.json", metrics)
    save_confusion_matrices(metrics, config.output_dir, spec.id)

    row = flatten_sampling_metrics(spec, metrics, latency_ms)
    row.update(
        {
            "n_train": int(masks["train"].sum()),
            "n_val": int(masks["val"].sum()),
            "n_test": int(masks["test"].sum()),
            "n_train_direction": int(((direction_mask > 0) & (y_direction >= 0) & masks["train"]).sum()),
            "n_val_direction": int(((direction_mask > 0) & (y_direction >= 0) & masks["val"]).sum()),
            "n_test_direction": int(((direction_mask > 0) & (y_direction >= 0) & masks["test"]).sum()),
            "fall_confusion_matrix": json.dumps(metrics.get("fall", {}).get("confusion_matrix")),
            "direction_confusion_matrix": json.dumps(metrics.get("direction", {}).get("confusion_matrix")),
        }
    )
    print(f"{spec.id} result: fall_f1={row.get('fall_f1')}, direction_macro_f1={row.get('direction_macro_f1')}")
    return row


def load_existing_row(output_dir: str | Path, spec: SamplingSpec) -> dict[str, Any] | None:
    run_id = spec.id.replace("-", "_")
    run_dir = Path(output_dir) / "runs" / f"SR_{run_id}_A5WCEFW"
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    latency_ms = metrics.get("inference_latency_ms")
    row = flatten_sampling_metrics(spec, metrics, latency_ms)
    split_counts_path = run_dir / "split_counts.json"
    if split_counts_path.exists():
        with split_counts_path.open("r", encoding="utf-8") as f:
            counts = json.load(f)
        row.update(
            {
                "n_train": counts.get("train"),
                "n_val": counts.get("val"),
                "n_test": counts.get("test"),
                "n_train_direction": counts.get("train_direction"),
                "n_val_direction": counts.get("val_direction"),
                "n_test_direction": counts.get("test_direction"),
            }
        )
    row["fall_confusion_matrix"] = json.dumps(metrics.get("fall", {}).get("confusion_matrix"))
    row["direction_confusion_matrix"] = json.dumps(metrics.get("direction", {}).get("confusion_matrix"))
    save_confusion_matrices(metrics, output_dir, spec.id)
    return row


def split_arrays(
    X: np.ndarray,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        X[mask],
        y_fall[mask].astype(np.int64),
        y_direction[mask].astype(np.int64),
        direction_mask[mask].astype(np.float32),
    )


def flatten_sampling_metrics(spec: SamplingSpec, metrics: dict[str, Any], latency_ms: float | None) -> dict[str, Any]:
    fall = metrics.get("fall", {})
    direction = metrics.get("direction", {})
    size = metrics.get("model_size", {})
    report = direction.get("classification_report", {})
    row: dict[str, Any] = {
        "experiment_id": spec.id,
        "dataset": spec.dataset,
        "sampling_rate": spec.sampling_rate,
        "window_length": spec.window_length,
        "source": spec.source,
        "note": spec.note,
        "model": "A5WCEFW",
        "feature_set": "tilt12",
        "direction_loss_type": "weighted_ce",
        "lambda_direction": 1.5,
        "fall_loss_weighted": True,
        "augment": False,
        "fall_accuracy": fall.get("accuracy"),
        "fall_precision": fall.get("precision"),
        "fall_recall": fall.get("recall"),
        "fall_f1": fall.get("f1"),
        "direction_num_supervised": direction.get("num_supervised", 0),
        "direction_accuracy": direction.get("accuracy"),
        "direction_macro_f1": direction.get("macro_f1"),
        "params": size.get("params"),
        "fp32_kb": size.get("fp32_kb"),
        "estimated_int8_kb": size.get("estimated_int8_kb"),
        "inference_latency_ms": latency_ms,
    }
    for label in ["forward", "backward", "lateral"]:
        if label in report:
            row[f"direction_{label}_precision"] = report[label].get("precision")
            row[f"direction_{label}_recall"] = report[label].get("recall")
            row[f"direction_{label}_f1"] = report[label].get("f1-score")
    return row


def save_confusion_matrices(metrics: dict[str, Any], output_dir: str | Path, experiment_id: str) -> None:
    fig_dir = ensure_dir(Path(output_dir) / "figures" / "sampling_rate_effect_confusion_matrices" / experiment_id)
    fall_cm = metrics.get("fall", {}).get("confusion_matrix")
    if fall_cm is not None:
        plot_confusion_matrix(fall_cm, ["non_fall", "fall"], f"{experiment_id} fall confusion matrix", fig_dir / "fall.png", show=False)
    direction_cm = metrics.get("direction", {}).get("confusion_matrix")
    if direction_cm is not None:
        plot_confusion_matrix(
            direction_cm,
            ["forward", "backward", "lateral"],
            f"{experiment_id} direction confusion matrix",
            fig_dir / "direction.png",
            show=False,
        )


def save_outputs(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    reports_dir = ensure_dir(Path(output_dir) / "reports")
    df = pd.DataFrame(rows)
    df.to_csv(reports_dir / "sampling_rate_effect_bits_weda.csv", index=False)
    (reports_dir / "sampling_rate_effect_bits_weda.md").write_text(build_markdown(df), encoding="utf-8")


def build_markdown(df: pd.DataFrame) -> str:
    lines = [
        "# Sampling Rate Effect: BITS and WEDA",
        "",
        "Fixed model: A5WCEFW / DS-Fall-RD with tilt12, task-specific attention, weighted CE fall, weighted CE direction.",
        "HIFD and UMAFall are not used. Jerk is computed with the experiment sampling rate.",
        "",
    ]
    if df.empty:
        return "\n".join(lines + ["_No rows yet._", ""]) + "\n"
    effect = sampling_effect_table(df)
    lines.extend(["## Table 1: Main Results", "", markdown_table(table1(df)), ""])
    lines.extend(["## Table 2: Sampling Effect", "", markdown_table(effect), ""])
    lines.extend(["## Table 3: Direction Per-Class F1", "", markdown_table(table3(df)), ""])
    lines.extend(["## Diagnosis", ""])
    lines.extend(diagnosis_lines(df, effect))
    lines.append("")
    return "\n".join(lines)


def table1(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "experiment_id",
        "dataset",
        "sampling_rate",
        "window_length",
        "fall_f1",
        "fall_precision",
        "fall_recall",
        "direction_macro_f1",
        "direction_accuracy",
        "direction_num_supervised",
        "params",
    ]
    out = df[[col for col in cols if col in df.columns]].rename(columns={"direction_num_supervised": "direction_n"})
    return format_float_columns(out)


def sampling_effect_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in ["bits", "weda"]:
        high_id = f"{dataset.upper()}-50"
        low_id = f"{dataset.upper()}-20"
        high = df[df["experiment_id"] == high_id]
        low = df[df["experiment_id"] == low_id]
        if high.empty or low.empty:
            continue
        h = high.iloc[0]
        l = low.iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "high_rate_config": high_id,
                "low_rate_config": low_id,
                "fall_f1_delta": h.get("fall_f1") - l.get("fall_f1"),
                "direction_macro_f1_delta": h.get("direction_macro_f1") - l.get("direction_macro_f1"),
                "direction_accuracy_delta": h.get("direction_accuracy") - l.get("direction_accuracy"),
            }
        )
    columns = [
        "dataset",
        "high_rate_config",
        "low_rate_config",
        "fall_f1_delta",
        "direction_macro_f1_delta",
        "direction_accuracy_delta",
    ]
    return format_float_columns(pd.DataFrame(rows, columns=columns))


def table3(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "experiment_id": "experiment_id",
        "direction_forward_f1": "forward_f1",
        "direction_backward_f1": "backward_f1",
        "direction_lateral_f1": "lateral_f1",
    }
    out = df[[col for col in cols if col in df.columns]].rename(columns=cols)
    return format_float_columns(out)


def diagnosis_lines(df: pd.DataFrame, effect: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    bits50 = metric(df, "BITS-50", "direction_macro_f1")
    bits20 = metric(df, "BITS-20", "direction_macro_f1")
    weda50 = metric(df, "WEDA-50", "direction_macro_f1")
    weda20 = metric(df, "WEDA-20", "direction_macro_f1")

    bits_delta = delta(effect, "bits", "direction_macro_f1_delta")
    weda_delta = delta(effect, "weda", "direction_macro_f1_delta")
    lines.append(
        f"1. BITS 50 Hz vs 20 Hz direction macro F1: {_fmt(bits50)} vs {_fmt(bits20)} "
        f"(high-low delta {_fmt(bits_delta)})."
    )
    if bits20 is not None:
        lines.append(
            f"2. BITS native 20 Hz is {'still usable' if bits20 >= 0.60 else 'weak'} for direction at macro F1 {_fmt(bits20)}."
        )
    else:
        lines.append("2. BITS native 20 Hz direction quality cannot be determined.")
    weda50_degenerate = is_degenerate(df, "WEDA-50")
    weda20_degenerate = is_degenerate(df, "WEDA-20")
    if weda50_degenerate or weda20_degenerate:
        lines.append(
            f"3. WEDA 50 Hz vs 20 Hz direction macro F1: {_fmt(weda50)} vs {_fmt(weda20)} "
            f"(high-low delta {_fmt(weda_delta)}), but the WEDA runs are degenerate: the selected checkpoint predicts no fall samples "
            "and only one direction class. Treat this WEDA sampling comparison as inconclusive."
        )
    else:
        lines.append(
            f"3. WEDA 50 Hz vs 20 Hz direction macro F1: {_fmt(weda50)} vs {_fmt(weda20)} "
            f"(high-low delta {_fmt(weda_delta)})."
        )
    low_values = [v for v in [bits20, weda20] if v is not None]
    if low_values:
        lines.append(
            f"4. Common 20 Hz feasibility: {'reasonable' if min(low_values) >= 0.60 else 'not strong enough'} "
            f"with minimum direction macro F1 {_fmt(min(low_values))}."
        )
    else:
        lines.append("4. Common 20 Hz feasibility cannot be determined.")
    if weda50_degenerate or weda20_degenerate:
        lines.append("5. Sampling rate is not proven to be the main cause of BITS/WEDA differences; WEDA needs a non-degenerate training run before this can be isolated cleanly.")
    elif bits_delta is not None and weda_delta is not None and abs(bits_delta) < 0.10 and abs(weda_delta) < 0.10:
        lines.append("5. Sampling rate does not look like the main cause of BITS/WEDA differences.")
    else:
        lines.append("5. Sampling rate has a measurable effect, but domain/label factors should still be considered alongside it.")
    if weda50_degenerate or weda20_degenerate:
        lines.append("6. For the paper, report BITS 50 Hz vs native 20 Hz as a valid sampling ablation, and rerun/fix WEDA checkpointing before making a 50 Hz vs 20 Hz claim for WEDA.")
    elif bits_delta is not None and weda_delta is not None:
        if bits_delta > 0.05 or weda_delta > 0.05:
            lines.append("6. For the paper, report 50 Hz as the main setup and include 20 Hz as a sampling-rate ablation.")
        else:
            lines.append("6. For the paper, reporting both 50 Hz and 20 Hz is useful; 20 Hz may be acceptable if latency/power matters.")
    else:
        lines.append("6. Paper recommendation cannot be finalized because some sampling-rate rows are missing.")
    return lines


def is_degenerate(df: pd.DataFrame, exp_id: str) -> bool:
    row = df[df["experiment_id"] == exp_id]
    if row.empty:
        return False
    item = row.iloc[0]
    fall_f1 = item.get("fall_f1")
    direction_macro = item.get("direction_macro_f1")
    try:
        fall_bad = float(fall_f1) == 0.0
    except (TypeError, ValueError):
        fall_bad = False
    try:
        direction_bad = float(direction_macro) <= 0.20
    except (TypeError, ValueError):
        direction_bad = False
    return fall_bad and direction_bad


def metric(df: pd.DataFrame, exp_id: str, col: str) -> float | None:
    row = df[df["experiment_id"] == exp_id]
    if row.empty or col not in row:
        return None
    value = row.iloc[0].get(col)
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def delta(effect: pd.DataFrame, dataset: str, col: str) -> float | None:
    if effect.empty or "dataset" not in effect:
        return None
    row = effect[effect["dataset"] == dataset]
    if row.empty or col not in row:
        return None
    value = row.iloc[0].get(col)
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in df.to_numpy()]
    return "\n".join([header, sep] + rows)


def format_float_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            if col in {"window_length", "direction_n", "params"}:
                out[col] = out[col].map(lambda value: "" if pd.isna(value) else str(int(value)))
            elif col == "sampling_rate":
                out[col] = out[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.0f}")
            else:
                out[col] = out[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out


def _fmt(value: Any) -> str:
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
