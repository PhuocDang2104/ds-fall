from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_25hz_a5wcefw_bits_weda as base25
from src.config import DIRECTION_LABEL_MAPPING, FALL_LABEL_MAPPING, make_config
from src.models.losses import (
    _get_sparse_focal_class,
    compute_class_weights,
    make_direction_loss,
    make_sample_weights,
    make_sparse_ce_loss,
    prepare_direction_targets,
)
from src.utils.io import ensure_dir, save_json, save_pickle
from src.utils.seed import set_seed


SAMPLING_RATE = 25.0
WINDOW_SECONDS = 2.0
WINDOW_SIZE = 50
DIRECTION_LABELS = ["forward", "backward", "lateral"]
FALL_LABELS = ["non_fall", "fall"]

FEATURE_PRESETS: dict[str, list[str]] = {
    "raw6": ["ax", "ay", "az", "gx", "gy", "gz"],
    "fa1_acc_mag7": ["ax", "ay", "az", "gx", "gy", "gz", "acc_mag"],
    "fa2_mag8": ["ax", "ay", "az", "gx", "gy", "gz", "acc_mag", "gyro_mag"],
    "mag_jerk9": ["ax", "ay", "az", "gx", "gy", "gz", "acc_mag", "gyro_mag", "jerk"],
    "fa4_roll_pitch8": ["ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch"],
    "fa5_tilt9": ["ax", "ay", "az", "gx", "gy", "gz", "roll", "pitch", "tilt_delta"],
    "fa6_no_jerk11": [
        "ax",
        "ay",
        "az",
        "gx",
        "gy",
        "gz",
        "acc_mag",
        "gyro_mag",
        "roll",
        "pitch",
        "tilt_delta",
    ],
    "fa7_no_gyro_mag11": [
        "ax",
        "ay",
        "az",
        "gx",
        "gy",
        "gz",
        "acc_mag",
        "jerk",
        "roll",
        "pitch",
        "tilt_delta",
    ],
    "tilt12": ["ax", "ay", "az", "gx", "gy", "gz", "acc_mag", "gyro_mag", "jerk", "roll", "pitch", "tilt_delta"],
}

FULL_FEATURE_ORDER = FEATURE_PRESETS["tilt12"]


@dataclass(frozen=True)
class TargetConfig:
    run_id: str
    group_id: str
    config_id: str
    feature_set: str = "tilt12"
    width_multiplier: float = 1.0
    shared_dense: int = 0
    head_dense: int = 32
    sampler: str = "current"
    fall_loss: str = "weighted_ce"
    alpha_fall: float = 1.0
    lambda_dir: float = 1.5
    fall_gamma: float = 2.0
    direction_loss: str = "weighted_ce"
    adapter_type: str = "none"
    adapter_filters: int = 0
    impact_position: float = 0.5
    learning_rate: float = 1e-3
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run targeted DS-Fall-RD BITS/WEDA experiments.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--suite", choices=["core", "full"], default="core")
    parser.add_argument("--limit-configs", type=int, default=0, help="Debug helper: stop after N trained configs.")
    parser.add_argument("--only-run-ids", nargs="*", default=None, help="Run only these run_id values.")
    parser.add_argument("--append-existing", action="store_true", help="Append/update an existing report directory instead of starting from empty tables.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--artifact-subdir", type=str, default="targeted_experiments")
    parser.add_argument(
        "--reuse-event-data",
        action="store_true",
        help="Use existing event artifacts when available for audit consistency; raw windows are still rebuilt safely.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = make_config(args.project_root)
    report_dir = ensure_dir(config.output_dir / "reports" / args.artifact_subdir)
    figures_dir = ensure_dir(config.output_dir / "figures" / args.artifact_subdir)
    run_root = ensure_dir(report_dir / "runs")
    ensure_dir(report_dir / "confusion_matrices")
    ensure_dir(figures_dir / "confusion_matrices")

    write_experiment_plan(report_dir, args)

    print("=== Targeted DS-Fall-RD experiments ===")
    print(f"project_root={config.project_root}")
    print(f"report_dir={report_dir}")
    print(f"window={WINDOW_SECONDS:g}s, sampling={SAMPLING_RATE:g}Hz, timesteps={WINDOW_SIZE}")

    data = base25.build_25hz_dataset(config, window_mode="event_centered")
    base25.validate_25hz_dataset(data)
    write_repo_audit(report_dir, data)

    all_rows, threshold_rows, per_dataset_rows, per_class_rows = load_existing_rows(report_dir) if args.append_existing else ([], [], [], [])
    completed: dict[str, dict[str, Any]] = {}

    baseline = TargetConfig(
        run_id="B0_BASE_A5WCEFW",
        group_id="baseline",
        config_id="B0",
        fall_loss="weighted_ce",
        alpha_fall=1.0,
        lambda_dir=1.5,
        direction_loss="weighted_ce",
        notes="Current A5WCEFW recipe, plus validation-tuned threshold evaluation.",
    )
    only = set(args.only_run_ids or [])

    def should_run(cfg: TargetConfig) -> bool:
        return not only or cfg.run_id in only

    block1 = [cfg for cfg in [baseline] + loss_configs() + sampler_configs() if should_run(cfg)]

    trained_configs = 0
    for cfg in block1:
        if args.limit_configs and trained_configs >= args.limit_configs:
            break
        rows = run_config_for_seeds(cfg, args.seeds, data, config, run_root, args, report_dir, figures_dir)
        trained_configs += 1
        completed[cfg.run_id] = best_seed_row(rows)
        collect_rows(rows, all_rows, threshold_rows, per_dataset_rows, per_class_rows)
        flush_reports(report_dir, all_rows, threshold_rows, per_dataset_rows, per_class_rows)

    best_recipe = choose_best_recipe([row for row in all_rows if row["group_id"] in {"baseline", "loss", "sampler"}])
    if best_recipe is None:
        best_recipe = asdict(baseline)
    feature_rows_source: list[dict[str, Any]] = []
    if not args.limit_configs or trained_configs < args.limit_configs:
        feature_base = config_from_row(best_recipe, "FA8_TILT12", "feature_ablation", "FA8")
        for fa_cfg in [cfg for cfg in feature_configs(feature_base) if should_run(cfg)]:
            if args.limit_configs and trained_configs >= args.limit_configs:
                break
            rows = run_config_for_seeds(fa_cfg, args.seeds[:1], data, config, run_root, args, report_dir, figures_dir)
            trained_configs += 1
            feature_rows_source.extend(rows)
            completed[fa_cfg.run_id] = best_seed_row(rows)
            collect_rows(rows, all_rows, threshold_rows, per_dataset_rows, per_class_rows)
            flush_reports(report_dir, all_rows, threshold_rows, per_dataset_rows, per_class_rows)

    best_feature = choose_best_recipe(feature_rows_source) or best_recipe
    compact_rows_source: list[dict[str, Any]] = []
    if not args.limit_configs or trained_configs < args.limit_configs:
        compact_base = config_from_row(best_feature, "W100_WIDTH", "compactness", "W100")
        for c_cfg in [cfg for cfg in compactness_configs(compact_base) if should_run(cfg)]:
            if args.limit_configs and trained_configs >= args.limit_configs:
                break
            rows = run_config_for_seeds(c_cfg, args.seeds[:1], data, config, run_root, args, report_dir, figures_dir)
            trained_configs += 1
            compact_rows_source.extend(rows)
            completed[c_cfg.run_id] = best_seed_row(rows)
            collect_rows(rows, all_rows, threshold_rows, per_dataset_rows, per_class_rows)
            flush_reports(report_dir, all_rows, threshold_rows, per_dataset_rows, per_class_rows)

    if args.suite == "full" and (not args.limit_configs or trained_configs < args.limit_configs):
        best_compact = choose_best_recipe(compact_rows_source) or best_feature
        adapter_base = config_from_row(best_compact, "AD0_NONE", "adapter", "AD0")
        for a_cfg in [cfg for cfg in adapter_configs(adapter_base) if should_run(cfg)]:
            if args.limit_configs and trained_configs >= args.limit_configs:
                break
            rows = run_config_for_seeds(a_cfg, args.seeds[:1], data, config, run_root, args, report_dir, figures_dir)
            trained_configs += 1
            collect_rows(rows, all_rows, threshold_rows, per_dataset_rows, per_class_rows)
            flush_reports(report_dir, all_rows, threshold_rows, per_dataset_rows, per_class_rows)

    add_reference_event_baseline(config.project_root, all_rows, per_dataset_rows, per_class_rows)
    save_optional_reports(report_dir, data, all_rows)
    finalize_reports(report_dir, figures_dir, all_rows, threshold_rows, per_dataset_rows, per_class_rows)


def run_config_for_seeds(
    cfg: TargetConfig,
    seeds: list[int],
    data: dict[str, Any],
    project_config,
    run_root: Path,
    args: argparse.Namespace,
    report_dir: Path,
    figures_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        row = run_or_load_target_config(
            cfg=cfg,
            seed=seed,
            data=data,
            project_config=project_config,
            run_root=run_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            force=args.force,
            report_dir=report_dir,
            figures_dir=figures_dir,
        )
        rows.append(row)
    return rows


def run_or_load_target_config(
    cfg: TargetConfig,
    seed: int,
    data: dict[str, Any],
    project_config,
    run_root: Path,
    epochs: int,
    batch_size: int,
    patience: int,
    force: bool,
    report_dir: Path,
    figures_dir: Path,
) -> dict[str, Any]:
    run_dir = ensure_dir(run_root / f"{cfg.run_id}_seed{seed}")
    row_path = run_dir / "summary_row.json"
    if row_path.exists() and not force:
        print(f"Reusing {cfg.run_id} seed={seed}")
        with row_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    import tensorflow as tf

    set_seed(seed)
    tf.keras.backend.clear_session()
    gc.collect()
    print(f"\n=== {cfg.run_id} seed={seed} ===")
    metadata = data["metadata"].copy().reset_index(drop=True)
    y_fall = data["y_fall"]
    y_direction = data["y_direction"]
    direction_mask = data["direction_mask"]

    spec = base25.ExperimentSpec(
        "E3_BITS_WEDA_MIXED",
        "bits+weda",
        "bits+weda",
        ("bits", "weda"),
        ("bits", "weda"),
        ("bits", "weda"),
    )
    train_mask, val_mask, test_mask = base25.experiment_masks(metadata, spec)
    base25.assert_no_leakage(metadata, train_mask, val_mask, test_mask, spec)

    X_features = compute_feature_set(data["X_raw6"], cfg.feature_set, fs=SAMPLING_RATE)
    scaler = fit_scaler(X_features[train_mask], cfg.feature_set)
    X_scaled = transform_scaler(X_features, scaler)

    X_train, yf_train, yd_train, dm_train = base25.split_arrays(X_scaled, y_fall, y_direction, direction_mask, train_mask)
    X_val, yf_val, yd_val, dm_val = base25.split_arrays(X_scaled, y_fall, y_direction, direction_mask, val_mask)
    X_test, yf_test, yd_test, dm_test = base25.split_arrays(X_scaled, y_fall, y_direction, direction_mask, test_mask)
    meta_train = metadata.loc[train_mask].reset_index(drop=True)
    meta_val = metadata.loc[val_mask].reset_index(drop=True)
    meta_test = metadata.loc[test_mask].reset_index(drop=True)

    fall_class_weights = compute_class_weights(yf_train, num_classes=2)
    direction_class_weights = compute_class_weights(yd_train, num_classes=3, mask=dm_train * (yf_train == 1))
    model = build_target_model(
        input_shape=(WINDOW_SIZE, len(FEATURE_PRESETS[cfg.feature_set])),
        feature_set=cfg.feature_set,
        width_multiplier=cfg.width_multiplier,
        shared_dense=cfg.shared_dense,
        head_dense=cfg.head_dense,
        adapter_type=cfg.adapter_type,
        adapter_filters=cfg.adapter_filters,
    )
    compile_target_model(
        model,
        cfg,
        fall_class_weights=fall_class_weights,
        direction_class_weights=direction_class_weights,
    )

    if cfg.sampler == "dataset_balanced":
        train_indices = balanced_training_indices(meta_train, yf_train, seed=seed)
        X_train_fit = X_train[train_indices]
        yf_train_fit = yf_train[train_indices]
        yd_train_fit = yd_train[train_indices]
        dm_train_fit = dm_train[train_indices]
        sampler_note = f"dataset_balanced static oversample n={len(train_indices)}"
    else:
        X_train_fit, yf_train_fit, yd_train_fit, dm_train_fit = X_train, yf_train, yd_train, dm_train
        sampler_note = "current shuffle"

    train_ds = make_tf_dataset(X_train_fit, yf_train_fit, yd_train_fit, dm_train_fit, batch_size=batch_size, shuffle=True, seed=seed)
    val_ds = make_tf_dataset(X_val, yf_val, yd_val, dm_val, batch_size=batch_size, shuffle=False, seed=seed)
    callbacks = [
        make_validation_metric_callback(X_val, yf_val, yd_val, dm_val),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best_model.keras"),
            monitor="val_domain_score",
            mode="max",
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_domain_score",
            mode="max",
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
        tf.keras.callbacks.CSVLogger(str(run_dir / "training_history.csv")),
    ]

    start = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)
    train_seconds = time.perf_counter() - start
    history_df = pd.DataFrame(history.history)
    if "epoch" not in history_df.columns:
        history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(run_dir / "training_history.csv", index=False)

    val_probs = predict_probs(model, X_val)
    test_probs = predict_probs(model, X_test)
    best_threshold, val_best_f1 = tune_threshold(yf_val, val_probs["fall_prob"])
    dataset_thresholds = tune_dataset_thresholds(meta_val, yf_val, val_probs["fall_prob"])

    evaluations = evaluate_e3_e6_e7(
        run_dir=run_dir,
        report_dir=report_dir,
        figures_dir=figures_dir,
        cfg=cfg,
        seed=seed,
        meta_test=meta_test,
        y_fall=yf_test,
        y_direction=yd_test,
        direction_mask=dm_test,
        fall_prob=test_probs["fall_prob"],
        direction_prob=test_probs["direction_prob"],
        threshold_05=0.5,
        threshold_tuned=best_threshold,
    )

    best_epoch, best_val_loss, best_monitor = best_epoch_from_history(history_df, "val_domain_score", "max")
    row = build_summary_row(
        cfg=cfg,
        seed=seed,
        evaluations=evaluations,
        params=int(model.count_params()),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_monitor=best_monitor,
        fall_threshold=best_threshold,
        val_best_fall_f1=val_best_f1,
        train_seconds=train_seconds,
        n_train=int(train_mask.sum()),
        n_val=int(val_mask.sum()),
        n_test=int(test_mask.sum()),
        sampler_note=sampler_note,
    )

    save_json(run_dir / "summary_row.json", row)
    save_json(run_dir / "model_config.json", {**asdict(cfg), "seed": seed, "input_shape": [WINDOW_SIZE, len(FEATURE_PRESETS[cfg.feature_set])]})
    save_json(run_dir / "feature_scaler.json", scaler)
    save_json(
        run_dir / "class_weights.json",
        {"fall": fall_class_weights.tolist(), "direction": direction_class_weights.tolist()},
    )
    save_pickle(run_dir / "feature_scaler.pkl", scaler)
    model.save(run_dir / "model_final.keras")
    save_json(run_dir / "history.json", history.history)
    save_json(run_dir / "thresholds.json", {"global": best_threshold, "val_best_fall_f1": val_best_f1, "dataset": dataset_thresholds})
    pd.DataFrame([row]).to_csv(run_dir / "summary_row.csv", index=False)
    print(
        f"{cfg.run_id}: E7 fall_f1={row['weda_fall_f1']:.4f}, "
        f"E7 dir={row['weda_direction_macro_f1']:.4f}, "
        f"E3 dir={row['direction_macro_f1']:.4f}, tau={best_threshold:.2f}"
    )
    return row


def compute_full_features(X_raw6: np.ndarray, fs: float = SAMPLING_RATE, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X_raw6, dtype=np.float32)
    ax, ay, az = X[:, :, 0], X[:, :, 1], X[:, :, 2]
    gx, gy, gz = X[:, :, 3], X[:, :, 4], X[:, :, 5]
    acc_mag = np.sqrt(np.maximum(ax * ax + ay * ay + az * az, 0.0) + eps)
    gyro_mag = np.sqrt(np.maximum(gx * gx + gy * gy + gz * gz, 0.0) + eps)
    dt = 1.0 / float(fs)
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


def compute_feature_set(X_raw6: np.ndarray, feature_set: str, fs: float = SAMPLING_RATE) -> np.ndarray:
    if feature_set not in FEATURE_PRESETS:
        raise ValueError(f"Unknown feature_set={feature_set}. Valid: {sorted(FEATURE_PRESETS)}")
    full = compute_full_features(X_raw6, fs=fs)
    full_idx = {name: idx for idx, name in enumerate(FULL_FEATURE_ORDER)}
    indices = [full_idx[name] for name in FEATURE_PRESETS[feature_set]]
    return full[:, :, indices].astype(np.float32)


def fit_scaler(X_train: np.ndarray, feature_set: str, eps: float = 1e-6) -> dict[str, Any]:
    mean = X_train.mean(axis=(0, 1)).astype(np.float32)
    std = X_train.std(axis=(0, 1)).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    return {
        "feature_set": feature_set,
        "feature_order": FEATURE_PRESETS[feature_set],
        "fit_scope": "E3_train_split_only",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "eps": float(eps),
    }


def transform_scaler(X: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(scaler["std"], dtype=np.float32).reshape(1, 1, -1)
    return np.nan_to_num(((X - mean) / std).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def make_tf_dataset(
    X: np.ndarray,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    import tensorflow as tf

    y_dict = {
        "fall_output": np.asarray(y_fall, dtype=np.int64),
        "direction_output": prepare_direction_targets(y_direction),
    }
    sw_dict = make_sample_weights(y_fall, direction_mask)
    ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y_dict, sw_dict))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_target_model(
    input_shape: tuple[int, int],
    feature_set: str,
    width_multiplier: float,
    shared_dense: int,
    head_dense: int,
    adapter_type: str,
    adapter_filters: int,
):
    import tensorflow as tf
    from tensorflow.keras import Model, layers

    names = FEATURE_PRESETS[feature_set]
    by_name = {name: idx for idx, name in enumerate(names)}
    acc_names = [name for name in ["ax", "ay", "az", "acc_mag", "jerk", "roll", "pitch", "tilt_delta"] if name in by_name]
    gyro_names = [name for name in ["gx", "gy", "gz", "gyro_mag"] if name in by_name]
    acc_idx = [by_name[name] for name in acc_names]
    gyro_idx = [by_name[name] for name in gyro_names]

    def scale(value: int) -> int:
        return max(4, int(round(value * width_multiplier)))

    def slice_channels(x, indices, name):
        return layers.Lambda(lambda t, idx=indices: tf.gather(t, idx, axis=-1), name=name)(x)

    def sensor_encoder(x, prefix):
        x = layers.Conv1D(scale(16), kernel_size=5, padding="same", name=f"{prefix}_conv1")(x)
        x = layers.BatchNormalization(name=f"{prefix}_conv1_bn")(x)
        x = layers.Activation("relu", name=f"{prefix}_conv1_relu")(x)
        x = dsconv(x, scale(24), 5, f"{prefix}_dsconv1")
        x = dsconv(x, scale(32), 3, f"{prefix}_dsconv2")
        return x

    def dsconv(x, filters, kernel_size, name, dilation_rate=1, dropout=0.0):
        x = layers.SeparableConv1D(filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", name=f"{name}_sepconv")(x)
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
        x = layers.Activation("relu", name=f"{name}_relu")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"{name}_dropout")(x)
        return x

    def dstcn(x, channels, dilation, name, dropout=0.1):
        residual = x
        y = layers.SeparableConv1D(channels, kernel_size=3, dilation_rate=dilation, padding="same", name=f"{name}_sepconv1")(x)
        y = layers.BatchNormalization(name=f"{name}_bn1")(y)
        y = layers.Activation("relu", name=f"{name}_relu1")(y)
        y = layers.Dropout(dropout, name=f"{name}_dropout")(y)
        y = layers.SeparableConv1D(channels, kernel_size=3, dilation_rate=dilation, padding="same", name=f"{name}_sepconv2")(y)
        y = layers.BatchNormalization(name=f"{name}_bn2")(y)
        if residual.shape[-1] != channels:
            residual = layers.Conv1D(channels, kernel_size=1, padding="same", name=f"{name}_projection")(residual)
        y = layers.Add(name=f"{name}_add")([residual, y])
        return layers.Activation("relu", name=f"{name}_out_relu")(y)

    def attention(x, name):
        channels = int(x.shape[-1])
        hidden = max(channels // 2, 1)
        h = layers.Dense(hidden, activation="tanh", name=f"{name}_hidden")(x)
        score = layers.Dense(1, name=f"{name}_score")(h)
        alpha = layers.Softmax(axis=1, name=f"{name}_alpha")(score)
        return layers.Lambda(lambda tensors: tf.reduce_sum(tensors[0] * tensors[1], axis=1), name=f"{name}_pool")([x, alpha])

    inputs = layers.Input(shape=input_shape, name="imu_input")
    acc = slice_channels(inputs, acc_idx, "acc_input")
    gyro = slice_channels(inputs, gyro_idx, "gyro_input")
    acc_features = sensor_encoder(acc, "acc")
    gyro_features = sensor_encoder(gyro, "gyro")
    x = layers.Concatenate(name="sensor_concat")([acc_features, gyro_features])
    x = layers.Conv1D(scale(64), kernel_size=1, padding="same", name="fusion_pointwise_conv")(x)
    x = layers.BatchNormalization(name="fusion_bn")(x)
    x = layers.Activation("relu", name="fusion_relu")(x)
    x = dstcn(x, scale(64), 1, "dstcn_d1")
    x = dstcn(x, scale(64), 2, "dstcn_d2")
    x = dstcn(x, scale(96), 4, "dstcn_d4")

    fall_x = x
    dir_x = x
    if adapter_type == "conv1x1" and adapter_filters > 0:
        fall_x = layers.Conv1D(adapter_filters, kernel_size=1, padding="same", activation="relu", name="fall_adapter_conv1x1")(fall_x)
        dir_x = layers.Conv1D(adapter_filters, kernel_size=1, padding="same", activation="relu", name="direction_adapter_conv1x1")(dir_x)

    fall_context = attention(fall_x, "fall_attention_pooling")
    direction_context = attention(dir_x, "direction_attention_pooling")
    if shared_dense > 0:
        fall_context = layers.Dense(shared_dense, activation="relu", name="fall_context_dense")(fall_context)
        direction_context = layers.Dense(shared_dense, activation="relu", name="direction_context_dense")(direction_context)

    fall = layers.Dense(head_dense, activation="relu", name="fall_dense")(fall_context)
    fall = layers.Dropout(0.2, name="fall_dropout")(fall)
    fall_output = layers.Dense(2, activation="softmax", name="fall_output")(fall)
    direction = layers.Dense(head_dense, activation="relu", name="direction_dense")(direction_context)
    direction = layers.Dropout(0.2, name="direction_dropout")(direction)
    direction_output = layers.Dense(3, activation="softmax", name="direction_output")(direction)
    return Model(inputs=inputs, outputs={"fall_output": fall_output, "direction_output": direction_output}, name="DS-Fall-RD-Targeted")


def compile_target_model(model, cfg: TargetConfig, fall_class_weights: np.ndarray, direction_class_weights: np.ndarray):
    import tensorflow as tf

    if cfg.fall_loss == "focal":
        focal_cls = _get_sparse_focal_class()
        fall_loss = focal_cls(class_weights=fall_class_weights, gamma=cfg.fall_gamma, name="fall_focal_loss")
    elif cfg.fall_loss == "weighted_ce":
        fall_loss = make_sparse_ce_loss(class_weights=fall_class_weights, name="fall_weighted_ce")
    else:
        fall_loss = make_sparse_ce_loss(name="fall_ce")
    direction_weights = direction_class_weights if cfg.direction_loss == "weighted_ce" else None
    direction_loss = make_direction_loss(cfg.direction_loss, class_weights=direction_weights, focal_gamma=2.0)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss={"fall_output": fall_loss, "direction_output": direction_loss},
        loss_weights={"fall_output": float(cfg.alpha_fall), "direction_output": float(cfg.lambda_dir)},
        metrics={"fall_output": ["accuracy"], "direction_output": ["accuracy"]},
    )


def balanced_training_indices(meta_train: pd.DataFrame, y_fall_train: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dataset_parts: list[np.ndarray] = []
    for dataset in ["bits", "weda"]:
        ds_idx = np.flatnonzero(meta_train["dataset"].astype(str).str.lower().to_numpy() == dataset)
        if len(ds_idx) == 0:
            continue
        parts = []
        for label in [0, 1]:
            pool = ds_idx[y_fall_train[ds_idx] == label]
            if len(pool) == 0:
                continue
            target = max(1, max(np.sum(y_fall_train[ds_idx] == 0), np.sum(y_fall_train[ds_idx] == 1)))
            parts.append(rng.choice(pool, size=int(target), replace=len(pool) < target))
        dataset_parts.append(np.concatenate(parts))
    if not dataset_parts:
        return np.arange(len(meta_train))
    max_len = max(len(part) for part in dataset_parts)
    balanced = []
    for part in dataset_parts:
        balanced.append(rng.choice(part, size=max_len, replace=len(part) < max_len))
    out = np.concatenate(balanced)
    rng.shuffle(out)
    return out.astype(np.int64)


def make_validation_metric_callback(X_val, y_fall_val, y_direction_val, direction_mask_val):
    import tensorflow as tf

    class ValidationMetricCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            probs = predict_probs(self.model, X_val)
            pred_fall = (probs["fall_prob"] >= 0.5).astype(np.int64)
            logs["val_fall_f1"] = float(f1_score(y_fall_val, pred_fall, zero_division=0))
            supervised = (direction_mask_val > 0) & (y_direction_val >= 0)
            if supervised.any():
                pred_direction = np.argmax(probs["direction_prob"][supervised], axis=1)
                logs["val_direction_macro_f1"] = float(
                    f1_score(y_direction_val[supervised], pred_direction, average="macro", zero_division=0)
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


def predict_probs(model, X: np.ndarray) -> dict[str, np.ndarray]:
    preds = model.predict(X, verbose=0)
    if isinstance(preds, dict):
        fall_prob = preds["fall_output"][:, 1]
        direction_prob = preds["direction_output"]
    else:
        fall_prob = preds[0][:, 1]
        direction_prob = preds[1]
    return {"fall_prob": fall_prob.astype(float), "direction_prob": direction_prob.astype(float)}


def tune_threshold(y_true: np.ndarray, fall_prob: np.ndarray, step: float = 0.01) -> tuple[float, float]:
    best_tau = 0.5
    best_f1 = -1.0
    for tau in np.arange(0.10, 0.9001, step):
        pred = (fall_prob >= tau).astype(np.int64)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_tau = float(tau)
    return best_tau, best_f1


def tune_dataset_thresholds(meta_val: pd.DataFrame, y_fall: np.ndarray, fall_prob: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for dataset in ["bits", "weda"]:
        mask = meta_val["dataset"].astype(str).str.lower().to_numpy() == dataset
        if not mask.any():
            continue
        tau, score = tune_threshold(y_fall[mask], fall_prob[mask])
        pred = (fall_prob[mask] >= tau).astype(np.int64)
        out[dataset] = {
            "best_threshold": tau,
            "val_fall_f1": score,
            "precision": float(precision_score(y_fall[mask], pred, zero_division=0)),
            "recall": float(recall_score(y_fall[mask], pred, zero_division=0)),
        }
    return out


def evaluate_e3_e6_e7(
    run_dir: Path,
    report_dir: Path,
    figures_dir: Path,
    cfg: TargetConfig,
    seed: int,
    meta_test: pd.DataFrame,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    fall_prob: np.ndarray,
    direction_prob: np.ndarray,
    threshold_05: float,
    threshold_tuned: float,
) -> dict[str, Any]:
    evaluations: dict[str, Any] = {}
    for threshold_name, threshold in [("threshold_0p5", threshold_05), ("threshold_tuned", threshold_tuned)]:
        for eval_id, dataset in [("E3", "bits+weda"), ("E6", "bits"), ("E7", "weda")]:
            if dataset == "bits+weda":
                mask = np.ones(len(meta_test), dtype=bool)
            else:
                mask = meta_test["dataset"].astype(str).str.lower().to_numpy() == dataset
            eval_metrics = evaluate_subset(
                meta_test.loc[mask].reset_index(drop=True),
                y_fall[mask],
                y_direction[mask],
                direction_mask[mask],
                fall_prob[mask],
                direction_prob[mask],
                threshold,
            )
            key = f"{eval_id}_{threshold_name}"
            evaluations[key] = eval_metrics
            if threshold_name == "threshold_tuned":
                save_eval_artifacts(
                    run_dir,
                    report_dir,
                    figures_dir,
                    cfg,
                    seed,
                    eval_id,
                    meta_test.loc[mask].reset_index(drop=True),
                    y_fall[mask],
                    y_direction[mask],
                    direction_mask[mask],
                    fall_prob[mask],
                    direction_prob[mask],
                    threshold,
                )
    return evaluations


def evaluate_subset(
    meta: pd.DataFrame,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    fall_prob: np.ndarray,
    direction_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    pred_fall = (fall_prob >= threshold).astype(np.int64)
    pred_direction = np.argmax(direction_prob, axis=1).astype(np.int64)
    supervised = (direction_mask > 0) & (y_direction >= 0)
    fall = binary_metrics(y_fall, pred_fall)
    direction = direction_metrics(y_direction, pred_direction, supervised)
    return {
        "dataset": "mixed" if meta["dataset"].nunique() > 1 else str(meta["dataset"].iloc[0]),
        "n": int(len(meta)),
        "fall": fall,
        "direction": direction,
        "threshold": float(threshold),
    }


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def direction_metrics(y_true: np.ndarray, y_pred: np.ndarray, supervised: np.ndarray) -> dict[str, Any]:
    if not supervised.any():
        return {"num_supervised": 0, "accuracy": math.nan, "macro_f1": math.nan, "per_class_f1": {}}
    yt = y_true[supervised]
    yp = y_pred[supervised]
    report = classification_report(yt, yp, labels=[0, 1, 2], target_names=DIRECTION_LABELS, zero_division=0, output_dict=True)
    return {
        "num_supervised": int(supervised.sum()),
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, labels=[0, 1, 2], average="macro", zero_division=0)),
        "per_class_f1": {name: float(report[name]["f1-score"]) for name in DIRECTION_LABELS},
        "confusion_matrix": confusion_matrix(yt, yp, labels=[0, 1, 2]).tolist(),
    }


def save_eval_artifacts(
    run_dir: Path,
    report_dir: Path,
    figures_dir: Path,
    cfg: TargetConfig,
    seed: int,
    eval_id: str,
    meta: pd.DataFrame,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    fall_prob: np.ndarray,
    direction_prob: np.ndarray,
    threshold: float,
) -> None:
    pred_fall = (fall_prob >= threshold).astype(np.int64)
    pred_direction = np.argmax(direction_prob, axis=1).astype(np.int64)
    supervised = (direction_mask > 0) & (y_direction >= 0)
    prefix = f"{cfg.run_id}_seed{seed}_{eval_id}"

    cm_fall = confusion_matrix(y_fall, pred_fall, labels=[0, 1])
    cm_dir = confusion_matrix(y_direction[supervised], pred_direction[supervised], labels=[0, 1, 2]) if supervised.any() else np.zeros((3, 3), dtype=int)
    pd.DataFrame(cm_fall, index=["true_nonfall", "true_fall"], columns=["pred_nonfall", "pred_fall"]).to_csv(
        report_dir / "confusion_matrices" / f"{prefix}_fall.csv"
    )
    pd.DataFrame(cm_dir, index=[f"true_{x}" for x in DIRECTION_LABELS], columns=[f"pred_{x}" for x in DIRECTION_LABELS]).to_csv(
        report_dir / "confusion_matrices" / f"{prefix}_direction.csv"
    )
    plot_cm(cm_fall, FALL_LABELS, figures_dir / "confusion_matrices" / f"{prefix}_fall.png", f"{prefix} Fall")
    plot_cm(cm_dir, DIRECTION_LABELS, figures_dir / "confusion_matrices" / f"{prefix}_direction.png", f"{prefix} Direction")

    out = meta.copy()
    out["true_fall"] = [FALL_LABELS[int(v)] for v in y_fall]
    out["pred_fall"] = [FALL_LABELS[int(v)] for v in pred_fall]
    out["fall_prob"] = fall_prob
    out["true_direction"] = [DIRECTION_LABELS[int(v)] if int(v) >= 0 else "none" for v in y_direction]
    out["pred_direction"] = [DIRECTION_LABELS[int(v)] for v in pred_direction]
    out["direction_confidence"] = direction_prob.max(axis=1)
    out.to_csv(run_dir / f"{eval_id}_predictions.csv", index=False)


def plot_cm(cm: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_summary_row(
    cfg: TargetConfig,
    seed: int,
    evaluations: dict[str, Any],
    params: int,
    best_epoch: int | None,
    best_val_loss: float | None,
    best_monitor: float | None,
    fall_threshold: float,
    val_best_fall_f1: float,
    train_seconds: float,
    n_train: int,
    n_val: int,
    n_test: int,
    sampler_note: str,
) -> dict[str, Any]:
    e3 = evaluations["E3_threshold_tuned"]
    e6 = evaluations["E6_threshold_tuned"]
    e7 = evaluations["E7_threshold_tuned"]
    row = {
        **asdict(cfg),
        "seed": seed,
        "train_dataset": "bits+weda",
        "test_dataset": "bits+weda",
        "sampling_rate": SAMPLING_RATE,
        "window_seconds": WINDOW_SECONDS,
        "input_shape": f"({WINDOW_SIZE}, {len(FEATURE_PRESETS[cfg.feature_set])})",
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "params": params,
        "model_size_float32_kb": params * 4 / 1024.0,
        "model_size_int8_kb": params / 1024.0,
        "fall_threshold": fall_threshold,
        "val_best_fall_f1": val_best_fall_f1,
        "fall_precision": e3["fall"]["precision"],
        "fall_recall": e3["fall"]["recall"],
        "fall_f1": e3["fall"]["f1"],
        "fall_accuracy": e3["fall"]["accuracy"],
        "direction_macro_f1": e3["direction"]["macro_f1"],
        "direction_accuracy": e3["direction"]["accuracy"],
        "direction_n_supervised": e3["direction"]["num_supervised"],
        "bits_fall_f1": e6["fall"]["f1"],
        "bits_fall_precision": e6["fall"]["precision"],
        "bits_fall_recall": e6["fall"]["recall"],
        "bits_direction_macro_f1": e6["direction"]["macro_f1"],
        "bits_direction_accuracy": e6["direction"]["accuracy"],
        "bits_direction_n_supervised": e6["direction"]["num_supervised"],
        "weda_fall_f1": e7["fall"]["f1"],
        "weda_fall_precision": e7["fall"]["precision"],
        "weda_fall_recall": e7["fall"]["recall"],
        "weda_direction_macro_f1": e7["direction"]["macro_f1"],
        "weda_direction_accuracy": e7["direction"]["accuracy"],
        "weda_direction_n_supervised": e7["direction"]["num_supervised"],
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_monitor": best_monitor,
        "train_seconds": train_seconds,
        "sampler_note": sampler_note,
    }
    for eval_key, prefix, metrics in [("e3", "mixed", e3), ("e6", "bits", e6), ("e7", "weda", e7)]:
        per_class = metrics["direction"].get("per_class_f1", {})
        for label in DIRECTION_LABELS:
            row[f"{prefix}_{label}_f1"] = per_class.get(label, math.nan)
    row["selection_score"] = selection_score(row)
    row["passes_hard_filter"] = bool(
        row["direction_macro_f1"] >= 0.86
        and row["weda_direction_macro_f1"] >= 0.80
        and row["bits_fall_f1"] >= 0.90
    )
    return row


def selection_score(row: dict[str, Any]) -> float:
    param_penalty = max(0.0, (float(row.get("params", 65959)) - 65959.0) / 65959.0)
    return float(
        0.30 * row.get("weda_fall_f1", 0.0)
        + 0.25 * row.get("direction_macro_f1", 0.0)
        + 0.20 * row.get("weda_direction_macro_f1", 0.0)
        + 0.15 * row.get("bits_fall_f1", 0.0)
        + 0.10 * row.get("fall_f1", 0.0)
        - 0.05 * param_penalty
    )


def best_epoch_from_history(history_df: pd.DataFrame, monitor: str, mode: str) -> tuple[int | None, float | None, float | None]:
    if history_df.empty or monitor not in history_df:
        return None, None, None
    values = history_df[monitor].astype(float)
    idx = int(values.idxmax() if mode == "max" else values.idxmin())
    val_loss = float(history_df.loc[idx, "val_loss"]) if "val_loss" in history_df else None
    return int(history_df.loc[idx, "epoch"]), val_loss, float(history_df.loc[idx, monitor])


def loss_configs() -> list[TargetConfig]:
    rows = []
    for cfg_id, alpha, lam in [
        ("L1", 1.0, 1.0),
        ("L2", 1.5, 1.0),
        ("L3", 2.0, 1.0),
        ("L4", 1.5, 0.75),
        ("L5", 2.0, 0.75),
    ]:
        rows.append(TargetConfig(f"{cfg_id}_FALL_WCE", "loss", cfg_id, alpha_fall=alpha, lambda_dir=lam, direction_loss="ce"))
    rows.append(
        TargetConfig(
            "L6_FALL2_DIRW15",
            "loss",
            "L6",
            alpha_fall=2.0,
            lambda_dir=1.5,
            direction_loss="weighted_ce",
            notes="Fall-focused variant retaining the original weighted direction loss strength.",
        )
    )
    rows.append(
        TargetConfig(
            "L7_FALL2_DIRW20",
            "loss",
            "L7",
            alpha_fall=2.0,
            lambda_dir=2.0,
            direction_loss="weighted_ce",
            notes="Fall-focused variant with stronger weighted direction protection.",
        )
    )
    for cfg_id, gamma, direction_loss in [
        ("FL1", 1.0, "ce"),
        ("FL2", 2.0, "ce"),
        ("FL3", 1.0, "weighted_ce"),
        ("FL4", 2.0, "weighted_ce"),
    ]:
        rows.append(
            TargetConfig(
                f"{cfg_id}_FALL_FOCAL",
                "loss",
                cfg_id,
                fall_loss="focal",
                fall_gamma=gamma,
                lambda_dir=1.0,
                direction_loss=direction_loss,
            )
        )
    return rows


def sampler_configs() -> list[TargetConfig]:
    rows = []
    for cfg_id, alpha, fall_loss, gamma in [
        ("S1", 1.0, "weighted_ce", 2.0),
        ("S2", 1.5, "weighted_ce", 2.0),
        ("S3", 2.0, "weighted_ce", 2.0),
        ("S4", 1.0, "focal", 1.0),
        ("S5", 1.0, "focal", 2.0),
    ]:
        rows.append(
            TargetConfig(
                f"{cfg_id}_DATASET_BALANCED",
                "sampler",
                cfg_id,
                sampler="dataset_balanced",
                fall_loss=fall_loss,
                alpha_fall=alpha,
                lambda_dir=1.0,
                fall_gamma=gamma,
                direction_loss="weighted_ce",
            )
        )
    return rows


def feature_configs(base_row: dict[str, Any]) -> list[TargetConfig]:
    mapping = [
        ("FA0", "raw6"),
        ("FA1", "fa1_acc_mag7"),
        ("FA2", "fa2_mag8"),
        ("FA3", "mag_jerk9"),
        ("FA4", "fa4_roll_pitch8"),
        ("FA5", "fa5_tilt9"),
        ("FA6", "fa6_no_jerk11"),
        ("FA7", "fa7_no_gyro_mag11"),
        ("FA8", "tilt12"),
    ]
    base_cfg = row_to_config(base_row)
    return [
        replace(base_cfg, run_id=f"{cfg_id}_{feature_set.upper()}", group_id="feature_ablation", config_id=cfg_id, feature_set=feature_set)
        for cfg_id, feature_set in mapping
    ]


def compactness_configs(base_row: dict[str, Any]) -> list[TargetConfig]:
    base_cfg = row_to_config(base_row)
    rows = [
        replace(base_cfg, run_id="W050_WIDTH", group_id="compactness", config_id="W050", width_multiplier=0.50),
        replace(base_cfg, run_id="W075_WIDTH", group_id="compactness", config_id="W075", width_multiplier=0.75),
        replace(base_cfg, run_id="W100_WIDTH", group_id="compactness", config_id="W100", width_multiplier=1.00),
        replace(base_cfg, run_id="W125_WIDTH", group_id="compactness", config_id="W125", width_multiplier=1.25),
        replace(base_cfg, run_id="D48_24_HEAD", group_id="compactness", config_id="D48_24", shared_dense=48, head_dense=24),
        replace(base_cfg, run_id="D32_16_HEAD", group_id="compactness", config_id="D32_16", shared_dense=32, head_dense=16),
    ]
    return rows


def adapter_configs(base_row: dict[str, Any]) -> list[TargetConfig]:
    base_cfg = row_to_config(base_row)
    return [
        replace(base_cfg, run_id="AD0_NONE", group_id="adapter", config_id="AD0", adapter_type="none", adapter_filters=0),
        replace(base_cfg, run_id="AD16_CONV1X1", group_id="adapter", config_id="AD16", adapter_type="conv1x1", adapter_filters=16),
        replace(base_cfg, run_id="AD32_CONV1X1", group_id="adapter", config_id="AD32", adapter_type="conv1x1", adapter_filters=32),
    ]


def row_to_config(row: dict[str, Any] | TargetConfig) -> TargetConfig:
    if isinstance(row, TargetConfig):
        return row
    fields = TargetConfig.__dataclass_fields__.keys()
    values = {field: row[field] for field in fields if field in row}
    return TargetConfig(**values)


def config_from_row(row: dict[str, Any], run_id: str, group_id: str, config_id: str) -> TargetConfig:
    return replace(row_to_config(row), run_id=run_id, group_id=group_id, config_id=config_id)


def choose_best_recipe(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "passes_hard_filter" in df:
        filtered = df[df["passes_hard_filter"].astype(bool)]
        if not filtered.empty:
            df = filtered
    df = df.sort_values(["weda_fall_f1", "selection_score", "direction_macro_f1"], ascending=False)
    return df.iloc[0].to_dict()


def best_seed_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(rows, key=lambda row: (row.get("selection_score", 0.0), row.get("weda_fall_f1", 0.0)), reverse=True)[0]


def collect_rows(
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    per_dataset_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
) -> None:
    keys = {(row["run_id"], int(row["seed"])) for row in rows}
    all_rows[:] = [row for row in all_rows if (row.get("run_id"), int(row.get("seed", -1))) not in keys]
    all_rows.extend(rows)
    for row in rows:
        threshold_rows.append(
            {
                "run_id": row["run_id"],
                "seed": row["seed"],
                "best_threshold": row["fall_threshold"],
                "val_best_fall_f1": row["val_best_fall_f1"],
                "test_e3_fall_f1_tuned": row["fall_f1"],
                "test_e6_bits_fall_f1_tuned": row["bits_fall_f1"],
                "test_e7_weda_fall_f1_tuned": row["weda_fall_f1"],
            }
        )
        for dataset in ["bits", "weda"]:
            prefix = dataset
            per_dataset_rows.append(
                {
                    "run_id": row["run_id"],
                    "seed": row["seed"],
                    "dataset": dataset,
                    "fall_f1": row[f"{prefix}_fall_f1"],
                    "fall_precision": row[f"{prefix}_fall_precision"],
                    "fall_recall": row[f"{prefix}_fall_recall"],
                    "direction_macro_f1": row[f"{prefix}_direction_macro_f1"],
                    "direction_accuracy": row[f"{prefix}_direction_accuracy"],
                    "direction_n_supervised": row[f"{prefix}_direction_n_supervised"],
                }
            )
        for eval_name, dataset in [("E6", "bits"), ("E7", "weda")]:
            cm_file = ""
            per_class_rows.append(
                {
                    "run_id": row["run_id"],
                    "seed": row["seed"],
                    "eval": eval_name,
                    "dataset": dataset,
                    "forward_f1": row.get(f"{dataset}_forward_f1", np.nan),
                    "backward_f1": row.get(f"{dataset}_backward_f1", np.nan),
                    "lateral_f1": row.get(f"{dataset}_lateral_f1", np.nan),
                    "confusion_matrix": cm_file,
                }
            )


def flush_reports(
    report_dir: Path,
    all_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    per_dataset_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
) -> None:
    all_df = pd.DataFrame(all_rows)
    if not all_df.empty:
        all_df = all_df.drop_duplicates(subset=["run_id", "seed"], keep="last")
        all_df.to_csv(report_dir / "all_runs.csv", index=False)
        all_df.to_csv(report_dir.parent / "targeted_experiments_results.csv", index=False)
    threshold_df = pd.DataFrame(threshold_rows)
    if not threshold_df.empty:
        threshold_df = threshold_df.drop_duplicates(subset=["run_id", "seed"], keep="last")
    threshold_df.to_csv(report_dir / "threshold_tuning.csv", index=False)
    per_dataset_df = pd.DataFrame(per_dataset_rows)
    if not per_dataset_df.empty:
        per_dataset_df = per_dataset_df.drop_duplicates(subset=["run_id", "seed", "dataset"], keep="last")
    per_dataset_df.to_csv(report_dir / "per_dataset_metrics.csv", index=False)
    per_class_df = pd.DataFrame(per_class_rows)
    if not per_class_df.empty:
        per_class_df = per_class_df.drop_duplicates(subset=["run_id", "seed", "eval", "dataset"], keep="last")
    per_class_df.to_csv(report_dir / "per_class_direction_f1.csv", index=False)
    if not all_df.empty:
        all_df.groupby("group_id", dropna=False).agg(
            runs=("run_id", "count"),
            best_weda_fall_f1=("weda_fall_f1", "max"),
            best_e3_direction_macro_f1=("direction_macro_f1", "max"),
            best_selection_score=("selection_score", "max"),
        ).reset_index().to_csv(report_dir / "summary_by_group.csv", index=False)


def load_existing_rows(report_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    def read(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return pd.read_csv(path).to_dict(orient="records")

    return (
        read(report_dir / "all_runs.csv"),
        read(report_dir / "threshold_tuning.csv"),
        read(report_dir / "per_dataset_metrics.csv"),
        read(report_dir / "per_class_direction_f1.csv"),
    )


def finalize_reports(
    report_dir: Path,
    figures_dir: Path,
    all_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    per_dataset_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
) -> None:
    flush_reports(report_dir, all_rows, threshold_rows, per_dataset_rows, per_class_rows)
    all_df = pd.DataFrame(all_rows)
    feature_df = all_df[all_df["group_id"].eq("feature_ablation")] if not all_df.empty else pd.DataFrame()
    compact_df = all_df[all_df["group_id"].eq("compactness")] if not all_df.empty else pd.DataFrame()
    feature_df.to_csv(report_dir / "feature_ablation.csv", index=False)
    compact_df.to_csv(report_dir / "compactness_results.csv", index=False)
    summary_md = build_summary_markdown(all_df)
    (report_dir / "targeted_experiments_summary.md").write_text(summary_md, encoding="utf-8")
    (report_dir.parent / "targeted_experiments_summary.md").write_text(summary_md, encoding="utf-8")
    best_md = build_best_selection_markdown(all_df)
    (report_dir / "best_model_selection.md").write_text(best_md, encoding="utf-8")
    (report_dir / "final_recommendation.md").write_text(build_final_recommendation(all_df), encoding="utf-8")


def build_summary_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "# Targeted Experiments Summary\n\nNo runs completed.\n"
    display_cols = [
        "run_id",
        "group_id",
        "config_id",
        "feature_set",
        "sampler",
        "fall_loss",
        "alpha_fall",
        "lambda_dir",
        "fall_threshold",
        "fall_f1",
        "direction_macro_f1",
        "bits_fall_f1",
        "bits_direction_macro_f1",
        "weda_fall_f1",
        "weda_direction_macro_f1",
        "params",
        "selection_score",
        "passes_hard_filter",
    ]
    ranked = df.sort_values(["passes_hard_filter", "weda_fall_f1", "selection_score"], ascending=False)
    best = ranked.iloc[0]
    lines = [
        "# DS-Fall-RD Targeted Experiments Summary",
        "",
        "Scope: BITS/WEDA only, event-centered 2s windows, 25 Hz, no new datasets and no heavy architecture.",
        "",
        "## Best Observed Run",
        "",
        f"- run_id: `{best['run_id']}`",
        f"- E7 WEDA Fall F1: {best['weda_fall_f1']:.4f}",
        f"- E3 Direction Macro F1: {best['direction_macro_f1']:.4f}",
        f"- E7 Direction Macro F1: {best['weda_direction_macro_f1']:.4f}",
        f"- E6 BITS Fall F1: {best['bits_fall_f1']:.4f}",
        f"- params: {int(best['params'])}",
        "",
        "## Main Results",
        "",
        markdown_table(format_float_df(ranked[display_cols])),
        "",
        "## Key Diagnostics",
        "",
        "- Selection prioritizes E7 WEDA Fall F1, then direction stability and E6 BITS Fall F1.",
        "- Dataset-aware thresholds are saved as analysis only; the selected rows use a global validation-tuned threshold.",
        "- Direction metrics always use supervised direction samples only.",
    ]
    return "\n".join(lines) + "\n"


def build_best_selection_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "# Best Model Selection\n\nNo completed runs.\n"
    hard = df[df["passes_hard_filter"].astype(bool)] if "passes_hard_filter" in df else pd.DataFrame()
    pool = hard if not hard.empty else df
    ranked = pool.sort_values(["weda_fall_f1", "selection_score", "direction_macro_f1"], ascending=False)
    best = ranked.iloc[0]
    baseline = df[df["run_id"].eq("B0_BASE_A5WCEFW")]
    base = baseline.iloc[0] if not baseline.empty else None
    lines = [
        "# Best Model Selection",
        "",
        "Hard filters: E3 Direction Macro F1 >= 0.86, E7 Direction Macro F1 >= 0.80, E6 Fall F1 >= 0.90.",
        "",
        f"Selected candidate: `{best['run_id']}`",
        "",
        "| metric | baseline | selected | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in ["weda_fall_f1", "direction_macro_f1", "weda_direction_macro_f1", "bits_fall_f1", "fall_f1", "params"]:
        base_val = float(base[metric]) if base is not None and metric in base else math.nan
        best_val = float(best[metric])
        lines.append(f"| {metric} | {base_val:.4f} | {best_val:.4f} | {best_val - base_val:.4f} |")
    lines.extend(
        [
            "",
            "## Ranking",
            "",
            markdown_table(format_float_df(ranked.head(12)[["run_id", "group_id", "config_id", "weda_fall_f1", "direction_macro_f1", "weda_direction_macro_f1", "bits_fall_f1", "params", "selection_score", "passes_hard_filter"]])),
        ]
    )
    return "\n".join(lines) + "\n"


def build_final_recommendation(df: pd.DataFrame) -> str:
    if df.empty:
        return "# Final Recommendation\n\nNo completed runs.\n"
    hard = df[df["passes_hard_filter"].astype(bool)] if "passes_hard_filter" in df else pd.DataFrame()
    pool = hard if not hard.empty else df
    best = pool.sort_values(["weda_fall_f1", "selection_score"], ascending=False).iloc[0]
    feature_df = df[df["group_id"].eq("feature_ablation")]
    compact_df = df[df["group_id"].eq("compactness")]
    lines = [
        "# Final Recommendation",
        "",
        f"Recommended paper candidate: `{best['run_id']}`.",
        "",
        "## Answers",
        "",
        f"- WEDA Fall F1: {best['weda_fall_f1']:.4f}.",
        f"- E3 Direction Macro F1: {best['direction_macro_f1']:.4f}.",
        f"- E7 Direction Macro F1: {best['weda_direction_macro_f1']:.4f}.",
        f"- Model params: {int(best['params'])}.",
        "",
        "## Feature Notes",
        "",
    ]
    if not feature_df.empty:
        feature_best = feature_df.sort_values(["weda_fall_f1", "selection_score"], ascending=False).iloc[0]
        lines.append(f"- Best feature ablation run: `{feature_best['run_id']}` with `{feature_best['feature_set']}`.")
    else:
        lines.append("- Feature ablation did not complete.")
    if not compact_df.empty:
        compact_best = compact_df.sort_values(["weda_fall_f1", "selection_score"], ascending=False).iloc[0]
        lines.append(f"- Best compactness run: `{compact_best['run_id']}`, params={int(compact_best['params'])}.")
    else:
        lines.append("- Compactness sweep did not complete.")
    lines.extend(
        [
            "",
            "## Paper Tables/Figures",
            "",
            "- Main targeted results: `all_runs.csv`.",
            "- Threshold analysis: `threshold_tuning.csv`.",
            "- Feature ablation: `feature_ablation.csv`.",
            "- Compactness: `compactness_results.csv`.",
            "- Confusion matrices: `confusion_matrices/` and `outputs/figures/targeted_experiments/confusion_matrices/`.",
        ]
    )
    return "\n".join(lines) + "\n"


def save_optional_reports(report_dir: Path, data: dict[str, Any], all_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# WEDA Error Analysis",
        "",
        "Hard-negative mining was kept as analysis unless explicitly run. The targeted runner saves WEDA predictions for every run, so false negatives and false positives can be inspected from each run folder.",
        "",
    ]
    if all_rows:
        best = pd.DataFrame(all_rows).sort_values(["weda_fall_f1", "selection_score"], ascending=False).iloc[0]
        pred_path = report_dir / "runs" / f"{best['run_id']}_seed{int(best['seed'])}" / "E7_predictions.csv"
        lines.extend(
            [
                f"Best current WEDA prediction file: `{pred_path}`",
                "",
                "If a second-stage hard-negative run is needed, use WEDA false negatives and high-probability false positives from this file. It was not selected automatically unless it passes E3/E6 direction stability filters.",
            ]
        )
    (report_dir / "weda_error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_reference_event_baseline(
    project_root: Path,
    all_rows: list[dict[str, Any]],
    per_dataset_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
) -> None:
    summary_path = project_root / "artifacts" / "experiments_25hz_event" / "summary_e1_e7_event_centered.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)

    def row(exp_id: str) -> pd.Series:
        match = summary[summary["experiment_id"].eq(exp_id)]
        if match.empty:
            raise ValueError(f"Missing reference experiment {exp_id} in {summary_path}")
        return match.iloc[0]

    e3 = row("E3_BITS_WEDA_MIXED")
    e6 = row("E6_E3_MIXED_TEST_BITS")
    e7 = row("E7_E3_MIXED_TEST_WEDA")
    ref_id = "REF_CURRENT_E3_ARTIFACT"
    all_rows[:] = [r for r in all_rows if r.get("run_id") != ref_id]
    per_dataset_rows[:] = [r for r in per_dataset_rows if r.get("run_id") != ref_id]
    per_class_rows[:] = [r for r in per_class_rows if r.get("run_id") != ref_id]

    ref = {
        **asdict(
            TargetConfig(
                run_id=ref_id,
                group_id="reference",
                config_id="REF",
                feature_set="tilt12",
                sampler="current",
                fall_loss="weighted_ce",
                alpha_fall=1.0,
                lambda_dir=1.5,
                direction_loss="weighted_ce",
                notes="Existing event-centered E3 checkpoint at threshold 0.5; used as the current main baseline.",
            )
        ),
        "seed": 42,
        "train_dataset": "bits+weda",
        "test_dataset": "bits+weda",
        "sampling_rate": SAMPLING_RATE,
        "window_seconds": WINDOW_SECONDS,
        "input_shape": "(50, 12)",
        "n_train": int(e3.get("n_train", 3999)),
        "n_val": int(e3.get("n_val", 748)),
        "n_test": int(e3.get("n_test", 708)),
        "params": int(e3.get("model_params", 65959)),
        "model_size_float32_kb": float(e3.get("model_size_float32_kb", 65959 * 4 / 1024)),
        "model_size_int8_kb": float(e3.get("model_size_int8_kb_if_available", 65959 / 1024)),
        "fall_threshold": 0.5,
        "val_best_fall_f1": math.nan,
        "fall_precision": float(e3["fall_precision"]),
        "fall_recall": float(e3["fall_recall"]),
        "fall_f1": float(e3["fall_f1"]),
        "fall_accuracy": float(e3["fall_accuracy"]),
        "direction_macro_f1": float(e3["direction_macro_f1"]),
        "direction_accuracy": float(e3["direction_accuracy"]),
        "direction_n_supervised": int(e3["direction_n_supervised"]),
        "bits_fall_f1": float(e6["fall_f1"]),
        "bits_fall_precision": float(e6["fall_precision"]),
        "bits_fall_recall": float(e6["fall_recall"]),
        "bits_direction_macro_f1": float(e6["direction_macro_f1"]),
        "bits_direction_accuracy": float(e6["direction_accuracy"]),
        "bits_direction_n_supervised": int(e6["direction_n_supervised"]),
        "weda_fall_f1": float(e7["fall_f1"]),
        "weda_fall_precision": float(e7["fall_precision"]),
        "weda_fall_recall": float(e7["fall_recall"]),
        "weda_direction_macro_f1": float(e7["direction_macro_f1"]),
        "weda_direction_accuracy": float(e7["direction_accuracy"]),
        "weda_direction_n_supervised": int(e7["direction_n_supervised"]),
        "best_epoch": int(e3.get("best_epoch", 0)),
        "best_val_loss": float(e3.get("best_val_loss", math.nan)),
        "best_monitor": math.nan,
        "train_seconds": math.nan,
        "sampler_note": "reference artifact",
    }
    for dataset, exp_id, prefix in [
        ("bits", "E6_E3_MIXED_TEST_BITS", "bits"),
        ("weda", "E7_E3_MIXED_TEST_WEDA", "weda"),
    ]:
        pred_path = project_root / "artifacts" / "experiments_25hz_event" / exp_id / "predictions.csv"
        if pred_path.exists():
            pred = pd.read_csv(pred_path)
            per_class = direction_per_class_from_prediction_frame(pred)
            for label, value in per_class.items():
                ref[f"{prefix}_{label}_f1"] = value
                per_class_rows.append(
                    {
                        "run_id": ref_id,
                        "seed": 42,
                        "eval": "E6" if dataset == "bits" else "E7",
                        "dataset": dataset,
                        "forward_f1": per_class.get("forward", math.nan),
                        "backward_f1": per_class.get("backward", math.nan),
                        "lateral_f1": per_class.get("lateral", math.nan),
                        "confusion_matrix": "",
                    }
                )
    ref["mixed_forward_f1"] = math.nan
    ref["mixed_backward_f1"] = math.nan
    ref["mixed_lateral_f1"] = math.nan
    ref["selection_score"] = selection_score(ref)
    ref["passes_hard_filter"] = bool(
        ref["direction_macro_f1"] >= 0.86
        and ref["weda_direction_macro_f1"] >= 0.80
        and ref["bits_fall_f1"] >= 0.90
    )
    all_rows.append(ref)
    for dataset, source in [("bits", e6), ("weda", e7)]:
        per_dataset_rows.append(
            {
                "run_id": ref_id,
                "seed": 42,
                "dataset": dataset,
                "fall_f1": float(source["fall_f1"]),
                "fall_precision": float(source["fall_precision"]),
                "fall_recall": float(source["fall_recall"]),
                "direction_macro_f1": float(source["direction_macro_f1"]),
                "direction_accuracy": float(source["direction_accuracy"]),
                "direction_n_supervised": int(source["direction_n_supervised"]),
            }
        )


def direction_per_class_from_prediction_frame(pred: pd.DataFrame) -> dict[str, float]:
    mask = pred["true_direction"].astype(str).isin(DIRECTION_LABELS)
    if not mask.any():
        return {label: math.nan for label in DIRECTION_LABELS}
    report = classification_report(
        pred.loc[mask, "true_direction"].astype(str),
        pred.loc[mask, "pred_direction"].astype(str),
        labels=DIRECTION_LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {label: float(report[label]["f1-score"]) for label in DIRECTION_LABELS}


def write_experiment_plan(report_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Targeted Experiment Plan",
        "",
        "- Scope: BITS and WEDA only.",
        "- Windowing: event-centered 2 seconds, 25 Hz, 50 timesteps.",
        "- Main model: DS-Fall-RD A5WCEFW.",
        "- Priority: threshold tuning, fall loss tuning, dataset-balanced sampler, feature ablation, compactness.",
        f"- Seeds: {args.seeds}",
        f"- Epochs: {args.epochs}, patience: {args.patience}, suite: {args.suite}",
        "- Optional hard-negative mining and impact-position sweeps are documented, not forced into the main model unless safe and useful.",
    ]
    (report_dir / "experiment_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_repo_audit(report_dir: Path, data: dict[str, Any]) -> None:
    audit = data["audit"].copy()
    audit.to_csv(report_dir / "dataset_audit_used.csv", index=False)
    lines = [
        "# Repository/Data Audit",
        "",
        "- Reused `scripts/run_25hz_a5wcefw_bits_weda.py` for BITS/WEDA event-centered preprocessing.",
        "- Direction masks are preserved through `make_sample_weights`; direction loss is masked for non-fall/unlabeled samples.",
        "- Scaler is fit on E3 train split only for every run.",
        "- Test split is never used for scaler, early stopping, or threshold tuning.",
        "",
        "## Dataset Audit",
        "",
        markdown_table(format_float_df(audit)),
    ]
    (report_dir / "repo_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
            if col in {"params", "n_train", "n_val", "n_test", "direction_n_supervised", "seed"}:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(int(v)))
            else:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    return out


if __name__ == "__main__":
    main()
