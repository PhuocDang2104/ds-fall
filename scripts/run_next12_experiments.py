from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_25hz_a5wcefw_bits_weda as base25
from scripts.run_ds_fall_rd_targeted_experiments import (
    FEATURE_PRESETS,
    compute_feature_set,
    fit_scaler,
    transform_scaler,
    tune_threshold,
)
from src.config import make_config
from src.models.losses import (
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
REF_METRICS = {
    "fall_f1": 0.8322981366459627,
    "direction_macro_f1": 0.8966626725247414,
    "bits_fall_f1": 0.945054945054945,
    "bits_direction_macro_f1": 0.9350469350469351,
    "weda_fall_f1": 0.6857142857142857,
    "weda_direction_macro_f1": 0.8390350877192981,
    "params": 65959,
}


@dataclass(frozen=True)
class NextConfig:
    run_id: str
    block: str
    alpha_fall: float = 1.0
    lambda_dir: float = 1.5
    sampler: str = "current"
    feature_set: str = "tilt12"
    fall_loss: str = "weighted_ce"
    direction_loss: str = "weighted_ce"
    adapter_mode: str = "none"
    train_mode: str = "standard"
    beta_kd: float = 0.0
    width_multiplier: float = 1.0
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed next-12 DS-Fall-RD experiments.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--teacher-epochs", type=int, default=80)
    parser.add_argument("--teacher-patience", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=25)
    parser.add_argument("--finetune-patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only-run-ids", nargs="*", default=None)
    return parser.parse_args()


def next12_configs() -> list[NextConfig]:
    return [
        NextConfig("B1", "loss_sampler_near_reference", alpha_fall=1.25, lambda_dir=1.50),
        NextConfig("B2", "loss_sampler_near_reference", alpha_fall=1.50, lambda_dir=1.50),
        NextConfig("B3", "loss_sampler_near_reference", alpha_fall=1.25, lambda_dir=1.25),
        NextConfig("C1", "loss_sampler_near_reference", alpha_fall=1.00, lambda_dir=1.50, sampler="dataset_balanced"),
        NextConfig("C2", "loss_sampler_near_reference", alpha_fall=1.25, lambda_dir=1.50, sampler="dataset_balanced"),
        NextConfig("C3", "loss_sampler_near_reference", alpha_fall=1.50, lambda_dir=1.50, sampler="dataset_balanced"),
        NextConfig(
            "D1",
            "direction_preserving_finetune",
            train_mode="fall_head_finetune",
            notes="Old Lambda checkpoint is not loadable; starts from an internally retrained serializable reference teacher.",
        ),
        NextConfig(
            "D2",
            "direction_preserving_finetune",
            train_mode="fall_head_finetune",
            adapter_mode="fall16",
            notes="Fall adapter Conv1D 1x1 16 filters; encoder and direction branch frozen.",
        ),
        NextConfig(
            "D3",
            "direction_preserving_finetune",
            alpha_fall=2.0,
            lambda_dir=1.0,
            direction_loss="ce",
            train_mode="kd_finetune",
            beta_kd=1.0,
            notes="L3-style fall-prioritized fine-tune with direction distillation beta=1.",
        ),
        NextConfig(
            "D4",
            "direction_preserving_finetune",
            alpha_fall=2.0,
            lambda_dir=1.0,
            direction_loss="ce",
            train_mode="kd_finetune",
            beta_kd=2.0,
            notes="L3-style fall-prioritized fine-tune with direction distillation beta=2.",
        ),
        NextConfig("E1", "lightweight_branch_separation", adapter_mode="both16"),
        NextConfig(
            "F1",
            "lightweight_branch_separation",
            train_mode="head_specific_feature_routing",
            width_multiplier=0.5,
            notes="Fall route uses mag_jerk9; direction route uses tilt12; width=0.5 to keep params <90k.",
        ),
    ]


def main() -> None:
    args = parse_args()
    project_config = make_config(args.project_root)
    report_dir = ensure_dir(project_config.output_dir / "reports" / "next12_experiments")
    figure_dir = ensure_dir(project_config.output_dir / "figures" / "next12_experiments" / "confusion_matrices")
    run_root = ensure_dir(report_dir / "runs")
    cm_dir = ensure_dir(report_dir / "confusion_matrices")

    set_seed(args.seed)
    write_plan(report_dir, args)

    print("=== Next 12 DS-Fall-RD experiments ===")
    print(f"project_root={project_config.project_root}")
    print(f"benchmark=BITS/WEDA event-centered, {WINDOW_SECONDS:g}s, {SAMPLING_RATE:g}Hz, tilt12")

    data = base25.build_25hz_dataset(project_config, window_mode="event_centered")
    base25.validate_25hz_dataset(data)
    prepared = prepare_data(data, args.seed)
    save_json(report_dir / "split_counts.json", prepared["split_counts"])

    load_note = check_old_artifact_loadability(project_config.project_root)
    (report_dir / "artifact_load_note.md").write_text(load_note, encoding="utf-8")

    configs = next12_configs()
    only = set(args.only_run_ids or [])
    configs = [cfg for cfg in configs if not only or cfg.run_id in only]

    teacher_bundle = None
    if any(cfg.train_mode in {"fall_head_finetune", "kd_finetune"} for cfg in configs):
        teacher_bundle = train_or_load_teacher(args, prepared, run_root, force=args.force)

    all_rows: list[dict[str, Any]] = []
    per_dataset_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []

    for cfg in configs:
        row, ds_rows, cls_rows = run_or_load_experiment(
            cfg=cfg,
            args=args,
            prepared=prepared,
            run_root=run_root,
            report_dir=report_dir,
            figure_dir=figure_dir,
            cm_dir=cm_dir,
            teacher_bundle=teacher_bundle,
            force=args.force,
        )
        all_rows.append(row)
        per_dataset_rows.extend(ds_rows)
        per_class_rows.extend(cls_rows)
        flush_outputs(report_dir, all_rows, per_dataset_rows, per_class_rows)

    flush_outputs(report_dir, all_rows, per_dataset_rows, per_class_rows)
    build_reports(report_dir, all_rows)


def prepare_data(data: dict[str, Any], seed: int) -> dict[str, Any]:
    metadata = data["metadata"].copy().reset_index(drop=True)
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
    X_features = compute_feature_set(data["X_raw6"], "tilt12", fs=SAMPLING_RATE)
    scaler = fit_scaler(X_features[train_mask], "tilt12")
    X_scaled = transform_scaler(X_features, scaler)
    y_fall = data["y_fall"]
    y_direction = data["y_direction"]
    direction_mask = data["direction_mask"]
    split_counts = {
        "train": int(train_mask.sum()),
        "val": int(val_mask.sum()),
        "test": int(test_mask.sum()),
        "train_direction_supervised": int(((direction_mask[train_mask] > 0) & (y_direction[train_mask] >= 0)).sum()),
        "val_direction_supervised": int(((direction_mask[val_mask] > 0) & (y_direction[val_mask] >= 0)).sum()),
        "test_direction_supervised": int(((direction_mask[test_mask] > 0) & (y_direction[test_mask] >= 0)).sum()),
    }
    return {
        "metadata": metadata,
        "X": X_scaled.astype(np.float32),
        "y_fall": y_fall.astype(np.int64),
        "y_direction": y_direction.astype(np.int64),
        "direction_mask": direction_mask.astype(np.float32),
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "scaler": scaler,
        "split_counts": split_counts,
    }


def train_or_load_teacher(args: argparse.Namespace, prepared: dict[str, Any], run_root: Path, force: bool) -> dict[str, Any]:
    run_dir = ensure_dir(run_root / "_internal_serializable_reference_teacher")
    row_path = run_dir / "teacher_info.json"
    import tensorflow as tf

    tf.keras.backend.clear_session()
    gc.collect()
    set_seed(args.seed)

    X, yf, yd, dm = prepared["X"], prepared["y_fall"], prepared["y_direction"], prepared["direction_mask"]
    tr, va = prepared["train_mask"], prepared["val_mask"]
    model = build_standard_model(adapter_mode="none", kd_output=False, width_multiplier=1.0)
    fall_weights = compute_class_weights(yf[tr], num_classes=2)
    dir_weights = compute_class_weights(yd[tr], num_classes=3, mask=dm[tr] * (yf[tr] == 1))
    compile_model(model, alpha_fall=1.0, lambda_dir=1.5, fall_weights=fall_weights, dir_weights=dir_weights)

    if row_path.exists() and not force and (run_dir / "model.weights.h5").exists():
        model.load_weights(run_dir / "model.weights.h5")
        with row_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        return {"model": model, "info": info, "weights_path": str(run_dir / "model.weights.h5")}

    train_ds = make_dataset(X[tr], yf[tr], yd[tr], dm[tr], args.batch_size, shuffle=True, seed=args.seed)
    val_ds = make_dataset(X[va], yf[va], yd[va], dm[va], args.batch_size, shuffle=False, seed=args.seed)
    callbacks = training_callbacks(run_dir, X[va], yf[va], yd[va], dm[va], patience=args.teacher_patience)
    t0 = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.teacher_epochs, callbacks=callbacks, verbose=1)
    train_seconds = time.perf_counter() - t0
    history_df = history_frame(history)
    history_df.to_csv(run_dir / "training_history.csv", index=False)
    model.save_weights(run_dir / "model.weights.h5")
    info = {
        "purpose": "internal serializable reference teacher for D1-D4 only",
        "seed": args.seed,
        "epochs": args.teacher_epochs,
        "patience": args.teacher_patience,
        "train_seconds": train_seconds,
        "params": int(model.count_params()),
        "note": "Old E3 artifact could not be loaded because of Lambda-layer deserialization; teacher was retrained with serializable ChannelSlice layers.",
    }
    save_json(row_path, info)
    return {"model": model, "info": info, "weights_path": str(run_dir / "model.weights.h5")}


def run_or_load_experiment(
    cfg: NextConfig,
    args: argparse.Namespace,
    prepared: dict[str, Any],
    run_root: Path,
    report_dir: Path,
    figure_dir: Path,
    cm_dir: Path,
    teacher_bundle: dict[str, Any] | None,
    force: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = ensure_dir(run_root / cfg.run_id)
    row_path = run_dir / "summary_row.json"
    if row_path.exists() and not force:
        print(f"Reusing {cfg.run_id}")
        with row_path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        return row, read_rows(run_dir / "per_dataset_metrics.csv"), read_rows(run_dir / "per_class_direction_f1.csv")
    return run_experiment(cfg, args, prepared, run_dir, report_dir, figure_dir, cm_dir, teacher_bundle)


def run_experiment(
    cfg: NextConfig,
    args: argparse.Namespace,
    prepared: dict[str, Any],
    run_dir: Path,
    report_dir: Path,
    figure_dir: Path,
    cm_dir: Path,
    teacher_bundle: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    import tensorflow as tf

    tf.keras.backend.clear_session()
    gc.collect()
    set_seed(args.seed)
    print(f"\n=== {cfg.run_id}: {cfg.block} ===")

    X, yf, yd, dm = prepared["X"], prepared["y_fall"], prepared["y_direction"], prepared["direction_mask"]
    tr, va, te = prepared["train_mask"], prepared["val_mask"], prepared["test_mask"]
    meta = prepared["metadata"]

    if cfg.train_mode == "head_specific_feature_routing":
        model = build_head_specific_model(width_multiplier=cfg.width_multiplier)
    else:
        kd_output = cfg.train_mode == "kd_finetune"
        model = build_standard_model(adapter_mode=cfg.adapter_mode, kd_output=kd_output, width_multiplier=cfg.width_multiplier)

    teacher_model = None
    if cfg.train_mode in {"fall_head_finetune", "kd_finetune"}:
        if teacher_bundle is None:
            raise RuntimeError(f"{cfg.run_id} requires teacher_bundle.")
        teacher_model = build_standard_model(adapter_mode="none", kd_output=False, width_multiplier=1.0)
        teacher_model.load_weights(teacher_bundle["weights_path"])
        copy_matching_weights(teacher_model, model)

    fall_weights = compute_class_weights(yf[tr], num_classes=2)
    dir_weights = compute_class_weights(yd[tr], num_classes=3, mask=dm[tr] * (yf[tr] == 1))
    compile_for_config(model, cfg, fall_weights, dir_weights)

    if cfg.train_mode == "fall_head_finetune":
        freeze_for_fall_finetune(model)
        compile_for_config(model, cfg, fall_weights, dir_weights)

    if cfg.train_mode == "kd_finetune":
        if teacher_bundle is None or teacher_model is None:
            raise RuntimeError("KD requires teacher model.")
        teacher_train = predict_direction(teacher_model, X[tr])
        teacher_val = predict_direction(teacher_model, X[va])
        train_ds = make_dataset(
            X[tr],
            yf[tr],
            yd[tr],
            dm[tr],
            args.batch_size,
            shuffle=True,
            seed=args.seed,
            teacher_direction=teacher_train,
        )
        val_ds = make_dataset(
            X[va],
            yf[va],
            yd[va],
            dm[va],
            args.batch_size,
            shuffle=False,
            seed=args.seed,
            teacher_direction=teacher_val,
        )
        epochs, patience = args.finetune_epochs, args.finetune_patience
    else:
        if cfg.sampler == "dataset_balanced":
            train_idx = balanced_training_indices(meta.loc[tr].reset_index(drop=True), yf[tr], seed=args.seed)
            X_train, yf_train, yd_train, dm_train = X[tr][train_idx], yf[tr][train_idx], yd[tr][train_idx], dm[tr][train_idx]
        else:
            X_train, yf_train, yd_train, dm_train = X[tr], yf[tr], yd[tr], dm[tr]
        train_ds = make_dataset(X_train, yf_train, yd_train, dm_train, args.batch_size, shuffle=True, seed=args.seed)
        val_ds = make_dataset(X[va], yf[va], yd[va], dm[va], args.batch_size, shuffle=False, seed=args.seed)
        if cfg.train_mode == "fall_head_finetune":
            epochs, patience = args.finetune_epochs, args.finetune_patience
        else:
            epochs, patience = args.epochs, args.patience

    callbacks = training_callbacks(run_dir, X[va], yf[va], yd[va], dm[va], patience=patience)
    t0 = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)
    train_seconds = time.perf_counter() - t0
    history_df = history_frame(history)
    history_df.to_csv(run_dir / "training_history.csv", index=False)

    val_probs = predict_outputs(model, X[va])
    threshold, val_f1 = tune_threshold(yf[va], val_probs["fall_prob"])
    test_probs = predict_outputs(model, X[te])
    meta_test = meta.loc[te].reset_index(drop=True)
    yft, ydt, dmt = yf[te], yd[te], dm[te]

    evals, ds_rows, class_rows = evaluate_all_splits(
        cfg,
        args.seed,
        run_dir,
        report_dir,
        figure_dir,
        cm_dir,
        meta_test,
        yft,
        ydt,
        dmt,
        test_probs["fall_prob"],
        test_probs["direction_prob"],
        threshold,
    )
    row = make_summary_row(cfg, args.seed, model, evals, threshold, val_f1, train_seconds, prepared["split_counts"], history_df)
    pd.DataFrame([row]).to_csv(run_dir / "summary_row.csv", index=False)
    save_json(run_dir / "summary_row.json", row)
    save_json(run_dir / "config.json", {**asdict(cfg), "seed": args.seed})
    save_json(run_dir / "feature_scaler.json", prepared["scaler"])
    save_pickle(run_dir / "feature_scaler.pkl", prepared["scaler"])
    save_json(run_dir / "class_weights.json", {"fall": fall_weights.tolist(), "direction": dir_weights.tolist()})
    model.save_weights(run_dir / "model.weights.h5")
    pd.DataFrame(ds_rows).to_csv(run_dir / "per_dataset_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(run_dir / "per_class_direction_f1.csv", index=False)
    print(
        f"{cfg.run_id}: E7 fall={row['weda_fall_f1']:.4f}, "
        f"E3 dir={row['direction_macro_f1']:.4f}, E7 dir={row['weda_direction_macro_f1']:.4f}, "
        f"params={row['params']}"
    )
    return row, ds_rows, class_rows


def compile_for_config(model, cfg: NextConfig, fall_weights: np.ndarray, dir_weights: np.ndarray) -> None:
    if cfg.train_mode == "fall_head_finetune":
        compile_model(model, cfg.alpha_fall, 0.0, fall_weights, dir_weights, direction_loss_type="weighted_ce")
    elif cfg.train_mode == "kd_finetune":
        compile_model(
            model,
            cfg.alpha_fall,
            cfg.lambda_dir,
            fall_weights,
            dir_weights,
            direction_loss_type=cfg.direction_loss,
            beta_kd=cfg.beta_kd,
            has_kd=True,
        )
    else:
        compile_model(model, cfg.alpha_fall, cfg.lambda_dir, fall_weights, dir_weights, direction_loss_type=cfg.direction_loss)


def compile_model(
    model,
    alpha_fall: float,
    lambda_dir: float,
    fall_weights: np.ndarray,
    dir_weights: np.ndarray,
    direction_loss_type: str = "weighted_ce",
    beta_kd: float = 0.0,
    has_kd: bool = False,
) -> None:
    import tensorflow as tf

    losses = {
        "fall_output": make_sparse_ce_loss(class_weights=fall_weights, name="fall_weighted_ce"),
        "direction_output": make_direction_loss(direction_loss_type, class_weights=dir_weights if direction_loss_type == "weighted_ce" else None),
    }
    loss_weights = {"fall_output": float(alpha_fall), "direction_output": float(lambda_dir)}
    metrics = {"fall_output": ["accuracy"], "direction_output": ["accuracy"]}
    if has_kd:
        losses["direction_kd_output"] = tf.keras.losses.KLDivergence(name="direction_kd")
        loss_weights["direction_kd_output"] = float(beta_kd)
        metrics["direction_kd_output"] = []
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )


def make_dataset(
    X: np.ndarray,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    teacher_direction: np.ndarray | None = None,
):
    import tensorflow as tf

    y_dict = {
        "fall_output": y_fall.astype(np.int64),
        "direction_output": prepare_direction_targets(y_direction),
    }
    sw_dict = make_sample_weights(y_fall, direction_mask)
    if teacher_direction is not None:
        y_dict["direction_kd_output"] = teacher_direction.astype(np.float32)
        supervised = (direction_mask > 0).astype(np.float32) * (y_fall.astype(np.int64) == 1)
        sw_dict["direction_kd_output"] = supervised.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y_dict, sw_dict))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def balanced_training_indices(meta_train: pd.DataFrame, y_fall_train: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    for dataset in ["bits", "weda"]:
        ds_idx = np.flatnonzero(meta_train["dataset"].astype(str).str.lower().to_numpy() == dataset)
        if len(ds_idx) == 0:
            continue
        n_non = int(np.sum(y_fall_train[ds_idx] == 0))
        n_fall = int(np.sum(y_fall_train[ds_idx] == 1))
        target = max(n_non, n_fall, 1)
        ds_parts = []
        for label in [0, 1]:
            pool = ds_idx[y_fall_train[ds_idx] == label]
            if len(pool):
                ds_parts.append(rng.choice(pool, size=target, replace=len(pool) < target))
        parts.append(np.concatenate(ds_parts))
    max_len = max(len(p) for p in parts)
    out = np.concatenate([rng.choice(p, size=max_len, replace=len(p) < max_len) for p in parts])
    rng.shuffle(out)
    return out.astype(np.int64)


def training_callbacks(run_dir: Path, X_val, y_fall_val, y_direction_val, direction_mask_val, patience: int):
    import tensorflow as tf

    class ValidationMetrics(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            probs = predict_outputs(self.model, X_val)
            pred_fall = (probs["fall_prob"] >= 0.5).astype(np.int64)
            logs["val_fall_f1"] = float(f1_score(y_fall_val, pred_fall, zero_division=0))
            supervised = (direction_mask_val > 0) & (y_direction_val >= 0)
            if supervised.any():
                pred_dir = np.argmax(probs["direction_prob"][supervised], axis=1)
                logs["val_direction_macro_f1"] = float(
                    f1_score(y_direction_val[supervised], pred_dir, labels=[0, 1, 2], average="macro", zero_division=0)
                )
            else:
                logs["val_direction_macro_f1"] = 0.0
            logs["val_domain_score"] = logs["val_fall_f1"] + logs["val_direction_macro_f1"]
            print(
                f" - val_fall_f1: {logs['val_fall_f1']:.4f}"
                f" - val_direction_macro_f1: {logs['val_direction_macro_f1']:.4f}"
                f" - val_domain_score: {logs['val_domain_score']:.4f}"
            )

    return [
        ValidationMetrics(),
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


def history_frame(history) -> pd.DataFrame:
    df = pd.DataFrame(history.history)
    if "epoch" not in df:
        df.insert(0, "epoch", np.arange(1, len(df) + 1))
    return df


def build_standard_model(adapter_mode: str = "none", kd_output: bool = False, width_multiplier: float = 1.0):
    import tensorflow as tf
    from tensorflow.keras import Model, layers

    names = FEATURE_PRESETS["tilt12"]
    by_name = {name: idx for idx, name in enumerate(names)}
    acc_idx = [by_name[name] for name in ["ax", "ay", "az", "acc_mag", "jerk", "roll", "pitch", "tilt_delta"]]
    gyro_idx = [by_name[name] for name in ["gx", "gy", "gz", "gyro_mag"]]

    def scale(v: int) -> int:
        return max(4, int(round(v * width_multiplier)))

    inp = layers.Input(shape=(WINDOW_SIZE, 12), name="imu_input")
    acc = ChannelSlice(acc_idx, name="acc_input")(inp)
    gyro = ChannelSlice(gyro_idx, name="gyro_input")(inp)
    acc_f = sensor_encoder(acc, "acc", scale)
    gyro_f = sensor_encoder(gyro, "gyro", scale)
    x = layers.Concatenate(name="sensor_concat")([acc_f, gyro_f])
    x = layers.Conv1D(scale(64), 1, padding="same", name="fusion_pointwise_conv")(x)
    x = layers.BatchNormalization(name="fusion_bn")(x)
    x = layers.Activation("relu", name="fusion_relu")(x)
    x = dstcn_block(x, scale(64), 1, "dstcn_d1")
    x = dstcn_block(x, scale(64), 2, "dstcn_d2")
    x = dstcn_block(x, scale(96), 4, "dstcn_d4")

    fall_x = x
    dir_x = x
    if adapter_mode in {"fall16", "both16"}:
        fall_x = layers.Conv1D(16, 1, padding="same", activation="relu", name="fall_adapter_conv1x1")(fall_x)
    if adapter_mode == "both16":
        dir_x = layers.Conv1D(16, 1, padding="same", activation="relu", name="direction_adapter_conv1x1")(dir_x)

    fall_context = attention_pool(fall_x, "fall_attention_pooling")
    dir_context = attention_pool(dir_x, "direction_attention_pooling")
    fall = layers.Dense(32, activation="relu", name="fall_dense")(fall_context)
    fall = layers.Dropout(0.2, name="fall_dropout")(fall)
    fall_out = layers.Dense(2, activation="softmax", name="fall_output")(fall)
    direction = layers.Dense(32, activation="relu", name="direction_dense")(dir_context)
    direction = layers.Dropout(0.2, name="direction_dropout")(direction)
    dir_out = layers.Dense(3, activation="softmax", name="direction_output")(direction)
    outputs = {"fall_output": fall_out, "direction_output": dir_out}
    if kd_output:
        outputs["direction_kd_output"] = layers.Activation("linear", name="direction_kd_output")(dir_out)
    return Model(inp, outputs, name="DS-Fall-RD-Next12")


def build_head_specific_model(width_multiplier: float = 0.5):
    from tensorflow.keras import Model, layers

    names = FEATURE_PRESETS["tilt12"]
    by_name = {name: idx for idx, name in enumerate(names)}

    def scale(v: int) -> int:
        return max(4, int(round(v * width_multiplier)))

    inp = layers.Input(shape=(WINDOW_SIZE, 12), name="imu_input")

    fall_acc = ChannelSlice([by_name[n] for n in ["ax", "ay", "az", "acc_mag", "jerk"]], name="fall_acc_input")(inp)
    fall_gyro = ChannelSlice([by_name[n] for n in ["gx", "gy", "gz", "gyro_mag"]], name="fall_gyro_input")(inp)
    dir_acc = ChannelSlice([by_name[n] for n in ["ax", "ay", "az", "acc_mag", "jerk", "roll", "pitch", "tilt_delta"]], name="dir_acc_input")(inp)
    dir_gyro = ChannelSlice([by_name[n] for n in ["gx", "gy", "gz", "gyro_mag"]], name="dir_gyro_input")(inp)

    fall_x = routed_encoder(fall_acc, fall_gyro, "fall_route", scale)
    dir_x = routed_encoder(dir_acc, dir_gyro, "dir_route", scale)

    fall_context = attention_pool(fall_x, "fall_attention_pooling")
    dir_context = attention_pool(dir_x, "direction_attention_pooling")
    fall = layers.Dense(32, activation="relu", name="fall_dense")(fall_context)
    fall = layers.Dropout(0.2, name="fall_dropout")(fall)
    fall_out = layers.Dense(2, activation="softmax", name="fall_output")(fall)
    direction = layers.Dense(32, activation="relu", name="direction_dense")(dir_context)
    direction = layers.Dropout(0.2, name="direction_dropout")(direction)
    dir_out = layers.Dense(3, activation="softmax", name="direction_output")(direction)
    return Model(inp, {"fall_output": fall_out, "direction_output": dir_out}, name="DS-Fall-RD-F1-Routed")


def routed_encoder(acc, gyro, prefix: str, scale):
    from tensorflow.keras import layers

    acc_f = sensor_encoder(acc, f"{prefix}_acc", scale)
    gyro_f = sensor_encoder(gyro, f"{prefix}_gyro", scale)
    x = layers.Concatenate(name=f"{prefix}_concat")([acc_f, gyro_f])
    x = layers.Conv1D(scale(64), 1, padding="same", name=f"{prefix}_fusion_conv")(x)
    x = layers.BatchNormalization(name=f"{prefix}_fusion_bn")(x)
    x = layers.Activation("relu", name=f"{prefix}_fusion_relu")(x)
    x = dstcn_block(x, scale(64), 1, f"{prefix}_dstcn_d1")
    x = dstcn_block(x, scale(64), 2, f"{prefix}_dstcn_d2")
    return dstcn_block(x, scale(96), 4, f"{prefix}_dstcn_d4")


class ChannelSlice:
    def __new__(cls, indices: list[int], name: str):
        import tensorflow as tf
        from tensorflow.keras import layers

        @tf.keras.utils.register_keras_serializable(package="DSFall")
        class _ChannelSlice(layers.Layer):
            def __init__(self, channel_indices, **kwargs):
                super().__init__(**kwargs)
                self.channel_indices = list(channel_indices)

            def call(self, inputs):
                return tf.gather(inputs, self.channel_indices, axis=-1)

            def get_config(self):
                config = super().get_config()
                config.update({"channel_indices": self.channel_indices})
                return config

        return _ChannelSlice(indices, name=name)


class TemporalWeightedSum:
    def __new__(cls, name: str):
        import tensorflow as tf
        from tensorflow.keras import layers

        @tf.keras.utils.register_keras_serializable(package="DSFall")
        class _TemporalWeightedSum(layers.Layer):
            def call(self, inputs):
                values, weights = inputs
                return tf.reduce_sum(values * weights, axis=1)

        return _TemporalWeightedSum(name=name)


def sensor_encoder(x, prefix: str, scale):
    from tensorflow.keras import layers

    x = layers.Conv1D(scale(16), 5, padding="same", name=f"{prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{prefix}_conv1_bn")(x)
    x = layers.Activation("relu", name=f"{prefix}_conv1_relu")(x)
    x = dsconv_block(x, scale(24), 5, f"{prefix}_dsconv1")
    x = dsconv_block(x, scale(32), 3, f"{prefix}_dsconv2")
    return x


def dsconv_block(x, filters: int, kernel_size: int, name: str):
    from tensorflow.keras import layers

    x = layers.SeparableConv1D(filters, kernel_size=kernel_size, padding="same", name=f"{name}_sepconv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def dstcn_block(x, channels: int, dilation: int, name: str):
    from tensorflow.keras import layers

    residual = x
    y = layers.SeparableConv1D(channels, 3, dilation_rate=dilation, padding="same", name=f"{name}_sepconv1")(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.Activation("relu", name=f"{name}_relu1")(y)
    y = layers.Dropout(0.1, name=f"{name}_dropout")(y)
    y = layers.SeparableConv1D(channels, 3, dilation_rate=dilation, padding="same", name=f"{name}_sepconv2")(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)
    if residual.shape[-1] != channels:
        residual = layers.Conv1D(channels, 1, padding="same", name=f"{name}_projection")(residual)
    y = layers.Add(name=f"{name}_add")([residual, y])
    return layers.Activation("relu", name=f"{name}_out_relu")(y)


def attention_pool(x, name: str):
    from tensorflow.keras import layers

    channels = int(x.shape[-1])
    h = layers.Dense(max(channels // 2, 1), activation="tanh", name=f"{name}_hidden")(x)
    score = layers.Dense(1, name=f"{name}_score")(h)
    alpha = layers.Softmax(axis=1, name=f"{name}_alpha")(score)
    return TemporalWeightedSum(name=f"{name}_pool")([x, alpha])


def copy_matching_weights(source, target) -> None:
    source_layers = {layer.name: layer for layer in source.layers}
    for layer in target.layers:
        src = source_layers.get(layer.name)
        if src is None:
            continue
        src_weights = src.get_weights()
        tgt_weights = layer.get_weights()
        if len(src_weights) == len(tgt_weights) and all(sw.shape == tw.shape for sw, tw in zip(src_weights, tgt_weights)):
            layer.set_weights(src_weights)


def freeze_for_fall_finetune(model) -> None:
    for layer in model.layers:
        layer.trainable = layer.name.startswith("fall_")


def predict_outputs(model, X: np.ndarray) -> dict[str, np.ndarray]:
    preds = model.predict(X, verbose=0)
    if isinstance(preds, dict):
        fall = preds["fall_output"]
        direction = preds["direction_output"]
    else:
        fall, direction = preds[:2]
    return {"fall_prob": fall[:, 1].astype(float), "direction_prob": direction.astype(float)}


def predict_direction(model, X: np.ndarray) -> np.ndarray:
    return predict_outputs(model, X)["direction_prob"].astype(np.float32)


def evaluate_all_splits(
    cfg: NextConfig,
    seed: int,
    run_dir: Path,
    report_dir: Path,
    figure_dir: Path,
    cm_dir: Path,
    meta_test: pd.DataFrame,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    fall_prob: np.ndarray,
    direction_prob: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    evals = {}
    per_dataset_rows = []
    per_class_rows = []
    for eval_id, dataset in [("E3", "bits+weda"), ("E6", "bits"), ("E7", "weda")]:
        mask = np.ones(len(meta_test), dtype=bool) if dataset == "bits+weda" else meta_test["dataset"].astype(str).str.lower().to_numpy() == dataset
        metrics, pred_df = evaluate_subset(
            meta_test.loc[mask].reset_index(drop=True),
            y_fall[mask],
            y_direction[mask],
            direction_mask[mask],
            fall_prob[mask],
            direction_prob[mask],
            threshold,
        )
        evals[eval_id] = metrics
        pred_df.to_csv(run_dir / f"{eval_id}_predictions.csv", index=False)
        save_confusions(cfg.run_id, eval_id, metrics, cm_dir, figure_dir)
        if eval_id in {"E6", "E7"}:
            per_dataset_rows.append(
                {
                    "run_id": cfg.run_id,
                    "seed": seed,
                    "eval": eval_id,
                    "dataset": dataset,
                    "fall_precision": metrics["fall_precision"],
                    "fall_recall": metrics["fall_recall"],
                    "fall_f1": metrics["fall_f1"],
                    "direction_macro_f1": metrics["direction_macro_f1"],
                    "direction_accuracy": metrics["direction_accuracy"],
                    "direction_n_supervised": metrics["direction_n_supervised"],
                }
            )
        per_class = metrics["direction_per_class_f1"]
        per_class_rows.append(
            {
                "run_id": cfg.run_id,
                "seed": seed,
                "eval": eval_id,
                "dataset": dataset,
                "forward_f1": per_class.get("forward", math.nan),
                "backward_f1": per_class.get("backward", math.nan),
                "lateral_f1": per_class.get("lateral", math.nan),
            }
        )
    return evals, per_dataset_rows, per_class_rows


def evaluate_subset(meta: pd.DataFrame, y_fall, y_direction, direction_mask, fall_prob, direction_prob, threshold):
    pred_fall = (fall_prob >= threshold).astype(np.int64)
    pred_direction = np.argmax(direction_prob, axis=1).astype(np.int64)
    supervised = (direction_mask > 0) & (y_direction >= 0)
    fall_cm = confusion_matrix(y_fall, pred_fall, labels=[0, 1])
    if supervised.any():
        dir_cm = confusion_matrix(y_direction[supervised], pred_direction[supervised], labels=[0, 1, 2])
        report = classification_report(
            y_direction[supervised],
            pred_direction[supervised],
            labels=[0, 1, 2],
            target_names=DIRECTION_LABELS,
            output_dict=True,
            zero_division=0,
        )
        direction_macro = float(f1_score(y_direction[supervised], pred_direction[supervised], labels=[0, 1, 2], average="macro", zero_division=0))
        direction_acc = float(accuracy_score(y_direction[supervised], pred_direction[supervised]))
        per_class = {label: float(report[label]["f1-score"]) for label in DIRECTION_LABELS}
    else:
        dir_cm = np.zeros((3, 3), dtype=int)
        direction_macro, direction_acc = math.nan, math.nan
        per_class = {label: math.nan for label in DIRECTION_LABELS}
    out = meta.copy()
    out["true_fall"] = [FALL_LABELS[int(v)] for v in y_fall]
    out["pred_fall"] = [FALL_LABELS[int(v)] for v in pred_fall]
    out["fall_prob"] = fall_prob
    out["true_direction"] = [DIRECTION_LABELS[int(v)] if int(v) >= 0 else "none" for v in y_direction]
    out["pred_direction"] = [DIRECTION_LABELS[int(v)] for v in pred_direction]
    metrics = {
        "fall_accuracy": float(accuracy_score(y_fall, pred_fall)),
        "fall_precision": float(precision_score(y_fall, pred_fall, zero_division=0)),
        "fall_recall": float(recall_score(y_fall, pred_fall, zero_division=0)),
        "fall_f1": float(f1_score(y_fall, pred_fall, zero_division=0)),
        "fall_cm": fall_cm.tolist(),
        "direction_accuracy": direction_acc,
        "direction_macro_f1": direction_macro,
        "direction_n_supervised": int(supervised.sum()),
        "direction_cm": dir_cm.tolist(),
        "direction_per_class_f1": per_class,
    }
    return metrics, out


def save_confusions(run_id: str, eval_id: str, metrics: dict[str, Any], cm_dir: Path, figure_dir: Path) -> None:
    fall_cm = np.asarray(metrics["fall_cm"], dtype=int)
    dir_cm = np.asarray(metrics["direction_cm"], dtype=int)
    pd.DataFrame(fall_cm, index=["true_nonfall", "true_fall"], columns=["pred_nonfall", "pred_fall"]).to_csv(cm_dir / f"{run_id}_{eval_id}_fall.csv")
    pd.DataFrame(dir_cm, index=[f"true_{x}" for x in DIRECTION_LABELS], columns=[f"pred_{x}" for x in DIRECTION_LABELS]).to_csv(cm_dir / f"{run_id}_{eval_id}_direction.csv")
    plot_cm(fall_cm, FALL_LABELS, figure_dir / f"{run_id}_{eval_id}_fall.png", f"{run_id} {eval_id} Fall")
    plot_cm(dir_cm, DIRECTION_LABELS, figure_dir / f"{run_id}_{eval_id}_direction.png", f"{run_id} {eval_id} Direction")


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


def make_summary_row(cfg: NextConfig, seed: int, model, evals: dict[str, Any], threshold: float, val_f1: float, train_seconds: float, split_counts: dict[str, int], history_df: pd.DataFrame) -> dict[str, Any]:
    e3, e6, e7 = evals["E3"], evals["E6"], evals["E7"]
    row = {
        **asdict(cfg),
        "seed": seed,
        "train_dataset": "bits+weda",
        "test_dataset": "bits+weda",
        "sampling_rate": SAMPLING_RATE,
        "window_seconds": WINDOW_SECONDS,
        "input_shape": "(50, 12)",
        "params": int(model.count_params()),
        "fall_threshold": float(threshold),
        "val_best_fall_f1": float(val_f1),
        "fall_precision": e3["fall_precision"],
        "fall_recall": e3["fall_recall"],
        "fall_f1": e3["fall_f1"],
        "direction_macro_f1": e3["direction_macro_f1"],
        "direction_accuracy": e3["direction_accuracy"],
        "direction_n_supervised": e3["direction_n_supervised"],
        "bits_fall_f1": e6["fall_f1"],
        "bits_direction_macro_f1": e6["direction_macro_f1"],
        "bits_direction_accuracy": e6["direction_accuracy"],
        "bits_direction_n_supervised": e6["direction_n_supervised"],
        "weda_fall_f1": e7["fall_f1"],
        "weda_fall_precision": e7["fall_precision"],
        "weda_fall_recall": e7["fall_recall"],
        "weda_direction_macro_f1": e7["direction_macro_f1"],
        "weda_direction_accuracy": e7["direction_accuracy"],
        "weda_direction_n_supervised": e7["direction_n_supervised"],
        "train_seconds": train_seconds,
        "best_epoch": int(history_df.loc[history_df["val_domain_score"].astype(float).idxmax(), "epoch"]) if "val_domain_score" in history_df else math.nan,
        **split_counts,
    }
    row["selection_score"] = selection_score(row)
    row["passes_hard_filter"] = bool(
        row["direction_macro_f1"] >= 0.86
        and row["weda_direction_macro_f1"] >= 0.80
        and row["bits_fall_f1"] >= 0.90
        and row["params"] <= 90000
    )
    return row


def selection_score(row: dict[str, Any]) -> float:
    param_penalty = max(0.0, (float(row.get("params", REF_METRICS["params"])) - REF_METRICS["params"]) / REF_METRICS["params"])
    return float(
        0.30 * row.get("weda_fall_f1", 0.0)
        + 0.25 * row.get("direction_macro_f1", 0.0)
        + 0.20 * row.get("weda_direction_macro_f1", 0.0)
        + 0.15 * row.get("bits_fall_f1", 0.0)
        + 0.10 * row.get("fall_f1", 0.0)
        - 0.05 * param_penalty
    )


def flush_outputs(report_dir: Path, all_rows: list[dict[str, Any]], per_dataset_rows: list[dict[str, Any]], per_class_rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(all_rows).to_csv(report_dir / "all_runs.csv", index=False)
    pd.DataFrame(per_dataset_rows).to_csv(report_dir / "per_dataset_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(report_dir / "per_class_direction_f1.csv", index=False)


def build_reports(report_dir: Path, all_rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(all_rows)
    ranked = df.sort_values(["passes_hard_filter", "weda_fall_f1", "selection_score"], ascending=False)
    hard = df[df["passes_hard_filter"].astype(bool)]
    best_hard = hard.sort_values(["weda_fall_f1", "selection_score"], ascending=False).iloc[0] if not hard.empty else None
    best_weda = df.sort_values(["weda_fall_f1", "selection_score"], ascending=False).iloc[0]
    trade_pool = df[(df["direction_macro_f1"] >= 0.86) & (df["weda_direction_macro_f1"] >= 0.80) & (df["bits_fall_f1"] >= 0.90) & (df["params"] <= 90000)]
    best_trade = trade_pool.sort_values(["weda_fall_f1", "selection_score"], ascending=False).iloc[0] if not trade_pool.empty else df.sort_values("selection_score", ascending=False).iloc[0]

    display_cols = [
        "run_id",
        "block",
        "alpha_fall",
        "lambda_dir",
        "sampler",
        "adapter_mode",
        "train_mode",
        "beta_kd",
        "fall_threshold",
        "fall_precision",
        "fall_recall",
        "fall_f1",
        "direction_macro_f1",
        "direction_accuracy",
        "direction_n_supervised",
        "bits_fall_f1",
        "bits_direction_macro_f1",
        "bits_direction_n_supervised",
        "weda_fall_f1",
        "weda_direction_macro_f1",
        "weda_direction_n_supervised",
        "params",
        "selection_score",
        "passes_hard_filter",
    ]
    summary = [
        "# Next 12 Experiments Summary",
        "",
        "Benchmark: BITS/WEDA only, event-centered 2s, 25 Hz, global validation-tuned fall threshold.",
        "",
        "## All Runs",
        "",
        markdown_table(format_df(ranked[display_cols])),
        "",
        "## Key Findings",
        "",
        f"- Runs passing hard filters: {', '.join(hard['run_id'].astype(str)) if not hard.empty else 'none'}.",
        f"- Best E7 WEDA Fall F1: `{best_weda['run_id']}` = {best_weda['weda_fall_f1']:.4f}.",
        f"- Best balanced trade-off: `{best_trade['run_id']}` with score {best_trade['selection_score']:.4f}.",
        "- Reference comparison is kept outside `all_runs.csv` so the table contains exactly the requested 12 experiments.",
        "- D1-D4 used an internally retrained serializable reference teacher because the old Lambda-layer artifact is not loadable in the current Keras runtime.",
    ]
    (report_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    selection_rows = []
    if best_hard is not None:
        selection_rows.append(selection_record("Best hard-filter pass", best_hard))
    selection_rows.append(selection_record("Best WEDA Fall F1", best_weda))
    selection_rows.append(selection_record("Best balanced trade-off", best_trade))
    selection_df = pd.DataFrame(selection_rows)
    selection = [
        "# Best Model Selection",
        "",
        "Hard filters: E3 Direction Macro F1 >= 0.86, E7 WEDA Direction Macro F1 >= 0.80, E6 BITS Fall F1 >= 0.90, Params <= 90k.",
        "",
        markdown_table(format_df(selection_df)),
        "",
        "## Recommendation",
        "",
    ]
    if best_hard is None:
        selection.extend(
            [
                "No next-12 run passed all hard filters. Keep `REF_CURRENT_E3_ARTIFACT` as the paper main model and report the best new run as an ablation/future direction.",
                f"The best WEDA Fall F1 run is `{best_weda['run_id']}` ({best_weda['weda_fall_f1']:.4f}), but it does not satisfy the full direction/fall stability criteria.",
            ]
        )
    else:
        improves_ref = best_hard["weda_fall_f1"] > REF_METRICS["weda_fall_f1"]
        if improves_ref:
            selection.append(f"`{best_hard['run_id']}` passes hard filters and improves E7 WEDA Fall F1 over reference.")
        else:
            selection.append("A run passed hard filters but did not improve E7 WEDA Fall F1 over reference; keep reference as main.")
    selection.append("")
    selection.append("## Fall/Direction Trade-off")
    selection.append("")
    selection.append(
        "Runs that push fall loss or freeze/fine-tune fall-specific parts can raise WEDA Fall F1, but the common failure mode is lower E3 Direction Macro F1 or lower E6 BITS Fall F1. The final choice should not optimize WEDA Fall F1 alone."
    )
    (report_dir / "best_model_selection.md").write_text("\n".join(selection) + "\n", encoding="utf-8")


def selection_record(role: str, row: pd.Series) -> dict[str, Any]:
    return {
        "role": role,
        "run_id": row["run_id"],
        "E7_WEDA_Fall_F1": row["weda_fall_f1"],
        "delta_E7_WEDA_Fall_vs_REF": row["weda_fall_f1"] - REF_METRICS["weda_fall_f1"],
        "E3_Direction_Macro_F1": row["direction_macro_f1"],
        "E3_Direction_Accuracy": row["direction_accuracy"],
        "E3_Direction_N": row["direction_n_supervised"],
        "E7_WEDA_Direction_Macro_F1": row["weda_direction_macro_f1"],
        "E6_BITS_Fall_F1": row["bits_fall_f1"],
        "params": row["params"],
        "selection_score": row["selection_score"],
        "passes_hard_filter": row["passes_hard_filter"],
    }


def check_old_artifact_loadability(project_root: Path) -> str:
    path = project_root / "artifacts" / "experiments_25hz_event" / "E3_BITS_WEDA_MIXED" / "best_model.keras"
    lines = ["# Artifact Load Note", ""]
    if not path.exists():
        lines.append(f"Old artifact not found: `{path}`.")
        return "\n".join(lines) + "\n"
    try:
        import tensorflow as tf

        tf.keras.models.load_model(path, compile=False, safe_mode=False)
        lines.append("Old E3 artifact loaded successfully.")
    except Exception as exc:
        lines.append("Old E3 artifact could not be loaded in the current Keras runtime.")
        lines.append("")
        lines.append(f"Error type: `{type(exc).__name__}`")
        lines.append("")
        lines.append("D1-D4 therefore rebuild/retrain a serializable reference-compatible teacher using `ChannelSlice` layers instead of Lambda slices.")
    return "\n".join(lines) + "\n"


def write_plan(report_dir: Path, args: argparse.Namespace) -> None:
    rows = [asdict(cfg) for cfg in next12_configs()]
    pd.DataFrame(rows).to_csv(report_dir / "experiment_configs.csv", index=False)
    lines = [
        "# Next 12 Experiment Plan",
        "",
        "- Scope: exactly B1, B2, B3, C1, C2, C3, D1, D2, D3, D4, E1, F1.",
        "- Benchmark: BITS/WEDA event-centered 2s, 25 Hz, tilt12 unless F1 routes fall features internally.",
        "- No new datasets, no Transformer, no dataset-aware threshold as main result.",
        f"- Seed: {args.seed}",
        f"- Standard epochs/patience: {args.epochs}/{args.patience}",
        f"- Fine-tune epochs/patience: {args.finetune_epochs}/{args.finetune_patience}",
    ]
    (report_dir / "experiment_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    compact = df.astype(object).where(pd.notna(df), "")
    header = "| " + " | ".join(map(str, compact.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(compact.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in compact.to_numpy()]
    return "\n".join([header, sep] + rows)


def format_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in {"params", "seed"} or col.endswith("_n_supervised") or col == "E3_Direction_N":
            values = pd.to_numeric(out[col], errors="coerce")
            out[col] = values.map(lambda v: "" if pd.isna(v) else str(int(v)))
        elif pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    return out


if __name__ == "__main__":
    main()
