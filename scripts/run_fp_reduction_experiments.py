from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_25hz_a5wcefw_bits_weda as base25
from scripts.analyze_25hz_center_fall_gap import extract_summary_features
from scripts.run_next12_experiments import build_standard_model, copy_matching_weights
from src.config import make_config
from src.models.losses import compute_class_weights, make_direction_loss, make_sparse_ce_loss, prepare_direction_targets
from src.utils.io import ensure_dir, save_json, save_pickle
from src.utils.seed import set_seed


SAMPLING_RATE = 25.0
WINDOW_SECONDS = 2.0
WINDOW_SIZE = 50
DIRECTION_LABELS = ["forward", "backward", "lateral"]
FALL_LABELS = ["non_fall", "fall"]
REF_METRICS = {
    "e3_direction_macro_f1": 0.8966626725247414,
    "e7_direction_macro_f1": 0.8390350877192981,
    "e6_bits_fall_f1": 0.945054945054945,
    "e7_weda_fall_f1": 0.6857142857142857,
    "e7_weda_precision": 0.5333333333333333,
    "e7_weda_recall": 0.96,
    "e7_weda_fp": 21,
    "params": 65959,
}


@dataclass(frozen=True)
class HNConfig:
    run_id: str
    method: str
    hard_negative_source: str
    hard_negative_weight: float = 2.0
    alpha_fall: float = 1.0
    lambda_dir: float = 1.5
    beta_kd: float = 0.0
    train_mode: str = "standard"
    epochs: int = 25
    patience: int = 7
    learning_rate: float = 1e-3


HN_CONFIGS = [
    HNConfig("HN1", "hard_negative_weighted_ce", "feature_rule", epochs=25, patience=7, learning_rate=1e-3),
    HNConfig("HN2", "hard_negative_weighted_ce", "prediction_rule", epochs=25, patience=7, learning_rate=1e-3),
    HNConfig("HN3", "two_stage_fall_branch_finetune", "prediction_rule", train_mode="fall_branch_finetune", epochs=15, patience=5, learning_rate=2e-4),
    HNConfig("HN4", "two_stage_fall_branch_direction_kd", "prediction_rule", train_mode="fall_branch_kd", beta_kd=1.0, epochs=15, patience=5, learning_rate=1e-4),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run threshold and hard-negative FP-reduction experiments.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "experiments_25hz_event")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    config = make_config(args.project_root)
    report_dir = ensure_dir(args.output_dir / "reports" / "fp_reduction_experiments")
    figure_dir = ensure_dir(args.output_dir / "figures" / "fp_reduction_experiments")
    cm_dir = ensure_dir(figure_dir / "confusion_matrices")
    run_dir = ensure_dir(report_dir / "runs")

    print("=== FP reduction experiments: threshold + hard negatives ===")
    print(f"project_root={args.project_root}")
    print(f"artifact_dir={args.artifact_dir}")
    print("Scope: BITS/WEDA only, 25 Hz, 2-second event-centered windows, tilt12.")

    data = base25.build_25hz_dataset(config, window_mode="event_centered")
    base25.validate_25hz_dataset(data)
    meta = data["metadata"].copy().reset_index(drop=True)
    X_features = np.asarray(data["X_features"], dtype=np.float32)
    X_raw6 = np.asarray(data["X_raw6"], dtype=np.float32)
    y_fall = np.asarray(data["y_fall"], dtype=np.int64)
    y_direction = np.asarray(data["y_direction"], dtype=np.int64)
    direction_mask = np.asarray(data["direction_mask"], dtype=np.float32)

    scaler = load_reference_scaler(args.artifact_dir)
    X_scaled = base25.transform_scaler(X_features, scaler)
    split_masks = make_split_masks(meta)
    save_json(
        report_dir / "split_counts.json",
        {name: int(mask.sum()) for name, mask in split_masks.items()},
    )

    reference = build_reference_model(args.artifact_dir)
    ref_pred = predict_model(reference, X_scaled, batch_size=args.batch_size)
    verify_reference_predictions(meta, split_masks["test"], ref_pred, args.artifact_dir, report_dir)

    summary_features = extract_summary_features(X_raw6, X_features)
    hn_info = build_hard_negative_flags(meta, y_fall, summary_features, ref_pred["fall_prob"])
    hn_info.to_csv(report_dir / "weda_fp_analysis.csv", index=False)

    threshold_rows = run_threshold_block(meta, y_fall, y_direction, direction_mask, ref_pred, split_masks, cm_dir, report_dir)
    pd.DataFrame(threshold_rows).to_csv(report_dir / "threshold_decision_results.csv", index=False)

    hard_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for hn_cfg in HN_CONFIGS:
        rows, cls_rows = run_hn_experiment(
            cfg=hn_cfg,
            seed=args.seed,
            run_root=run_dir,
            report_dir=report_dir,
            cm_dir=cm_dir,
            meta=meta,
            X_scaled=X_scaled,
            y_fall=y_fall,
            y_direction=y_direction,
            direction_mask=direction_mask,
            split_masks=split_masks,
            reference=reference,
            ref_pred=ref_pred,
            hn_info=hn_info,
            batch_size=args.batch_size,
            force=args.force,
            artifact_dir=args.artifact_dir,
        )
        hard_rows.extend(rows)
        per_class_rows.extend(cls_rows)
        pd.DataFrame(hard_rows).to_csv(report_dir / "hard_negative_runs.csv", index=False)
        pd.DataFrame(per_class_rows).to_csv(report_dir / "per_class_direction_f1.csv", index=False)

    threshold_df = pd.DataFrame(threshold_rows)
    hard_df = pd.DataFrame(hard_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    write_reports(report_dir, threshold_df, hard_df, per_class_df)
    print(f"Saved reports to {report_dir}")


def load_reference_scaler(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / "E3_BITS_WEDA_MIXED" / "scaler.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing reference scaler: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def build_reference_model(artifact_dir: Path):
    weights_path = artifact_dir / "E3_BITS_WEDA_MIXED" / "best_model.keras"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing reference weights artifact: {weights_path}")
    model = build_standard_model(adapter_mode="none", kd_output=False, width_multiplier=1.0)
    model.load_weights(str(weights_path))
    return model


def make_split_masks(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    split = meta["split"].astype(str).str.lower()
    dataset = meta["dataset"].astype(str).str.lower()
    return {
        "train": split.eq("train").to_numpy(),
        "val": split.eq("val").to_numpy(),
        "test": split.eq("test").to_numpy(),
        "test_bits": (split.eq("test") & dataset.eq("bits")).to_numpy(),
        "test_weda": (split.eq("test") & dataset.eq("weda")).to_numpy(),
    }


def predict_model(model, X: np.ndarray, batch_size: int = 128) -> dict[str, np.ndarray]:
    preds = model.predict(X, batch_size=batch_size, verbose=0)
    if isinstance(preds, dict):
        fall = preds["fall_output"]
        direction = preds["direction_output"]
    else:
        fall, direction = preds[:2]
    return {
        "fall_prob": fall[:, 1].astype(np.float64),
        "fall_probs": fall.astype(np.float64),
        "direction_probs": direction.astype(np.float64),
        "direction_pred": np.argmax(direction, axis=1).astype(np.int64),
    }


def verify_reference_predictions(meta: pd.DataFrame, test_mask: np.ndarray, pred: dict[str, np.ndarray], artifact_dir: Path, report_dir: Path) -> None:
    saved_path = artifact_dir / "E3_BITS_WEDA_MIXED" / "predictions.csv"
    note = {
        "reference_model_load": "Keras model object is not deserializable because of Lambda layers, but weights load into the serializable equivalent architecture.",
        "weights_path": str(artifact_dir / "E3_BITS_WEDA_MIXED" / "best_model.keras"),
        "saved_predictions_path": str(saved_path),
        "max_abs_fall_prob_diff_on_test": None,
        "mean_abs_fall_prob_diff_on_test": None,
    }
    if saved_path.exists():
        saved = pd.read_csv(saved_path)
        check = (
            meta.loc[test_mask, ["window_id"]]
            .reset_index()
            .assign(rebuilt_fall_prob=pred["fall_prob"][test_mask])
            .merge(saved[["window_id", "fall_prob"]], on="window_id", how="inner")
        )
        if not check.empty:
            diff = np.abs(check["rebuilt_fall_prob"].to_numpy() - check["fall_prob"].to_numpy())
            note["max_abs_fall_prob_diff_on_test"] = float(np.max(diff))
            note["mean_abs_fall_prob_diff_on_test"] = float(np.mean(diff))
    save_json(report_dir / "reference_load_note.json", note)


def build_hard_negative_flags(meta: pd.DataFrame, y_fall: np.ndarray, features: pd.DataFrame, ref_fall_prob: np.ndarray) -> pd.DataFrame:
    out = meta[["window_id", "dataset", "split", "subject_id", "trial_id", "activity", "fall_label"]].copy()
    out["ref_fall_prob"] = ref_fall_prob
    dataset = out["dataset"].astype(str).str.lower()
    split = out["split"].astype(str).str.lower()
    weda_nonfall_trainval = dataset.eq("weda") & split.isin(["train", "val"]) & (y_fall == 0)

    feature_names = [
        "jerk_p95",
        "jerk_max",
        "acc_mag_std",
        "acc_mag_range",
        "tilt_delta_max",
        "tilt_delta_std",
        "ay_range",
        "ay_std",
        "gyro_mag_max",
        "gyro_mag_range",
    ]
    score = np.full(len(out), np.nan, dtype=np.float64)
    rank_parts = []
    idx = np.flatnonzero(weda_nonfall_trainval.to_numpy())
    for name in feature_names:
        values = features.loc[idx, name].to_numpy(dtype=np.float64)
        ranks = pd.Series(values).rank(pct=True).to_numpy(dtype=np.float64)
        rank_parts.append(ranks)
    if len(idx):
        score[idx] = np.mean(np.column_stack(rank_parts), axis=1)
    out["hard_negative_score"] = score
    feature_threshold = np.nanquantile(score[idx], 0.80) if len(idx) else np.nan
    out["hn_feature_rule"] = False
    if np.isfinite(feature_threshold):
        out.loc[idx, "hn_feature_rule"] = score[idx] >= feature_threshold

    prob_threshold = 0.5
    pred_flag = np.zeros(len(out), dtype=bool)
    if len(idx):
        pred_flag[idx] = ref_fall_prob[idx] >= prob_threshold
        if pred_flag[idx].sum() == 0:
            cutoff = np.quantile(ref_fall_prob[idx], 0.80)
            pred_flag[idx] = ref_fall_prob[idx] >= cutoff
            prob_threshold = float(cutoff)
    out["hn_prediction_rule"] = pred_flag
    out["prediction_rule_threshold"] = prob_threshold

    test_weda_nonfall = dataset.eq("weda") & split.eq("test") & (y_fall == 0)
    out["saved_reference_test_fp"] = False
    out.loc[test_weda_nonfall, "saved_reference_test_fp"] = ref_fall_prob[test_weda_nonfall.to_numpy()] >= 0.5
    out["feature_rule_threshold"] = feature_threshold
    return out


def run_threshold_block(
    meta: pd.DataFrame,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    pred: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    cm_dir: Path,
    report_dir: Path,
) -> list[dict[str, Any]]:
    rules = select_threshold_rules(y_fall[masks["val"]], pred["fall_prob"][masks["val"]])
    save_json(report_dir / "threshold_rule_config.json", rules)
    rows = []
    for rule_id, rule in rules.items():
        threshold = float(rule["threshold"])
        for eval_id, mask_name, dataset_name in [
            ("E3", "test", "bits+weda"),
            ("E6", "test_bits", "bits"),
            ("E7", "test_weda", "weda"),
        ]:
            metrics = evaluate_predictions(
                y_fall[mask_name_to_mask(masks, mask_name)],
                y_direction[mask_name_to_mask(masks, mask_name)],
                direction_mask[mask_name_to_mask(masks, mask_name)],
                pred["fall_prob"][mask_name_to_mask(masks, mask_name)],
                pred["direction_pred"][mask_name_to_mask(masks, mask_name)],
                threshold,
            )
            row = {
                "block": "threshold_decision",
                "run_id": rule_id,
                "rule_description": rule["description"],
                "eval": eval_id,
                "dataset": dataset_name,
                "threshold": threshold,
                **metrics,
                "weda_fp_reduction_vs_ref": REF_METRICS["e7_weda_fp"] - metrics["FP"] if eval_id == "E7" else np.nan,
                "weda_precision_gain_vs_ref": metrics["fall_precision"] - REF_METRICS["e7_weda_precision"] if eval_id == "E7" else np.nan,
                "weda_recall_delta_vs_ref": metrics["fall_recall"] - REF_METRICS["e7_weda_recall"] if eval_id == "E7" else np.nan,
            }
            rows.append(row)
            save_confusion_figure(cm_dir, f"{rule_id}_{eval_id}_fall", metrics["fall_confusion_matrix"], ["non_fall", "fall"])
            save_confusion_figure(cm_dir, f"{rule_id}_{eval_id}_direction", metrics["direction_confusion_matrix"], DIRECTION_LABELS)
    return rows


def mask_name_to_mask(masks: dict[str, np.ndarray], name: str) -> np.ndarray:
    return masks[name]


def select_threshold_rules(y_val: np.ndarray, prob_val: np.ndarray) -> dict[str, dict[str, Any]]:
    thresholds = candidate_thresholds(prob_val)
    rows = []
    for threshold in thresholds:
        pred = (prob_val >= threshold).astype(np.int64)
        cm = confusion_matrix(y_val, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        precision = precision_score(y_val, pred, zero_division=0)
        recall = recall_score(y_val, pred, zero_division=0)
        f1 = f1_score(y_val, pred, zero_division=0)
        rows.append({"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1, "FP": int(fp), "FN": int(fn), "TP": int(tp), "TN": int(tn)})
    df = pd.DataFrame(rows)

    def best_f1(frame: pd.DataFrame) -> pd.Series:
        if frame.empty:
            frame = df
        return frame.sort_values(["f1", "precision", "threshold"], ascending=[False, False, False]).iloc[0]

    t1 = best_f1(df)
    t2 = best_f1(df[df["precision"] >= 0.65])
    t3 = best_f1(df[df["precision"] >= 0.70])
    t4_pool = df[df["recall"] >= 0.88]
    t4 = t4_pool.sort_values(["FP", "f1", "threshold"], ascending=[True, False, False]).iloc[0] if not t4_pool.empty else best_f1(df)
    return {
        "T1": {"threshold": float(t1["threshold"]), "description": "global threshold maximizing validation Fall F1"},
        "T2": {"threshold": float(t2["threshold"]), "description": "global threshold maximizing validation Fall F1 with validation precision >= 0.65"},
        "T3": {"threshold": float(t3["threshold"]), "description": "global threshold maximizing validation Fall F1 with validation precision >= 0.70"},
        "T4": {"threshold": float(t4["threshold"]), "description": "global threshold minimizing validation FP with validation recall >= 0.88"},
    }


def candidate_thresholds(prob: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, 1001)
    unique = np.unique(np.clip(prob, 0.0, 1.0))
    eps_down = np.clip(unique - 1e-7, 0.0, 1.0)
    return np.unique(np.concatenate([grid, unique, eps_down]))


def run_hn_experiment(
    cfg: HNConfig,
    seed: int,
    run_root: Path,
    report_dir: Path,
    cm_dir: Path,
    meta: pd.DataFrame,
    X_scaled: np.ndarray,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    split_masks: dict[str, np.ndarray],
    reference,
    ref_pred: dict[str, np.ndarray],
    hn_info: pd.DataFrame,
    batch_size: int,
    force: bool,
    artifact_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import tensorflow as tf

    this_run = ensure_dir(run_root / cfg.run_id)
    summary_path = this_run / "summary_rows.csv"
    per_class_path = this_run / "per_class_rows.csv"
    if summary_path.exists() and per_class_path.exists() and not force:
        return pd.read_csv(summary_path).to_dict(orient="records"), pd.read_csv(per_class_path).to_dict(orient="records")

    set_seed(seed)
    print(f"\n=== {cfg.run_id}: {cfg.method}, source={cfg.hard_negative_source} ===")
    hn_col = "hn_feature_rule" if cfg.hard_negative_source == "feature_rule" else "hn_prediction_rule"
    hn_flags = hn_info[hn_col].to_numpy(dtype=bool)
    train_mask = split_masks["train"]
    val_mask = split_masks["val"]

    kd_output = cfg.train_mode == "fall_branch_kd"
    model = build_standard_model(adapter_mode="none", kd_output=kd_output, width_multiplier=1.0)
    if cfg.train_mode in {"fall_branch_finetune", "fall_branch_kd"}:
        if kd_output:
            reference_shape_model = build_standard_model(adapter_mode="none", kd_output=False, width_multiplier=1.0)
            reference_shape_model.load_weights(str(artifact_dir / "E3_BITS_WEDA_MIXED" / "best_model.keras"))
            copy_matching_weights(reference_shape_model, model)
        else:
            model.load_weights(str(artifact_dir / "E3_BITS_WEDA_MIXED" / "best_model.keras"))
        configure_finetune_trainability(model, mode=cfg.train_mode)

    fall_weights = compute_class_weights(y_fall[train_mask], num_classes=2)
    direction_weights = compute_class_weights(y_direction[train_mask], num_classes=3, mask=direction_mask[train_mask] * (y_fall[train_mask] == 1))
    compile_hn_model(model, cfg, fall_weights, direction_weights)

    teacher_direction = ref_pred["direction_probs"].astype(np.float32)
    train_ds = make_dataset(
        X_scaled[train_mask],
        y_fall[train_mask],
        y_direction[train_mask],
        direction_mask[train_mask],
        hn_flags[train_mask],
        cfg.hard_negative_weight,
        batch_size,
        seed,
        shuffle=True,
        kd_targets=teacher_direction[train_mask] if kd_output else None,
    )
    val_ds = make_dataset(
        X_scaled[val_mask],
        y_fall[val_mask],
        y_direction[val_mask],
        direction_mask[val_mask],
        np.zeros(int(val_mask.sum()), dtype=bool),
        1.0,
        batch_size,
        seed,
        shuffle=False,
        kd_targets=teacher_direction[val_mask] if kd_output else None,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=cfg.patience, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=max(2, cfg.patience // 2), min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(str(this_run / "training_history.csv")),
    ]
    start = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=callbacks, verbose=1)
    train_seconds = time.perf_counter() - start
    pd.DataFrame(history.history).to_csv(this_run / "history.csv", index=False)

    model.save_weights(this_run / "model.weights.h5")
    save_json(this_run / "config.json", asdict(cfg) | {"seed": seed, "train_seconds": train_seconds, "params": int(model.count_params())})
    save_json(this_run / "class_weights.json", {"fall": fall_weights.tolist(), "direction": direction_weights.tolist()})

    pred = predict_model(model, X_scaled, batch_size=batch_size)
    rules = select_threshold_rules(y_fall[val_mask], pred["fall_prob"][val_mask])
    threshold = float(rules["T1"]["threshold"])
    save_json(this_run / "thresholds.json", rules)

    rows = []
    per_class_rows = []
    for eval_id, mask_name, dataset_name in [
        ("E3", "test", "bits+weda"),
        ("E6", "test_bits", "bits"),
        ("E7", "test_weda", "weda"),
    ]:
        mask = split_masks[mask_name]
        metrics = evaluate_predictions(y_fall[mask], y_direction[mask], direction_mask[mask], pred["fall_prob"][mask], pred["direction_pred"][mask], threshold)
        row = {
            "block": "hard_negative_training",
            "run_id": cfg.run_id,
            "method": cfg.method,
            "hard_negative_source": cfg.hard_negative_source,
            "hard_negative_weight": cfg.hard_negative_weight,
            "alpha_fall": cfg.alpha_fall,
            "lambda_dir": cfg.lambda_dir,
            "beta_kd": cfg.beta_kd,
            "train_mode": cfg.train_mode,
            "eval": eval_id,
            "dataset": dataset_name,
            "threshold": threshold,
            "params": int(model.count_params()),
            "train_seconds": train_seconds,
            "train_hard_negative_count": int((hn_flags & train_mask).sum()),
            "val_hard_negative_count": int((hn_flags & val_mask).sum()),
            **metrics,
            "weda_fp_reduction_vs_ref": REF_METRICS["e7_weda_fp"] - metrics["FP"] if eval_id == "E7" else np.nan,
            "weda_precision_gain_vs_ref": metrics["fall_precision"] - REF_METRICS["e7_weda_precision"] if eval_id == "E7" else np.nan,
            "weda_recall_delta_vs_ref": metrics["fall_recall"] - REF_METRICS["e7_weda_recall"] if eval_id == "E7" else np.nan,
        }
        rows.append(row)
        per_class = metrics["direction_per_class_f1"]
        per_class_rows.append(
            {
                "run_id": cfg.run_id,
                "eval": eval_id,
                "dataset": dataset_name,
                "forward_f1": per_class.get("forward", np.nan),
                "backward_f1": per_class.get("backward", np.nan),
                "lateral_f1": per_class.get("lateral", np.nan),
            }
        )
        save_confusion_figure(cm_dir, f"{cfg.run_id}_{eval_id}_fall", metrics["fall_confusion_matrix"], ["non_fall", "fall"])
        save_confusion_figure(cm_dir, f"{cfg.run_id}_{eval_id}_direction", metrics["direction_confusion_matrix"], DIRECTION_LABELS)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False)
    return rows, per_class_rows


def configure_finetune_trainability(model, mode: str) -> None:
    for layer in model.layers:
        layer.trainable = False
        if mode == "fall_branch_finetune":
            if layer.name.startswith("fall_attention_pooling") or layer.name.startswith("fall_dense") or layer.name.startswith("fall_output"):
                layer.trainable = True
        elif mode == "fall_branch_kd":
            if (
                layer.name.startswith("dstcn_d4")
                or layer.name.startswith("fall_attention_pooling")
                or layer.name.startswith("fall_dense")
                or layer.name.startswith("fall_output")
            ):
                layer.trainable = True


def compile_hn_model(model, cfg: HNConfig, fall_weights: np.ndarray, direction_weights: np.ndarray) -> None:
    import tensorflow as tf

    losses = {
        "fall_output": make_sparse_ce_loss(class_weights=fall_weights, name="fall_weighted_ce"),
        "direction_output": make_direction_loss("weighted_ce", class_weights=direction_weights),
    }
    loss_weights = {"fall_output": cfg.alpha_fall, "direction_output": cfg.lambda_dir}
    if cfg.train_mode == "fall_branch_kd":
        losses["direction_kd_output"] = tf.keras.losses.KLDivergence(name="direction_kd")
        loss_weights["direction_kd_output"] = cfg.beta_kd
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate), loss=losses, loss_weights=loss_weights)


def make_dataset(
    X: np.ndarray,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    hard_negative: np.ndarray,
    hard_negative_weight: float,
    batch_size: int,
    seed: int,
    shuffle: bool,
    kd_targets: np.ndarray | None = None,
):
    import tensorflow as tf

    yd = prepare_direction_targets(y_direction)
    fall_sw = np.ones(len(y_fall), dtype=np.float32)
    fall_sw[np.asarray(hard_negative, dtype=bool)] *= float(hard_negative_weight)
    dir_sw = (np.asarray(direction_mask, dtype=np.float32) * (np.asarray(y_fall) == 1)).astype(np.float32)
    y = {"fall_output": y_fall.astype(np.int64), "direction_output": yd.astype(np.int64)}
    sw = {"fall_output": fall_sw, "direction_output": dir_sw}
    if kd_targets is not None:
        y["direction_kd_output"] = kd_targets.astype(np.float32)
        sw["direction_kd_output"] = dir_sw
    ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y, sw))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 8192), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def evaluate_predictions(
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    fall_prob: np.ndarray,
    direction_pred: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    pred_fall = (fall_prob >= threshold).astype(np.int64)
    cm = confusion_matrix(y_fall, pred_fall, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    supervised = (direction_mask > 0) & (y_direction >= 0)
    if supervised.any():
        dir_cm = confusion_matrix(y_direction[supervised], direction_pred[supervised], labels=[0, 1, 2])
        direction_macro = f1_score(y_direction[supervised], direction_pred[supervised], labels=[0, 1, 2], average="macro", zero_division=0)
        direction_acc = accuracy_score(y_direction[supervised], direction_pred[supervised])
        per_class = {
            name: f1_score(y_direction[supervised], direction_pred[supervised], labels=[i], average="macro", zero_division=0)
            for i, name in enumerate(DIRECTION_LABELS)
        }
    else:
        dir_cm = np.zeros((3, 3), dtype=int)
        direction_macro = np.nan
        direction_acc = np.nan
        per_class = {name: np.nan for name in DIRECTION_LABELS}
    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "fall_precision": precision_score(y_fall, pred_fall, zero_division=0),
        "fall_recall": recall_score(y_fall, pred_fall, zero_division=0),
        "fall_f1": f1_score(y_fall, pred_fall, zero_division=0),
        "fall_accuracy": accuracy_score(y_fall, pred_fall),
        "direction_macro_f1": direction_macro,
        "direction_accuracy": direction_acc,
        "direction_n_supervised": int(supervised.sum()),
        "fall_confusion_matrix": cm.tolist(),
        "direction_confusion_matrix": dir_cm.tolist(),
        "direction_per_class_f1": per_class,
    }


def save_confusion_figure(cm_dir: Path, name: str, matrix: list[list[int]], labels: list[str]) -> None:
    matrix_arr = np.asarray(matrix, dtype=int)
    pd.DataFrame(matrix_arr, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels]).to_csv(cm_dir / f"{name}.csv")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(matrix_arr, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(matrix_arr.shape[0]):
        for j in range(matrix_arr.shape[1]):
            ax.text(j, i, str(matrix_arr[i, j]), ha="center", va="center", color="black")
    ax.set_title(name)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(cm_dir / f"{name}.png", dpi=170)
    plt.close(fig)


def flatten_metrics_for_report(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [
        "run_id",
        "eval",
        "dataset",
        "threshold",
        "TP",
        "FP",
        "FN",
        "TN",
        "fall_precision",
        "fall_recall",
        "fall_f1",
        "direction_macro_f1",
        "direction_n_supervised",
        "params",
        "weda_fp_reduction_vs_ref",
        "weda_precision_gain_vs_ref",
        "weda_recall_delta_vs_ref",
    ]
    return df[[c for c in cols if c in df.columns]].copy()


def write_reports(report_dir: Path, threshold_df: pd.DataFrame, hard_df: pd.DataFrame, per_class_df: pd.DataFrame) -> None:
    best_threshold = select_best_threshold_result(threshold_df)
    best_hn = select_best_hn_result(hard_df)
    threshold_filter_rows = threshold_hard_filter_passes(threshold_df)
    hard_filter_rows = hard_filter_passes(hard_df)
    hard_target_rows = target_fp_passes(hard_df)
    selection_lines = [
        "# Best FP Reduction Selection",
        "",
        "Hard filters: E3 Direction Macro F1 >= 0.86, E7 WEDA Direction Macro F1 >= 0.80, E6 BITS Fall F1 >= 0.90, E7 WEDA Recall >= 0.88, Params <= 90k.",
        "Desired FP target: E7 WEDA FP < 12, precision > 0.65, recall >= 0.88.",
        "",
        "## Threshold-only Best",
        "",
        markdown_table(format_df(best_threshold)),
        "",
        "## Hard-negative Best",
        "",
        markdown_table(format_df(best_hn)),
        "",
        "## Threshold Rules Passing Hard Filters",
        "",
        markdown_table(format_df(threshold_filter_rows)),
        "",
        "## Hard-negative Runs Passing Hard Filters",
        "",
        markdown_table(format_df(hard_filter_rows)),
        "",
        "## Hard-negative Runs Passing Hard Filters And FP Target",
        "",
        markdown_table(format_df(hard_target_rows)),
        "",
        "## Recommendation",
        "",
    ]
    if hard_target_rows.empty:
        selection_lines.append(
            "No hard-negative run satisfies both the hard filters and the desired FP target. Keep `REF_CURRENT_E3_ARTIFACT` as the main model. "
            "Use T1 as a decision-threshold ablation, HN1 as aggressive FP-reduction ablation, and HN4 as direction-safe but modest FP-reduction ablation."
        )
    else:
        chosen = hard_target_rows.sort_values(["E7_WEDA_FP", "E7_WEDA_Fall_F1"], ascending=[True, False]).iloc[0]
        selection_lines.append(f"`{chosen['run_id']}` passes hard filters and the FP target; repeat across seeds before replacing the reference.")
    (report_dir / "best_fp_reduction_selection.md").write_text("\n".join(selection_lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# FP Reduction Experiments Summary",
        "",
        "Scope: BITS/WEDA only, 25 Hz, 2-second event-centered windows, tilt12. No new dataset and no large architecture change.",
        "",
        "## Block A: Threshold / Decision Analysis",
        "",
        markdown_table(format_df(flatten_metrics_for_report(threshold_df))),
        "",
        "## Block B: Hard-negative Training",
        "",
        markdown_table(format_df(flatten_metrics_for_report(hard_df))),
        "",
        "## Per-class Direction F1",
        "",
        markdown_table(format_df(per_class_df)),
        "",
        "## Answers",
        "",
        answer_text(threshold_df, hard_df),
    ]
    (report_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def select_best_threshold_result(threshold_df: pd.DataFrame) -> pd.DataFrame:
    if threshold_df.empty:
        return pd.DataFrame()
    e7 = threshold_df[threshold_df["eval"].eq("E7")].copy()
    e3 = threshold_df[threshold_df["eval"].eq("E3")][["run_id", "direction_macro_f1"]].rename(columns={"direction_macro_f1": "E3_Direction_Macro_F1"})
    e6 = threshold_df[threshold_df["eval"].eq("E6")][["run_id", "fall_f1", "direction_macro_f1"]].rename(columns={"fall_f1": "E6_BITS_Fall_F1", "direction_macro_f1": "E6_BITS_Direction_Macro_F1"})
    merged = e7.merge(e3, on="run_id").merge(e6, on="run_id")
    merged = merged.rename(columns={"fall_f1": "E7_WEDA_Fall_F1", "fall_precision": "E7_WEDA_Precision", "fall_recall": "E7_WEDA_Recall", "direction_macro_f1": "E7_WEDA_Direction_Macro_F1", "FP": "E7_WEDA_FP", "FN": "E7_WEDA_FN"})
    return merged.sort_values(["E7_WEDA_FP", "E7_WEDA_Fall_F1"], ascending=[True, False]).head(1)


def select_best_hn_result(hard_df: pd.DataFrame) -> pd.DataFrame:
    if hard_df.empty:
        return pd.DataFrame()
    merged = pivot_run_metrics(hard_df)
    return merged.sort_values(["E7_WEDA_FP", "E7_WEDA_Fall_F1", "E3_Direction_Macro_F1"], ascending=[True, False, False]).head(1)


def hard_filter_passes(hard_df: pd.DataFrame) -> pd.DataFrame:
    merged = pivot_run_metrics(hard_df)
    if merged.empty:
        return merged
    passed = merged[
        (merged["E3_Direction_Macro_F1"] >= 0.86)
        & (merged["E7_WEDA_Direction_Macro_F1"] >= 0.80)
        & (merged["E6_BITS_Fall_F1"] >= 0.90)
        & (merged["E7_WEDA_Recall"] >= 0.88)
        & (merged["params"] <= 90000)
    ].copy()
    return passed


def threshold_hard_filter_passes(threshold_df: pd.DataFrame) -> pd.DataFrame:
    merged = pivot_threshold_metrics(threshold_df)
    if merged.empty:
        return merged
    return merged[
        (merged["E3_Direction_Macro_F1"] >= 0.86)
        & (merged["E7_WEDA_Direction_Macro_F1"] >= 0.80)
        & (merged["E6_BITS_Fall_F1"] >= 0.90)
        & (merged["E7_WEDA_Recall"] >= 0.88)
        & (merged["params"] <= 90000)
    ].copy()


def target_fp_passes(hard_df: pd.DataFrame) -> pd.DataFrame:
    passed = hard_filter_passes(hard_df)
    if passed.empty:
        return passed
    return passed[
        (passed["E7_WEDA_FP"] < 12)
        & (passed["E7_WEDA_Precision"] > 0.65)
        & (passed["E7_WEDA_Recall"] >= 0.88)
    ].copy()


def pivot_threshold_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    e3 = df[df["eval"].eq("E3")][["run_id", "threshold", "direction_macro_f1", "fall_f1"]].rename(
        columns={"direction_macro_f1": "E3_Direction_Macro_F1", "fall_f1": "E3_Fall_F1"}
    )
    e6 = df[df["eval"].eq("E6")][["run_id", "fall_f1", "direction_macro_f1"]].rename(
        columns={"fall_f1": "E6_BITS_Fall_F1", "direction_macro_f1": "E6_BITS_Direction_Macro_F1"}
    )
    e7 = df[df["eval"].eq("E7")][
        ["run_id", "FP", "FN", "TP", "TN", "fall_precision", "fall_recall", "fall_f1", "direction_macro_f1"]
    ].rename(
        columns={
            "FP": "E7_WEDA_FP",
            "FN": "E7_WEDA_FN",
            "TP": "E7_WEDA_TP",
            "TN": "E7_WEDA_TN",
            "fall_precision": "E7_WEDA_Precision",
            "fall_recall": "E7_WEDA_Recall",
            "fall_f1": "E7_WEDA_Fall_F1",
            "direction_macro_f1": "E7_WEDA_Direction_Macro_F1",
        }
    )
    out = e7.merge(e3, on="run_id").merge(e6, on="run_id")
    out["params"] = REF_METRICS["params"]
    return out


def pivot_run_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    e3 = df[df["eval"].eq("E3")][["run_id", "direction_macro_f1", "fall_f1"]].rename(columns={"direction_macro_f1": "E3_Direction_Macro_F1", "fall_f1": "E3_Fall_F1"})
    e6 = df[df["eval"].eq("E6")][["run_id", "fall_f1", "direction_macro_f1"]].rename(columns={"fall_f1": "E6_BITS_Fall_F1", "direction_macro_f1": "E6_BITS_Direction_Macro_F1"})
    e7 = df[df["eval"].eq("E7")][
        ["run_id", "method", "hard_negative_source", "threshold", "params", "FP", "FN", "TP", "TN", "fall_precision", "fall_recall", "fall_f1", "direction_macro_f1"]
    ].rename(
        columns={
            "FP": "E7_WEDA_FP",
            "FN": "E7_WEDA_FN",
            "TP": "E7_WEDA_TP",
            "TN": "E7_WEDA_TN",
            "fall_precision": "E7_WEDA_Precision",
            "fall_recall": "E7_WEDA_Recall",
            "fall_f1": "E7_WEDA_Fall_F1",
            "direction_macro_f1": "E7_WEDA_Direction_Macro_F1",
        }
    )
    return e7.merge(e3, on="run_id").merge(e6, on="run_id")


def answer_text(threshold_df: pd.DataFrame, hard_df: pd.DataFrame) -> str:
    best_t = select_best_threshold_result(threshold_df)
    best_h = select_best_hn_result(hard_df)
    threshold_passes = threshold_hard_filter_passes(threshold_df)
    hn_passes = hard_filter_passes(hard_df)
    hn_target_passes = target_fp_passes(hard_df)
    t_line = "No threshold result available."
    if not best_t.empty:
        t = best_t.iloc[0]
        t_line = f"Threshold-only can reduce WEDA FP to {int(t['E7_WEDA_FP'])} with precision {t['E7_WEDA_Precision']:.4f}, recall {t['E7_WEDA_Recall']:.4f}, Fall F1 {t['E7_WEDA_Fall_F1']:.4f}. Best rule by FP is {t['run_id']}."
    h_line = "No hard-negative run available."
    if not best_h.empty:
        h = best_h.iloc[0]
        h_line = f"Best hard-negative run by WEDA FP is {h['run_id']} ({h['method']}, {h['hard_negative_source']}): FP={int(h['E7_WEDA_FP'])}, precision={h['E7_WEDA_Precision']:.4f}, recall={h['E7_WEDA_Recall']:.4f}, Fall F1={h['E7_WEDA_Fall_F1']:.4f}."
    if hn_target_passes.empty:
        replace = (
            "No HN run reaches both the hard filters and the desired FP target. "
            "Keep `REF_CURRENT_E3_ARTIFACT` as the paper main model; report T1/HN1/HN4 as FP-aware ablations."
        )
    else:
        replace = "At least one HN run passes hard filters and the FP target; repeat it across seeds before replacing the reference."
    threshold_pass_line = (
        f"Threshold rules passing hard filters: {', '.join(threshold_passes['run_id'].astype(str).tolist())}."
        if not threshold_passes.empty
        else "No threshold rule passes all hard filters."
    )
    hn_pass_line = (
        f"HN runs passing hard filters: {', '.join(hn_passes['run_id'].astype(str).tolist())}."
        if not hn_passes.empty
        else "No HN run passes all hard filters."
    )
    return "\n".join(
        [
            f"1. {t_line}",
            f"2. The best T1-T4 rule is the row selected above by lowest WEDA FP, with Fall F1 as tie-breaker. {threshold_pass_line}",
            f"3. {h_line} It reduces FP more than threshold-only, but check recall/direction hard filters before treating it as a replacement.",
            "4. HN1-HN4 comparison is in `hard_negative_runs.csv`; the best row above is selected by WEDA FP first, then WEDA Fall F1.",
            f"5. {hn_pass_line} {replace}",
            "6. If no run passes hard filters, keep `REF_CURRENT_E3_ARTIFACT` and describe hard-negative training as an FP-aware ablation.",
            "7. Paper-facing: WEDA fall weakness is a hard-negative ADL precision problem; FP-aware thresholding/training quantifies how far precision can be improved before recall/direction trade-offs appear.",
        ]
    )


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
    drop_cols = [c for c in out.columns if c.endswith("confusion_matrix") or c == "direction_per_class_f1"]
    out = out.drop(columns=drop_cols, errors="ignore")
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            if col in {"TP", "FP", "FN", "TN", "params", "direction_n_supervised", "E7_WEDA_FP", "E7_WEDA_FN", "E7_WEDA_TP", "E7_WEDA_TN"}:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(int(v)))
            else:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    return out


if __name__ == "__main__":
    main()
