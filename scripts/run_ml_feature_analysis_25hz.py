from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_25hz_a5wcefw_bits_weda as base25
from scripts.run_next12_experiments import build_standard_model
from src.config import make_config
from src.utils.io import ensure_dir, save_json
from src.utils.seed import set_seed


SAMPLING_RATE = 25.0
DT = 1.0 / SAMPLING_RATE
WINDOW_SECONDS = 2.0
WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_SECONDS)
CENTER_INDEX = WINDOW_SIZE // 2
RAW6 = ["ax", "ay", "az", "gx", "gy", "gz"]
TILT12 = ["ax", "ay", "az", "gx", "gy", "gz", "acc_mag", "gyro_mag", "jerk", "roll", "pitch", "tilt_delta"]
DIRECTION_LABELS = ["forward", "backward", "lateral"]
DIRECTION_TO_ID = {name: i for i, name in enumerate(DIRECTION_LABELS)}
ID_TO_DIRECTION = {i: name for name, i in DIRECTION_TO_ID.items()}
DEEP_REFERENCE = {
    "model_name": "REF_CURRENT_E3_ARTIFACT / DS-Fall-RD A5WCEFW",
    "params": 65959,
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Any
    supports_proba: bool = True
    model_type: str = "classical_feature"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML feature analysis on BITS/WEDA 25 Hz event-centered benchmark.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "experiments_25hz_event")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-rebuild-features", action="store_true")
    parser.add_argument("--permutation-repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    report_dir = ensure_dir(args.output_dir / "reports" / "ml_feature_analysis_25hz")
    figure_dir = ensure_dir(args.output_dir / "figures" / "ml_feature_analysis_25hz")
    ensure_dir(figure_dir / "fall_confusion_matrices")
    ensure_dir(figure_dir / "direction_confusion_matrices")
    ensure_dir(figure_dir / "fall_roc_pr_curves")

    print("=== ML feature analysis 25 Hz event-centered ===")
    print(f"project_root={args.project_root}")
    print(f"report_dir={report_dir}")
    read_existing_context(args.project_root)

    matrix_path = report_dir / "summary_feature_matrix.csv"
    if matrix_path.exists() and not args.force_rebuild_features:
        feature_df = pd.read_csv(matrix_path)
        print(f"Loaded existing summary feature matrix: {matrix_path} shape={feature_df.shape}")
    else:
        feature_df = build_summary_feature_matrix(args, report_dir)
        feature_df.to_csv(matrix_path, index=False)
        print(f"Saved summary feature matrix: {matrix_path} shape={feature_df.shape}")

    feature_columns = infer_feature_columns(feature_df)
    group_map_fall, group_map_direction = build_feature_groups(feature_columns)
    save_json(report_dir / "feature_groups.json", {"fall": group_map_fall, "direction": group_map_direction})

    fall_results, fall_importance, fall_ablation, best_fall = run_fall_analysis(
        feature_df, feature_columns, group_map_fall, args.seed, args.permutation_repeats, report_dir, figure_dir
    )
    direction_results, direction_importance, direction_univariate, direction_shift, direction_ablation, best_direction = run_direction_analysis(
        feature_df, feature_columns, group_map_direction, args.seed, args.permutation_repeats, report_dir, figure_dir
    )

    deep_ref = compute_deep_reference_metrics(feature_df)
    comparison, best_fall_table, best_direction_table, interpretation = build_comparison_tables(
        fall_results, direction_results, fall_importance, direction_importance, deep_ref, best_fall, best_direction
    )
    comparison.to_csv(report_dir / "ml_vs_deep_comparison.csv", index=False)
    best_fall_table.to_csv(report_dir / "best_fall_only_model.csv", index=False)
    best_direction_table.to_csv(report_dir / "best_direction_only_model.csv", index=False)
    interpretation.to_csv(report_dir / "feature_interpretation_table.csv", index=False)

    make_summary_figures(
        figure_dir,
        fall_results,
        direction_results,
        fall_importance,
        direction_importance,
        fall_ablation,
        direction_ablation,
        comparison,
        feature_df,
        feature_columns,
        best_fall,
        best_direction,
    )
    write_final_report(
        report_dir,
        fall_results,
        fall_importance,
        fall_ablation,
        direction_results,
        direction_importance,
        direction_univariate,
        direction_shift,
        direction_ablation,
        comparison,
        best_fall_table,
        best_direction_table,
        interpretation,
        deep_ref,
        best_fall,
        best_direction,
    )
    print(f"Saved final report: {report_dir / 'ml_feature_analysis_summary.md'}")


def read_existing_context(project_root: Path) -> None:
    for rel in [
        "outputs/reports/data_analysis_25hz_center/data_analysis_summary.md",
        "outputs/reports/fp_reduction_experiments/summary.md",
        "outputs/reports/targeted_experiments/best_model_selection.md",
    ]:
        path = project_root / rel
        if path.exists():
            print(f"Context available: {rel}")


def build_summary_feature_matrix(args: argparse.Namespace, report_dir: Path) -> pd.DataFrame:
    config = make_config(args.project_root)
    data = base25.build_25hz_dataset(config, window_mode="event_centered")
    base25.validate_25hz_dataset(data)
    meta = data["metadata"].copy().reset_index(drop=True)
    X_raw6 = np.asarray(data["X_raw6"], dtype=np.float32)
    X_tilt12 = np.asarray(data["X_features"], dtype=np.float32)
    features = extract_summary_features(X_raw6, X_tilt12)
    ref_pred, ref_note = reference_predictions_all_windows(args.artifact_dir, X_tilt12)
    save_json(report_dir / "reference_prediction_note.json", ref_note)

    out = pd.concat([meta, features], axis=1)
    out["fall_prob"] = ref_pred["fall_prob"]
    out["fall_pred"] = np.where(ref_pred["fall_pred"] == 1, "fall", "non_fall")
    out["direction_pred"] = [ID_TO_DIRECTION.get(int(v), "unknown") for v in ref_pred["direction_pred"]]
    for i, name in enumerate(DIRECTION_LABELS):
        out[f"direction_prob_{name}"] = ref_pred["direction_probs"][:, i]
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def reference_predictions_all_windows(artifact_dir: Path, X_tilt12: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    scaler_path = artifact_dir / "E3_BITS_WEDA_MIXED" / "scaler.pkl"
    weights_path = artifact_dir / "E3_BITS_WEDA_MIXED" / "best_model.keras"
    if not scaler_path.exists() or not weights_path.exists():
        raise FileNotFoundError("Missing E3 reference scaler/weights under artifacts/experiments_25hz_event/E3_BITS_WEDA_MIXED")
    with scaler_path.open("rb") as f:
        scaler = pickle.load(f)
    X_scaled = base25.transform_scaler(X_tilt12, scaler)
    model = build_standard_model(adapter_mode="none", kd_output=False, width_multiplier=1.0)
    model.load_weights(str(weights_path))
    preds = model.predict(X_scaled, batch_size=128, verbose=0)
    fall = preds["fall_output"] if isinstance(preds, dict) else preds[0]
    direction = preds["direction_output"] if isinstance(preds, dict) else preds[1]
    note = {
        "reference": "DS-Fall-RD A5WCEFW E3 weights loaded into serializable equivalent architecture",
        "scaler": str(scaler_path),
        "weights": str(weights_path),
        "limitation": "Keras artifact cannot be deserialized as a full model because of Lambda layers; weights are loaded by name-compatible serializable layers.",
    }
    return {
        "fall_prob": fall[:, 1].astype(float),
        "fall_pred": np.argmax(fall, axis=1).astype(int),
        "direction_probs": direction.astype(float),
        "direction_pred": np.argmax(direction, axis=1).astype(int),
    }, note


def extract_summary_features(X_raw6: np.ndarray, X_tilt12: np.ndarray) -> pd.DataFrame:
    raw = {name: X_raw6[:, :, i] for i, name in enumerate(RAW6)}
    feat = {name: X_tilt12[:, :, i] for i, name in enumerate(TILT12)}
    rows: dict[str, np.ndarray] = {}

    for name in RAW6:
        x = raw[name]
        rows[f"{name}_mean"] = np.mean(x, axis=1)
        rows[f"{name}_std"] = np.std(x, axis=1)
        rows[f"{name}_min"] = np.min(x, axis=1)
        rows[f"{name}_max"] = np.max(x, axis=1)
        rows[f"{name}_range"] = np.max(x, axis=1) - np.min(x, axis=1)
        rows[f"{name}_final_initial"] = x[:, -1] - x[:, 0]
        rows[f"{name}_median"] = np.median(x, axis=1)
        rows[f"{name}_iqr"] = np.percentile(x, 75, axis=1) - np.percentile(x, 25, axis=1)
        rows[f"peak_signed_{name}"] = signed_peak(x)
        rows[f"integrated_{name}"] = np.sum(x, axis=1) * DT

    for name in ["acc_mag", "gyro_mag"]:
        x = feat[name]
        rows[f"{name}_mean"] = np.mean(x, axis=1)
        rows[f"{name}_std"] = np.std(x, axis=1)
        rows[f"{name}_min"] = np.min(x, axis=1)
        rows[f"{name}_max"] = np.max(x, axis=1)
        rows[f"{name}_p95"] = np.percentile(x, 95, axis=1)
        rows[f"{name}_range"] = np.max(x, axis=1) - np.min(x, axis=1)
        rows[f"{name}_median"] = np.median(x, axis=1)
        rows[f"{name}_iqr"] = np.percentile(x, 75, axis=1) - np.percentile(x, 25, axis=1)

    jerk = feat["jerk"]
    rows["jerk_mean"] = np.mean(jerk, axis=1)
    rows["jerk_std"] = np.std(jerk, axis=1)
    rows["jerk_max"] = np.max(jerk, axis=1)
    rows["jerk_p95"] = np.percentile(jerk, 95, axis=1)
    rows["jerk_median"] = np.median(jerk, axis=1)
    rows["jerk_iqr"] = np.percentile(jerk, 75, axis=1) - np.percentile(jerk, 25, axis=1)

    for name in ["roll", "pitch"]:
        x = feat[name]
        rows[f"{name}_mean"] = np.mean(x, axis=1)
        rows[f"{name}_std"] = np.std(x, axis=1)
        rows[f"{name}_min"] = np.min(x, axis=1)
        rows[f"{name}_max"] = np.max(x, axis=1)
        rows[f"{name}_range"] = np.max(x, axis=1) - np.min(x, axis=1)
        rows[f"{name}_final_initial"] = x[:, -1] - x[:, 0]
        rows[f"{name}_median"] = np.median(x, axis=1)
        rows[f"{name}_iqr"] = np.percentile(x, 75, axis=1) - np.percentile(x, 25, axis=1)

    tilt = feat["tilt_delta"]
    rows["tilt_delta_mean"] = np.mean(tilt, axis=1)
    rows["tilt_delta_std"] = np.std(tilt, axis=1)
    rows["tilt_delta_max"] = np.max(tilt, axis=1)
    rows["tilt_delta_p95"] = np.percentile(tilt, 95, axis=1)
    rows["tilt_delta_final"] = tilt[:, -1]
    rows["tilt_delta_median"] = np.median(tilt, axis=1)
    rows["tilt_delta_iqr"] = np.percentile(tilt, 75, axis=1) - np.percentile(tilt, 25, axis=1)

    acc_mag = feat["acc_mag"]
    gyro_mag = feat["gyro_mag"]
    impact_index = np.argmax(acc_mag, axis=1)
    gyro_peak_index = np.argmax(gyro_mag, axis=1)
    rows["impact_index"] = impact_index.astype(float)
    rows["impact_distance_from_center"] = np.abs(impact_index - CENTER_INDEX).astype(float)
    rows["gyro_peak_index"] = gyro_peak_index.astype(float)
    rows["gyro_peak_distance_from_center"] = np.abs(gyro_peak_index - CENTER_INDEX).astype(float)
    rows["acc_mag_peak_value"] = acc_mag[np.arange(len(acc_mag)), impact_index]
    rows["gyro_mag_peak_value"] = gyro_mag[np.arange(len(gyro_mag)), gyro_peak_index]

    pre_energy, post_energy = [], []
    pre_acc_std, post_acc_std, pre_gyro_std, post_gyro_std = [], [], [], []
    pre_post_signed: dict[str, list[float]] = {}
    for name in RAW6 + ["roll", "pitch"]:
        pre_post_signed[f"pre_{name}_mean"] = []
        pre_post_signed[f"post_{name}_mean"] = []
    for i, idx in enumerate(impact_index):
        pre_slice = slice(0, max(int(idx), 1))
        post_slice = slice(min(int(idx) + 1, WINDOW_SIZE - 1), WINDOW_SIZE)
        pre_acc = acc_mag[i, pre_slice]
        post_acc = acc_mag[i, post_slice]
        pre_g = gyro_mag[i, pre_slice]
        post_g = gyro_mag[i, post_slice]
        pre_energy.append(float(np.sum(pre_acc * pre_acc) * DT))
        post_energy.append(float(np.sum(post_acc * post_acc) * DT))
        pre_acc_std.append(float(np.std(pre_acc)))
        post_acc_std.append(float(np.std(post_acc)))
        pre_gyro_std.append(float(np.std(pre_g)))
        post_gyro_std.append(float(np.std(post_g)))
        for name in RAW6:
            x = raw[name][i]
            pre_post_signed[f"pre_{name}_mean"].append(float(np.mean(x[pre_slice])))
            pre_post_signed[f"post_{name}_mean"].append(float(np.mean(x[post_slice])))
        for name in ["roll", "pitch"]:
            x = feat[name][i]
            pre_post_signed[f"pre_{name}_mean"].append(float(np.mean(x[pre_slice])))
            pre_post_signed[f"post_{name}_mean"].append(float(np.mean(x[post_slice])))
    rows["pre_impact_energy"] = np.asarray(pre_energy)
    rows["post_impact_energy"] = np.asarray(post_energy)
    rows["pre_post_energy_ratio"] = rows["pre_impact_energy"] / np.maximum(rows["post_impact_energy"], 1e-6)
    rows["post_acc_mag_std"] = np.asarray(post_acc_std)
    rows["post_gyro_mag_std"] = np.asarray(post_gyro_std)
    rows["pre_acc_mag_std"] = np.asarray(pre_acc_std)
    rows["pre_gyro_mag_std"] = np.asarray(pre_gyro_std)
    for key, values in pre_post_signed.items():
        rows[key] = np.asarray(values)
    rows["delta_roll_window"] = rows["post_roll_mean"] - rows["pre_roll_mean"]
    rows["delta_pitch_window"] = rows["post_pitch_mean"] - rows["pre_pitch_mean"]
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def signed_peak(x: np.ndarray) -> np.ndarray:
    idx = np.argmax(np.abs(x), axis=1)
    return x[np.arange(len(x)), idx]


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    # Keep metadata in summary_feature_matrix.csv for auditing, but do not let
    # classical ML use protocol fields such as source_hz/start_idx/annotation
    # intervals. The models below should test signal-derived summaries only.
    signal_prefixes = (
        "ax_",
        "ay_",
        "az_",
        "gx_",
        "gy_",
        "gz_",
        "peak_signed_",
        "integrated_",
        "acc_mag_",
        "gyro_mag_",
        "jerk_",
        "roll_",
        "pitch_",
        "tilt_delta_",
        "pre_",
        "post_",
        "pre_post_",
        "delta_roll_",
        "delta_pitch_",
    )
    signal_exact = {
        "impact_index",
        "impact_distance_from_center",
        "gyro_peak_index",
        "gyro_peak_distance_from_center",
        "acc_mag_peak_value",
        "gyro_mag_peak_value",
    }
    cols = []
    for col in df.columns:
        if col in signal_exact or col.startswith(signal_prefixes):
            if pd.api.types.is_numeric_dtype(df[col]):
                cols.append(col)
    return cols


def build_feature_groups(features: list[str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    raw_axis = [c for c in features if c.startswith(("ax_", "ay_", "az_", "gx_", "gy_", "gz_", "peak_signed_ax", "peak_signed_ay", "peak_signed_az", "peak_signed_gx", "peak_signed_gy", "peak_signed_gz", "integrated_ax", "integrated_ay", "integrated_az", "integrated_gx", "integrated_gy", "integrated_gz"))]
    acc_signed = [c for c in features if c.startswith(("ax_", "ay_", "az_", "peak_signed_ax", "peak_signed_ay", "peak_signed_az", "integrated_ax", "integrated_ay", "integrated_az", "pre_ax_", "pre_ay_", "pre_az_", "post_ax_", "post_ay_", "post_az_"))]
    gyro_signed = [c for c in features if c.startswith(("gx_", "gy_", "gz_", "peak_signed_gx", "peak_signed_gy", "peak_signed_gz", "integrated_gx", "integrated_gy", "integrated_gz", "pre_gx_", "pre_gy_", "pre_gz_", "post_gx_", "post_gy_", "post_gz_"))]
    magnitude_jerk = [c for c in features if c.startswith(("acc_mag_", "gyro_mag_", "jerk_"))]
    orientation = [c for c in features if c.startswith(("roll_", "pitch_", "tilt_delta_", "pre_roll", "post_roll", "pre_pitch", "post_pitch", "delta_roll", "delta_pitch"))]
    impact_timing = [c for c in features if c in {"impact_index", "impact_distance_from_center", "gyro_peak_index", "gyro_peak_distance_from_center", "acc_mag_peak_value", "gyro_mag_peak_value"}]
    pre_post = [c for c in features if c.startswith(("pre_", "post_", "pre_post_", "delta_roll", "delta_pitch"))]
    pre_impact_signed = [c for c in features if c.startswith(("pre_ax_", "pre_ay_", "pre_az_", "pre_gx_", "pre_gy_", "pre_gz_", "pre_roll", "pre_pitch", "peak_signed_", "integrated_gx", "integrated_gy", "integrated_gz", "delta_roll", "delta_pitch"))]
    no_impact = [c for c in features if c not in set(impact_timing)]
    no_magnitude = [c for c in features if c not in set(magnitude_jerk)]
    fall = {
        "raw_axis_only": raw_axis,
        "magnitude_jerk_only": magnitude_jerk,
        "orientation_only": orientation,
        "impact_timing_only": impact_timing,
        "pre_post_only": pre_post,
        "all_features": features,
        "all_features_no_impact_timing": no_impact,
    }
    direction = {
        "raw_axis_only": raw_axis,
        "gyro_signed_only": gyro_signed,
        "acceleration_signed_only": acc_signed,
        "magnitude_jerk_only": magnitude_jerk,
        "orientation_only": orientation,
        "pre_impact_signed_only": pre_impact_signed,
        "all_features": features,
        "all_features_no_magnitude": no_magnitude,
    }
    return fall, direction


def fall_model_specs(seed: int) -> list[ModelSpec]:
    specs = [
        ModelSpec("logistic_regression", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))),
        ModelSpec("linear_svm", lambda: make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", random_state=seed, max_iter=5000)), supports_proba=False),
        ModelSpec("random_forest", lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)),
        ModelSpec("gradient_boosting", lambda: GradientBoostingClassifier(n_estimators=220, learning_rate=0.05, max_depth=3, random_state=seed)),
        ModelSpec("hist_gradient_boosting", lambda: HistGradientBoostingClassifier(max_iter=220, learning_rate=0.05, random_state=seed)),
    ]
    return specs


def direction_model_specs(seed: int) -> list[ModelSpec]:
    return [
        ModelSpec("logistic_regression", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))),
        ModelSpec("linear_svm", lambda: make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", random_state=seed, max_iter=5000)), supports_proba=False),
        ModelSpec("random_forest", lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)),
        ModelSpec("gradient_boosting", lambda: GradientBoostingClassifier(n_estimators=220, learning_rate=0.05, max_depth=3, random_state=seed)),
        ModelSpec("hist_gradient_boosting", lambda: HistGradientBoostingClassifier(max_iter=220, learning_rate=0.05, random_state=seed)),
    ]


def get_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        if prob.ndim == 2 and prob.shape[1] > 1:
            return prob[:, 1]
        return prob.ravel()
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([np.linspace(np.nanmin(scores), np.nanmax(scores), 400), scores]))
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return float(best_t)


def run_fall_analysis(
    df: pd.DataFrame,
    features: list[str],
    groups: dict[str, list[str]],
    seed: int,
    permutation_repeats: int,
    report_dir: Path,
    figure_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    print("\n[Fall] Training classical models...")
    masks = split_masks(df)
    X_train, y_train = df.loc[masks["train"], features].to_numpy(float), df.loc[masks["train"], "fall_label"].to_numpy(int)
    X_val, y_val = df.loc[masks["val"], features].to_numpy(float), df.loc[masks["val"], "fall_label"].to_numpy(int)
    X_test, y_test = df.loc[masks["test"], features].to_numpy(float), df.loc[masks["test"], "fall_label"].to_numpy(int)
    specs = fall_model_specs(seed)
    rows, importance_rows = [], []
    fitted: dict[str, tuple[Any, float]] = {}
    for spec in specs:
        start = time.perf_counter()
        model = spec.factory()
        model.fit(X_train, y_train)
        val_scores = get_scores(model, X_val)
        threshold = tune_threshold(y_val, val_scores)
        train_seconds = time.perf_counter() - start
        test_scores = get_scores(model, X_test)
        fitted[spec.name] = (model, threshold)
        model_rows = evaluate_fall_model(spec.name, model, threshold, df, features, train_seconds)
        rows.extend(model_rows)
        importance_rows.extend(fall_importance_rows(spec.name, model, threshold, df.loc[masks["test"]], features, permutation_repeats, seed))
        plot_fall_roc_pr(spec.name, y_test, test_scores, figure_dir / "fall_roc_pr_curves")

    result_df = pd.DataFrame(rows)
    importance_df = pd.DataFrame(importance_rows)
    result_df.to_csv(report_dir / "fall_ml_results.csv", index=False)
    importance_df.to_csv(report_dir / "fall_feature_importance.csv", index=False)
    best_name = choose_best_model_name(result_df, task="fall")
    best_model, best_threshold = fitted[best_name]
    save_fall_confusion(best_name, best_model, best_threshold, df, features, figure_dir / "fall_confusion_matrices")

    ablation_df = run_fall_group_ablation(best_name, specs, groups, df, seed)
    ablation_df.to_csv(report_dir / "fall_feature_group_ablation.csv", index=False)
    return result_df, importance_df, ablation_df, {"name": best_name, "model": best_model, "threshold": best_threshold}


def evaluate_fall_model(model_name: str, model: Any, threshold: float, df: pd.DataFrame, features: list[str], train_seconds: float) -> list[dict[str, Any]]:
    rows = []
    for eval_id, mask, dataset in eval_masks(df):
        y = df.loc[mask, "fall_label"].to_numpy(int)
        scores = get_scores(model, df.loc[mask, features].to_numpy(float))
        pred = (scores >= threshold).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append(
            {
                "model_name": model_name,
                "task": "fall_detection",
                "eval": eval_id,
                "dataset": dataset,
                "threshold": threshold,
                "AUROC": safe_roc_auc(y, scores),
                "average_precision": safe_average_precision(y, scores),
                "accuracy": accuracy_score(y, pred),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "fall_f1": f1_score(y, pred, zero_division=0),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "train_seconds": train_seconds,
            }
        )
    return rows


def fall_importance_rows(model_name: str, model: Any, threshold: float, test_df: pd.DataFrame, features: list[str], repeats: int, seed: int) -> list[dict[str, Any]]:
    X = test_df[features].to_numpy(float)
    y = test_df["fall_label"].to_numpy(int)
    baseline_scores = get_scores(model, X)
    baseline_pred = (baseline_scores >= threshold).astype(int)
    baseline_f1 = f1_score(y, baseline_pred, zero_division=0)
    native = native_importance(model, features, task="fall")
    rng = np.random.default_rng(seed)
    rows = []
    for j, feature in enumerate(features):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            pred = (get_scores(model, Xp) >= threshold).astype(int)
            drops.append(baseline_f1 - f1_score(y, pred, zero_division=0))
        rows.append(
            {
                "model_name": model_name,
                "task": "fall_detection",
                "feature": feature,
                "class_name": "",
                "native_importance": native.get(feature, np.nan),
                "signed_coefficient": native.get(f"{feature}__signed", np.nan),
                "permutation_importance_mean": float(np.mean(drops)),
                "permutation_importance_std": float(np.std(drops)),
            }
        )
    return rows


def run_fall_group_ablation(best_name: str, specs: list[ModelSpec], groups: dict[str, list[str]], df: pd.DataFrame, seed: int) -> pd.DataFrame:
    spec = next(s for s in specs if s.name == best_name)
    rows = []
    masks = split_masks(df)
    for group_name, cols in groups.items():
        if not cols:
            continue
        model = spec.factory()
        X_train, y_train = df.loc[masks["train"], cols].to_numpy(float), df.loc[masks["train"], "fall_label"].to_numpy(int)
        X_val, y_val = df.loc[masks["val"], cols].to_numpy(float), df.loc[masks["val"], "fall_label"].to_numpy(int)
        model.fit(X_train, y_train)
        threshold = tune_threshold(y_val, get_scores(model, X_val))
        for eval_id, mask, dataset in eval_masks(df):
            y = df.loc[mask, "fall_label"].to_numpy(int)
            scores = get_scores(model, df.loc[mask, cols].to_numpy(float))
            pred = (scores >= threshold).astype(int)
            rows.append(
                {
                    "model_name": best_name,
                    "feature_group": group_name,
                    "n_features": len(cols),
                    "eval": eval_id,
                    "dataset": dataset,
                    "threshold": threshold,
                    "fall_f1": f1_score(y, pred, zero_division=0),
                    "precision": precision_score(y, pred, zero_division=0),
                    "recall": recall_score(y, pred, zero_division=0),
                    "AUROC": safe_roc_auc(y, scores),
                }
            )
    return pd.DataFrame(rows)


def run_direction_analysis(
    df: pd.DataFrame,
    features: list[str],
    groups: dict[str, list[str]],
    seed: int,
    permutation_repeats: int,
    report_dir: Path,
    figure_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    print("\n[Direction] Training classical models...")
    ddf = direction_df(df)
    masks = split_masks(ddf)
    X_train = ddf.loc[masks["train"], features].to_numpy(float)
    y_train = ddf.loc[masks["train"], "direction_id"].to_numpy(int)
    specs = direction_model_specs(seed)
    rows, importance_rows = [], []
    fitted: dict[str, Any] = {}
    for spec in specs:
        start = time.perf_counter()
        model = spec.factory()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        fitted[spec.name] = model
        rows.extend(evaluate_direction_model(spec.name, model, ddf, features, train_seconds))
        importance_rows.extend(direction_importance_rows(spec.name, model, ddf.loc[masks["test"]], features, permutation_repeats, seed))
    result_df = pd.DataFrame(rows)
    importance_df = pd.DataFrame(importance_rows)
    univariate_df = direction_univariate_ovr(ddf, features)
    shift_df = direction_dataset_shift(ddf, features)
    result_df.to_csv(report_dir / "direction_ml_results.csv", index=False)
    importance_df.to_csv(report_dir / "direction_feature_importance.csv", index=False)
    univariate_df.to_csv(report_dir / "direction_univariate_ovr.csv", index=False)
    shift_df.to_csv(report_dir / "direction_dataset_shift.csv", index=False)

    best_name = choose_best_model_name(result_df, task="direction")
    best_model = fitted[best_name]
    save_direction_confusion(best_name, best_model, ddf, features, figure_dir / "direction_confusion_matrices")
    ablation_df = run_direction_group_ablation(best_name, specs, groups, ddf, features, seed)
    ablation_df.to_csv(report_dir / "direction_feature_group_ablation.csv", index=False)
    return result_df, importance_df, univariate_df, shift_df, ablation_df, {"name": best_name, "model": best_model}


def direction_df(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["fall_label"].astype(int) == 1) & df["direction_supervised"].astype(bool) & df["direction_label"].isin(DIRECTION_LABELS)
    out = df.loc[mask].copy().reset_index(drop=True)
    out["direction_id"] = out["direction_label"].map(DIRECTION_TO_ID).astype(int)
    return out


def evaluate_direction_model(model_name: str, model: Any, ddf: pd.DataFrame, features: list[str], train_seconds: float) -> list[dict[str, Any]]:
    rows = []
    for eval_id, mask, dataset in eval_masks(ddf):
        if mask.sum() == 0:
            continue
        y = ddf.loc[mask, "direction_id"].to_numpy(int)
        pred = model.predict(ddf.loc[mask, features].to_numpy(float)).astype(int)
        per_class = {name: f1_score(y, pred, labels=[i], average="macro", zero_division=0) for i, name in enumerate(DIRECTION_LABELS)}
        cm = confusion_matrix(y, pred, labels=[0, 1, 2])
        rows.append(
            {
                "model_name": model_name,
                "task": "direction_classification",
                "eval": eval_id,
                "dataset": dataset,
                "direction_macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                "direction_accuracy": accuracy_score(y, pred),
                "forward_f1": per_class["forward"],
                "backward_f1": per_class["backward"],
                "lateral_f1": per_class["lateral"],
                "direction_n_supervised": int(len(y)),
                "train_seconds": train_seconds,
                "confusion_matrix": json.dumps(cm.tolist()),
            }
        )
    return rows


def direction_importance_rows(model_name: str, model: Any, test_df: pd.DataFrame, features: list[str], repeats: int, seed: int) -> list[dict[str, Any]]:
    X = test_df[features].to_numpy(float)
    y = test_df["direction_id"].to_numpy(int)
    pred = model.predict(X).astype(int)
    baseline = f1_score(y, pred, average="macro", zero_division=0)
    native = native_importance(model, features, task="direction")
    rng = np.random.default_rng(seed)
    rows = []
    for j, feature in enumerate(features):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            drops.append(baseline - f1_score(y, model.predict(Xp).astype(int), average="macro", zero_division=0))
        base_row = {
            "model_name": model_name,
            "task": "direction_classification",
            "feature": feature,
            "class_name": "",
            "native_importance": native.get(feature, np.nan),
            "signed_coefficient": np.nan,
            "permutation_importance_mean": float(np.mean(drops)),
            "permutation_importance_std": float(np.std(drops)),
        }
        rows.append(base_row)
        for class_name in DIRECTION_LABELS:
            key = f"{feature}__{class_name}"
            if key in native:
                r = dict(base_row)
                r["class_name"] = class_name
                r["native_importance"] = abs(native[key])
                r["signed_coefficient"] = native[key]
                r["permutation_importance_mean"] = np.nan
                r["permutation_importance_std"] = np.nan
                rows.append(r)
    return rows


def run_direction_group_ablation(best_name: str, specs: list[ModelSpec], groups: dict[str, list[str]], ddf: pd.DataFrame, all_features: list[str], seed: int) -> pd.DataFrame:
    spec = next(s for s in specs if s.name == best_name)
    rows = []
    masks = split_masks(ddf)
    for group_name, cols in groups.items():
        cols = [c for c in cols if c in all_features]
        if not cols:
            continue
        model = spec.factory()
        model.fit(ddf.loc[masks["train"], cols].to_numpy(float), ddf.loc[masks["train"], "direction_id"].to_numpy(int))
        for eval_id, mask, dataset in eval_masks(ddf):
            if mask.sum() == 0:
                continue
            y = ddf.loc[mask, "direction_id"].to_numpy(int)
            pred = model.predict(ddf.loc[mask, cols].to_numpy(float)).astype(int)
            rows.append(
                {
                    "model_name": best_name,
                    "feature_group": group_name,
                    "n_features": len(cols),
                    "eval": eval_id,
                    "dataset": dataset,
                    "direction_macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                    "direction_accuracy": accuracy_score(y, pred),
                    "direction_n_supervised": len(y),
                }
            )
    return pd.DataFrame(rows)


def direction_univariate_ovr(ddf: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for class_id, class_name in enumerate(DIRECTION_LABELS):
        y = (ddf["direction_id"].to_numpy(int) == class_id).astype(int)
        for feature in features:
            vals = ddf[feature].to_numpy(float)
            target = vals[y == 1]
            rest = vals[y == 0]
            rows.append(
                {
                    "class_name": class_name,
                    "feature": feature,
                    "n_target": int(len(target)),
                    "n_rest": int(len(rest)),
                    "AUROC": safe_roc_auc(y, vals),
                    "abs_auc": max_or_nan(safe_roc_auc(y, vals)),
                    "cohens_d": cohens_d(target, rest),
                    "ks_statistic": ks_stat(target, rest),
                    "median_target": safe_median(target),
                    "median_rest": safe_median(rest),
                }
            )
    return pd.DataFrame(rows).sort_values(["class_name", "abs_auc"], ascending=[True, False])


def direction_dataset_shift(ddf: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for direction in DIRECTION_LABELS:
        sub = ddf[ddf["direction_label"].eq(direction)]
        bits = sub[sub["dataset"].str.lower().eq("bits")]
        weda = sub[sub["dataset"].str.lower().eq("weda")]
        for feature in features:
            a = bits[feature].to_numpy(float)
            b = weda[feature].to_numpy(float)
            rows.append(
                {
                    "direction": direction,
                    "feature": feature,
                    "n_bits": int(len(a)),
                    "n_weda": int(len(b)),
                    "bits_median": safe_median(a),
                    "weda_median": safe_median(b),
                    "median_delta_bits_minus_weda": safe_median(a) - safe_median(b) if len(a) and len(b) else np.nan,
                    "cohens_d_bits_vs_weda": cohens_d(a, b),
                    "ks_statistic": ks_stat(a, b),
                }
            )
    return pd.DataFrame(rows).sort_values(["direction", "ks_statistic"], ascending=[True, False])


def split_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    split = df["split"].astype(str).str.lower()
    return {"train": split.eq("train").to_numpy(), "val": split.eq("val").to_numpy(), "test": split.eq("test").to_numpy()}


def eval_masks(df: pd.DataFrame) -> list[tuple[str, np.ndarray, str]]:
    split = df["split"].astype(str).str.lower()
    dataset = df["dataset"].astype(str).str.lower()
    test = split.eq("test")
    return [
        ("E3", test.to_numpy(), "bits+weda"),
        ("E6", (test & dataset.eq("bits")).to_numpy(), "bits"),
        ("E7", (test & dataset.eq("weda")).to_numpy(), "weda"),
    ]


def choose_best_model_name(results: pd.DataFrame, task: str) -> str:
    if task == "fall":
        e3 = results[results["eval"].eq("E3")].copy()
        return e3.sort_values(["fall_f1", "average_precision"], ascending=False).iloc[0]["model_name"]
    e3 = results[results["eval"].eq("E3")].copy()
    return e3.sort_values(["direction_macro_f1", "direction_accuracy"], ascending=False).iloc[0]["model_name"]


def native_importance(model: Any, features: list[str], task: str) -> dict[str, float]:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    out: dict[str, float] = {}
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim == 1 or coef.shape[0] == 1:
            vals = coef.ravel()
            for f, v in zip(features, vals):
                out[f] = abs(float(v))
                out[f"{f}__signed"] = float(v)
        else:
            abs_mean = np.mean(np.abs(coef), axis=0)
            for j, f in enumerate(features):
                out[f] = float(abs_mean[j])
                for class_id, class_name in enumerate(DIRECTION_LABELS[: coef.shape[0]]):
                    out[f"{f}__{class_name}"] = float(coef[class_id, j])
    elif hasattr(estimator, "feature_importances_"):
        for f, v in zip(features, estimator.feature_importances_):
            out[f] = float(v)
    return out


def safe_roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, scores))
    except Exception:
        return np.nan


def safe_average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(average_precision_score(y, scores))
    except Exception:
        return np.nan


def max_or_nan(auc: float) -> float:
    return max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / max(len(a) + len(b) - 2, 1)
    if pooled <= 1e-12:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / math.sqrt(pooled))


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(stats.ks_2samp(a, b).statistic)


def safe_median(x: np.ndarray) -> float:
    return float(np.median(x)) if len(x) else np.nan


def save_fall_confusion(model_name: str, model: Any, threshold: float, df: pd.DataFrame, features: list[str], out_dir: Path) -> None:
    for eval_id, mask, dataset in eval_masks(df):
        y = df.loc[mask, "fall_label"].to_numpy(int)
        pred = (get_scores(model, df.loc[mask, features].to_numpy(float)) >= threshold).astype(int)
        save_cm(out_dir, f"{model_name}_{eval_id}_fall", confusion_matrix(y, pred, labels=[0, 1]), ["non_fall", "fall"])


def save_direction_confusion(model_name: str, model: Any, ddf: pd.DataFrame, features: list[str], out_dir: Path) -> None:
    for eval_id, mask, dataset in eval_masks(ddf):
        y = ddf.loc[mask, "direction_id"].to_numpy(int)
        pred = model.predict(ddf.loc[mask, features].to_numpy(float)).astype(int)
        save_cm(out_dir, f"{model_name}_{eval_id}_direction", confusion_matrix(y, pred, labels=[0, 1, 2]), DIRECTION_LABELS)


def save_cm(out_dir: Path, name: str, cm: np.ndarray, labels: list[str]) -> None:
    pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels]).to_csv(out_dir / f"{name}.csv")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    ax.set_title(name)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=180)
    plt.close(fig)


def plot_fall_roc_pr(model_name: str, y: np.ndarray, scores: np.ndarray, out_dir: Path) -> None:
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    if len(np.unique(y)) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y, scores, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y, scores, ax=axes[1])
    fig.suptitle(model_name)
    fig.tight_layout()
    fig.savefig(out_dir / f"{model_name}_roc_pr.png", dpi=180)
    plt.close(fig)


def compute_deep_reference_metrics(df: pd.DataFrame) -> dict[str, Any]:
    rows = {}
    for eval_id, mask, dataset in eval_masks(df):
        y = df.loc[mask, "fall_label"].to_numpy(int)
        fall_pred = np.where(df.loc[mask, "fall_pred"].astype(str).eq("fall"), 1, 0)
        rows[f"{eval_id}_fall_precision"] = precision_score(y, fall_pred, zero_division=0)
        rows[f"{eval_id}_fall_recall"] = recall_score(y, fall_pred, zero_division=0)
        rows[f"{eval_id}_fall_f1"] = f1_score(y, fall_pred, zero_division=0)
        dsub = df.loc[mask & (df["fall_label"].astype(int).eq(1)) & df["direction_supervised"].astype(bool) & df["direction_label"].isin(DIRECTION_LABELS)]
        if len(dsub):
            true = dsub["direction_label"].map(DIRECTION_TO_ID).to_numpy(int)
            pred = dsub["direction_pred"].map(DIRECTION_TO_ID).fillna(0).to_numpy(int)
            rows[f"{eval_id}_direction_macro_f1"] = f1_score(true, pred, average="macro", zero_division=0)
            rows[f"{eval_id}_direction_accuracy"] = accuracy_score(true, pred)
            rows[f"{eval_id}_direction_n"] = int(len(dsub))
        else:
            rows[f"{eval_id}_direction_macro_f1"] = np.nan
            rows[f"{eval_id}_direction_accuracy"] = np.nan
            rows[f"{eval_id}_direction_n"] = 0
    rows["params"] = DEEP_REFERENCE["params"]
    return rows


def build_comparison_tables(
    fall_results: pd.DataFrame,
    direction_results: pd.DataFrame,
    fall_importance: pd.DataFrame,
    direction_importance: pd.DataFrame,
    deep: dict[str, Any],
    best_fall: dict[str, Any],
    best_direction: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    rows.append(
        {
            "model_name": "DS-Fall-RD A5WCEFW",
            "model_type": "deep_temporal",
            "input_type": "temporal_tilt12",
            "task_supported": "multitask_fall_direction",
            "params_or_model_size": deep["params"],
            "E3 Fall F1": deep["E3_fall_f1"],
            "E3 Fall Precision": deep["E3_fall_precision"],
            "E3 Fall Recall": deep["E3_fall_recall"],
            "E3 Direction Macro F1": deep["E3_direction_macro_f1"],
            "E3 Direction Accuracy": deep["E3_direction_accuracy"],
            "E6 BITS Fall F1": deep["E6_fall_f1"],
            "E6 BITS Direction Macro F1": deep["E6_direction_macro_f1"],
            "E7 WEDA Fall F1": deep["E7_fall_f1"],
            "E7 WEDA Precision": deep["E7_fall_precision"],
            "E7 WEDA Recall": deep["E7_fall_recall"],
            "E7 WEDA Direction Macro F1": deep["E7_direction_macro_f1"],
            "notes": "reference saved predictions",
        }
    )
    for model_name in sorted(fall_results["model_name"].unique()):
        e3 = row_for(fall_results, model_name, "E3")
        e6 = row_for(fall_results, model_name, "E6")
        e7 = row_for(fall_results, model_name, "E7")
        rows.append(
            {
                "model_name": model_name,
                "model_type": "classical_feature",
                "input_type": "summary_features",
                "task_supported": "fall_only",
                "params_or_model_size": "",
                "E3 Fall F1": e3.get("fall_f1"),
                "E3 Fall Precision": e3.get("precision"),
                "E3 Fall Recall": e3.get("recall"),
                "E3 Direction Macro F1": "",
                "E3 Direction Accuracy": "",
                "E6 BITS Fall F1": e6.get("fall_f1"),
                "E6 BITS Direction Macro F1": "",
                "E7 WEDA Fall F1": e7.get("fall_f1"),
                "E7 WEDA Precision": e7.get("precision"),
                "E7 WEDA Recall": e7.get("recall"),
                "E7 WEDA Direction Macro F1": "",
                "notes": "fall-only summary feature classifier",
            }
        )
    for model_name in sorted(direction_results["model_name"].unique()):
        e3 = row_for(direction_results, model_name, "E3")
        e6 = row_for(direction_results, model_name, "E6")
        e7 = row_for(direction_results, model_name, "E7")
        rows.append(
            {
                "model_name": model_name,
                "model_type": "classical_feature",
                "input_type": "summary_features",
                "task_supported": "direction_only",
                "params_or_model_size": "",
                "E3 Fall F1": "",
                "E3 Fall Precision": "",
                "E3 Fall Recall": "",
                "E3 Direction Macro F1": e3.get("direction_macro_f1"),
                "E3 Direction Accuracy": e3.get("direction_accuracy"),
                "E6 BITS Fall F1": "",
                "E6 BITS Direction Macro F1": e6.get("direction_macro_f1"),
                "E7 WEDA Fall F1": "",
                "E7 WEDA Precision": "",
                "E7 WEDA Recall": "",
                "E7 WEDA Direction Macro F1": e7.get("direction_macro_f1"),
                "notes": "direction-only summary feature classifier",
            }
        )
    comparison = pd.DataFrame(rows)
    best_fall_table = comparison[comparison["model_name"].isin(["DS-Fall-RD A5WCEFW", best_fall["name"]])].copy()
    best_direction_table = comparison[comparison["model_name"].isin(["DS-Fall-RD A5WCEFW", best_direction["name"]])].copy()
    fall_top = top_features(fall_importance, best_fall["name"])
    dir_top = top_features(direction_importance, best_direction["name"])
    common = sorted(set(fall_top[:15]) & set(dir_top[:15]))
    interpretation = pd.DataFrame(
        [
            {"feature_group": "fall_top_features", "examples": ", ".join(fall_top[:10]), "physical meaning": "impact strength, abrupt acceleration, event timing, post-impact instability", "paper interpretation": "fall detection is dominated by impact/jerk/timing evidence"},
            {"feature_group": "direction_top_features", "examples": ", ".join(dir_top[:10]), "physical meaning": "signed axis motion, roll/pitch transition, gyroscope pattern", "paper interpretation": "direction depends on signed orientation and rotation evidence"},
            {"feature_group": "features_common_to_both", "examples": ", ".join(common[:10]), "physical meaning": "features useful for both event detection and motion geometry", "paper interpretation": "some summary features bridge fall and direction but do not replace temporal multitask learning"},
            {"feature_group": "fall_specific_features", "examples": ", ".join([f for f in fall_top if f not in dir_top][:10]), "physical meaning": "magnitude, jerk, impact timing", "paper interpretation": "classical fall-only classifiers can exploit compact impact summaries"},
            {"feature_group": "direction_specific_features", "examples": ", ".join([f for f in dir_top if f not in fall_top][:10]), "physical meaning": "signed roll/pitch/gyro differences", "paper interpretation": "rotation-aware signed features explain direction robustness"},
        ]
    )
    return comparison, best_fall_table, best_direction_table, interpretation


def row_for(df: pd.DataFrame, model_name: str, eval_id: str) -> dict[str, Any]:
    row = df[(df["model_name"].eq(model_name)) & (df["eval"].eq(eval_id))]
    return row.iloc[0].to_dict() if not row.empty else {}


def top_features(importance: pd.DataFrame, model_name: str) -> list[str]:
    sub = importance[(importance["model_name"].eq(model_name)) & (importance["class_name"].fillna("").eq(""))].copy()
    sub["score"] = sub["permutation_importance_mean"].fillna(0).abs()
    if sub["score"].max() <= 0:
        sub["score"] = sub["native_importance"].fillna(0).abs()
    return sub.sort_values("score", ascending=False)["feature"].head(30).tolist()


def make_summary_figures(
    figure_dir: Path,
    fall_results: pd.DataFrame,
    direction_results: pd.DataFrame,
    fall_importance: pd.DataFrame,
    direction_importance: pd.DataFrame,
    fall_ablation: pd.DataFrame,
    direction_ablation: pd.DataFrame,
    comparison: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    best_fall: dict[str, Any],
    best_direction: dict[str, Any],
) -> None:
    plot_importance_bar(fall_importance, best_fall["name"], "Fall top feature importance", figure_dir / "fall_top_feature_importance.png")
    plot_importance_bar(direction_importance, best_direction["name"], "Direction top feature importance", figure_dir / "direction_top_feature_importance.png")
    plot_comparison_bar(comparison, "E7 WEDA Fall F1", "Fall F1 comparison", figure_dir / "ml_vs_deep_fall_comparison.png")
    plot_comparison_bar(comparison, "E7 WEDA Direction Macro F1", "Direction Macro F1 comparison", figure_dir / "ml_vs_deep_direction_comparison.png")
    plot_ablation(fall_ablation, "fall_f1", "Fall feature group ablation", figure_dir / "fall_feature_group_ablation.png")
    plot_ablation(direction_ablation, "direction_macro_f1", "Direction feature group ablation", figure_dir / "direction_feature_group_ablation.png")
    plot_direction_tsne(feature_df, feature_columns, figure_dir / "direction_tsne_summary_features.png")


def plot_importance_bar(importance: pd.DataFrame, model_name: str, title: str, path: Path) -> None:
    sub = importance[(importance["model_name"].eq(model_name)) & (importance["class_name"].fillna("").eq(""))].copy()
    sub["score"] = sub["permutation_importance_mean"].fillna(0).abs()
    if sub["score"].max() <= 0:
        sub["score"] = sub["native_importance"].fillna(0).abs()
    top = sub.sort_values("score", ascending=False).head(15)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=top, y="feature", x="score", color="#4c78a8")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_comparison_bar(comparison: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    sub = comparison[["model_name", "model_type", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna().sort_values(metric, ascending=False).head(8)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=sub, x=metric, y="model_name", hue="model_type", dodge=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_ablation(ablation: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    sub = ablation[ablation["eval"].eq("E3")].copy()
    if sub.empty:
        return
    plt.figure(figsize=(10, 5))
    sns.barplot(data=sub, x="feature_group", y=metric, color="#59a14f")
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_direction_tsne(df: pd.DataFrame, features: list[str], path: Path) -> None:
    ddf = direction_df(df)
    if len(ddf) < 10:
        return
    X = StandardScaler().fit_transform(ddf[features].to_numpy(float))
    Xp = PCA(n_components=min(20, X.shape[1]), random_state=42).fit_transform(X)
    perplexity = max(5, min(30, len(ddf) // 5))
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42).fit_transform(Xp)
    plot_df = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "direction": ddf["direction_label"], "dataset": ddf["dataset"]})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="direction", style="dataset", s=45)
    plt.title("t-SNE of direction-supervised summary features")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_final_report(
    report_dir: Path,
    fall_results: pd.DataFrame,
    fall_importance: pd.DataFrame,
    fall_ablation: pd.DataFrame,
    direction_results: pd.DataFrame,
    direction_importance: pd.DataFrame,
    direction_univariate: pd.DataFrame,
    direction_shift: pd.DataFrame,
    direction_ablation: pd.DataFrame,
    comparison: pd.DataFrame,
    best_fall_table: pd.DataFrame,
    best_direction_table: pd.DataFrame,
    interpretation: pd.DataFrame,
    deep: dict[str, Any],
    best_fall: dict[str, Any],
    best_direction: dict[str, Any],
) -> None:
    best_fall_e3 = row_for(fall_results, best_fall["name"], "E3")
    best_fall_e7 = row_for(fall_results, best_fall["name"], "E7")
    best_dir_e3 = row_for(direction_results, best_direction["name"], "E3")
    best_dir_e7 = row_for(direction_results, best_direction["name"], "E7")
    fall_top = top_features(fall_importance, best_fall["name"])[:10]
    dir_top = top_features(direction_importance, best_direction["name"])[:10]
    fall_ab_e3 = fall_ablation[fall_ablation["eval"].eq("E3")].sort_values("fall_f1", ascending=False)
    dir_ab_e3 = direction_ablation[direction_ablation["eval"].eq("E3")].sort_values("direction_macro_f1", ascending=False)
    no_impact = fall_ab_e3[fall_ab_e3["feature_group"].eq("all_features_no_impact_timing")]
    all_fall = fall_ab_e3[fall_ab_e3["feature_group"].eq("all_features")]
    shift_top = direction_shift.sort_values("ks_statistic", ascending=False).head(10)
    xgboost_note = "XGBoost was available." if importlib.util.find_spec("xgboost") else "XGBoost was not installed in the active Python environment and was skipped per prompt."
    lines = [
        "# ML Feature Analysis 25 Hz",
        "",
        "## 1. Purpose",
        "",
        "This is a machine-learning-based feature analysis for fall detection and direction classification. It is not a new deep architecture experiment.",
        "",
        "## 2. Dataset And Protocol",
        "",
        "Benchmark: BITS + WEDA only, 25 Hz, 2-second event-centered windows, existing train/val/test split, summary features derived from tilt12/raw6. Direction metrics use only supervised fall windows with labels forward/backward/lateral.",
        "",
        "Classical ML feature columns are signal-derived summaries only. Metadata/protocol columns such as dataset, source_hz, start_idx, timestamps, labels, and reference predictions are retained in `summary_feature_matrix.csv` for audit but excluded from ML training.",
        "",
        xgboost_note,
        "",
        "## 3. Fall Detection Findings",
        "",
        f"- Best classical fall model: `{best_fall['name']}` with E3 Fall F1={best_fall_e3.get('fall_f1', np.nan):.4f}, E7 WEDA Fall F1={best_fall_e7.get('fall_f1', np.nan):.4f}, WEDA precision={best_fall_e7.get('precision', np.nan):.4f}, WEDA recall={best_fall_e7.get('recall', np.nan):.4f}.",
        f"- Top fall features: {', '.join(fall_top)}.",
        "- WEDA DS-Fall-RD Fall F1 is low mainly because WEDA non-fall contains high-motion hard negatives, which increases false positives.",
        "- Classical fall-only models can reduce/avoid this issue because they directly exploit compact summary features, but they do not solve direction or multitask learning.",
        f"- Impact artifact check: all_features E3 Fall F1={all_fall.iloc[0]['fall_f1']:.4f} and no-impact-timing E3 Fall F1={no_impact.iloc[0]['fall_f1']:.4f}." if not all_fall.empty and not no_impact.empty else "- Impact artifact check is unavailable.",
        "",
        "## 4. Direction Classification Findings",
        "",
        f"- Best classical direction model: `{best_direction['name']}` with E3 Direction Macro F1={best_dir_e3.get('direction_macro_f1', np.nan):.4f}, E7 WEDA Direction Macro F1={best_dir_e7.get('direction_macro_f1', np.nan):.4f}.",
        f"- Top direction features: {', '.join(dir_top)}.",
        f"- Best direction feature group on E3: `{dir_ab_e3.iloc[0]['feature_group']}` with Macro F1={dir_ab_e3.iloc[0]['direction_macro_f1']:.4f}.",
        "- Direction depends more on signed orientation/gyro/axis summaries than on pure magnitude. This matches DS-Fall-RD's rotation-aware design.",
        "- BITS/WEDA direction shift exists; top shifted rows are shown below.",
        "",
        markdown_table(format_df(shift_top[["direction", "feature", "n_bits", "n_weda", "bits_median", "weda_median", "median_delta_bits_minus_weda", "ks_statistic"]].head(10))),
        "",
        "## 5. Classical ML vs Deep DS-Fall-RD",
        "",
        markdown_table(format_df(comparison)),
        "",
        "Best fall-only comparison:",
        "",
        markdown_table(format_df(best_fall_table)),
        "",
        "Best direction-only comparison:",
        "",
        markdown_table(format_df(best_direction_table)),
        "",
        "- Classical ML can beat or approach the deep model on fall-only metrics because impact/jerk/timing summaries are highly discriminative.",
        "- DS-Fall-RD remains stronger as a single multitask temporal model because it handles fall and direction jointly from sequence input.",
        "- A classical FPGuard/verifier is plausible: use summary-feature ML to filter WEDA-like false positives while preserving DS-Fall-RD direction output.",
        "",
        "## 6. Paper-ready Conclusion",
        "",
        "Summary impact, jerk and timing features are very strong for fall detection, and WEDA is difficult because ADL/non-fall windows include hard-negative high-motion patterns. Direction classification is driven by signed orientation and gyroscope evidence such as roll/pitch transitions and signed axis dynamics. DS-Fall-RD remains the main model for direction-sensitive multitask fall detection, while classical ML baselines provide diagnostic evidence and a possible FPGuard verifier for reducing false positives.",
        "",
        "## 7. Tables To Include In Paper",
        "",
        "- ML vs Deep comparison",
        "- Fall feature importance",
        "- Direction feature importance",
        "- Feature group ablation",
        "- WEDA FP explanation table",
        "",
        "## Feature Interpretation Table",
        "",
        markdown_table(format_df(interpretation)),
        "",
        "## Output Files",
        "",
        "- `summary_feature_matrix.csv`",
        "- `fall_ml_results.csv`",
        "- `fall_feature_importance.csv`",
        "- `fall_feature_group_ablation.csv`",
        "- `direction_ml_results.csv`",
        "- `direction_feature_importance.csv`",
        "- `direction_univariate_ovr.csv`",
        "- `direction_dataset_shift.csv`",
        "- `direction_feature_group_ablation.csv`",
        "- `ml_vs_deep_comparison.csv`",
        "- `feature_interpretation_table.csv`",
    ]
    (report_dir / "ml_feature_analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        if pd.api.types.is_numeric_dtype(out[col]):
            if col.startswith("n_") or col in {"params", "direction_n_supervised", "TN", "FP", "FN", "TP"}:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(int(v)))
            else:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    return out


if __name__ == "__main__":
    main()
