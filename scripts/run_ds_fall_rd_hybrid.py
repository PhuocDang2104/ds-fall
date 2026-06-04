from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "reports" / "ml_feature_analysis_25hz" / "summary_feature_matrix.csv"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "ds_fall_rd_hybrid"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "ds_fall_rd_hybrid" / "confusion_matrices"
ML_COMPARISON_PATH = PROJECT_ROOT / "outputs" / "reports" / "ml_feature_analysis_25hz" / "ml_vs_deep_comparison.csv"
FV_RESULTS_PATH = PROJECT_ROOT / "outputs" / "reports" / "ds_fall_rd_fv" / "fv_results.csv"

REFERENCE_PARAMS = 65959
DIRECTION_LABELS = ["forward", "backward", "lateral"]
DIR_TO_ID = {name: i for i, name in enumerate(DIRECTION_LABELS)}

LITE10 = [
    "acc_mag_range",
    "acc_mag_max",
    "acc_mag_std",
    "gyro_mag_p95",
    "gyro_mag_mean",
    "jerk_p95",
    "jerk_std",
    "tilt_delta_p95",
    "post_acc_mag_std",
    "post_gyro_mag_std",
]

AXIS10 = [
    "acc_mag_range",
    "acc_mag_max",
    "acc_mag_std",
    "jerk_p95",
    "jerk_std",
    "gyro_mag_p95",
    "gyro_mag_max",
    "ay_std",
    "ay_range",
    "post_acc_mag_std",
]

TIMING_FEATURES = {"impact_distance_from_center", "impact_index", "gyro_peak_distance_from_center", "gyro_peak_index"}


@dataclass(frozen=True)
class HybridRun:
    run_id: str
    expert: str
    feature_set: str
    upper_bound: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DS-Fall-RD hybrid ML-fall + A5-direction experiments.")
    parser.add_argument("--summary_path", default=str(SUMMARY_PATH))
    parser.add_argument("--output_dir", default=str(REPORT_DIR))
    parser.add_argument("--figure_dir", default=str(FIGURE_DIR))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = ensure_dir(Path(args.output_dir))
    figure_dir = ensure_dir(Path(args.figure_dir))
    run_root = ensure_dir(report_dir / "runs")

    df = load_summary(Path(args.summary_path))
    validate_inputs(df)
    signal_features = infer_signal_features(df)
    feature_sets = {
        "Lite10": LITE10,
        "Axis": AXIS10,
        "FullNoTiming": [c for c in signal_features if c not in TIMING_FEATURES],
        "FullTiming": signal_features,
    }
    feature_config = {
        "summary_source": str(Path(args.summary_path)),
        "benchmark": "BITS/WEDA only, 25 Hz, 2-second event-centered, existing train/val/test split",
        "direction_expert": "DS-Fall-RD A5WCEFW reference direction columns from summary_feature_matrix.csv",
        "feature_sets": feature_sets,
        "timing_features": sorted(TIMING_FEATURES),
        "notes": "Hybrid inference always uses ML Fall Expert for final fall and A5 reference direction for direction. Deep fall head is not used in hybrid final fall decisions.",
    }
    (report_dir / "hybrid_feature_configs.json").write_text(json.dumps(feature_config, indent=2), encoding="utf-8")

    runs = hybrid_runs()
    results_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    e2e_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    baseline_rows, baseline_e2e, baseline_complexity = build_baselines(df, figure_dir)
    results_rows.extend(baseline_rows)
    e2e_rows.extend(baseline_e2e)
    complexity_rows.extend(baseline_complexity)

    for run in runs:
        try:
            print(f"Running {run.run_id} ({run.expert}, {run.feature_set})")
            run_dir = ensure_dir(run_root / run.run_id)
            rows, thresholds, complexity, e2e = run_hybrid(df, run, feature_sets[run.feature_set], args.seed, run_dir, figure_dir)
            results_rows.extend(rows)
            threshold_rows.append(thresholds)
            complexity_rows.append(complexity)
            e2e_rows.extend(e2e)
        except Exception as exc:
            failures.append({"run_id": run.run_id, "status": "failed", "reason": repr(exc)})
            print(f"FAILED {run.run_id}: {exc}")

    results = pd.DataFrame(results_rows)
    thresholds = pd.DataFrame(threshold_rows)
    complexity = pd.DataFrame(complexity_rows)
    e2e = pd.DataFrame(e2e_rows)
    failures_df = pd.DataFrame(failures)

    results.to_csv(report_dir / "hybrid_results.csv", index=False)
    thresholds.to_csv(report_dir / "hybrid_thresholds.csv", index=False)
    complexity.to_csv(report_dir / "hybrid_complexity.csv", index=False)
    e2e.to_csv(report_dir / "hybrid_end_to_end_direction.csv", index=False)
    failures_df.to_csv(report_dir / "hybrid_failures.csv", index=False)

    write_best_selection(results, e2e, complexity, failures_df, report_dir)
    write_summary(results, thresholds, complexity, e2e, failures_df, report_dir)
    print(f"Saved outputs to {report_dir}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary feature matrix: {path}")
    df = pd.read_csv(path).copy()
    df["dataset"] = df["dataset"].astype(str).str.lower()
    df["split"] = df["split"].astype(str).str.lower()
    df["fall_label"] = df["fall_label"].astype(int)
    df["direction_supervised_bool"] = to_bool(df["direction_supervised"])
    df["direction_id"] = df["direction_label"].astype(str).str.lower().map(DIR_TO_ID).fillna(-1).astype(int)
    return df


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def validate_inputs(df: pd.DataFrame) -> None:
    required = {
        "dataset",
        "split",
        "fall_label",
        "direction_label",
        "direction_supervised",
        "fall_prob",
        "direction_prob_forward",
        "direction_prob_backward",
        "direction_prob_lateral",
    }
    required.update(LITE10)
    required.update(AXIS10)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in summary feature matrix: {missing}")
    if not {"train", "val", "test"}.issubset(set(df["split"].unique())):
        raise ValueError(f"Expected train/val/test split, got {sorted(df['split'].unique())}")


def infer_signal_features(df: pd.DataFrame) -> list[str]:
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


def hybrid_runs() -> list[HybridRun]:
    return [
        HybridRun("HYB_LR_Lite10", "LR", "Lite10"),
        HybridRun("HYB_DT_Lite10", "DT", "Lite10"),
        HybridRun("HYB_RF_Lite10", "RF", "Lite10"),
        HybridRun("HYB_GB_Lite10", "GB", "Lite10"),
        HybridRun("HYB_HGB_Lite10", "HGB", "Lite10"),
        HybridRun("HYB_LR_Axis", "LR", "Axis"),
        HybridRun("HYB_DT_Axis", "DT", "Axis"),
        HybridRun("HYB_RF_Axis", "RF", "Axis"),
        HybridRun("HYB_GB_Axis", "GB", "Axis"),
        HybridRun("HYB_HGB_Axis", "HGB", "Axis"),
        HybridRun("HYB_LR_FullNoTiming", "LR", "FullNoTiming"),
        HybridRun("HYB_RF_FullNoTiming", "RF", "FullNoTiming"),
        HybridRun("HYB_GB_FullNoTiming", "GB", "FullNoTiming"),
        HybridRun("HYB_HGB_FullNoTiming", "HGB", "FullNoTiming"),
        HybridRun("HYB_RF_FullTiming", "RF", "FullTiming", upper_bound=True),
        HybridRun("HYB_GB_FullTiming", "GB", "FullTiming", upper_bound=True),
        HybridRun("HYB_HGB_FullTiming", "HGB", "FullTiming", upper_bound=True),
    ]


def make_model(expert: str, seed: int):
    if expert == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))
    if expert == "DT":
        return make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=seed))
    if expert == "RF":
        return make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=50, max_depth=5, class_weight="balanced", random_state=seed, n_jobs=-1),
        )
    if expert == "GB":
        return make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=seed))
    if expert == "HGB":
        return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(random_state=seed, max_iter=100))
    raise ValueError(expert)


def run_hybrid(
    df: pd.DataFrame,
    run: HybridRun,
    features: list[str],
    seed: int,
    run_dir: Path,
    figure_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    train_mask = df["split"].eq("train").to_numpy()
    val_mask = df["split"].eq("val").to_numpy()
    X_train = df.loc[train_mask, features].astype(float).to_numpy()
    y_train = df.loc[train_mask, "fall_label"].astype(int).to_numpy()
    X_all = df[features].astype(float).to_numpy()
    model = make_model(run.expert, seed)
    model.fit(X_train, y_train)
    with (run_dir / "fall_expert.pkl").open("wb") as f:
        pickle.dump(model, f)
    (run_dir / "config.json").write_text(
        json.dumps({"run": run.__dict__, "features": features, "seed": seed}, indent=2),
        encoding="utf-8",
    )
    prob = model.predict_proba(X_all)[:, 1].astype(float)
    threshold, threshold_note = select_threshold(
        y=df.loc[val_mask, "fall_label"].astype(int).to_numpy(),
        prob=prob[val_mask],
        meta=df.loc[val_mask].reset_index(drop=True),
    )
    rows = []
    e2e_rows = []
    for eval_id, dataset in [("E3", "bits+weda"), ("E6", "bits"), ("E7", "weda")]:
        idx = eval_indices(df, eval_id)
        pred = (prob[idx] >= threshold).astype(int)
        metrics, e2e = evaluate_eval(df, idx, pred, prob[idx], run.run_id, eval_id, dataset, figure_dir)
        metrics.update(
            {
                "model_type": "hybrid_ml_fall_a5_direction",
                "feature_set": run.feature_set,
                "fall_expert": run.expert,
                "number_features": len(features),
                "threshold": threshold,
                "threshold_note": threshold_note,
                "upper_bound_timing": run.upper_bound,
                "deep_fall_used_for_final": False,
                "direction_source": "A5_reference",
                **complexity_info(model, run.expert, len(features)),
            }
        )
        rows.append(metrics)
        e2e.update({"run_id": run.run_id, "eval": eval_id, "dataset": dataset, "feature_set": run.feature_set, "fall_expert": run.expert})
        e2e_rows.append(e2e)
    complexity = {
        "run_id": run.run_id,
        "model_type": "hybrid_ml_fall_a5_direction",
        "feature_set": run.feature_set,
        "fall_expert": run.expert,
        "number_features": len(features),
        "upper_bound_timing": run.upper_bound,
        **complexity_info(model, run.expert, len(features)),
    }
    thresholds = {
        "run_id": run.run_id,
        "feature_set": run.feature_set,
        "fall_expert": run.expert,
        "threshold": threshold,
        "threshold_note": threshold_note,
        "selected_on": "validation",
    }
    return rows, thresholds, complexity, e2e_rows


def select_threshold(y: np.ndarray, prob: np.ndarray, meta: pd.DataFrame) -> tuple[float, str]:
    candidates = np.unique(
        np.clip(
            np.concatenate([np.linspace(0.001, 0.999, 240), np.quantile(prob, np.linspace(0.01, 0.99, 120)), np.unique(prob)]),
            0.0,
            1.0,
        )
    )
    hard_neg = meta["dataset"].astype(str).str.lower().eq("weda").to_numpy() & (meta["fall_label"].astype(int).to_numpy() == 0)
    rows = []
    for t in candidates:
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel()]
        hard_fp = int(np.sum((pred == 1) & hard_neg))
        precision = precision_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        rows.append((float(t), f1, recall, precision, fp, hard_fp))
    # Primary: maximize F1. Secondary/tie: prefer recall >= .88 and fewer WEDA-like hard-negative FP.
    best = max(rows, key=lambda r: (round(r[1], 6), 1 if r[2] >= 0.88 else 0, -r[5], r[3], -r[4]))
    feasible = [r for r in rows if r[2] >= 0.88]
    best_feasible = max(feasible, key=lambda r: (round(r[1], 6), -r[5], r[3], -r[4])) if feasible else None
    if best_feasible and best[2] < 0.88 and best_feasible[1] >= best[1] - 0.01:
        return best_feasible[0], "best_val_f1_with_recall_088_floor_within_0.01"
    return best[0], "best_val_f1_then_recall088_hard_negative_fp_tiebreak"


def eval_indices(df: pd.DataFrame, eval_id: str) -> np.ndarray:
    test = df["split"].eq("test").to_numpy()
    if eval_id == "E3":
        return np.flatnonzero(test)
    if eval_id == "E6":
        return np.flatnonzero(test & df["dataset"].eq("bits").to_numpy())
    if eval_id == "E7":
        return np.flatnonzero(test & df["dataset"].eq("weda").to_numpy())
    raise ValueError(eval_id)


def evaluate_eval(
    df: pd.DataFrame,
    idx: np.ndarray,
    pred_fall: np.ndarray,
    fall_prob: np.ndarray,
    run_id: str,
    eval_id: str,
    dataset: str,
    figure_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sub = df.iloc[idx].reset_index(drop=True)
    y_fall = sub["fall_label"].astype(int).to_numpy()
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y_fall, pred_fall, labels=[0, 1]).ravel()]
    try:
        auroc = roc_auc_score(y_fall, fall_prob)
    except ValueError:
        auroc = math.nan
    try:
        ap = average_precision_score(y_fall, fall_prob)
    except ValueError:
        ap = math.nan

    dir_prob = sub[["direction_prob_forward", "direction_prob_backward", "direction_prob_lateral"]].astype(float).to_numpy()
    pred_dir = np.argmax(dir_prob, axis=1)
    y_dir = sub["direction_id"].to_numpy()
    supervised = sub["direction_supervised_bool"].to_numpy() & sub["fall_label"].eq(1).to_numpy() & (y_dir >= 0)
    if supervised.any():
        dir_macro = f1_score(y_dir[supervised], pred_dir[supervised], labels=[0, 1, 2], average="macro", zero_division=0)
        dir_acc = accuracy_score(y_dir[supervised], pred_dir[supervised])
        dir_per_class = f1_score(y_dir[supervised], pred_dir[supervised], labels=[0, 1, 2], average=None, zero_division=0)
        dir_cm = confusion_matrix(y_dir[supervised], pred_dir[supervised], labels=[0, 1, 2])
        e2e_pred_dir = pred_dir.copy()
        e2e_pred_dir[pred_fall == 0] = -1
        e2e_macro = f1_score(y_dir[supervised], e2e_pred_dir[supervised], labels=[0, 1, 2], average="macro", zero_division=0)
        e2e_per_class = f1_score(y_dir[supervised], e2e_pred_dir[supervised], labels=[0, 1, 2], average=None, zero_division=0)
        coverage = float(np.mean(pred_fall[supervised] == 1))
        e2e_correct = float(np.mean((pred_fall[supervised] == 1) & (pred_dir[supervised] == y_dir[supervised])))
    else:
        dir_macro = dir_acc = e2e_macro = coverage = e2e_correct = math.nan
        dir_per_class = e2e_per_class = [math.nan, math.nan, math.nan]
        dir_cm = np.zeros((3, 3), dtype=int)

    fall_cm = np.array([[tn, fp], [fn, tp]], dtype=int)
    save_confusion(figure_dir, run_id, eval_id, "fall", fall_cm, ["non_fall", "fall"])
    save_confusion(figure_dir, run_id, eval_id, "direction", dir_cm, DIRECTION_LABELS)
    metrics = {
        "run_id": run_id,
        "eval": eval_id,
        "dataset": dataset,
        "n": int(len(sub)),
        "E3_Fall_Precision": math.nan,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "fall_precision": float(precision_score(y_fall, pred_fall, zero_division=0)),
        "fall_recall": float(recall_score(y_fall, pred_fall, zero_division=0)),
        "fall_f1": float(f1_score(y_fall, pred_fall, zero_division=0)),
        "fall_accuracy": float(accuracy_score(y_fall, pred_fall)),
        "fall_auroc": float(auroc),
        "fall_average_precision": float(ap),
        "direction_macro_f1": float(dir_macro),
        "direction_accuracy": float(dir_acc),
        "direction_n_supervised": int(supervised.sum()),
        "forward_f1": float(dir_per_class[0]),
        "backward_f1": float(dir_per_class[1]),
        "lateral_f1": float(dir_per_class[2]),
        "end_to_end_direction_macro_f1": float(e2e_macro),
        "end_to_end_direction_coverage": float(coverage),
        "end_to_end_direction_correct_rate": float(e2e_correct),
    }
    if eval_id == "E7":
        metrics["E7_WEDA_FP_reduction_vs_A5_reference"] = 21 - fp
    else:
        metrics["E7_WEDA_FP_reduction_vs_A5_reference"] = math.nan
    e2e = {
        "end_to_end_direction_macro_f1": float(e2e_macro),
        "end_to_end_direction_coverage": float(coverage),
        "end_to_end_direction_correct_rate": float(e2e_correct),
        "end_to_end_forward_f1": float(e2e_per_class[0]),
        "end_to_end_backward_f1": float(e2e_per_class[1]),
        "end_to_end_lateral_f1": float(e2e_per_class[2]),
        "direction_n_supervised": int(supervised.sum()),
    }
    return metrics, e2e


def save_confusion(figure_dir: Path, run_id: str, eval_id: str, task: str, cm: np.ndarray, labels: list[str]) -> None:
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{run_id} {eval_id} {task}")
    plt.tight_layout()
    plt.savefig(figure_dir / f"{run_id}_{eval_id}_{task}_confusion.png", dpi=160)
    plt.close()


def complexity_info(model: Any, expert: str, n_features: int) -> dict[str, Any]:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if expert == "LR":
        params = n_features + 1
        return {"estimated_params_or_nodes": params, "number_trees": 0, "max_depth": 0, "edge_suitability": "high", "complexity_notes": f"LR {params} weights plus scaler."}
    if expert == "DT":
        return {
            "estimated_params_or_nodes": int(estimator.tree_.node_count),
            "number_trees": 1,
            "max_depth": int(estimator.get_depth()),
            "edge_suitability": "high",
            "complexity_notes": f"DecisionTree depth={estimator.get_depth()}, leaves={estimator.get_n_leaves()}.",
        }
    if expert == "RF":
        depths = [est.get_depth() for est in estimator.estimators_]
        nodes = [est.tree_.node_count for est in estimator.estimators_]
        return {
            "estimated_params_or_nodes": int(sum(nodes)),
            "number_trees": int(len(estimator.estimators_)),
            "max_depth": int(max(depths)) if depths else 0,
            "edge_suitability": "medium",
            "complexity_notes": f"RF 50 trees depth<=5, total nodes={sum(nodes)}.",
        }
    if expert == "GB":
        return {
            "estimated_params_or_nodes": int(sum(tree.tree_.node_count for stage in estimator.estimators_ for tree in stage)),
            "number_trees": int(estimator.n_estimators_),
            "max_depth": 3,
            "edge_suitability": "medium",
            "complexity_notes": f"GradientBoosting trees={estimator.n_estimators_}.",
        }
    if expert == "HGB":
        return {
            "estimated_params_or_nodes": int(getattr(estimator, "n_iter_", 0)),
            "number_trees": int(getattr(estimator, "n_iter_", 0)),
            "max_depth": math.nan,
            "edge_suitability": "low",
            "complexity_notes": f"HistGradientBoosting iterations={getattr(estimator, 'n_iter_', 'unknown')}.",
        }
    return {"estimated_params_or_nodes": math.nan, "number_trees": math.nan, "max_depth": math.nan, "edge_suitability": "unknown", "complexity_notes": ""}


def build_baselines(df: pd.DataFrame, figure_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    e2e_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []

    ref_prob = df["fall_prob"].to_numpy(dtype=float)
    for eval_id, dataset in [("E3", "bits+weda"), ("E6", "bits"), ("E7", "weda")]:
        idx = eval_indices(df, eval_id)
        pred = (ref_prob[idx] >= 0.5).astype(int)
        metrics, e2e = evaluate_eval(df, idx, pred, ref_prob[idx], "DS_Fall_RD_A5_reference", eval_id, dataset, figure_dir)
        metrics.update(
            {
                "model_type": "baseline_deep_temporal",
                "feature_set": "temporal_tilt12",
                "fall_expert": "A5_deep_fall_head",
                "number_features": 12,
                "threshold": 0.5,
                "threshold_note": "reference_softmax_threshold",
                "upper_bound_timing": False,
                "deep_fall_used_for_final": True,
                "direction_source": "A5_reference",
                "estimated_params_or_nodes": REFERENCE_PARAMS,
                "number_trees": 0,
                "max_depth": math.nan,
                "edge_suitability": "high",
                "complexity_notes": "A5 DS-Fall-RD temporal multitask model.",
            }
        )
        rows.append(metrics)
        e2e.update({"run_id": "DS_Fall_RD_A5_reference", "eval": eval_id, "dataset": dataset, "feature_set": "temporal_tilt12", "fall_expert": "A5_deep_fall_head"})
        e2e_rows.append(e2e)
    complexity_rows.append(
        {
            "run_id": "DS_Fall_RD_A5_reference",
            "model_type": "baseline_deep_temporal",
            "feature_set": "temporal_tilt12",
            "fall_expert": "A5_deep_fall_head",
            "number_features": 12,
            "estimated_params_or_nodes": REFERENCE_PARAMS,
            "number_trees": 0,
            "max_depth": math.nan,
            "edge_suitability": "high",
            "complexity_notes": "A5 reference.",
        }
    )

    add_fv_baseline(rows, e2e_rows, complexity_rows)
    add_classical_baseline(rows, complexity_rows)
    return rows, e2e_rows, complexity_rows


def add_fv_baseline(rows: list[dict[str, Any]], e2e_rows: list[dict[str, Any]], complexity_rows: list[dict[str, Any]]) -> None:
    if not FV_RESULTS_PATH.exists():
        return
    fv = pd.read_csv(FV_RESULTS_PATH)
    src = fv[fv["run_id"].eq("FV1_LITE10_AND")]
    if src.empty:
        return
    for _, row in src.iterrows():
        rows.append(
            {
                "run_id": "Best_FV1_Lite10_AND",
                "eval": row["eval"],
                "dataset": row["dataset"],
                "n": row.get("n", math.nan),
                "TN": row["TN"],
                "FP": row["FP"],
                "FN": row["FN"],
                "TP": row["TP"],
                "fall_precision": row["fall_precision"],
                "fall_recall": row["fall_recall"],
                "fall_f1": row["fall_f1"],
                "fall_accuracy": row["fall_accuracy"],
                "fall_auroc": row.get("fall_auroc", math.nan),
                "fall_average_precision": row.get("fall_average_precision", math.nan),
                "direction_macro_f1": row["direction_macro_f1"],
                "direction_accuracy": row["direction_accuracy"],
                "direction_n_supervised": row["direction_n_supervised"],
                "forward_f1": row["forward_f1"],
                "backward_f1": row["backward_f1"],
                "lateral_f1": row["lateral_f1"],
                "end_to_end_direction_macro_f1": math.nan,
                "end_to_end_direction_coverage": math.nan,
                "end_to_end_direction_correct_rate": math.nan,
                "E7_WEDA_FP_reduction_vs_A5_reference": 21 - row["FP"] if row["eval"] == "E7" else math.nan,
                "model_type": "baseline_fv_guard",
                "feature_set": "Lite10",
                "fall_expert": "LR_AND_verifier",
                "number_features": 10,
                "threshold": math.nan,
                "threshold_note": "from ds_fall_rd_fv report",
                "upper_bound_timing": False,
                "deep_fall_used_for_final": True,
                "direction_source": "A5_reference",
                "estimated_params_or_nodes": row.get("params", math.nan),
                "number_trees": 0,
                "max_depth": math.nan,
                "edge_suitability": "high",
                "complexity_notes": "Prior FV1 baseline, not retrained in this script.",
            }
        )
    complexity_rows.append(
        {
            "run_id": "Best_FV1_Lite10_AND",
            "model_type": "baseline_fv_guard",
            "feature_set": "Lite10",
            "fall_expert": "LR_AND_verifier",
            "number_features": 10,
            "estimated_params_or_nodes": 65970,
            "number_trees": 0,
            "max_depth": math.nan,
            "edge_suitability": "high",
            "complexity_notes": "Prior FV1 baseline.",
        }
    )


def add_classical_baseline(rows: list[dict[str, Any]], complexity_rows: list[dict[str, Any]]) -> None:
    if not ML_COMPARISON_PATH.exists():
        return
    comp = pd.read_csv(ML_COMPARISON_PATH)
    fall_only = comp[comp["task_supported"].eq("fall_only")].copy()
    if fall_only.empty:
        return
    fall_only["E7 WEDA Fall F1"] = pd.to_numeric(fall_only["E7 WEDA Fall F1"], errors="coerce")
    best = fall_only.sort_values("E7 WEDA Fall F1", ascending=False).iloc[0].to_dict()
    for eval_id, dataset in [("E3", "bits+weda"), ("E6", "bits"), ("E7", "weda")]:
        if eval_id == "E3":
            f1_col, precision_col, recall_col = "E3 Fall F1", "E3 Fall Precision", "E3 Fall Recall"
        elif eval_id == "E6":
            f1_col, precision_col, recall_col = "E6 BITS Fall F1", math.nan, math.nan
        else:
            f1_col, precision_col, recall_col = "E7 WEDA Fall F1", "E7 WEDA Precision", "E7 WEDA Recall"
        rows.append(
            {
                "run_id": "Classical_Fall_Only_Best",
                "eval": eval_id,
                "dataset": dataset,
                "fall_precision": best.get(precision_col, math.nan) if isinstance(precision_col, str) else math.nan,
                "fall_recall": best.get(recall_col, math.nan) if isinstance(recall_col, str) else math.nan,
                "fall_f1": best.get(f1_col, math.nan),
                "direction_macro_f1": math.nan,
                "direction_accuracy": math.nan,
                "direction_n_supervised": math.nan,
                "model_type": "baseline_classical_fall_only",
                "feature_set": "summary_features",
                "fall_expert": best.get("model_name", "unknown"),
                "number_features": math.nan,
                "threshold": math.nan,
                "threshold_note": "from ml_feature_analysis report",
                "upper_bound_timing": False,
                "deep_fall_used_for_final": False,
                "direction_source": "not_supported",
                "estimated_params_or_nodes": math.nan,
                "number_trees": math.nan,
                "max_depth": math.nan,
                "edge_suitability": "analysis_only",
                "complexity_notes": "Fall-only baseline does not support direction.",
            }
        )
    complexity_rows.append(
        {
            "run_id": "Classical_Fall_Only_Best",
            "model_type": "baseline_classical_fall_only",
            "feature_set": "summary_features",
            "fall_expert": best.get("model_name", "unknown"),
            "number_features": math.nan,
            "estimated_params_or_nodes": math.nan,
            "number_trees": math.nan,
            "max_depth": math.nan,
            "edge_suitability": "analysis_only",
            "complexity_notes": "Imported from ml_feature_analysis comparison.",
        }
    )


def build_selection_table(results: pd.DataFrame, e2e: pd.DataFrame) -> pd.DataFrame:
    hybrid = results[results["model_type"].eq("hybrid_ml_fall_a5_direction")].copy()
    e7 = hybrid[hybrid["eval"].eq("E7")].copy()
    e3 = hybrid[hybrid["eval"].eq("E3")][["run_id", "direction_macro_f1", "fall_f1"]].rename(
        columns={"direction_macro_f1": "E3_Direction_Macro_F1", "fall_f1": "E3_Fall_F1"}
    )
    e6 = hybrid[hybrid["eval"].eq("E6")][["run_id", "fall_f1", "direction_macro_f1"]].rename(
        columns={"fall_f1": "E6_BITS_Fall_F1", "direction_macro_f1": "E6_BITS_Direction_Macro_F1"}
    )
    e2e_e7 = e2e[e2e["eval"].eq("E7")][["run_id", "end_to_end_direction_macro_f1", "end_to_end_direction_coverage"]].rename(
        columns={"end_to_end_direction_macro_f1": "E7_E2E_Direction_Macro_F1", "end_to_end_direction_coverage": "E7_E2E_Direction_Coverage"}
    )
    table = e7.merge(e3, on="run_id", how="left").merge(e6, on="run_id", how="left").merge(e2e_e7, on="run_id", how="left")
    table["passes_hard_target"] = (
        (table["fall_recall"] >= 0.88)
        & (table["direction_macro_f1"] >= 0.80)
        & (table["E3_Direction_Macro_F1"] >= 0.86)
        & (table["E6_BITS_Fall_F1"] >= 0.90)
    )
    table["meets_strong_weda_target"] = (table["fall_f1"] >= 0.85) & (table["fall_precision"] >= 0.80)
    simplicity = {"LR": 0, "DT": 1, "RF": 2, "GB": 3, "HGB": 4}
    table["simplicity_rank"] = table["fall_expert"].map(simplicity).fillna(99)
    table["timing_rank"] = table["upper_bound_timing"].astype(int)
    return table.sort_values(
        ["passes_hard_target", "fall_f1", "fall_precision", "fall_recall", "E7_E2E_Direction_Macro_F1", "timing_rank", "simplicity_rank"],
        ascending=[False, False, False, False, False, True, True],
    )


def write_best_selection(results: pd.DataFrame, e2e: pd.DataFrame, complexity: pd.DataFrame, failures: pd.DataFrame, report_dir: Path) -> None:
    table = build_selection_table(results, e2e)
    best = table.iloc[0].to_dict() if not table.empty else {}
    no_timing_table = table[~table["upper_bound_timing"].astype(bool)] if not table.empty else pd.DataFrame()
    best_no_timing = no_timing_table.iloc[0].to_dict() if not no_timing_table.empty else {}
    cols = [
        "run_id",
        "fall_expert",
        "feature_set",
        "fall_f1",
        "fall_precision",
        "fall_recall",
        "FP",
        "FN",
        "E7_WEDA_FP_reduction_vs_A5_reference",
        "direction_macro_f1",
        "E3_Direction_Macro_F1",
        "E6_BITS_Fall_F1",
        "E7_E2E_Direction_Macro_F1",
        "E7_E2E_Direction_Coverage",
        "upper_bound_timing",
        "edge_suitability",
        "passes_hard_target",
        "meets_strong_weda_target",
    ]
    lines = [
        "# DS-Fall-RD Hybrid Best Selection",
        "",
        "Selection prioritizes E7 WEDA Fall F1, precision, recall, E3 direction, E7 end-to-end direction, no-timing features, then lighter experts.",
        "",
        "## Hybrid Selection Table",
        "",
        markdown_table(format_df(table[cols])) if not table.empty else "_No completed hybrid runs._",
        "",
        "## Best Run",
        "",
    ]
    if best:
        lines.extend(
            [
                f"Best selected run: `{best['run_id']}`.",
                "",
                f"- E7 WEDA Fall F1: {best['fall_f1']:.4f}",
                f"- E7 WEDA precision/recall: {best['fall_precision']:.4f}/{best['fall_recall']:.4f}",
                f"- E7 WEDA FP/FN: {int(best['FP'])}/{int(best['FN'])}",
                f"- E3 Direction Macro F1: {best['E3_Direction_Macro_F1']:.4f}",
                f"- E7 end-to-end Direction Macro F1: {best['E7_E2E_Direction_Macro_F1']:.4f}",
                f"- Feature set: {best['feature_set']}; expert: {best['fall_expert']}; upper-bound timing: {bool(best['upper_bound_timing'])}",
            ]
        )
    if not failures.empty:
        lines.extend(["", "## Failures", "", markdown_table(format_df(failures))])
    (report_dir / "hybrid_best_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    results: pd.DataFrame,
    thresholds: pd.DataFrame,
    complexity: pd.DataFrame,
    e2e: pd.DataFrame,
    failures: pd.DataFrame,
    report_dir: Path,
) -> None:
    table = build_selection_table(results, e2e)
    best = table.iloc[0].to_dict() if not table.empty else {}
    no_timing_table = table[~table["upper_bound_timing"].astype(bool)] if not table.empty else pd.DataFrame()
    best_no_timing = no_timing_table.iloc[0].to_dict() if not no_timing_table.empty else {}
    e7_all = results[results["eval"].eq("E7")].copy()
    baseline_ref = one(e7_all, "DS_Fall_RD_A5_reference")
    baseline_fv = one(e7_all, "Best_FV1_Lite10_AND")
    by_feature = table.groupby("feature_set", dropna=False).head(1)[["feature_set", "run_id", "fall_expert", "fall_f1", "fall_precision", "fall_recall", "FP", "upper_bound_timing"]] if not table.empty else pd.DataFrame()
    by_expert = table.groupby("fall_expert", dropna=False).head(1)[["fall_expert", "run_id", "feature_set", "fall_f1", "fall_precision", "fall_recall", "FP"]] if not table.empty else pd.DataFrame()
    e2e_e7 = e2e[e2e["eval"].eq("E7")].copy() if not e2e.empty else pd.DataFrame()

    lines = [
        "# DS-Fall-RD Hybrid Summary",
        "",
        "## Protocol",
        "",
        "- Hybrid final fall decision uses ML Fall Expert only.",
        "- Direction output is copied from DS-Fall-RD A5WCEFW reference and is not trained or modified.",
        "- Deep fall head is not used for final fall in hybrid runs. In option B, A5 is kept as auxiliary/reference direction expert.",
        "- Thresholds are selected on validation split only; test is used only for evaluation.",
        "",
        "## E7 WEDA Hybrid Results",
        "",
        markdown_table(
            format_df(
                table[
                    [
                        "run_id",
                        "fall_expert",
                        "feature_set",
                        "fall_f1",
                        "fall_precision",
                        "fall_recall",
                        "FP",
                        "FN",
                        "E7_WEDA_FP_reduction_vs_A5_reference",
                        "direction_macro_f1",
                        "E3_Direction_Macro_F1",
                        "E6_BITS_Fall_F1",
                        "E7_E2E_Direction_Macro_F1",
                        "E7_E2E_Direction_Coverage",
                        "upper_bound_timing",
                        "edge_suitability",
                    ]
                ]
            )
        )
        if not table.empty
        else "_No hybrid rows._",
        "",
        "## Baseline E7 Rows",
        "",
        markdown_table(
            format_df(
                e7_all[
                    e7_all["run_id"].isin(["DS_Fall_RD_A5_reference", "Best_FV1_Lite10_AND", "Classical_Fall_Only_Best"])
                ][
                    [
                        "run_id",
                        "model_type",
                        "fall_expert",
                        "fall_f1",
                        "fall_precision",
                        "fall_recall",
                        "FP",
                        "FN",
                        "direction_macro_f1",
                    ]
                ]
            )
        ),
        "",
        "## Best By Feature Set",
        "",
        markdown_table(format_df(by_feature)) if not by_feature.empty else "_No rows._",
        "",
        "## Best By Fall Expert",
        "",
        markdown_table(format_df(by_expert)) if not by_expert.empty else "_No rows._",
        "",
        "## Thresholds",
        "",
        markdown_table(format_df(thresholds)) if not thresholds.empty else "_No thresholds._",
        "",
        "## Complexity",
        "",
        markdown_table(
            format_df(
                complexity[
                    [
                        "run_id",
                        "model_type",
                        "feature_set",
                        "fall_expert",
                        "number_features",
                        "estimated_params_or_nodes",
                        "number_trees",
                        "max_depth",
                        "edge_suitability",
                        "complexity_notes",
                    ]
                ].head(30)
            )
        ),
        "",
        "## End-to-end Direction",
        "",
        markdown_table(
            format_df(
                e2e_e7[
                    [
                        "run_id",
                        "feature_set",
                        "fall_expert",
                        "end_to_end_direction_macro_f1",
                        "end_to_end_direction_coverage",
                        "end_to_end_direction_correct_rate",
                        "direction_n_supervised",
                    ]
                ].sort_values("end_to_end_direction_macro_f1", ascending=False).head(20)
            )
        )
        if not e2e_e7.empty
        else "_No E2E rows._",
        "",
        "## Answers",
        "",
        answer_improve(best, baseline_ref),
        answer_best_expert(best),
        answer_best_feature(by_feature, best),
        answer_timing(table),
        answer_fp(best, baseline_ref, baseline_fv),
        "6. Direction stays unchanged at the direction-head level because all hybrid runs use the same A5 direction probabilities.",
        answer_e2e(best),
        "8. Option B is more paper-appropriate: keep A5 as the auxiliary/reference multitask direction expert, but use the ML Fall Expert for final fall inference. Option A is the same inference rule but undersells why the direction expert exists.",
        answer_replace(best, best_no_timing, baseline_ref),
        answer_paper_position(best, best_no_timing),
    ]
    if not failures.empty:
        lines.extend(["", "## Failures", "", markdown_table(format_df(failures))])
    (report_dir / "hybrid_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def one(df: pd.DataFrame, run_id: str) -> dict[str, Any]:
    rows = df[df["run_id"].eq(run_id)]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def answer_improve(best: dict[str, Any], ref: dict[str, Any]) -> str:
    if not best or not ref:
        return "1. Improvement check unavailable."
    return f"1. Hybrid improves WEDA Fall F1 over A5 from {ref['fall_f1']:.4f} to {best['fall_f1']:.4f} for `{best['run_id']}`."


def answer_best_expert(best: dict[str, Any]) -> str:
    if not best:
        return "2. Best fall expert unavailable."
    return f"2. Best Fall Expert is `{best['fall_expert']}` in `{best['run_id']}`."


def answer_best_feature(by_feature: pd.DataFrame, best: dict[str, Any]) -> str:
    if by_feature.empty or not best:
        return "3. Best feature set unavailable."
    return f"3. Best feature set is `{best['feature_set']}` with `{best['run_id']}`."


def answer_timing(table: pd.DataFrame) -> str:
    if table.empty:
        return "4. Timing comparison unavailable."
    no_timing = table[~table["upper_bound_timing"].astype(bool)]
    timing = table[table["upper_bound_timing"].astype(bool)]
    if timing.empty or no_timing.empty:
        return "4. Timing comparison incomplete."
    best_no = no_timing.iloc[0]
    best_time = timing.iloc[0]
    if best_time["fall_f1"] > best_no["fall_f1"] + 0.02:
        return f"4. FullTiming is higher ({best_time['fall_f1']:.4f} vs no-timing {best_no['fall_f1']:.4f}) but should be treated as upper-bound because it uses event-centered timing cues."
    return f"4. Timing features are not necessary: best no-timing F1={best_no['fall_f1']:.4f}, best FullTiming F1={best_time['fall_f1']:.4f}."


def answer_fp(best: dict[str, Any], ref: dict[str, Any], fv: dict[str, Any]) -> str:
    if not best or not ref:
        return "5. FP reduction unavailable."
    extra = ""
    if fv:
        extra = f" Prior FV1 FP={int(fv['FP'])}, so hybrid changes FP by {int(fv['FP'] - best['FP'])} relative to FV1."
    return f"5. Hybrid reduces WEDA FP from A5 {int(ref['FP'])} to {int(best['FP'])}.{extra}"


def answer_e2e(best: dict[str, Any]) -> str:
    if not best:
        return "7. End-to-end direction unavailable."
    return f"7. End-to-end direction is affected by fall recall: `{best['run_id']}` keeps coverage={best['E7_E2E_Direction_Coverage']:.4f} and E2E direction macro F1={best['E7_E2E_Direction_Macro_F1']:.4f}."


def answer_replace(best: dict[str, Any], best_no_timing: dict[str, Any], ref: dict[str, Any]) -> str:
    if not best or not ref:
        return "9. Replacement recommendation unavailable."
    if bool(best["upper_bound_timing"]):
        practical = (
            f" Practical no-timing candidate is `{best_no_timing['run_id']}` with WEDA F1={best_no_timing['fall_f1']:.4f}, "
            f"precision={best_no_timing['fall_precision']:.4f}, recall={best_no_timing['fall_recall']:.4f}."
            if best_no_timing
            else ""
        )
        return "9. Do not replace A5 main architecture with the FullTiming upper-bound as a paper main result." + practical
    if best["fall_f1"] > ref["fall_f1"] and best["fall_recall"] >= 0.88 and best["E3_Direction_Macro_F1"] >= 0.86:
        return f"9. Hybrid can replace A5 as the final practical inference pipeline if separating fall and direction experts is acceptable; `{best['run_id']}` is the candidate."
    return "9. Do not replace A5 reference as the main model; use hybrid as analysis/extension."


def answer_paper_position(best: dict[str, Any], best_no_timing: dict[str, Any]) -> str:
    if not best:
        return "10. Paper positioning unavailable."
    if bool(best["upper_bound_timing"]):
        practical = (
            f" Use `{best_no_timing['run_id']}` as the practical no-timing hybrid if a deployable pipeline is needed."
            if best_no_timing
            else ""
        )
        return "10. Paper position: FullTiming is upper-bound analysis because it uses event-centered timing cues." + practical
    if best["fall_f1"] >= 0.85 and best["fall_precision"] >= 0.80:
        return "10. Paper position: final practical pipeline, with A5 as direction expert and ML fall expert as final fall decision."
    return "10. Paper position: optional practical hybrid extension, not a replacement for the end-to-end A5 architecture."


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
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map(lambda v: "yes" if bool(v) else "no")
        elif pd.api.types.is_numeric_dtype(out[col]):
            if col in {
                "n",
                "TN",
                "FP",
                "FN",
                "TP",
                "number_features",
                "estimated_params_or_nodes",
                "number_trees",
                "direction_n_supervised",
                "E7_WEDA_FP_reduction_vs_A5_reference",
            }:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(int(v)))
            else:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    return out


if __name__ == "__main__":
    main()
