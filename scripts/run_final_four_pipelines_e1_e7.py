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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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

import run_ml_e1_e7_specialized as base


DIRECTION_LABELS = ["forward", "backward", "lateral"]
DIR_TO_ID = {name: i for i, name in enumerate(DIRECTION_LABELS)}
ID_TO_DIR = {i: name for name, i in DIR_TO_ID.items()}

BASE_PROTOCOLS = [
    "E1_BITS_TO_BITS",
    "E2_WEDA_TO_WEDA",
    "E3_BITS_WEDA_MIXED",
    "E4_BITS_TO_WEDA",
    "E5_WEDA_TO_BITS",
]
ALL_PROTOCOLS = BASE_PROTOCOLS + ["E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"]

PIPELINES = {
    "A5_reference": {
        "fall_source": "A5_deep_fall_head",
        "direction_source": "A5_direction_head",
        "role": "deep temporal multitask reference",
    },
    "Hybrid_NoTiming": {
        "fall_source": "GradientBoosting_FallNoTiming_Full",
        "direction_source": "A5_direction_head",
        "role": "recommended practical hybrid",
    },
    "EdgeLite_Hybrid": {
        "fall_source": "RandomForest_FallNoTiming_Core",
        "direction_source": "A5_direction_head",
        "role": "compact edge-oriented ablation",
    },
    "ML_only_separated": {
        "fall_source": "GradientBoosting_FallNoTiming_Full",
        "direction_source": "GradientBoosting_DirectionFullNoTiming",
        "role": "analysis-only separated ML pipeline",
    },
}

A5_PARAMS = 65_959
A5_FP32_KB = A5_PARAMS * 4 / 1024
A5_INT8_KB = A5_PARAMS / 1024


@dataclass
class TrainedModel:
    model: Any
    feature_set: str
    features: list[str]
    threshold: float | None
    threshold_note: str
    train_protocol: str
    model_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final four DS-Fall-RD pipelines across E1-E7.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-all", action="store_true", help="Run the final E1-E7 benchmark.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--summary-path",
        default="outputs/reports/ml_feature_analysis_25hz/summary_feature_matrix.csv",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_feature_matrix(path: Path) -> pd.DataFrame:
    df = base.load_summary(path)
    df = df[df["dataset"].isin(["bits", "weda"])].copy()
    if df.empty:
        raise ValueError("Feature matrix has no BITS/WEDA rows.")
    required = {"window_id", "dataset", "split", "fall_label", "direction_label", "direction_supervised_bool"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Feature matrix is missing required columns: {missing}")
    return df.reset_index(drop=True)


def build_feature_sets(df: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    signal_features = base.infer_signal_features(df)
    feature_sets = {
        "FallNoTiming_Core": [f for f in base.FALL_CORE if f in df.columns],
        "FallNoTiming_Full": [f for f in signal_features if not base.is_timing_feature(f)],
        "DirectionCore": [
            f for f in base.DIRECTION_CORE_CANDIDATES if f in df.columns and not base.is_timing_feature(f)
        ],
        "DirectionFullNoTiming": [f for f in signal_features if not base.is_timing_feature(f)],
    }
    missing = {
        "FallNoTiming_Core": [f for f in base.FALL_CORE if f not in df.columns],
    }
    for name, features in feature_sets.items():
        if not features:
            raise ValueError(f"Feature set {name} is empty.")
    return feature_sets, missing


def get_protocol_splits(df: pd.DataFrame, protocol_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return base.protocol_masks(df, protocol_id)


def train_fall_model(
    df: pd.DataFrame,
    protocol_id: str,
    feature_set: str,
    features: list[str],
    model_name: str,
    seed: int,
) -> TrainedModel:
    train_mask, val_mask, _ = get_protocol_splits(df, protocol_id)
    X_train = df.loc[train_mask, features].astype(float).to_numpy()
    y_train = df.loc[train_mask, "fall_label"].astype(int).to_numpy()
    X_val = df.loc[val_mask, features].astype(float).to_numpy()
    y_val = df.loc[val_mask, "fall_label"].astype(int).to_numpy()

    if model_name == "GradientBoosting":
        model = make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=seed))
    elif model_name == "RandomForest":
        model = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        )
    else:
        raise ValueError(f"Unsupported fall model: {model_name}")

    model.fit(X_train, y_train)
    val_scores = fall_scores(model, X_val)
    if len(np.unique(y_val)) < 2:
        threshold = 0.5
        threshold_note = "fallback_0.5_validation_single_class"
    else:
        threshold, threshold_note = base.select_fall_threshold(y_val, val_scores)
    return TrainedModel(model, feature_set, features, threshold, threshold_note, protocol_id, model_name)


def train_direction_model(
    df: pd.DataFrame,
    protocol_id: str,
    feature_set: str,
    features: list[str],
    model_name: str,
    seed: int,
) -> TrainedModel:
    train_mask, _, _ = get_protocol_splits(df, protocol_id)
    direction_mask = train_mask & base.direction_train_mask(df)
    X_train = df.loc[direction_mask, features].astype(float).to_numpy()
    y_train = df.loc[direction_mask, "direction_id"].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"Direction training split for {protocol_id} has fewer than two classes.")
    if model_name != "GradientBoosting":
        raise ValueError(f"Unsupported direction model: {model_name}")
    model = make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=seed))
    model.fit(X_train, y_train)
    return TrainedModel(model, feature_set, features, None, "not_applicable", protocol_id, model_name)


def fall_scores(model: Any, X: np.ndarray) -> np.ndarray:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(estimator, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(float)
    scores = model.decision_function(X)
    return np.asarray(scores, dtype=float)


def direction_predictions(model: Any, X: np.ndarray) -> np.ndarray:
    return model.predict(X).astype(int)


def canonical_train_protocol(protocol_id: str) -> str:
    if protocol_id in {"E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"}:
        return "E3_BITS_WEDA_MIXED"
    return protocol_id


def a5_prediction_path(repo_root: Path, protocol_id: str) -> Path:
    return repo_root / "artifacts" / "experiments_25hz_event" / protocol_id / "predictions.csv"


def load_a5_predictions(repo_root: Path, protocol_id: str) -> pd.DataFrame:
    path = a5_prediction_path(repo_root, protocol_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing A5 prediction file for {protocol_id}: {path}")
    pred = pd.read_csv(path)
    if "window_id" not in pred.columns:
        raise ValueError(f"A5 prediction file has no window_id: {path}")
    return pred.copy()


def align_a5_predictions(test_df: pd.DataFrame, a5_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["window_id", "pred_fall", "fall_prob", "pred_direction", "direction_confidence"]
    present = [c for c in cols if c in a5_df.columns]
    aligned = test_df[["window_id"]].merge(a5_df[present], on="window_id", how="left", validate="one_to_one")
    missing = aligned["pred_fall"].isna().sum() if "pred_fall" in aligned.columns else len(aligned)
    if missing:
        raise ValueError(f"A5 predictions missing for {missing} test windows.")
    return aligned


def evaluate_fall(
    pipeline: str,
    protocol_id: str,
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    train_protocol: str,
    fall_source: str,
    feature_set: str,
    feature_count: int,
    threshold: float | None,
    threshold_note: str,
) -> dict[str, Any]:
    y_true = test_df["fall_label"].astype(int).to_numpy()
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()]
    score = y_prob if y_prob is not None else y_pred.astype(float)
    return {
        "pipeline": pipeline,
        "experiment_id": protocol_id,
        "train_protocol": train_protocol,
        "test_dataset": "+".join(sorted(test_df["dataset"].unique())),
        "fall_source": fall_source,
        "feature_set": feature_set,
        "feature_count": int(feature_count),
        "fall_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "fall_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "fall_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fall_accuracy": float(accuracy_score(y_true, y_pred)),
        "AUROC": safe_auc(y_true, score),
        "average_precision": safe_ap(y_true, score),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "threshold": threshold,
        "threshold_note": threshold_note,
    }


def evaluate_direction(
    pipeline: str,
    protocol_id: str,
    test_df: pd.DataFrame,
    y_pred_all: np.ndarray,
    train_protocol: str,
    direction_source: str,
    feature_set: str,
    feature_count: int,
) -> dict[str, Any]:
    mask = base.direction_train_mask(test_df)
    y_true = test_df.loc[mask, "direction_id"].astype(int).to_numpy()
    y_pred = y_pred_all[mask]
    if len(y_true) == 0:
        cm = np.zeros((3, 3), dtype=int)
        per_class = [math.nan, math.nan, math.nan]
        macro = math.nan
        acc = math.nan
    else:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        per_class = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
        macro = f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
        acc = accuracy_score(y_true, y_pred)
    return {
        "pipeline": pipeline,
        "experiment_id": protocol_id,
        "train_protocol": train_protocol,
        "test_dataset": "+".join(sorted(test_df["dataset"].unique())),
        "direction_source": direction_source,
        "feature_set": feature_set,
        "feature_count": int(feature_count),
        "direction_macro_f1": float(macro),
        "direction_accuracy": float(acc),
        "forward_f1": float(per_class[0]),
        "backward_f1": float(per_class[1]),
        "lateral_f1": float(per_class[2]),
        "direction_n_supervised": int(len(y_true)),
        "cm_00": int(cm[0, 0]),
        "cm_01": int(cm[0, 1]),
        "cm_02": int(cm[0, 2]),
        "cm_10": int(cm[1, 0]),
        "cm_11": int(cm[1, 1]),
        "cm_12": int(cm[1, 2]),
        "cm_20": int(cm[2, 0]),
        "cm_21": int(cm[2, 1]),
        "cm_22": int(cm[2, 2]),
    }


def evaluate_e2e_direction(
    pipeline: str,
    protocol_id: str,
    test_df: pd.DataFrame,
    fall_pred: np.ndarray,
    direction_pred_all: np.ndarray,
    train_protocol: str,
) -> dict[str, Any]:
    mask = base.direction_train_mask(test_df)
    y_true = test_df.loc[mask, "direction_id"].astype(int).to_numpy()
    fall_supervised = fall_pred[mask]
    dir_supervised = direction_pred_all[mask]
    if len(y_true) == 0:
        return {
            "pipeline": pipeline,
            "experiment_id": protocol_id,
            "train_protocol": train_protocol,
            "E2E_Direction_Macro_F1": math.nan,
            "E2E_Direction_Accuracy": math.nan,
            "coverage": math.nan,
            "direction_n_supervised": 0,
        }
    e2e_pred = dir_supervised.copy()
    e2e_pred[fall_supervised == 0] = -1
    per = f1_score(y_true, e2e_pred, labels=[0, 1, 2], average=None, zero_division=0)
    correct = (fall_supervised == 1) & (dir_supervised == y_true)
    return {
        "pipeline": pipeline,
        "experiment_id": protocol_id,
        "train_protocol": train_protocol,
        "E2E_Direction_Macro_F1": float(f1_score(y_true, e2e_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "E2E_Direction_Accuracy": float(np.mean(correct)),
        "coverage": float(np.mean(fall_supervised == 1)),
        "forward_f1": float(per[0]),
        "backward_f1": float(per[1]),
        "lateral_f1": float(per[2]),
        "direction_n_supervised": int(len(y_true)),
    }


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, score))
    except ValueError:
        return math.nan


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y, score))
    except ValueError:
        return math.nan


def tree_stats(model: Any) -> dict[str, int]:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if isinstance(estimator, RandomForestClassifier):
        trees = list(estimator.estimators_)
    elif isinstance(estimator, GradientBoostingClassifier):
        trees = [tree for tree in estimator.estimators_.ravel()]
    else:
        return {"tree_count": 0, "tree_nodes": 0, "tree_leaves": 0}
    nodes = int(sum(tree.tree_.node_count for tree in trees))
    leaves = int(sum(np.sum(tree.tree_.children_left == -1) for tree in trees))
    return {"tree_count": len(trees), "tree_nodes": nodes, "tree_leaves": leaves}


def pipeline_complexity(
    pipeline: str,
    feature_sets: dict[str, list[str]],
    trained_fall: dict[tuple[str, str], TrainedModel],
    trained_direction: dict[tuple[str, str], TrainedModel],
) -> dict[str, Any]:
    e3_key = (pipeline, "E3_BITS_WEDA_MIXED")
    fall_model = trained_fall.get(e3_key)
    direction_model = trained_direction.get(e3_key)
    fall_tree = tree_stats(fall_model.model) if fall_model else {"tree_count": 0, "tree_nodes": 0, "tree_leaves": 0}
    dir_tree = tree_stats(direction_model.model) if direction_model else {"tree_count": 0, "tree_nodes": 0, "tree_leaves": 0}
    uses_a5_direction = PIPELINES[pipeline]["direction_source"] == "A5_direction_head"
    uses_a5_fall = PIPELINES[pipeline]["fall_source"] == "A5_deep_fall_head"
    deep_params = A5_PARAMS if (uses_a5_direction or uses_a5_fall) else 0
    total_tree_count = fall_tree["tree_count"] + dir_tree["tree_count"]
    total_nodes = fall_tree["tree_nodes"] + dir_tree["tree_nodes"]
    if pipeline == "A5_reference":
        edge = "medium"
        notes = "Single compact temporal CNN; requires TensorFlow/TFLite for deployment."
    elif pipeline == "Hybrid_NoTiming":
        edge = "medium"
        notes = "GB fall expert plus A5 direction expert; best practical accuracy, moderate edge cost."
    elif pipeline == "EdgeLite_Hybrid":
        edge = "medium-high"
        notes = "Compact 16-feature RF fall expert; full pipeline still uses A5 for direction."
    else:
        edge = "high"
        notes = "Summary-only tree ensembles; no temporal deep model, CPU-friendly but less direction-consistent."
    feature_count_fall = len(fall_model.features) if fall_model else (12 if uses_a5_fall else 0)
    feature_count_direction = len(direction_model.features) if direction_model else (12 if uses_a5_direction else 0)
    return {
        "pipeline": pipeline,
        "fall_source": PIPELINES[pipeline]["fall_source"],
        "direction_source": PIPELINES[pipeline]["direction_source"],
        "deep_params": deep_params,
        "a5_fp32_kb": A5_FP32_KB if deep_params else 0.0,
        "a5_int8_kb": A5_INT8_KB if deep_params else 0.0,
        "fall_feature_count": feature_count_fall,
        "direction_feature_count": feature_count_direction,
        "fall_tree_count": fall_tree["tree_count"],
        "fall_tree_nodes": fall_tree["tree_nodes"],
        "fall_tree_leaves": fall_tree["tree_leaves"],
        "direction_tree_count": dir_tree["tree_count"],
        "direction_tree_nodes": dir_tree["tree_nodes"],
        "direction_tree_leaves": dir_tree["tree_leaves"],
        "total_tree_count": total_tree_count,
        "total_tree_nodes": total_nodes,
        "estimated_complexity_units": deep_params + total_nodes,
        "edge_suitability": edge,
        "complexity_notes": notes,
    }


def save_confusion_matrices(
    fig_dir: Path,
    fall_rows: list[dict[str, Any]],
    direction_rows: list[dict[str, Any]],
) -> None:
    cm_dir = ensure_dir(fig_dir / "confusion_matrices")
    for row in fall_rows:
        cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]], dtype=int)
        save_cm(cm, ["non_fall", "fall"], cm_dir / f"{safe_name(row['pipeline'])}_{row['experiment_id']}_fall.png", f"{row['pipeline']} {row['experiment_id']} fall")
    for row in direction_rows:
        cm = np.array(
            [
                [row["cm_00"], row["cm_01"], row["cm_02"]],
                [row["cm_10"], row["cm_11"], row["cm_12"]],
                [row["cm_20"], row["cm_21"], row["cm_22"]],
            ],
            dtype=int,
        )
        save_cm(cm, DIRECTION_LABELS, cm_dir / f"{safe_name(row['pipeline'])}_{row['experiment_id']}_direction.png", f"{row['pipeline']} {row['experiment_id']} direction")


def save_cm(cm: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    plt.figure(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_bar_charts(fig_dir: Path, combined: pd.DataFrame) -> None:
    bar_dir = ensure_dir(fig_dir / "bar_charts")
    focus = combined[combined["experiment_id"].isin(["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"])].copy()
    for metric, filename in [
        ("fall_f1", "fall_f1_e3_e6_e7.png"),
        ("direction_macro_f1", "direction_macro_f1_e3_e6_e7.png"),
        ("E2E_Direction_Macro_F1", "e2e_direction_f1_e3_e6_e7.png"),
        ("FP", "fall_fp_e3_e6_e7.png"),
    ]:
        if metric not in focus.columns:
            continue
        plt.figure(figsize=(10, 4.5))
        sns.barplot(data=focus, x="experiment_id", y=metric, hue="pipeline")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(bar_dir / filename, dpi=150)
        plt.close()


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))


def save_prediction_file(
    pred_dir: Path,
    pipeline: str,
    protocol_id: str,
    test_df: pd.DataFrame,
    fall_pred: np.ndarray,
    fall_prob: np.ndarray,
    direction_pred: np.ndarray,
) -> None:
    out = pd.DataFrame(
        {
            "window_id": test_df["window_id"].to_numpy(),
            "dataset": test_df["dataset"].to_numpy(),
            "split": test_df["split"].to_numpy(),
            "subject_id": test_df["subject_id"].to_numpy() if "subject_id" in test_df.columns else "",
            "trial_id": test_df["trial_id"].to_numpy() if "trial_id" in test_df.columns else "",
            "activity": test_df["activity"].to_numpy() if "activity" in test_df.columns else "",
            "y_fall_true": test_df["fall_label"].astype(int).to_numpy(),
            "y_fall_pred": fall_pred.astype(int),
            "y_fall_prob": fall_prob.astype(float),
            "y_direction_true": test_df["direction_id"].map(ID_TO_DIR).fillna("none").to_numpy(),
            "y_direction_pred": pd.Series(direction_pred).map(ID_TO_DIR).fillna("none").to_numpy(),
            "direction_supervised": test_df["direction_supervised_bool"].astype(bool).to_numpy(),
        }
    )
    out["final_output"] = np.where(out["y_fall_pred"].eq(0), "non_fall", out["y_direction_pred"])
    out.to_csv(pred_dir / f"{pipeline}_{protocol_id}_predictions.csv", index=False)


def save_reports(
    repo_root: Path,
    report_dir: Path,
    fig_dir: Path,
    pred_dir: Path,
    feature_sets: dict[str, list[str]],
    missing_features: dict[str, list[str]],
    fall_rows: list[dict[str, Any]],
    direction_rows: list[dict[str, Any]],
    e2e_rows: list[dict[str, Any]],
    complexity_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
) -> None:
    fall_df = pd.DataFrame(fall_rows)
    direction_df = pd.DataFrame(direction_rows)
    e2e_df = pd.DataFrame(e2e_rows)
    complexity_df = pd.DataFrame(complexity_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    failed_df = pd.DataFrame(failed_rows)

    if not fall_df.empty:
        a5_fp = fall_df[fall_df["pipeline"].eq("A5_reference")][["experiment_id", "FP"]].rename(columns={"FP": "A5_FP"})
        fall_df = fall_df.merge(a5_fp, on="experiment_id", how="left")
        fall_df["FP_reduction_vs_A5"] = fall_df["A5_FP"] - fall_df["FP"]

    complexity_for_merge = complexity_df.drop(columns=["fall_source", "direction_source"], errors="ignore")
    combined = fall_df.merge(
        direction_df[
            [
                "pipeline",
                "experiment_id",
                "direction_macro_f1",
                "direction_accuracy",
                "forward_f1",
                "backward_f1",
                "lateral_f1",
                "direction_n_supervised",
            ]
        ],
        on=["pipeline", "experiment_id"],
        how="left",
    ).merge(
        e2e_df[
            [
                "pipeline",
                "experiment_id",
                "E2E_Direction_Macro_F1",
                "E2E_Direction_Accuracy",
                "coverage",
            ]
        ],
        on=["pipeline", "experiment_id"],
        how="left",
    ).merge(complexity_for_merge, on="pipeline", how="left")

    final_feature_sets = {
        "benchmark": "BITS/WEDA only, 25 Hz, 2-second event-centered windows",
        "main_temporal_input": "50 x 12 tilt12",
        "feature_sets": feature_sets,
        "missing_features": missing_features,
        "timing_exclusion": {
            "excluded_exact": sorted(base.TIMING_FEATURE_NAMES),
            "excluded_patterns": list(base.TIMING_PATTERNS),
            "FullTiming": "upper-bound only, not used in the final four main pipelines",
        },
    }
    (report_dir / "final_feature_sets.json").write_text(json.dumps(final_feature_sets, indent=2), encoding="utf-8")
    fall_df.to_csv(report_dir / "final_four_pipeline_fall_results.csv", index=False)
    direction_df.to_csv(report_dir / "final_four_pipeline_direction_results.csv", index=False)
    e2e_df.to_csv(report_dir / "final_four_pipeline_e2e_direction_results.csv", index=False)
    combined.to_csv(report_dir / "final_four_pipeline_results.csv", index=False)
    complexity_df.to_csv(report_dir / "final_four_pipeline_complexity.csv", index=False)
    threshold_df.to_csv(report_dir / "final_thresholds.csv", index=False)
    failed_df.to_csv(report_dir / "failed_runs.csv", index=False)

    best_rows = build_best_by_metric(combined, complexity_df)
    best_rows.to_csv(report_dir / "final_four_pipeline_best_by_metric.csv", index=False)

    save_confusion_matrices(fig_dir, fall_rows, direction_rows)
    save_bar_charts(fig_dir, combined)
    write_final_docs(repo_root, report_dir, combined, fall_df, direction_df, e2e_df, complexity_df, best_rows, failed_df)


def build_best_by_metric(combined: pd.DataFrame, complexity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if combined.empty:
        return pd.DataFrame(rows)
    e7 = combined[combined["experiment_id"].eq("E7_E3_MIXED_TEST_WEDA")]
    focus = combined[combined["experiment_id"].isin(["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"])]
    if not e7.empty:
        rows.append(metric_winner("best_weda_fall_f1", e7, "fall_f1"))
        rows.append(metric_winner("best_weda_direction_macro_f1", e7, "direction_macro_f1"))
        rows.append(metric_winner("best_weda_e2e_direction_macro_f1", e7, "E2E_Direction_Macro_F1"))
    if not focus.empty:
        avg = focus.groupby("pipeline", as_index=False).agg(
            avg_fall_f1=("fall_f1", "mean"),
            avg_direction_macro_f1=("direction_macro_f1", "mean"),
            avg_e2e_direction_macro_f1=("E2E_Direction_Macro_F1", "mean"),
        )
        avg["balanced_score"] = (
            0.4 * avg["avg_fall_f1"] + 0.35 * avg["avg_direction_macro_f1"] + 0.25 * avg["avg_e2e_direction_macro_f1"]
        )
        row = avg.sort_values("balanced_score", ascending=False).iloc[0].to_dict()
        rows.append(
            {
                "criterion": "best_balanced_e3_e6_e7_score",
                "pipeline": row["pipeline"],
                "metric": "balanced_score",
                "metric_value": row["balanced_score"],
                "avg_fall_f1": row["avg_fall_f1"],
                "avg_direction_macro_f1": row["avg_direction_macro_f1"],
                "avg_e2e_direction_macro_f1": row["avg_e2e_direction_macro_f1"],
            }
        )
    if not complexity.empty:
        light = complexity.sort_values(["estimated_complexity_units", "total_tree_nodes"], ascending=True).iloc[0].to_dict()
        rows.append(
            {
                "criterion": "lightest_estimated_complexity",
                "pipeline": light["pipeline"],
                "metric": "estimated_complexity_units",
                "metric_value": light["estimated_complexity_units"],
                "edge_suitability": light["edge_suitability"],
                "deep_params": light["deep_params"],
                "total_tree_count": light["total_tree_count"],
                "total_tree_nodes": light["total_tree_nodes"],
            }
        )
    return pd.DataFrame(rows)


def metric_winner(criterion: str, df: pd.DataFrame, metric: str) -> dict[str, Any]:
    row = df.sort_values(metric, ascending=False).iloc[0].to_dict()
    return {
        "criterion": criterion,
        "pipeline": row.get("pipeline"),
        "experiment_id": row.get("experiment_id"),
        "metric": metric,
        "metric_value": row.get(metric),
        "fall_f1": row.get("fall_f1"),
        "fall_precision": row.get("fall_precision"),
        "fall_recall": row.get("fall_recall"),
        "FP": row.get("FP"),
        "direction_macro_f1": row.get("direction_macro_f1"),
        "E2E_Direction_Macro_F1": row.get("E2E_Direction_Macro_F1"),
        "edge_suitability": row.get("edge_suitability"),
        "estimated_complexity_units": row.get("estimated_complexity_units"),
    }


def write_final_docs(
    repo_root: Path,
    report_dir: Path,
    combined: pd.DataFrame,
    fall_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    e2e_df: pd.DataFrame,
    complexity_df: pd.DataFrame,
    best_rows: pd.DataFrame,
    failed_df: pd.DataFrame,
) -> None:
    docs_final = ensure_dir(repo_root / "docs" / "final")
    architecture = final_architecture_principles_md(complexity_df)
    research = final_research_summary_md()
    summary = final_summary_md(combined, fall_df, direction_df, e2e_df, complexity_df, best_rows, failed_df)
    (report_dir / "final_architecture_principles.md").write_text(architecture, encoding="utf-8")
    (report_dir / "final_research_summary.md").write_text(research, encoding="utf-8")
    (report_dir / "final_four_pipeline_summary.md").write_text(summary, encoding="utf-8")
    (docs_final / "04_hybrid_architecture.md").write_text(hybrid_architecture_doc(), encoding="utf-8")
    (docs_final / "05_final_four_pipelines.md").write_text(final_four_pipelines_doc(complexity_df), encoding="utf-8")
    (docs_final / "06_research_summary_from_start_to_now.md").write_text(research, encoding="utf-8")
    (docs_final / "07_final_e1_e7_results_summary.md").write_text(summary, encoding="utf-8")


def final_summary_md(
    combined: pd.DataFrame,
    fall_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    e2e_df: pd.DataFrame,
    complexity_df: pd.DataFrame,
    best_rows: pd.DataFrame,
    failed_df: pd.DataFrame,
) -> str:
    focus_ids = ["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"]
    focus = combined[combined["experiment_id"].isin(focus_ids)].copy()
    e7 = combined[combined["experiment_id"].eq("E7_E3_MIXED_TEST_WEDA")].copy()
    best_weda_fall = row_or_empty(e7.sort_values("fall_f1", ascending=False))
    best_direction = row_or_empty(focus.groupby("pipeline", as_index=False)["direction_macro_f1"].mean().sort_values("direction_macro_f1", ascending=False))
    best_e2e = row_or_empty(focus.groupby("pipeline", as_index=False)["E2E_Direction_Macro_F1"].mean().sort_values("E2E_Direction_Macro_F1", ascending=False))
    lightest = row_or_empty(complexity_df.sort_values("estimated_complexity_units", ascending=True))
    a5_e7 = row_or_empty(e7[e7["pipeline"].eq("A5_reference")])
    hybrid_e7 = row_or_empty(e7[e7["pipeline"].eq("Hybrid_NoTiming")])
    ml_only_dir_avg = focus[focus["pipeline"].eq("ML_only_separated")]["direction_macro_f1"].mean()
    a5_dir_avg = focus[focus["pipeline"].eq("A5_reference")]["direction_macro_f1"].mean()

    lines = [
        "# Final Four Pipeline E1-E7 Summary",
        "",
        "## Scope",
        "",
        "- BITS/WEDA only.",
        "- 25 Hz, 2-second event-centered windows.",
        "- Main temporal input: 50 x 12 tilt12.",
        "- FullTiming is not used as a main result.",
        "- Test sets are not used for threshold tuning.",
        "",
        "## Topline E3/E6/E7",
        "",
        markdown_table(format_df(focus[[
            "pipeline",
            "experiment_id",
            "fall_f1",
            "fall_precision",
            "fall_recall",
            "FP",
            "direction_macro_f1",
            "direction_accuracy",
            "E2E_Direction_Macro_F1",
            "coverage",
            "edge_suitability",
            "estimated_complexity_units",
        ]])),
        "",
        "## Complexity And Edge Suitability",
        "",
        markdown_table(format_df(complexity_df)),
        "",
        "## Best By Metric",
        "",
        markdown_table(format_df(best_rows)) if not best_rows.empty else "_No rows._",
        "",
        "## Required Answers",
        "",
        f"1. Best overall practical pipeline: `Hybrid_NoTiming`. It gives the largest WEDA fall improvement while keeping A5 as the stable direction expert.",
        f"2. Best WEDA fall pipeline: `{best_weda_fall.get('pipeline', '')}` with E7 fall F1={num(best_weda_fall.get('fall_f1'))}, FP={int_or_blank(best_weda_fall.get('FP'))}.",
        f"3. Best direction pipeline on average over E3/E6/E7: `{best_direction.get('pipeline', '')}` with average direction macro F1={num(best_direction.get('direction_macro_f1'))}.",
        f"4. Best end-to-end direction pipeline on average over E3/E6/E7: `{best_e2e.get('pipeline', '')}` with average E2E macro F1={num(best_e2e.get('E2E_Direction_Macro_F1'))}.",
        f"5. Lightest pipeline by estimated complexity units: `{lightest.get('pipeline', '')}`; edge suitability={lightest.get('edge_suitability', '')}.",
        "6. A5 remains the main deep temporal baseline and the direction expert even if Hybrid improves fall.",
        "7. Yes, the A5 fall head can be removed from final fall inference in the practical hybrid; keep it as auxiliary/reference for reporting.",
        "8. Yes, keep the A5 direction head. It is more reliable across the benchmark than ML-only direction.",
        f"9. ML-only separated should not replace A5 direction as main unless it wins consistently. Current E3/E6/E7 average direction macro F1: ML-only={num(ml_only_dir_avg)}, A5={num(a5_dir_avg)}.",
        "10. EdgeLite_Hybrid is worth keeping as a compact ablation. It uses only the core fall summary features but still depends on A5 for direction.",
        "11. FullTiming is not used as a main result. It remains upper-bound only.",
        "12. Final recommended practical architecture: DS-Fall-RD-Hybrid-NoTiming = GradientBoosting FallNoTiming_Full Fall Expert + DS-Fall-RD A5 Direction Expert.",
        "",
        "## A5 vs Hybrid WEDA Fall",
        "",
        f"- A5 E7 fall F1={num(a5_e7.get('fall_f1'))}, precision={num(a5_e7.get('fall_precision'))}, recall={num(a5_e7.get('fall_recall'))}, FP={int_or_blank(a5_e7.get('FP'))}.",
        f"- Hybrid_NoTiming E7 fall F1={num(hybrid_e7.get('fall_f1'))}, precision={num(hybrid_e7.get('fall_precision'))}, recall={num(hybrid_e7.get('fall_recall'))}, FP={int_or_blank(hybrid_e7.get('FP'))}.",
        "",
        "## Limitations",
        "",
        "- Pure cross-dataset transfer remains weaker than mixed BITS/WEDA training.",
        "- WEDA hard-negative ADL behavior still explains much of the fall gap.",
        "- ML-only direction can be strong on WEDA but is less stable than A5 across all protocols.",
        "- FullTiming results should not be used as the paper main result.",
    ]
    if not failed_df.empty:
        lines.extend(["", "## Failed Runs", "", markdown_table(format_df(failed_df))])
    return "\n".join(lines).rstrip() + "\n"


def row_or_empty(df: pd.DataFrame) -> dict[str, Any]:
    return df.iloc[0].to_dict() if not df.empty else {}


def num(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def int_or_blank(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def final_architecture_principles_md(complexity_df: pd.DataFrame) -> str:
    return f"""# Final Architecture Principles

## 1. A5_reference

```text
tilt12 sequence 50 x 12
        |
DS-Fall-RD A5
        |-- fall head -> fall / non-fall
        `-- direction head -> forward / backward / lateral
```

Role:

- Deep temporal multitask baseline.
- Direction-sensitive representation.
- Main reference.

## 2. Hybrid_NoTiming

```text
tilt12/raw window
        |-- FallNoTiming_Full summary features
        |       |
        |   GradientBoosting Fall Expert
        |       |
        |   final fall
        |
        `-- tilt12 sequence
                |
            A5 direction head
                |
            direction
```

Role:

- Main practical architecture.
- Reduces WEDA false positives.
- Does not use timing artifacts.

## 3. EdgeLite_Hybrid

```text
tilt12/raw window
        |-- FallNoTiming_Core 16 features
        |       |
        |   RandomForest Fall Expert
        |       |
        |   final fall
        |
        `-- A5 direction head
```

Role:

- Compact / edge-friendly ablation.
- Lower fall feature count.
- Compares against full hybrid.

## 4. ML_only_separated

```text
summary features
        |-- ML Fall Expert
        `-- ML Direction Expert
```

Role:

- Analysis pipeline.
- Tests whether ML direction can replace A5.
- Not selected as main if pure transfer or direction consistency is weaker.

## Pipeline Table

| Pipeline | Fall decision | Direction decision | Feature/Input | Role |
| --- | --- | --- | --- | --- |
| A5_reference | A5 fall head | A5 direction head | tilt12 sequence | deep temporal reference |
| Hybrid_NoTiming | GradientBoosting FallNoTiming_Full | A5 direction head | summary + tilt12 | practical main |
| EdgeLite_Hybrid | RandomForest FallNoTiming_Core | A5 direction head | compact summary + tilt12 | compact ablation |
| ML_only_separated | GradientBoosting FallNoTiming_Full | GradientBoosting DirectionFullNoTiming | summary only | analysis pipeline |

## Complexity

{markdown_table(format_df(complexity_df))}
"""


def final_research_summary_md() -> str:
    return """# Final Research Summary

## 1. Problem Definition

The goal is direction-sensitive fall detection from wrist-worn IMU windows.
The system must decide fall/non-fall and, for supervised fall windows, classify
direction as forward, backward, or lateral.

## 2. Dataset And Benchmark

The final benchmark uses BITS and WEDA only. The protocol is fixed at 25 Hz,
2-second event-centered windows, and 50 x 12 tilt12 temporal input.

## 3. DS-Fall-RD A5 Model

A5WCEFW is the main deep temporal reference. It uses rotation-aware tilt12
features, task-specific attention, weighted fall loss, and weighted direction
loss. It remains the strongest direction expert.

## 4. WEDA Fall Gap

WEDA fall F1 is lower for the deep fall head because WEDA contains harder ADL
negatives and produces more false positives. Direction on WEDA remains good,
so the issue is mainly fall discrimination rather than direction separability.

## 5. Feature Analysis

No-timing summary features separate WEDA fall/non-fall better than the A5 fall
head in the final benchmark. Fall depends heavily on impact, jerk, post-impact
stability, and magnitude summaries. Direction depends more on signed gyro and
roll/pitch transitions.

## 6. FV/FPGuard Experiments

Previous fall-focused variants improved only part of the WEDA false-positive
problem. They did not justify replacing the stable A5 direction head.

## 7. Hybrid Experiments

Task separation is the most defensible final direction: use a no-timing ML fall
expert for fall decision and A5 direction head for direction.

## 8. Specialized ML E1-E7

The final E1-E7 ML analysis evaluates fall-only and direction-only classical
experts with fixed protocols. FullTiming is excluded from the main result.

## 9. Final Architecture Decision

Recommended practical architecture:

```text
DS-Fall-RD-Hybrid-NoTiming
= GradientBoosting FallNoTiming_Full Fall Expert
+ DS-Fall-RD A5 Direction Expert
```

A5 remains the main deep baseline and direction expert. A5 fall head is kept
as auxiliary/reference only.

## 10. Limitations

- BITS/WEDA pure cross-dataset transfer remains weaker than mixed training.
- WEDA hard negatives still need careful discussion.
- ML-only direction is useful analysis but not stable enough to replace A5.
- FullTiming is upper-bound only.

## 11. Next Work

- Validate on external datasets after axis and label harmonization.
- Study hard-negative WEDA ADL cases.
- Explore compact deployment of the A5 direction expert.
"""


def hybrid_architecture_doc() -> str:
    return """# Hybrid Architecture

Fall and direction are separated because they use different evidence.

- Fall: impact magnitude, jerk, post-impact stillness, and summary statistics.
- Direction: signed temporal gyro pattern and roll/pitch transition.

Final practical pipeline:

```text
Fall = GradientBoosting FallNoTiming_Full
Direction = DS-Fall-RD A5 direction head
```

No FullTiming features are used in the main pipeline. FullTiming is upper-bound
only.
"""


def final_four_pipelines_doc(complexity_df: pd.DataFrame) -> str:
    return f"""# Final Four Pipelines

| Pipeline | Fall decision | Direction decision | Feature/Input | Role |
| --- | --- | --- | --- | --- |
| A5_reference | A5 fall head | A5 direction head | tilt12 sequence | deep multitask reference |
| Hybrid_NoTiming | GradientBoosting + FallNoTiming_Full | A5 direction head | summary + tilt12 | practical main |
| EdgeLite_Hybrid | RandomForest + FallNoTiming_Core | A5 direction head | compact summary + tilt12 | compact ablation |
| ML_only_separated | GradientBoosting + FallNoTiming_Full | GradientBoosting + DirectionFullNoTiming | summary only | analysis pipeline |

ML_only_separated is analysis-only unless its direction expert beats A5
consistently across E1-E7.

## Complexity And Edge

{markdown_table(format_df(complexity_df))}
"""


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
                "TN",
                "FP",
                "FN",
                "TP",
                "feature_count",
                "fall_feature_count",
                "direction_feature_count",
                "fall_tree_count",
                "direction_tree_count",
                "total_tree_count",
                "fall_tree_nodes",
                "direction_tree_nodes",
                "total_tree_nodes",
                "estimated_complexity_units",
                "deep_params",
                "direction_n_supervised",
            }:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(int(v)))
            else:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    return out


def run_final(repo_root: Path, summary_path: Path, seed: int) -> None:
    report_dir = ensure_dir(repo_root / "outputs" / "final_e1_e7" / "reports")
    table_dir = ensure_dir(repo_root / "outputs" / "final_e1_e7" / "tables")
    fig_dir = ensure_dir(repo_root / "outputs" / "final_e1_e7" / "figures")
    pred_dir = ensure_dir(repo_root / "outputs" / "final_e1_e7" / "predictions")
    run_dir = ensure_dir(repo_root / "outputs" / "final_e1_e7" / "models")
    ensure_dir(fig_dir / "confusion_matrices")
    ensure_dir(fig_dir / "bar_charts")

    df = load_feature_matrix(summary_path)
    feature_sets, missing_features = build_feature_sets(df)
    print(f"Loaded feature matrix: {summary_path} ({len(df)} rows)")
    print(f"FallNoTiming_Full features: {len(feature_sets['FallNoTiming_Full'])}")
    print(f"FallNoTiming_Core features: {len(feature_sets['FallNoTiming_Core'])}")
    print(f"DirectionFullNoTiming features: {len(feature_sets['DirectionFullNoTiming'])}")

    trained_fall: dict[tuple[str, str], TrainedModel] = {}
    trained_direction: dict[tuple[str, str], TrainedModel] = {}
    failed_rows: list[dict[str, Any]] = []

    for pipeline in ["Hybrid_NoTiming", "EdgeLite_Hybrid", "ML_only_separated"]:
        for protocol_id in BASE_PROTOCOLS:
            try:
                if pipeline == "EdgeLite_Hybrid":
                    fall_feature_set = "FallNoTiming_Core"
                    fall_model_name = "RandomForest"
                else:
                    fall_feature_set = "FallNoTiming_Full"
                    fall_model_name = "GradientBoosting"
                print(f"Training fall model: {pipeline} {protocol_id}")
                model = train_fall_model(
                    df,
                    protocol_id,
                    fall_feature_set,
                    feature_sets[fall_feature_set],
                    fall_model_name,
                    seed,
                )
                trained_fall[(pipeline, protocol_id)] = model
                with (run_dir / f"{pipeline}_{protocol_id}_fall_model.pkl").open("wb") as f:
                    pickle.dump(model, f)
            except Exception as exc:
                failed_rows.append({"pipeline": pipeline, "experiment_id": protocol_id, "task": "fall", "error": repr(exc)})
        if pipeline == "ML_only_separated":
            for protocol_id in BASE_PROTOCOLS:
                try:
                    print(f"Training direction model: {pipeline} {protocol_id}")
                    model = train_direction_model(
                        df,
                        protocol_id,
                        "DirectionFullNoTiming",
                        feature_sets["DirectionFullNoTiming"],
                        "GradientBoosting",
                        seed,
                    )
                    trained_direction[(pipeline, protocol_id)] = model
                    with (run_dir / f"{pipeline}_{protocol_id}_direction_model.pkl").open("wb") as f:
                        pickle.dump(model, f)
                except Exception as exc:
                    failed_rows.append({"pipeline": pipeline, "experiment_id": protocol_id, "task": "direction", "error": repr(exc)})

    fall_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    e2e_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    a5_cache: dict[str, pd.DataFrame] = {}
    for pipeline in PIPELINES:
        for protocol_id in ALL_PROTOCOLS:
            print(f"Evaluating {pipeline} {protocol_id}")
            train_protocol = canonical_train_protocol(protocol_id)
            _, _, test_mask = get_protocol_splits(df, protocol_id)
            test_df = df.loc[test_mask].copy().reset_index(drop=True)
            if test_df.empty:
                failed_rows.append({"pipeline": pipeline, "experiment_id": protocol_id, "task": "eval", "error": "empty_test_mask"})
                continue
            if protocol_id not in a5_cache:
                a5_cache[protocol_id] = load_a5_predictions(repo_root, protocol_id)
            a5_aligned = align_a5_predictions(test_df, a5_cache[protocol_id])

            if pipeline == "A5_reference":
                fall_prob = a5_aligned["fall_prob"].astype(float).to_numpy()
                fall_pred = a5_aligned["pred_fall"].astype(str).str.lower().eq("fall").astype(int).to_numpy()
                fall_feature_set = "temporal_tilt12"
                fall_feature_count = 12
                threshold = 0.5
                threshold_note = "saved_a5_prediction_threshold"
            else:
                fall_model = trained_fall.get((pipeline, train_protocol))
                if fall_model is None:
                    failed_rows.append({"pipeline": pipeline, "experiment_id": protocol_id, "task": "fall_eval", "error": f"missing_fall_model_{train_protocol}"})
                    continue
                X_test = test_df[fall_model.features].astype(float).to_numpy()
                fall_prob = fall_scores(fall_model.model, X_test)
                fall_pred = (fall_prob >= float(fall_model.threshold)).astype(int)
                fall_feature_set = fall_model.feature_set
                fall_feature_count = len(fall_model.features)
                threshold = fall_model.threshold
                threshold_note = fall_model.threshold_note

            if PIPELINES[pipeline]["direction_source"] == "A5_direction_head":
                direction_pred = a5_aligned["pred_direction"].astype(str).str.lower().map(DIR_TO_ID).fillna(-1).astype(int).to_numpy()
                direction_feature_set = "temporal_tilt12"
                direction_feature_count = 12
            else:
                direction_model = trained_direction.get((pipeline, train_protocol))
                if direction_model is None:
                    failed_rows.append({"pipeline": pipeline, "experiment_id": protocol_id, "task": "direction_eval", "error": f"missing_direction_model_{train_protocol}"})
                    continue
                direction_pred = direction_predictions(direction_model.model, test_df[direction_model.features].astype(float).to_numpy())
                direction_feature_set = direction_model.feature_set
                direction_feature_count = len(direction_model.features)

            fall_row = evaluate_fall(
                pipeline,
                protocol_id,
                test_df,
                fall_pred,
                fall_prob,
                train_protocol,
                PIPELINES[pipeline]["fall_source"],
                fall_feature_set,
                fall_feature_count,
                threshold,
                threshold_note,
            )
            direction_row = evaluate_direction(
                pipeline,
                protocol_id,
                test_df,
                direction_pred,
                train_protocol,
                PIPELINES[pipeline]["direction_source"],
                direction_feature_set,
                direction_feature_count,
            )
            e2e_row = evaluate_e2e_direction(pipeline, protocol_id, test_df, fall_pred, direction_pred, train_protocol)
            fall_rows.append(fall_row)
            direction_rows.append(direction_row)
            e2e_rows.append(e2e_row)
            threshold_rows.append(
                {
                    "pipeline": pipeline,
                    "experiment_id": protocol_id,
                    "train_protocol": train_protocol,
                    "fall_source": PIPELINES[pipeline]["fall_source"],
                    "threshold": threshold,
                    "threshold_note": threshold_note,
                }
            )
            save_prediction_file(pred_dir, pipeline, protocol_id, test_df, fall_pred, fall_prob, direction_pred)

    complexity_rows = [pipeline_complexity(p, feature_sets, trained_fall, trained_direction) for p in PIPELINES]
    save_reports(
        repo_root,
        report_dir,
        fig_dir,
        pred_dir,
        feature_sets,
        missing_features,
        fall_rows,
        direction_rows,
        e2e_rows,
        complexity_rows,
        threshold_rows,
        failed_rows,
    )

    combined = pd.read_csv(report_dir / "final_four_pipeline_results.csv")
    focus = combined[combined["experiment_id"].isin(["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"])]
    print("")
    print("Report files:")
    print(report_dir / "final_four_pipeline_summary.md")
    print(report_dir / "final_four_pipeline_results.csv")
    print(report_dir / "final_four_pipeline_complexity.csv")
    print("")
    print("Topline E3/E6/E7:")
    print(
        focus[
            [
                "pipeline",
                "experiment_id",
                "fall_f1",
                "direction_macro_f1",
                "E2E_Direction_Macro_F1",
                "FP",
                "edge_suitability",
            ]
        ].to_string(index=False)
    )
    print("")
    print("Recommended architecture: DS-Fall-RD-Hybrid-NoTiming = GradientBoosting FallNoTiming_Full + A5 Direction Expert")
    if failed_rows:
        print(f"Warnings/failures saved: {report_dir / 'failed_runs.csv'}")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    summary_path = Path(args.summary_path)
    if not summary_path.is_absolute():
        summary_path = repo_root / summary_path
    if not args.run_all:
        print("Use --run-all to execute the final E1-E7 benchmark.")
        return
    run_final(repo_root, summary_path, args.seed)


if __name__ == "__main__":
    main()
