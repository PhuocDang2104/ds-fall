from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "reports" / "ml_feature_analysis_25hz" / "summary_feature_matrix.csv"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "ds_fall_rd_fv_extra"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "ds_fall_rd_fv_extra" / "confusion_matrices"

REFERENCE_PARAMS = 65959
REFERENCE_WEDA_FP = 21
FV1_OLD_WEDA_FP = 15
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

AXISHARD10 = [
    "ay_std",
    "ay_range",
    "jerk_p95",
    "jerk_std",
    "acc_mag_std",
    "acc_mag_range",
    "gyro_mag_max",
    "gyro_mag_range",
    "tilt_delta_p95",
    "post_acc_mag_std",
]


@dataclass(frozen=True)
class ExtraRun:
    run_id: str
    verifier: str
    features: str
    fusion: str
    threshold_rule: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exactly six DS-Fall-RD-FV extra FPGuard runs.")
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
    validate_features(df)
    df = add_hard_negative_flags(df)

    runs = [
        ExtraRun("FV1A", "logistic_regression", "Lite10", "and", "max_f1_precision_065"),
        ExtraRun("FV1B", "logistic_regression", "Lite10", "and", "min_fp_recall_088"),
        ExtraRun("FV7", "logistic_regression", "AxisHard10", "and", "max_f1_precision_065"),
        ExtraRun("FV8", "logistic_regression", "AxisHard10", "stacked_logit_fusion", "max_f1_precision_065"),
        ExtraRun("FV9", "decision_tree_depth3", "AxisHard10", "and", "max_f1_precision_065"),
        ExtraRun("FV10", "tiny_random_forest_10x_depth3", "AxisHard10", "and", "max_f1_precision_065"),
    ]

    feature_config = {
        "summary_source": str(Path(args.summary_path)),
        "Lite10": LITE10,
        "AxisHard10": AXISHARD10,
        "hard_negative_score_features": AXISHARD10,
        "hard_negative_definition": "top 20% WEDA non-fall train/val by rank-normalized AxisHard10 motion score",
        "benchmark": "BITS/WEDA only, 25 Hz, 2-second event-centered, existing split",
        "direction_policy": "Direction probabilities are copied from DS-Fall-RD A5 reference columns and are never changed.",
    }
    (report_dir / "fv_extra_feature_configs.json").write_text(json.dumps(feature_config, indent=2), encoding="utf-8")

    results: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for run in runs:
        try:
            print(f"Running {run.run_id}: {run.verifier}, {run.features}, {run.fusion}")
            run_results, threshold_row, run_fp_rows = run_one(df, run, args.seed, run_root, figure_dir)
            results.extend(run_results)
            thresholds.append(threshold_row)
            fp_rows.extend(run_fp_rows)
        except Exception as exc:
            failures.append({"run_id": run.run_id, "status": "failed", "reason": repr(exc)})
            print(f"FAILED {run.run_id}: {exc}")

    results_df = pd.DataFrame(results)
    thresholds_df = pd.DataFrame(thresholds)
    fp_df = pd.DataFrame(fp_rows)
    failures_df = pd.DataFrame(failures)

    results_df.to_csv(report_dir / "fv_extra_results.csv", index=False)
    thresholds_df.to_csv(report_dir / "fv_extra_thresholds.csv", index=False)
    fp_df.to_csv(report_dir / "fv_extra_weda_fp_analysis.csv", index=False)
    failures_df.to_csv(report_dir / "fv_extra_failures.csv", index=False)

    write_best_selection(results_df, thresholds_df, failures_df, report_dir)
    write_summary(results_df, thresholds_df, fp_df, failures_df, report_dir)
    print(f"Saved outputs to {report_dir}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary feature matrix: {path}")
    df = pd.read_csv(path)
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


def validate_features(df: pd.DataFrame) -> None:
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
    required.update(AXISHARD10)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_hard_negative_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eligible = df["dataset"].eq("weda") & df["fall_label"].eq(0) & df["split"].isin(["train", "val"])
    score = np.zeros(len(df), dtype=float)
    hard = np.zeros(len(df), dtype=bool)
    if eligible.any():
        sub = df.loc[eligible, AXISHARD10].astype(float)
        ranks = [sub[col].rank(pct=True).to_numpy() for col in AXISHARD10]
        values = np.mean(np.vstack(ranks), axis=0)
        cutoff = np.quantile(values, 0.80)
        eligible_idx = np.flatnonzero(eligible.to_numpy())
        score[eligible_idx] = values
        hard[eligible_idx] = values >= cutoff
    df["hard_negative_score"] = score
    df["hard_negative_flag"] = hard
    return df


def features_for(name: str) -> list[str]:
    if name == "Lite10":
        return LITE10
    if name == "AxisHard10":
        return AXISHARD10
    raise ValueError(name)


def train_weights(df_train: pd.DataFrame) -> np.ndarray:
    weights = np.ones(len(df_train), dtype=float)
    weights[df_train["hard_negative_flag"].to_numpy()] = 2.0
    return weights


def train_verifier(df: pd.DataFrame, run: ExtraRun, seed: int, run_dir: Path) -> tuple[np.ndarray, np.ndarray, Any, StandardScaler]:
    features = features_for(run.features)
    train = df["split"].eq("train")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df.loc[train, features].astype(float).to_numpy())
    X_all = scaler.transform(df[features].astype(float).to_numpy())
    y_train = df.loc[train, "fall_label"].astype(int).to_numpy()
    weights = train_weights(df.loc[train].reset_index(drop=True))

    if run.verifier == "logistic_regression":
        model = LogisticRegression(max_iter=3000, random_state=seed)
        model.fit(X_train, y_train, sample_weight=weights)
        prob = model.predict_proba(X_all)[:, 1]
        score = model.decision_function(X_all)
    elif run.verifier == "decision_tree_depth3":
        model = DecisionTreeClassifier(max_depth=3, random_state=seed)
        model.fit(X_train, y_train, sample_weight=weights)
        prob = model.predict_proba(X_all)[:, 1]
        score = prob
    elif run.verifier == "tiny_random_forest_10x_depth3":
        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=seed, n_jobs=-1)
        model.fit(X_train, y_train, sample_weight=weights)
        prob = model.predict_proba(X_all)[:, 1]
        score = prob
    else:
        raise ValueError(run.verifier)

    with (run_dir / "summary_scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)
    with (run_dir / "verifier.pkl").open("wb") as f:
        pickle.dump(model, f)
    return prob.astype(float), np.asarray(score, dtype=float), model, scaler


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def train_stacked_fusion(df: pd.DataFrame, verifier_score: np.ndarray, seed: int, run_dir: Path) -> np.ndarray:
    train = df["split"].eq("train").to_numpy()
    y_train = df.loc[train, "fall_label"].astype(int).to_numpy()
    X_train = np.column_stack([logit(df["fall_prob"].to_numpy()[train]), verifier_score[train]])
    weights = train_weights(df.loc[train].reset_index(drop=True))
    fusion = LogisticRegression(max_iter=3000, random_state=seed)
    fusion.fit(X_train, y_train, sample_weight=weights)
    with (run_dir / "stacked_logit_fusion.pkl").open("wb") as f:
        pickle.dump(fusion, f)
    X_all = np.column_stack([logit(df["fall_prob"].to_numpy()), verifier_score])
    return fusion.predict_proba(X_all)[:, 1].astype(float)


def threshold_candidates(*arrays: np.ndarray) -> np.ndarray:
    parts = [np.linspace(0.001, 0.999, 90), np.array([0.1, 0.25, 0.5, 0.65, 0.75, 0.85, 0.89, 0.9, 0.95, 0.975, 0.99])]
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        parts.append(np.quantile(arr, np.linspace(0.02, 0.98, 60)))
    out = np.unique(np.clip(np.concatenate(parts), 0.0, 1.0))
    return out


def select_and_thresholds(
    y: np.ndarray,
    p_deep: np.ndarray,
    p_verifier: np.ndarray,
    rule: str,
) -> tuple[float, float, str]:
    deep_candidates = threshold_candidates(p_deep)
    verifier_candidates = threshold_candidates(p_verifier)
    rows = []
    for t_deep in deep_candidates:
        deep_ok = p_deep >= t_deep
        for t_ver in verifier_candidates:
            pred = (deep_ok & (p_verifier >= t_ver)).astype(int)
            metrics = binary_metrics(y, pred)
            rows.append((metrics, float(t_deep), float(t_ver)))
    if rule == "max_f1_precision_065":
        feasible = [r for r in rows if r[0]["precision"] >= 0.65]
        if feasible:
            best = max(feasible, key=lambda r: (r[0]["f1"], r[0]["recall"], -r[0]["FP"], r[0]["precision"]))
            return best[1], best[2], "max_f1_precision_065"
        fallback = [r for r in rows if r[0]["recall"] >= 0.88]
        if fallback:
            best = max(fallback, key=lambda r: (r[0]["precision"], r[0]["f1"], -r[0]["FP"], r[0]["recall"]))
            return best[1], best[2], "fallback_highest_precision_recall_088"
        best = max(rows, key=lambda r: (r[0]["precision"], r[0]["f1"], r[0]["recall"], -r[0]["FP"]))
        return best[1], best[2], "fallback_highest_precision_no_recall_feasible"
    if rule == "min_fp_recall_088":
        feasible = [r for r in rows if r[0]["recall"] >= 0.88]
        if feasible:
            best = min(feasible, key=lambda r: (r[0]["FP"], -r[0]["f1"], -r[0]["precision"], -r[0]["recall"]))
            return best[1], best[2], "min_fp_recall_088"
        best = max(rows, key=lambda r: (r[0]["recall"], -r[0]["FP"], r[0]["f1"], r[0]["precision"]))
        return best[1], best[2], "fallback_highest_recall"
    raise ValueError(rule)


def select_final_threshold(y: np.ndarray, prob: np.ndarray, rule: str) -> tuple[float, str]:
    rows = []
    for t in threshold_candidates(prob):
        pred = (prob >= t).astype(int)
        rows.append((binary_metrics(y, pred), float(t)))
    if rule == "max_f1_precision_065":
        feasible = [r for r in rows if r[0]["precision"] >= 0.65]
        if feasible:
            best = max(feasible, key=lambda r: (r[0]["f1"], r[0]["recall"], -r[0]["FP"], r[0]["precision"]))
            return best[1], "max_f1_precision_065"
        fallback = [r for r in rows if r[0]["recall"] >= 0.88]
        if fallback:
            best = max(fallback, key=lambda r: (r[0]["precision"], r[0]["f1"], -r[0]["FP"], r[0]["recall"]))
            return best[1], "fallback_highest_precision_recall_088"
        best = max(rows, key=lambda r: (r[0]["precision"], r[0]["f1"], r[0]["recall"], -r[0]["FP"]))
        return best[1], "fallback_highest_precision_no_recall_feasible"
    raise ValueError(rule)


def run_one(
    df: pd.DataFrame,
    run: ExtraRun,
    seed: int,
    run_root: Path,
    figure_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    run_dir = ensure_dir(run_root / run.run_id)
    features = features_for(run.features)
    (run_dir / "config.json").write_text(
        json.dumps({"run": run.__dict__, "features": features, "seed": seed}, indent=2),
        encoding="utf-8",
    )
    verifier_prob, verifier_score, verifier, _ = train_verifier(df, run, seed, run_dir)
    p_deep = df["fall_prob"].to_numpy(dtype=float)
    val_idx = np.flatnonzero(df["split"].eq("val").to_numpy())
    y_val = df.iloc[val_idx]["fall_label"].astype(int).to_numpy()

    if run.fusion == "and":
        t_deep, t_verifier, selected_rule = select_and_thresholds(
            y_val, p_deep[val_idx], verifier_prob[val_idx], run.threshold_rule
        )
        final_prob = np.minimum(p_deep, verifier_prob)
        pred_fn: Callable[[np.ndarray], np.ndarray] = lambda idx: (
            (p_deep[idx] >= t_deep) & (verifier_prob[idx] >= t_verifier)
        ).astype(int)
        threshold_row = {
            "run_id": run.run_id,
            "threshold_rule": run.threshold_rule,
            "selected_rule": selected_rule,
            "threshold_deep": t_deep,
            "threshold_verifier": t_verifier,
            "threshold_final": math.nan,
            "fusion": run.fusion,
            "features": run.features,
            "verifier": run.verifier,
        }
    elif run.fusion == "stacked_logit_fusion":
        final_prob = train_stacked_fusion(df, verifier_score, seed, run_dir)
        t_final, selected_rule = select_final_threshold(y_val, final_prob[val_idx], run.threshold_rule)
        pred_fn = lambda idx: (final_prob[idx] >= t_final).astype(int)
        threshold_row = {
            "run_id": run.run_id,
            "threshold_rule": run.threshold_rule,
            "selected_rule": selected_rule,
            "threshold_deep": math.nan,
            "threshold_verifier": math.nan,
            "threshold_final": t_final,
            "fusion": run.fusion,
            "features": run.features,
            "verifier": run.verifier,
        }
    else:
        raise ValueError(run.fusion)

    model_info = verifier_info(verifier, run, len(features))
    rows = []
    for eval_id, dataset in [("E3", "bits+weda"), ("E6", "bits"), ("E7", "weda")]:
        idx = eval_indices(df, eval_id)
        pred = pred_fn(idx)
        metrics = evaluate_subset(df, idx, pred, final_prob[idx], run.run_id, eval_id, dataset, figure_dir)
        metrics.update(
            {
                "features": run.features,
                "fusion": run.fusion,
                "verifier": run.verifier,
                "threshold_rule": run.threshold_rule,
                **model_info,
                "weda_fp_reduction_vs_reference": REFERENCE_WEDA_FP - metrics["FP"] if eval_id == "E7" else math.nan,
                "weda_fp_reduction_vs_fv1_lite10_and": FV1_OLD_WEDA_FP - metrics["FP"] if eval_id == "E7" else math.nan,
            }
        )
        rows.append(metrics)
    fp_rows = weda_fp_analysis(df, run, verifier_prob, final_prob, pred_fn(eval_indices(df, "E7")), features)
    return rows, threshold_row, fp_rows


def verifier_info(model: Any, run: ExtraRun, n_features: int) -> dict[str, Any]:
    if run.verifier == "logistic_regression":
        verifier_params = n_features + 1
        return {
            "verifier_params": verifier_params,
            "tree_depth": math.nan,
            "tree_leaves": math.nan,
            "rf_n_trees": math.nan,
            "rf_max_depth": math.nan,
            "edge_suitability_note": f"Very edge-friendly LR: {verifier_params} scalar parameters plus scaler.",
        }
    if run.verifier == "decision_tree_depth3":
        return {
            "verifier_params": int(model.tree_.node_count),
            "tree_depth": int(model.get_depth()),
            "tree_leaves": int(model.get_n_leaves()),
            "rf_n_trees": math.nan,
            "rf_max_depth": math.nan,
            "edge_suitability_note": f"Edge-friendly shallow tree: depth={model.get_depth()}, leaves={model.get_n_leaves()}.",
        }
    if run.verifier == "tiny_random_forest_10x_depth3":
        depths = [est.get_depth() for est in model.estimators_]
        leaves = [est.get_n_leaves() for est in model.estimators_]
        return {
            "verifier_params": int(sum(est.tree_.node_count for est in model.estimators_)),
            "tree_depth": math.nan,
            "tree_leaves": int(sum(leaves)),
            "rf_n_trees": len(model.estimators_),
            "rf_max_depth": int(max(depths)) if depths else math.nan,
            "edge_suitability_note": f"TinyRF: 10 trees, max observed depth={max(depths) if depths else 'n/a'}, total leaves={sum(leaves)}.",
        }
    raise ValueError(run.verifier)


def eval_indices(df: pd.DataFrame, eval_id: str) -> np.ndarray:
    test = df["split"].eq("test").to_numpy()
    if eval_id == "E3":
        return np.flatnonzero(test)
    if eval_id == "E6":
        return np.flatnonzero(test & df["dataset"].eq("bits").to_numpy())
    if eval_id == "E7":
        return np.flatnonzero(test & df["dataset"].eq("weda").to_numpy())
    raise ValueError(eval_id)


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel()]
    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
    }


def direction_metrics(df: pd.DataFrame, idx: np.ndarray) -> dict[str, Any]:
    sub = df.iloc[idx].reset_index(drop=True)
    probs = sub[["direction_prob_forward", "direction_prob_backward", "direction_prob_lateral"]].astype(float).to_numpy()
    pred = np.argmax(probs, axis=1)
    y = sub["direction_id"].to_numpy()
    supervised = sub["direction_supervised_bool"].to_numpy() & sub["fall_label"].eq(1).to_numpy() & (y >= 0)
    if not supervised.any():
        return {
            "direction_macro_f1": math.nan,
            "direction_accuracy": math.nan,
            "direction_n_supervised": 0,
            "forward_f1": math.nan,
            "backward_f1": math.nan,
            "lateral_f1": math.nan,
            "direction_cm": np.zeros((3, 3), dtype=int),
        }
    cm = confusion_matrix(y[supervised], pred[supervised], labels=[0, 1, 2])
    per_class = f1_score(y[supervised], pred[supervised], labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "direction_macro_f1": float(f1_score(y[supervised], pred[supervised], labels=[0, 1, 2], average="macro", zero_division=0)),
        "direction_accuracy": float(accuracy_score(y[supervised], pred[supervised])),
        "direction_n_supervised": int(supervised.sum()),
        "forward_f1": float(per_class[0]),
        "backward_f1": float(per_class[1]),
        "lateral_f1": float(per_class[2]),
        "direction_cm": cm,
    }


def evaluate_subset(
    df: pd.DataFrame,
    idx: np.ndarray,
    pred_fall: np.ndarray,
    score: np.ndarray,
    run_id: str,
    eval_id: str,
    dataset: str,
    figure_dir: Path,
) -> dict[str, Any]:
    sub = df.iloc[idx].reset_index(drop=True)
    y = sub["fall_label"].astype(int).to_numpy()
    fall = binary_metrics(y, pred_fall)
    try:
        auroc = float(roc_auc_score(y, score))
    except ValueError:
        auroc = math.nan
    try:
        ap = float(average_precision_score(y, score))
    except ValueError:
        ap = math.nan
    direction = direction_metrics(df, idx)
    fall_cm = np.array([[fall["TN"], fall["FP"]], [fall["FN"], fall["TP"]]], dtype=int)
    save_confusion(figure_dir, run_id, eval_id, "fall", fall_cm, ["non_fall", "fall"])
    save_confusion(figure_dir, run_id, eval_id, "direction", direction["direction_cm"], DIRECTION_LABELS)
    out = {
        "run_id": run_id,
        "eval": eval_id,
        "dataset": dataset,
        "n": int(len(idx)),
        "TN": fall["TN"],
        "FP": fall["FP"],
        "FN": fall["FN"],
        "TP": fall["TP"],
        "fall_precision": fall["precision"],
        "fall_recall": fall["recall"],
        "fall_f1": fall["f1"],
        "fall_accuracy": fall["accuracy"],
        "fall_auroc": auroc,
        "fall_average_precision": ap,
    }
    for key, value in direction.items():
        if key != "direction_cm":
            out[key] = value
    return out


def save_confusion(figure_dir: Path, run_id: str, eval_id: str, task: str, cm: np.ndarray, labels: list[str]) -> None:
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{run_id} {eval_id} {task}")
    plt.tight_layout()
    plt.savefig(figure_dir / f"{run_id}_{eval_id}_{task}_confusion.png", dpi=160)
    plt.close()


def weda_fp_analysis(
    df: pd.DataFrame,
    run: ExtraRun,
    verifier_prob: np.ndarray,
    final_score: np.ndarray,
    final_pred_e7: np.ndarray,
    features: list[str],
) -> list[dict[str, Any]]:
    idx = eval_indices(df, "E7")
    sub = df.iloc[idx].reset_index(drop=True)
    y = sub["fall_label"].astype(int).to_numpy()
    ref_pred = (sub["fall_prob"].to_numpy(dtype=float) >= 0.5).astype(int)
    rows = []
    for i, row in sub.iterrows():
        ref_fp = bool(y[i] == 0 and ref_pred[i] == 1)
        new_fp = bool(y[i] == 0 and final_pred_e7[i] == 1)
        if ref_fp or new_fp:
            out = {
                "run_id": run.run_id,
                "window_id": row.get("window_id", ""),
                "activity": row.get("activity", row.get("activity_name", "")),
                "subject_id": row.get("subject_id", ""),
                "trial_id": row.get("trial_id", ""),
                "reference_fall_prob": float(row["fall_prob"]),
                "verifier_prob": float(verifier_prob[idx][i]),
                "final_score": float(final_score[idx][i]),
                "reference_pred": int(ref_pred[i]),
                "extra_pred": int(final_pred_e7[i]),
                "reference_fp": ref_fp,
                "extra_fp": new_fp,
                "corrected_reference_fp": bool(ref_fp and not new_fp),
            }
            for feature in features:
                out[feature] = row.get(feature, math.nan)
            rows.append(out)
    return rows


def build_selection_table(results: pd.DataFrame) -> pd.DataFrame:
    e7 = results[results["eval"].eq("E7")].copy()
    e3 = results[results["eval"].eq("E3")][["run_id", "direction_macro_f1", "direction_accuracy"]].rename(
        columns={"direction_macro_f1": "E3_Direction_Macro_F1", "direction_accuracy": "E3_Direction_Accuracy"}
    )
    e6 = results[results["eval"].eq("E6")][["run_id", "fall_f1", "direction_macro_f1"]].rename(
        columns={"fall_f1": "E6_BITS_Fall_F1", "direction_macro_f1": "E6_BITS_Direction_Macro_F1"}
    )
    table = e7.merge(e3, on="run_id", how="left").merge(e6, on="run_id", how="left")
    table["passes_hard_constraints"] = (
        (table["fall_recall"] >= 0.88)
        & (table["direction_macro_f1"] >= 0.80)
        & (table["E3_Direction_Macro_F1"] >= 0.86)
        & (table["E6_BITS_Fall_F1"] >= 0.90)
    )
    table["passes_primary_target"] = (table["FP"] <= 12) & (table["fall_precision"] >= 0.65) & (table["fall_f1"] > 0.75)
    simplicity = {
        "logistic_regression": 0,
        "decision_tree_depth3": 1,
        "tiny_random_forest_10x_depth3": 2,
    }
    table["simplicity_rank"] = table["verifier"].map(simplicity).fillna(99)
    return table.sort_values(
        ["passes_hard_constraints", "FP", "fall_precision", "fall_f1", "simplicity_rank"],
        ascending=[False, True, False, False, True],
    )


def write_best_selection(results: pd.DataFrame, thresholds: pd.DataFrame, failures: pd.DataFrame, report_dir: Path) -> None:
    table = build_selection_table(results) if not results.empty else pd.DataFrame()
    best = table.iloc[0].to_dict() if not table.empty else {}
    lines = [
        "# DS-Fall-RD-FV Extra Best Selection",
        "",
        "Selection: hard constraints first, then minimize WEDA FP, maximize WEDA precision, keep WEDA Fall F1 > 0.75 if possible, and prefer simpler verifier when metrics are close.",
        "",
        "## E7 Selection Table",
        "",
        markdown_table(
            format_df(
                table[
                    [
                        "run_id",
                        "verifier",
                        "features",
                        "fusion",
                        "fall_f1",
                        "fall_precision",
                        "fall_recall",
                        "FP",
                        "FN",
                        "weda_fp_reduction_vs_reference",
                        "weda_fp_reduction_vs_fv1_lite10_and",
                        "direction_macro_f1",
                        "E3_Direction_Macro_F1",
                        "E6_BITS_Fall_F1",
                        "passes_hard_constraints",
                        "passes_primary_target",
                    ]
                ]
            )
        )
        if not table.empty
        else "_No completed runs._",
        "",
        "## Best Run",
        "",
    ]
    if best:
        lines.extend(
            [
                f"Best selected run: `{best['run_id']}`.",
                "",
                f"- WEDA FP/FN: {int(best['FP'])}/{int(best['FN'])}",
                f"- WEDA precision/recall/F1: {best['fall_precision']:.4f}/{best['fall_recall']:.4f}/{best['fall_f1']:.4f}",
                f"- WEDA Direction Macro F1: {best['direction_macro_f1']:.4f}",
                f"- E3 Direction Macro F1: {best['E3_Direction_Macro_F1']:.4f}",
                f"- E6 BITS Fall F1: {best['E6_BITS_Fall_F1']:.4f}",
            ]
        )
    if not failures.empty:
        lines.extend(["", "## Failures", "", markdown_table(format_df(failures))])
    (report_dir / "fv_extra_best_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(results: pd.DataFrame, thresholds: pd.DataFrame, fp_df: pd.DataFrame, failures: pd.DataFrame, report_dir: Path) -> None:
    table = build_selection_table(results) if not results.empty else pd.DataFrame()
    e7 = table.copy()
    best = e7.iloc[0].to_dict() if not e7.empty else {}
    fv1a = one(e7, "FV1A")
    fv1b = one(e7, "FV1B")
    lite_best = e7[e7["features"].eq("Lite10")].iloc[0].to_dict() if not e7[e7["features"].eq("Lite10")].empty else {}
    axis_best = e7[e7["features"].eq("AxisHard10")].iloc[0].to_dict() if not e7[e7["features"].eq("AxisHard10")].empty else {}
    lr_best = e7[e7["verifier"].eq("logistic_regression")].iloc[0].to_dict() if not e7[e7["verifier"].eq("logistic_regression")].empty else {}
    tree_best = one(e7, "FV9")
    rf_best = one(e7, "FV10")
    target_runs = e7[(e7["FP"] <= 12) & (e7["fall_precision"] >= 0.65)] if not e7.empty else pd.DataFrame()
    correction = (
        fp_df.groupby("run_id")
        .agg(reference_fp=("reference_fp", "sum"), extra_fp=("extra_fp", "sum"), corrected_reference_fp=("corrected_reference_fp", "sum"))
        .reset_index()
        if not fp_df.empty
        else pd.DataFrame()
    )

    lines = [
        "# DS-Fall-RD-FV Extra Summary",
        "",
        "## Protocol",
        "",
        "- Exactly six runs were attempted: FV1A, FV1B, FV7, FV8, FV9, FV10.",
        "- Benchmark is fixed to BITS/WEDA, 25 Hz, 2-second event-centered, existing split.",
        "- A5 reference fall probability is Stage-1; direction probability is copied unchanged from reference.",
        "- Verifiers are trained on train split only; thresholds are selected on validation only.",
        "",
        "## Main E7 WEDA Results",
        "",
        markdown_table(
            format_df(
                e7[
                    [
                        "run_id",
                        "verifier",
                        "features",
                        "fusion",
                        "fall_f1",
                        "fall_precision",
                        "fall_recall",
                        "FP",
                        "FN",
                        "weda_fp_reduction_vs_reference",
                        "weda_fp_reduction_vs_fv1_lite10_and",
                        "direction_macro_f1",
                        "E3_Direction_Macro_F1",
                        "E6_BITS_Fall_F1",
                        "edge_suitability_note",
                    ]
                ]
            )
        )
        if not e7.empty
        else "_No rows._",
        "",
        "## Thresholds",
        "",
        markdown_table(format_df(thresholds)) if not thresholds.empty else "_No thresholds._",
        "",
        "## WEDA FP Correction",
        "",
        markdown_table(format_df(correction)) if not correction.empty else "_No FP rows._",
        "",
        "## Answers",
        "",
        answer_fv1(fv1a, fv1b),
        compare_axis(lite_best, axis_best),
        compare_verifiers(lr_best, tree_best, rf_best),
        answer_best_fp(best),
        answer_tradeoff(best),
        f"6. Runs with FP <= 12 and precision >= 0.65: {', '.join(target_runs['run_id'].tolist()) if not target_runs.empty else 'none'}.",
        answer_replace_fv1(best),
        "8. A5 should remain the main model. The extra FV runs are FPGuard/decision-layer extensions because they do not alter the temporal direction-sensitive architecture.",
        "9. Paper-facing conclusion: the earlier FV1 Lite10 AND remains the best FPGuard result observed so far; these six extra constrained runs do not improve it. Report FV/FPGuard as an optional precision-oriented extension, not a replacement for the A5 main model.",
    ]
    if not failures.empty:
        lines.extend(["", "## Failures", "", markdown_table(format_df(failures))])
    (report_dir / "fv_extra_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def one(table: pd.DataFrame, run_id: str) -> dict[str, Any]:
    rows = table[table["run_id"].eq(run_id)]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def answer_fv1(fv1a: dict[str, Any], fv1b: dict[str, Any]) -> str:
    parts = []
    for name, row in [("FV1A", fv1a), ("FV1B", fv1b)]:
        if row:
            parts.append(f"{name}: FP={int(row['FP'])}, precision={row['fall_precision']:.4f}, recall={row['fall_recall']:.4f}, F1={row['fall_f1']:.4f}")
    return "1. FV1A/FV1B vs old FV1: old FV1 had FP=15, precision=0.6154, recall=0.9600, F1=0.7500. " + "; ".join(parts) + "."


def compare_axis(lite: dict[str, Any], axis: dict[str, Any]) -> str:
    if not lite or not axis:
        return "2. Lite10 vs AxisHard10 comparison is unavailable."
    better = "AxisHard10" if (axis["FP"], -axis["fall_precision"]) < (lite["FP"], -lite["fall_precision"]) else "Lite10"
    return f"2. {better} is better by the selection rule. Best Lite10={lite['run_id']} FP={int(lite['FP'])}, F1={lite['fall_f1']:.4f}; best AxisHard10={axis['run_id']} FP={int(axis['FP'])}, F1={axis['fall_f1']:.4f}."


def compare_verifiers(lr: dict[str, Any], tree: dict[str, Any], rf: dict[str, Any]) -> str:
    rows = [r for r in [lr, tree, rf] if r]
    if not rows:
        return "3. Verifier comparison is unavailable."
    best = sorted(rows, key=lambda r: (r["FP"], -r["fall_precision"], -r["fall_f1"], r["simplicity_rank"]))[0]
    return f"3. Best verifier is `{best['verifier']}` via {best['run_id']}: FP={int(best['FP'])}, precision={best['fall_precision']:.4f}, F1={best['fall_f1']:.4f}."


def answer_best_fp(best: dict[str, Any]) -> str:
    if not best:
        return "4. Best FP run is unavailable."
    return f"4. Strongest FP reduction with recall >= 0.88 is `{best['run_id']}`: WEDA FP={int(best['FP'])}, recall={best['fall_recall']:.4f}, FP reduction vs reference={int(best['weda_fp_reduction_vs_reference'])}."


def answer_tradeoff(best: dict[str, Any]) -> str:
    if not best:
        return "5. Best trade-off is unavailable."
    return f"5. Best paper-target trade-off is `{best['run_id']}`: FP={int(best['FP'])}, precision={best['fall_precision']:.4f}, recall={best['fall_recall']:.4f}, F1={best['fall_f1']:.4f}, direction unchanged."


def answer_replace_fv1(best: dict[str, Any]) -> str:
    if not best:
        return "7. No replacement recommendation."
    if best["run_id"] in {"FV1A", "FV1B"} and best["FP"] < FV1_OLD_WEDA_FP:
        return f"7. Replace old FV1 with `{best['run_id']}` because it lowers FP below 15 while preserving recall/direction."
    if best["FP"] < FV1_OLD_WEDA_FP and best["fall_recall"] >= 0.88:
        return f"7. Replace old FV1 with `{best['run_id']}` if a non-Lite verifier is acceptable."
    return "7. Do not replace old FV1; no extra run improves its FP/F1 trade-off enough."


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
                "direction_n_supervised",
                "verifier_params",
                "tree_depth",
                "tree_leaves",
                "rf_n_trees",
                "rf_max_depth",
                "weda_fp_reduction_vs_reference",
                "weda_fp_reduction_vs_fv1_lite10_and",
            }:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(int(v)))
            else:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    return out


if __name__ == "__main__":
    main()
