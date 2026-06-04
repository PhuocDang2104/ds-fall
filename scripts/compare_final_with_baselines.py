from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
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
from sklearn.svm import SVC

import run_final_compact_hybrid as final
import run_hybrid_feature_reduction_and_lightweight_direction as hybrid
import run_ml_e1_e7_specialized as mlbase


FOCUS_PROTOCOLS = ["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"]
DIRECTION_LABELS = ["forward", "backward", "lateral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare final compact hybrid with baselines.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-svm", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_fall_baselines(seed: int, include_svm: bool, skipped: list[dict[str, str]]) -> dict[str, Any]:
    models: dict[str, Any] = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
        "RandomForest": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)),
        "GradientBoosting": make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=seed)),
        "HistGradientBoosting": make_pipeline(StandardScaler(), HistGradientBoostingClassifier(random_state=seed, max_iter=140)),
    }
    if include_svm:
        models["SVM_RBF"] = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed))
    else:
        skipped.append({"stage": "fall_baseline", "model": "SVM_RBF", "error": "skipped because --include-svm was not set"})
    try:
        from xgboost import XGBClassifier  # type: ignore

        models["XGBoost"] = make_pipeline(
            StandardScaler(),
            XGBClassifier(
                n_estimators=120,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=seed,
            ),
        )
    except Exception as exc:
        skipped.append({"stage": "fall_baseline", "model": "XGBoost", "error": repr(exc)})
    return models


def make_direction_baselines(seed: int, include_svm: bool, skipped: list[dict[str, str]]) -> dict[str, Any]:
    models: dict[str, Any] = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
        "RandomForest": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)),
        "GradientBoosting": make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=seed)),
        "HistGradientBoosting": make_pipeline(StandardScaler(), HistGradientBoostingClassifier(random_state=seed, max_iter=140)),
    }
    if include_svm:
        models["SVM_RBF"] = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed))
    else:
        skipped.append({"stage": "direction_baseline", "model": "SVM_RBF", "error": "skipped because --include-svm was not set"})
    return models


def direction_features(df: pd.DataFrame) -> list[str]:
    signal = mlbase.infer_signal_features(df)
    return [f for f in signal if not mlbase.is_timing_feature(f)]


def fall_scores(model: Any, X: np.ndarray) -> np.ndarray:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(estimator, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(float)
    return np.asarray(model.decision_function(X), dtype=float)


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return math.nan


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y, score))
    except Exception:
        return math.nan


def evaluate_fall(df: pd.DataFrame, mask: np.ndarray, pred: np.ndarray, score: np.ndarray, method: str, protocol_id: str, feature_count: int, complexity: str) -> dict[str, Any]:
    y = df.loc[mask, "fall_label"].astype(int).to_numpy()
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel()]
    return {
        "method": method,
        "experiment_id": protocol_id,
        "feature_input": "event-window statistical features",
        "feature_count": int(feature_count),
        "fall_precision": float(precision_score(y, pred, zero_division=0)),
        "fall_recall": float(recall_score(y, pred, zero_division=0)),
        "fall_f1": float(f1_score(y, pred, zero_division=0)),
        "fall_accuracy": float(accuracy_score(y, pred)),
        "AUROC": safe_auc(y, score),
        "AUPRC": safe_ap(y, score),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "complexity": complexity,
    }


def evaluate_direction(df: pd.DataFrame, mask: np.ndarray, pred_all: np.ndarray, method: str, protocol_id: str, input_type: str, params: int = 0) -> dict[str, Any]:
    sub = df.loc[mask].reset_index(drop=True)
    sup = mlbase.direction_train_mask(sub)
    y = sub.loc[sup, "direction_id"].astype(int).to_numpy()
    pred = pred_all[sup]
    per = f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0) if len(y) else [np.nan, np.nan, np.nan]
    cm = confusion_matrix(y, pred, labels=[0, 1, 2]) if len(y) else np.zeros((3, 3), dtype=int)
    return {
        "method": method,
        "experiment_id": protocol_id,
        "input": input_type,
        "params": int(params),
        "estimated_fp32_kb": params * 4 / 1024,
        "estimated_int8_kb": params / 1024,
        "direction_macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)) if len(y) else np.nan,
        "direction_accuracy": float(accuracy_score(y, pred)) if len(y) else np.nan,
        "forward_f1": float(per[0]),
        "backward_f1": float(per[1]),
        "lateral_f1": float(per[2]),
        "direction_n_supervised": int(len(y)),
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


def tree_complexity(model: Any) -> str:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "estimators_"):
        arr = np.ravel(est.estimators_)
        nodes = sum(int(t.tree_.node_count) for t in arr if hasattr(t, "tree_"))
        return f"trees={len(arr)}, nodes={nodes}"
    if hasattr(est, "_predictors"):
        count = sum(len(stage) for stage in est._predictors)
        return f"hist_gb_predictors={count}"
    return est.__class__.__name__


def load_final_outputs(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    report_dir = repo_root / "outputs" / "reports" / "final_compact_hybrid"
    model_dir = repo_root / "outputs" / "models" / "final_compact_hybrid"
    fall = pd.read_csv(report_dir / "final_compact_hybrid_fall_metrics.csv")
    direction = pd.read_csv(report_dir / "final_compact_hybrid_direction_metrics.csv")
    e2e = pd.read_csv(report_dir / "final_compact_hybrid_e2e_metrics.csv")
    config = json.loads((model_dir / "final_config.json").read_text(encoding="utf-8"))
    return fall, direction, e2e, config


def load_a5_direction_predictions(repo_root: Path, df: pd.DataFrame, protocol_id: str) -> np.ndarray:
    _, _, mask = hybrid.get_protocol_splits(df, protocol_id)
    test = df.loc[mask].reset_index(drop=True)
    p = repo_root / "artifacts" / "experiments_25hz_event" / protocol_id / "predictions.csv"
    pred = pd.read_csv(p)
    aligned = test[["window_id"]].merge(pred[["window_id", "pred_direction"]], on="window_id", how="left", validate="one_to_one")
    return aligned["pred_direction"].astype(str).str.lower().map({"forward": 0, "backward": 1, "lateral": 2}).fillna(-1).astype(int).to_numpy()


def load_a5_fall_predictions(repo_root: Path, df: pd.DataFrame, protocol_id: str) -> tuple[np.ndarray, np.ndarray]:
    _, _, mask = hybrid.get_protocol_splits(df, protocol_id)
    test = df.loc[mask].reset_index(drop=True)
    p = repo_root / "artifacts" / "experiments_25hz_event" / protocol_id / "predictions.csv"
    pred = pd.read_csv(p)
    aligned = test[["window_id"]].merge(pred[["window_id", "pred_fall", "fall_prob"]], on="window_id", how="left", validate="one_to_one")
    fall_pred = aligned["pred_fall"].astype(str).str.lower().eq("fall").astype(int).to_numpy()
    fall_prob = aligned["fall_prob"].astype(float).to_numpy()
    return fall_pred, fall_prob


def e2e_row(df: pd.DataFrame, mask: np.ndarray, fall_pred: np.ndarray, direction_pred: np.ndarray, method: str, protocol_id: str) -> dict[str, Any]:
    row = hybrid.evaluate_e2e_direction(
        df=df,
        mask=mask,
        fall_pred=fall_pred,
        direction_pred=direction_pred,
        pipeline=method,
        experiment_id=protocol_id,
        train_protocol="E3_BITS_WEDA_MIXED",
        source=method,
    )
    row["method"] = method
    return row


def train_baselines(repo_root: Path, df: pd.DataFrame, output_dir: Path, seed: int, include_svm: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fall_features = final.statistical_fall_features(df)
    dir_features = direction_features(df)
    train_mask, val_mask, _ = hybrid.get_protocol_splits(df, "E3_BITS_WEDA_MIXED")
    fall_rows: list[dict[str, Any]] = []
    dir_rows: list[dict[str, Any]] = []
    e2e_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    fall_predictions: dict[str, dict[str, np.ndarray]] = {}
    direction_predictions: dict[str, dict[str, np.ndarray]] = {}

    for name, model in make_fall_baselines(seed, include_svm, skipped).items():
        print(f"[baseline-fall] {name}")
        model.fit(df.loc[train_mask, fall_features].astype(float).to_numpy(), df.loc[train_mask, "fall_label"].astype(int).to_numpy())
        val_score = fall_scores(model, df.loc[val_mask, fall_features].astype(float).to_numpy())
        threshold, _ = mlbase.select_fall_threshold(df.loc[val_mask, "fall_label"].astype(int).to_numpy(), val_score)
        fall_predictions[name] = {}
        for protocol_id in FOCUS_PROTOCOLS:
            _, _, mask = hybrid.get_protocol_splits(df, protocol_id)
            score = fall_scores(model, df.loc[mask, fall_features].astype(float).to_numpy())
            pred = (score >= threshold).astype(int)
            fall_predictions[name][protocol_id] = pred
            fall_rows.append(evaluate_fall(df, mask, pred, score, f"Classical ML Fall - {name}", protocol_id, len(fall_features), tree_complexity(model)))

    dtrain = train_mask & mlbase.direction_train_mask(df)
    dval = val_mask & mlbase.direction_train_mask(df)
    for name, model in make_direction_baselines(seed, include_svm, skipped).items():
        print(f"[baseline-direction] {name}")
        model.fit(df.loc[dtrain, dir_features].astype(float).to_numpy(), df.loc[dtrain, "direction_id"].astype(int).to_numpy())
        direction_predictions[name] = {}
        for protocol_id in FOCUS_PROTOCOLS:
            _, _, mask = hybrid.get_protocol_splits(df, protocol_id)
            pred = model.predict(df.loc[mask, dir_features].astype(float).to_numpy()).astype(int)
            direction_predictions[name][protocol_id] = pred
            dir_rows.append(evaluate_direction(df, mask, pred, f"Classical ML Direction - {name}", protocol_id, "event-window statistical features", params=0))

    fall_df = pd.DataFrame(fall_rows)
    dir_df = pd.DataFrame(dir_rows)
    best_fall_method = (
        fall_df[fall_df["experiment_id"].eq("E3_BITS_WEDA_MIXED")]
        .sort_values(["fall_f1", "fall_precision", "fall_recall"], ascending=[False, False, False])
        .iloc[0]["method"]
        .replace("Classical ML Fall - ", "")
    )
    best_dir_method = (
        dir_df[dir_df["experiment_id"].eq("E3_BITS_WEDA_MIXED")]
        .sort_values(["direction_macro_f1", "direction_accuracy"], ascending=[False, False])
        .iloc[0]["method"]
        .replace("Classical ML Direction - ", "")
    )
    for protocol_id in FOCUS_PROTOCOLS:
        _, _, mask = hybrid.get_protocol_splits(df, protocol_id)
        e2e_rows.append(e2e_row(df, mask, fall_predictions[best_fall_method][protocol_id], direction_predictions[best_dir_method][protocol_id], "Statistical-ML Baseline", protocol_id))

    return (
        fall_df,
        dir_df,
        pd.DataFrame(e2e_rows),
        pd.DataFrame([{"best_fall_baseline": best_fall_method, "best_direction_baseline": best_dir_method, "selection_note": "selected by E3 test for compact paper table analysis"}]),
        pd.DataFrame(skipped, columns=["stage", "model", "error"]),
    )


def dataset_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in df.groupby("dataset"):
        rows.append(
            {
                "dataset": dataset,
                "sampling_rate_hz": 25,
                "window_seconds": 2,
                "input_shape": "50 x 12",
                "n_windows": len(group),
                "n_fall": int(group["fall_label"].sum()),
                "n_nonfall": int((group["fall_label"] == 0).sum()),
                "n_supervised_direction": int(mlbase.direction_train_mask(group).sum()),
            }
        )
    return pd.DataFrame(rows)


def pivot_metric(df: pd.DataFrame, method_col: str, metric: str, prefix: str) -> pd.DataFrame:
    focus = df[df["experiment_id"].isin(FOCUS_PROTOCOLS)].copy()
    out = focus.pivot_table(index=method_col, columns="experiment_id", values=metric, aggfunc="mean").reset_index()
    rename = {
        "E3_BITS_WEDA_MIXED": f"E3_{prefix}",
        "E6_E3_MIXED_TEST_BITS": f"E6_{prefix}",
        "E7_E3_MIXED_TEST_WEDA": f"E7_{prefix}",
    }
    return out.rename(columns=rename)


def make_paper_tables(
    repo_root: Path,
    df: pd.DataFrame,
    final_fall: pd.DataFrame,
    final_dir: pd.DataFrame,
    final_e2e: pd.DataFrame,
    config: dict[str, Any],
    fall_baselines: pd.DataFrame,
    dir_baselines: pd.DataFrame,
    e2e_baselines: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    table1 = dataset_table(df)
    fall_path = repo_root / "outputs" / "models" / "final_compact_hybrid" / "statistical_fall_expert.pkl"
    with fall_path.open("rb") as f:
        fall_expert = pickle.load(f)
    final_fall_copy = final_fall.copy()
    final_fall_copy["method"] = "DS-Fall-RD Compact Hybrid"
    final_fall_copy["complexity"] = tree_complexity(fall_expert["model"])
    fall_all = pd.concat([fall_baselines, final_fall_copy], ignore_index=True)
    table2 = pivot_metric(fall_all, "method", "fall_f1", "Fall_F1")
    e7 = fall_all[fall_all["experiment_id"].eq("E7_E3_MIXED_TEST_WEDA")][["method", "fall_precision", "fall_recall", "FP", "complexity"] if "complexity" in fall_all.columns else ["method", "fall_precision", "fall_recall", "FP"]]
    table2 = table2.merge(e7, on="method", how="left").rename(columns={"fall_precision": "E7_precision", "fall_recall": "E7_recall", "FP": "E7_FP"})
    feature_input = fall_all.groupby("method", as_index=False).agg(**{"Feature/Input": ("feature_input", "first")})
    table2 = table2.merge(feature_input, on="method", how="left")
    table2 = table2.rename(
        columns={
            "method": "Method",
            "E3_Fall_F1": "E3 Fall F1",
            "E6_Fall_F1": "E6 Fall F1",
            "E7_Fall_F1": "E7 Fall F1",
            "E7_precision": "E7 Precision",
            "E7_recall": "E7 Recall",
            "complexity": "Params/Complexity",
        }
    )
    table2 = table2[["Method", "Feature/Input", "E3 Fall F1", "E6 Fall F1", "E7 Fall F1", "E7 Precision", "E7 Recall", "E7_FP", "Params/Complexity"]]

    # Historical A5 direction from saved predictions.
    hist_rows = []
    hist_e2e_rows = []
    # Final statistical fall expert predictions are used for Hybrid A5 reference E2E.
    for protocol_id in FOCUS_PROTOCOLS:
        _, _, mask = hybrid.get_protocol_splits(df, protocol_id)
        pred = load_a5_direction_predictions(repo_root, df, protocol_id)
        hist_rows.append(evaluate_direction(df, mask, pred, "DS-Fall-RD A5 Hybrid Reference", protocol_id, "historical A5 direction head", params=65_959))
        fall_mask, fall_pred, _ = final.fall_predictions_for_protocol(df, fall_expert, protocol_id)
        hist_e2e_rows.append(e2e_row(df, fall_mask, fall_pred, pred, "DS-Fall-RD A5 Hybrid Reference", protocol_id))
    hist_dir = pd.DataFrame(hist_rows)
    hist_e2e = pd.DataFrame(hist_e2e_rows)
    final_dir_copy = final_dir.copy()
    final_dir_copy["method"] = "DS-Fall-RD Compact Hybrid"
    dir_all = pd.concat([dir_baselines, final_dir_copy, hist_dir], ignore_index=True)
    table3 = pivot_metric(dir_all, "method", "direction_macro_f1", "Direction_F1")
    table3["Avg_E3_E6_E7"] = table3[["E3_Direction_F1", "E6_Direction_F1", "E7_Direction_F1"]].mean(axis=1)
    params = dir_all.groupby("method", as_index=False).agg(params=("params", "max"), estimated_int8_kb=("estimated_int8_kb", "max"))
    inputs = dir_all.groupby("method", as_index=False).agg(Input=("input", "first"))
    table3 = table3.merge(inputs, on="method", how="left").merge(params, on="method", how="left")
    table3 = table3.rename(
        columns={
            "method": "Method",
            "params": "Params",
            "E3_Direction_F1": "E3 Direction F1",
            "E6_Direction_F1": "E6 Direction F1",
            "E7_Direction_F1": "E7 Direction F1",
            "estimated_int8_kb": "INT8 KB",
        }
    )
    table3 = table3[["Method", "Input", "Params", "E3 Direction F1", "E6 Direction F1", "E7 Direction F1", "Avg_E3_E6_E7", "INT8 KB"]]

    final_e2e_copy = final_e2e.copy()
    final_e2e_copy["method"] = "DS-Fall-RD Compact Hybrid"
    e2e_all = pd.concat([e2e_baselines, final_e2e_copy, hist_e2e], ignore_index=True)
    table4 = pivot_metric(e2e_all, "method", "E2E_Direction_Macro_F1", "E2E_F1")
    table4["Avg_E2E_F1"] = table4[["E3_E2E_F1", "E6_E2E_F1", "E7_E2E_F1"]].mean(axis=1)
    table4["edge_suitability"] = table4["method"].map(
        {
            "DS-Fall-RD Compact Hybrid": "high",
            "DS-Fall-RD A5 Hybrid Reference": "medium",
            "Statistical-ML Baseline": "medium",
        }
    ).fillna("medium")
    table4["Fall Expert"] = table4["method"].map(
        {
            "DS-Fall-RD Compact Hybrid": "Statistical Fall Expert",
            "DS-Fall-RD A5 Hybrid Reference": "Statistical Fall Expert",
            "Statistical-ML Baseline": "Best classical ML fall baseline",
        }
    )
    table4["Direction Expert"] = table4["method"].map(
        {
            "DS-Fall-RD Compact Hybrid": "LDX1 Wide-DSConv",
            "DS-Fall-RD A5 Hybrid Reference": "historical multitask A5 direction head",
            "Statistical-ML Baseline": "Best classical ML direction baseline",
        }
    )
    table4 = table4.rename(
        columns={
            "method": "Method",
            "E3_E2E_F1": "E3 E2E F1",
            "E6_E2E_F1": "E6 E2E F1",
            "E7_E2E_F1": "E7 E2E F1",
            "Avg_E2E_F1": "Avg E2E F1",
            "edge_suitability": "Edge suitability",
        }
    )
    table4 = table4[["Method", "Fall Expert", "Direction Expert", "E3 E2E F1", "E6 E2E F1", "E7 E2E F1", "Avg E2E F1", "Edge suitability"]]

    final_e7_fall = final_fall[final_fall["experiment_id"].eq("E7_E3_MIXED_TEST_WEDA")]["fall_f1"].mean()
    final_avg_dir = final_dir[final_dir["experiment_id"].isin(FOCUS_PROTOCOLS)]["direction_macro_f1"].mean()
    final_avg_e2e = final_e2e[final_e2e["experiment_id"].isin(FOCUS_PROTOCOLS)]["E2E_Direction_Macro_F1"].mean()
    ref_avg_dir = hist_dir["direction_macro_f1"].mean()
    ref_avg_e2e = hist_e2e["E2E_Direction_Macro_F1"].mean()
    table5 = pd.DataFrame(
        [
            {
                "Model": "DS-Fall-RD Compact Hybrid",
                "Fall feature type": "event-window statistical features",
                "Direction model": "LDX1 Wide-DSConv",
                "Direction params": config["direction_expert"]["params"],
                "Direction param reduction": 1.0 - config["direction_expert"]["params"] / 65_959,
                "Fall E7 F1": final_e7_fall,
                "Direction avg F1": final_avg_dir,
                "E2E avg F1": final_avg_e2e,
                "Limitation": "paper-safe seed selected by validation; compact direction is below historical A5 if so observed",
            },
            {
                "Model": "DS-Fall-RD A5 Hybrid Reference",
                "Fall feature type": "event-window statistical features",
                "Direction model": "historical multitask A5 direction head",
                "Direction params": 65_959,
                "Direction param reduction": 0.0,
                "Fall E7 F1": final_e7_fall,
                "Direction avg F1": ref_avg_dir,
                "E2E avg F1": ref_avg_e2e,
                "Limitation": "larger temporal direction expert",
            },
        ]
    )
    return {"table1": table1, "table2": table2, "table3": table3, "table4": table4, "table5": table5}


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    x = df.copy()
    for col in x.columns:
        if pd.api.types.is_float_dtype(x[col]):
            x[col] = x[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    cols = list(x.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in x.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines)


def supplementary_reference_context(repo_root: Path, report_dir: Path) -> str:
    lines: list[str] = []
    seed_path = report_dir / "ldx1_seed_search_results.csv"
    if seed_path.exists():
        seed_df = pd.read_csv(seed_path)
        seed_summary = (
            seed_df.groupby("learning_rate", as_index=False)
            .agg(
                n_runs=("seed", "count"),
                mean_E3_direction_f1=("E3_direction_macro_f1", "mean"),
                mean_E6_direction_f1=("E6_direction_macro_f1", "mean"),
                mean_E7_direction_f1=("E7_direction_macro_f1", "mean"),
                mean_score=("score_best_observed", "mean"),
                std_score=("score_best_observed", "std"),
                best_score=("score_best_observed", "max"),
            )
            .sort_values("mean_score", ascending=False)
        )
        seed_summary.to_csv(report_dir / "ldx1_seed_search_summary.csv", index=False)
        best_observed = seed_df.sort_values("score_best_observed", ascending=False).head(1)
        lines += [
            "## LDX1 Multi-Seed Summary",
            "",
            "Full final seed-search summary for the compact direction expert:",
            "",
            md_table(seed_summary),
            "",
            "Best-observed seed is analysis-only because it is ranked with E3/E6/E7 test metrics:",
            "",
            md_table(best_observed[["seed", "learning_rate", "best_val_macro_f1", "E3_direction_macro_f1", "E6_direction_macro_f1", "E7_direction_macro_f1", "score_best_observed", "params"]]),
            "",
        ]
    legacy_candidates = list((repo_root / "outputs").rglob("a5_vs_ldx1_seed_summary.csv"))
    legacy_path = legacy_candidates[0] if legacy_candidates else repo_root / "outputs" / "reports" / "a5_direction_vs_ldx1_multiseed" / "a5_vs_ldx1_seed_summary.csv"
    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path)
        cols = [
            "model_name",
            "learning_rate",
            "params",
            "mean_avg_e3_e6_e7_f1",
            "std_avg_e3_e6_e7_f1",
            "mean_e3_f1",
            "mean_e6_f1",
            "mean_e7_f1",
            "estimated_int8_kb",
        ]
        legacy_out = legacy[[c for c in cols if c in legacy.columns]].copy()
        legacy_out.to_csv(report_dir / "legacy_a5_direction_only_ldx1_summary.csv", index=False)
        lines += [
            "## Historical Direction-Only Context",
            "",
            "Prior A5 Direction-Only and LDX1 multi-seed runs are kept as context, not as the final selected paper-safe model:",
            "",
            md_table(legacy_out),
            "",
        ]
    return "\n".join(lines)


def write_paper_report(report_dir: Path, tables: dict[str, pd.DataFrame], final_config: dict[str, Any], selected_seed: dict[str, Any], baseline_selection: pd.DataFrame, supplementary_context: str) -> None:
    lines = [
        "# Final Paper Readiness Report",
        "",
        "## Final Model",
        "",
        "- Name: DS-Fall-RD Compact Hybrid.",
        "- Fall: Statistical Fall Expert, GradientBoostingClassifier, event-window statistical features excluding timing-index artifacts.",
        "- Direction: LDX1 Wide-DSConv Direction Expert, temporal tilt12 input `50 x 12`.",
        "",
        "## Questions",
        "",
        f"1. Selected seed/lr: seed `{selected_seed['selected_seed']}`, learning rate `{selected_seed['selected_learning_rate']}`, mode `{selected_seed['selection_mode']}`.",
        f"2. Reason: {selected_seed['selected_reason']}.",
        "3. Final direction metrics are in Table 3 below and `final_compact_hybrid_direction_metrics.csv`.",
        "4. Final E2E metrics are in Table 4 below and `final_compact_hybrid_e2e_metrics.csv`.",
        "5. Comparisons include classical ML-only baselines, historical A5 hybrid reference, and final compact hybrid.",
        f"6. Direction params: `{final_config['direction_expert']['params']}`, estimated INT8 `{final_config['direction_expert']['estimated_int8_kb']:.2f} KB`.",
        f"7. Direction parameter reduction vs historical A5: `{(1.0 - final_config['direction_expert']['params'] / 65959) * 100:.2f}%`.",
        "8. Paper safety: if selection mode is `paper_safe`, seed is validation-selected. If `best_observed`, mark as optimistic analysis only.",
        "9. Limitations: BITS/WEDA only, TensorFlow/TFLite unavailable in current environment if logged, and seed sensitivity should be disclosed.",
        "10. Repo readiness: final scripts, model card, tables, configs, and artifacts are saved under `final_compact_hybrid`; inspect failed runs before submission.",
        "11. Supplementary files: seed search table, selected seed JSON, final config, model card, paper tables, class counts, and confusion matrices.",
        "12. A5 Direction-Only and LDX1 multi-seed context is included below when prior reports are available.",
        "",
        "## Baseline Selection",
        "",
        md_table(baseline_selection),
        "",
        "## Table 1: Dataset and Window Statistics",
        "",
        md_table(tables["table1"]),
        "",
        "## Table 2: Fall Detection Comparison",
        "",
        md_table(tables["table2"]),
        "",
        "## Table 3: Direction Classification Comparison",
        "",
        md_table(tables["table3"]),
        "",
        "## Table 4: End-to-End Hybrid Comparison",
        "",
        md_table(tables["table4"]),
        "",
        "## Table 5: Final Compact Model vs Main Reference",
        "",
        md_table(tables["table5"]),
        "",
        supplementary_context,
    ]
    text = "\n".join(lines)
    (report_dir / "final_paper_readiness_report.md").write_text(text, encoding="utf-8")
    (report_dir / "paper_tables.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.run_all:
        print("Use --run-all to compare final model with baselines.")
        return
    repo_root = Path(args.repo_root).resolve()
    report_dir = ensure_dir(repo_root / "outputs" / "reports" / "final_compact_hybrid")
    df, _ = final.load_data(repo_root)
    final_fall, final_dir, final_e2e, config = load_final_outputs(repo_root)
    fall_baselines, dir_baselines, e2e_baselines, baseline_selection, skipped = train_baselines(repo_root, df, report_dir, args.seed, args.include_svm)
    tables = make_paper_tables(repo_root, df, final_fall, final_dir, final_e2e, config, fall_baselines, dir_baselines, e2e_baselines)
    names = {
        "table1": "paper_table_1_dataset_stats.csv",
        "table2": "paper_table_2_fall_detection.csv",
        "table3": "paper_table_3_direction.csv",
        "table4": "paper_table_4_e2e.csv",
        "table5": "paper_table_5_final_comparison.csv",
    }
    for key, name in names.items():
        tables[key].to_csv(report_dir / name, index=False)
    fall_baselines.to_csv(report_dir / "baseline_fall_results.csv", index=False)
    dir_baselines.to_csv(report_dir / "baseline_direction_results.csv", index=False)
    e2e_baselines.to_csv(report_dir / "baseline_e2e_results.csv", index=False)
    baseline_selection.to_csv(report_dir / "baseline_selection.csv", index=False)
    selected = json.loads((report_dir / "ldx1_selected_seed.json").read_text(encoding="utf-8"))
    context = supplementary_reference_context(repo_root, report_dir)
    write_paper_report(report_dir, tables, config, selected, baseline_selection, context)
    skipped.to_csv(report_dir / "compare_failed_runs.csv", index=False)
    print("[compare] done")


if __name__ == "__main__":
    main()
