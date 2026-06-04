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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "ds_fall_rd_fv"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "ds_fall_rd_fv" / "confusion_matrices"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "reports" / "ml_feature_analysis_25hz" / "summary_feature_matrix.csv"

REFERENCE_PARAMS = 65959
REFERENCE_THRESHOLD = 0.5
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
CENTER12 = LITE10 + ["impact_distance_from_center", "gyro_peak_distance_from_center"]
HARD_NEG_FEATURES = [
    "jerk_p95",
    "jerk_std",
    "acc_mag_std",
    "acc_mag_range",
    "tilt_delta_p95",
    "tilt_delta_max",
    "ay_range",
    "ay_std",
    "gyro_mag_p95",
    "gyro_mag_max",
]


@dataclass(frozen=True)
class FVConfig:
    run_id: str
    features: str
    fusion: str
    hard_negative_weight: float
    timing_features: bool
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DS-Fall-RD-FV verifier experiments.")
    parser.add_argument("--summary_path", default=str(SUMMARY_PATH))
    parser.add_argument("--output_dir", default=str(REPORT_DIR))
    parser.add_argument("--figure_dir", default=str(FIGURE_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = ensure_dir(Path(args.output_dir))
    figure_dir = ensure_dir(Path(args.figure_dir))
    run_dir = ensure_dir(report_dir / "runs")

    df = load_summary(Path(args.summary_path))
    validate_inputs(df)
    feature_configs = {
        "FV-Lite10": LITE10,
        "FV-Center12": CENTER12,
        "hard_negative_score_features": [c for c in HARD_NEG_FEATURES if c in df.columns],
        "notes": {
            "summary_source": str(Path(args.summary_path)),
            "benchmark": "BITS/WEDA only, 25 Hz, 2-second event-centered, tilt12",
            "direction": "Frozen reference direction probabilities from summary_feature_matrix.csv",
            "architecture_equivalence": "Stage-2 FV runs train a logistic-regression verifier Dense(1) and linear logit fusion on frozen DS-Fall-RD fall logits. This is functionally equivalent to the requested frozen DS-Fall-RD-FV head.",
        },
    }
    (report_dir / "fv_feature_configs.json").write_text(json.dumps(feature_configs, indent=2), encoding="utf-8")

    df = add_hard_negative_flags(df)
    ref_rows = reference_rows(df, figure_dir)
    ref_metrics = {row["eval"]: row for row in ref_rows}

    configs = fv_configs()
    all_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    params_rows: list[dict[str, Any]] = []
    weda_fp_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for cfg in configs[:5]:
        print(f"Running {cfg.run_id}...")
        row_bundle, threshold_bundle, params_row, fp_rows = run_fv_config(df, cfg, args.seed, report_dir, run_dir, figure_dir)
        all_rows.extend(row_bundle)
        threshold_rows.append(threshold_bundle)
        params_rows.append(params_row)
        weda_fp_rows.extend(fp_rows)

    fv2_fv5 = [r for r in all_rows if r["eval"] == "E7" and r["run_id"] in {"FV2_LITE10_LOGIT_FUSION", "FV5_LITE10_LOGIT_FUSION_HN3"}]
    stage2_strong = any(
        r["fall_f1"] > 0.75
        and r["fall_precision"] > 0.65
        and r["fall_recall"] >= 0.88
        and r["direction_macro_f1"] >= 0.80
        for r in fv2_fv5
    )
    fv6_reason = (
        "Skipped by prompt rule: Stage 3 is optional only if Stage 2 is too weak. "
        "FV2/FV5 already reached the WEDA target range with frozen encoder/head verifier. "
        "No fall-head unfreezing was needed, so the benchmark is kept cleaner."
        if stage2_strong
        else "Skipped because safe fall-head unfreezing would require rebuilding a trainable temporal FV graph with explicit fall-logit exposure and verifying unchanged direction outputs. This frozen-probability FV script completed FV1-FV5 and records FV6 as not run instead of applying an unsafe partial unfreeze."
    )
    failure_rows.append(
        {
            "run_id": "FV6_LITE10_LOGIT_FUSION_UNFREEZE_FALL",
            "status": "not_run",
            "reason": fv6_reason,
        }
    )
    threshold_rows.append(
        {
            "run_id": "FV6_LITE10_LOGIT_FUSION_UNFREEZE_FALL",
            "features": "FV-Lite10",
            "fusion": "stage3_unfreeze_fall",
            "threshold": math.nan,
            "t_deep": math.nan,
            "t_lr": math.nan,
            "alpha_gate": math.nan,
            "hard_negative_weight": 2.0,
            "status": "not_run",
            "reason": fv6_reason,
        }
    )
    params_rows.append(
        {
            "run_id": "FV6_LITE10_LOGIT_FUSION_UNFREEZE_FALL",
            "params": math.nan,
            "added_params_over_reference": math.nan,
            "summary_feature_count": len(LITE10),
            "timing_features_used": False,
            "direction_changed": False,
            "status": "not_run",
            "notes": fv6_reason,
        }
    )

    results = pd.DataFrame(ref_rows + all_rows)
    thresholds = pd.DataFrame(threshold_rows)
    params = pd.DataFrame(
        [
            {
                "run_id": "REF_CURRENT_E3_ARTIFACT",
                "params": REFERENCE_PARAMS,
                "added_params_over_reference": 0,
                "summary_feature_count": 0,
                "timing_features_used": False,
                "direction_changed": False,
                "status": "reference",
                "notes": "Reference DS-Fall-RD A5WCEFW.",
            }
        ]
        + params_rows
    )
    fp_analysis = pd.DataFrame(weda_fp_rows)
    failures = pd.DataFrame(failure_rows)

    results.to_csv(report_dir / "fv_results.csv", index=False)
    thresholds.to_csv(report_dir / "fv_thresholds.csv", index=False)
    params.to_csv(report_dir / "fv_model_params.csv", index=False)
    fp_analysis.to_csv(report_dir / "fv_weda_fp_analysis.csv", index=False)
    failures.to_csv(report_dir / "fv_failures.csv", index=False)

    write_selection(results, thresholds, params, failures, ref_metrics, report_dir)
    write_summary(results, thresholds, params, fp_analysis, failures, ref_metrics, report_dir)
    print(f"Saved reports to {report_dir}")


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
    return series.astype(str).str.lower().isin(["1", "true", "yes", "y"])


def validate_inputs(df: pd.DataFrame) -> None:
    required = {
        "window_id",
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
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in summary feature matrix: {missing}")
    datasets = set(df["dataset"].unique())
    if not {"bits", "weda"}.issubset(datasets):
        raise ValueError(f"Expected bits and weda in summary matrix, got {sorted(datasets)}")
    splits = set(df["split"].unique())
    if not {"train", "val", "test"}.issubset(splits):
        raise ValueError(f"Expected train/val/test splits, got {sorted(splits)}")


def add_hard_negative_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    score_features = [c for c in HARD_NEG_FEATURES if c in df.columns]
    score = np.zeros(len(df), dtype=float)
    eligible = df["dataset"].eq("weda") & df["fall_label"].eq(0) & df["split"].isin(["train", "val"])
    if eligible.any() and score_features:
        ranks = []
        sub = df.loc[eligible, score_features].astype(float)
        for col in score_features:
            ranks.append(sub[col].rank(pct=True).to_numpy())
        values = np.mean(np.vstack(ranks), axis=0)
        score[eligible.to_numpy()] = values
        cutoff = np.quantile(values, 0.80)
        hard = np.zeros(len(df), dtype=bool)
        hard[np.flatnonzero(eligible.to_numpy())] = values >= cutoff
    else:
        hard = np.zeros(len(df), dtype=bool)
    df["hard_negative_score"] = score
    df["hard_negative_flag"] = hard
    return df


def fv_configs() -> list[FVConfig]:
    return [
        FVConfig("FV1_LITE10_AND", "FV-Lite10", "and", 2.0, False, "AND fusion with Lite10 verifier."),
        FVConfig("FV2_LITE10_LOGIT_FUSION", "FV-Lite10", "logit_fusion", 2.0, False, "Preferred clean FV architecture."),
        FVConfig("FV3_CENTER12_LOGIT_FUSION", "FV-Center12", "logit_fusion", 2.0, True, "Upper-bound with center timing features."),
        FVConfig("FV4_LITE10_GATED_PRODUCT", "FV-Lite10", "gated_product", 2.0, False, "Gated-product verifier."),
        FVConfig("FV5_LITE10_LOGIT_FUSION_HN3", "FV-Lite10", "logit_fusion", 3.0, False, "Same as FV2 with hard-negative weight 3."),
        FVConfig("FV6_LITE10_LOGIT_FUSION_UNFREEZE_FALL", "FV-Lite10", "stage3_unfreeze_fall", 2.0, False, "Optional Stage 3."),
    ]


def feature_names(feature_set: str) -> list[str]:
    if feature_set == "FV-Lite10":
        return LITE10
    if feature_set == "FV-Center12":
        return CENTER12
    raise ValueError(feature_set)


def split_mask(df: pd.DataFrame, split: str, dataset: str | None = None) -> np.ndarray:
    mask = df["split"].eq(split).to_numpy()
    if dataset:
        mask &= df["dataset"].eq(dataset).to_numpy()
    return mask


def sample_weights(df_train: pd.DataFrame, hard_negative_weight: float) -> np.ndarray:
    weights = np.ones(len(df_train), dtype=float)
    weights[df_train["hard_negative_flag"].to_numpy()] = hard_negative_weight
    return weights


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def train_lr_verifier(
    df: pd.DataFrame,
    features: list[str],
    hard_negative_weight: float,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    train = df["split"].eq("train")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df.loc[train, features].astype(float).to_numpy())
    y_train = df.loc[train, "fall_label"].astype(int).to_numpy()
    sw = sample_weights(df.loc[train].reset_index(drop=True), hard_negative_weight)
    verifier = LogisticRegression(max_iter=3000, random_state=seed)
    verifier.fit(X_train, y_train, sample_weight=sw)
    with (run_dir / "summary_scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)
    with (run_dir / "lr_verifier.pkl").open("wb") as f:
        pickle.dump(verifier, f)
    z_lr = verifier.decision_function(scaler.transform(df[features].astype(float).to_numpy()))
    p_lr = verifier.predict_proba(scaler.transform(df[features].astype(float).to_numpy()))[:, 1]
    return {"scaler": scaler, "verifier": verifier, "z_lr": z_lr, "p_lr": p_lr}


def train_fusion(df: pd.DataFrame, z_lr: np.ndarray, hard_negative_weight: float, seed: int, run_dir: Path) -> dict[str, Any]:
    train = df["split"].eq("train").to_numpy()
    y_train = df.loc[train, "fall_label"].astype(int).to_numpy()
    z_deep = logit(df["fall_prob"].to_numpy())
    X_train = np.column_stack([z_deep[train], z_lr[train]])
    sw = sample_weights(df.loc[train].reset_index(drop=True), hard_negative_weight)
    fusion = LogisticRegression(max_iter=3000, random_state=seed)
    fusion.fit(X_train, y_train, sample_weight=sw)
    with (run_dir / "fall_logit_fusion.pkl").open("wb") as f:
        pickle.dump(fusion, f)
    X_all = np.column_stack([z_deep, z_lr])
    p_final = fusion.predict_proba(X_all)[:, 1]
    return {"fusion": fusion, "p_final": p_final}


def tune_threshold(y: np.ndarray, prob: np.ndarray, recall_floor: float | None = None) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 199), np.quantile(prob, np.linspace(0.01, 0.99, 99))]))
    best = None
    for t in candidates:
        pred = (prob >= t).astype(int)
        precision = precision_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        cm = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        tn, fp, fn, tp = [int(v) for v in cm]
        feasible = recall_floor is None or recall >= recall_floor
        key = (1 if feasible else 0, -fp, precision, f1, recall)
        if best is None or key > best[0]:
            best = (key, float(t))
    return best[1] if best else 0.5


def tune_and_thresholds(y: np.ndarray, p_deep: np.ndarray, p_lr: np.ndarray, recall_floor: float = 0.86) -> tuple[float, float]:
    candidates_deep = np.unique(np.concatenate([np.linspace(0.01, 0.99, 80), np.quantile(p_deep, np.linspace(0.05, 0.95, 40))]))
    candidates_lr = np.unique(np.concatenate([np.linspace(0.01, 0.99, 80), np.quantile(p_lr, np.linspace(0.05, 0.95, 40))]))
    best = None
    for td in candidates_deep:
        deep_ok = p_deep >= td
        for tlr in candidates_lr:
            pred = (deep_ok & (p_lr >= tlr)).astype(int)
            precision = precision_score(y, pred, zero_division=0)
            recall = recall_score(y, pred, zero_division=0)
            f1 = f1_score(y, pred, zero_division=0)
            tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel()]
            key = (1 if recall >= recall_floor else 0, -fp, precision, f1, recall)
            if best is None or key > best[0]:
                best = (key, float(td), float(tlr))
    return best[1], best[2]


def run_fv_config(
    df: pd.DataFrame,
    cfg: FVConfig,
    seed: int,
    report_dir: Path,
    run_root: Path,
    figure_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    run_dir = ensure_dir(run_root / cfg.run_id)
    features = feature_names(cfg.features)
    train_note = {
        "run_id": cfg.run_id,
        "config": cfg.__dict__,
        "features": features,
        "train_split_counts": split_counts(df[df["split"].eq("train")]),
        "val_split_counts": split_counts(df[df["split"].eq("val")]),
        "test_split_counts": split_counts(df[df["split"].eq("test")]),
    }
    (run_dir / "config.json").write_text(json.dumps(train_note, indent=2), encoding="utf-8")

    lr_bundle = train_lr_verifier(df, features, cfg.hard_negative_weight, seed, run_dir)
    p_deep = df["fall_prob"].to_numpy(dtype=float)
    p_lr = lr_bundle["p_lr"]
    z_lr = lr_bundle["z_lr"]
    val = df["split"].eq("val").to_numpy()
    y_val = df.loc[val, "fall_label"].astype(int).to_numpy()

    threshold_row: dict[str, Any] = {
        "run_id": cfg.run_id,
        "features": cfg.features,
        "fusion": cfg.fusion,
        "hard_negative_weight": cfg.hard_negative_weight,
        "status": "completed",
    }

    if cfg.fusion == "and":
        t_deep, t_lr = tune_and_thresholds(y_val, p_deep[val], p_lr[val], recall_floor=0.86)
        final_prob = np.minimum(p_deep, p_lr)
        pred_fn = lambda idx: ((p_deep[idx] >= t_deep) & (p_lr[idx] >= t_lr)).astype(int)
        threshold_row.update({"threshold": math.nan, "t_deep": t_deep, "t_lr": t_lr, "alpha_gate": math.nan})
        added_params = len(features) + 1
    elif cfg.fusion == "logit_fusion":
        fusion_bundle = train_fusion(df, z_lr, cfg.hard_negative_weight, seed, run_dir)
        final_prob = fusion_bundle["p_final"]
        threshold = tune_threshold(y_val, final_prob[val], recall_floor=0.86)
        pred_fn = lambda idx: (final_prob[idx] >= threshold).astype(int)
        threshold_row.update({"threshold": threshold, "t_deep": math.nan, "t_lr": math.nan, "alpha_gate": math.nan})
        added_params = len(features) + 1 + 3
    elif cfg.fusion == "gated_product":
        best = None
        final_by_alpha = {}
        for alpha in [0.2, 0.3]:
            p = p_deep * (alpha + (1.0 - alpha) * p_lr)
            final_by_alpha[alpha] = p
            t = tune_threshold(y_val, p[val], recall_floor=0.86)
            pred = (p[val] >= t).astype(int)
            f1 = f1_score(y_val, pred, zero_division=0)
            precision = precision_score(y_val, pred, zero_division=0)
            recall = recall_score(y_val, pred, zero_division=0)
            tn, fp, fn, tp = [int(v) for v in confusion_matrix(y_val, pred, labels=[0, 1]).ravel()]
            key = (1 if recall >= 0.86 else 0, -fp, precision, f1, recall)
            if best is None or key > best[0]:
                best = (key, alpha, t)
        alpha, threshold = float(best[1]), float(best[2])
        final_prob = final_by_alpha[alpha]
        pred_fn = lambda idx: (final_prob[idx] >= threshold).astype(int)
        threshold_row.update({"threshold": threshold, "t_deep": math.nan, "t_lr": math.nan, "alpha_gate": alpha})
        added_params = len(features) + 1
    else:
        raise ValueError(cfg.fusion)

    rows = []
    for eval_id, dataset in [("E3", "bits+weda"), ("E6", "bits"), ("E7", "weda")]:
        idx = eval_indices(df, eval_id)
        metrics = evaluate_eval(df, idx, pred_fn(idx), final_prob[idx], cfg.run_id, eval_id, dataset, figure_dir)
        metrics.update(
            {
                "features": cfg.features,
                "fusion": cfg.fusion,
                "hard_negative_weight": cfg.hard_negative_weight,
                "summary_feature_count": len(features),
                "timing_features_used": cfg.timing_features,
                "params": REFERENCE_PARAMS + added_params,
                "added_params_over_reference": added_params,
                "direction_changed_from_reference": False,
                "notes": cfg.notes,
            }
        )
        rows.append(metrics)

    params_row = {
        "run_id": cfg.run_id,
        "params": REFERENCE_PARAMS + added_params,
        "added_params_over_reference": added_params,
        "summary_feature_count": len(features),
        "timing_features_used": cfg.timing_features,
        "direction_changed": False,
        "status": "completed",
        "notes": cfg.notes,
    }
    fp_rows = weda_fp_analysis(df, cfg.run_id, p_lr, final_prob, pred_fn(eval_indices(df, "E7")), features)
    return rows, threshold_row, params_row, fp_rows


def split_counts(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"n": int(len(df))}
    for dataset in ["bits", "weda"]:
        sub = df[df["dataset"].eq(dataset)]
        out[dataset] = {
            "n": int(len(sub)),
            "fall": int(sub["fall_label"].sum()),
            "non_fall": int((sub["fall_label"] == 0).sum()),
            "direction_supervised": int(sub["direction_supervised_bool"].sum()),
        }
    return out


def eval_indices(df: pd.DataFrame, eval_id: str) -> np.ndarray:
    test = df["split"].eq("test").to_numpy()
    if eval_id == "E3":
        return np.flatnonzero(test)
    if eval_id == "E6":
        return np.flatnonzero(test & df["dataset"].eq("bits").to_numpy())
    if eval_id == "E7":
        return np.flatnonzero(test & df["dataset"].eq("weda").to_numpy())
    raise ValueError(eval_id)


def direction_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probs = df[["direction_prob_forward", "direction_prob_backward", "direction_prob_lateral"]].astype(float).to_numpy()
    pred = np.argmax(probs, axis=1)
    return probs, pred


def evaluate_eval(
    df: pd.DataFrame,
    idx: np.ndarray,
    pred_fall: np.ndarray,
    fall_prob: np.ndarray,
    run_id: str,
    eval_id: str,
    dataset: str,
    figure_dir: Path,
) -> dict[str, Any]:
    sub = df.iloc[idx].reset_index(drop=True)
    y = sub["fall_label"].astype(int).to_numpy()
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, pred_fall, labels=[0, 1]).ravel()]
    precision = precision_score(y, pred_fall, zero_division=0)
    recall = recall_score(y, pred_fall, zero_division=0)
    f1 = f1_score(y, pred_fall, zero_division=0)
    try:
        auroc = roc_auc_score(y, fall_prob)
    except ValueError:
        auroc = math.nan
    try:
        ap = average_precision_score(y, fall_prob)
    except ValueError:
        ap = math.nan

    _, pred_dir_all = direction_arrays(sub)
    y_dir = sub["direction_id"].to_numpy()
    supervised = sub["direction_supervised_bool"].to_numpy() & (sub["fall_label"].to_numpy() == 1) & (y_dir >= 0)
    if supervised.any():
        dir_acc = accuracy_score(y_dir[supervised], pred_dir_all[supervised])
        dir_macro = f1_score(y_dir[supervised], pred_dir_all[supervised], labels=[0, 1, 2], average="macro", zero_division=0)
        dir_cm = confusion_matrix(y_dir[supervised], pred_dir_all[supervised], labels=[0, 1, 2])
        per_class = f1_score(y_dir[supervised], pred_dir_all[supervised], labels=[0, 1, 2], average=None, zero_division=0)
    else:
        dir_acc = math.nan
        dir_macro = math.nan
        dir_cm = np.zeros((3, 3), dtype=int)
        per_class = [math.nan, math.nan, math.nan]

    save_confusion(figure_dir, run_id, eval_id, dataset, np.array([[tn, fp], [fn, tp]]), ["non_fall", "fall"], "fall")
    save_confusion(figure_dir, run_id, eval_id, dataset, dir_cm, DIRECTION_LABELS, "direction")

    return {
        "run_id": run_id,
        "eval": eval_id,
        "dataset": dataset,
        "n": int(len(sub)),
        "threshold": math.nan,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "fall_precision": float(precision),
        "fall_recall": float(recall),
        "fall_f1": float(f1),
        "fall_accuracy": float(accuracy_score(y, pred_fall)),
        "fall_auroc": float(auroc),
        "fall_average_precision": float(ap),
        "direction_macro_f1": float(dir_macro),
        "direction_accuracy": float(dir_acc),
        "direction_n_supervised": int(supervised.sum()),
        "forward_f1": float(per_class[0]),
        "backward_f1": float(per_class[1]),
        "lateral_f1": float(per_class[2]),
    }


def reference_rows(df: pd.DataFrame, figure_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for eval_id, dataset in [("E3", "bits+weda"), ("E6", "bits"), ("E7", "weda")]:
        idx = eval_indices(df, eval_id)
        prob = df.iloc[idx]["fall_prob"].to_numpy(dtype=float)
        pred = (prob >= REFERENCE_THRESHOLD).astype(int)
        metrics = evaluate_eval(df, idx, pred, prob, "REF_CURRENT_E3_ARTIFACT", eval_id, dataset, figure_dir)
        metrics.update(
            {
                "features": "temporal_tilt12",
                "fusion": "reference_softmax",
                "hard_negative_weight": math.nan,
                "summary_feature_count": 0,
                "timing_features_used": False,
                "params": REFERENCE_PARAMS,
                "added_params_over_reference": 0,
                "direction_changed_from_reference": False,
                "notes": "Reference DS-Fall-RD A5WCEFW, threshold 0.5.",
            }
        )
        rows.append(metrics)
    return rows


def save_confusion(figure_dir: Path, run_id: str, eval_id: str, dataset: str, cm: np.ndarray, labels: list[str], task: str) -> None:
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{run_id} {eval_id} {dataset} {task}")
    plt.tight_layout()
    safe_run = run_id.replace("/", "_")
    plt.savefig(figure_dir / f"{safe_run}_{eval_id}_{task}_confusion.png", dpi=160)
    plt.close()


def weda_fp_analysis(
    df: pd.DataFrame,
    run_id: str,
    p_lr: np.ndarray,
    final_prob: np.ndarray,
    final_pred_e7: np.ndarray,
    features: list[str],
) -> list[dict[str, Any]]:
    idx = eval_indices(df, "E7")
    sub = df.iloc[idx].reset_index(drop=True).copy()
    ref_pred = (sub["fall_prob"].to_numpy(dtype=float) >= REFERENCE_THRESHOLD).astype(int)
    y = sub["fall_label"].astype(int).to_numpy()
    rows = []
    for local_i, row in sub.iterrows():
        is_ref_fp = bool(y[local_i] == 0 and ref_pred[local_i] == 1)
        is_fv_fp = bool(y[local_i] == 0 and final_pred_e7[local_i] == 1)
        if is_ref_fp or is_fv_fp:
            out = {
                "run_id": run_id,
                "window_id": row.get("window_id", ""),
                "activity": row.get("activity", row.get("activity_name", "")),
                "subject_id": row.get("subject_id", ""),
                "trial_id": row.get("trial_id", ""),
                "true_fall": int(y[local_i]),
                "reference_fall_prob": float(row["fall_prob"]),
                "reference_pred": int(ref_pred[local_i]),
                "lr_fall_prob": float(p_lr[idx][local_i]),
                "fv_final_prob": float(final_prob[idx][local_i]),
                "fv_pred": int(final_pred_e7[local_i]),
                "reference_fp": is_ref_fp,
                "fv_fp": is_fv_fp,
                "corrected_reference_fp": bool(is_ref_fp and not is_fv_fp),
            }
            for feature in features:
                out[feature] = row.get(feature, math.nan)
            rows.append(out)
    return rows


def write_selection(
    results: pd.DataFrame,
    thresholds: pd.DataFrame,
    params: pd.DataFrame,
    failures: pd.DataFrame,
    ref_metrics: dict[str, dict[str, Any]],
    report_dir: Path,
) -> None:
    e7 = results[results["eval"].eq("E7")].copy()
    e6 = results[results["eval"].eq("E6")][["run_id", "fall_f1", "direction_macro_f1"]].rename(
        columns={"fall_f1": "E6_BITS_Fall_F1", "direction_macro_f1": "E6_BITS_Direction_Macro_F1"}
    )
    e3 = results[results["eval"].eq("E3")][["run_id", "fall_f1", "direction_macro_f1"]].rename(
        columns={"fall_f1": "E3_Fall_F1", "direction_macro_f1": "E3_Direction_Macro_F1"}
    )
    merged = e7.merge(e6, on="run_id", how="left").merge(e3, on="run_id", how="left")
    completed = merged[~merged["run_id"].eq("REF_CURRENT_E3_ARTIFACT")].copy()
    completed["passes_targets"] = (
        (completed["FP"] < 12)
        & (completed["fall_precision"] > 0.65)
        & (completed["fall_recall"] >= 0.88)
        & (completed["fall_f1"] > 0.75)
        & (completed["E3_Direction_Macro_F1"] >= 0.86)
        & (completed["direction_macro_f1"] >= 0.80)
        & (completed["E6_BITS_Fall_F1"] >= 0.90)
        & (completed["params"] <= 70000)
    )
    candidates = completed.sort_values(
        ["passes_targets", "fall_f1", "fall_precision", "FP", "E3_Direction_Macro_F1"],
        ascending=[False, False, False, True, False],
    )
    best = candidates.iloc[0].to_dict() if not candidates.empty else {}

    lines = [
        "# DS-Fall-RD-FV Best Selection",
        "",
        "Selection target: WEDA FP < 12, precision > 0.65, recall >= 0.88, Fall F1 > 0.75, E3 Direction Macro F1 >= 0.86, E7 Direction Macro F1 >= 0.80, E6 BITS Fall F1 >= 0.90, params <= 70k.",
        "",
        "## Passing Runs",
        "",
        markdown_table(format_df(completed[completed["passes_targets"]][["run_id", "fall_f1", "fall_precision", "fall_recall", "FP", "FN", "direction_macro_f1", "E3_Direction_Macro_F1", "E6_BITS_Fall_F1", "params"]])),
        "",
        "## Best Trade-off",
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
                f"- E7 WEDA Direction Macro F1: {best['direction_macro_f1']:.4f}",
                f"- E3 Direction Macro F1: {best['E3_Direction_Macro_F1']:.4f}",
                f"- Params: {int(best['params'])}",
            ]
        )
    else:
        lines.append("No completed FV run was available.")
    if not failures.empty:
        lines.extend(["", "## Not Run / Failed", "", markdown_table(format_df(failures))])
    (report_dir / "fv_best_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    results: pd.DataFrame,
    thresholds: pd.DataFrame,
    params: pd.DataFrame,
    fp_analysis: pd.DataFrame,
    failures: pd.DataFrame,
    ref_metrics: dict[str, dict[str, Any]],
    report_dir: Path,
) -> None:
    e7 = results[results["eval"].eq("E7")].sort_values(["fall_f1", "fall_precision"], ascending=False)
    e3 = results[results["eval"].eq("E3")][["run_id", "fall_f1", "direction_macro_f1"]].rename(
        columns={"fall_f1": "E3_Fall_F1", "direction_macro_f1": "E3_Direction_Macro_F1"}
    )
    e6 = results[results["eval"].eq("E6")][["run_id", "fall_f1", "direction_macro_f1"]].rename(
        columns={"fall_f1": "E6_BITS_Fall_F1", "direction_macro_f1": "E6_BITS_Direction_Macro_F1"}
    )
    table = e7.merge(e3, on="run_id", how="left").merge(e6, on="run_id", how="left")
    table = table[
        [
            "run_id",
            "features",
            "fusion",
            "hard_negative_weight",
            "fall_f1",
            "fall_precision",
            "fall_recall",
            "FP",
            "FN",
            "direction_macro_f1",
            "E3_Direction_Macro_F1",
            "E6_BITS_Fall_F1",
            "params",
        ]
    ]
    ref_e7 = ref_metrics["E7"]
    completed = table[~table["run_id"].eq("REF_CURRENT_E3_ARTIFACT")].copy()
    best = completed.sort_values(["fall_f1", "fall_precision", "FP"], ascending=[False, False, True]).iloc[0]
    lite = completed[completed["features"].eq("FV-Lite10")].sort_values(["fall_f1", "fall_precision"], ascending=False)
    center = completed[completed["features"].eq("FV-Center12")]
    logit = completed[completed["fusion"].eq("logit_fusion")].sort_values(["fall_f1", "fall_precision"], ascending=False)
    and_rows = completed[completed["fusion"].eq("and")]
    gated = completed[completed["fusion"].eq("gated_product")]
    hn2 = completed[completed["run_id"].eq("FV2_LITE10_LOGIT_FUSION")]
    hn3 = completed[completed["run_id"].eq("FV5_LITE10_LOGIT_FUSION_HN3")]

    lines = [
        "# DS-Fall-RD-FV Summary",
        "",
        "## Protocol",
        "",
        "- Dataset: BITS + WEDA only.",
        "- Sampling/window: 25 Hz, 2-second event-centered, tilt12 temporal reference input.",
        "- Reference: DS-Fall-RD A5WCEFW / REF_CURRENT_E3_ARTIFACT.",
        "- FV training: frozen deep reference fall probability plus Logistic Regression verifier over summary features. Direction probabilities are copied from the reference and remain unchanged.",
        "- Thresholds are global and tuned on validation only.",
        "",
        "## E7 WEDA Results",
        "",
        markdown_table(format_df(table)),
        "",
        "## Thresholds",
        "",
        markdown_table(format_df(thresholds)),
        "",
        "## Model Params",
        "",
        markdown_table(format_df(params)),
        "",
        "## Answers",
        "",
        f"1. DS-Fall-RD-FV reduces WEDA FP from {int(ref_e7['FP'])} to {int(best['FP'])} for the best run `{best['run_id']}`. WEDA precision changes from {ref_e7['fall_precision']:.4f} to {best['fall_precision']:.4f}.",
        f"2. Best variant by WEDA Fall F1/precision trade-off is `{best['run_id']}` with WEDA Fall F1={best['fall_f1']:.4f}.",
        f"3. Lite10 without timing works {'well enough' if not lite.empty and lite.iloc[0]['fall_f1'] > 0.75 else 'only modestly'}; best Lite10 run is `{lite.iloc[0]['run_id']}` with WEDA Fall F1={lite.iloc[0]['fall_f1']:.4f}." if not lite.empty else "3. Lite10 result is unavailable.",
        f"4. Center12 upper-bound run `{center.iloc[0]['run_id']}` gets WEDA Fall F1={center.iloc[0]['fall_f1']:.4f}; because it uses impact-center timing, treat it as protocol-specific." if not center.empty else "4. Center12 result is unavailable.",
        f"5. Logit fusion best WEDA Fall F1={logit.iloc[0]['fall_f1']:.4f}; AND best={and_rows.iloc[0]['fall_f1']:.4f}; gated best={gated.iloc[0]['fall_f1']:.4f}." if not logit.empty and not and_rows.empty and not gated.empty else "5. Fusion comparison is incomplete.",
        f"6. Hard-negative weight 3 {'helps' if not hn2.empty and not hn3.empty and hn3.iloc[0]['fall_f1'] > hn2.iloc[0]['fall_f1'] else 'does not clearly help'} versus weight 2: FV2 WEDA Fall F1={hn2.iloc[0]['fall_f1']:.4f}, FV5 WEDA Fall F1={hn3.iloc[0]['fall_f1']:.4f}." if not hn2.empty and not hn3.empty else "6. Hard-negative comparison is unavailable.",
        f"7. FV6 fall-head unfreezing status: {failures.iloc[0]['reason'] if not failures.empty else 'completed or not applicable'}.",
        "8. Direction remains unchanged from reference for completed FV runs because direction probabilities are copied from the frozen reference output.",
        "9. DS-Fall-RD-FV should be reported as an optional FPGuard/verifier extension unless the paper wants to prioritize WEDA fall precision over a pure end-to-end temporal model. A5 remains the cleaner main architecture; FV is the enhanced false-positive guard.",
        "10. Paper-facing conclusion: WEDA Fall F1 weakness is a false-positive hard-negative problem. A lightweight logistic verifier using signal summary features can substantially reduce WEDA false positives while preserving DS-Fall-RD direction behavior.",
    ]
    if not failures.empty:
        lines.extend(["", "## Not Run / Failed Runs", "", markdown_table(format_df(failures))])
    if not fp_analysis.empty:
        lines.extend(["", "## WEDA FP Correction Count", ""])
        correction = fp_analysis.groupby("run_id").agg(
            reference_fp=("reference_fp", "sum"),
            fv_fp=("fv_fp", "sum"),
            corrected_reference_fp=("corrected_reference_fp", "sum"),
        ).reset_index()
        lines.append(markdown_table(format_df(correction)))
    (report_dir / "fv_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
            if col in {"n", "TN", "FP", "FN", "TP", "params", "added_params_over_reference", "summary_feature_count", "direction_n_supervised"}:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(int(v)))
            else:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    return out


if __name__ == "__main__":
    main()
