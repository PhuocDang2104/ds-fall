from __future__ import annotations

import argparse
import json
import math
import sys
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
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_25hz_a5wcefw_bits_weda as base25
from src.config import make_config
from src.utils.io import ensure_dir


SAMPLING_RATE = 25.0
DT = 1.0 / SAMPLING_RATE
CENTER_INDEX = int(SAMPLING_RATE * 2.0 / 2.0)
RAW6 = ["ax", "ay", "az", "gx", "gy", "gz"]
TILT12 = ["ax", "ay", "az", "gx", "gy", "gz", "acc_mag", "gyro_mag", "jerk", "roll", "pitch", "tilt_delta"]
GROUP_ORDER = ["BITS fall", "BITS non-fall", "WEDA fall", "WEDA non-fall", "WEDA TP", "WEDA FN", "WEDA FP", "WEDA TN"]


@dataclass(frozen=True)
class Paths:
    project_root: Path
    artifact_dir: Path
    report_dir: Path
    figure_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze why E7 WEDA fall F1 is lower than BITS in 25 Hz event-centered benchmark.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "experiments_25hz_event")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths(
        project_root=args.project_root.resolve(),
        artifact_dir=args.artifact_dir.resolve(),
        report_dir=ensure_dir(args.output_dir / "reports" / "data_analysis_25hz_center"),
        figure_dir=ensure_dir(args.output_dir / "figures" / "data_analysis_25hz_center"),
    )
    ensure_dir(paths.figure_dir / "feature_distributions")
    ensure_dir(paths.figure_dir / "temporal_curves")

    config = make_config(paths.project_root)
    print("=== 25 Hz event-centered fall-gap data analysis ===")
    print(f"project_root={paths.project_root}")
    print(f"artifact_dir={paths.artifact_dir}")
    print("Rebuilding event-centered BITS/WEDA windows for raw6/tilt12 features; no DS-Fall training is run.")

    data = base25.build_25hz_dataset(config, window_mode="event_centered")
    base25.validate_25hz_dataset(data)

    metadata = data["metadata"].copy().reset_index(drop=True)
    X_raw6 = np.asarray(data["X_raw6"], dtype=np.float32)
    X_tilt12 = np.asarray(data["X_features"], dtype=np.float32)
    y_fall = np.asarray(data["y_fall"], dtype=np.int64)
    if X_tilt12.shape[-1] != len(TILT12):
        raise ValueError(f"Expected tilt12 features, got shape {X_tilt12.shape}")

    summary = extract_summary_features(X_raw6, X_tilt12)
    feature_df = pd.concat([metadata.reset_index(drop=True), summary], axis=1)

    pred_df, prediction_note = load_reference_predictions(paths.artifact_dir)
    analysis_df = merge_predictions(feature_df, pred_df)
    if analysis_df["fall_prob"].isna().any():
        total_missing = int(analysis_df["fall_prob"].isna().sum())
        test_missing = int(
            analysis_df.loc[analysis_df["split"].astype(str).str.lower().eq("test"), "fall_prob"].isna().sum()
        )
        if test_missing:
            prediction_note += (
                f"\nWARNING: {test_missing} test windows and {total_missing} total windows had no saved reference prediction; "
                "prediction-dependent analysis uses matched rows only."
            )
        else:
            prediction_note += (
                f"\nSaved predictions cover all test windows. {total_missing} train/val windows have no saved prediction, "
                "which is expected because the reference artifact stores test predictions."
            )
    analysis_df["fall_true_int"] = y_fall
    analysis_df["pred_fall_int"] = analysis_df["pred_fall"].map({"non_fall": 0, "fall": 1}).fillna(-1).astype(int)
    analysis_df["fall_error_group"] = error_group(analysis_df["fall_true_int"], analysis_df["pred_fall_int"])
    analysis_df.to_csv(paths.report_dir / "summary_feature_matrix_with_predictions.csv", index=False)

    test_df = analysis_df[analysis_df["split"].astype(str).str.lower() == "test"].copy()
    test_df = test_df[test_df["dataset"].astype(str).str.lower().isin(["bits", "weda"])].copy()
    if test_df.empty:
        raise ValueError("No BITS/WEDA test rows available for analysis.")

    numeric_features = list(summary.columns)
    univariate = run_univariate_analysis(test_df, numeric_features)
    univariate.to_csv(paths.report_dir / "univariate_fall_importance.csv", index=False)

    weda_error = run_weda_error_analysis(test_df, numeric_features, univariate, top_n=max(25, args.top_n))
    weda_error.to_csv(paths.report_dir / "weda_error_feature_comparison.csv", index=False)

    prob_summary = fall_probability_distribution(test_df)
    prob_summary.to_csv(paths.report_dir / "fall_probability_distribution.csv", index=False)

    impact_report = impact_alignment_report(test_df)
    impact_report.to_csv(paths.report_dir / "impact_alignment_report.csv", index=False)

    classical = run_classical_models(analysis_df, numeric_features, seed=args.seed)
    classical.to_csv(paths.report_dir / "classical_fall_feature_importance.csv", index=False)

    top_features = choose_top_features(univariate, args.top_n)
    plot_probability_distribution(test_df, paths.figure_dir)
    plot_impact_distribution(test_df, paths.figure_dir)
    plot_feature_distributions(test_df, top_features, paths.figure_dir)
    plot_error_distributions(test_df, top_features, paths.figure_dir)
    plot_temporal_curves(test_df, X_tilt12, paths.figure_dir)

    write_report(
        paths=paths,
        test_df=test_df,
        univariate=univariate,
        weda_error=weda_error,
        prob_summary=prob_summary,
        impact_report=impact_report,
        classical=classical,
        prediction_note=prediction_note,
        top_features=top_features,
    )
    print(f"Saved report to {paths.report_dir / 'data_analysis_summary.md'}")


def extract_summary_features(X_raw6: np.ndarray, X_tilt12: np.ndarray) -> pd.DataFrame:
    rows: dict[str, np.ndarray] = {}
    raw = {name: X_raw6[:, :, i] for i, name in enumerate(RAW6)}
    feat = {name: X_tilt12[:, :, i] for i, name in enumerate(TILT12)}

    for name in RAW6:
        x = raw[name]
        rows[f"{name}_mean"] = np.mean(x, axis=1)
        rows[f"{name}_std"] = np.std(x, axis=1)
        rows[f"{name}_min"] = np.min(x, axis=1)
        rows[f"{name}_max"] = np.max(x, axis=1)
        rows[f"{name}_range"] = rows[f"{name}_max"] - rows[f"{name}_min"]

    for name, stats_list in {
        "acc_mag": ["mean", "std", "max", "p95", "range"],
        "gyro_mag": ["mean", "std", "max", "p95", "range"],
        "jerk": ["mean", "std", "max", "p95"],
        "roll": ["mean", "std", "min", "max", "range", "final_initial"],
        "pitch": ["mean", "std", "min", "max", "range", "final_initial"],
        "tilt_delta": ["mean", "std", "max", "p95", "final"],
    }.items():
        x = feat[name]
        if "mean" in stats_list:
            rows[f"{name}_mean"] = np.mean(x, axis=1)
        if "std" in stats_list:
            rows[f"{name}_std"] = np.std(x, axis=1)
        if "min" in stats_list:
            rows[f"{name}_min"] = np.min(x, axis=1)
        if "max" in stats_list:
            rows[f"{name}_max"] = np.max(x, axis=1)
        if "p95" in stats_list:
            rows[f"{name}_p95"] = np.percentile(x, 95, axis=1)
        if "range" in stats_list:
            rows[f"{name}_range"] = np.max(x, axis=1) - np.min(x, axis=1)
        if "final_initial" in stats_list:
            rows[f"{name}_final_initial"] = x[:, -1] - x[:, 0]
        if "final" in stats_list:
            rows[f"{name}_final"] = x[:, -1]

    acc_mag = feat["acc_mag"]
    gyro_mag = feat["gyro_mag"]
    impact_index = np.argmax(acc_mag, axis=1)
    gyro_peak_index = np.argmax(gyro_mag, axis=1)
    rows["impact_index"] = impact_index.astype(float)
    rows["gyro_peak_index"] = gyro_peak_index.astype(float)
    rows["impact_distance_from_center"] = np.abs(impact_index - CENTER_INDEX).astype(float)
    rows["gyro_peak_distance_from_center"] = np.abs(gyro_peak_index - CENTER_INDEX).astype(float)

    pre_energy = []
    post_energy = []
    post_acc_std = []
    post_gyro_std = []
    for i, idx in enumerate(impact_index):
        pre = acc_mag[i, : max(idx, 1)]
        post = acc_mag[i, min(idx + 1, acc_mag.shape[1] - 1) :]
        gpost = gyro_mag[i, min(idx + 1, gyro_mag.shape[1] - 1) :]
        pre_energy.append(float(np.sum(pre * pre) * DT))
        post_energy.append(float(np.sum(post * post) * DT))
        post_acc_std.append(float(np.std(post)) if len(post) else 0.0)
        post_gyro_std.append(float(np.std(gpost)) if len(gpost) else 0.0)
    rows["pre_impact_energy"] = np.asarray(pre_energy)
    rows["post_impact_energy"] = np.asarray(post_energy)
    rows["post_acc_mag_std"] = np.asarray(post_acc_std)
    rows["post_gyro_mag_std"] = np.asarray(post_gyro_std)
    rows["pre_post_energy_ratio"] = rows["pre_impact_energy"] / np.maximum(rows["post_impact_energy"], 1e-6)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_reference_predictions(artifact_dir: Path) -> tuple[pd.DataFrame, str]:
    candidates = [
        artifact_dir / "E3_BITS_WEDA_MIXED" / "predictions.csv",
        artifact_dir / "E6_E3_MIXED_TEST_BITS" / "predictions.csv",
        artifact_dir / "E7_E3_MIXED_TEST_WEDA" / "predictions.csv",
    ]
    frames = []
    used = []
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
            used.append(str(path))
    if not frames:
        return pd.DataFrame(columns=["window_id"]), "No saved reference prediction CSV was found; prediction-dependent analysis is limited."
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["window_id"], keep="first")
    note = "Loaded saved reference predictions from:\n" + "\n".join(f"- {p}" for p in used)
    return out, note


def merge_predictions(feature_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["window_id", "pred_fall", "fall_prob", "pred_direction", "direction_confidence", "experiment_id"]
    available = [c for c in cols if c in pred_df.columns]
    merged = feature_df.merge(pred_df[available], on="window_id", how="left", suffixes=("", "_pred"))
    if "pred_fall" not in merged.columns:
        merged["pred_fall"] = np.nan
    if "fall_prob" not in merged.columns:
        merged["fall_prob"] = np.nan
    return merged


def error_group(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> list[str]:
    groups = []
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            groups.append("TP")
        elif true == 1 and pred == 0:
            groups.append("FN")
        elif true == 0 and pred == 1:
            groups.append("FP")
        elif true == 0 and pred == 0:
            groups.append("TN")
        else:
            groups.append("unmatched")
    return groups


def run_univariate_analysis(test_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    specs = [
        ("bits_fall_vs_nonfall", test_df["dataset"].str.lower().eq("bits"), "fall_label", 1, 0),
        ("weda_fall_vs_nonfall", test_df["dataset"].str.lower().eq("weda"), "fall_label", 1, 0),
        ("bits_fall_vs_weda_fall", test_df["fall_label"].eq(1), "dataset", "bits", "weda"),
        ("bits_nonfall_vs_weda_nonfall", test_df["fall_label"].eq(0), "dataset", "bits", "weda"),
    ]
    rows = []
    for comparison, base_mask, label_col, positive_value, negative_value in specs:
        subset = test_df.loc[base_mask].copy()
        if subset.empty:
            continue
        y = np.where(subset[label_col].astype(str if isinstance(positive_value, str) else int).eq(positive_value), 1, 0)
        valid = subset[label_col].astype(str if isinstance(positive_value, str) else int).isin([positive_value, negative_value]).to_numpy()
        subset = subset.loc[valid]
        y = y[valid]
        for feature in features:
            values = pd.to_numeric(subset[feature], errors="coerce").to_numpy(dtype=float)
            row = statistical_row(comparison, feature, y, values)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["comparison", "abs_auc"], ascending=[True, False])


def statistical_row(comparison: str, feature: str, y: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(values) & np.isfinite(y)
    y = y[mask].astype(int)
    values = values[mask].astype(float)
    pos = values[y == 1]
    neg = values[y == 0]
    out = {
        "comparison": comparison,
        "feature": feature,
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "median_pos": safe_median(pos),
        "median_neg": safe_median(neg),
        "iqr_pos": safe_iqr(pos),
        "iqr_neg": safe_iqr(neg),
        "cohens_d": cohens_d(pos, neg),
        "cliffs_delta": cliffs_delta(pos, neg),
        "ks_statistic": np.nan,
        "mannwhitney_p": np.nan,
        "auroc": np.nan,
        "abs_auc": np.nan,
        "average_precision": np.nan,
    }
    if len(np.unique(y)) == 2 and len(pos) > 0 and len(neg) > 0:
        try:
            auc = float(roc_auc_score(y, values))
            out["auroc"] = auc
            out["abs_auc"] = max(auc, 1.0 - auc)
        except Exception:
            pass
        try:
            out["average_precision"] = float(average_precision_score(y, values))
        except Exception:
            pass
        try:
            out["ks_statistic"] = float(stats.ks_2samp(pos, neg).statistic)
        except Exception:
            pass
        try:
            out["mannwhitney_p"] = float(stats.mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
        except Exception:
            pass
    return out


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / max(len(a) + len(b) - 2, 1)
    if pooled <= 1e-12:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / math.sqrt(pooled))


def cliffs_delta(a: np.ndarray, b: np.ndarray, max_pairs: int = 2_000_000) -> float:
    if len(a) == 0 or len(b) == 0:
        return np.nan
    total = len(a) * len(b)
    if total <= max_pairs:
        diff = a[:, None] - b[None, :]
        return float((np.sum(diff > 0) - np.sum(diff < 0)) / total)
    rng = np.random.default_rng(42)
    ai = rng.choice(a, size=max_pairs, replace=True)
    bi = rng.choice(b, size=max_pairs, replace=True)
    return float((np.sum(ai > bi) - np.sum(ai < bi)) / max_pairs)


def safe_median(x: np.ndarray) -> float:
    return float(np.median(x)) if len(x) else np.nan


def safe_iqr(x: np.ndarray) -> float:
    return float(np.percentile(x, 75) - np.percentile(x, 25)) if len(x) else np.nan


def run_weda_error_analysis(test_df: pd.DataFrame, features: list[str], univariate: pd.DataFrame, top_n: int) -> pd.DataFrame:
    weda = test_df[test_df["dataset"].str.lower().eq("weda")].copy()
    top = choose_top_features(univariate, top_n)
    comparisons = [("weda_tp_vs_fn", "TP", "FN"), ("weda_fp_vs_tn", "FP", "TN"), ("weda_fn_vs_bits_tp", "FN", "BITS_TP")]
    bits_tp = test_df[(test_df["dataset"].str.lower().eq("bits")) & (test_df["fall_error_group"].eq("TP"))].copy()
    rows = []
    for comparison, pos_group, neg_group in comparisons:
        if neg_group == "BITS_TP":
            pos_df = weda[weda["fall_error_group"].eq(pos_group)]
            neg_df = bits_tp
        else:
            pos_df = weda[weda["fall_error_group"].eq(pos_group)]
            neg_df = weda[weda["fall_error_group"].eq(neg_group)]
        for feature in top:
            pos = pd.to_numeric(pos_df[feature], errors="coerce").dropna().to_numpy(dtype=float)
            neg = pd.to_numeric(neg_df[feature], errors="coerce").dropna().to_numpy(dtype=float)
            row = {
                "comparison": comparison,
                "feature": feature,
                "pos_group": pos_group,
                "neg_group": neg_group,
                "n_pos": int(len(pos)),
                "n_neg": int(len(neg)),
                "median_pos": safe_median(pos),
                "median_neg": safe_median(neg),
                "iqr_pos": safe_iqr(pos),
                "iqr_neg": safe_iqr(neg),
                "median_delta": safe_median(pos) - safe_median(neg) if len(pos) and len(neg) else np.nan,
                "cohens_d": cohens_d(pos, neg),
                "ks_statistic": np.nan,
                "mannwhitney_p": np.nan,
            }
            if len(pos) and len(neg):
                try:
                    row["ks_statistic"] = float(stats.ks_2samp(pos, neg).statistic)
                except Exception:
                    pass
                try:
                    row["mannwhitney_p"] = float(stats.mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
                except Exception:
                    pass
            rows.append(row)
    return pd.DataFrame(rows)


def fall_probability_distribution(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {
        "bits_fall": test_df["dataset"].str.lower().eq("bits") & test_df["fall_label"].eq(1),
        "bits_nonfall": test_df["dataset"].str.lower().eq("bits") & test_df["fall_label"].eq(0),
        "weda_fall": test_df["dataset"].str.lower().eq("weda") & test_df["fall_label"].eq(1),
        "weda_nonfall": test_df["dataset"].str.lower().eq("weda") & test_df["fall_label"].eq(0),
        "weda_tp": test_df["dataset"].str.lower().eq("weda") & test_df["fall_error_group"].eq("TP"),
        "weda_fn": test_df["dataset"].str.lower().eq("weda") & test_df["fall_error_group"].eq("FN"),
        "weda_fp": test_df["dataset"].str.lower().eq("weda") & test_df["fall_error_group"].eq("FP"),
        "weda_tn": test_df["dataset"].str.lower().eq("weda") & test_df["fall_error_group"].eq("TN"),
    }
    for name, mask in groups.items():
        p = pd.to_numeric(test_df.loc[mask, "fall_prob"], errors="coerce").dropna().to_numpy(dtype=float)
        rows.append(
            {
                "group": name,
                "n": int(len(p)),
                "mean_fall_prob": float(np.mean(p)) if len(p) else np.nan,
                "median_fall_prob": safe_median(p),
                "p05": float(np.percentile(p, 5)) if len(p) else np.nan,
                "p25": float(np.percentile(p, 25)) if len(p) else np.nan,
                "p75": float(np.percentile(p, 75)) if len(p) else np.nan,
                "p95": float(np.percentile(p, 95)) if len(p) else np.nan,
                "min": float(np.min(p)) if len(p) else np.nan,
                "max": float(np.max(p)) if len(p) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def impact_alignment_report(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {
        "bits_fall": test_df["dataset"].str.lower().eq("bits") & test_df["fall_label"].eq(1),
        "weda_fall": test_df["dataset"].str.lower().eq("weda") & test_df["fall_label"].eq(1),
        "weda_tp": test_df["dataset"].str.lower().eq("weda") & test_df["fall_error_group"].eq("TP"),
        "weda_fn": test_df["dataset"].str.lower().eq("weda") & test_df["fall_error_group"].eq("FN"),
        "weda_nonfall": test_df["dataset"].str.lower().eq("weda") & test_df["fall_label"].eq(0),
        "weda_fp": test_df["dataset"].str.lower().eq("weda") & test_df["fall_error_group"].eq("FP"),
    }
    for name, mask in groups.items():
        sub = test_df.loc[mask]
        idx = pd.to_numeric(sub["impact_index"], errors="coerce").dropna().to_numpy(dtype=float)
        peak = pd.to_numeric(sub["acc_mag_max"], errors="coerce").dropna().to_numpy(dtype=float)
        dist = np.abs(idx - CENTER_INDEX)
        rows.append(
            {
                "group": name,
                "n": int(len(idx)),
                "impact_index_mean": float(np.mean(idx)) if len(idx) else np.nan,
                "impact_index_median": safe_median(idx),
                "impact_index_iqr": safe_iqr(idx),
                "distance_from_center_median": safe_median(dist),
                "distance_from_center_mean": float(np.mean(dist)) if len(dist) else np.nan,
                "percent_center_20_30": float(np.mean((idx >= 20) & (idx <= 30)) * 100.0) if len(idx) else np.nan,
                "percent_early_lt20": float(np.mean(idx < 20) * 100.0) if len(idx) else np.nan,
                "percent_late_gt30": float(np.mean(idx > 30) * 100.0) if len(idx) else np.nan,
                "acc_mag_peak_median": safe_median(peak),
                "acc_mag_peak_iqr": safe_iqr(peak),
            }
        )
    return pd.DataFrame(rows)


def run_classical_models(df: pd.DataFrame, features: list[str], seed: int) -> pd.DataFrame:
    train = df[df["split"].astype(str).str.lower().eq("train")].copy()
    val = df[df["split"].astype(str).str.lower().eq("val")].copy()
    test = df[df["split"].astype(str).str.lower().eq("test")].copy()
    train_fit = pd.concat([train, val], ignore_index=True)
    X_train = train[features].to_numpy(dtype=float)
    y_train = train["fall_label"].to_numpy(dtype=int)
    X_val = val[features].to_numpy(dtype=float)
    y_val = val["fall_label"].to_numpy(dtype=int)
    X_fit = train_fit[features].to_numpy(dtype=float)
    y_fit = train_fit["fall_label"].to_numpy(dtype=int)
    X_test = test[features].to_numpy(dtype=float)
    y_test = test["fall_label"].to_numpy(dtype=int)

    models: list[tuple[str, Any, str]] = [
        (
            "logistic_regression",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed, solver="lbfgs"),
            ),
            "coef",
        ),
        (
            "random_forest",
            RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1, max_depth=None),
            "feature_importances",
        ),
    ]
    try:
        import xgboost as xgb  # type: ignore

        models.append(("xgboost", xgb.XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=seed), "feature_importances"))
    except Exception:
        models.append(("gradient_boosting", GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, random_state=seed), "feature_importances"))
        models.append(("hist_gradient_boosting", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, random_state=seed), "permutation_only"))

    rows = []
    for model_name, model, importance_kind in models:
        model.fit(X_train, y_train)
        val_prob = predict_positive_prob(model, X_val)
        threshold = tune_f1_threshold(y_val, val_prob)
        if len(np.unique(y_fit)) == 2:
            model.fit(X_fit, y_fit)
        test_prob = predict_positive_prob(model, X_test)
        test_pred = (test_prob >= threshold).astype(int)
        overall_auroc = safe_auc(y_test, test_prob)
        overall_f1 = f1_score(y_test, test_pred, zero_division=0)
        bits_mask = test["dataset"].str.lower().eq("bits").to_numpy()
        weda_mask = test["dataset"].str.lower().eq("weda").to_numpy()
        bits_f1 = f1_score(y_test[bits_mask], test_pred[bits_mask], zero_division=0) if bits_mask.any() else np.nan
        weda_f1 = f1_score(y_test[weda_mask], test_pred[weda_mask], zero_division=0) if weda_mask.any() else np.nan
        bits_auroc = safe_auc(y_test[bits_mask], test_prob[bits_mask]) if bits_mask.any() else np.nan
        weda_auroc = safe_auc(y_test[weda_mask], test_prob[weda_mask]) if weda_mask.any() else np.nan
        perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=seed, scoring="roc_auc", n_jobs=-1)
        base_importance = model_feature_importance(model, features, importance_kind)
        for i, feature in enumerate(features):
            rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "importance": float(base_importance.get(feature, np.nan)),
                    "permutation_importance_mean": float(perm.importances_mean[i]),
                    "permutation_importance_std": float(perm.importances_std[i]),
                    "threshold_tuned_on_val": float(threshold),
                    "test_auroc": overall_auroc,
                    "test_f1": float(overall_f1),
                    "bits_test_auroc": bits_auroc,
                    "bits_test_f1": float(bits_f1),
                    "weda_test_auroc": weda_auroc,
                    "weda_test_f1": float(weda_f1),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "permutation_importance_mean"], ascending=[True, False])


def predict_positive_prob(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-z))
    return np.asarray(model.predict(X), dtype=float)


def tune_f1_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        f1 = f1_score(y, (prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def safe_auc(y: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, prob))
    except Exception:
        return np.nan


def model_feature_importance(model: Any, features: list[str], kind: str) -> dict[str, float]:
    target = model
    if hasattr(model, "steps"):
        target = model.steps[-1][1]
    if kind == "coef" and hasattr(target, "coef_"):
        vals = np.abs(target.coef_).ravel()
    elif hasattr(target, "feature_importances_"):
        vals = np.asarray(target.feature_importances_, dtype=float)
    else:
        vals = np.full(len(features), np.nan)
    return {feature: float(vals[i]) for i, feature in enumerate(features)}


def choose_top_features(univariate: pd.DataFrame, top_n: int) -> list[str]:
    focus = univariate[univariate["comparison"].isin(["bits_fall_vs_nonfall", "weda_fall_vs_nonfall"])].copy()
    if focus.empty:
        focus = univariate.copy()
    ranked = focus.groupby("feature", as_index=False)["abs_auc"].max().sort_values("abs_auc", ascending=False)
    return ranked["feature"].head(top_n).tolist()


def plot_probability_distribution(test_df: pd.DataFrame, figure_dir: Path) -> None:
    df = test_df.copy()
    dataset_prefix = pd.Series(np.where(df["dataset"].str.lower().eq("bits"), "BITS ", "WEDA "), index=df.index)
    fall_suffix = pd.Series(np.where(df["fall_label"].eq(1), "fall", "non-fall"), index=df.index)
    df["prob_group"] = dataset_prefix + fall_suffix
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="fall_prob", hue="prob_group", bins=40, stat="density", common_norm=False, element="step")
    plt.title("Reference fall probability distribution")
    plt.xlabel("fall probability")
    plt.tight_layout()
    plt.savefig(figure_dir / "fall_probability_histogram.png", dpi=180)
    plt.close()


def plot_impact_distribution(test_df: pd.DataFrame, figure_dir: Path) -> None:
    df = test_df.copy()
    df["impact_group"] = np.select(
        [
            df["dataset"].str.lower().eq("bits") & df["fall_label"].eq(1),
            df["dataset"].str.lower().eq("weda") & df["fall_label"].eq(1),
            df["dataset"].str.lower().eq("weda") & df["fall_error_group"].eq("FP"),
            df["dataset"].str.lower().eq("weda") & df["fall_error_group"].eq("TN"),
        ],
        ["BITS fall", "WEDA fall", "WEDA FP", "WEDA TN"],
        default="other",
    )
    df = df[df["impact_group"].ne("other")]
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="impact_index", hue="impact_group", bins=np.arange(0, 52, 2), multiple="layer", stat="density", common_norm=False)
    plt.axvline(CENTER_INDEX, color="black", linestyle="--", linewidth=1)
    plt.title("Impact index distribution")
    plt.tight_layout()
    plt.savefig(figure_dir / "impact_index_distribution.png", dpi=180)
    plt.close()


def plot_feature_distributions(test_df: pd.DataFrame, top_features: list[str], figure_dir: Path) -> None:
    out_dir = ensure_dir(figure_dir / "feature_distributions")
    for feature in top_features:
        df = test_df.copy()
        dataset_prefix = pd.Series(np.where(df["dataset"].str.lower().eq("bits"), "BITS ", "WEDA "), index=df.index)
        fall_suffix = pd.Series(np.where(df["fall_label"].eq(1), "fall", "non-fall"), index=df.index)
        df["fall_group"] = dataset_prefix + fall_suffix
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(data=df, x="fall_group", y=feature, ax=axes[0])
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].set_title(f"{feature} boxplot")
        for group, sub in df.groupby("fall_group"):
            values = pd.to_numeric(sub[feature], errors="coerce").dropna()
            if len(values) > 1:
                sns.kdeplot(values, label=group, ax=axes[1])
        axes[1].set_title(f"{feature} KDE")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"fall_nonfall_{safe_name(feature)}.png", dpi=180)
        plt.close(fig)


def plot_error_distributions(test_df: pd.DataFrame, top_features: list[str], figure_dir: Path) -> None:
    out_dir = ensure_dir(figure_dir / "feature_distributions")
    weda = test_df[test_df["dataset"].str.lower().eq("weda")].copy()
    for feature in top_features:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(data=weda[weda["fall_error_group"].isin(["TP", "FN"])], x="fall_error_group", y=feature, ax=axes[0], order=["TP", "FN"])
        axes[0].set_title(f"WEDA TP vs FN: {feature}")
        sns.boxplot(data=weda[weda["fall_error_group"].isin(["FP", "TN"])], x="fall_error_group", y=feature, ax=axes[1], order=["FP", "TN"])
        axes[1].set_title(f"WEDA FP vs TN: {feature}")
        fig.tight_layout()
        fig.savefig(out_dir / f"weda_error_{safe_name(feature)}.png", dpi=180)
        plt.close(fig)


def plot_temporal_curves(test_df: pd.DataFrame, X_tilt12: np.ndarray, figure_dir: Path) -> None:
    out_dir = ensure_dir(figure_dir / "temporal_curves")
    index = test_df.index.to_numpy()
    channel_map = {name: i for i, name in enumerate(TILT12)}
    groups = {
        "BITS fall": test_df["dataset"].str.lower().eq("bits") & test_df["fall_label"].eq(1),
        "WEDA fall": test_df["dataset"].str.lower().eq("weda") & test_df["fall_label"].eq(1),
        "WEDA TP": test_df["dataset"].str.lower().eq("weda") & test_df["fall_error_group"].eq("TP"),
        "WEDA FN": test_df["dataset"].str.lower().eq("weda") & test_df["fall_error_group"].eq("FN"),
        "WEDA non-fall": test_df["dataset"].str.lower().eq("weda") & test_df["fall_label"].eq(0),
        "WEDA FP": test_df["dataset"].str.lower().eq("weda") & test_df["fall_error_group"].eq("FP"),
    }
    time = np.arange(X_tilt12.shape[1]) / SAMPLING_RATE
    for channel in ["acc_mag", "gyro_mag", "jerk", "tilt_delta"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for group, mask in groups.items():
            rows = index[mask.to_numpy()]
            if len(rows) == 0:
                continue
            x = X_tilt12[rows, :, channel_map[channel]]
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
            ax.plot(time, mean, label=f"{group} (n={len(rows)})")
            ax.fill_between(time, mean - std, mean + std, alpha=0.12)
        ax.axvline(CENTER_INDEX / SAMPLING_RATE, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Temporal mean +/- std: {channel}")
        ax.set_xlabel("seconds")
        ax.set_ylabel(channel)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"temporal_{channel}.png", dpi=180)
        plt.close(fig)


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in name)


def write_report(
    paths: Paths,
    test_df: pd.DataFrame,
    univariate: pd.DataFrame,
    weda_error: pd.DataFrame,
    prob_summary: pd.DataFrame,
    impact_report: pd.DataFrame,
    classical: pd.DataFrame,
    prediction_note: str,
    top_features: list[str],
) -> None:
    weda = test_df[test_df["dataset"].str.lower().eq("weda")]
    bits = test_df[test_df["dataset"].str.lower().eq("bits")]
    weda_cm = confusion_matrix(weda["fall_label"], weda["pred_fall_int"], labels=[0, 1])
    bits_cm = confusion_matrix(bits["fall_label"], bits["pred_fall_int"], labels=[0, 1])
    weda_prec, weda_rec, weda_f1, _ = precision_recall_fscore_support(weda["fall_label"], weda["pred_fall_int"], labels=[1], average="binary", zero_division=0)
    bits_prec, bits_rec, bits_f1, _ = precision_recall_fscore_support(bits["fall_label"], bits["pred_fall_int"], labels=[1], average="binary", zero_division=0)
    fp = int(weda_cm[0, 1])
    fn = int(weda_cm[1, 0])
    tp = int(weda_cm[1, 1])
    tn = int(weda_cm[0, 0])

    top_bits = top_univariate(univariate, "bits_fall_vs_nonfall", 10)
    top_weda = top_univariate(univariate, "weda_fall_vs_nonfall", 10)
    bits_weda_fall_shift = top_univariate(univariate, "bits_fall_vs_weda_fall", 10)
    nonfall_shift = top_univariate(univariate, "bits_nonfall_vs_weda_nonfall", 10)
    classical_top = top_classical(classical, 15)
    weda_fp_tn = weda_error[weda_error["comparison"].eq("weda_fp_vs_tn")].sort_values("ks_statistic", ascending=False).head(10)
    weda_tp_fn = weda_error[weda_error["comparison"].eq("weda_tp_vs_fn")].sort_values("ks_statistic", ascending=False).head(10)

    prob_lookup = prob_summary.set_index("group").to_dict(orient="index")
    impact_lookup = impact_report.set_index("group").to_dict(orient="index")

    lines = [
        "# Data Analysis: BITS vs WEDA Fall Gap at 25 Hz Event-centered",
        "",
        "Scope: BITS/WEDA only, 25 Hz, 2-second event-centered windows, tilt12 features. This is data analysis only; no DS-Fall architecture experiment was run.",
        "",
        "## Prediction Source",
        "",
        prediction_note,
        "",
        "## Reference Fall Metrics Recomputed From Saved Predictions",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    {"dataset": "bits", "n": len(bits), "TN": int(bits_cm[0, 0]), "FP": int(bits_cm[0, 1]), "FN": int(bits_cm[1, 0]), "TP": int(bits_cm[1, 1]), "precision": bits_prec, "recall": bits_rec, "fall_f1": bits_f1},
                    {"dataset": "weda", "n": len(weda), "TN": tn, "FP": fp, "FN": fn, "TP": tp, "precision": weda_prec, "recall": weda_rec, "fall_f1": weda_f1},
                ]
            )
        ),
        "",
        "## Top Variables",
        "",
        "Top fall/non-fall separators on BITS:",
        "",
        markdown_table(format_small(top_bits[["feature", "abs_auc", "cohens_d", "ks_statistic", "median_pos", "median_neg"]])),
        "",
        "Top fall/non-fall separators on WEDA:",
        "",
        markdown_table(format_small(top_weda[["feature", "abs_auc", "cohens_d", "ks_statistic", "median_pos", "median_neg"]])),
        "",
        "Largest BITS-fall vs WEDA-fall shifts:",
        "",
        markdown_table(format_small(bits_weda_fall_shift[["feature", "abs_auc", "cohens_d", "ks_statistic", "median_pos", "median_neg"]])),
        "",
        "Largest BITS-non-fall vs WEDA-non-fall shifts:",
        "",
        markdown_table(format_small(nonfall_shift[["feature", "abs_auc", "cohens_d", "ks_statistic", "median_pos", "median_neg"]])),
        "",
        "## WEDA Error Analysis",
        "",
        f"WEDA fall F1 is mainly limited by false positives: FP={fp}, FN={fn}, TP={tp}, TN={tn}. Recall is high ({weda_rec:.4f}), but precision is low ({weda_prec:.4f}).",
        "",
        "Strongest WEDA FP vs TN feature differences:",
        "",
        markdown_table(format_small(weda_fp_tn[["feature", "n_pos", "n_neg", "median_pos", "median_neg", "median_delta", "ks_statistic"]])),
        "",
        "WEDA TP vs FN feature differences. Interpret cautiously if FN count is very small:",
        "",
        markdown_table(format_small(weda_tp_fn[["feature", "n_pos", "n_neg", "median_pos", "median_neg", "median_delta", "ks_statistic"]])),
        "",
        "## Probability And Impact Alignment",
        "",
        markdown_table(format_small(prob_summary)),
        "",
        markdown_table(format_small(impact_report)),
        "",
        "## Classical Feature-importance Models",
        "",
        "Classical models are trained only on summary features using the existing train/val/test split. Thresholds are tuned on validation for F1, then evaluated on test.",
        "",
        markdown_table(format_small(classical_top)),
        "",
        "## Diagnosis",
        "",
        diagnosis_text(fp, fn, top_bits, top_weda, bits_weda_fall_shift, nonfall_shift, weda_fp_tn, prob_lookup, impact_lookup),
        "",
        "## Paper-facing Conclusion",
        "",
        paper_text(fp, fn, weda_prec, weda_rec, top_weda, nonfall_shift, impact_lookup),
        "",
        "## Output Files",
        "",
        "- `univariate_fall_importance.csv`",
        "- `weda_error_feature_comparison.csv`",
        "- `fall_probability_distribution.csv`",
        "- `impact_alignment_report.csv`",
        "- `classical_fall_feature_importance.csv`",
        "- `summary_feature_matrix_with_predictions.csv`",
        "- figures under `outputs/figures/data_analysis_25hz_center/`",
    ]
    (paths.report_dir / "data_analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def top_univariate(univariate: pd.DataFrame, comparison: str, n: int) -> pd.DataFrame:
    return univariate[univariate["comparison"].eq(comparison)].sort_values("abs_auc", ascending=False).head(n).copy()


def top_classical(classical: pd.DataFrame, n: int) -> pd.DataFrame:
    return classical.sort_values("permutation_importance_mean", ascending=False).head(n)[
        ["model", "feature", "importance", "permutation_importance_mean", "test_auroc", "test_f1", "bits_test_f1", "weda_test_f1"]
    ].copy()


def diagnosis_text(
    fp: int,
    fn: int,
    top_bits: pd.DataFrame,
    top_weda: pd.DataFrame,
    fall_shift: pd.DataFrame,
    nonfall_shift: pd.DataFrame,
    weda_fp_tn: pd.DataFrame,
    prob_lookup: dict[str, dict[str, float]],
    impact_lookup: dict[str, dict[str, float]],
) -> str:
    bits_top = ", ".join(top_bits["feature"].head(5).tolist())
    weda_top = ", ".join(top_weda["feature"].head(5).tolist())
    shift_top = ", ".join(fall_shift["feature"].head(5).tolist())
    nonfall_top = ", ".join(nonfall_shift["feature"].head(5).tolist())
    fp_top = ", ".join(weda_fp_tn["feature"].head(5).tolist())
    weda_fn_prob = prob_lookup.get("weda_fn", {}).get("median_fall_prob", np.nan)
    weda_fp_prob = prob_lookup.get("weda_fp", {}).get("median_fall_prob", np.nan)
    weda_tn_prob = prob_lookup.get("weda_tn", {}).get("median_fall_prob", np.nan)
    bits_peak = impact_lookup.get("bits_fall", {}).get("acc_mag_peak_median", np.nan)
    weda_peak = impact_lookup.get("weda_fall", {}).get("acc_mag_peak_median", np.nan)
    weda_fp_center = impact_lookup.get("weda_fp", {}).get("percent_center_20_30", np.nan)
    return "\n".join(
        [
            f"- Top variables affecting fall separation on BITS are: {bits_top}.",
            f"- Top variables affecting fall separation on WEDA are: {weda_top}.",
            f"- Features with the strongest BITS-fall vs WEDA-fall shift are: {shift_top}. This indicates that the fall class itself is not identically distributed across datasets.",
            f"- Features with the strongest BITS-non-fall vs WEDA-non-fall shift are: {nonfall_top}. This is evidence that WEDA ADL/non-fall windows have a different motion distribution.",
            f"- WEDA Fall F1 is lower mainly because of false positives, not false negatives: FP={fp}, FN={fn}.",
            f"- WEDA FP differs from WEDA TN most on: {fp_top}. These are likely hard-negative ADL windows that resemble fall-like high-motion segments.",
            f"- WEDA FN median fall probability is {weda_fn_prob:.4f}; WEDA FP median fall probability is {weda_fp_prob:.4f}; WEDA TN median fall probability is {weda_tn_prob:.4f}.",
            f"- Median fall acc peak is BITS={bits_peak:.4f}, WEDA={weda_peak:.4f}. WEDA FP impact-center percentage is {weda_fp_center:.2f}%, so some non-fall WEDA windows contain centered high-energy peaks that look event-like.",
            "- WEDA direction can remain good while fall is weak because direction uses signed roll/pitch/gyro patterns on supervised fall windows, whereas fall precision is hurt by WEDA non-fall windows whose impact/jerk/energy statistics overlap with falls.",
        ]
    )


def paper_text(
    fp: int,
    fn: int,
    weda_prec: float,
    weda_rec: float,
    top_weda: pd.DataFrame,
    nonfall_shift: pd.DataFrame,
    impact_lookup: dict[str, dict[str, float]],
) -> str:
    weda_top = ", ".join(top_weda["feature"].head(4).tolist())
    nonfall_top = ", ".join(nonfall_shift["feature"].head(4).tolist())
    bits_valid = impact_lookup.get("bits_fall", {}).get("percent_center_20_30", np.nan)
    weda_valid = impact_lookup.get("weda_fall", {}).get("percent_center_20_30", np.nan)
    return "\n".join(
        [
            f"On the BITS/WEDA 25 Hz event-centered benchmark, the lower WEDA fall F1 is not a direction-label failure. The E3 model keeps high WEDA fall recall ({weda_rec:.4f}) but has low precision ({weda_prec:.4f}), producing FP={fp} and FN={fn}.",
            f"The data-level explanation is that WEDA non-fall windows are closer to fall windows in high-motion summary features, especially {weda_top}, and WEDA non-fall distribution differs from BITS on {nonfall_top}.",
            f"Impact alignment is not the primary issue for true fall windows: centered-impact percentage is BITS={bits_valid:.2f}% and WEDA={weda_valid:.2f}%. The harder part is WEDA hard-negative ADL windows that create fall-like impact/jerk evidence.",
            "For the paper, report WEDA as a harder fall-precision subset in the clean BITS/WEDA benchmark, while noting that WEDA direction remains strong because signed orientation features remain separable on supervised fall windows.",
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


def format_small(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            if col.startswith("n") or col in {"FP", "FN", "TP", "TN"}:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(int(v)))
            else:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    return out


if __name__ == "__main__":
    main()
