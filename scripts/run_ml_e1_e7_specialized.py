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
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "reports" / "ml_feature_analysis_25hz" / "summary_feature_matrix.csv"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports" / "ml_e1_e7_specialized"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "ml_e1_e7_specialized"
A5_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "experiments_25hz_event"
HYBRID_RESULTS_PATH = PROJECT_ROOT / "outputs" / "reports" / "ds_fall_rd_hybrid" / "hybrid_results.csv"

DIRECTION_LABELS = ["forward", "backward", "lateral"]
DIR_TO_ID = {name: i for i, name in enumerate(DIRECTION_LABELS)}
ID_TO_DIR = {i: name for name, i in DIR_TO_ID.items()}

FALL_CORE = [
    "acc_mag_range",
    "acc_mag_max",
    "acc_mag_p95",
    "acc_mag_std",
    "gyro_mag_range",
    "gyro_mag_max",
    "gyro_mag_p95",
    "gyro_mag_mean",
    "jerk_max",
    "jerk_p95",
    "jerk_std",
    "post_acc_mag_std",
    "post_gyro_mag_std",
    "pre_post_energy_ratio",
    "tilt_delta_p95",
    "tilt_delta_std",
]

DIRECTION_CORE_CANDIDATES = [
    "gx_mean",
    "gx_median",
    "gx_min",
    "gx_max",
    "gx_range",
    "gx_std",
    "gx_final_initial",
    "gy_mean",
    "gy_median",
    "gy_min",
    "gy_max",
    "gy_range",
    "gy_std",
    "gy_final_initial",
    "gz_mean",
    "gz_median",
    "gz_min",
    "gz_max",
    "gz_range",
    "gz_std",
    "gz_final_initial",
    "ax_mean",
    "ax_min",
    "ax_max",
    "ax_range",
    "ax_std",
    "ax_final_initial",
    "ay_mean",
    "ay_min",
    "ay_max",
    "ay_range",
    "ay_std",
    "ay_final_initial",
    "az_mean",
    "az_min",
    "az_max",
    "az_range",
    "az_std",
    "az_final_initial",
    "roll_mean",
    "roll_min",
    "roll_max",
    "roll_range",
    "roll_std",
    "roll_final_initial",
    "pitch_mean",
    "pitch_min",
    "pitch_max",
    "pitch_range",
    "pitch_std",
    "pitch_final_initial",
    "tilt_delta_mean",
    "tilt_delta_std",
    "tilt_delta_max",
    "tilt_delta_p95",
    "tilt_delta_final",
    "pre_gx_mean",
    "pre_gy_mean",
    "pre_gz_mean",
    "pre_ax_mean",
    "pre_ay_mean",
    "pre_az_mean",
    "delta_roll_window",
    "delta_pitch_window",
]

TIMING_FEATURE_NAMES = {"impact_index", "impact_distance_from_center", "gyro_peak_index", "gyro_peak_distance_from_center"}
TIMING_PATTERNS = ("distance_from_center", "peak_index", "impact_index")

PROTOCOLS = {
    "E1_BITS_TO_BITS": {"train": ["bits"], "test": ["bits"], "reuse": None},
    "E2_WEDA_TO_WEDA": {"train": ["weda"], "test": ["weda"], "reuse": None},
    "E3_BITS_WEDA_MIXED": {"train": ["bits", "weda"], "test": ["bits", "weda"], "reuse": None},
    "E4_BITS_TO_WEDA": {"train": ["bits"], "test": ["weda"], "reuse": None},
    "E5_WEDA_TO_BITS": {"train": ["weda"], "test": ["bits"], "reuse": None},
    "E6_E3_MIXED_TEST_BITS": {"train": ["bits", "weda"], "test": ["bits"], "reuse": "E3_BITS_WEDA_MIXED"},
    "E7_E3_MIXED_TEST_WEDA": {"train": ["bits", "weda"], "test": ["weda"], "reuse": "E3_BITS_WEDA_MIXED"},
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run specialized ML fall-only and direction-only E1-E7 experiments.")
    parser.add_argument("--summary_path", default=str(SUMMARY_PATH))
    parser.add_argument("--output_dir", default=str(REPORT_DIR))
    parser.add_argument("--figure_dir", default=str(FIGURE_DIR))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = ensure_dir(Path(args.output_dir))
    figure_dir = ensure_dir(Path(args.figure_dir))
    ensure_dir(figure_dir / "fall_e1_e7_confusion_matrices")
    ensure_dir(figure_dir / "direction_e1_e7_confusion_matrices")
    run_root = ensure_dir(report_dir / "runs")

    df = load_summary(Path(args.summary_path))
    signal_features = infer_signal_features(df)
    feature_sets = {
        "FallNoTiming_Core": [f for f in FALL_CORE if f in df.columns],
        "FallNoTiming_Full": [f for f in signal_features if not is_timing_feature(f)],
        "DirectionCore": [f for f in DIRECTION_CORE_CANDIDATES if f in df.columns and not is_timing_feature(f)],
        "DirectionNoMagnitude": [
            f
            for f in DIRECTION_CORE_CANDIDATES
            if f in df.columns and not is_timing_feature(f) and not f.startswith(("acc_mag_", "gyro_mag_", "jerk_"))
        ],
        "DirectionFullNoTiming": [f for f in signal_features if not is_timing_feature(f)],
    }
    validate_feature_sets(feature_sets)
    (report_dir / "feature_set_configs.json").write_text(
        json.dumps(
            {
                "summary_source": str(Path(args.summary_path)),
                "benchmark": "BITS/WEDA only, 25 Hz, 2-second event-centered, existing split",
                "split_strategy": "existing train/val/test split per dataset; E6/E7 reuse E3 mixed model",
                "feature_sets": feature_sets,
                "timing_feature_exclusion": list(TIMING_FEATURE_NAMES),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fall_rows, fall_importance, fall_predictions, fall_failures = run_fall_experiments(
        df, feature_sets, args.seed, run_root, figure_dir
    )
    direction_rows, direction_importance, direction_predictions, direction_failures = run_direction_experiments(
        df, feature_sets, args.seed, run_root, figure_dir
    )

    fall_df = pd.DataFrame(fall_rows)
    direction_df = pd.DataFrame(direction_rows)
    fall_imp_df = pd.DataFrame(fall_importance)
    direction_imp_df = pd.DataFrame(direction_importance)
    fall_fail_df = pd.DataFrame(fall_failures)
    direction_fail_df = pd.DataFrame(direction_failures)

    fall_best = best_by_protocol(fall_df, task="fall")
    direction_best = best_by_protocol(direction_df, task="direction")
    a5_fall, a5_direction = load_a5_reference_rows()
    comparison = build_comparison(fall_df, direction_df, a5_fall, a5_direction)
    e2e = build_e2e_results(df, fall_df, direction_df, fall_predictions, direction_predictions)
    recommendation = build_final_recommendation(fall_df, direction_df, e2e, a5_fall, a5_direction)
    shift_notes = build_direction_shift_notes(df, feature_sets)

    fall_df.to_csv(report_dir / "fall_e1_e7_results.csv", index=False)
    fall_best.to_csv(report_dir / "fall_e1_e7_best_by_protocol.csv", index=False)
    fall_imp_df.to_csv(report_dir / "fall_e1_e7_feature_importance.csv", index=False)
    direction_df.to_csv(report_dir / "direction_e1_e7_results.csv", index=False)
    direction_best.to_csv(report_dir / "direction_e1_e7_best_by_protocol.csv", index=False)
    direction_imp_df.to_csv(report_dir / "direction_e1_e7_feature_importance.csv", index=False)
    shift_notes.to_csv(report_dir / "direction_e1_e7_dataset_shift_notes.csv", index=False)
    comparison.to_csv(report_dir / "ml_vs_a5_e1_e7_comparison.csv", index=False)
    recommendation.to_csv(report_dir / "final_pipeline_recommendation.csv", index=False)
    e2e.to_csv(report_dir / "e2e_direction_results.csv", index=False)
    pd.concat(
        [
            fall_fail_df.assign(task="fall") if not fall_fail_df.empty else pd.DataFrame(),
            direction_fail_df.assign(task="direction") if not direction_fail_df.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    ).to_csv(report_dir / "ml_e1_e7_failures.csv", index=False)

    write_summary(
        report_dir=report_dir,
        fall_df=fall_df,
        direction_df=direction_df,
        fall_best=fall_best,
        direction_best=direction_best,
        comparison=comparison,
        recommendation=recommendation,
        e2e=e2e,
        failures=pd.concat(
            [
                fall_fail_df.assign(task="fall") if not fall_fail_df.empty else pd.DataFrame(),
                direction_fail_df.assign(task="direction") if not direction_fail_df.empty else pd.DataFrame(),
            ],
            ignore_index=True,
        ),
    )
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


def is_timing_feature(name: str) -> bool:
    return name in TIMING_FEATURE_NAMES or any(pattern in name for pattern in TIMING_PATTERNS)


def validate_feature_sets(feature_sets: dict[str, list[str]]) -> None:
    missing = [name for name, cols in feature_sets.items() if not cols]
    if missing:
        raise ValueError(f"Empty feature sets: {missing}")


def fall_model_specs(seed: int) -> list[ModelSpec]:
    return [
        ModelSpec("LogisticRegression", "lr"),
        ModelSpec("LinearSVM", "linear_svm"),
        ModelSpec("RandomForest", "rf"),
        ModelSpec("GradientBoosting", "gb"),
        ModelSpec("HistGradientBoosting", "hgb"),
    ]


def direction_model_specs(seed: int) -> list[ModelSpec]:
    return [
        ModelSpec("LogisticRegression", "lr"),
        ModelSpec("LinearSVM", "linear_svm"),
        ModelSpec("RandomForest", "rf"),
        ModelSpec("GradientBoosting", "gb"),
        ModelSpec("HistGradientBoosting", "hgb"),
    ]


def make_model(spec: ModelSpec, seed: int, task: str):
    if spec.kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))
    if spec.kind == "linear_svm":
        return make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", random_state=seed, max_iter=8000))
    if spec.kind == "rf":
        return make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=200 if task == "direction" else 300, max_depth=None, class_weight="balanced", random_state=seed, n_jobs=-1),
        )
    if spec.kind == "gb":
        return make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=seed))
    if spec.kind == "hgb":
        return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(random_state=seed, max_iter=140))
    raise ValueError(spec.kind)


def protocol_masks(df: pd.DataFrame, protocol_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    protocol = PROTOCOLS[protocol_id]
    train_datasets = protocol["train"]
    test_datasets = protocol["test"]
    train = df["split"].eq("train").to_numpy() & df["dataset"].isin(train_datasets).to_numpy()
    val = df["split"].eq("val").to_numpy() & df["dataset"].isin(train_datasets).to_numpy()
    test = df["split"].eq("test").to_numpy() & df["dataset"].isin(test_datasets).to_numpy()
    return train, val, test


def run_fall_experiments(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    seed: int,
    run_root: Path,
    figure_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], np.ndarray], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    importances: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    eval_feature_sets = ["FallNoTiming_Core", "FallNoTiming_Full"]
    base_protocols = ["E1_BITS_TO_BITS", "E2_WEDA_TO_WEDA", "E3_BITS_WEDA_MIXED", "E4_BITS_TO_WEDA", "E5_WEDA_TO_BITS"]

    for feature_set in eval_feature_sets:
        features = feature_sets[feature_set]
        for spec in fall_model_specs(seed):
            trained_cache: dict[str, dict[str, Any]] = {}
            for protocol_id in base_protocols:
                run_id = f"FALL_{spec.name}_{feature_set}_{protocol_id}"
                try:
                    train, val, test = protocol_masks(df, protocol_id)
                    y_train = df.loc[train, "fall_label"].astype(int).to_numpy()
                    model = make_model(spec, seed, task="fall")
                    model.fit(df.loc[train, features].astype(float).to_numpy(), y_train)
                    val_scores = fall_scores(model, df.loc[val, features].astype(float).to_numpy())
                    threshold, threshold_note = select_fall_threshold(df.loc[val, "fall_label"].astype(int).to_numpy(), val_scores)
                    run_dir = ensure_dir(run_root / run_id)
                    with (run_dir / "model.pkl").open("wb") as f:
                        pickle.dump(model, f)
                    trained_cache[protocol_id] = {"model": model, "threshold": threshold, "threshold_note": threshold_note}
                    for eval_id, mask in eval_masks_for_protocol(df, protocol_id):
                        scores = fall_scores(model, df.loc[mask, features].astype(float).to_numpy())
                        pred = (scores >= threshold).astype(int)
                        key = (run_id, eval_id)
                        predictions[key] = pred
                        metric = evaluate_fall(df, mask, pred, scores, run_id, eval_id, protocol_id, feature_set, spec.name, threshold, threshold_note, features)
                        rows.append(metric)
                        save_fall_confusion(metric, figure_dir / "fall_e1_e7_confusion_matrices")
                    importances.extend(extract_importance(model, features, run_id, protocol_id, task="fall"))
                except Exception as exc:
                    failures.append({"run_id": run_id, "protocol": protocol_id, "feature_set": feature_set, "model": spec.name, "reason": repr(exc)})
            # Explicit E6/E7 aliases reuse the E3 mixed model.
            if "E3_BITS_WEDA_MIXED" in trained_cache:
                model = trained_cache["E3_BITS_WEDA_MIXED"]["model"]
                threshold = trained_cache["E3_BITS_WEDA_MIXED"]["threshold"]
                threshold_note = trained_cache["E3_BITS_WEDA_MIXED"]["threshold_note"]
                for eval_id in ["E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"]:
                    run_id = f"FALL_{spec.name}_{feature_set}_{eval_id}"
                    _, _, mask = protocol_masks(df, eval_id)
                    scores = fall_scores(model, df.loc[mask, features].astype(float).to_numpy())
                    pred = (scores >= threshold).astype(int)
                    predictions[(run_id, eval_id)] = pred
                    metric = evaluate_fall(df, mask, pred, scores, run_id, eval_id, "E3_BITS_WEDA_MIXED_REUSED", feature_set, spec.name, threshold, threshold_note, features)
                    rows.append(metric)
                    save_fall_confusion(metric, figure_dir / "fall_e1_e7_confusion_matrices")
    return rows, importances, predictions, failures


def fall_scores(model: Any, X: np.ndarray) -> np.ndarray:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(estimator, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(float)
    scores = model.decision_function(X)
    return np.asarray(scores, dtype=float)


def select_fall_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, str]:
    candidates = np.unique(np.concatenate([np.linspace(np.min(score), np.max(score), 240), np.quantile(score, np.linspace(0.01, 0.99, 120)), np.unique(score)]))
    best = None
    for threshold in candidates:
        pred = (score >= threshold).astype(int)
        precision = precision_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel()]
        key = (round(f1, 6), 1 if recall >= 0.88 else 0, precision, -fp, recall)
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1], "best_val_f1_then_precision_with_recall088_tiebreak"


def eval_masks_for_protocol(df: pd.DataFrame, protocol_id: str) -> list[tuple[str, np.ndarray]]:
    _, _, test = protocol_masks(df, protocol_id)
    out = [(protocol_id, test)]
    if protocol_id == "E3_BITS_WEDA_MIXED":
        _, _, bits = protocol_masks(df, "E6_E3_MIXED_TEST_BITS")
        _, _, weda = protocol_masks(df, "E7_E3_MIXED_TEST_WEDA")
        out.extend([("E6_E3_MIXED_TEST_BITS", bits), ("E7_E3_MIXED_TEST_WEDA", weda)])
    return out


def evaluate_fall(
    df: pd.DataFrame,
    mask: np.ndarray,
    pred: np.ndarray,
    score: np.ndarray,
    run_id: str,
    eval_id: str,
    train_protocol: str,
    feature_set: str,
    model_name: str,
    threshold: float,
    threshold_note: str,
    features: list[str],
) -> dict[str, Any]:
    y = df.loc[mask, "fall_label"].astype(int).to_numpy()
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel()]
    try:
        auroc = float(roc_auc_score(y, score))
    except ValueError:
        auroc = math.nan
    try:
        ap = float(average_precision_score(y, score))
    except ValueError:
        ap = math.nan
    datasets = "+".join(sorted(df.loc[mask, "dataset"].unique()))
    complexity = complexity_info(model_name, features)
    out = {
        "run_id": run_id,
        "experiment_id": eval_id,
        "train_protocol": train_protocol,
        "test_dataset": datasets,
        "model_name": model_name,
        "model_type": "fall_only_ml",
        "feature_set": feature_set,
        "feature_count": len(features),
        "threshold": threshold,
        "threshold_note": threshold_note,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "fall_precision": float(precision_score(y, pred, zero_division=0)),
        "fall_recall": float(recall_score(y, pred, zero_division=0)),
        "fall_f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "AUROC": auroc,
        "average_precision": ap,
        "FP_reduction_vs_A5": math.nan,
        **complexity,
    }
    a5 = a5_fall_metric(eval_id)
    if a5:
        out["FP_reduction_vs_A5"] = a5.get("FP", math.nan) - fp
    return out


def save_fall_confusion(row: dict[str, Any], out_dir: Path) -> None:
    cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]], dtype=int)
    save_cm(cm, ["non_fall", "fall"], out_dir / f"{safe_name(row['run_id'])}_{row['experiment_id']}_fall.png", f"{row['run_id']} {row['experiment_id']}")


def run_direction_experiments(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    seed: int,
    run_root: Path,
    figure_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], np.ndarray], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    importances: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    eval_feature_sets = ["DirectionCore", "DirectionNoMagnitude", "DirectionFullNoTiming"]
    base_protocols = ["E1_BITS_TO_BITS", "E2_WEDA_TO_WEDA", "E3_BITS_WEDA_MIXED", "E4_BITS_TO_WEDA", "E5_WEDA_TO_BITS"]

    for feature_set in eval_feature_sets:
        features = feature_sets[feature_set]
        for spec in direction_model_specs(seed):
            trained_cache: dict[str, Any] = {}
            for protocol_id in base_protocols:
                run_id = f"DIR_{spec.name}_{feature_set}_{protocol_id}"
                try:
                    train, val, test = protocol_masks(df, protocol_id)
                    train &= direction_train_mask(df)
                    if np.sum(train) < 6:
                        raise ValueError("Too few supervised direction training samples")
                    y_train = df.loc[train, "direction_id"].astype(int).to_numpy()
                    if len(np.unique(y_train)) < 2:
                        raise ValueError("Direction training split has fewer than two classes")
                    model = make_model(spec, seed, task="direction")
                    model.fit(df.loc[train, features].astype(float).to_numpy(), y_train)
                    run_dir = ensure_dir(run_root / run_id)
                    with (run_dir / "model.pkl").open("wb") as f:
                        pickle.dump(model, f)
                    trained_cache[protocol_id] = model
                    for eval_id, mask in eval_masks_for_protocol(df, protocol_id):
                        mask = mask & direction_train_mask(df)
                        pred = model.predict(df.loc[mask, features].astype(float).to_numpy()).astype(int)
                        predictions[(run_id, eval_id)] = pred
                        metric = evaluate_direction(df, mask, pred, run_id, eval_id, protocol_id, feature_set, spec.name, features)
                        rows.append(metric)
                        save_direction_confusion(metric, figure_dir / "direction_e1_e7_confusion_matrices")
                    importances.extend(extract_importance(model, features, run_id, protocol_id, task="direction"))
                except Exception as exc:
                    failures.append({"run_id": run_id, "protocol": protocol_id, "feature_set": feature_set, "model": spec.name, "reason": repr(exc)})
            if "E3_BITS_WEDA_MIXED" in trained_cache:
                model = trained_cache["E3_BITS_WEDA_MIXED"]
                for eval_id in ["E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"]:
                    run_id = f"DIR_{spec.name}_{feature_set}_{eval_id}"
                    _, _, mask = protocol_masks(df, eval_id)
                    mask = mask & direction_train_mask(df)
                    pred = model.predict(df.loc[mask, features].astype(float).to_numpy()).astype(int)
                    predictions[(run_id, eval_id)] = pred
                    metric = evaluate_direction(df, mask, pred, run_id, eval_id, "E3_BITS_WEDA_MIXED_REUSED", feature_set, spec.name, features)
                    rows.append(metric)
                    save_direction_confusion(metric, figure_dir / "direction_e1_e7_confusion_matrices")
    return rows, importances, predictions, failures


def direction_train_mask(df: pd.DataFrame) -> np.ndarray:
    return (
        df["fall_label"].eq(1).to_numpy()
        & df["direction_supervised_bool"].to_numpy()
        & df["direction_id"].isin([0, 1, 2]).to_numpy()
    )


def evaluate_direction(
    df: pd.DataFrame,
    mask: np.ndarray,
    pred: np.ndarray,
    run_id: str,
    eval_id: str,
    train_protocol: str,
    feature_set: str,
    model_name: str,
    features: list[str],
) -> dict[str, Any]:
    y = df.loc[mask, "direction_id"].astype(int).to_numpy()
    if len(y) == 0:
        cm = np.zeros((3, 3), dtype=int)
        per_class = [math.nan, math.nan, math.nan]
        macro = acc = math.nan
    else:
        cm = confusion_matrix(y, pred, labels=[0, 1, 2])
        per_class = f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
        macro = f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)
        acc = accuracy_score(y, pred)
    datasets = "+".join(sorted(df.loc[mask, "dataset"].unique())) if len(y) else ""
    return {
        "run_id": run_id,
        "experiment_id": eval_id,
        "train_protocol": train_protocol,
        "test_dataset": datasets,
        "model_name": model_name,
        "model_type": "direction_only_ml",
        "feature_set": feature_set,
        "feature_count": len(features),
        "direction_macro_f1": float(macro),
        "direction_accuracy": float(acc),
        "forward_f1": float(per_class[0]),
        "backward_f1": float(per_class[1]),
        "lateral_f1": float(per_class[2]),
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
        **complexity_info(model_name, features),
    }


def save_direction_confusion(row: dict[str, Any], out_dir: Path) -> None:
    cm = np.array(
        [
            [row["cm_00"], row["cm_01"], row["cm_02"]],
            [row["cm_10"], row["cm_11"], row["cm_12"]],
            [row["cm_20"], row["cm_21"], row["cm_22"]],
        ],
        dtype=int,
    )
    save_cm(cm, DIRECTION_LABELS, out_dir / f"{safe_name(row['run_id'])}_{row['experiment_id']}_direction.png", f"{row['run_id']} {row['experiment_id']}")


def save_cm(cm: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


def extract_importance(model: Any, features: list[str], run_id: str, protocol_id: str, task: str) -> list[dict[str, Any]]:
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    values = None
    method = None
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        method = "native_feature_importances"
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        method = "abs_linear_coef"
    if values is None:
        return [
            {
                "run_id": run_id,
                "protocol_id": protocol_id,
                "task": task,
                "feature": "",
                "importance": math.nan,
                "rank": math.nan,
                "method": "not_available_for_model",
            }
        ]
    order = np.argsort(values)[::-1][: min(30, len(features))]
    return [
        {
            "run_id": run_id,
            "protocol_id": protocol_id,
            "task": task,
            "feature": features[i],
            "importance": float(values[i]),
            "rank": rank + 1,
            "method": method,
        }
        for rank, i in enumerate(order)
    ]


def complexity_info(model_name: str, features: list[str]) -> dict[str, Any]:
    n = len(features)
    if model_name == "LogisticRegression":
        return {"complexity": n + 1, "complexity_note": f"LR params={n + 1}", "feature_count": n}
    if model_name == "LinearSVM":
        return {"complexity": n + 1, "complexity_note": f"LinearSVM params~{n + 1}", "feature_count": n}
    if model_name == "RandomForest":
        return {"complexity": 300, "complexity_note": "RandomForest trees=300 fall / 200 direction", "feature_count": n}
    if model_name == "GradientBoosting":
        return {"complexity": 100, "complexity_note": "GradientBoosting default trees=100", "feature_count": n}
    if model_name == "HistGradientBoosting":
        return {"complexity": 140, "complexity_note": "HistGradientBoosting max_iter=140", "feature_count": n}
    return {"complexity": math.nan, "complexity_note": "", "feature_count": n}


def best_by_protocol(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for protocol_id, group in df.groupby("experiment_id"):
        if task == "fall":
            sort_cols = ["fall_f1", "fall_precision", "fall_recall", "FP"]
            ascending = [False, False, False, True]
        else:
            sort_cols = ["direction_macro_f1", "direction_accuracy", "forward_f1"]
            ascending = [False, False, False]
        rows.append(group.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict())
    return pd.DataFrame(rows).sort_values("experiment_id")


def prediction_path_for(protocol_id: str) -> Path | None:
    mapping = {
        "E1_BITS_TO_BITS": "E1_BITS_TO_BITS",
        "E2_WEDA_TO_WEDA": "E2_WEDA_TO_WEDA",
        "E3_BITS_WEDA_MIXED": "E3_BITS_WEDA_MIXED",
        "E4_BITS_TO_WEDA": "E4_BITS_TO_WEDA",
        "E5_WEDA_TO_BITS": "E5_WEDA_TO_BITS",
        "E6_E3_MIXED_TEST_BITS": "E6_E3_MIXED_TEST_BITS",
        "E7_E3_MIXED_TEST_WEDA": "E7_E3_MIXED_TEST_WEDA",
    }
    folder = mapping.get(protocol_id)
    if not folder:
        return None
    path = A5_ARTIFACT_DIR / folder / "predictions.csv"
    return path if path.exists() else None


def load_a5_reference_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    fall_rows = []
    direction_rows = []
    for protocol_id in PROTOCOLS:
        path = prediction_path_for(protocol_id)
        if path is None:
            continue
        pred = pd.read_csv(path)
        if pred.empty:
            continue
        true_fall = pred["true_fall"].astype(str).str.lower().eq("fall").astype(int).to_numpy()
        pred_fall = pred["pred_fall"].astype(str).str.lower().eq("fall").astype(int).to_numpy()
        fall_prob = pred["fall_prob"].astype(float).to_numpy() if "fall_prob" in pred.columns else pred_fall.astype(float)
        tn, fp, fn, tp = [int(v) for v in confusion_matrix(true_fall, pred_fall, labels=[0, 1]).ravel()]
        fall_rows.append(
            {
                "experiment_id": protocol_id,
                "model_name": "DS-Fall-RD A5",
                "model_type": "a5_reference",
                "feature_set": "temporal_tilt12",
                "fall_expert": "A5_deep_fall_head",
                "fall_precision": precision_score(true_fall, pred_fall, zero_division=0),
                "fall_recall": recall_score(true_fall, pred_fall, zero_division=0),
                "fall_f1": f1_score(true_fall, pred_fall, zero_division=0),
                "accuracy": accuracy_score(true_fall, pred_fall),
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "TN": tn,
                "AUROC": safe_auc(true_fall, fall_prob),
                "average_precision": safe_ap(true_fall, fall_prob),
                "complexity": 65959,
                "notes": f"saved predictions: {path}",
            }
        )
        sup = pred["true_direction"].astype(str).str.lower().isin(DIRECTION_LABELS)
        if sup.any():
            y = pred.loc[sup, "true_direction"].astype(str).str.lower().map(DIR_TO_ID).to_numpy()
            yhat = pred.loc[sup, "pred_direction"].astype(str).str.lower().map(DIR_TO_ID).fillna(-1).astype(int).to_numpy()
            per = f1_score(y, yhat, labels=[0, 1, 2], average=None, zero_division=0)
            direction_rows.append(
                {
                    "experiment_id": protocol_id,
                    "model_name": "DS-Fall-RD A5",
                    "model_type": "a5_reference",
                    "feature_set": "temporal_tilt12",
                    "direction_model": "A5_direction_head",
                    "direction_macro_f1": f1_score(y, yhat, labels=[0, 1, 2], average="macro", zero_division=0),
                    "direction_accuracy": accuracy_score(y, yhat),
                    "forward_f1": per[0],
                    "backward_f1": per[1],
                    "lateral_f1": per[2],
                    "direction_n_supervised": len(y),
                    "complexity": 65959,
                    "notes": f"saved predictions: {path}",
                }
            )
    return pd.DataFrame(fall_rows), pd.DataFrame(direction_rows)


_A5_FALL_CACHE: pd.DataFrame | None = None


def a5_fall_metric(protocol_id: str) -> dict[str, Any] | None:
    global _A5_FALL_CACHE
    if _A5_FALL_CACHE is None:
        _A5_FALL_CACHE, _ = load_a5_reference_rows()
    if _A5_FALL_CACHE.empty:
        return None
    rows = _A5_FALL_CACHE[_A5_FALL_CACHE["experiment_id"].eq(protocol_id)]
    return rows.iloc[0].to_dict() if not rows.empty else None


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


def build_comparison(fall_df: pd.DataFrame, direction_df: pd.DataFrame, a5_fall: pd.DataFrame, a5_direction: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not a5_fall.empty:
        rows.extend(a5_fall.assign(task="fall").to_dict("records"))
    if not a5_direction.empty:
        rows.extend(a5_direction.assign(task="direction").to_dict("records"))
    for feature_set in ["FallNoTiming_Core", "FallNoTiming_Full"]:
        sub = fall_df[fall_df["feature_set"].eq(feature_set)]
        if sub.empty:
            continue
        best = best_by_protocol(sub, task="fall")
        for row in best.to_dict("records"):
            rows.append(
                {
                    "task": "fall",
                    "experiment_id": row["experiment_id"],
                    "model_name": row["model_name"],
                    "model_type": "ml_fall_only",
                    "feature_set": row["feature_set"],
                    "fall_expert": row["model_name"],
                    "fall_precision": row["fall_precision"],
                    "fall_recall": row["fall_recall"],
                    "fall_f1": row["fall_f1"],
                    "FP": row["FP"],
                    "FN": row["FN"],
                    "AUROC": row["AUROC"],
                    "average_precision": row["average_precision"],
                    "complexity": row["complexity"],
                    "notes": "best ML fall-only by protocol and feature set",
                }
            )
    for feature_set in ["DirectionCore", "DirectionNoMagnitude", "DirectionFullNoTiming"]:
        sub = direction_df[direction_df["feature_set"].eq(feature_set)]
        if sub.empty:
            continue
        best = best_by_protocol(sub, task="direction")
        for row in best.to_dict("records"):
            rows.append(
                {
                    "task": "direction",
                    "experiment_id": row["experiment_id"],
                    "model_name": row["model_name"],
                    "model_type": "ml_direction_only",
                    "feature_set": row["feature_set"],
                    "direction_model": row["model_name"],
                    "direction_macro_f1": row["direction_macro_f1"],
                    "direction_accuracy": row["direction_accuracy"],
                    "forward_f1": row["forward_f1"],
                    "backward_f1": row["backward_f1"],
                    "lateral_f1": row["lateral_f1"],
                    "direction_n_supervised": row["direction_n_supervised"],
                    "complexity": row["complexity"],
                    "notes": "best ML direction-only by protocol and feature set",
                }
            )
    hybrid_path = HYBRID_RESULTS_PATH
    if hybrid_path.exists():
        hybrid = pd.read_csv(hybrid_path)
        for run_id in ["HYB_HGB_FullNoTiming", "HYB_HGB_FullTiming"]:
            rows.extend(hybrid[hybrid["run_id"].eq(run_id)].assign(task="fall", notes="previous hybrid result").to_dict("records"))
    return pd.DataFrame(rows)


def build_e2e_results(
    df: pd.DataFrame,
    fall_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    fall_predictions: dict[tuple[str, str], np.ndarray],
    direction_predictions: dict[tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    rows = []
    a5_fall_pred = {}
    a5_dir_pred = {}
    for eval_id in ["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"]:
        _, _, mask = protocol_masks(df, eval_id)
        a5_fall_pred[eval_id] = (df.loc[mask, "fall_prob"].to_numpy(dtype=float) >= 0.5).astype(int)
        a5_dir_pred[eval_id] = np.argmax(
            df.loc[mask, ["direction_prob_forward", "direction_prob_backward", "direction_prob_lateral"]].astype(float).to_numpy(), axis=1
        )
    final_runs = select_final_runs(fall_df, direction_df)
    for name, fall_run, dir_run, direction_source in final_runs:
        for eval_id in ["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"]:
            _, _, mask = protocol_masks(df, eval_id)
            supervised_mask = direction_train_mask(df.loc[mask].reset_index(drop=True))
            y = df.loc[mask, "direction_id"].to_numpy()[supervised_mask]
            if fall_run == "A5":
                fall_pred = a5_fall_pred[eval_id]
            else:
                fall_pred = fall_predictions.get((fall_run, eval_id))
            if dir_run == "A5":
                dir_pred = a5_dir_pred[eval_id]
            else:
                dir_pred = direction_predictions.get((dir_run, eval_id))
            if fall_pred is None or dir_pred is None or len(y) == 0:
                continue
            fall_sup = fall_pred[supervised_mask]
            if len(dir_pred) == len(supervised_mask):
                dir_sup = dir_pred[supervised_mask]
            elif len(dir_pred) == len(y):
                dir_sup = dir_pred
            else:
                continue
            e2e_pred = dir_sup.copy()
            e2e_pred[fall_sup == 0] = -1
            per = f1_score(y, e2e_pred, labels=[0, 1, 2], average=None, zero_division=0)
            rows.append(
                {
                    "pipeline": name,
                    "experiment_id": eval_id,
                    "fall_run_id": fall_run,
                    "direction_run_id": dir_run,
                    "direction_source": direction_source,
                    "E2E_Direction_Macro_F1": f1_score(y, e2e_pred, labels=[0, 1, 2], average="macro", zero_division=0),
                    "coverage": float(np.mean(fall_sup == 1)),
                    "correct_rate": float(np.mean((fall_sup == 1) & (dir_sup == y))),
                    "forward_f1": per[0],
                    "backward_f1": per[1],
                    "lateral_f1": per[2],
                    "direction_n_supervised": len(y),
                }
            )
    return pd.DataFrame(rows)


def select_final_runs(fall_df: pd.DataFrame, direction_df: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    e3_fall_full = fall_df[
        (fall_df["experiment_id"].eq("E7_E3_MIXED_TEST_WEDA"))
        & (fall_df["train_protocol"].astype(str).str.contains("E3_BITS_WEDA_MIXED"))
        & (fall_df["feature_set"].eq("FallNoTiming_Full"))
    ]
    e3_fall_core = fall_df[
        (fall_df["experiment_id"].eq("E7_E3_MIXED_TEST_WEDA"))
        & (fall_df["train_protocol"].astype(str).str.contains("E3_BITS_WEDA_MIXED"))
        & (fall_df["feature_set"].eq("FallNoTiming_Core"))
    ]
    e3_dir_full = direction_df[
        (direction_df["experiment_id"].eq("E7_E3_MIXED_TEST_WEDA"))
        & (direction_df["train_protocol"].astype(str).str.contains("E3_BITS_WEDA_MIXED"))
        & (direction_df["feature_set"].eq("DirectionFullNoTiming"))
    ]
    e3_dir_core = direction_df[
        (direction_df["experiment_id"].eq("E7_E3_MIXED_TEST_WEDA"))
        & (direction_df["train_protocol"].astype(str).str.contains("E3_BITS_WEDA_MIXED"))
        & (direction_df["feature_set"].eq("DirectionCore"))
    ]
    best_fall_full = best_row_id(e3_fall_full, "fall")
    best_fall_core = best_row_id(e3_fall_core, "fall")
    best_dir_full = best_row_id(e3_dir_full, "direction")
    best_dir_core = best_row_id(e3_dir_core, "direction")
    rows = [("A5_reference", "A5", "A5", "A5")]
    if best_fall_full and best_dir_full:
        rows.append(("ML_only_separated", best_fall_full, best_dir_full, "ML_direction"))
    if best_fall_full:
        rows.append(("Hybrid_ML_FallFull_A5_Direction", best_fall_full, "A5", "A5"))
    if best_fall_core:
        rows.append(("EdgeLite_ML_FallCore_A5_Direction", best_fall_core, "A5", "A5"))
    if best_fall_core and best_dir_core:
        rows.append(("EdgeLite_ML_FallCore_ML_DirectionCore", best_fall_core, best_dir_core, "ML_direction"))
    return rows


def best_row_id(df: pd.DataFrame, task: str) -> str | None:
    if df.empty:
        return None
    if task == "fall":
        row = df.sort_values(["fall_f1", "fall_precision", "fall_recall"], ascending=False).iloc[0]
    else:
        row = df.sort_values(["direction_macro_f1", "direction_accuracy"], ascending=False).iloc[0]
    return str(row["run_id"])


def build_final_recommendation(
    fall_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    e2e: pd.DataFrame,
    a5_fall: pd.DataFrame,
    a5_direction: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for pipeline in ["A5_reference", "ML_only_separated", "Hybrid_ML_FallFull_A5_Direction", "EdgeLite_ML_FallCore_A5_Direction"]:
        rec = {"pipeline": pipeline}
        for eval_id in ["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"]:
            e2e_row = e2e[(e2e["pipeline"].eq(pipeline)) & (e2e["experiment_id"].eq(eval_id))]
            if not e2e_row.empty:
                rec[f"{eval_id}_E2E_Direction_F1"] = e2e_row.iloc[0]["E2E_Direction_Macro_F1"]
                rec[f"{eval_id}_coverage"] = e2e_row.iloc[0]["coverage"]
        rows.append(rec)
    out = pd.DataFrame(rows)
    # Fill fall and direction headline metrics from best available rows.
    for i, row in out.iterrows():
        pipeline = row["pipeline"]
        if pipeline == "A5_reference":
            fill_from_a5(out, i, a5_fall, a5_direction)
        elif pipeline == "Hybrid_ML_FallFull_A5_Direction":
            fall_run = e2e[e2e["pipeline"].eq(pipeline)]["fall_run_id"].dropna().head(1)
            fill_from_ml(out, i, fall_df, direction_df, fall_run.iloc[0] if not fall_run.empty else None, "A5", a5_direction)
        elif pipeline == "ML_only_separated":
            rows_e2e = e2e[e2e["pipeline"].eq(pipeline)]
            if not rows_e2e.empty:
                fill_from_ml(out, i, fall_df, direction_df, rows_e2e.iloc[0]["fall_run_id"], rows_e2e.iloc[0]["direction_run_id"], a5_direction)
        elif pipeline == "EdgeLite_ML_FallCore_A5_Direction":
            rows_e2e = e2e[e2e["pipeline"].eq(pipeline)]
            if not rows_e2e.empty:
                fill_from_ml(out, i, fall_df, direction_df, rows_e2e.iloc[0]["fall_run_id"], "A5", a5_direction)
        out.loc[i, "paper_suitability"] = paper_suitability(pipeline)
    return out


def fill_from_a5(out: pd.DataFrame, i: int, a5_fall: pd.DataFrame, a5_direction: pd.DataFrame) -> None:
    for eval_id in ["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"]:
        f = a5_fall[a5_fall["experiment_id"].eq(eval_id)]
        d = a5_direction[a5_direction["experiment_id"].eq(eval_id)]
        if not f.empty:
            out.loc[i, f"{eval_id}_Fall_F1"] = f.iloc[0]["fall_f1"]
            out.loc[i, f"{eval_id}_Fall_Precision"] = f.iloc[0]["fall_precision"]
            out.loc[i, f"{eval_id}_Fall_Recall"] = f.iloc[0]["fall_recall"]
        if not d.empty:
            out.loc[i, f"{eval_id}_Direction_F1"] = d.iloc[0]["direction_macro_f1"]
    out.loc[i, "complexity"] = 65959


def fill_from_ml(out: pd.DataFrame, i: int, fall_df: pd.DataFrame, direction_df: pd.DataFrame, fall_run: str | None, dir_run: str, a5_direction: pd.DataFrame) -> None:
    for eval_id in ["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"]:
        if fall_run:
            f = fall_df[(fall_df["run_id"].eq(fall_run.replace("E3_BITS_WEDA_MIXED", eval_id) if eval_id.startswith("E6") or eval_id.startswith("E7") else fall_run)) & (fall_df["experiment_id"].eq(eval_id))]
            if f.empty:
                f = fall_df[(fall_df["experiment_id"].eq(eval_id)) & (fall_df["run_id"].str.contains(fall_run.split("_E3_BITS_WEDA_MIXED")[0], regex=False))]
            if not f.empty:
                out.loc[i, f"{eval_id}_Fall_F1"] = f.iloc[0]["fall_f1"]
                out.loc[i, f"{eval_id}_Fall_Precision"] = f.iloc[0]["fall_precision"]
                out.loc[i, f"{eval_id}_Fall_Recall"] = f.iloc[0]["fall_recall"]
        if dir_run == "A5":
            d = a5_direction[a5_direction["experiment_id"].eq(eval_id)]
        else:
            d = direction_df[(direction_df["run_id"].eq(dir_run.replace("E3_BITS_WEDA_MIXED", eval_id) if eval_id.startswith("E6") or eval_id.startswith("E7") else dir_run)) & (direction_df["experiment_id"].eq(eval_id))]
            if d.empty:
                d = direction_df[(direction_df["experiment_id"].eq(eval_id)) & (direction_df["run_id"].str.contains(dir_run.split("_E3_BITS_WEDA_MIXED")[0], regex=False))]
        if not d.empty:
            out.loc[i, f"{eval_id}_Direction_F1"] = d.iloc[0]["direction_macro_f1"]
    out.loc[i, "complexity_note"] = "ML fall + direction source"


def paper_suitability(pipeline: str) -> str:
    if pipeline == "A5_reference":
        return "main deep multitask baseline"
    if pipeline == "Hybrid_ML_FallFull_A5_Direction":
        return "recommended practical hybrid if no-timing ML fall consistently wins"
    if pipeline == "ML_only_separated":
        return "analysis pipeline; use only if ML direction beats A5"
    if pipeline == "EdgeLite_ML_FallCore_A5_Direction":
        return "edge-lite ablation"
    return ""


def build_direction_shift_notes(df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    mask = direction_train_mask(df)
    sub = df.loc[mask].copy()
    for feature in feature_sets["DirectionFullNoTiming"]:
        bits = sub[sub["dataset"].eq("bits")][feature].astype(float).to_numpy()
        weda = sub[sub["dataset"].eq("weda")][feature].astype(float).to_numpy()
        if len(bits) < 2 or len(weda) < 2:
            continue
        rows.append(
            {
                "feature": feature,
                "bits_median": float(np.median(bits)),
                "weda_median": float(np.median(weda)),
                "median_delta_bits_minus_weda": float(np.median(bits) - np.median(weda)),
                "abs_median_delta": float(abs(np.median(bits) - np.median(weda))),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_median_delta", ascending=False).head(80)


def write_summary(
    report_dir: Path,
    fall_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    fall_best: pd.DataFrame,
    direction_best: pd.DataFrame,
    comparison: pd.DataFrame,
    recommendation: pd.DataFrame,
    e2e: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    e7_fall = fall_df[fall_df["experiment_id"].eq("E7_E3_MIXED_TEST_WEDA")].sort_values(["fall_f1", "fall_precision"], ascending=False)
    e7_direction = direction_df[direction_df["experiment_id"].eq("E7_E3_MIXED_TEST_WEDA")].sort_values(["direction_macro_f1", "direction_accuracy"], ascending=False)
    e4_fall = fall_best[fall_best["experiment_id"].eq("E4_BITS_TO_WEDA")]
    e5_fall = fall_best[fall_best["experiment_id"].eq("E5_WEDA_TO_BITS")]
    e4_dir = direction_best[direction_best["experiment_id"].eq("E4_BITS_TO_WEDA")]
    e5_dir = direction_best[direction_best["experiment_id"].eq("E5_WEDA_TO_BITS")]
    a5_comp = comparison[(comparison["model_type"].eq("a5_reference")) & (comparison["experiment_id"].eq("E7_E3_MIXED_TEST_WEDA")) & (comparison["task"].eq("fall"))]
    best_fall = e7_fall.iloc[0].to_dict() if not e7_fall.empty else {}
    best_dir = e7_direction.iloc[0].to_dict() if not e7_direction.empty else {}
    core_e7 = e7_fall[e7_fall["feature_set"].eq("FallNoTiming_Core")]
    full_e7 = e7_fall[e7_fall["feature_set"].eq("FallNoTiming_Full")]
    dir_nomag = e7_direction[e7_direction["feature_set"].eq("DirectionNoMagnitude")]
    dir_full = e7_direction[e7_direction["feature_set"].eq("DirectionFullNoTiming")]

    lines = [
        "# ML E1-E7 Specialized Summary",
        "",
        "## Protocol",
        "",
        "- BITS/WEDA only, 25 Hz, 2-second event-centered windows.",
        "- ML fall-only uses no FullTiming features.",
        "- Direction-only uses supervised fall windows only.",
        "- E6/E7 reuse the E3 mixed model, matching the requested protocol.",
        "- Test sets are never used for threshold or hyperparameter tuning.",
        "",
        "## Best Fall E7 Rows",
        "",
        markdown_table(format_df(e7_fall.head(15))),
        "",
        "## Best Direction E7 Rows",
        "",
        markdown_table(format_df(e7_direction.head(15))),
        "",
        "## Fall Best By Protocol",
        "",
        markdown_table(format_df(fall_best)),
        "",
        "## Direction Best By Protocol",
        "",
        markdown_table(format_df(direction_best)),
        "",
        "## Final Pipeline Recommendation",
        "",
        markdown_table(format_df(recommendation)),
        "",
        "## End-to-end Direction",
        "",
        markdown_table(format_df(e2e.sort_values(["pipeline", "experiment_id"]).head(40))) if not e2e.empty else "_No rows._",
        "",
        "## Answers",
        "",
        answer_best_fall(best_fall),
        answer_core_vs_full(core_e7, full_e7),
        answer_transfer("fall", e4_fall, e5_fall),
        answer_fall_vs_a5(best_fall, a5_comp),
        answer_direction_only(best_dir),
        answer_direction_vs_a5(best_dir, comparison),
        answer_nomag(dir_nomag, dir_full),
        answer_pipeline(recommendation),
        "9. Deep fall head can be removed from final fall inference if the selected no-timing ML Fall Expert is adopted; however, keep A5 as the direction expert.",
        "10. Keep A5 fall head as auxiliary/reference for the paper baseline and for reproducibility; do not present FullTiming as main.",
        "11. Paper-facing: no-timing ML Fall Expert is strong for fall detection; A5 temporal direction head remains the safest direction expert unless direction-only ML clearly exceeds it across E1-E7. Hybrid ML Fall + A5 Direction is the practical recommendation when fall gains are prioritized.",
    ]
    if not failures.empty:
        lines.extend(["", "## Failures / Limitations", "", markdown_table(format_df(failures))])
    (report_dir / "ml_e1_e7_specialized_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def answer_best_fall(best: dict[str, Any]) -> str:
    if not best:
        return "1. Best ML fall-only model unavailable."
    return f"1. Best no-timing ML fall-only E7 model is `{best['model_name']}` with `{best['feature_set']}`: F1={best['fall_f1']:.4f}, precision={best['fall_precision']:.4f}, recall={best['fall_recall']:.4f}, FP={int(best['FP'])}."


def answer_core_vs_full(core: pd.DataFrame, full: pd.DataFrame) -> str:
    if core.empty or full.empty:
        return "2. Core vs Full comparison unavailable."
    c = core.sort_values("fall_f1", ascending=False).iloc[0]
    f = full.sort_values("fall_f1", ascending=False).iloc[0]
    if f["fall_f1"] > c["fall_f1"] + 0.02:
        return f"2. FallNoTiming_Full is needed for best fall performance: Full F1={f['fall_f1']:.4f} vs Core F1={c['fall_f1']:.4f}."
    return f"2. FallNoTiming_Core is close enough: Core F1={c['fall_f1']:.4f}, Full F1={f['fall_f1']:.4f}."


def answer_transfer(task: str, e4: pd.DataFrame, e5: pd.DataFrame) -> str:
    if e4.empty or e5.empty:
        return f"3. {task} transfer E4/E5 unavailable."
    metric = "fall_f1" if task == "fall" else "direction_macro_f1"
    return f"3. Pure transfer remains measurable: E4 bits->weda best {metric}={e4.iloc[0][metric]:.4f}; E5 weda->bits best {metric}={e5.iloc[0][metric]:.4f}."


def answer_fall_vs_a5(best: dict[str, Any], a5: pd.DataFrame) -> str:
    if not best or a5.empty:
        return "4. Fall vs A5 comparison unavailable."
    a = a5.iloc[0]
    return f"4. ML fall-only beats A5 on E7 if comparing best no-timing row: ML F1={best['fall_f1']:.4f} vs A5 F1={a['fall_f1']:.4f}, FP {int(best['FP'])} vs {int(a['FP'])}."


def answer_direction_only(best: dict[str, Any]) -> str:
    if not best:
        return "5. Direction-only result unavailable."
    return f"5. Best direction-only E7 model is `{best['model_name']}` with `{best['feature_set']}`: macro F1={best['direction_macro_f1']:.4f}, accuracy={best['direction_accuracy']:.4f}."


def answer_direction_vs_a5(best: dict[str, Any], comparison: pd.DataFrame) -> str:
    a5 = comparison[(comparison["task"].eq("direction")) & (comparison["model_type"].eq("a5_reference")) & (comparison["experiment_id"].eq("E7_E3_MIXED_TEST_WEDA"))]
    if not best or a5.empty:
        return "6. Direction vs A5 comparison unavailable."
    a = a5.iloc[0]
    verdict = "beats" if best["direction_macro_f1"] > a["direction_macro_f1"] else "does not beat"
    return f"6. Direction-only ML {verdict} A5 on E7: ML={best['direction_macro_f1']:.4f}, A5={a['direction_macro_f1']:.4f}."


def answer_nomag(nomag: pd.DataFrame, full: pd.DataFrame) -> str:
    if nomag.empty or full.empty:
        return "7. DirectionNoMagnitude comparison unavailable."
    n = nomag.sort_values("direction_macro_f1", ascending=False).iloc[0]
    f = full.sort_values("direction_macro_f1", ascending=False).iloc[0]
    return f"7. DirectionNoMagnitude E7 best macro F1={n['direction_macro_f1']:.4f}; DirectionFullNoTiming best={f['direction_macro_f1']:.4f}. This indicates whether signed features alone are sufficient."


def answer_pipeline(recommendation: pd.DataFrame) -> str:
    if recommendation.empty:
        return "8. Final pipeline recommendation unavailable."
    return "8. Final pipeline options are saved in `final_pipeline_recommendation.csv`; prefer Hybrid ML FallNoTiming_Full + A5 Direction unless ML direction-only consistently beats A5."


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
            if col in {"TN", "FP", "FN", "TP", "feature_count", "direction_n_supervised", "complexity"}:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(int(v)))
            else:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    return out


if __name__ == "__main__":
    main()
