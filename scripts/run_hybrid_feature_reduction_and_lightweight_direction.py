from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from collections import Counter, defaultdict
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
from sklearn.inspection import permutation_importance
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
from sklearn.utils.class_weight import compute_class_weight

import run_25hz_a5wcefw_bits_weda as a5data
import run_final_four_pipelines_e1_e7 as final4
import run_ml_e1_e7_specialized as mlbase

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf  # noqa: F401

    TENSORFLOW_AVAILABLE = True
    TENSORFLOW_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - environment dependent
    TENSORFLOW_AVAILABLE = False
    TENSORFLOW_IMPORT_ERROR = repr(exc)


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

FALL_TOP8_IMPACT_ENERGY = [
    "acc_mag_range",
    "acc_mag_max",
    "acc_mag_p95",
    "acc_mag_std",
    "gyro_mag_p95",
    "jerk_p95",
    "jerk_std",
    "post_acc_mag_std",
]

FALL_TOP10_IMPACT_GYRO_POSTURE = [
    "acc_mag_range",
    "acc_mag_max",
    "acc_mag_p95",
    "acc_mag_std",
    "gyro_mag_p95",
    "gyro_mag_mean",
    "jerk_p95",
    "jerk_std",
    "tilt_delta_p95",
    "post_acc_mag_std",
]

FALL_TOP12_NO_TIMING_CORE = [
    "acc_mag_range",
    "acc_mag_max",
    "acc_mag_p95",
    "acc_mag_std",
    "gyro_mag_p95",
    "gyro_mag_mean",
    "gyro_mag_range",
    "jerk_max",
    "jerk_p95",
    "jerk_std",
    "tilt_delta_p95",
    "post_acc_mag_std",
]

FALL_TOP12_WITH_DIRECTION_RELEVANT = [
    "acc_mag_range",
    "acc_mag_max",
    "acc_mag_std",
    "gyro_mag_p95",
    "jerk_p95",
    "jerk_std",
    "tilt_delta_p95",
    "roll_range",
    "pitch_range",
    "gy_range",
    "pre_post_energy_ratio",
    "post_acc_mag_std",
]

FALL_FULL_PROMPT_FEATURES = [
    "ax_mean",
    "ax_std",
    "ax_min",
    "ax_max",
    "ax_range",
    "ax_final_initial",
    "ax_median",
    "ax_iqr",
    "peak_signed_ax",
    "integrated_ax",
    "ay_mean",
    "ay_std",
    "ay_min",
    "ay_max",
    "ay_range",
    "ay_final_initial",
    "ay_median",
    "ay_iqr",
    "peak_signed_ay",
    "integrated_ay",
    "az_mean",
    "az_std",
    "az_min",
    "az_max",
    "az_range",
    "az_final_initial",
    "az_median",
    "az_iqr",
    "peak_signed_az",
    "integrated_az",
    "gx_mean",
    "gx_std",
    "gx_min",
    "gx_max",
    "gx_range",
    "gx_final_initial",
    "gx_median",
    "gx_iqr",
    "peak_signed_gx",
    "integrated_gx",
    "gy_mean",
    "gy_std",
    "gy_min",
    "gy_max",
    "gy_range",
    "gy_final_initial",
    "gy_median",
    "gy_iqr",
    "peak_signed_gy",
    "integrated_gy",
    "gz_mean",
    "gz_std",
    "gz_min",
    "gz_max",
    "gz_range",
    "gz_final_initial",
    "gz_median",
    "gz_iqr",
    "peak_signed_gz",
    "integrated_gz",
    "acc_mag_mean",
    "acc_mag_std",
    "acc_mag_min",
    "acc_mag_max",
    "acc_mag_p95",
    "acc_mag_range",
    "acc_mag_median",
    "acc_mag_iqr",
    "gyro_mag_mean",
    "gyro_mag_std",
    "gyro_mag_min",
    "gyro_mag_max",
    "gyro_mag_p95",
    "gyro_mag_range",
    "gyro_mag_median",
    "gyro_mag_iqr",
    "jerk_mean",
    "jerk_std",
    "jerk_max",
    "jerk_p95",
    "jerk_median",
    "jerk_iqr",
    "roll_mean",
    "roll_std",
    "roll_min",
    "roll_max",
    "roll_range",
    "roll_final_initial",
    "roll_median",
    "roll_iqr",
    "pitch_mean",
    "pitch_std",
    "pitch_min",
    "pitch_max",
    "pitch_range",
    "pitch_final_initial",
    "pitch_median",
    "pitch_iqr",
    "tilt_delta_mean",
    "tilt_delta_std",
    "tilt_delta_max",
    "tilt_delta_p95",
    "tilt_delta_final",
    "tilt_delta_median",
    "tilt_delta_iqr",
    "acc_mag_peak_value",
    "gyro_mag_peak_value",
    "pre_impact_energy",
    "post_impact_energy",
    "pre_post_energy_ratio",
    "post_acc_mag_std",
    "post_gyro_mag_std",
    "pre_acc_mag_std",
    "pre_gyro_mag_std",
    "pre_ax_mean",
    "post_ax_mean",
    "pre_ay_mean",
    "post_ay_mean",
    "pre_az_mean",
    "post_az_mean",
    "pre_gx_mean",
    "post_gx_mean",
    "pre_gy_mean",
    "post_gy_mean",
    "pre_gz_mean",
    "post_gz_mean",
    "pre_roll_mean",
    "post_roll_mean",
    "pre_pitch_mean",
    "post_pitch_mean",
    "delta_roll_window",
    "delta_pitch_window",
]


@dataclass
class FallModelSpec:
    model_name: str
    model_family: str
    n_estimators: int | None = None
    max_depth: int | None = None


@dataclass
class FallTrained:
    model: Any
    features: list[str]
    feature_set: str
    threshold: float
    threshold_note: str
    model_spec: FallModelSpec
    protocol_id: str


@dataclass
class DirectionTrained:
    model: Any
    model_name: str
    feature_source: str
    features: list[str] | None
    params: int
    protocol_id: str
    seed: int
    learning_rate: float
    best_epoch: int
    best_val_macro_f1: float
    val_loss_at_best: float
    complexity: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid fall feature reduction and lightweight direction experiments.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-fall-reduction-only", action="store_true")
    parser.add_argument("--run-lightweight-direction-only", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--quick", action="store_true", help="Run a small validation subset.")
    parser.add_argument("--full", action="store_true", help="Alias for --run-all with full settings.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_summary_feature_matrix(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "outputs" / "reports" / "ml_feature_analysis_25hz" / "summary_feature_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary feature matrix: {path}")
    df = mlbase.load_summary(path)
    df = df[df["dataset"].isin(["bits", "weda"])].copy()
    if df.empty:
        raise ValueError("No BITS/WEDA rows in summary feature matrix.")
    return df.reset_index(drop=True)


def load_temporal_data(repo_root: Path, summary_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    cache_dir = ensure_dir(repo_root / "outputs" / "models" / "hybrid_feature_reduction_direction_lightweight")
    x_cache = cache_dir / "cached_event_25hz_tilt12_X.npy"
    meta_cache = cache_dir / "cached_event_25hz_metadata.csv"
    if x_cache.exists() and meta_cache.exists():
        X = np.load(x_cache)
        meta = pd.read_csv(meta_cache)
    else:
        config = a5data.make_config(str(repo_root))
        data = a5data.build_25hz_dataset(config, window_mode="event_centered")
        X = np.asarray(data["X_features"], dtype=np.float32)
        meta = data["metadata"].copy().reset_index(drop=True)
        np.save(x_cache, X)
        meta.to_csv(meta_cache, index=False)
    if X.shape[1:] != (50, 12):
        raise ValueError(f"Expected temporal shape (N,50,12), got {X.shape}")
    meta["dataset"] = meta["dataset"].astype(str).str.lower()
    meta["split"] = meta["split"].astype(str).str.lower()
    need = summary_df[["window_id"]].copy()
    aligned_meta = need.merge(meta, on="window_id", how="left", validate="one_to_one")
    missing = aligned_meta["dataset"].isna().sum()
    if missing:
        raise ValueError(f"Temporal data missing {missing} summary window_id rows.")
    order = meta.set_index("window_id").loc[summary_df["window_id"], :].index
    # Build integer position lookup to preserve exact summary row order.
    pos = pd.Series(np.arange(len(meta)), index=meta["window_id"])
    idx = pos.loc[summary_df["window_id"]].to_numpy(dtype=int)
    return X[idx], aligned_meta


def build_fall_feature_sets(df: pd.DataFrame) -> tuple[dict[str, list[str]], pd.DataFrame]:
    signal_features = mlbase.infer_signal_features(df)
    full = [f for f in FALL_FULL_PROMPT_FEATURES if f in df.columns and not mlbase.is_timing_feature(f)]
    if not full:
        full = [f for f in signal_features if not mlbase.is_timing_feature(f)]
    fixed = {
        "FallTop8_ImpactEnergy": FALL_TOP8_IMPACT_ENERGY,
        "FallTop10_ImpactGyroPosture": FALL_TOP10_IMPACT_GYRO_POSTURE,
        "FallTop12_NoTimingCore": FALL_TOP12_NO_TIMING_CORE,
        "FallTop12_WithDirectionRelevant": FALL_TOP12_WITH_DIRECTION_RELEVANT,
        "FallNoTiming_Core_16": mlbase.FALL_CORE,
        "FallNoTiming_Full_132": full,
    }
    rows = []
    out = {}
    for name, features in fixed.items():
        present = [f for f in features if f in df.columns and not mlbase.is_timing_feature(f)]
        missing = [f for f in features if f not in df.columns]
        out[name] = present
        for f in missing:
            rows.append({"feature_set": name, "feature": f, "reason": "missing_from_summary_feature_matrix"})
    out["FallTop10_SelectedByImportance"] = []
    out["FallTop12_SelectedByImportance"] = []
    return out, pd.DataFrame(rows)


def build_direction_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    signal_features = mlbase.infer_signal_features(df)
    direction_core = [f for f in mlbase.DIRECTION_CORE_CANDIDATES if f in df.columns and not mlbase.is_timing_feature(f)]
    direction_no_mag = [
        f
        for f in direction_core
        if not f.startswith(("acc_mag_", "gyro_mag_", "jerk_")) and not mlbase.is_timing_feature(f)
    ]
    return {
        "DirectionCore": direction_core,
        "DirectionNoMagnitude": direction_no_mag,
        "DirectionFullNoTiming": [f for f in signal_features if not mlbase.is_timing_feature(f)],
    }


def get_protocol_splits(df: pd.DataFrame, protocol_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return final4.get_protocol_splits(df, protocol_id)


def canonical_train_protocol(protocol_id: str) -> str:
    return "E3_BITS_WEDA_MIXED" if protocol_id in {"E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"} else protocol_id


def fall_model_specs(quick: bool = False) -> list[FallModelSpec]:
    specs = [
        FallModelSpec("LogisticRegression", "lr"),
        FallModelSpec("GradientBoosting", "gb"),
        FallModelSpec("HistGradientBoosting", "hgb"),
        FallModelSpec("RandomForest", "rf"),
        FallModelSpec("TinyGB10_d2", "tiny_gb", 10, 2),
        FallModelSpec("TinyGB20_d2", "tiny_gb", 20, 2),
        FallModelSpec("TinyGB30_d2", "tiny_gb", 30, 2),
        FallModelSpec("TinyGB10_d3", "tiny_gb", 10, 3),
        FallModelSpec("TinyGB20_d3", "tiny_gb", 20, 3),
        FallModelSpec("TinyGB30_d3", "tiny_gb", 30, 3),
    ]
    if quick:
        return [specs[1], specs[4]]
    return specs


def make_fall_model(spec: FallModelSpec, seed: int) -> Any:
    if spec.model_family == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))
    if spec.model_family == "gb":
        return make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=seed))
    if spec.model_family == "hgb":
        return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(random_state=seed, max_iter=140))
    if spec.model_family == "rf":
        return make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced", random_state=seed, n_jobs=-1),
        )
    if spec.model_family == "tiny_gb":
        return make_pipeline(
            StandardScaler(),
            GradientBoostingClassifier(n_estimators=spec.n_estimators, max_depth=spec.max_depth, random_state=seed),
        )
    raise ValueError(spec.model_family)


def train_fall_expert(
    df: pd.DataFrame,
    protocol_id: str,
    feature_set: str,
    features: list[str],
    spec: FallModelSpec,
    seed: int,
) -> FallTrained:
    train_mask, val_mask, _ = get_protocol_splits(df, protocol_id)
    model = make_fall_model(spec, seed)
    model.fit(df.loc[train_mask, features].astype(float).to_numpy(), df.loc[train_mask, "fall_label"].astype(int).to_numpy())
    val_scores = fall_scores(model, df.loc[val_mask, features].astype(float).to_numpy())
    threshold, note = tune_fall_threshold(df.loc[val_mask, "fall_label"].astype(int).to_numpy(), val_scores)
    return FallTrained(model, features, feature_set, threshold, note, spec, protocol_id)


def tune_fall_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, str]:
    return mlbase.select_fall_threshold(y, scores)


def fall_scores(model: Any, X: np.ndarray) -> np.ndarray:
    return final4.fall_scores(model, X)


def evaluate_fall(
    df: pd.DataFrame,
    mask: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    experiment_id: str,
    train_protocol: str,
    run_id: str,
    feature_set: str,
    model_name: str,
    threshold: float,
    selected_features: list[str],
    a5_fall_rows: pd.DataFrame,
) -> dict[str, Any]:
    y = df.loc[mask, "fall_label"].astype(int).to_numpy()
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, y_pred, labels=[0, 1]).ravel()]
    a5_fp = np.nan
    if not a5_fall_rows.empty:
        rows = a5_fall_rows[a5_fall_rows["experiment_id"].eq(experiment_id)]
        if not rows.empty:
            a5_fp = int(rows.iloc[0]["FP"])
    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "train_protocol": train_protocol,
        "test_dataset": "+".join(sorted(df.loc[mask, "dataset"].unique())),
        "model_name": model_name,
        "feature_set": feature_set,
        "feature_count": len(selected_features),
        "fall_precision": precision_score(y, y_pred, zero_division=0),
        "fall_recall": recall_score(y, y_pred, zero_division=0),
        "fall_f1": f1_score(y, y_pred, zero_division=0),
        "fall_accuracy": accuracy_score(y, y_pred),
        "AUROC": safe_auc(y, y_score),
        "average_precision": safe_ap(y, y_score),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "threshold": threshold,
        "A5_FP": a5_fp,
        "FP_reduction_vs_A5": a5_fp - fp if not pd.isna(a5_fp) else np.nan,
        "selected_features": ";".join(selected_features),
    }


def evaluate_e2e_direction(
    df: pd.DataFrame,
    mask: np.ndarray,
    fall_pred: np.ndarray,
    direction_pred: np.ndarray,
    pipeline: str,
    experiment_id: str,
    train_protocol: str,
    source: str,
) -> dict[str, Any]:
    sub = df.loc[mask].reset_index(drop=True)
    sup = mlbase.direction_train_mask(sub)
    y = sub.loc[sup, "direction_id"].astype(int).to_numpy()
    if len(y) == 0:
        return {
            "pipeline": pipeline,
            "experiment_id": experiment_id,
            "train_protocol": train_protocol,
            "source": source,
            "E2E_Direction_Macro_F1": np.nan,
            "coverage": np.nan,
            "correct_rate": np.nan,
            "direction_n_supervised": 0,
        }
    fall_sup = fall_pred[sup]
    dir_sup = direction_pred[sup]
    e2e_pred = dir_sup.copy()
    e2e_pred[fall_sup == 0] = -1
    per = f1_score(y, e2e_pred, labels=[0, 1, 2], average=None, zero_division=0)
    correct = (fall_sup == 1) & (dir_sup == y)
    return {
        "pipeline": pipeline,
        "experiment_id": experiment_id,
        "train_protocol": train_protocol,
        "source": source,
        "E2E_Direction_Macro_F1": f1_score(y, e2e_pred, labels=[0, 1, 2], average="macro", zero_division=0),
        "coverage": float(np.mean(fall_sup == 1)),
        "correct_rate": float(np.mean(correct)),
        "forward_f1": per[0],
        "backward_f1": per[1],
        "lateral_f1": per[2],
        "direction_n_supervised": len(y),
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


def load_a5_predictions(repo_root: Path, df: pd.DataFrame, protocol_id: str) -> tuple[np.ndarray, np.ndarray]:
    _, _, mask = get_protocol_splits(df, protocol_id)
    test = df.loc[mask].reset_index(drop=True)
    path = repo_root / "artifacts" / "experiments_25hz_event" / protocol_id / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    pred = pd.read_csv(path)
    aligned = test[["window_id"]].merge(
        pred[["window_id", "pred_fall", "fall_prob", "pred_direction"]],
        on="window_id",
        how="left",
        validate="one_to_one",
    )
    if aligned["pred_fall"].isna().any():
        raise ValueError(f"A5 prediction alignment failed for {protocol_id}")
    fall_pred = aligned["pred_fall"].astype(str).str.lower().eq("fall").astype(int).to_numpy()
    direction_pred = aligned["pred_direction"].astype(str).str.lower().map(DIR_TO_ID).fillna(-1).astype(int).to_numpy()
    return fall_pred, direction_pred


def run_fall_feature_reduction(
    repo_root: Path,
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    a5_fall: pd.DataFrame,
    a5_direction: pd.DataFrame,
    report_dir: Path,
    figure_dir: Path,
    model_dir: Path,
    seed: int,
    quick: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    rows: list[dict[str, Any]] = []
    e2e_rows: list[dict[str, Any]] = []
    imp_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    trained_cache: dict[tuple[str, str, str], FallTrained] = {}
    selected_cache: dict[tuple[str, str], list[str]] = {}
    feature_names = list(feature_sets)
    if quick:
        feature_names = ["FallTop8_ImpactEnergy", "FallNoTiming_Full_132"]
    for protocol_id in BASE_PROTOCOLS:
        for k_name, k in [("FallTop10_SelectedByImportance", 10), ("FallTop12_SelectedByImportance", 12)]:
            try:
                selected = select_top_k_features(df, protocol_id, feature_sets["FallNoTiming_Full_132"], k, seed)
                selected_cache[(protocol_id, k_name)] = selected
                for rank, feature in enumerate(selected, 1):
                    selected_rows.append({"protocol_id": protocol_id, "feature_set": k_name, "rank": rank, "feature": feature})
            except Exception as exc:
                failed.append({"stage": "select_features", "protocol_id": protocol_id, "feature_set": k_name, "model_name": "", "error": repr(exc)})
    for feature_set in feature_names:
        for spec in fall_model_specs(quick):
            for protocol_id in BASE_PROTOCOLS:
                try:
                    features = selected_cache.get((protocol_id, feature_set), feature_sets.get(feature_set, []))
                    if not features:
                        raise ValueError(f"No features for {feature_set}")
                    print(f"[fall] {feature_set} {spec.model_name} {protocol_id}")
                    trained = train_fall_expert(df, protocol_id, feature_set, features, spec, seed)
                    trained_cache[(feature_set, spec.model_name, protocol_id)] = trained
                    run_id = f"FALLRED_{spec.model_name}_{feature_set}_{protocol_id}"
                    with (model_dir / f"{run_id}.pkl").open("wb") as f:
                        pickle.dump(trained, f)
                    imp_rows.extend(extract_fall_importance(trained.model, features, run_id, protocol_id, feature_set, spec.model_name))
                except Exception as exc:
                    failed.append({"stage": "train_fall", "protocol_id": protocol_id, "feature_set": feature_set, "model_name": spec.model_name, "error": repr(exc)})
            if (feature_set, spec.model_name, "E3_BITS_WEDA_MIXED") in trained_cache:
                # E6/E7 reuse E3 automatically at evaluation time.
                pass
    for feature_set in feature_names:
        for spec in fall_model_specs(quick):
            for eval_id in ALL_PROTOCOLS:
                train_protocol = canonical_train_protocol(eval_id)
                trained = trained_cache.get((feature_set, spec.model_name, train_protocol))
                if trained is None:
                    continue
                try:
                    _, _, mask = get_protocol_splits(df, eval_id)
                    X = df.loc[mask, trained.features].astype(float).to_numpy()
                    scores = fall_scores(trained.model, X)
                    pred = (scores >= trained.threshold).astype(int)
                    predictions[(f"{spec.model_name}_{feature_set}", eval_id)] = pred
                    row = evaluate_fall(
                        df,
                        mask,
                        pred,
                        scores,
                        eval_id,
                        train_protocol,
                        f"FALLRED_{spec.model_name}_{feature_set}_{train_protocol}",
                        feature_set,
                        spec.model_name,
                        trained.threshold,
                        trained.features,
                        a5_fall,
                    )
                    rows.append(row)
                    _, a5_dir = load_a5_predictions(repo_root, df, eval_id)
                    e2e_rows.append(
                        evaluate_e2e_direction(
                            df,
                            mask,
                            pred,
                            a5_dir,
                            f"{spec.model_name}_{feature_set}_A5Direction",
                            eval_id,
                            train_protocol,
                            "fall_reduction_with_a5_direction",
                        )
                    )
                    save_fall_cm(row, figure_dir / "confusion_matrices")
                except Exception as exc:
                    failed.append({"stage": "eval_fall", "protocol_id": eval_id, "feature_set": feature_set, "model_name": spec.model_name, "error": repr(exc)})
    failed_df = pd.DataFrame(failed)
    return pd.DataFrame(rows), pd.DataFrame(imp_rows), pd.DataFrame(selected_rows), pd.DataFrame(e2e_rows), failed_df, predictions


def select_top_k_features(df: pd.DataFrame, protocol_id: str, features: list[str], k: int, seed: int) -> list[str]:
    train_mask, val_mask, _ = get_protocol_splits(df, protocol_id)
    model = GradientBoostingClassifier(random_state=seed)
    X_train = df.loc[train_mask, features].astype(float).to_numpy()
    y_train = df.loc[train_mask, "fall_label"].astype(int).to_numpy()
    model.fit(X_train, y_train)
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = permutation_importance(
            model,
            df.loc[val_mask, features].astype(float).to_numpy(),
            df.loc[val_mask, "fall_label"].astype(int).to_numpy(),
            n_repeats=5,
            random_state=seed,
            scoring="f1",
        ).importances_mean
    order = np.argsort(values)[::-1]
    return [features[i] for i in order[:k]]


def extract_fall_importance(model: Any, features: list[str], run_id: str, protocol_id: str, feature_set: str, model_name: str) -> list[dict[str, Any]]:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    values = getattr(est, "feature_importances_", None)
    if values is None:
        if model_name == "LogisticRegression":
            values = np.abs(est.coef_).mean(axis=0)
        else:
            return []
    return [
        {
            "run_id": run_id,
            "protocol_id": protocol_id,
            "feature_set": feature_set,
            "model_name": model_name,
            "feature": feature,
            "importance": float(value),
        }
        for feature, value in sorted(zip(features, values), key=lambda x: abs(x[1]), reverse=True)
    ]


if TORCH_AVAILABLE:

    class TinyDSConvDirection(nn.Module):
        def __init__(self, input_channels: int = 12):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(input_channels, input_channels, 5, padding=2, groups=input_channels),
                nn.Conv1d(input_channels, 16, 1),
                nn.ReLU(),
                nn.BatchNorm1d(16),
                nn.Conv1d(16, 16, 3, padding=1, groups=16),
                nn.Conv1d(16, 24, 1),
                nn.ReLU(),
                nn.BatchNorm1d(24),
                nn.Conv1d(24, 24, 3, padding=1, groups=24),
                nn.Conv1d(24, 32, 1),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 3))

        def forward(self, x):
            return self.head(self.net(x.transpose(1, 2)))


    class MicroTCNDirection(nn.Module):
        def __init__(self, input_channels: int = 12):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(input_channels, 16, 3, padding=1), nn.ReLU())
            self.blocks = nn.Sequential(
                nn.Conv1d(16, 16, 3, padding=1, groups=16),
                nn.Conv1d(16, 24, 1),
                nn.ReLU(),
                nn.Conv1d(24, 24, 3, padding=2, dilation=2, groups=24),
                nn.Conv1d(24, 24, 1),
                nn.ReLU(),
                nn.Conv1d(24, 24, 3, padding=4, dilation=4, groups=24),
                nn.Conv1d(24, 32, 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(32, 24), nn.ReLU(), nn.Dropout(0.1), nn.Linear(24, 3))

        def forward(self, x):
            return self.head(self.blocks(self.stem(x.transpose(1, 2))))


    class A5SmallDirection(nn.Module):
        def __init__(self, width: float, input_channels: int = 12):
            super().__init__()
            c1 = max(4, int(16 * width))
            c2 = max(6, int(24 * width))
            c3 = max(8, int(32 * width))
            c4 = max(12, int(64 * width))
            self.net = nn.Sequential(
                nn.Conv1d(input_channels, c1, 5, padding=2),
                nn.BatchNorm1d(c1),
                nn.ReLU(),
                nn.Conv1d(c1, c1, 5, padding=2, groups=c1),
                nn.Conv1d(c1, c2, 1),
                nn.BatchNorm1d(c2),
                nn.ReLU(),
                nn.Conv1d(c2, c2, 3, padding=1, groups=c2),
                nn.Conv1d(c2, c3, 1),
                nn.BatchNorm1d(c3),
                nn.ReLU(),
                nn.Conv1d(c3, c4, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(c4, c4, 3, padding=2, dilation=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(c4, max(8, int(32 * width))), nn.ReLU(), nn.Dropout(0.1), nn.Linear(max(8, int(32 * width)), 3))

        def forward(self, x):
            return self.head(self.net(x.transpose(1, 2)))


    class TinyGRUDirection(nn.Module):
        def __init__(self, input_channels: int = 12):
            super().__init__()
            self.gru = nn.GRU(input_channels, 16, batch_first=True)
            self.head = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Dropout(0.1), nn.Linear(16, 3))

        def forward(self, x):
            _, h = self.gru(x)
            return self.head(h[-1])


    class SummaryMLPDirection(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 3))

        def forward(self, x):
            return self.net(x)


def direction_model_specs(quick: bool = False) -> list[dict[str, Any]]:
    specs = [
        {"model_name": "D1_Tiny_DSConv_Direction", "kind": "temporal", "builder": lambda: TinyDSConvDirection()},
        {"model_name": "D2_Micro_TCN_Direction", "kind": "temporal", "builder": lambda: MicroTCNDirection()},
        {"model_name": "D3_A5_Direction_Small_050", "kind": "temporal", "builder": lambda: A5SmallDirection(0.5)},
        {"model_name": "D4_A5_Direction_Small_025", "kind": "temporal", "builder": lambda: A5SmallDirection(0.25)},
        {"model_name": "D5_Tiny_GRU_Direction", "kind": "temporal", "builder": lambda: TinyGRUDirection()},
        {"model_name": "D6_Summary_MLP_DirectionCore", "kind": "summary", "feature_set": "DirectionCore", "builder": None},
    ]
    if quick:
        return [specs[1], specs[5]]
    return specs


def train_direction_model(
    df: pd.DataFrame,
    X_temporal: np.ndarray,
    protocol_id: str,
    spec: dict[str, Any],
    direction_feature_sets: dict[str, list[str]],
    seed: int,
    lr: float,
    epochs: int,
    batch_size: int,
) -> DirectionTrained:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available.")
    set_seed(seed)
    train_mask, val_mask, _ = get_protocol_splits(df, protocol_id)
    train_mask = train_mask & mlbase.direction_train_mask(df)
    val_mask = val_mask & mlbase.direction_train_mask(df)
    if train_mask.sum() < 3 or val_mask.sum() < 1:
        raise ValueError(f"Insufficient direction samples for {protocol_id}")
    y_train = df.loc[train_mask, "direction_id"].astype(int).to_numpy()
    y_val = df.loc[val_mask, "direction_id"].astype(int).to_numpy()
    if spec["kind"] == "summary":
        features = direction_feature_sets[spec["feature_set"]]
        scaler = StandardScaler()
        X_train_np = scaler.fit_transform(df.loc[train_mask, features].astype(float).to_numpy()).astype(np.float32)
        X_val_np = scaler.transform(df.loc[val_mask, features].astype(float).to_numpy()).astype(np.float32)
        model = SummaryMLPDirection(X_train_np.shape[1])
        feature_source = spec["feature_set"]
    else:
        features = None
        mean = X_temporal[train_mask].mean(axis=(0, 1), keepdims=True)
        std = X_temporal[train_mask].std(axis=(0, 1), keepdims=True) + 1e-6
        X_train_np = ((X_temporal[train_mask] - mean) / std).astype(np.float32)
        X_val_np = ((X_temporal[val_mask] - mean) / std).astype(np.float32)
        model = spec["builder"]()
        feature_source = "temporal_tilt12"
        scaler = {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}
    classes = np.array([0, 1, 2])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_np, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
    )
    X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
    best_state = None
    best_macro = -1.0
    best_loss = math.inf
    best_epoch = 0
    no_improve = 0
    patience = 12
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(X_val_t)
            val_loss = float(criterion(logits, torch.tensor(y_val, dtype=torch.long)).item())
            pred = torch.argmax(logits, dim=1).cpu().numpy()
        macro = f1_score(y_val, pred, labels=[0, 1, 2], average="macro", zero_division=0)
        if macro > best_macro + 1e-8 or (abs(macro - best_macro) <= 1e-8 and val_loss < best_loss):
            best_macro = float(macro)
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    params = int(sum(p.numel() for p in model.parameters()))
    complexity = estimate_complexity(model, params, spec["kind"])
    trained = DirectionTrained(
        model=model,
        model_name=spec["model_name"],
        feature_source=feature_source,
        features=features,
        params=params,
        protocol_id=protocol_id,
        seed=seed,
        learning_rate=lr,
        best_epoch=best_epoch,
        best_val_macro_f1=best_macro,
        val_loss_at_best=best_loss,
        complexity=complexity,
    )
    trained.scaler = scaler  # type: ignore[attr-defined]
    return trained


def predict_direction(trained: DirectionTrained, df: pd.DataFrame, X_temporal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if trained.feature_source == "temporal_tilt12":
        scaler = trained.scaler  # type: ignore[attr-defined]
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        std = np.asarray(scaler["std"], dtype=np.float32)
        X = ((X_temporal[mask] - mean) / std).astype(np.float32)
    else:
        scaler = trained.scaler  # type: ignore[attr-defined]
        X = scaler.transform(df.loc[mask, trained.features].astype(float).to_numpy()).astype(np.float32)
    trained.model.eval()
    with torch.no_grad():
        logits = trained.model(torch.tensor(X, dtype=torch.float32))
        pred = torch.argmax(logits, dim=1).cpu().numpy().astype(int)
    return pred


def evaluate_direction(
    df: pd.DataFrame,
    mask: np.ndarray,
    pred_all: np.ndarray,
    run_id: str,
    experiment_id: str,
    train_protocol: str,
    trained: DirectionTrained,
    a5_direction_rows: pd.DataFrame,
) -> dict[str, Any]:
    sub = df.loc[mask].reset_index(drop=True)
    sup = mlbase.direction_train_mask(sub)
    y = sub.loc[sup, "direction_id"].astype(int).to_numpy()
    pred = pred_all[sup]
    cm = confusion_matrix(y, pred, labels=[0, 1, 2]) if len(y) else np.zeros((3, 3), dtype=int)
    per = f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0) if len(y) else [np.nan, np.nan, np.nan]
    a5_f1 = np.nan
    rows = a5_direction_rows[a5_direction_rows["experiment_id"].eq(experiment_id)]
    if not rows.empty:
        a5_f1 = float(rows.iloc[0]["direction_macro_f1"])
    macro = f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0) if len(y) else np.nan
    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "train_protocol": train_protocol,
        "model_name": trained.model_name,
        "feature_source": trained.feature_source,
        "seed": trained.seed,
        "learning_rate": trained.learning_rate,
        "best_epoch": trained.best_epoch,
        "best_val_macro_f1": trained.best_val_macro_f1,
        "direction_macro_f1": macro,
        "direction_accuracy": accuracy_score(y, pred) if len(y) else np.nan,
        "forward_f1": per[0],
        "backward_f1": per[1],
        "lateral_f1": per[2],
        "direction_n_supervised": len(y),
        "A5_direction_macro_f1": a5_f1,
        "delta_vs_A5": macro - a5_f1 if not pd.isna(a5_f1) else np.nan,
        "params": trained.params,
        "estimated_fp32_kb": trained.params * 4 / 1024,
        "estimated_int8_kb": trained.params / 1024,
        "tflite_convert": "not_available_tensorflow_import_failed" if not TENSORFLOW_AVAILABLE else "not_attempted_pytorch_model",
        "int8_tflite_convert": "not_available_tensorflow_import_failed" if not TENSORFLOW_AVAILABLE else "not_attempted_pytorch_model",
        "cm_00": int(cm[0, 0]),
        "cm_01": int(cm[0, 1]),
        "cm_02": int(cm[0, 2]),
        "cm_10": int(cm[1, 0]),
        "cm_11": int(cm[1, 1]),
        "cm_12": int(cm[1, 2]),
        "cm_20": int(cm[2, 0]),
        "cm_21": int(cm[2, 1]),
        "cm_22": int(cm[2, 2]),
        **trained.complexity,
    }


def estimate_complexity(model: Any, params: int, kind: str) -> dict[str, Any]:
    fp32 = params * 4 / 1024
    int8 = params / 1024
    if params < 10_000:
        edge = "high"
    elif params < 30_000:
        edge = "medium-high"
    else:
        edge = "medium"
    return {
        "estimated_fp32_kb": fp32,
        "estimated_int8_kb": int8,
        "macs_estimate": np.nan,
        "edge_suitability": edge,
        "complexity_note": f"PyTorch {kind}; params={params}; TFLite unavailable because TensorFlow import status: {TENSORFLOW_AVAILABLE}",
    }


def run_lightweight_direction(
    repo_root: Path,
    df: pd.DataFrame,
    X_temporal: np.ndarray,
    direction_feature_sets: dict[str, list[str]],
    best_fall_key: str,
    fall_predictions: dict[tuple[str, str], np.ndarray],
    a5_direction: pd.DataFrame,
    report_dir: Path,
    figure_dir: Path,
    model_dir: Path,
    seeds: list[int],
    quick: bool,
    epochs: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, Any]] = []
    e2e_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    trained_cache: dict[tuple[str, str, int, float], DirectionTrained] = {}
    if not TORCH_AVAILABLE:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(
            [{"stage": "direction", "error": "PyTorch is not available"}]
        )
    lrs = [1e-3] if quick else [1e-3, 5e-4]
    protocols = ["E3_BITS_WEDA_MIXED"] if quick else BASE_PROTOCOLS
    for spec in direction_model_specs(quick):
        for seed in seeds:
            for lr in lrs:
                for protocol_id in protocols:
                    try:
                        print(f"[direction] {spec['model_name']} seed={seed} lr={lr} {protocol_id}")
                        trained = train_direction_model(df, X_temporal, protocol_id, spec, direction_feature_sets, seed, lr, epochs, batch_size)
                        key = (spec["model_name"], protocol_id, seed, lr)
                        trained_cache[key] = trained
                        torch.save(
                            {
                                "model_name": trained.model_name,
                                "state_dict": trained.model.state_dict(),
                                "feature_source": trained.feature_source,
                                "features": trained.features,
                                "params": trained.params,
                                "seed": seed,
                                "learning_rate": lr,
                            },
                            model_dir / f"{trained.model_name}_{protocol_id}_seed{seed}_lr{lr}.pt",
                        )
                        complexity_rows.append(
                            {
                                "model_name": trained.model_name,
                                "protocol_id": protocol_id,
                                "seed": seed,
                                "learning_rate": lr,
                                "params": trained.params,
                                "estimated_fp32_kb": trained.params * 4 / 1024,
                                "estimated_int8_kb": trained.params / 1024,
                                "best_val_macro_f1": trained.best_val_macro_f1,
                                "tflite_convert": "not_available_tensorflow_import_failed" if not TENSORFLOW_AVAILABLE else "not_attempted_pytorch_model",
                                "int8_tflite_convert": "not_available_tensorflow_import_failed" if not TENSORFLOW_AVAILABLE else "not_attempted_pytorch_model",
                                **trained.complexity,
                            }
                        )
                    except Exception as exc:
                        failed.append({"stage": "train_direction", "model_name": spec["model_name"], "protocol_id": protocol_id, "seed": seed, "learning_rate": lr, "error": repr(exc)})
    eval_protocols = ["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"] if quick else ALL_PROTOCOLS
    for key, trained in trained_cache.items():
        model_name, protocol_id, seed, lr = key
        for eval_id in eval_protocols:
            train_protocol = canonical_train_protocol(eval_id)
            if train_protocol != protocol_id:
                continue
            try:
                _, _, mask = get_protocol_splits(df, eval_id)
                pred = predict_direction(trained, df, X_temporal, mask)
                run_id = f"{model_name}_{protocol_id}_seed{seed}_lr{lr}"
                results.append(evaluate_direction(df, mask, pred, run_id, eval_id, protocol_id, trained, a5_direction))
                fall_pred = fall_predictions.get((best_fall_key, eval_id))
                if fall_pred is not None:
                    e2e_rows.append(evaluate_e2e_direction(df, mask, fall_pred, pred, run_id, eval_id, protocol_id, "lightweight_direction_with_best_fall"))
                save_direction_cm(results[-1], figure_dir / "confusion_matrices")
            except Exception as exc:
                failed.append({"stage": "eval_direction", "model_name": model_name, "protocol_id": eval_id, "seed": seed, "learning_rate": lr, "error": repr(exc)})
    return pd.DataFrame(results), best_by_protocol_direction(pd.DataFrame(results)), pd.DataFrame(complexity_rows), pd.DataFrame(e2e_rows), pd.DataFrame(failed)


def best_by_protocol_direction(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for experiment_id, group in df.groupby("experiment_id"):
        rows.append(group.sort_values(["direction_macro_f1", "direction_accuracy", "params"], ascending=[False, False, True]).iloc[0].to_dict())
    return pd.DataFrame(rows).sort_values("experiment_id")


def best_by_protocol_fall(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for experiment_id, group in df.groupby("experiment_id"):
        rows.append(group.sort_values(["fall_f1", "fall_precision", "fall_recall", "FP"], ascending=[False, False, False, True]).iloc[0].to_dict())
    return pd.DataFrame(rows).sort_values("experiment_id")


def save_fall_cm(row: dict[str, Any], out_dir: Path) -> None:
    ensure_dir(out_dir)
    cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]], dtype=int)
    save_cm(cm, ["non_fall", "fall"], out_dir / f"{safe_name(row['run_id'])}_{row['experiment_id']}_fall.png", f"{row['run_id']} {row['experiment_id']} fall")


def save_direction_cm(row: dict[str, Any], out_dir: Path) -> None:
    ensure_dir(out_dir)
    cm = np.array(
        [
            [row["cm_00"], row["cm_01"], row["cm_02"]],
            [row["cm_10"], row["cm_11"], row["cm_12"]],
            [row["cm_20"], row["cm_21"], row["cm_22"]],
        ],
        dtype=int,
    )
    save_cm(cm, DIRECTION_LABELS, out_dir / f"{safe_name(row['run_id'])}_{row['experiment_id']}_direction.png", f"{row['run_id']} {row['experiment_id']} direction")


def save_cm(cm: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    plt.figure(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))


def safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return math.nan


def make_plots(report_dir: Path, figure_dir: Path) -> None:
    fall_path = report_dir / "fall_feature_reduction_results.csv"
    dir_path = report_dir / "lightweight_direction_results.csv"
    if fall_path.exists():
        fall = pd.read_csv(fall_path)
        focus = fall[fall["experiment_id"].isin(["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"])].copy()
        if not focus.empty:
            plt.figure(figsize=(10, 5))
            sns.scatterplot(data=focus, x="feature_count", y="fall_f1", hue="feature_set", style="experiment_id", alpha=0.8)
            plt.tight_layout()
            plt.savefig(figure_dir / "fall_feature_count_vs_f1.png", dpi=150)
            plt.close()
    if dir_path.exists():
        d = pd.read_csv(dir_path)
        focus = d[d["experiment_id"].isin(["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"])].copy()
        if not focus.empty:
            plt.figure(figsize=(9, 5))
            sns.scatterplot(data=focus, x="params", y="direction_macro_f1", hue="model_name", style="experiment_id")
            plt.tight_layout()
            plt.savefig(figure_dir / "params_vs_f1.png", dpi=150)
            plt.close()
    if fall_path.exists() and dir_path.exists():
        fall = pd.read_csv(fall_path)
        focus = fall[fall["experiment_id"].isin(["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"])].copy()
        if not focus.empty:
            top = focus.sort_values(["experiment_id", "fall_f1"], ascending=[True, False]).groupby("experiment_id").head(8)
            plt.figure(figsize=(12, 5))
            sns.barplot(data=top, x="experiment_id", y="fall_f1", hue="feature_set")
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.savefig(figure_dir / "e3_e6_e7_comparison.png", dpi=150)
            plt.close()


def save_reports(
    report_dir: Path,
    figure_dir: Path,
    feature_sets: dict[str, list[str]],
    direction_feature_sets: dict[str, list[str]],
    missing_features: pd.DataFrame,
    fall_results: pd.DataFrame,
    fall_importance: pd.DataFrame,
    selected_features: pd.DataFrame,
    fall_e2e: pd.DataFrame,
    fall_failed: pd.DataFrame,
    lightweight_results: pd.DataFrame,
    lightweight_best: pd.DataFrame,
    lightweight_complexity: pd.DataFrame,
    lightweight_e2e: pd.DataFrame,
    lightweight_failed: pd.DataFrame,
    a5_fall: pd.DataFrame,
    a5_direction: pd.DataFrame,
) -> None:
    fall_best = best_by_protocol_fall(fall_results)
    fall_results.to_csv(report_dir / "fall_feature_reduction_results.csv", index=False)
    fall_best.to_csv(report_dir / "fall_feature_reduction_best_by_protocol.csv", index=False)
    fall_importance.to_csv(report_dir / "fall_feature_reduction_feature_importance.csv", index=False)
    selected_features.to_csv(report_dir / "fall_selected_features_by_protocol.csv", index=False)
    fall_e2e.to_csv(report_dir / "fall_feature_reduction_e2e_direction.csv", index=False)
    lightweight_results.to_csv(report_dir / "lightweight_direction_results.csv", index=False)
    lightweight_best.to_csv(report_dir / "lightweight_direction_best_by_protocol.csv", index=False)
    lightweight_complexity.to_csv(report_dir / "lightweight_direction_complexity.csv", index=False)
    lightweight_e2e.to_csv(report_dir / "lightweight_direction_e2e_results.csv", index=False)
    failed = pd.concat([fall_failed, lightweight_failed], ignore_index=True)
    failed.to_csv(report_dir / "failed_runs.csv", index=False)
    missing_features.to_csv(report_dir / "missing_features.csv", index=False)
    vs_a5 = build_lightweight_vs_a5(lightweight_results, a5_direction)
    vs_a5.to_csv(report_dir / "lightweight_direction_vs_a5.csv", index=False)
    final_cmp = build_final_comparison(fall_results, fall_e2e, lightweight_results, lightweight_e2e, a5_fall, a5_direction)
    final_cmp.to_csv(report_dir / "final_hybrid_variants_comparison.csv", index=False)
    configs = {
        "fall_feature_sets": feature_sets,
        "direction_feature_sets": direction_feature_sets,
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "tensorflow_import_error": TENSORFLOW_IMPORT_ERROR,
        "torch_available": TORCH_AVAILABLE,
        "fulltiming_used": False,
    }
    (report_dir / "feature_set_configs.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")
    make_plots(report_dir, figure_dir)
    write_markdown_summary(report_dir, fall_results, fall_best, selected_features, lightweight_results, lightweight_best, lightweight_complexity, lightweight_e2e, final_cmp, failed)


def build_lightweight_vs_a5(results: pd.DataFrame, a5_direction: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows = []
    for row in results.to_dict("records"):
        a5 = a5_direction[a5_direction["experiment_id"].eq(row["experiment_id"])]
        a5_f1 = float(a5.iloc[0]["direction_macro_f1"]) if not a5.empty else np.nan
        rows.append({**row, "a5_direction_macro_f1": a5_f1, "macro_f1_delta_vs_a5": row["direction_macro_f1"] - a5_f1 if not pd.isna(a5_f1) else np.nan})
    return pd.DataFrame(rows)


def build_final_comparison(fall_results: pd.DataFrame, fall_e2e: pd.DataFrame, lightweight_results: pd.DataFrame, lightweight_e2e: pd.DataFrame, a5_fall: pd.DataFrame, a5_direction: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exp in ["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"]:
        a5f = a5_fall[a5_fall["experiment_id"].eq(exp)]
        a5d = a5_direction[a5_direction["experiment_id"].eq(exp)]
        if not a5f.empty and not a5d.empty:
            rows.append(
                {
                    "variant": "A5_reference",
                    "experiment_id": exp,
                    "fall_f1": a5f.iloc[0]["fall_f1"],
                    "fall_precision": a5f.iloc[0]["fall_precision"],
                    "fall_recall": a5f.iloc[0]["fall_recall"],
                    "FP": a5f.iloc[0]["FP"],
                    "direction_macro_f1": a5d.iloc[0]["direction_macro_f1"],
                    "params_direction": 65959,
                    "fall_feature_count": 12,
                }
            )
    focus_fall = fall_results[fall_results["experiment_id"].isin(["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"])]
    if not focus_fall.empty:
        for name, group in focus_fall.groupby(["model_name", "feature_set"]):
            avg = group["fall_f1"].mean()
            e7 = group[group["experiment_id"].eq("E7_E3_MIXED_TEST_WEDA")]
            if not e7.empty:
                rows.append(
                    {
                        "variant": f"FallReduced_{name[0]}_{name[1]}_A5Direction",
                        "experiment_id": "E3/E6/E7_avg",
                        "avg_fall_f1": avg,
                        "E7_fall_f1": e7.iloc[0]["fall_f1"],
                        "E7_FP": e7.iloc[0]["FP"],
                        "fall_feature_count": e7.iloc[0]["feature_count"],
                    }
                )
    if not lightweight_results.empty:
        focus = lightweight_results[lightweight_results["experiment_id"].isin(["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"])]
        for name, group in focus.groupby(["model_name", "seed", "learning_rate"]):
            rows.append(
                {
                    "variant": f"LightDir_{name[0]}_seed{name[1]}_lr{name[2]}",
                    "experiment_id": "E3/E6/E7_avg",
                    "avg_direction_macro_f1": group["direction_macro_f1"].mean(),
                    "params_direction": group["params"].iloc[0],
                    "E7_direction_macro_f1": group.loc[group["experiment_id"].eq("E7_E3_MIXED_TEST_WEDA"), "direction_macro_f1"].mean(),
                }
            )
    return pd.DataFrame(rows)


def write_markdown_summary(
    report_dir: Path,
    fall_results: pd.DataFrame,
    fall_best: pd.DataFrame,
    selected_features: pd.DataFrame,
    lightweight_results: pd.DataFrame,
    lightweight_best: pd.DataFrame,
    lightweight_complexity: pd.DataFrame,
    lightweight_e2e: pd.DataFrame,
    final_cmp: pd.DataFrame,
    failed: pd.DataFrame,
) -> None:
    lines = ["# Hybrid Feature Reduction And Lightweight Direction Summary", ""]
    lines += ["## Fall Feature Reduction", ""]
    focus = fall_results[fall_results["experiment_id"].isin(["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"])] if not fall_results.empty else pd.DataFrame()
    if not focus.empty:
        agg = focus.groupby(["model_name", "feature_set", "feature_count"], as_index=False).agg(
            avg_fall_f1=("fall_f1", "mean"),
            e7_fall_f1=("fall_f1", lambda s: s[focus.loc[s.index, "experiment_id"].eq("E7_E3_MIXED_TEST_WEDA")].mean()),
            e7_recall=("fall_recall", lambda s: s[focus.loc[s.index, "experiment_id"].eq("E7_E3_MIXED_TEST_WEDA")].mean()),
            e7_fp=("FP", lambda s: s[focus.loc[s.index, "experiment_id"].eq("E7_E3_MIXED_TEST_WEDA")].mean()),
        ).sort_values(["e7_fall_f1", "avg_fall_f1"], ascending=False)
        lines += [markdown_table(format_df(agg.head(30))), ""]
        best_e7 = agg.sort_values(["e7_fall_f1", "feature_count"], ascending=[False, True]).iloc[0]
        compact = agg[(agg["feature_count"] < 12) & (agg["e7_fall_f1"] >= 0.82)]
        compact_row = compact.sort_values(["feature_count", "e7_fall_f1"], ascending=[True, False]).iloc[0] if not compact.empty else None
        lines += [
            "### Fall Answers",
            "",
            f"1. Full 132 is not automatically required if a compact row stays close; best E7 fall row is `{best_e7['model_name']} + {best_e7['feature_set']}` with E7 F1={best_e7['e7_fall_f1']:.4f}.",
            f"2. Best <12 feature candidate: `{compact_row['model_name']} + {compact_row['feature_set']}` with {int(compact_row['feature_count'])} features and E7 F1={compact_row['e7_fall_f1']:.4f}." if compact_row is not None else "2. No <12 feature candidate reached E7 Fall F1 >= 0.82.",
            "3. Check `fall_feature_reduction_results.csv` for exact E7 FP/recall per model.",
            "4. SelectedByImportance repeat features are summarized below.",
            "",
        ]
    if not selected_features.empty:
        top_sel = selected_features["feature"].value_counts().head(20).reset_index()
        top_sel.columns = ["feature", "times_selected"]
        lines += ["### Repeated SelectedByImportance Features", "", markdown_table(top_sel), ""]
    lines += ["## Lightweight Direction", ""]
    if not lightweight_results.empty:
        focusd = lightweight_results[lightweight_results["experiment_id"].isin(["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"])]
        aggd = focusd.groupby(["model_name", "seed", "learning_rate", "params"], as_index=False).agg(
            avg_direction_macro_f1=("direction_macro_f1", "mean"),
            e3_direction_macro_f1=("direction_macro_f1", lambda s: s[focusd.loc[s.index, "experiment_id"].eq("E3_BITS_WEDA_MIXED")].mean()),
            e7_direction_macro_f1=("direction_macro_f1", lambda s: s[focusd.loc[s.index, "experiment_id"].eq("E7_E3_MIXED_TEST_WEDA")].mean()),
        ).sort_values(["avg_direction_macro_f1", "params"], ascending=[False, True])
        lines += [markdown_table(format_df(aggd.head(30))), ""]
        best = aggd.iloc[0]
        lt30 = aggd[(aggd["params"] < 30000) & (aggd["e3_direction_macro_f1"] >= 0.86)]
        lt10 = aggd[(aggd["params"] < 10000) & (aggd["e3_direction_macro_f1"] >= 0.84)]
        lines += [
            "### Direction Answers",
            "",
            f"1. Best lightweight direction model overall: `{best['model_name']}` seed={int(best['seed'])} lr={best['learning_rate']} with avg E3/E6/E7 macro F1={best['avg_direction_macro_f1']:.4f}, params={int(best['params'])}.",
            "2. A <30k params model reaches E3 Direction Macro F1 >= 0.86." if not lt30.empty else "2. No <30k params model reaches E3 Direction Macro F1 >= 0.86.",
            "3. A <10k params model reaches E3 Direction Macro F1 >= 0.84." if not lt10.empty else "3. No <10k params model reaches E3 Direction Macro F1 >= 0.84.",
            f"4. TFLite/INT8 conversion status: TensorFlow available={TENSORFLOW_AVAILABLE}; error={TENSORFLOW_IMPORT_ERROR or 'none'}.",
            "5. Lightweight direction should replace A5 only if E3/E6/E7 consistency is close to A5; otherwise keep A5 direction.",
            "",
        ]
    else:
        lines += ["No lightweight direction rows were produced.", ""]
    lines += [
        "## Final Recommendation",
        "",
        "- Practical main: keep `Hybrid_NoTiming_Full` unless compact fall feature sets match E7 F1 and FP.",
        "- Compact fall: choose the smallest feature set with E7 Fall F1 >= 0.82 and acceptable FP.",
        "- Direction: keep A5 direction head unless a lightweight model stays within 0.04 macro F1 on E3 and remains stable on E6/E7.",
        "- FullTiming is not used as a main result.",
        "",
    ]
    if not failed.empty:
        lines += ["## Failed Runs / Limitations", "", markdown_table(format_df(failed)), ""]
    summary = "\n".join(lines)
    (report_dir / "experiment_summary.md").write_text(summary, encoding="utf-8")
    (report_dir / "final_recommendation.md").write_text(summary, encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    compact = df.astype(object).where(pd.notna(df), "")
    header = "| " + " | ".join(map(str, compact.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(compact.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in compact.to_numpy()]
    return "\n".join([header, sep] + rows)


def format_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            if col in {"feature_count", "FP", "FN", "TP", "TN", "params", "seed"}:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(int(v)))
            else:
                out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    return out


def main() -> None:
    args = parse_args()
    if args.full:
        args.run_all = True
    if not (args.run_all or args.run_fall_reduction_only or args.run_lightweight_direction_only):
        print("Use --run-all, --run-fall-reduction-only, or --run-lightweight-direction-only.")
        return
    repo_root = Path(args.repo_root).resolve()
    report_dir = ensure_dir(repo_root / "outputs" / "reports" / "hybrid_feature_reduction_direction_lightweight")
    figure_dir = ensure_dir(repo_root / "outputs" / "figures" / "hybrid_feature_reduction_direction_lightweight")
    model_dir = ensure_dir(repo_root / "outputs" / "models" / "hybrid_feature_reduction_direction_lightweight")
    ensure_dir(figure_dir / "confusion_matrices")
    seed = args.seeds[0]
    set_seed(seed)
    print("[1/7] Loading summary feature matrix")
    df = load_summary_feature_matrix(repo_root)
    print("[2/7] Loading/rebuilding temporal tilt12 data")
    X_temporal, temporal_meta = load_temporal_data(repo_root, df)
    print(f"Temporal shape: {X_temporal.shape}")
    print("[3/7] Building feature sets")
    fall_feature_sets, missing = build_fall_feature_sets(df)
    direction_feature_sets = build_direction_feature_sets(df)
    a5_fall, a5_direction = mlbase.load_a5_reference_rows()
    fall_results = pd.DataFrame()
    fall_importance = pd.DataFrame()
    selected_features = pd.DataFrame()
    fall_e2e = pd.DataFrame()
    fall_failed = pd.DataFrame()
    fall_predictions: dict[tuple[str, str], np.ndarray] = {}
    if args.run_all or args.run_fall_reduction_only:
        print("[4/7] Running fall feature reduction")
        fall_results, fall_importance, selected_features, fall_e2e, fall_failed, fall_predictions = run_fall_feature_reduction(
            repo_root,
            df,
            fall_feature_sets,
            a5_fall,
            a5_direction,
            report_dir,
            figure_dir,
            model_dir,
            seed,
            args.quick,
        )
    else:
        old = report_dir / "fall_feature_reduction_results.csv"
        if old.exists():
            fall_results = pd.read_csv(old)
    best_fall_key = choose_best_fall_key(fall_results)
    if not fall_predictions and not fall_results.empty:
        # Rebuild only the baseline full GB fall predictions if direction-only is run after a previous fall run.
        fall_predictions = {}
    lightweight_results = pd.DataFrame()
    lightweight_best = pd.DataFrame()
    lightweight_complexity = pd.DataFrame()
    lightweight_e2e = pd.DataFrame()
    lightweight_failed = pd.DataFrame()
    if args.run_all or args.run_lightweight_direction_only:
        if not fall_predictions:
            print("[5/7] Recomputing baseline fall predictions for direction E2E")
            _, _, _, _, _, fall_predictions = run_fall_feature_reduction(
                repo_root,
                df,
                {"FallNoTiming_Full_132": fall_feature_sets["FallNoTiming_Full_132"]},
                a5_fall,
                a5_direction,
                report_dir,
                figure_dir,
                model_dir,
                seed,
                True,
            )
            best_fall_key = choose_best_fall_key(pd.read_csv(report_dir / "fall_feature_reduction_results.csv") if (report_dir / "fall_feature_reduction_results.csv").exists() else fall_results)
        print("[6/7] Running lightweight direction")
        lightweight_results, lightweight_best, lightweight_complexity, lightweight_e2e, lightweight_failed = run_lightweight_direction(
            repo_root,
            df,
            X_temporal,
            direction_feature_sets,
            best_fall_key,
            fall_predictions,
            a5_direction,
            report_dir,
            figure_dir,
            model_dir,
            args.seeds,
            args.quick,
            args.epochs if not args.quick else min(args.epochs, 30),
            args.batch_size,
        )
    print("[7/7] Saving reports")
    save_reports(
        report_dir,
        figure_dir,
        fall_feature_sets,
        direction_feature_sets,
        missing,
        fall_results,
        fall_importance,
        selected_features,
        fall_e2e,
        fall_failed,
        lightweight_results,
        lightweight_best,
        lightweight_complexity,
        lightweight_e2e,
        lightweight_failed,
        a5_fall,
        a5_direction,
    )
    print("Reports saved:")
    print(report_dir / "experiment_summary.md")
    print(report_dir / "final_recommendation.md")
    print(report_dir / "fall_feature_reduction_results.csv")
    print(report_dir / "lightweight_direction_results.csv")
    if not fall_results.empty:
        focus = fall_results[fall_results["experiment_id"].eq("E7_E3_MIXED_TEST_WEDA")].sort_values(["fall_f1", "fall_precision"], ascending=False).head(10)
        print("\nTop E7 fall rows:")
        print(focus[["model_name", "feature_set", "feature_count", "fall_f1", "fall_precision", "fall_recall", "FP"]].to_string(index=False))
    if not lightweight_results.empty:
        focus = lightweight_results[lightweight_results["experiment_id"].isin(["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"])]
        print("\nTop direction rows:")
        print(focus.sort_values(["direction_macro_f1", "params"], ascending=[False, True])[["model_name", "experiment_id", "params", "direction_macro_f1", "direction_accuracy"]].head(12).to_string(index=False))


def choose_best_fall_key(fall_results: pd.DataFrame) -> str:
    if fall_results.empty:
        return "GradientBoosting_FallNoTiming_Full_132"
    e7 = fall_results[fall_results["experiment_id"].eq("E7_E3_MIXED_TEST_WEDA")]
    if e7.empty:
        return "GradientBoosting_FallNoTiming_Full_132"
    row = e7.sort_values(["fall_f1", "fall_precision", "fall_recall"], ascending=False).iloc[0]
    return f"{row['model_name']}_{row['feature_set']}"


if __name__ == "__main__":
    main()
