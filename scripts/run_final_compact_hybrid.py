from __future__ import annotations

import argparse
import json
import math
import pickle
import platform
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
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

import run_a5_direction_vs_ldx1_multiseed as edgebase
import run_hybrid_feature_reduction_and_lightweight_direction as hybrid
import run_ml_e1_e7_specialized as mlbase

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf  # noqa: F401

    TENSORFLOW_AVAILABLE = True
    TENSORFLOW_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    TENSORFLOW_AVAILABLE = False
    TENSORFLOW_IMPORT_ERROR = repr(exc)


DIRECTION_LABELS = ["forward", "backward", "lateral"]
FOCUS_PROTOCOLS = ["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"]
ALL_PROTOCOLS = [
    "E1_BITS_TO_BITS",
    "E2_WEDA_TO_WEDA",
    "E3_BITS_WEDA_MIXED",
    "E4_BITS_TO_WEDA",
    "E5_WEDA_TO_BITS",
    "E6_E3_MIXED_TEST_BITS",
    "E7_E3_MIXED_TEST_WEDA",
]
DEFAULT_SEEDS = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 42, 43, 44, 45, 46, 50, 60, 70, 80, 90, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final DS-Fall-RD Compact Hybrid.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--seed-search", action="store_true")
    parser.add_argument("--build-final", action="store_true")
    parser.add_argument("--selection-mode", choices=["paper_safe", "best_observed"], default="paper_safe")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--learning-rates", "--lrs", dest="learning_rates", nargs="+", type=float, default=[1e-3, 5e-4])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def canonical_train_protocol(protocol_id: str) -> str:
    return "E3_BITS_WEDA_MIXED" if protocol_id in {"E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"} else protocol_id


def direction_mask(df: pd.DataFrame) -> np.ndarray:
    return mlbase.direction_train_mask(df)


def load_data(repo_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    final_report_dir = repo_root / "outputs" / "reports" / "final_compact_hybrid"
    final_model_dir = repo_root / "outputs" / "models" / "final_compact_hybrid"
    summary_cache = final_report_dir / "summary_feature_matrix_bits_weda.csv"
    x_cache = final_model_dir / "cached_event_25hz_tilt12_X.npy"
    meta_cache = final_model_dir / "cached_event_25hz_metadata.csv"
    if summary_cache.exists() and x_cache.exists() and meta_cache.exists():
        df = mlbase.load_summary(summary_cache)
        meta = pd.read_csv(meta_cache)
        X_all = np.load(x_cache)
        pos = pd.Series(np.arange(len(meta)), index=meta["window_id"])
        idx = pos.loc[df["window_id"]].to_numpy(dtype=int)
        X_temporal = X_all[idx]
    else:
        df = hybrid.load_summary_feature_matrix(repo_root)
        X_temporal, meta = hybrid.load_temporal_data(repo_root, df)
        ensure_dir(final_report_dir)
        ensure_dir(final_model_dir)
        df.to_csv(summary_cache, index=False)
        np.save(x_cache, X_temporal)
        meta.to_csv(meta_cache, index=False)
    return df, X_temporal.astype(np.float32)


def statistical_fall_features(df: pd.DataFrame) -> list[str]:
    features = [f for f in hybrid.FALL_FULL_PROMPT_FEATURES if f in df.columns and not mlbase.is_timing_feature(f)]
    if not features:
        signal = mlbase.infer_signal_features(df)
        features = [f for f in signal if not mlbase.is_timing_feature(f)]
    forbidden = [f for f in features if mlbase.is_timing_feature(f)]
    if forbidden:
        raise ValueError(f"Timing-index artifacts leaked into fall features: {forbidden}")
    return features


def train_statistical_fall_expert(df: pd.DataFrame, features: list[str], seed: int = 42) -> dict[str, Any]:
    train_mask, val_mask, _ = hybrid.get_protocol_splits(df, "E3_BITS_WEDA_MIXED")
    model = make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=seed))
    model.fit(df.loc[train_mask, features].astype(float).to_numpy(), df.loc[train_mask, "fall_label"].astype(int).to_numpy())
    val_scores = hybrid.fall_scores(model, df.loc[val_mask, features].astype(float).to_numpy())
    threshold, note = mlbase.select_fall_threshold(df.loc[val_mask, "fall_label"].astype(int).to_numpy(), val_scores)
    return {"model": model, "features": features, "threshold": float(threshold), "threshold_note": note, "seed": seed}


def fall_predictions_for_protocol(df: pd.DataFrame, fall_expert: dict[str, Any], protocol_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, mask = hybrid.get_protocol_splits(df, protocol_id)
    X = df.loc[mask, fall_expert["features"]].astype(float).to_numpy()
    scores = hybrid.fall_scores(fall_expert["model"], X)
    pred = (scores >= fall_expert["threshold"]).astype(int)
    return mask, pred, scores


def evaluate_fall(df: pd.DataFrame, mask: np.ndarray, pred: np.ndarray, scores: np.ndarray, method: str, protocol_id: str, feature_count: int) -> dict[str, Any]:
    y = df.loc[mask, "fall_label"].astype(int).to_numpy()
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel()]
    return {
        "method": method,
        "experiment_id": protocol_id,
        "test_dataset": "+".join(sorted(df.loc[mask, "dataset"].unique())),
        "feature_input": "event-window statistical features",
        "feature_count": int(feature_count),
        "fall_precision": float(precision_score(y, pred, zero_division=0)),
        "fall_recall": float(recall_score(y, pred, zero_division=0)),
        "fall_f1": float(f1_score(y, pred, zero_division=0)),
        "fall_accuracy": float(accuracy_score(y, pred)),
        "AUROC": safe_auc(y, scores),
        "AUPRC": safe_ap(y, scores),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


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


def evaluate_direction(df: pd.DataFrame, mask: np.ndarray, pred_all: np.ndarray, method: str, protocol_id: str, params: int) -> dict[str, Any]:
    sub = df.loc[mask].reset_index(drop=True)
    sup = direction_mask(sub)
    y = sub.loc[sup, "direction_id"].astype(int).to_numpy()
    pred = pred_all[sup]
    cm = confusion_matrix(y, pred, labels=[0, 1, 2]) if len(y) else np.zeros((3, 3), dtype=int)
    per = f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0) if len(y) else [np.nan, np.nan, np.nan]
    return {
        "method": method,
        "experiment_id": protocol_id,
        "test_dataset": "+".join(sorted(sub["dataset"].unique())),
        "input": "temporal tilt12 sequence 50x12",
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


def evaluate_e2e(df: pd.DataFrame, mask: np.ndarray, fall_pred: np.ndarray, direction_pred: np.ndarray, method: str, protocol_id: str) -> dict[str, Any]:
    row = hybrid.evaluate_e2e_direction(
        df=df,
        mask=mask,
        fall_pred=fall_pred,
        direction_pred=direction_pred,
        pipeline=method,
        experiment_id=protocol_id,
        train_protocol="E3_BITS_WEDA_MIXED",
        source="DS-Fall-RD Compact Hybrid",
    )
    row["method"] = method
    return row


def run_ldx1_seed_search(
    repo_root: Path,
    df: pd.DataFrame,
    X_temporal: np.ndarray,
    fall_expert: dict[str, Any],
    report_dir: Path,
    figure_dir: Path,
    model_dir: Path,
    seeds: list[int],
    learning_rates: list[float],
    epochs: int,
    batch_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    spec = [s for s in edgebase.model_specs() if s["model_name"] == "LDX1_D1_Wide_DSConv"][0]
    for seed in seeds:
        for lr in learning_rates:
            print(f"[seed-search] LDX1 seed={seed} lr={lr}")
            run = edgebase.train_direction_model(df, X_temporal, "E3_BITS_WEDA_MIXED", spec, seed, lr, epochs, batch_size)
            run_id = f"LDX1_D1_Wide_DSConv_E3_BITS_WEDA_MIXED_seed{seed}_lr{lr}"
            torch.save(
                {
                    "model_name": run.model_name,
                    "state_dict": run.model.state_dict(),
                    "scaler": run.scaler,
                    "seed": seed,
                    "learning_rate": lr,
                    "params": run.params,
                },
                model_dir / f"{run_id}.pt",
            )
            metrics: dict[str, float] = {}
            e2e_vals: dict[str, float] = {}
            val_test_gap = np.nan
            for protocol_id in FOCUS_PROTOCOLS:
                _, _, mask = hybrid.get_protocol_splits(df, protocol_id)
                pred = edgebase.predict_direction(run, X_temporal, mask)
                drow = evaluate_direction(df, mask, pred, "LDX1 Wide-DSConv Direction Expert", protocol_id, run.params)
                metrics[protocol_id] = drow["direction_macro_f1"]
                fall_mask, fall_pred, _ = fall_predictions_for_protocol(df, fall_expert, protocol_id)
                e2e = evaluate_e2e(df, fall_mask, fall_pred, pred, "DS-Fall-RD Compact Hybrid Seed Search", protocol_id)
                e2e_vals[protocol_id] = e2e["E2E_Direction_Macro_F1"]
            score = 0.40 * metrics["E3_BITS_WEDA_MIXED"] + 0.30 * metrics["E6_E3_MIXED_TEST_BITS"] + 0.30 * metrics["E7_E3_MIXED_TEST_WEDA"]
            vals = np.array([metrics[p] for p in FOCUS_PROTOCOLS], dtype=float)
            val_test_gap = float(run.best_val_macro_f1 - metrics["E3_BITS_WEDA_MIXED"])
            rows.append(
                {
                    "seed": seed,
                    "learning_rate": lr,
                    "best_epoch": run.best_epoch,
                    "best_val_macro_f1": run.best_val_macro_f1,
                    "best_val_loss": run.best_val_loss,
                    "E3_direction_macro_f1": metrics["E3_BITS_WEDA_MIXED"],
                    "E6_direction_macro_f1": metrics["E6_E3_MIXED_TEST_BITS"],
                    "E7_direction_macro_f1": metrics["E7_E3_MIXED_TEST_WEDA"],
                    "score_best_observed": float(score),
                    "min_E3_E6_E7": float(np.nanmin(vals)),
                    "std_E3_E6_E7": float(np.nanstd(vals, ddof=1)),
                    "E3_E2E_direction_macro_f1": e2e_vals["E3_BITS_WEDA_MIXED"],
                    "E6_E2E_direction_macro_f1": e2e_vals["E6_E3_MIXED_TEST_BITS"],
                    "E7_E2E_direction_macro_f1": e2e_vals["E7_E3_MIXED_TEST_WEDA"],
                    "avg_E2E_direction_macro_f1": float(np.nanmean(list(e2e_vals.values()))),
                    "validation_test_gap_E3": val_test_gap,
                    "params": run.params,
                    "model_path": str(model_dir / f"{run_id}.pt"),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(report_dir / "ldx1_seed_search_results.csv", index=False)
    top = out.sort_values(
        ["score_best_observed", "min_E3_E6_E7", "std_E3_E6_E7", "avg_E2E_direction_macro_f1", "validation_test_gap_E3", "best_epoch"],
        ascending=[False, False, True, False, True, True],
    ).head(10)
    top.to_csv(report_dir / "ldx1_seed_search_top10.csv", index=False)
    plt.figure(figsize=(8, 4.8))
    sns.histplot(out["score_best_observed"], bins=20, kde=True)
    plt.xlabel("Best-observed weighted seed score")
    plt.tight_layout()
    plt.savefig(figure_dir / "ldx1_seed_score_distribution.png", dpi=150)
    plt.close()
    return out


def select_seed(seed_df: pd.DataFrame, selection_mode: str) -> dict[str, Any]:
    if seed_df.empty:
        raise ValueError("Seed search table is empty.")
    paper = seed_df.sort_values(
        ["best_val_macro_f1", "best_val_loss", "validation_test_gap_E3", "best_epoch"],
        ascending=[False, True, True, True],
    ).iloc[0]
    observed = seed_df.sort_values(
        ["score_best_observed", "min_E3_E6_E7", "std_E3_E6_E7", "avg_E2E_direction_macro_f1", "validation_test_gap_E3", "best_epoch"],
        ascending=[False, False, True, False, True, True],
    ).iloc[0]
    selected = paper if selection_mode == "paper_safe" else observed
    return {
        "selection_mode": selection_mode,
        "selected_seed": int(selected["seed"]),
        "selected_learning_rate": float(selected["learning_rate"]),
        "selected_reason": "highest validation direction macro F1, with val loss/gap/epoch tie-breakers"
        if selection_mode == "paper_safe"
        else "highest weighted E3/E6/E7 test score; analysis only, optimistic",
        "paper_safe_seed": int(paper["seed"]),
        "paper_safe_learning_rate": float(paper["learning_rate"]),
        "best_observed_seed": int(observed["seed"]),
        "best_observed_learning_rate": float(observed["learning_rate"]),
        "paper_safe_row": paper.to_dict(),
        "best_observed_row": observed.to_dict(),
    }


def build_final_model(
    repo_root: Path,
    df: pd.DataFrame,
    X_temporal: np.ndarray,
    fall_expert: dict[str, Any],
    selected: dict[str, Any],
    report_dir: Path,
    model_dir: Path,
    epochs: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    spec = [s for s in edgebase.model_specs() if s["model_name"] == "LDX1_D1_Wide_DSConv"][0]
    seed = int(selected["selected_seed"])
    lr = float(selected["selected_learning_rate"])
    print(f"[build-final] LDX1 seed={seed} lr={lr}")
    direction_run = edgebase.train_direction_model(df, X_temporal, "E3_BITS_WEDA_MIXED", spec, seed, lr, epochs, batch_size)
    fall_path = model_dir / "statistical_fall_expert.pkl"
    with fall_path.open("wb") as f:
        pickle.dump(fall_expert, f)
    direction_path = model_dir / "ldx1_wide_dsconv_direction.pt"
    torch.save(
        {
            "model_name": direction_run.model_name,
            "state_dict": direction_run.model.state_dict(),
            "scaler": direction_run.scaler,
            "seed": seed,
            "learning_rate": lr,
            "params": direction_run.params,
            "selection_mode": selected["selection_mode"],
        },
        direction_path,
    )
    fall_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    e2e_rows: list[dict[str, Any]] = []
    for protocol_id in FOCUS_PROTOCOLS:
        mask, fall_pred, fall_scores = fall_predictions_for_protocol(df, fall_expert, protocol_id)
        d_pred = edgebase.predict_direction(direction_run, X_temporal, mask)
        fall_rows.append(evaluate_fall(df, mask, fall_pred, fall_scores, "DS-Fall-RD Compact Hybrid", protocol_id, len(fall_expert["features"])))
        direction_rows.append(evaluate_direction(df, mask, d_pred, "DS-Fall-RD Compact Hybrid", protocol_id, direction_run.params))
        e2e_rows.append(evaluate_e2e(df, mask, fall_pred, d_pred, "DS-Fall-RD Compact Hybrid", protocol_id))
    fall_df = pd.DataFrame(fall_rows)
    direction_df = pd.DataFrame(direction_rows)
    e2e_df = pd.DataFrame(e2e_rows)
    config = {
        "model_name": "DS-Fall-RD Compact Hybrid",
        "selection_mode": selected["selection_mode"],
        "selected_seed": seed,
        "selected_learning_rate": lr,
        "fall_expert": {
            "name": "Statistical Fall Expert",
            "model": "GradientBoostingClassifier",
            "feature_type": "event-window statistical features",
            "feature_count": len(fall_expert["features"]),
            "threshold": fall_expert["threshold"],
            "threshold_note": fall_expert["threshold_note"],
            "artifact": str(fall_path),
        },
        "direction_expert": {
            "name": "LDX1 Wide-DSConv Direction Expert",
            "input": "temporal tilt12 sequence",
            "input_shape": [50, 12],
            "params": direction_run.params,
            "estimated_fp32_kb": direction_run.params * 4 / 1024,
            "estimated_int8_kb": direction_run.params / 1024,
            "artifact": str(direction_path),
        },
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "tensorflow_import_error": TENSORFLOW_IMPORT_ERROR,
        "tflite_status": "not_available_tensorflow_import_failed" if not TENSORFLOW_AVAILABLE else "not_exported_by_pytorch_script",
    }
    (model_dir / "final_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    fall_df.to_csv(report_dir / "final_compact_hybrid_fall_metrics.csv", index=False)
    direction_df.to_csv(report_dir / "final_compact_hybrid_direction_metrics.csv", index=False)
    e2e_df.to_csv(report_dir / "final_compact_hybrid_e2e_metrics.csv", index=False)
    return fall_df, direction_df, e2e_df, config


def write_model_card(report_dir: Path, config: dict[str, Any], fall_df: pd.DataFrame, direction_df: pd.DataFrame, e2e_df: pd.DataFrame) -> None:
    lines = [
        "# DS-Fall-RD Compact Hybrid Model Card",
        "",
        "## Model",
        "",
        "- Name: DS-Fall-RD Compact Hybrid",
        "- Fall: Statistical Fall Expert, `GradientBoostingClassifier`, event-window statistical features.",
        "- Direction: LDX1 Wide-DSConv Direction Expert, temporal `tilt12` input `50 x 12`.",
        "- FullTiming/timing-index artifacts: not used.",
        "",
        "## Selected Direction Seed",
        "",
        f"- Selection mode: `{config['selection_mode']}`.",
        f"- Seed: `{config['selected_seed']}`.",
        f"- Learning rate: `{config['selected_learning_rate']}`.",
        "",
        "## Final Metrics",
        "",
        "### Fall",
        md_table(fall_df[["experiment_id", "fall_precision", "fall_recall", "fall_f1", "fall_accuracy", "FP", "FN"]]),
        "",
        "### Direction",
        md_table(direction_df[["experiment_id", "direction_macro_f1", "direction_accuracy", "forward_f1", "backward_f1", "lateral_f1", "direction_n_supervised"]]),
        "",
        "### End-to-End Direction",
        md_table(e2e_df[["experiment_id", "E2E_Direction_Macro_F1", "coverage", "correct_rate", "direction_n_supervised"]]),
        "",
        "## Artifacts",
        "",
        f"- Fall expert: `{config['fall_expert']['artifact']}`",
        f"- Direction expert: `{config['direction_expert']['artifact']}`",
        f"- Config: `outputs/models/final_compact_hybrid/final_config.json`",
        "",
        "## Limitations",
        "",
        "- TensorFlow/TFLite export is unavailable in this environment if TensorFlow import fails.",
        "- If `best_observed` seed selection is used, it is optimistic and should be reported as analysis only.",
    ]
    (report_dir / "final_model_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def write_readme(repo_root: Path) -> None:
    text = """# DS-Fall-RD Compact Hybrid

Final paper-ready pipeline for BITS/WEDA 25 Hz event-centered 2-second wrist IMU windows.

## Inputs

- Fall expert: event-window statistical feature vector from the 2-second IMU window.
- Direction expert: temporal `tilt12` sequence with shape `50 x 12`.

The fall feature vector excludes explicit timing-index artifacts such as `impact_index`, `peak_index`, and distance-to-center variables. Peak magnitude values are allowed because they are signal values.

## Model

- Statistical Fall Expert: `GradientBoostingClassifier`.
- LDX1 Wide-DSConv Direction Expert: lightweight temporal depthwise-separable CNN.
- Inference: if the fall expert predicts non-fall, output non-fall; otherwise output fall with direction predicted by LDX1.

## Reproduce

```bash
python scripts/run_final_compact_hybrid.py --repo-root . --seed-search --build-final --selection-mode paper_safe
python scripts/compare_final_with_baselines.py --repo-root . --run-all
python scripts/evaluate_final_compact_hybrid.py --repo-root .
```

Outputs are written to `outputs/reports/final_compact_hybrid`, `outputs/models/final_compact_hybrid`, and `outputs/figures/final_compact_hybrid`.
"""
    (repo_root / "README_FINAL_MODEL.md").write_text(text, encoding="utf-8")


def write_environment(report_dir: Path, repo_root: Path) -> None:
    text = "\n".join(
        [
            f"python={platform.python_version()}",
            f"platform={platform.platform()}",
            f"torch_available={TORCH_AVAILABLE}",
            f"torch_version={getattr(torch, '__version__', 'not_available') if TORCH_AVAILABLE else 'not_available'}",
            f"tensorflow_available={TENSORFLOW_AVAILABLE}",
            f"tensorflow_import_error={TENSORFLOW_IMPORT_ERROR}",
            f"repo_root={repo_root}",
        ]
    )
    (report_dir / "environment_info.txt").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not (args.seed_search or args.build_final):
        args.seed_search = True
        args.build_final = True
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for final LDX1 training.")
    repo_root = Path(args.repo_root).resolve()
    report_dir = ensure_dir(repo_root / "outputs" / "reports" / "final_compact_hybrid")
    figure_dir = ensure_dir(repo_root / "outputs" / "figures" / "final_compact_hybrid")
    model_dir = ensure_dir(repo_root / "outputs" / "models" / "final_compact_hybrid")
    write_environment(report_dir, repo_root)
    write_readme(repo_root)
    seeds = args.seeds[:3] if args.quick else args.seeds
    learning_rates = args.learning_rates[:1] if args.quick else args.learning_rates
    print("[1/5] Load data")
    df, X_temporal = load_data(repo_root)
    features = statistical_fall_features(df)
    print(f"[2/5] Train Statistical Fall Expert with {len(features)} features")
    fall_expert = train_statistical_fall_expert(df, features, seed=42)
    failed = pd.DataFrame(columns=["stage", "seed", "learning_rate", "error"])
    seed_df = pd.DataFrame()
    if args.seed_search:
        print("[3/5] LDX1 seed search")
        seed_df = run_ldx1_seed_search(repo_root, df, X_temporal, fall_expert, report_dir, figure_dir, model_dir, seeds, learning_rates, args.epochs, args.batch_size)
    else:
        seed_path = report_dir / "ldx1_seed_search_results.csv"
        if not seed_path.exists():
            raise FileNotFoundError("Seed search results not found. Run with --seed-search first.")
        seed_df = pd.read_csv(seed_path)
    selected = select_seed(seed_df, args.selection_mode)
    (report_dir / "ldx1_selected_seed.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    if args.build_final:
        print("[4/5] Build final compact hybrid")
        fall_df, direction_df, e2e_df, config = build_final_model(repo_root, df, X_temporal, fall_expert, selected, report_dir, model_dir, args.epochs, args.batch_size)
        write_model_card(report_dir, config, fall_df, direction_df, e2e_df)
    failed.to_csv(report_dir / "failed_runs.csv", index=False)
    print("[5/5] Done")
    print(json.dumps({"selected_seed": selected["selected_seed"], "selected_learning_rate": selected["selected_learning_rate"], "selection_mode": args.selection_mode}, indent=2))


if __name__ == "__main__":
    main()
