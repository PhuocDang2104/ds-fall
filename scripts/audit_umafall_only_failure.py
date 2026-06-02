from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import compute_imu_features  # noqa: E402
from src.experiments.ablation_rd import load_processed_training_data  # noqa: E402
from src.utils.io import ensure_dir  # noqa: E402


DIRECTION_NAMES = {0: "forward", 1: "backward", 2: "lateral", -1: "unsupervised"}
FALL_ACTIVITY_KEYWORDS = ("fall", "falling")
EXPECTED_FALL_DIRECTIONS = ("backwardfall", "forwardfall", "lateralfall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit UMAFall-only DS-Fall failure.")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--run-dir", type=str, default="outputs/runs/DOMAIN_DS_UMAFALL_A5WCEFW")
    parser.add_argument("--processed-x-is-raw", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    reports_dir = ensure_dir(output_dir / "reports")

    data = load_processed_training_data(args.processed_dir, processed_x_is_normalized=not args.processed_x_is_raw)
    X_raw6 = data["X_raw6"]
    metadata = data["metadata"].copy()
    y_fall = data["y_fall"].astype(np.int64)
    y_direction = data["y_direction"].astype(np.int64)
    direction_mask = data["direction_mask"].astype(np.float32)

    if "dataset" not in metadata:
        raise ValueError("metadata.csv must contain a dataset column")
    uma_mask = metadata["dataset"].astype(str).str.lower().to_numpy() == "umafall"
    if not uma_mask.any():
        raise ValueError("No rows with dataset == 'umafall' found in processed data")

    X_uma = X_raw6[uma_mask]
    meta_uma = metadata.loc[uma_mask].reset_index(drop=True)
    yf_uma = y_fall[uma_mask]
    yd_uma = y_direction[uma_mask]
    dm_uma = direction_mask[uma_mask]

    split_counts = build_split_counts(meta_uma, yf_uma, yd_uma, dm_uma)
    label_mapping = build_label_mapping(meta_uma, yf_uma, yd_uma)
    split_counts.to_csv(reports_dir / "umafall_only_split_counts.csv", index=False)
    label_mapping.to_csv(reports_dir / "umafall_only_label_mapping.csv", index=False)

    warnings = label_mapping_warnings(label_mapping)
    trainability = run_fall_trainability_baselines(X_uma, meta_uma, yf_uma, seed=args.seed)
    training_log = parse_training_log(Path(args.run_dir))
    class_weights = load_json_if_exists(Path(args.run_dir) / "class_weights.json")
    ds_metrics = load_json_if_exists(Path(args.run_dir) / "metrics.json")
    ds_fall_eval = extract_fall_confusion(ds_metrics)

    report = render_report(
        meta_uma=meta_uma,
        yf_uma=yf_uma,
        yd_uma=yd_uma,
        dm_uma=dm_uma,
        split_counts=split_counts,
        label_mapping=label_mapping,
        warnings=warnings,
        trainability=trainability,
        training_log=training_log,
        class_weights=class_weights,
        ds_fall_eval=ds_fall_eval,
        run_dir=Path(args.run_dir),
    )
    (reports_dir / "umafall_only_audit.md").write_text(report, encoding="utf-8")
    print(report)


def build_split_counts(
    metadata: pd.DataFrame,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
) -> pd.DataFrame:
    meta = metadata.copy()
    meta["_y_fall"] = y_fall
    meta["_direction_name"] = [
        DIRECTION_NAMES.get(int(label), str(label)) if bool(mask) and int(label) >= 0 else "unsupervised"
        for label, mask in zip(y_direction, direction_mask)
    ]
    if "split" not in meta:
        meta["split"] = "all"

    rows: list[dict[str, Any]] = []
    split_order = sorted(meta["split"].astype(str).unique())
    for split in split_order:
        group = meta[meta["split"].astype(str) == split]
        rows.append({"section": "total_windows", "split": split, "label": "all", "count": int(len(group))})
        rows.append({"section": "fall_label", "split": split, "label": "non_fall", "count": int((group["_y_fall"] == 0).sum())})
        rows.append({"section": "fall_label", "split": split, "label": "fall", "count": int((group["_y_fall"] == 1).sum())})
        for label, count in group["_direction_name"].value_counts(dropna=False).sort_index().items():
            rows.append({"section": "direction_label", "split": split, "label": str(label), "count": int(count)})
        for col in ["subject_id", "trial_id", "activity_name", "activity_id"]:
            if col in group:
                rows.append({"section": "unique_count", "split": split, "label": col, "count": int(group[col].nunique(dropna=True))})
        activity_col = activity_column(group)
        if activity_col is not None:
            for label, count in group[activity_col].fillna("missing").astype(str).value_counts().sort_index().items():
                rows.append({"section": "activity_count", "split": split, "label": label, "count": int(count)})
    return pd.DataFrame(rows)


def build_label_mapping(metadata: pd.DataFrame, y_fall: np.ndarray, y_direction: np.ndarray) -> pd.DataFrame:
    meta = metadata.copy()
    meta["_y_fall"] = y_fall
    meta["_direction_name"] = [DIRECTION_NAMES.get(int(label), str(label)) for label in y_direction]
    activity_col = activity_column(meta)
    if activity_col is None:
        activity_col = "_activity_missing"
        meta[activity_col] = "not_available"

    group_cols = [activity_col]
    if "activity_id" in meta and activity_col != "activity_id":
        group_cols.append("activity_id")

    rows: list[dict[str, Any]] = []
    for key, group in meta.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {
            "activity_name": str(key[0]),
            "activity_id": str(key[1]) if len(key) > 1 else "",
            "n_windows": int(len(group)),
            "fall_0_count": int((group["_y_fall"] == 0).sum()),
            "fall_1_count": int((group["_y_fall"] == 1).sum()),
        }
        for direction in ["forward", "backward", "lateral", "unsupervised"]:
            row[f"direction_{direction}_count"] = int((group["_direction_name"] == direction).sum())
        for split, count in group["split"].fillna("missing").astype(str).value_counts().items() if "split" in group else []:
            row[f"split_{split}_count"] = int(count)
        rows.append(row)
    return pd.DataFrame(rows)


def label_mapping_warnings(label_mapping: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if label_mapping.empty:
        return ["No UMAFall activity/fall_type mapping rows were found."]

    label_mapping = label_mapping.copy()
    label_mapping["_norm_activity"] = label_mapping["activity_name"].astype(str).map(normalize_text)
    for expected in EXPECTED_FALL_DIRECTIONS:
        match = label_mapping[label_mapping["_norm_activity"].str.contains(expected, na=False)]
        if match.empty:
            warnings.append(f"Expected activity containing {expected!r} was not found.")
        elif int(match["fall_1_count"].sum()) == 0:
            warnings.append(f"Activity {expected!r} exists but has no fall_label=1 windows.")

    adl_rows = label_mapping[~label_mapping["_norm_activity"].str.contains("fall", na=False)]
    bad_adl = adl_rows[adl_rows["fall_1_count"] > 0]
    for _, row in bad_adl.iterrows():
        warnings.append(
            f"Potential ADL activity {row['activity_name']!r} has fall_label=1 count {int(row['fall_1_count'])}."
        )
    return warnings


def run_fall_trainability_baselines(
    X_raw6: np.ndarray,
    metadata: pd.DataFrame,
    y_fall: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    if "split" not in metadata:
        return {"warning": "metadata has no split column"}
    split = metadata["split"].astype(str).str.lower().to_numpy()
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    if not train_mask.any() or not test_mask.any():
        return {"warning": "train/test split is not available"}

    X_summary, feature_names = fall_summary_features(X_raw6)
    out: dict[str, Any] = {"feature_names": feature_names}
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }
    for name, model in models.items():
        model.fit(X_summary[train_mask], y_fall[train_mask])
        out[name] = {}
        for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
            if not mask.any():
                continue
            y_pred = model.predict(X_summary[mask])
            cm = confusion_matrix(y_fall[mask], y_pred, labels=[0, 1])
            out[name][split_name] = {
                "accuracy": float(accuracy_score(y_fall[mask], y_pred)),
                "precision": float(precision_score(y_fall[mask], y_pred, zero_division=0)),
                "recall": float(recall_score(y_fall[mask], y_pred, zero_division=0)),
                "fall_f1": float(f1_score(y_fall[mask], y_pred, average="binary", zero_division=0)),
                "confusion_matrix": cm.tolist(),
            }
    return out


def fall_summary_features(X_raw6: np.ndarray, fs: float = 50.0) -> tuple[np.ndarray, list[str]]:
    features = compute_imu_features(X_raw6, feature_set="tilt12", fs=fs)
    acc_mag = features[:, :, 6]
    gyro_mag = features[:, :, 7]
    jerk = features[:, :, 8]
    post = slice(max(0, X_raw6.shape[1] - 20), X_raw6.shape[1])
    out = np.column_stack(
        [
            acc_mag.max(axis=1),
            gyro_mag.max(axis=1),
            np.abs(jerk).max(axis=1),
            acc_mag[:, post].std(axis=1),
            gyro_mag[:, post].std(axis=1),
        ]
    ).astype(np.float32)
    names = ["max_acc_mag", "max_gyro_mag", "max_jerk", "post_acc_std", "post_gyro_std"]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), names


def parse_training_log(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "training_log.csv"
    if not path.exists():
        return {"available": False, "path": str(path)}
    df = pd.read_csv(path)
    out: dict[str, Any] = {"available": True, "path": str(path), "epochs_logged": int(len(df))}
    if df.empty:
        return out

    metric_cols = [
        "epoch",
        "loss",
        "fall_output_accuracy",
        "fall_output_loss",
        "direction_output_accuracy",
        "direction_output_loss",
        "val_loss",
        "val_fall_output_accuracy",
        "val_fall_output_loss",
        "val_direction_output_accuracy",
        "val_direction_output_loss",
        "val_fall_f1",
        "val_direction_macro_f1",
        "val_domain_score",
        "learning_rate",
    ]

    def row_at(idx: int) -> dict[str, Any]:
        row = df.iloc[int(idx)]
        return {col: scalar(row[col]) for col in metric_cols if col in df.columns}

    out["last_epoch"] = row_at(len(df) - 1)
    for col in ["val_fall_f1", "val_direction_macro_f1", "val_domain_score", "val_loss"]:
        if col in df.columns and df[col].notna().any():
            idx = df[col].idxmin() if col == "val_loss" else df[col].idxmax()
            out[f"best_by_{col}"] = row_at(int(idx))
    return out


def extract_fall_confusion(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {"available": False}
    fall = metrics.get("fall", {})
    cm = np.asarray(fall.get("confusion_matrix", []), dtype=np.int64)
    if cm.shape != (2, 2):
        return {"available": False, "reason": "fall confusion matrix not found"}
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    predicted_fall = tp + fp
    total = int(cm.sum())
    return {
        "available": True,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "predicted_fall": predicted_fall,
        "total": total,
        "predicted_fall_pct": predicted_fall / total if total else 0.0,
        "accuracy": fall.get("accuracy"),
        "precision": fall.get("precision"),
        "recall": fall.get("recall"),
        "f1": fall.get("f1"),
    }


def render_report(
    meta_uma: pd.DataFrame,
    yf_uma: np.ndarray,
    yd_uma: np.ndarray,
    dm_uma: np.ndarray,
    split_counts: pd.DataFrame,
    label_mapping: pd.DataFrame,
    warnings: list[str],
    trainability: dict[str, Any],
    training_log: dict[str, Any],
    class_weights: dict[str, Any] | None,
    ds_fall_eval: dict[str, Any],
    run_dir: Path,
) -> str:
    lines: list[str] = [
        "# UMAFall-Only Failure Audit",
        "",
        "Scope: processed UMAFall windows after unit harmonization. This audit does not modify preprocessing or model code.",
        "",
        "## Basic Counts",
        "",
        f"- Total UMAFall windows: {len(meta_uma)}",
        f"- Fall windows: {int((yf_uma == 1).sum())}",
        f"- Non-fall windows: {int((yf_uma == 0).sum())}",
        f"- Supervised direction windows: {int(((dm_uma > 0) & (yd_uma >= 0)).sum())}",
        "",
        "### Split / Label Counts",
        "",
        dataframe_to_markdown(split_counts.head(80)),
        "",
        "## Label Mapping",
        "",
        dataframe_to_markdown(label_mapping),
        "",
        "### Label Mapping Warnings",
        "",
    ]
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- No obvious fall/ADL label mapping warnings found.")

    lines.extend(["", "## Trainability Baseline", ""])
    lines.extend(render_trainability(trainability))

    lines.extend(["", "## DS-Fall UMAFall-Only Training Log", "", f"- Run dir: `{run_dir}`"])
    if not training_log.get("available"):
        lines.append("- Training log not available.")
    else:
        lines.append(f"- Epochs logged: {training_log.get('epochs_logged')}")
        for key in ["best_by_val_domain_score", "best_by_val_direction_macro_f1", "best_by_val_fall_f1", "best_by_val_loss", "last_epoch"]:
            if key in training_log:
                lines.append(f"- {key}: `{compact_dict(training_log[key])}`")
    if class_weights:
        lines.extend(["", "### Class Weights", "", f"```json\n{json.dumps(class_weights, indent=2)}\n```"])

    lines.extend(["", "## DS-Fall UMAFall-Only Fall Confusion", ""])
    if ds_fall_eval.get("available"):
        lines.extend(
            [
                f"- TN: {ds_fall_eval['tn']}",
                f"- FP: {ds_fall_eval['fp']}",
                f"- FN: {ds_fall_eval['fn']}",
                f"- TP: {ds_fall_eval['tp']}",
                f"- Predicted fall windows: {ds_fall_eval['predicted_fall']} / {ds_fall_eval['total']} ({ds_fall_eval['predicted_fall_pct']:.4f})",
                f"- Precision: {fmt(ds_fall_eval.get('precision'))}",
                f"- Recall: {fmt(ds_fall_eval.get('recall'))}",
                f"- F1: {fmt(ds_fall_eval.get('f1'))}",
            ]
        )
    else:
        lines.append("- DS-Fall fall confusion matrix not available.")

    lines.extend(["", "## Diagnosis", ""])
    lines.extend(diagnosis_lines(meta_uma, yf_uma, warnings, trainability, training_log, ds_fall_eval))
    lines.append("")
    return "\n".join(lines)


def render_trainability(trainability: dict[str, Any]) -> list[str]:
    if trainability.get("warning"):
        return [f"- {trainability['warning']}"]
    lines = [f"- Summary features: {', '.join(trainability.get('feature_names', []))}", ""]
    rows = []
    for model_name in ["logistic_regression", "random_forest"]:
        for split_name, result in trainability.get(model_name, {}).items():
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "accuracy": result["accuracy"],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "fall_f1": result["fall_f1"],
                    "confusion_matrix": result["confusion_matrix"],
                }
            )
    lines.append(dataframe_to_markdown(pd.DataFrame(rows)))
    return lines


def diagnosis_lines(
    meta_uma: pd.DataFrame,
    yf_uma: np.ndarray,
    warnings: list[str],
    trainability: dict[str, Any],
    training_log: dict[str, Any],
    ds_fall_eval: dict[str, Any],
) -> list[str]:
    lines: list[str] = []
    split = meta_uma["split"].astype(str).str.lower() if "split" in meta_uma else pd.Series(["all"] * len(meta_uma))
    fall_by_split = {
        name: int(((split == name).to_numpy() & (yf_uma == 1)).sum())
        for name in ["train", "val", "test"]
    }
    enough = all(count >= 20 for count in fall_by_split.values())
    lines.append(
        f"- UMAFall-only split has fall samples train/val/test = "
        f"{fall_by_split.get('train', 0)}/{fall_by_split.get('val', 0)}/{fall_by_split.get('test', 0)}; "
        f"{'enough for a coarse audit' if enough else 'validation/test fall sample count is small'}."
    )
    if warnings:
        lines.append("- Label mapping has warnings; inspect `umafall_only_label_mapping.csv` before treating results as final.")
    else:
        lines.append("- Fall label mapping looks internally consistent from activity names: fall activities are fall_label=1 and ADL are non-fall.")

    predicted_fall_pct = ds_fall_eval.get("predicted_fall_pct")
    if predicted_fall_pct is not None:
        if predicted_fall_pct < 0.05:
            lines.append("- DS-Fall UMAFall-only is effectively predicting almost all windows as non-fall.")
        else:
            lines.append("- DS-Fall UMAFall-only is not a pure all-non-fall predictor.")

    rf_test_f1 = trainability.get("random_forest", {}).get("test", {}).get("fall_f1")
    ds_f1 = ds_fall_eval.get("f1")
    if rf_test_f1 is not None and ds_f1 is not None:
        if rf_test_f1 >= 0.80 and ds_f1 < 0.30:
            lines.append(
                "- RF fall baseline is high while DS-Fall is low: likely training config/checkpointing/model calibration issue, not split/window/label difficulty."
            )
        elif rf_test_f1 < 0.50:
            lines.append("- RF fall baseline is also low: likely split/window/label separability issue.")
        else:
            lines.append("- RF baseline is moderate; failure may involve both model training behavior and UMAFall split/window difficulty.")

    if training_log.get("available"):
        best_fall = training_log.get("best_by_val_fall_f1", {}).get("val_fall_f1")
        best_dir = training_log.get("best_by_val_direction_macro_f1", {}).get("val_direction_macro_f1")
        if best_fall == 0 or best_fall is None:
            lines.append("- Training log never reaches nonzero validation fall F1; fall head is not learning UMAFall-only validation under this setup.")
        lines.append(f"- Best logged val_direction_macro_f1: {fmt(best_dir)}; best logged val_fall_f1: {fmt(best_fall)}.")

    lines.append(
        "- Most likely cause from this audit: not unit harmonization; prioritize training/checkpoint behavior and UMAFall-only fall separability, then manually verify UMAFall activity/window labels."
    )
    return lines


def activity_column(df: pd.DataFrame) -> str | None:
    for col in ["activity_name", "fall_type", "activity_id"]:
        if col in df:
            return col
    return None


def normalize_text(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    compact = df.copy()
    for col in compact.select_dtypes(include=[float]).columns:
        compact[col] = compact[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    header = "| " + " | ".join(compact.columns.astype(str)) + " |"
    sep = "| " + " | ".join(["---"] * len(compact.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in compact.to_numpy()]
    return "\n".join([header, sep] + rows)


def compact_dict(values: dict[str, Any]) -> str:
    keep = []
    for key, value in values.items():
        if isinstance(value, float):
            keep.append(f"{key}={value:.4f}")
        else:
            keep.append(f"{key}={value}")
    return ", ".join(keep)


def scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


if __name__ == "__main__":
    main()
