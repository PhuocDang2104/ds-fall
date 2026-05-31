from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.features import compute_imu_features, feature_channel_names, handcrafted_direction_features
from src.training.evaluate import plot_confusion_matrix
from src.utils.io import ensure_dir, save_json


DIRECTION_NAMES = {0: "forward", 1: "backward", 2: "lateral"}


def generate_data_quality_report(
    X_raw6: np.ndarray,
    X_features: np.ndarray,
    metadata: pd.DataFrame,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    output_dir: str | Path,
    feature_set: str = "tilt12",
    fs: float = 50.0,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    reports_dir = ensure_dir(output_dir / "reports")
    figures_dir = ensure_dir(output_dir / "figures" / "data_quality")

    raw_names = feature_channel_names("raw6")
    feature_names = feature_channel_names(feature_set)
    X_raw6 = np.asarray(X_raw6, dtype=np.float32)
    X_features = np.asarray(X_features, dtype=np.float32)

    raw_stats = channel_stats_dataframe(X_raw6, raw_names)
    feature_stats = channel_stats_dataframe(X_features, feature_names)
    raw_stats.to_csv(reports_dir / "data_quality_channel_stats_raw6.csv", index=False)
    feature_stats.to_csv(reports_dir / "data_quality_channel_stats_features.csv", index=False)

    distribution_paths = _save_distribution_tables(metadata, reports_dir)

    acc_mag = np.sqrt(np.sum(np.square(X_raw6[:, :, :3]), axis=2))
    gyro_mag = np.sqrt(np.sum(np.square(X_raw6[:, :, 3:6]), axis=2))
    acc_peak = acc_mag.max(axis=1)
    gyro_peak = gyro_mag.max(axis=1)
    peak_df = pd.DataFrame({"acc_mag_peak": acc_peak, "gyro_mag_peak": gyro_peak})
    peak_df.to_csv(reports_dir / "data_quality_peak_distributions.csv", index=False)
    _plot_histogram(peak_df["acc_mag_peak"], "acc_mag peak distribution", figures_dir / "acc_mag_peak_distribution.png")
    _plot_histogram(peak_df["gyro_mag_peak"], "gyro_mag peak distribution", figures_dir / "gyro_mag_peak_distribution.png")

    fall_mask = np.asarray(y_fall).astype(int) == 1
    impact_indices = np.argmax(acc_mag[fall_mask], axis=1) if fall_mask.any() else np.array([], dtype=int)
    impact_df = pd.DataFrame({"impact_index": impact_indices.astype(int)})
    impact_df.to_csv(reports_dir / "data_quality_impact_indices.csv", index=False)
    if len(impact_indices):
        _plot_histogram(impact_df["impact_index"], "fall impact index distribution", figures_dir / "impact_index_distribution.png")

    impact_summary = _impact_boundary_summary(impact_indices)
    warnings = _build_quality_warnings(metadata, y_direction, direction_mask, impact_summary)

    report = {
        "feature_set": feature_set,
        "num_windows": int(len(X_raw6)),
        "nan_inf_before_feature_engineering": _nan_inf_counts(X_raw6),
        "nan_inf_after_feature_engineering": _nan_inf_counts(X_features),
        "supervised_direction_samples": int(np.asarray(direction_mask).astype(bool).sum()),
        "missing_direction_labels": int((np.asarray(y_direction) < 0).sum()),
        "impact_boundary_summary": impact_summary,
        "warnings": warnings,
        "distribution_paths": {name: str(path) for name, path in distribution_paths.items()},
    }
    save_json(reports_dir / "data_quality_report.json", report)
    _write_markdown_report(
        reports_dir / "data_quality_report.md",
        report,
        metadata,
        raw_stats,
        feature_stats,
        peak_df,
    )
    return report


def channel_stats_dataframe(X: np.ndarray, channel_names: list[str]) -> pd.DataFrame:
    rows = []
    for idx, name in enumerate(channel_names):
        values = X[:, :, idx]
        rows.append(
            {
                "channel": name,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        )
    return pd.DataFrame(rows)


def plot_direction_signal_diagnostics(
    X_raw6: np.ndarray,
    metadata: pd.DataFrame,
    direction_mask: np.ndarray,
    output_dir: str | Path,
    fs: float = 50.0,
) -> None:
    output_dir = Path(output_dir)
    fig_dir = ensure_dir(output_dir / "figures" / "direction_signal_diagnostics")
    labels = _direction_label_series(metadata, direction_mask)

    _plot_mean_std_by_direction(
        X_raw6,
        feature_channel_names("raw6"),
        labels,
        ["ax", "ay", "az", "gx", "gy", "gz"],
        fig_dir / "raw6_mean_std_by_direction.png",
        fs=fs,
    )

    tilt_features = compute_imu_features(X_raw6, feature_set="tilt12", fs=fs)
    _plot_mean_std_by_direction(
        tilt_features,
        feature_channel_names("tilt12"),
        labels,
        ["roll", "pitch"],
        fig_dir / "roll_pitch_mean_std_by_direction.png",
        fs=fs,
    )


def run_handcrafted_direction_baseline(
    X_raw6: np.ndarray,
    metadata: pd.DataFrame,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    output_dir: str | Path,
    seed: int = 42,
    fs: float = 50.0,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    reports_dir = ensure_dir(output_dir / "reports")
    figures_dir = ensure_dir(output_dir / "figures" / "direction_signal_diagnostics")

    X_summary, summary_names = handcrafted_direction_features(X_raw6, fs=fs)
    summary_df = pd.DataFrame(X_summary, columns=summary_names)
    summary_df.to_csv(reports_dir / "direction_handcrafted_summary_features.csv", index=False)

    supervised = np.asarray(direction_mask).astype(bool) & (np.asarray(y_direction) >= 0)
    if "split" not in metadata:
        return {"warning": "metadata has no split column; handcrafted baseline skipped"}

    train_mask = supervised & metadata["split"].eq("train").to_numpy()
    eval_split = "test" if metadata["split"].eq("test").any() else "val"
    eval_mask = supervised & metadata["split"].eq(eval_split).to_numpy()

    if train_mask.sum() < 3 or eval_mask.sum() < 1:
        return {"warning": "not enough supervised direction samples for handcrafted baseline"}
    if len(np.unique(y_direction[train_mask])) < 2:
        return {"warning": "handcrafted baseline needs at least two direction classes in train split"}

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    clf.fit(X_summary[train_mask], y_direction[train_mask])
    pred = clf.predict(X_summary[eval_mask])
    y_true = y_direction[eval_mask]
    result = {
        "eval_split": eval_split,
        "num_train": int(train_mask.sum()),
        "num_eval": int(eval_mask.sum()),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            pred,
            labels=[0, 1, 2],
            target_names=["forward", "backward", "lateral"],
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1, 2]).tolist(),
        "warning": None,
    }
    if result["macro_f1"] < 0.4:
        result["warning"] = (
            "Handcrafted direction baseline macro F1 is very low; direction labels may be noisy, "
            "insufficient, or not physically separable from wrist IMU."
        )

    pd.DataFrame([_flatten_baseline_result(result)]).to_csv(
        reports_dir / "direction_handcrafted_baseline.csv",
        index=False,
    )
    plot_confusion_matrix(
        result["confusion_matrix"],
        ["forward", "backward", "lateral"],
        "Handcrafted direction baseline confusion matrix",
        figures_dir / "handcrafted_baseline_confusion_matrix.png",
        show=False,
    )
    save_json(reports_dir / "direction_handcrafted_baseline.json", result)
    return result


def _save_distribution_tables(metadata: pd.DataFrame, reports_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if metadata.empty:
        return paths
    specs = {
        "windows_by_dataset": ["dataset"],
        "windows_by_subject": ["dataset", "subject_id"],
        "fall_distribution": ["fall_label"],
        "direction_distribution": ["direction_label"],
        "direction_by_dataset": ["dataset", "direction_label"],
        "direction_by_subject": ["dataset", "subject_id", "direction_label"],
    }
    for name, cols in specs.items():
        available = [col for col in cols if col in metadata]
        if len(available) != len(cols):
            continue
        table = metadata.groupby(cols).size().reset_index(name="count").sort_values(cols)
        path = reports_dir / f"data_quality_{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return paths


def _build_quality_warnings(
    metadata: pd.DataFrame,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    impact_summary: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    supervised = np.asarray(direction_mask).astype(bool) & (np.asarray(y_direction) >= 0)
    supervised_labels = np.asarray(y_direction)[supervised]
    counts = pd.Series(supervised_labels).map(DIRECTION_NAMES).value_counts()
    for label in ["forward", "backward", "lateral"]:
        count = int(counts.get(label, 0))
        if count < 20:
            warnings.append(f"Direction class {label!r} has fewer than 20 supervised samples: {count}")

    if len(counts) > 1 and counts.min() > 0 and counts.max() / counts.min() >= 3.0:
        warnings.append(f"Direction labels are highly imbalanced: {counts.to_dict()}")

    if "dataset" in metadata:
        supervised_meta = metadata.loc[supervised]
        dataset_counts = supervised_meta.groupby("dataset").size()
        for dataset, count in dataset_counts.items():
            if int(count) < 10:
                warnings.append(f"Dataset {dataset!r} has fewer than 10 supervised direction samples: {int(count)}")
        for dataset in sorted(set(metadata["dataset"]) - set(dataset_counts.index)):
            warnings.append(f"Dataset {dataset!r} has no supervised direction samples")

    early_pct = impact_summary.get("early_pct", 0.0)
    late_pct = impact_summary.get("late_pct", 0.0)
    if early_pct > 20.0:
        warnings.append(f"{early_pct:.1f}% of fall windows have impact index < 20")
    if late_pct > 20.0:
        warnings.append(f"{late_pct:.1f}% of fall windows have impact index > 80")

    warnings.extend(_split_leakage_warnings(metadata))
    return warnings


def _split_leakage_warnings(metadata: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if "split" not in metadata:
        return ["metadata has no split column; cannot verify leakage"]
    if {"dataset", "subject_id"}.issubset(metadata.columns):
        subject_splits = metadata.groupby(["dataset", "subject_id"])["split"].nunique()
        leaked = subject_splits[subject_splits > 1]
        if not leaked.empty:
            warnings.append(f"Subject-level leakage detected for {len(leaked)} dataset/subject pairs")
    else:
        warnings.append("metadata lacks subject_id; split may be window-random")

    if {"dataset", "source_path"}.issubset(metadata.columns):
        trial_splits = metadata.groupby(["dataset", "source_path"])["split"].nunique()
        leaked_trials = trial_splits[trial_splits > 1]
        if not leaked_trials.empty:
            warnings.append(f"Trial/source leakage detected for {len(leaked_trials)} dataset/source pairs")
    else:
        warnings.append("metadata lacks source_path; cannot verify trial-level leakage")

    window_cols = [c for c in ["dataset", "source_path", "window_start_idx", "window_end_idx"] if c in metadata]
    if len(window_cols) == 4:
        dup_splits = metadata.groupby(window_cols)["split"].nunique()
        leaked_windows = dup_splits[dup_splits > 1]
        if not leaked_windows.empty:
            warnings.append(f"Exact window leakage detected for {len(leaked_windows)} windows")
    return warnings


def _impact_boundary_summary(impact_indices: np.ndarray) -> dict[str, Any]:
    n = int(len(impact_indices))
    if n == 0:
        return {"num_fall_windows": 0, "early": 0, "valid": 0, "late": 0, "early_pct": 0.0, "valid_pct": 0.0, "late_pct": 0.0}
    early = int((impact_indices < 20).sum())
    valid = int(((impact_indices >= 20) & (impact_indices <= 80)).sum())
    late = int((impact_indices > 80).sum())
    return {
        "num_fall_windows": n,
        "early": early,
        "valid": valid,
        "late": late,
        "early_pct": 100.0 * early / n,
        "valid_pct": 100.0 * valid / n,
        "late_pct": 100.0 * late / n,
    }


def _nan_inf_counts(X: np.ndarray) -> dict[str, int]:
    return {"nan": int(np.isnan(X).sum()), "inf": int(np.isinf(X).sum())}


def _plot_histogram(values: pd.Series, title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(values, bins=30, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _direction_label_series(metadata: pd.DataFrame, direction_mask: np.ndarray) -> pd.Series:
    labels = metadata.get("direction_label", pd.Series(["none"] * len(metadata))).copy()
    labels = labels.where(np.asarray(direction_mask).astype(bool), "none")
    return labels


def _plot_mean_std_by_direction(
    X: np.ndarray,
    channel_names: list[str],
    labels: pd.Series,
    channels: list[str],
    save_path: Path,
    fs: float,
) -> None:
    idx = {name: i for i, name in enumerate(channel_names)}
    t = np.arange(X.shape[1]) / float(fs)
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 2.6 * len(channels)), sharex=True)
    if len(channels) == 1:
        axes = [axes]
    for ax, channel in zip(axes, channels):
        if channel not in idx:
            continue
        for direction in ["forward", "backward", "lateral"]:
            mask = labels.eq(direction).to_numpy()
            if not mask.any():
                continue
            series = X[mask, :, idx[channel]]
            mean = series.mean(axis=0)
            std = series.std(axis=0)
            ax.plot(t, mean, label=direction)
            ax.fill_between(t, mean - std, mean + std, alpha=0.15)
        ax.set_title(channel)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_markdown_report(
    path: Path,
    report: dict[str, Any],
    metadata: pd.DataFrame,
    raw_stats: pd.DataFrame,
    feature_stats: pd.DataFrame,
    peak_df: pd.DataFrame,
) -> None:
    lines = [
        "# DS-Fall-RD Data Quality Report",
        "",
        f"- Feature set: `{report['feature_set']}`",
        f"- Windows: {report['num_windows']}",
        f"- Supervised direction samples: {report['supervised_direction_samples']}",
        f"- Missing direction labels: {report['missing_direction_labels']}",
        f"- NaN/Inf before feature engineering: {report['nan_inf_before_feature_engineering']}",
        f"- NaN/Inf after feature engineering: {report['nan_inf_after_feature_engineering']}",
        "",
        "## Impact Index",
        "",
        "```text",
        str(report["impact_boundary_summary"]),
        "```",
        "",
        "## Distributions",
        "",
        "```text",
    ]
    if not metadata.empty:
        if "dataset" in metadata:
            lines.append("windows_by_dataset:")
            lines.append(metadata["dataset"].value_counts().to_string())
        if "fall_label" in metadata:
            lines.append("")
            lines.append("fall_distribution:")
            lines.append(metadata["fall_label"].value_counts().to_string())
        if "direction_label" in metadata:
            lines.append("")
            lines.append("direction_distribution:")
            lines.append(metadata["direction_label"].value_counts().to_string())
    lines.extend(
        [
            "```",
            "",
            "## Raw6 Channel Stats",
            "",
            "```text",
            raw_stats.to_string(index=False),
            "```",
            "",
            "## Feature Channel Stats",
            "",
            "```text",
            feature_stats.to_string(index=False),
            "```",
            "",
            "## Peak Summary",
            "",
            "```text",
            peak_df.describe().to_string(),
            "```",
            "",
            "## Warnings",
            "",
        ]
    )
    if report["warnings"]:
        lines.extend([f"- {warning}" for warning in report["warnings"]])
    else:
        lines.append("- No warnings generated.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _flatten_baseline_result(result: dict[str, Any]) -> dict[str, Any]:
    row = {
        "eval_split": result.get("eval_split"),
        "num_train": result.get("num_train"),
        "num_eval": result.get("num_eval"),
        "accuracy": result.get("accuracy"),
        "macro_f1": result.get("macro_f1"),
        "warning": result.get("warning"),
    }
    report = result.get("classification_report", {})
    for label in ["forward", "backward", "lateral"]:
        if label in report:
            row[f"{label}_precision"] = report[label].get("precision")
            row[f"{label}_recall"] = report[label].get("recall")
            row[f"{label}_f1"] = report[label].get("f1-score")
    return row
