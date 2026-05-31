from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RAW6_CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
DIRECTION_ORDER = ["forward", "backward", "lateral"]
DIRECTION_TO_ID = {"forward": 0, "backward": 1, "lateral": 2}
SUMMARY_FEATURES = [
    "mean_ax_pre",
    "mean_ay_pre",
    "mean_az_pre",
    "mean_ax_post",
    "mean_ay_post",
    "mean_az_post",
    "mean_gx_pre",
    "mean_gy_pre",
    "mean_gz_pre",
    "mean_gx_post",
    "mean_gy_post",
    "mean_gz_post",
    "peak_signed_ax",
    "peak_signed_ay",
    "peak_signed_az",
    "peak_signed_gx",
    "peak_signed_gy",
    "peak_signed_gz",
    "acc_mag_peak",
    "gyro_mag_peak",
    "roll_pre_mean",
    "roll_post_mean",
    "pitch_pre_mean",
    "pitch_post_mean",
    "delta_roll_window",
    "delta_pitch_window",
    "integrated_gx",
    "integrated_gy",
    "integrated_gz",
]
ACC_DOMINANT_COLS = ["peak_signed_ax", "peak_signed_ay", "peak_signed_az"]
GYRO_DOMINANT_COLS = ["integrated_gx", "integrated_gy", "integrated_gz"]
TILT_DOMINANT_COLS = ["delta_roll_window", "delta_pitch_window"]
AXIS_TRANSFORMS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "identity": ((0, 1, 2), (1, 1, 1)),
    "flip_x": ((0, 1, 2), (-1, 1, 1)),
    "flip_y": ((0, 1, 2), (1, -1, 1)),
    "flip_z": ((0, 1, 2), (1, 1, -1)),
    "swap_xy": ((1, 0, 2), (1, 1, 1)),
    "swap_xz": ((2, 1, 0), (1, 1, 1)),
    "swap_yz": ((0, 2, 1), (1, 1, 1)),
    "swap_xy_flip_x": ((1, 0, 2), (-1, 1, 1)),
    "swap_xy_flip_y": ((1, 0, 2), (1, -1, 1)),
    "swap_xz_flip_x": ((2, 1, 0), (-1, 1, 1)),
    "swap_yz_flip_y": ((0, 2, 1), (1, -1, 1)),
}


@dataclass
class ProcessedData:
    X_raw6: np.ndarray
    metadata: pd.DataFrame
    channel_order_source: str
    input_note: str


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    reports_dir = ensure_dir(output_dir / "reports")
    figures_dir = ensure_dir(output_dir / "figures" / "direction_axis_semantics")

    data = load_processed_data(processed_dir)
    datasets = list(args.datasets)
    supervised_mask = make_supervised_direction_mask(data.metadata)
    supervised_meta = data.metadata.loc[supervised_mask].reset_index(drop=True)
    supervised_X = data.X_raw6[supervised_mask]

    print("Loaded processed data:", processed_dir)
    print("Present datasets:", sorted(data.metadata["dataset"].dropna().astype(str).unique()))
    print("Requested datasets:", datasets)
    print("Supervised direction fall windows:", len(supervised_meta))
    print("Channel order:", data.channel_order_source)
    print(data.input_note)

    if len(supervised_meta) == 0:
        raise RuntimeError("No supervised direction fall windows found.")

    window_features = compute_signed_summary_features(supervised_X, fs=args.fs)
    window_features.insert(0, "direction", supervised_meta["direction_label"].astype(str).to_numpy())
    window_features.insert(0, "dataset", supervised_meta["dataset"].astype(str).to_numpy())
    if "activity_id" in supervised_meta:
        window_features["activity_id"] = supervised_meta["activity_id"].astype(str).to_numpy()
    if "subject_id" in supervised_meta:
        window_features["subject_id"] = supervised_meta["subject_id"].astype(str).to_numpy()

    summary_df = build_axis_summary(window_features, datasets)
    permutation_df = run_umafall_axis_permutation_baseline(
        supervised_X,
        supervised_meta,
        seed=args.seed,
        fs=args.fs,
    )

    save_csv(summary_df, reports_dir / "direction_axis_summary.csv")
    save_csv(permutation_df, reports_dir / "umafall_axis_permutation_baseline.csv")

    plot_direction_curves(supervised_X, supervised_meta, datasets, figures_dir, fs=args.fs)
    plot_summary_boxplots(window_features, datasets, figures_dir)

    report = build_report(
        data=data,
        datasets=datasets,
        axis_summary=summary_df,
        permutation_df=permutation_df,
        window_features=window_features,
    )
    report_path = reports_dir / "direction_axis_semantics_report.md"
    report_path.write_text(report, encoding="utf-8")

    print("Saved:", reports_dir / "direction_axis_summary.csv")
    print("Saved:", reports_dir / "umafall_axis_permutation_baseline.csv")
    print("Saved:", report_path)
    print("Saved figures under:", figures_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit direction-axis semantics without training DS-Fall.")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--datasets", nargs="+", default=["bits", "hifd", "weda", "umafall"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fs", type=float, default=50.0)
    return parser.parse_args()


def load_processed_data(processed_dir: Path) -> ProcessedData:
    if not processed_dir.exists():
        raise FileNotFoundError(f"processed_dir does not exist: {processed_dir}")
    metadata_path = processed_dir / "metadata.csv"
    if not metadata_path.exists():
        parquet_path = processed_dir / "metadata.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing metadata.csv/metadata.parquet in {processed_dir}")
        metadata = pd.read_parquet(parquet_path)
    else:
        metadata = pd.read_csv(metadata_path)
    if "dataset" not in metadata.columns:
        raise ValueError("metadata must contain a 'dataset' column.")

    X_path = processed_dir / "X.npy"
    if not X_path.exists():
        raise FileNotFoundError(f"Missing X.npy in {processed_dir}")
    X = np.load(X_path).astype(np.float32)
    if X.ndim != 3 or X.shape[-1] < 6:
        raise ValueError(f"Expected X.npy shape (N, T, >=6), got {X.shape}")
    if len(X) != len(metadata):
        raise ValueError(f"X length {len(X)} does not match metadata length {len(metadata)}")

    scaler = load_scaler(processed_dir / "scaler.pkl")
    channel_order_source = validate_channel_order(scaler)
    X_raw6 = X[:, :, :6].astype(np.float32)
    input_note = f"Loaded X from {X_path}."
    if scaler is not None and scaler.get("channel_names") == RAW6_CHANNELS:
        mean = np.asarray(scaler["mean"], dtype=np.float32).reshape(1, 1, 6)
        std = np.asarray(scaler["std"], dtype=np.float32).reshape(1, 1, 6)
        X_raw6 = X_raw6 * std + mean
        input_note += " X.npy was converted back to harmonized raw6 values using scaler.pkl."
    else:
        input_note += " X.npy is treated as raw6 because no compatible scaler.pkl was found."
    X_raw6 = np.nan_to_num(X_raw6, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return ProcessedData(
        X_raw6=X_raw6,
        metadata=metadata,
        channel_order_source=channel_order_source,
        input_note=input_note,
    )


def load_scaler(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def validate_channel_order(scaler: dict[str, Any] | None) -> str:
    if scaler is not None and scaler.get("channel_names") == RAW6_CHANNELS:
        return "scaler.pkl channel_names = [ax, ay, az, gx, gy, gz]"
    raise ValueError(
        "Channel order is not explicitly available. Refusing to audit axis semantics because wrong "
        "channel order would invalidate signed-axis diagnostics."
    )


def make_supervised_direction_mask(metadata: pd.DataFrame) -> np.ndarray:
    if "fall_label" not in metadata or "direction_label" not in metadata:
        raise ValueError("metadata must contain fall_label and direction_label columns.")
    mask = metadata["fall_label"].astype(int).eq(1).to_numpy(copy=True)
    mask &= metadata["direction_label"].astype(str).isin(DIRECTION_ORDER).to_numpy(copy=True)
    if "direction_supervised" in metadata:
        mask &= metadata["direction_supervised"].astype(bool).to_numpy(copy=True)
    return mask


def compute_signed_summary_features(
    X: np.ndarray,
    fs: float = 50.0,
    pre_steps: int = 20,
    post_steps: int = 20,
) -> pd.DataFrame:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[-1] != 6:
        raise ValueError(f"Expected X shape (N, T, 6), got {X.shape}")
    dt = 1.0 / float(fs)
    acc = X[:, :, :3]
    gyro = X[:, :, 3:6]
    acc_mag = np.linalg.norm(acc, axis=2)
    gyro_mag = np.linalg.norm(gyro, axis=2)
    roll = np.unwrap(np.arctan2(acc[:, :, 1], acc[:, :, 2]), axis=1)
    pitch = np.unwrap(
        np.arctan2(-acc[:, :, 0], np.sqrt(acc[:, :, 1] ** 2 + acc[:, :, 2] ** 2 + 1e-8)),
        axis=1,
    )
    pre = slice(0, min(pre_steps, X.shape[1]))
    post = slice(max(0, X.shape[1] - post_steps), X.shape[1])
    rows = {
        "mean_ax_pre": acc[:, pre, 0].mean(axis=1),
        "mean_ay_pre": acc[:, pre, 1].mean(axis=1),
        "mean_az_pre": acc[:, pre, 2].mean(axis=1),
        "mean_ax_post": acc[:, post, 0].mean(axis=1),
        "mean_ay_post": acc[:, post, 1].mean(axis=1),
        "mean_az_post": acc[:, post, 2].mean(axis=1),
        "mean_gx_pre": gyro[:, pre, 0].mean(axis=1),
        "mean_gy_pre": gyro[:, pre, 1].mean(axis=1),
        "mean_gz_pre": gyro[:, pre, 2].mean(axis=1),
        "mean_gx_post": gyro[:, post, 0].mean(axis=1),
        "mean_gy_post": gyro[:, post, 1].mean(axis=1),
        "mean_gz_post": gyro[:, post, 2].mean(axis=1),
        "peak_signed_ax": signed_peak(acc[:, :, 0]),
        "peak_signed_ay": signed_peak(acc[:, :, 1]),
        "peak_signed_az": signed_peak(acc[:, :, 2]),
        "peak_signed_gx": signed_peak(gyro[:, :, 0]),
        "peak_signed_gy": signed_peak(gyro[:, :, 1]),
        "peak_signed_gz": signed_peak(gyro[:, :, 2]),
        "acc_mag_peak": acc_mag.max(axis=1),
        "gyro_mag_peak": gyro_mag.max(axis=1),
        "roll_pre_mean": roll[:, pre].mean(axis=1),
        "roll_post_mean": roll[:, post].mean(axis=1),
        "pitch_pre_mean": pitch[:, pre].mean(axis=1),
        "pitch_post_mean": pitch[:, post].mean(axis=1),
        "delta_roll_window": roll[:, post].mean(axis=1) - roll[:, pre].mean(axis=1),
        "delta_pitch_window": pitch[:, post].mean(axis=1) - pitch[:, pre].mean(axis=1),
        "integrated_gx": gyro[:, :, 0].sum(axis=1) * dt,
        "integrated_gy": gyro[:, :, 1].sum(axis=1) * dt,
        "integrated_gz": gyro[:, :, 2].sum(axis=1) * dt,
    }
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def signed_peak(values: np.ndarray) -> np.ndarray:
    idx = np.argmax(np.abs(values), axis=1)
    return values[np.arange(values.shape[0]), idx]


def build_axis_summary(window_features: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subset = window_features[window_features["dataset"].isin(datasets)].copy()
    for (dataset, direction), group in subset.groupby(["dataset", "direction"], sort=True):
        row: dict[str, Any] = {
            "dataset": dataset,
            "direction": direction,
            "n": int(len(group)),
        }
        for feature in SUMMARY_FEATURES:
            row[f"{feature}_mean"] = float(group[feature].mean())
            row[f"{feature}_std"] = float(group[feature].std(ddof=0))
        row["dominant_acc_axis"] = dominant_signed_axis(group, ACC_DOMINANT_COLS)
        row["dominant_gyro_axis"] = dominant_signed_axis(group, GYRO_DOMINANT_COLS)
        row["dominant_tilt_axis"] = dominant_signed_axis(group, TILT_DOMINANT_COLS)
        rows.append(row)
    return pd.DataFrame(rows)


def dominant_signed_axis(group: pd.DataFrame, cols: list[str]) -> str:
    means = group[cols].mean()
    if means.empty:
        return "not_available"
    col = means.abs().idxmax()
    value = float(means[col])
    axis = col.replace("peak_signed_", "").replace("integrated_", "").replace("delta_", "")
    sign = "+" if value >= 0 else "-"
    return f"{sign}{axis}"


def run_umafall_axis_permutation_baseline(
    X: np.ndarray,
    metadata: pd.DataFrame,
    seed: int,
    fs: float,
) -> pd.DataFrame:
    mask = metadata["dataset"].astype(str).eq("umafall").to_numpy()
    if not mask.any():
        return pd.DataFrame(
            [{"transform": "not_available", "classifier": "not_available", "warning": "UMAFall not found"}]
        )
    X_u = X[mask]
    meta_u = metadata.loc[mask].reset_index(drop=True)
    y = meta_u["direction_label"].astype(str).map(DIRECTION_TO_ID).to_numpy()
    unique, counts = np.unique(y, return_counts=True)
    if len(X_u) < 30 or len(unique) < 2 or counts.min() < 2:
        return pd.DataFrame(
            [
                {
                    "transform": "not_available",
                    "classifier": "not_available",
                    "n_supervised": len(X_u),
                    "warning": "Not enough UMAFall direction samples for stratified split",
                }
            ]
        )

    rows: list[dict[str, Any]] = []
    for transform_name in AXIS_TRANSFORMS:
        X_t = apply_axis_transform(X_u, transform_name)
        features = compute_signed_summary_features(X_t, fs=fs)
        feature_matrix = features[SUMMARY_FEATURES].to_numpy(dtype=np.float32)
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix,
            y,
            test_size=0.30,
            random_state=seed,
            stratify=y,
        )
        classifiers = {
            "logistic_regression": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                random_state=seed,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
            ),
        }
        for classifier_name, clf in classifiers.items():
            warning = ""
            try:
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
                accuracy = float(accuracy_score(y_test, pred))
                macro_f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
            except Exception as exc:
                accuracy = np.nan
                macro_f1 = np.nan
                warning = str(exc)
            rows.append(
                {
                    "transform": transform_name,
                    "classifier": classifier_name,
                    "n_supervised": int(len(X_u)),
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "warning": warning,
                }
            )
    return pd.DataFrame(rows)


def apply_axis_transform(X: np.ndarray, transform_name: str) -> np.ndarray:
    if transform_name not in AXIS_TRANSFORMS:
        raise ValueError(f"Unknown transform: {transform_name}")
    perm, signs = AXIS_TRANSFORMS[transform_name]
    signs_arr = np.asarray(signs, dtype=np.float32).reshape(1, 1, 3)
    out = np.asarray(X, dtype=np.float32).copy()
    out[:, :, :3] = out[:, :, list(perm)] * signs_arr
    out[:, :, 3:6] = out[:, :, [3 + p for p in perm]] * signs_arr
    return out


def plot_direction_curves(
    X: np.ndarray,
    metadata: pd.DataFrame,
    datasets: list[str],
    figures_dir: Path,
    fs: float,
) -> None:
    ensure_dir(figures_dir)
    for dataset in datasets:
        mask = metadata["dataset"].astype(str).eq(dataset).to_numpy()
        if not mask.any():
            continue
        X_d = X[mask]
        labels = metadata.loc[mask, "direction_label"].astype(str).reset_index(drop=True)
        plot_axis_group(
            X_d[:, :, :3],
            ["ax", "ay", "az"],
            labels,
            dataset,
            f"{dataset} accelerometer axes by direction",
            figures_dir / f"{dataset}_mean_std_ax_ay_az_by_direction.png",
            fs=fs,
        )
        plot_axis_group(
            X_d[:, :, 3:6],
            ["gx", "gy", "gz"],
            labels,
            dataset,
            f"{dataset} gyroscope axes by direction",
            figures_dir / f"{dataset}_mean_std_gx_gy_gz_by_direction.png",
            fs=fs,
        )
        plot_roll_pitch_group(
            X_d,
            labels,
            dataset,
            figures_dir / f"{dataset}_mean_std_roll_pitch_by_direction.png",
            fs=fs,
        )


def plot_axis_group(
    values: np.ndarray,
    channel_names: list[str],
    labels: pd.Series,
    dataset: str,
    title: str,
    path: Path,
    fs: float,
) -> None:
    t = np.arange(values.shape[1]) / float(fs)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for axis_idx, channel in enumerate(channel_names):
        ax = axes[axis_idx]
        for direction in DIRECTION_ORDER:
            mask = labels.eq(direction).to_numpy()
            if not mask.any():
                continue
            series = values[mask, :, axis_idx]
            mean = series.mean(axis=0)
            std = series.std(axis=0)
            ax.plot(t, mean, label=direction)
            ax.fill_between(t, mean - std, mean + std, alpha=0.14)
        ax.set_title(f"{dataset} {channel}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roll_pitch_group(X: np.ndarray, labels: pd.Series, dataset: str, path: Path, fs: float) -> None:
    acc = X[:, :, :3]
    roll = np.unwrap(np.arctan2(acc[:, :, 1], acc[:, :, 2]), axis=1)
    pitch = np.unwrap(
        np.arctan2(-acc[:, :, 0], np.sqrt(acc[:, :, 1] ** 2 + acc[:, :, 2] ** 2 + 1e-8)),
        axis=1,
    )
    t = np.arange(X.shape[1]) / float(fs)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, values, name in [(axes[0], roll, "roll"), (axes[1], pitch, "pitch")]:
        for direction in DIRECTION_ORDER:
            mask = labels.eq(direction).to_numpy()
            if not mask.any():
                continue
            series = values[mask]
            mean = series.mean(axis=0)
            std = series.std(axis=0)
            ax.plot(t, mean, label=direction)
            ax.fill_between(t, mean - std, mean + std, alpha=0.14)
        ax.set_title(f"{dataset} {name}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_boxplots(window_features: pd.DataFrame, datasets: list[str], figures_dir: Path) -> None:
    ensure_dir(figures_dir)
    df = window_features[window_features["dataset"].isin(datasets)].copy()
    if df.empty:
        return
    plot_box(df, "delta_roll_window", figures_dir / "boxplot_delta_roll_window_by_dataset_direction.png")
    plot_box(df, "delta_pitch_window", figures_dir / "boxplot_delta_pitch_window_by_dataset_direction.png")
    for feature in ["integrated_gx", "integrated_gy", "integrated_gz"]:
        plot_box(df, feature, figures_dir / f"boxplot_{feature}_by_dataset_direction.png")


def plot_box(df: pd.DataFrame, feature: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="dataset", y=feature, hue="direction", hue_order=DIRECTION_ORDER, ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(f"{feature} by dataset and direction")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_report(
    data: ProcessedData,
    datasets: list[str],
    axis_summary: pd.DataFrame,
    permutation_df: pd.DataFrame,
    window_features: pd.DataFrame,
) -> str:
    diagnosis = diagnose(axis_summary, permutation_df, window_features)
    lines = [
        "# Direction Axis Semantics Report",
        "",
        "This audit uses only supervised fall windows with direction labels. It does not modify data and does not train DS-Fall.",
        "",
        f"- Window shape: {tuple(data.X_raw6.shape[1:])}",
        f"- Channel order source: {data.channel_order_source}",
        f"- Input note: {data.input_note}",
        "",
        "## Direction Counts",
        "",
        dataframe_to_markdown(direction_count_table(window_features, datasets)),
        "",
        "## Dominant Axis Summary",
        "",
        dataframe_to_markdown(
            axis_summary[
                [
                    "dataset",
                    "direction",
                    "n",
                    "dominant_acc_axis",
                    "dominant_gyro_axis",
                    "dominant_tilt_axis",
                    "integrated_gx_mean",
                    "integrated_gy_mean",
                    "integrated_gz_mean",
                    "delta_roll_window_mean",
                    "delta_pitch_window_mean",
                ]
            ]
        ),
        "",
        "## UMAFall Axis Permutation Baseline",
        "",
        dataframe_to_markdown(permutation_df.sort_values(["classifier", "macro_f1"], ascending=[True, False])),
        "",
        "## Diagnosis",
        "",
    ]
    lines.extend([f"- {item}" for item in diagnosis])
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `outputs/reports/direction_axis_summary.csv`",
            "- `outputs/reports/umafall_axis_permutation_baseline.csv`",
            "- `outputs/figures/direction_axis_semantics/*.png`",
        ]
    )
    return "\n".join(lines) + "\n"


def direction_count_table(window_features: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    df = window_features[window_features["dataset"].isin(datasets)]
    return (
        df.groupby(["dataset", "direction"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["dataset", "direction"])
    )


def diagnose(axis_summary: pd.DataFrame, permutation_df: pd.DataFrame, window_features: pd.DataFrame) -> list[str]:
    items: list[str] = []
    items.append(diagnose_umafall_axis_similarity(axis_summary))
    items.append(diagnose_forward_backward_signs(axis_summary))
    items.append(diagnose_lateral_merge(window_features))
    items.append(diagnose_axis_permutation(permutation_df))
    items.append(diagnose_likely_cause(axis_summary, permutation_df, window_features))
    return items


def diagnose_umafall_axis_similarity(axis_summary: pd.DataFrame) -> str:
    if axis_summary.empty or "umafall" not in set(axis_summary["dataset"]):
        return "UMAFall dominant-axis comparison is not available."
    comparisons = []
    for direction in DIRECTION_ORDER:
        ref = axis_summary[
            axis_summary["dataset"].isin(["bits", "weda"]) & axis_summary["direction"].eq(direction)
        ]
        uma = axis_summary[axis_summary["dataset"].eq("umafall") & axis_summary["direction"].eq(direction)]
        if ref.empty or uma.empty:
            continue
        ref_gyro = majority_unsigned_axis(ref["dominant_gyro_axis"].tolist())
        ref_tilt = majority_unsigned_axis(ref["dominant_tilt_axis"].tolist())
        uma_gyro = unsigned_axis(str(uma.iloc[0]["dominant_gyro_axis"]))
        uma_tilt = unsigned_axis(str(uma.iloc[0]["dominant_tilt_axis"]))
        comparisons.append(f"{direction}: gyro {uma_gyro} vs ref {ref_gyro}, tilt {uma_tilt} vs ref {ref_tilt}")
    mismatch = [c for c in comparisons if "vs ref" in c and not same_axes_from_text(c)]
    verdict = "UMAFall does not consistently use the same dominant gyro/tilt axes as BITS/WEDA." if mismatch else "UMAFall dominant axes look broadly similar to BITS/WEDA."
    return verdict + " " + "; ".join(comparisons)


def same_axes_from_text(text: str) -> bool:
    # Conservative parser for strings like "forward: gyro gy vs ref gx, tilt roll_window vs ref pitch_window".
    try:
        gyro_part = text.split("gyro ", 1)[1].split(", tilt", 1)[0]
        gyro_left, gyro_right = gyro_part.split(" vs ref ")
        tilt_part = text.split("tilt ", 1)[1]
        tilt_left, tilt_right = tilt_part.split(" vs ref ")
        return gyro_left == gyro_right and tilt_left == tilt_right
    except Exception:
        return False


def diagnose_forward_backward_signs(axis_summary: pd.DataFrame) -> str:
    rows = []
    for dataset in sorted(axis_summary["dataset"].unique()) if not axis_summary.empty else []:
        fwd = axis_summary[axis_summary["dataset"].eq(dataset) & axis_summary["direction"].eq("forward")]
        bwd = axis_summary[axis_summary["dataset"].eq(dataset) & axis_summary["direction"].eq("backward")]
        if fwd.empty or bwd.empty:
            continue
        fg = str(fwd.iloc[0]["dominant_gyro_axis"])
        bg = str(bwd.iloc[0]["dominant_gyro_axis"])
        ft = str(fwd.iloc[0]["dominant_tilt_axis"])
        bt = str(bwd.iloc[0]["dominant_tilt_axis"])
        rows.append(f"{dataset}: forward/backward gyro {fg}/{bg}, tilt {ft}/{bt}")
    return "Forward/backward sign consistency by dataset: " + "; ".join(rows)


def diagnose_lateral_merge(window_features: pd.DataFrame) -> str:
    flags = []
    for dataset, group in window_features[window_features["direction"].eq("lateral")].groupby("dataset"):
        candidates = ["integrated_gx", "integrated_gy", "integrated_gz", "delta_roll_window", "delta_pitch_window"]
        ratios = []
        for col in candidates:
            values = group[col].to_numpy(dtype=float)
            if len(values) < 5:
                continue
            positive_frac = float(np.mean(values > 0))
            mean = float(np.mean(values))
            std = float(np.std(values))
            if 0.35 <= positive_frac <= 0.65 and std > max(1e-6, 2.0 * abs(mean)):
                ratios.append(f"{col} pos_frac={positive_frac:.2f}")
        if ratios:
            flags.append(f"{dataset} lateral may contain opposite signs ({', '.join(ratios[:3])})")
    if not flags:
        return "No strong lateral left/right merge signal detected by sign-balance heuristic."
    return "Lateral merge diagnostic: " + "; ".join(flags)


def diagnose_axis_permutation(permutation_df: pd.DataFrame) -> str:
    if permutation_df.empty or "macro_f1" not in permutation_df:
        return "UMAFall axis permutation baseline is not available."
    rf = permutation_df[permutation_df["classifier"].eq("random_forest")].copy()
    if rf.empty:
        rf = permutation_df.copy()
    identity = rf[rf["transform"].eq("identity")]["macro_f1"]
    identity_f1 = float(identity.iloc[0]) if not identity.empty and pd.notna(identity.iloc[0]) else np.nan
    best = rf.sort_values("macro_f1", ascending=False).iloc[0]
    best_f1 = float(best["macro_f1"])
    delta = best_f1 - identity_f1 if np.isfinite(identity_f1) else np.nan
    if np.isfinite(delta) and delta >= 0.05:
        verdict = "A UMAFall axis permutation improves RF macro F1 substantially, supporting an axis-convention issue."
    else:
        verdict = "No UMAFall axis permutation improves RF macro F1 substantially over identity."
    return f"{verdict} Identity RF={identity_f1:.4f}, best={best['transform']} RF={best_f1:.4f}, delta={delta:.4f}."


def diagnose_likely_cause(
    axis_summary: pd.DataFrame,
    permutation_df: pd.DataFrame,
    window_features: pd.DataFrame,
) -> str:
    perm_text = diagnose_axis_permutation(permutation_df)
    lateral_text = diagnose_lateral_merge(window_features)
    axis_text = diagnose_umafall_axis_similarity(axis_summary)
    if "substantially" in perm_text and "No UMAFall" not in perm_text:
        likely = "axis convention"
    elif "opposite signs" in lateral_text:
        likely = "label semantics, especially lateral left/right merge"
    elif "does not consistently use" in axis_text:
        likely = "axis convention or cross-dataset sensor-frame mismatch"
    else:
        likely = "model limitation or cross-dataset domain conflict more than simple unit/axis permutation"
    return f"Most likely current direction bottleneck: {likely}."


def majority_unsigned_axis(values: list[str]) -> str:
    axes = [unsigned_axis(v) for v in values]
    if not axes:
        return "not_available"
    return pd.Series(axes).value_counts().index[0]


def unsigned_axis(value: str) -> str:
    value = str(value)
    if value.startswith(("+", "-")):
        value = value[1:]
    return value


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    compact = df.copy()
    for col in compact.columns:
        if pd.api.types.is_float_dtype(compact[col]):
            compact[col] = compact[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    header = "| " + " | ".join(map(str, compact.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(compact.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in compact.to_numpy()]
    return "\n".join([header, sep] + rows)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
