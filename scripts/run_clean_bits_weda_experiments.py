from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import make_config
from src.experiments.ablation_rd import load_processed_training_data
from src.experiments.domain_conflict import DomainExperimentSpec, run_single_domain_experiment
from src.reporting.analysis_notes import update_analysis_notes_from_reports
from src.training.evaluate import plot_confusion_matrix
from src.utils.io import ensure_dir


DATASETS = ("bits", "weda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clean-direction A5WCEFW experiments on BITS and WEDA only.")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--processed-x-is-raw", action="store_true", help="Use if data/processed/X.npy is raw6, not normalized.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = make_config(args.project_root)

    data = load_processed_training_data(
        config.processed_dir,
        processed_x_is_normalized=not args.processed_x_is_raw,
    )
    metadata = data["metadata"].copy()
    metadata["dataset"] = metadata["dataset"].astype(str).str.lower()

    keep = metadata["dataset"].isin(DATASETS).to_numpy()
    if not keep.any():
        raise ValueError(f"No windows found for required datasets: {DATASETS}")
    present = sorted(metadata.loc[keep, "dataset"].unique().tolist())
    missing = sorted(set(DATASETS) - set(present))
    if missing:
        raise ValueError(f"Missing datasets in processed metadata: {missing}")

    X_raw6 = data["X_raw6"][keep]
    y_fall = data["y_fall"][keep]
    y_direction = data["y_direction"][keep]
    direction_mask = data["direction_mask"][keep]
    metadata = metadata.loc[keep].reset_index(drop=True)

    specs = [
        DomainExperimentSpec(
            id="CBW_E1",
            experiment_type="clean_bits_weda",
            train_datasets=("bits",),
            test_datasets=("weda",),
            test_split=None,
            note="Train BITS only; test all WEDA windows. HIFD/UMAFall are filtered out before splits.",
        ),
        DomainExperimentSpec(
            id="CBW_E2",
            experiment_type="clean_bits_weda",
            train_datasets=("weda",),
            test_datasets=("bits",),
            test_split=None,
            note="Train WEDA only; test all BITS windows. HIFD/UMAFall are filtered out before splits.",
        ),
        DomainExperimentSpec(
            id="CBW_E3",
            experiment_type="clean_bits_weda_mixed",
            train_datasets=("bits", "weda"),
            test_datasets=("bits", "weda"),
            test_split="test",
            note="Train BITS+WEDA train splits, validate on BITS+WEDA val splits, test on BITS+WEDA test splits.",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = run_single_domain_experiment(
            spec,
            config=config,
            X_raw6=X_raw6,
            metadata=metadata,
            y_fall=y_fall,
            y_direction=y_direction,
            direction_mask=direction_mask,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        rows.append(row)
        save_outputs(rows, config.output_dir)
        copy_confusion_matrices(config.output_dir, spec.id)

    save_outputs(rows, config.output_dir)


def save_outputs(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    reports_dir = ensure_dir(Path(output_dir) / "reports")
    df = pd.DataFrame([enrich_row(row, output_dir) for row in rows])
    df.to_csv(reports_dir / "clean_bits_weda_experiments.csv", index=False)
    (reports_dir / "clean_bits_weda_experiments.md").write_text(build_markdown(df, output_dir), encoding="utf-8")
    update_analysis_notes_from_reports(project_root=Path(output_dir).parent, output_dir=output_dir)


def enrich_row(row: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    out = dict(row)
    out["experiment_id"] = str(out.get("id", "")).replace("CBW_", "")
    out["direction_loss_type"] = "weighted_ce"
    out["lambda_direction"] = 1.5
    out["fall_loss_weighted"] = True
    out["augment"] = False

    metrics_path = Path(output_dir) / "runs" / f"DOMAIN_{out.get('id')}_A5WCEFW" / "metrics.json"
    if not metrics_path.exists():
        return out
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    fall_cm = metrics.get("fall", {}).get("confusion_matrix")
    direction_cm = metrics.get("direction", {}).get("confusion_matrix")
    if fall_cm is not None:
        out["fall_confusion_matrix"] = json.dumps(fall_cm)
    if direction_cm is not None:
        out["direction_confusion_matrix"] = json.dumps(direction_cm)
    return out


def copy_confusion_matrices(output_dir: str | Path, experiment_id: str) -> None:
    run_metrics = Path(output_dir) / "runs" / f"DOMAIN_{experiment_id}_A5WCEFW" / "metrics.json"
    if not run_metrics.exists():
        return
    with run_metrics.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    fig_dir = ensure_dir(Path(output_dir) / "figures" / "clean_bits_weda_confusion_matrices" / experiment_id)
    fall_cm = metrics.get("fall", {}).get("confusion_matrix")
    if fall_cm is not None:
        plot_confusion_matrix(
            fall_cm,
            ["non_fall", "fall"],
            f"{experiment_id} fall confusion matrix",
            fig_dir / "fall.png",
            show=False,
        )
    direction_cm = metrics.get("direction", {}).get("confusion_matrix")
    if direction_cm is not None:
        plot_confusion_matrix(
            direction_cm,
            ["forward", "backward", "lateral"],
            f"{experiment_id} direction confusion matrix",
            fig_dir / "direction.png",
            show=False,
        )


def build_markdown(df: pd.DataFrame, output_dir: str | Path) -> str:
    lines = [
        "# Clean BITS-WEDA Experiments",
        "",
        "Fixed model: B0 / A5WCEFW = DS-Fall-RD, tilt12, task-specific attention, weighted CE fall, weighted CE direction.",
        "Only BITS and WEDA are retained before split construction. HIFD and UMAFall are not used.",
        "",
    ]
    if df.empty:
        return "\n".join(lines + ["_No rows yet._", ""]) + "\n"

    lines.extend(["## Table 1: Clean BITS-WEDA Results", "", markdown_table(table1(df)), ""])
    lines.extend(["## Table 2: Per-dataset Results for E3", "", markdown_table(table2_e3(df, output_dir)), ""])
    lines.extend(["## Table 3: Direction Per-class F1", "", markdown_table(table3(df)), ""])
    drop = table4_cross_dataset_drop(df, output_dir)
    lines.extend(["## Table 4: Cross-dataset Drop", "", markdown_table(drop), ""])
    lines.extend(["## Diagnosis", ""])
    lines.extend(diagnosis_lines(df, drop))
    lines.append("")
    return "\n".join(lines)


def table1(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "experiment_id": "experiment_id",
        "train_datasets": "train_datasets",
        "test_datasets": "test_datasets",
        "fall_f1": "fall_f1",
        "direction_macro_f1": "direction_macro_f1",
        "direction_accuracy": "direction_accuracy",
        "direction_num_supervised": "direction_n",
        "params": "params",
    }
    out = df[[col for col in cols if col in df.columns]].rename(columns=cols)
    return format_float_columns(out)


def table2_e3(df: pd.DataFrame, output_dir: str | Path) -> pd.DataFrame:
    e3 = df[df["experiment_id"] == "E3"]
    if e3.empty:
        return pd.DataFrame(columns=["dataset", "fall_f1", "direction_macro_f1", "direction_accuracy", "direction_n"])
    row = e3.iloc[0]
    rows = []
    for dataset in DATASETS:
        rows.append(
            {
                "dataset": dataset,
                "fall_f1": row.get(f"{dataset}_fall_f1"),
                "direction_macro_f1": row.get(f"{dataset}_direction_macro_f1"),
                "direction_accuracy": read_per_dataset_direction_accuracy(output_dir, "CBW_E3", dataset),
                "direction_n": row.get(f"{dataset}_direction_num_supervised"),
            }
        )
    return format_float_columns(pd.DataFrame(rows))


def read_per_dataset_direction_accuracy(output_dir: str | Path, experiment_id: str, dataset: str) -> float | None:
    path = Path(output_dir) / "runs" / f"DOMAIN_{experiment_id}_A5WCEFW" / "metrics.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics.get("per_dataset", {}).get(dataset, {}).get("direction", {}).get("accuracy")


def table3(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "experiment_id": "experiment_id",
        "test_datasets": "test_dataset",
        "direction_forward_f1": "forward_f1",
        "direction_backward_f1": "backward_f1",
        "direction_lateral_f1": "lateral_f1",
    }
    out = df[[col for col in cols if col in df.columns]].rename(columns=cols)
    return format_float_columns(out)


def table4_cross_dataset_drop(df: pd.DataFrame, output_dir: str | Path) -> pd.DataFrame:
    baseline_path = Path(output_dir) / "reports" / "domain_conflict_experiments.csv"
    if not baseline_path.exists():
        return pd.DataFrame(columns=["comparison", "dataset_specific_direction_f1", "cross_dataset_direction_f1", "drop"])
    base = pd.read_csv(baseline_path)
    rows = []
    comparisons = [
        ("bits-only vs train weda -> test bits", "bits", "E2"),
        ("weda-only vs train bits -> test weda", "weda", "E1"),
    ]
    for label, dataset, exp_id in comparisons:
        ds = base[
            (base.get("experiment_type", pd.Series(dtype=str)).astype(str) == "dataset_specific")
            & (base.get("train_datasets", pd.Series(dtype=str)).astype(str) == dataset)
        ]
        exp = df[df["experiment_id"] == exp_id]
        if ds.empty or exp.empty:
            continue
        ds_f1 = ds.iloc[0].get("direction_macro_f1")
        cross_f1 = exp.iloc[0].get("direction_macro_f1")
        rows.append(
            {
                "comparison": label,
                "dataset_specific_direction_f1": ds_f1,
                "cross_dataset_direction_f1": cross_f1,
                "drop": ds_f1 - cross_f1,
            }
        )
    return format_float_columns(pd.DataFrame(rows))


def diagnosis_lines(df: pd.DataFrame, drop: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    e1 = first_row(df, "E1")
    e2 = first_row(df, "E2")
    e3 = first_row(df, "E3")

    e1_dir = value(e1, "direction_macro_f1")
    e2_dir = value(e2, "direction_macro_f1")
    e3_dir = value(e3, "direction_macro_f1")

    lines.append(
        f"1. Train bits -> test weda: direction macro F1 {_fmt(e1_dir)}; "
        f"{'generalizes moderately' if e1_dir is not None and e1_dir >= 0.50 else 'does not generalize well'}."
    )
    lines.append(
        f"2. Train weda -> test bits: direction macro F1 {_fmt(e2_dir)}; "
        f"{'generalizes moderately' if e2_dir is not None and e2_dir >= 0.50 else 'does not generalize well'}."
    )
    if e3 is not None:
        best_cross = max([v for v in [e1_dir, e2_dir] if v is not None], default=None)
        better = e3_dir is not None and best_cross is not None and e3_dir > best_cross
        lines.append(
            f"3. Train bits+weda then test both gives direction macro F1 {_fmt(e3_dir)}; "
            f"{'it is better than either cross-dataset direction run' if better else 'it is not clearly better than the best cross-dataset run'}."
        )
    else:
        lines.append("3. E3 is missing, so the mixed clean setup cannot be judged.")

    if e1_dir is not None and e2_dir is not None:
        easier = "bits" if e2_dir > e1_dir else "weda"
        lines.append(f"4. Easier unseen dataset in this setup: {easier}.")
    else:
        lines.append("4. Easier unseen dataset cannot be determined.")

    if not drop.empty:
        parts = [f"{row['comparison']}: {_fmt(row['drop'])}" for _, row in drop.iterrows()]
        mean_drop = pd.to_numeric(drop["drop"], errors="coerce").dropna().mean()
        lines.append(f"5. Direction drop when test is unseen: {', '.join(parts)}; mean drop {_fmt(mean_drop)}.")
    else:
        lines.append("5. Direction drop when test is unseen is not available.")

    if e3_dir is not None and e3_dir >= 0.65:
        lines.append("6. BITS+WEDA is strong enough as a clean-direction benchmark for this model/config.")
    elif e3_dir is not None:
        lines.append("6. BITS+WEDA is cleaner than the 3-dataset setup but still not a fully strong clean-direction benchmark at the current score.")
    else:
        lines.append("6. BITS+WEDA benchmark suitability cannot be judged without E3.")

    if not drop.empty and (pd.to_numeric(drop["drop"], errors="coerce") > 0.10).any():
        lines.append("7. There is still evidence of label/domain conflict between BITS and WEDA.")
    else:
        lines.append("7. Evidence of BITS/WEDA label-domain conflict is limited in this run.")
    return lines


def first_row(df: pd.DataFrame, exp_id: str) -> pd.Series | None:
    match = df[df["experiment_id"] == exp_id]
    if match.empty:
        return None
    return match.iloc[0]


def value(row: pd.Series | None, key: str) -> float | None:
    if row is None or key not in row:
        return None
    raw = row.get(key)
    try:
        if pd.isna(raw):
            return None
    except TypeError:
        pass
    return float(raw)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in df.to_numpy()]
    return "\n".join([header, sep] + rows)


def format_float_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            if col in {"params", "direction_n"}:
                out[col] = out[col].map(lambda item: "" if pd.isna(item) else str(int(item)))
            else:
                out[col] = out[col].map(lambda item: "" if pd.isna(item) else f"{float(item):.4f}")
    return out


def _fmt(raw: Any) -> str:
    try:
        if pd.isna(raw):
            return "NA"
    except TypeError:
        pass
    try:
        return f"{float(raw):.4f}"
    except (TypeError, ValueError):
        return str(raw)


if __name__ == "__main__":
    main()
