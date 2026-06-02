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


DATASETS = ("bits", "weda", "hifd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run train-2-test-1 cross-dataset evaluation on BITS, WEDA, and HIFD only."
    )
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
            id="E1",
            experiment_type="train2_test1_bits_weda_hifd",
            train_datasets=("bits", "weda"),
            test_datasets=("hifd",),
            test_split=None,
            note="Train BITS+WEDA, test all HIFD windows. UMAFall is filtered out before splits.",
        ),
        DomainExperimentSpec(
            id="E2",
            experiment_type="train2_test1_bits_weda_hifd",
            train_datasets=("bits", "hifd"),
            test_datasets=("weda",),
            test_split=None,
            note="Train BITS+HIFD, test all WEDA windows. UMAFall is filtered out before splits.",
        ),
        DomainExperimentSpec(
            id="E3",
            experiment_type="train2_test1_bits_weda_hifd",
            train_datasets=("weda", "hifd"),
            test_datasets=("bits",),
            test_split=None,
            note="Train WEDA+HIFD, test all BITS windows. UMAFall is filtered out before splits.",
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
    csv_path = reports_dir / "train2_test1_bits_weda_hifd.csv"
    df.to_csv(csv_path, index=False)
    md_path = reports_dir / "train2_test1_bits_weda_hifd.md"
    md_path.write_text(build_markdown(df, output_dir), encoding="utf-8")
    update_analysis_notes_from_reports(project_root=Path(output_dir).parent, output_dir=output_dir)


def enrich_row(row: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    out = dict(row)
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

    fig_dir = ensure_dir(Path(output_dir) / "figures" / "train2_test1_confusion_matrices" / experiment_id)
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
        "# Train-2-Test-1: BITS, WEDA, HIFD",
        "",
        "Fixed model: B0 / A5WCEFW = DS-Fall-RD, tilt12, task-specific attention, weighted CE fall, weighted CE direction.",
        "UMAFall is excluded before split construction. Held-out test datasets are not used for scaler fitting, early stopping, or threshold tuning.",
        "",
    ]
    if df.empty:
        return "\n".join(lines + ["_No rows yet._", ""]) + "\n"

    lines.extend(["## Table 1: Train 2 Test 1 Results", "", markdown_table(table1(df)), ""])
    compare = table2_compare_dataset_specific(df, output_dir)
    if not compare.empty:
        lines.extend(["## Table 2: Compare with Dataset-Specific", "", markdown_table(compare), ""])
    else:
        lines.extend(["## Table 2: Compare with Dataset-Specific", "", "_Dataset-specific baseline rows were not found._", ""])
    lines.extend(["## Table 3: Direction Per-class F1", "", markdown_table(table3(df)), ""])

    mixed = find_mixed_3dataset_baseline(output_dir)
    lines.extend(["## Mixed 3-Dataset Baseline", ""])
    if mixed is None:
        lines.extend(["No mixed BITS+WEDA+HIFD-only baseline was found in existing reports, and this task did not request running an extra mixed baseline.", ""])
    else:
        lines.extend([mixed, ""])

    lines.extend(["## Diagnosis", ""])
    lines.extend(diagnosis_lines(df, compare))
    lines.append("")
    return "\n".join(lines)


def table1(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "id": "experiment_id",
        "train_datasets": "train_datasets",
        "test_datasets": "test_dataset",
        "fall_f1": "fall_f1",
        "direction_macro_f1": "direction_macro_f1",
        "direction_accuracy": "direction_accuracy",
        "direction_num_supervised": "direction_n",
        "params": "params",
    }
    out = df[[col for col in cols if col in df.columns]].rename(columns=cols)
    return format_float_columns(out)


def table2_compare_dataset_specific(df: pd.DataFrame, output_dir: str | Path) -> pd.DataFrame:
    baseline_path = Path(output_dir) / "reports" / "domain_conflict_experiments.csv"
    if not baseline_path.exists():
        return pd.DataFrame()
    base = pd.read_csv(baseline_path)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        test_dataset = str(row.get("test_datasets"))
        match = base[
            (base.get("experiment_type", pd.Series(dtype=str)).astype(str) == "dataset_specific")
            & (base.get("train_datasets", pd.Series(dtype=str)).astype(str) == test_dataset)
        ]
        if match.empty:
            continue
        ds_f1 = match.iloc[0].get("direction_macro_f1")
        train2_f1 = row.get("direction_macro_f1")
        rows.append(
            {
                "test_dataset": test_dataset,
                "dataset_specific_direction_f1": ds_f1,
                "train2_test1_direction_f1": train2_f1,
                "drop": ds_f1 - train2_f1,
            }
        )
    return format_float_columns(pd.DataFrame(rows))


def table3(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "id": "experiment_id",
        "test_datasets": "test_dataset",
        "direction_forward_f1": "forward_f1",
        "direction_backward_f1": "backward_f1",
        "direction_lateral_f1": "lateral_f1",
    }
    out = df[[col for col in cols if col in df.columns]].rename(columns=cols)
    return format_float_columns(out)


def find_mixed_3dataset_baseline(output_dir: str | Path) -> str | None:
    path = Path(output_dir) / "reports" / "domain_conflict_experiments.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "train_datasets" not in df or "test_datasets" not in df:
        return None
    target = "+".join(DATASETS)
    candidates = df[
        (df["train_datasets"].astype(str) == target)
        & (df["test_datasets"].astype(str) == target)
    ]
    if candidates.empty:
        return None
    row = candidates.iloc[0]
    return (
        f"Existing mixed BITS+WEDA+HIFD row: fall F1 {_fmt(row.get('fall_f1'))}, "
        f"direction macro F1 {_fmt(row.get('direction_macro_f1'))}."
    )


def diagnosis_lines(df: pd.DataFrame, compare: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if df.empty:
        return ["- No completed experiment rows are available."]

    valid_dir = df.dropna(subset=["direction_macro_f1"]).copy()
    if not valid_dir.empty:
        best = valid_dir.sort_values("direction_macro_f1", ascending=False).iloc[0]
        worst = valid_dir.sort_values("direction_macro_f1", ascending=True).iloc[0]
        lines.append(
            f"1. Best unseen direction generalization: {best['test_datasets']} "
            f"(train {best['train_datasets']}), direction macro F1 {_fmt(best['direction_macro_f1'])}."
        )
        lines.append(
            f"2. Hardest unseen dataset: {worst['test_datasets']} "
            f"(train {worst['train_datasets']}), direction macro F1 {_fmt(worst['direction_macro_f1'])}."
        )
    else:
        lines.append("1. Direction generalization cannot be ranked because direction metrics are missing.")
        lines.append("2. Hardest unseen dataset cannot be determined because direction metrics are missing.")

    if compare.empty:
        lines.append("3. Direction drop versus dataset-specific baselines is not available.")
    else:
        parts = [
            f"{row['test_dataset']}: {_fmt(row['drop'])}"
            for _, row in compare.iterrows()
        ]
        mean_drop = pd.to_numeric(compare["drop"], errors="coerce").dropna().mean()
        lines.append(f"3. Direction drop versus dataset-specific baseline: {', '.join(parts)}; mean drop {_fmt(mean_drop)}.")

    fall_mean = pd.to_numeric(df.get("fall_f1"), errors="coerce").dropna().mean()
    dir_mean = pd.to_numeric(df.get("direction_macro_f1"), errors="coerce").dropna().mean()
    if pd.notna(fall_mean) and pd.notna(dir_mean):
        verdict = "yes" if fall_mean > dir_mean else "no"
        lines.append(
            f"4. Fall detection generalizes better than direction: {verdict} "
            f"(mean fall F1 {_fmt(fall_mean)} vs mean direction macro F1 {_fmt(dir_mean)})."
        )
    else:
        lines.append("4. Fall-vs-direction generalization cannot be compared because metrics are incomplete.")

    e1 = df[df["id"] == "E1"]
    if not e1.empty:
        e1_dir = e1.iloc[0].get("direction_macro_f1")
        if pd.notna(e1_dir) and float(e1_dir) < 0.50:
            lines.append("5. BITS+WEDA is a plausible clean-direction setup, while HIFD should be treated as an external stress-test.")
        else:
            lines.append("5. BITS+WEDA -> HIFD is not uniquely poor in this run; HIFD need not be isolated solely as a stress-test based on these rows.")
    else:
        lines.append("5. The BITS+WEDA clean-direction question cannot be answered because E1 is missing.")

    if not compare.empty:
        large_drops = pd.to_numeric(compare["drop"], errors="coerce") > 0.10
        if large_drops.any():
            lines.append("6. There is still evidence of direction label/domain conflict among BITS, WEDA, and HIFD.")
        else:
            lines.append("6. Direction label/domain conflict among these three datasets appears limited relative to dataset-specific baselines.")
    else:
        lines.append("6. Direction label/domain conflict cannot be assessed without dataset-specific comparisons.")
    return lines


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
                out[col] = out[col].map(lambda value: "" if pd.isna(value) else str(int(value)))
            else:
                out[col] = out[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out


def _fmt(value: Any) -> str:
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
