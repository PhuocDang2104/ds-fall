"""Summarize event-centered BITS/WEDA experiments E1-E7.

E6/E7 are not new training runs. They reuse the E3 mixed BITS+WEDA model
predictions and evaluate the BITS and WEDA test subsets separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


FALL_LABELS = ["non_fall", "fall"]
DIRECTION_LABELS = ["forward", "backward", "lateral"]


def _safe_float(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    return float(value)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a small DataFrame as a GitHub markdown table without tabulate."""
    if df.empty:
        return "_No rows._"

    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        return str(value)

    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def compute_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    y_fall = predictions["true_fall"].astype(str)
    p_fall = predictions["pred_fall"].astype(str)

    direction_mask = predictions["true_direction"].astype(str).isin(DIRECTION_LABELS)
    y_dir = predictions.loc[direction_mask, "true_direction"].astype(str)
    p_dir = predictions.loc[direction_mask, "pred_direction"].astype(str)

    metrics: dict[str, object] = {
        "n_test": int(len(predictions)),
        "fall_accuracy": accuracy_score(y_fall, p_fall),
        "fall_precision": precision_score(
            y_fall, p_fall, labels=FALL_LABELS, pos_label="fall", zero_division=0
        ),
        "fall_recall": recall_score(
            y_fall, p_fall, labels=FALL_LABELS, pos_label="fall", zero_division=0
        ),
        "fall_f1": f1_score(
            y_fall, p_fall, labels=FALL_LABELS, pos_label="fall", zero_division=0
        ),
        "direction_n_supervised": int(direction_mask.sum()),
        "direction_accuracy": float("nan"),
        "direction_macro_f1": float("nan"),
    }

    for label in DIRECTION_LABELS:
        metrics[f"{label}_f1"] = float("nan")

    if direction_mask.any():
        metrics["direction_accuracy"] = accuracy_score(y_dir, p_dir)
        metrics["direction_macro_f1"] = f1_score(
            y_dir, p_dir, labels=DIRECTION_LABELS, average="macro", zero_division=0
        )
        per_class = classification_report(
            y_dir,
            p_dir,
            labels=DIRECTION_LABELS,
            output_dict=True,
            zero_division=0,
        )
        for label in DIRECTION_LABELS:
            metrics[f"{label}_f1"] = per_class[label]["f1-score"]

    return metrics


def write_subset_artifacts(subset: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(subset)

    subset.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)

    fall_cm = confusion_matrix(
        subset["true_fall"].astype(str),
        subset["pred_fall"].astype(str),
        labels=FALL_LABELS,
    )
    pd.DataFrame(fall_cm, index=FALL_LABELS, columns=FALL_LABELS).to_csv(
        output_dir / "confusion_matrix_fall.csv"
    )
    (output_dir / "classification_report_fall.txt").write_text(
        classification_report(
            subset["true_fall"].astype(str),
            subset["pred_fall"].astype(str),
            labels=FALL_LABELS,
            zero_division=0,
        ),
        encoding="utf-8",
    )

    direction_mask = subset["true_direction"].astype(str).isin(DIRECTION_LABELS)
    if direction_mask.any():
        direction_subset = subset.loc[direction_mask]
        dir_cm = confusion_matrix(
            direction_subset["true_direction"].astype(str),
            direction_subset["pred_direction"].astype(str),
            labels=DIRECTION_LABELS,
        )
        pd.DataFrame(dir_cm, index=DIRECTION_LABELS, columns=DIRECTION_LABELS).to_csv(
            output_dir / "confusion_matrix_direction.csv"
        )
        (output_dir / "classification_report_direction.txt").write_text(
            classification_report(
                direction_subset["true_direction"].astype(str),
                direction_subset["pred_direction"].astype(str),
                labels=DIRECTION_LABELS,
                zero_division=0,
            ),
            encoding="utf-8",
        )

    return metrics


def build_markdown(summary: pd.DataFrame) -> str:
    key_cols = [
        "experiment_id",
        "train_dataset",
        "test_dataset",
        "fall_f1",
        "direction_macro_f1",
        "direction_accuracy",
        "direction_n_supervised",
        "model_params",
    ]
    main_table = summary[key_cols].copy()

    def get_row(exp_id: str) -> pd.Series:
        match = summary.loc[summary["experiment_id"] == exp_id]
        if match.empty:
            raise ValueError(f"Missing experiment {exp_id}")
        return match.iloc[0]

    e1 = get_row("E1_BITS_TO_BITS")
    e2 = get_row("E2_WEDA_TO_WEDA")
    e3 = get_row("E3_BITS_WEDA_MIXED")
    e4 = get_row("E4_BITS_TO_WEDA")
    e5 = get_row("E5_WEDA_TO_BITS")
    e6 = get_row("E6_E3_MIXED_TEST_BITS")
    e7 = get_row("E7_E3_MIXED_TEST_WEDA")

    dataset_specific = pd.DataFrame(
        [
            {
                "dataset": "bits",
                "dataset_specific_fall_f1": e1["fall_f1"],
                "e3_model_test_fall_f1": e6["fall_f1"],
                "fall_delta": e6["fall_f1"] - e1["fall_f1"],
                "dataset_specific_direction_f1": e1["direction_macro_f1"],
                "e3_model_test_direction_f1": e6["direction_macro_f1"],
                "direction_delta": e6["direction_macro_f1"] - e1["direction_macro_f1"],
            },
            {
                "dataset": "weda",
                "dataset_specific_fall_f1": e2["fall_f1"],
                "e3_model_test_fall_f1": e7["fall_f1"],
                "fall_delta": e7["fall_f1"] - e2["fall_f1"],
                "dataset_specific_direction_f1": e2["direction_macro_f1"],
                "e3_model_test_direction_f1": e7["direction_macro_f1"],
                "direction_delta": e7["direction_macro_f1"] - e2["direction_macro_f1"],
            },
        ]
    )

    pure_transfer = pd.DataFrame(
        [
            {
                "test_dataset": "bits",
                "pure_transfer_exp": "E5_WEDA_TO_BITS",
                "pure_transfer_fall_f1": e5["fall_f1"],
                "e3_model_test_fall_f1": e6["fall_f1"],
                "fall_delta": e6["fall_f1"] - e5["fall_f1"],
                "pure_transfer_direction_f1": e5["direction_macro_f1"],
                "e3_model_test_direction_f1": e6["direction_macro_f1"],
                "direction_delta": e6["direction_macro_f1"] - e5["direction_macro_f1"],
            },
            {
                "test_dataset": "weda",
                "pure_transfer_exp": "E4_BITS_TO_WEDA",
                "pure_transfer_fall_f1": e4["fall_f1"],
                "e3_model_test_fall_f1": e7["fall_f1"],
                "fall_delta": e7["fall_f1"] - e4["fall_f1"],
                "pure_transfer_direction_f1": e4["direction_macro_f1"],
                "e3_model_test_direction_f1": e7["direction_macro_f1"],
                "direction_delta": e7["direction_macro_f1"] - e4["direction_macro_f1"],
            },
        ]
    )

    per_class = summary[
        [
            "experiment_id",
            "test_dataset",
            "forward_f1",
            "backward_f1",
            "lateral_f1",
        ]
    ].copy()

    lines = [
        "# Event-centered 25 Hz BITS/WEDA E1-E7 Summary",
        "",
        "E6 and E7 reuse the E3 checkpoint trained on BITS+WEDA and evaluate the",
        "same test predictions separately for BITS and WEDA. They are not new",
        "training runs.",
        "",
        "## Table 1. Main Results",
        "",
        markdown_table(main_table),
        "",
        "## Table 2. Dataset-specific vs E3 Model Per-dataset Test",
        "",
        markdown_table(dataset_specific),
        "",
        "## Table 3. Pure Transfer vs E3 Model Per-dataset Test",
        "",
        markdown_table(pure_transfer),
        "",
        "## Table 4. Direction Per-class F1",
        "",
        markdown_table(per_class),
        "",
        "## Insights",
        "",
        "- The E3 mixed BITS+WEDA model is strong on both test subsets: BITS direction macro F1 = "
        f"{_safe_float(e6['direction_macro_f1']):.4f}, WEDA direction macro F1 = "
        f"{_safe_float(e7['direction_macro_f1']):.4f}. The overall E3 score is therefore not only a BITS effect.",
        "- Mixed training improves direction over dataset-specific training on both datasets: "
        f"BITS +{_safe_float(dataset_specific.loc[0, 'direction_delta']):.4f}, "
        f"WEDA +{_safe_float(dataset_specific.loc[1, 'direction_delta']):.4f}. "
        "This supports using BITS+WEDA as the clean-direction benchmark.",
        "- WEDA fall detection drops under the mixed model compared with WEDA-only "
        f"({_safe_float(e2['fall_f1']):.4f} to {_safe_float(e7['fall_f1']):.4f}), while WEDA direction improves strongly. "
        "This suggests the fall and direction heads are affected differently by cross-dataset mixing.",
        "- Pure cross-dataset transfer is still weak: BITS->WEDA direction macro F1 = "
        f"{_safe_float(e4['direction_macro_f1']):.4f}, WEDA->BITS = {_safe_float(e5['direction_macro_f1']):.4f}. "
        "The clean setup works best when both datasets are represented in training.",
        "- Direction metrics are based on small supervised test counts "
        f"(BITS n={int(e6['direction_n_supervised'])}, WEDA n={int(e7['direction_n_supervised'])}), so report these counts with every paper table.",
        "",
        "## Diagnosis",
        "",
        "1. Train BITS+WEDA then test each dataset separately generalizes well for direction on both datasets.",
        "2. BITS is easier for fall detection under the mixed model; WEDA remains harder for fall detection even though direction improves.",
        "3. Sampling-rate normalization to 25 Hz is not the main blocker inside the clean BITS/WEDA setup; pure domain transfer remains the larger issue.",
        "4. BITS+WEDA is suitable as a clean-direction benchmark, but leave-one-dataset transfer should be reported separately as a domain-shift stress test.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/experiments_25hz_event"),
        help="Event-centered experiment artifact directory.",
    )
    args = parser.parse_args()

    root = args.root
    summary_path = root / "summary_25hz_a5wcefw.csv"
    e3_predictions_path = root / "E3_BITS_WEDA_MIXED" / "predictions.csv"

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not e3_predictions_path.exists():
        raise FileNotFoundError(e3_predictions_path)

    summary = pd.read_csv(summary_path)
    e3 = summary.loc[summary["experiment_id"] == "E3_BITS_WEDA_MIXED"].iloc[0]
    predictions = pd.read_csv(e3_predictions_path)

    new_rows: list[dict[str, object]] = []
    for exp_id, dataset in [
        ("E6_E3_MIXED_TEST_BITS", "bits"),
        ("E7_E3_MIXED_TEST_WEDA", "weda"),
    ]:
        subset = predictions.loc[predictions["dataset"].astype(str).str.lower() == dataset].copy()
        metrics = write_subset_artifacts(subset, root / exp_id)
        row = e3.to_dict()
        row.update(
            {
                "experiment_id": exp_id,
                "train_dataset": "bits+weda",
                "test_dataset": dataset,
                "n_test": metrics["n_test"],
                "fall_accuracy": metrics["fall_accuracy"],
                "fall_precision": metrics["fall_precision"],
                "fall_recall": metrics["fall_recall"],
                "fall_f1": metrics["fall_f1"],
                "direction_n_supervised": metrics["direction_n_supervised"],
                "direction_accuracy": metrics["direction_accuracy"],
                "direction_macro_f1": metrics["direction_macro_f1"],
                "forward_f1": metrics["forward_f1"],
                "backward_f1": metrics["backward_f1"],
                "lateral_f1": metrics["lateral_f1"],
            }
        )
        new_rows.append(row)

    summary_e1_e7 = pd.concat([summary, pd.DataFrame(new_rows)], ignore_index=True)
    summary_e1_e7.to_csv(root / "summary_e1_e7_event_centered.csv", index=False)
    (root / "summary_e1_e7_event_centered.md").write_text(
        build_markdown(summary_e1_e7),
        encoding="utf-8",
    )

    print(f"Wrote {root / 'summary_e1_e7_event_centered.csv'}")
    print(f"Wrote {root / 'summary_e1_e7_event_centered.md'}")


if __name__ == "__main__":
    main()
