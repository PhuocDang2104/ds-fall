from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

import run_a5_direction_vs_ldx1_multiseed as edgebase
import run_final_compact_hybrid as final
import run_hybrid_feature_reduction_and_lightweight_direction as hybrid


FOCUS_PROTOCOLS = ["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved DS-Fall-RD Compact Hybrid artifacts.")
    parser.add_argument("--repo-root", default=".")
    return parser.parse_args()


def load_direction_run(model_path: Path) -> edgebase.DirectionRun:
    if not edgebase.TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to evaluate the saved LDX1 artifact.")
    checkpoint = edgebase.torch.load(model_path, map_location="cpu")
    spec = [s for s in edgebase.model_specs() if s["model_name"] == checkpoint["model_name"]][0]
    model = spec["builder"]()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return edgebase.DirectionRun(
        model=model,
        model_name=checkpoint["model_name"],
        params=int(checkpoint["params"]),
        protocol_id="E3_BITS_WEDA_MIXED",
        seed=int(checkpoint["seed"]),
        learning_rate=float(checkpoint["learning_rate"]),
        best_epoch=-1,
        best_val_macro_f1=np.nan,
        best_val_loss=np.nan,
        scaler=checkpoint["scaler"],
        complexity=edgebase.estimate_complexity(checkpoint["model_name"], int(checkpoint["params"])),
        architecture_note=spec["architecture_note"],
    )


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    cols = list(out.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines)


def write_eval_report(report_dir: Path, config: dict[str, Any], fall_df: pd.DataFrame, direction_df: pd.DataFrame, e2e_df: pd.DataFrame) -> None:
    lines = [
        "# Final Compact Hybrid Evaluation",
        "",
        "This report is regenerated from saved artifacts, not from model retraining.",
        "",
        "## Artifact Config",
        "",
        f"- Model: `{config['model_name']}`",
        f"- Selected seed: `{config['selected_seed']}`",
        f"- Selected learning rate: `{config['selected_learning_rate']}`",
        f"- Direction params: `{config['direction_expert']['params']}`",
        f"- Direction input: `{config['direction_expert']['input_shape']}`",
        f"- Fall feature count: `{config['fall_expert']['feature_count']}`",
        f"- Fall threshold: `{config['fall_expert']['threshold']:.6f}`",
        "",
        "## Fall Metrics",
        "",
        md_table(fall_df[["experiment_id", "test_dataset", "fall_precision", "fall_recall", "fall_f1", "fall_accuracy", "TN", "FP", "FN", "TP"]]),
        "",
        "## Direction Metrics",
        "",
        md_table(direction_df[["experiment_id", "test_dataset", "direction_macro_f1", "direction_accuracy", "forward_f1", "backward_f1", "lateral_f1", "direction_n_supervised"]]),
        "",
        "## End-to-End Metrics",
        "",
        md_table(e2e_df[["experiment_id", "E2E_Direction_Macro_F1", "coverage", "correct_rate", "direction_n_supervised"]]),
        "",
    ]
    (report_dir / "final_compact_hybrid_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_cm(cm: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.8, 4.0))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_confusion_figures(figure_dir: Path, fall_df: pd.DataFrame, direction_df: pd.DataFrame) -> None:
    cm_dir = figure_dir / "confusion_matrices"
    for _, row in fall_df.iterrows():
        cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]], dtype=int)
        plot_cm(cm, ["nonfall", "fall"], f"Fall CM - {row['experiment_id']}", cm_dir / f"{row['experiment_id']}_fall.png")
    for _, row in direction_df.iterrows():
        cm = np.array(
            [
                [row["cm_00"], row["cm_01"], row["cm_02"]],
                [row["cm_10"], row["cm_11"], row["cm_12"]],
                [row["cm_20"], row["cm_21"], row["cm_22"]],
            ],
            dtype=int,
        )
        plot_cm(cm, ["forward", "backward", "lateral"], f"Direction CM - {row['experiment_id']}", cm_dir / f"{row['experiment_id']}_direction.png")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    report_dir = repo_root / "outputs" / "reports" / "final_compact_hybrid"
    figure_dir = repo_root / "outputs" / "figures" / "final_compact_hybrid"
    model_dir = repo_root / "outputs" / "models" / "final_compact_hybrid"
    config = json.loads((model_dir / "final_config.json").read_text(encoding="utf-8"))
    df, X_temporal = final.load_data(repo_root)
    with (model_dir / "statistical_fall_expert.pkl").open("rb") as f:
        fall_expert = pickle.load(f)
    direction_run = load_direction_run(model_dir / "ldx1_wide_dsconv_direction.pt")

    fall_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    e2e_rows: list[dict[str, Any]] = []
    for protocol_id in FOCUS_PROTOCOLS:
        mask, fall_pred, fall_scores = final.fall_predictions_for_protocol(df, fall_expert, protocol_id)
        d_pred = edgebase.predict_direction(direction_run, X_temporal, mask)
        fall_rows.append(final.evaluate_fall(df, mask, fall_pred, fall_scores, "DS-Fall-RD Compact Hybrid", protocol_id, len(fall_expert["features"])))
        direction_rows.append(final.evaluate_direction(df, mask, d_pred, "DS-Fall-RD Compact Hybrid", protocol_id, direction_run.params))
        e2e_rows.append(final.evaluate_e2e(df, mask, fall_pred, d_pred, "DS-Fall-RD Compact Hybrid", protocol_id))

    fall_df = pd.DataFrame(fall_rows)
    direction_df = pd.DataFrame(direction_rows)
    e2e_df = pd.DataFrame(e2e_rows)
    fall_df.to_csv(report_dir / "final_compact_hybrid_fall_metrics_recomputed.csv", index=False)
    direction_df.to_csv(report_dir / "final_compact_hybrid_direction_metrics_recomputed.csv", index=False)
    e2e_df.to_csv(report_dir / "final_compact_hybrid_e2e_metrics_recomputed.csv", index=False)
    write_eval_report(report_dir, config, fall_df, direction_df, e2e_df)
    save_confusion_figures(figure_dir, fall_df, direction_df)
    print("[evaluate] done")
    print(direction_df[["experiment_id", "direction_macro_f1", "direction_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
