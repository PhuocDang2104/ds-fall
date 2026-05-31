from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


UNIT_METADATA_FIELDS = [
    "acc_unit_before",
    "gyro_unit_before",
    "acc_unit_after",
    "gyro_unit_after",
    "conversion_factor_acc",
    "conversion_factor_gyro",
]


def harmonize_record_units(
    records: list[dict[str, Any]],
    dataset_unit_map: dict[str, dict[str, float | str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert record X arrays to common physical units before scaling/features."""
    converted: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    warnings: list[str] = []

    for rec in records:
        dataset = str(rec.get("dataset", "")).lower()
        unit_cfg = dataset_unit_map.get(dataset)
        if unit_cfg is None:
            warnings.append(f"No unit map for dataset {dataset!r}; leaving record unchanged.")
            unit_cfg = {
                "acc_unit_before": "unknown",
                "gyro_unit_before": "unknown",
                "acc_unit_after": "unknown",
                "gyro_unit_after": "unknown",
                "conversion_factor_acc": 1.0,
                "conversion_factor_gyro": 1.0,
            }

        out = dict(rec)
        X = np.asarray(out["X"], dtype=np.float32).copy()
        acc_factor = float(unit_cfg.get("conversion_factor_acc", 1.0))
        gyro_factor = float(unit_cfg.get("conversion_factor_gyro", 1.0))
        X[:, :3] *= np.float32(acc_factor)
        X[:, 3:6] *= np.float32(gyro_factor)
        out["X"] = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        out["dataset_name"] = dataset
        for field in UNIT_METADATA_FIELDS:
            out[field] = unit_cfg.get(field)
        out["unit_harmonized"] = True
        counts[dataset] += 1
        converted.append(out)

    summary = {
        "target_acc_unit": "m/s^2",
        "target_gyro_unit": "rad/s",
        "records_by_dataset": dict(counts),
        "dataset_unit_map": dataset_unit_map,
        "warnings": sorted(set(warnings)),
    }
    return converted, summary


def unit_harmonization_markdown(metadata: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines = [
        "# Unit Harmonization Report",
        "",
        "Raw windows are converted before train-set normalization and before DS-Fall-RD feature engineering.",
        "",
        f"- Target accelerometer unit: `{summary.get('target_acc_unit')}`",
        f"- Target gyroscope unit: `{summary.get('target_gyro_unit')}`",
        "",
        "## Dataset Unit Map",
        "",
        dataframe_to_markdown(_unit_map_dataframe(summary.get("dataset_unit_map", {}))),
        "",
        "## Metadata Counts",
        "",
    ]
    if metadata.empty or "dataset" not in metadata:
        lines.append("_No metadata rows._")
    else:
        cols = ["dataset"] + [col for col in UNIT_METADATA_FIELDS if col in metadata]
        table = metadata[cols].drop_duplicates().sort_values("dataset").reset_index(drop=True)
        lines.append(dataframe_to_markdown(table))
    lines.extend(["", "## Warnings", ""])
    warnings = summary.get("warnings") or []
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _unit_map_dataframe(unit_map: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for dataset, cfg in sorted(unit_map.items()):
        row = {"dataset": dataset}
        row.update({field: cfg.get(field) for field in UNIT_METADATA_FIELDS})
        rows.append(row)
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    compact = df.copy()
    for col in compact.columns:
        if pd.api.types.is_float_dtype(compact[col]):
            compact[col] = compact[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    header = "| " + " | ".join(map(str, compact.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(compact.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in compact.to_numpy()]
    return "\n".join([header, sep] + rows)
