from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


START_MARKER = "<!-- AUTO_CROSS_DATASET_ANALYSIS_START -->"
END_MARKER = "<!-- AUTO_CROSS_DATASET_ANALYSIS_END -->"

DEFAULT_REPORT_FILENAMES = [
    "domain_conflict_experiments.csv",
    "train2_test1_bits_weda_hifd.csv",
    "clean_bits_weda_experiments.csv",
]

POSSIBLE_CAUSES = [
    "Sampling protocol mismatch.",
    "Sensor placement / device mismatch.",
    "Activity definition mismatch.",
    "Direction label mismatch.",
    "Subject movement style mismatch.",
    "Fall impact interval annotation mismatch.",
    "BITS 20 Hz interpolation to 25 Hz may smooth impact peaks.",
    "WEDA 50 Hz downsampling to 25 Hz may reduce high-frequency impact detail.",
]


def update_analysis_notes_from_reports(
    project_root: str | Path,
    output_dir: str | Path | None = None,
    report_paths: list[str | Path] | None = None,
    direction_threshold: float = 0.60,
    fall_threshold: float = 0.70,
) -> Path:
    """Create/update analysis_notes.md with automatic cross-dataset failure notes."""
    root = Path(project_root)
    reports = _resolve_report_paths(root, output_dir, report_paths)
    rows = _collect_cross_dataset_rows(reports, root)
    low_rows = [
        row
        for row in rows
        if _is_low(row.get("direction_macro_f1"), direction_threshold)
        or _is_low(row.get("fall_f1"), fall_threshold)
    ]

    section = build_cross_dataset_section(
        rows=rows,
        low_rows=low_rows,
        direction_threshold=direction_threshold,
        fall_threshold=fall_threshold,
    )
    notes_path = root / "analysis_notes.md"
    existing = notes_path.read_text(encoding="utf-8") if notes_path.exists() else "# Analysis Notes\n\n"
    notes_path.write_text(_replace_section(existing, section), encoding="utf-8")
    return notes_path


def build_cross_dataset_section(
    rows: list[dict[str, Any]],
    low_rows: list[dict[str, Any]],
    direction_threshold: float,
    fall_threshold: float,
) -> str:
    lines = [
        START_MARKER,
        "## Automatic Cross-Dataset Analysis",
        "",
        "This section is generated from cross-dataset result CSV files. It is a hypothesis list for debugging, not a confirmed causal analysis.",
        "",
        (
            f"Low-performance rule: `direction_macro_f1 < {direction_threshold:.2f}` "
            f"or `fall_f1 < {fall_threshold:.2f}` on cross-dataset rows."
        ),
        "",
    ]

    if not rows:
        lines.extend(
            [
                "Status: no cross-dataset result rows were found.",
                "",
                "Possible causes to check once cross-dataset runs are available:",
                "",
            ]
        )
        lines.extend([f"{idx}. {cause}" for idx, cause in enumerate(POSSIBLE_CAUSES, start=1)])
        lines.extend(["", END_MARKER])
        return "\n".join(lines) + "\n"

    if low_rows:
        lines.extend(["Status: low cross-dataset performance detected.", "", "Low rows:", "", _rows_table(low_rows), ""])
        lines.extend(["Possible causes:", ""])
        lines.extend([f"{idx}. {cause}" for idx, cause in enumerate(POSSIBLE_CAUSES, start=1)])
        lines.extend(
            [
                "",
                "Recommended checks:",
                "",
                "- Verify resampling is done per trial and preserves acc/gyro alignment.",
                "- Compare per-dataset axis semantics using signed gyro, roll, pitch, and impact-centered curves.",
                "- Recheck direction label definitions and whether lateral merges left/right patterns.",
                "- Inspect impact index and fall interval annotation consistency before changing the model.",
            ]
        )
    else:
        lines.extend(["Status: no low cross-dataset rows were detected with the current thresholds.", "", "Checked rows:", "", _rows_table(rows), ""])

    lines.append(END_MARKER)
    return "\n".join(lines) + "\n"


def _resolve_report_paths(
    root: Path,
    output_dir: str | Path | None,
    report_paths: list[str | Path] | None,
) -> list[Path]:
    if report_paths:
        return [Path(path) if Path(path).is_absolute() else root / path for path in report_paths]

    out_dir = Path(output_dir) if output_dir is not None else root / "outputs"
    reports_dir = out_dir / "reports"
    paths = [reports_dir / name for name in DEFAULT_REPORT_FILENAMES]
    artifacts_dir = root / "artifacts" / "experiments"
    if artifacts_dir.exists():
        paths.extend(sorted(artifacts_dir.glob("*summary*.csv")))
    return paths


def _collect_cross_dataset_rows(report_paths: list[Path], root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in report_paths:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        for _, item in df.iterrows():
            row = _extract_row(path, item, root)
            if row is not None:
                rows.append(row)
    return rows


def _extract_row(path: Path, item: pd.Series, root: Path) -> dict[str, Any] | None:
    experiment_type = str(item.get("experiment_type", "")).lower()
    if experiment_type == "dataset_specific":
        return None

    train = _first_present(item, ["train_datasets", "train_dataset", "train"])
    test = _first_present(item, ["test_datasets", "test_dataset", "test"])
    exp_id = _first_present(item, ["experiment_id", "id"])
    if train is None or test is None or exp_id is None:
        return None

    train_s = str(train)
    test_s = str(test)
    if train_s == test_s and "cross" not in experiment_type and "test1" not in experiment_type:
        return None

    direction_value = _first_present(item, ["direction_macro_f1", "final_direction_macro_f1"])
    if direction_value is None and "fall_f1" not in item:
        return None

    return {
        "source": _display_path(path, root),
        "experiment_id": str(exp_id),
        "experiment_type": experiment_type or "unknown",
        "train": train_s,
        "test": test_s,
        "fall_f1": _to_float(item.get("fall_f1")),
        "direction_macro_f1": _to_float(direction_value),
        "direction_accuracy": _to_float(item.get("direction_accuracy")),
        "direction_n": _to_float(_first_present(item, ["direction_num_supervised", "final_direction_num_supervised", "direction_n"])),
    }


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _first_present(item: pd.Series, keys: list[str]) -> Any | None:
    for key in keys:
        if key in item and not pd.isna(item.get(key)):
            return item.get(key)
    return None


def _to_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_low(value: Any, threshold: float) -> bool:
    numeric = _to_float(value)
    return numeric is not None and numeric < threshold


def _rows_table(rows: list[dict[str, Any]]) -> str:
    table = pd.DataFrame(rows)
    if table.empty:
        return "_No rows._"
    show_cols = [
        "experiment_id",
        "experiment_type",
        "train",
        "test",
        "fall_f1",
        "direction_macro_f1",
        "direction_accuracy",
        "direction_n",
        "source",
    ]
    compact = table[[col for col in show_cols if col in table.columns]].copy()
    for col in ["fall_f1", "direction_macro_f1", "direction_accuracy", "direction_n"]:
        if col in compact:
            compact[col] = compact[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    header = "| " + " | ".join(compact.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(compact.columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in compact.to_numpy()]
    return "\n".join([header, sep] + body)


def _replace_section(existing: str, section: str) -> str:
    if START_MARKER in existing and END_MARKER in existing:
        before = existing.split(START_MARKER, 1)[0].rstrip()
        after = existing.split(END_MARKER, 1)[1].lstrip()
        return f"{before}\n\n{section}\n{after}".rstrip() + "\n"
    return existing.rstrip() + "\n\n" + section
