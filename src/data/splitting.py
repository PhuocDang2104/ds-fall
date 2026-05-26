from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


def _split_subjects(subjects: list[str], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    subjects = list(subjects)
    rng.shuffle(subjects)
    n = len(subjects)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n == 1:
        return {"train": subjects, "val": [], "test": []}
    if n == 2:
        return {"train": subjects[:1], "val": [], "test": subjects[1:]}

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = max(1, n - n_val - n_test)

    return {
        "train": subjects[:n_train],
        "val": subjects[n_train : n_train + n_val],
        "test": subjects[n_train + n_val :],
    }


def subject_wise_split(
    metadata: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    metadata = metadata.copy()
    metadata["split"] = "unassigned"
    split_subjects: dict[str, Any] = {}

    for dataset in sorted(metadata["dataset"].dropna().unique()):
        mask = metadata["dataset"] == dataset
        subjects = sorted(metadata.loc[mask, "subject_id"].astype(str).unique())
        dataset_offset = sum(ord(ch) for ch in str(dataset))
        parts = _split_subjects(subjects, train_ratio, val_ratio, seed + dataset_offset)
        split_subjects[dataset] = parts
        for split_name, split_subjs in parts.items():
            split_mask = mask & metadata["subject_id"].astype(str).isin(split_subjs)
            metadata.loc[split_mask, "split"] = split_name

    unassigned = metadata["split"].eq("unassigned").sum()
    if unassigned:
        raise RuntimeError(f"{unassigned} samples were not assigned to a split")

    _assert_no_subject_leakage(metadata)
    return metadata, split_subjects


def _assert_no_subject_leakage(metadata: pd.DataFrame) -> None:
    seen: dict[tuple[str, str], str] = {}
    conflicts: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for _, row in metadata.iterrows():
        key = (str(row["dataset"]), str(row["subject_id"]))
        split = str(row["split"])
        if key in seen and seen[key] != split:
            conflicts[key].update([seen[key], split])
        seen[key] = split
    if conflicts:
        raise RuntimeError(f"Subject leakage detected: {dict(conflicts)}")
