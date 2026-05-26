from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.utils.io import ensure_dir
from src.utils.metrics import classification_metrics


def _predict_outputs(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    preds = model.predict(X, verbose=0)
    if isinstance(preds, dict):
        fall_probs = preds["fall_output"]
        direction_probs = preds["direction_output"]
    else:
        fall_probs, direction_probs = preds
    return fall_probs, direction_probs


def evaluate_fall_detection(model, X: np.ndarray, y_fall: np.ndarray) -> dict[str, Any]:
    fall_probs, _ = _predict_outputs(model, X)
    y_pred = np.argmax(fall_probs, axis=1)
    metrics = classification_metrics(y_fall, y_pred, average="binary")
    metrics["confusion_matrix"] = confusion_matrix(y_fall, y_pred, labels=[0, 1]).tolist()
    metrics["classification_report"] = classification_report(
        y_fall,
        y_pred,
        labels=[0, 1],
        target_names=["non_fall", "fall"],
        zero_division=0,
        output_dict=True,
    )
    return metrics


def evaluate_direction(
    model,
    X: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
) -> dict[str, Any]:
    supervised = (direction_mask.astype(bool)) & (y_direction >= 0)
    if supervised.sum() == 0:
        return {"num_supervised": 0}
    _, direction_probs = _predict_outputs(model, X[supervised])
    y_true = y_direction[supervised]
    y_pred = np.argmax(direction_probs, axis=1)
    return {
        "num_supervised": int(supervised.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=["forward", "backward", "lateral"],
            zero_division=0,
            output_dict=True,
        ),
    }


def evaluate_per_dataset(
    model,
    X: np.ndarray,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in sorted(metadata["dataset"].unique()):
        mask = metadata["dataset"].to_numpy() == dataset
        if mask.sum() == 0:
            continue
        out[dataset] = {
            "fall": evaluate_fall_detection(model, X[mask], y_fall[mask]),
            "direction": evaluate_direction(model, X[mask], y_direction[mask], direction_mask[mask]),
        }
    if "bits" in out:
        out["bits"]["bits_preprocessing"] = "20Hz row-order uniform interpolation to 50Hz"
    return out


def plot_confusion_matrix(
    cm: np.ndarray | list[list[int]],
    labels: list[str],
    title: str,
    save_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.asarray(cm), annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def save_metrics_json(path: str | Path, metrics: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(metrics), f, indent=2)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(_jsonable(k)): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
