from __future__ import annotations

import json
import time
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


def evaluate_all(
    model,
    X: np.ndarray,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    metadata: pd.DataFrame | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "fall": evaluate_fall_detection(model, X, y_fall),
        "direction": evaluate_direction(model, X, y_direction, direction_mask),
    }
    if metadata is not None:
        metrics["per_dataset"] = evaluate_per_dataset(model, X, y_fall, y_direction, direction_mask, metadata)
        metrics["per_subject"] = evaluate_per_subject(model, X, y_fall, y_direction, direction_mask, metadata)
    metrics["model_size"] = model_size_summary(model)
    return metrics


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
    if "umafall" in out:
        out["umafall"]["umafall_preprocessing"] = "20Hz timestamp interpolation to 50Hz using WRIST SensorTag"
    return out


def evaluate_per_subject(
    model,
    X: np.ndarray,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    if "subject_id" not in metadata:
        return {}
    meta = metadata.reset_index(drop=True)
    group_cols = ["subject_id"]
    if "dataset" in meta:
        group_cols = ["dataset", "subject_id"]

    out: dict[str, Any] = {}
    for key, group in meta.groupby(group_cols, sort=True):
        idx = group.index.to_numpy()
        if len(idx) == 0:
            continue
        if isinstance(key, tuple):
            out_key = "/".join(str(part) for part in key)
        else:
            out_key = str(key)
        out[out_key] = {
            "num_windows": int(len(idx)),
            "fall": evaluate_fall_detection(model, X[idx], y_fall[idx]),
            "direction": evaluate_direction(model, X[idx], y_direction[idx], direction_mask[idx]),
        }
    return out


def plot_confusion_matrix(
    cm: np.ndarray | list[list[int]],
    labels: list[str],
    title: str,
    save_path: str | Path | None = None,
    show: bool = True,
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
    if show:
        plt.show()
    else:
        plt.close(fig)


def model_size_summary(model) -> dict[str, Any]:
    params = int(model.count_params())
    fp32_bytes = params * 4
    int8_bytes = params
    return {
        "params": params,
        "fp32_bytes": fp32_bytes,
        "fp32_kb": fp32_bytes / 1024.0,
        "estimated_int8_bytes": int8_bytes,
        "estimated_int8_kb": int8_bytes / 1024.0,
    }


def estimate_inference_latency_ms(
    model,
    X: np.ndarray,
    warmup: int = 3,
    repeats: int = 30,
) -> float | None:
    if len(X) == 0:
        return None
    sample = np.asarray(X[:1], dtype=np.float32)
    for _ in range(warmup):
        model.predict(sample, verbose=0)
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(sample, verbose=0)
    elapsed = time.perf_counter() - start
    return float((elapsed / repeats) * 1000.0)


def flatten_ablation_metrics(
    ablation_id: str,
    spec: dict[str, Any],
    metrics: dict[str, Any],
    latency_ms: float | None = None,
) -> dict[str, Any]:
    fall = metrics.get("fall", {})
    direction = metrics.get("direction", {})
    size = metrics.get("model_size", {})
    row: dict[str, Any] = {
        "id": ablation_id,
        "model": spec.get("model_name"),
        "feature_set": spec.get("feature_set"),
        "direction_loss_type": spec.get("direction_loss_type"),
        "lambda_direction": spec.get("lambda_direction"),
        "fall_loss_weighted": spec.get("fall_loss_weighted"),
        "augment": spec.get("augment"),
        "fall_accuracy": fall.get("accuracy"),
        "fall_precision": fall.get("precision"),
        "fall_recall": fall.get("recall"),
        "fall_f1": fall.get("f1"),
        "direction_num_supervised": direction.get("num_supervised", 0),
        "direction_accuracy": direction.get("accuracy"),
        "direction_macro_f1": direction.get("macro_f1"),
        "params": size.get("params"),
        "fp32_kb": size.get("fp32_kb"),
        "estimated_int8_kb": size.get("estimated_int8_kb"),
        "inference_latency_ms": latency_ms,
    }
    for dataset, result in metrics.get("per_dataset", {}).items():
        row[f"{dataset}_fall_f1"] = result.get("fall", {}).get("f1")
        row[f"{dataset}_direction_macro_f1"] = result.get("direction", {}).get("macro_f1")
        row[f"{dataset}_direction_num_supervised"] = result.get("direction", {}).get("num_supervised", 0)
    report = direction.get("classification_report", {})
    for label in ["forward", "backward", "lateral"]:
        if label in report:
            row[f"direction_{label}_precision"] = report[label].get("precision")
            row[f"direction_{label}_recall"] = report[label].get("recall")
            row[f"direction_{label}_f1"] = report[label].get("f1-score")
    return row


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
