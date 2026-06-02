from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import DIRECTION_LABEL_MAPPING, FALL_LABEL_MAPPING, ProjectConfig
from src.data.features import (
    compute_imu_features,
    direction_summary_features,
    feature_channel_names,
    fit_feature_scaler,
    fit_summary_scaler,
    transform_summary_features,
    transform_with_feature_scaler,
)
from src.experiments.ablation_rd import load_processed_training_data
from src.models.losses import compute_class_weights, make_fall_loss, make_sparse_ce_loss
from src.training.evaluate import model_size_summary, plot_confusion_matrix, save_metrics_json
from src.utils.io import ensure_dir, load_json, save_json, save_pickle
from src.utils.metrics import classification_metrics
from src.utils.seed import set_seed


PLANE_LABEL_MAPPING = {"sagittal": 0, "lateral": 1}
SAGITTAL_LABEL_MAPPING = {"forward": 0, "backward": 1}


@dataclass(frozen=True)
class HierarchicalSpec:
    id: str
    model_name: str
    feature_set: str = "tilt12"
    use_summary_branch: bool = False
    lambda_plane: float = 1.0
    lambda_sagittal: float = 1.0
    fall_loss_weighted: bool = True


def default_hierarchical_specs() -> list[HierarchicalSpec]:
    return [
        HierarchicalSpec("B1", "A10_HIER_DIR", use_summary_branch=False),
        HierarchicalSpec("B2", "A11_HIER_DIR_SUMMARY", use_summary_branch=True),
    ]


def run_hierarchical_direction_suite(
    config: ProjectConfig,
    ablation_ids: list[str] | None = None,
    epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    processed_x_is_normalized: bool = True,
    include_b0: bool = True,
) -> pd.DataFrame:
    set_seed(config.seed)
    data = load_processed_training_data(config.processed_dir, processed_x_is_normalized=processed_x_is_normalized)
    X_raw6 = data["X_raw6"]
    metadata = data["metadata"]
    y_fall = data["y_fall"]
    y_direction = data["y_direction"]
    direction_mask = data["direction_mask"]

    wanted = {item.upper() for item in ablation_ids} if ablation_ids else {"B0", "B1", "B2"}
    rows: list[dict[str, Any]] = []

    if include_b0 and "B0" in wanted:
        b0_row = load_b0_a5wcefw_row(config.output_dir)
        if b0_row is not None:
            rows.append(b0_row)
            save_b0_confusion_matrices(config.output_dir)
        else:
            print("B0 A5WCEFW metrics not found; run scripts/run_ds_fall_rd_ablation.py --ablation A5WCEFW first.")

    specs = [spec for spec in default_hierarchical_specs() if spec.id.upper() in wanted]
    missing = sorted((wanted - {"B0"}) - {spec.id.upper() for spec in specs})
    if missing:
        raise ValueError(f"Unknown hierarchical ablation ids: {missing}")

    save_hierarchical_results(rows, config.output_dir)
    for spec in specs:
        row = run_single_hierarchical_ablation(
            spec,
            config=config,
            X_raw6=X_raw6,
            metadata=metadata,
            y_fall=y_fall,
            y_direction=y_direction,
            direction_mask=direction_mask,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        rows.append(row)
        save_hierarchical_results(rows, config.output_dir)

    results = pd.DataFrame(rows)
    save_hierarchical_model_selection_report(results, config.output_dir)
    return results


def run_single_hierarchical_ablation(
    spec: HierarchicalSpec,
    config: ProjectConfig,
    X_raw6: np.ndarray,
    metadata: pd.DataFrame,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    from src.models.ds_fall import build_ds_fall_hier_dir_model

    set_seed(config.seed)
    run_name = f"{spec.id}_{spec.model_name}_{spec.feature_set}"
    run_dir = ensure_dir(config.output_dir / "runs" / run_name)
    print(
        f"\n=== {spec.id}: model={spec.model_name}, feature_set={spec.feature_set}, "
        f"summary_branch={spec.use_summary_branch} ==="
    )

    splits = prepare_hierarchical_splits(
        X_raw6,
        metadata,
        y_fall,
        y_direction,
        direction_mask,
        spec=spec,
        fs=config.model_fs,
    )
    train = splits["train"]
    val = splits["val"]
    test = splits["test"]

    fall_class_weights = compute_class_weights(train["y_fall"], num_classes=2) if spec.fall_loss_weighted else None
    plane_class_weights = compute_class_weights(train["y_plane"], num_classes=2, mask=train["plane_mask"])
    sagittal_class_weights = compute_class_weights(train["y_sagittal"], num_classes=2, mask=train["sagittal_mask"])

    input_shape = (train["X"].shape[1], train["X"].shape[2])
    summary_shape = (train["X_summary"].shape[1],) if spec.use_summary_branch else (12,)
    model = build_ds_fall_hier_dir_model(
        input_shape=input_shape,
        feature_set=spec.feature_set,
        use_summary_branch=spec.use_summary_branch,
        summary_shape=summary_shape,
        show_summary=False,
    )
    compile_hierarchical_model(
        model,
        learning_rate=learning_rate,
        fall_class_weights=fall_class_weights,
        plane_class_weights=plane_class_weights,
        sagittal_class_weights=sagittal_class_weights,
        lambda_plane=spec.lambda_plane,
        lambda_sagittal=spec.lambda_sagittal,
    )

    train_ds = make_hierarchical_tf_dataset(
        train,
        batch_size=batch_size,
        shuffle=True,
        seed=config.seed,
        use_summary_branch=spec.use_summary_branch,
    )
    val_ds = make_hierarchical_tf_dataset(
        val,
        batch_size=batch_size,
        shuffle=False,
        seed=config.seed,
        use_summary_branch=spec.use_summary_branch,
    )
    callbacks = build_hierarchical_callbacks(
        run_dir,
        val_inputs=make_model_inputs(val, spec.use_summary_branch),
        val=val,
        patience=14,
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)

    model.save(run_dir / "model_final.keras")
    save_json(run_dir / "history.json", history.history)
    save_json(
        run_dir / "feature_config.json",
        {"feature_set": spec.feature_set, "channel_names": feature_channel_names(spec.feature_set)},
    )
    save_json(run_dir / "model_config.json", asdict(spec) | {"input_shape": list(input_shape), "learning_rate": learning_rate})
    save_json(run_dir / "feature_scaler.json", splits["feature_scaler"])
    save_pickle(run_dir / "feature_scaler.pkl", splits["feature_scaler"])
    if spec.use_summary_branch:
        save_json(run_dir / "summary_scaler.json", splits["summary_scaler"])
        save_pickle(run_dir / "summary_scaler.pkl", splits["summary_scaler"])
    save_json(
        run_dir / "label_mapping.json",
        {
            "fall": FALL_LABEL_MAPPING,
            "direction": DIRECTION_LABEL_MAPPING,
            "plane": PLANE_LABEL_MAPPING,
            "sagittal": SAGITTAL_LABEL_MAPPING,
        },
    )
    save_json(
        run_dir / "class_weights.json",
        {
            "fall": fall_class_weights.tolist() if fall_class_weights is not None else None,
            "plane": plane_class_weights.tolist(),
            "sagittal": sagittal_class_weights.tolist(),
        },
    )

    test_inputs = make_model_inputs(test, spec.use_summary_branch)
    metrics = evaluate_hierarchical_model(model, test_inputs, test, metadata=splits["meta_test"])
    latency_ms = estimate_hierarchical_latency_ms(model, test_inputs)
    metrics["inference_latency_ms"] = latency_ms
    save_metrics_json(run_dir / "metrics.json", metrics)
    save_hierarchical_confusion_matrix_artifacts(metrics, config.output_dir, spec.id)

    row = flatten_hierarchical_metrics(spec.id, asdict(spec), metrics, latency_ms=latency_ms)
    print(
        f"{spec.id} result: fall_f1={row.get('fall_f1')}, "
        f"final_direction_macro_f1={row.get('final_direction_macro_f1')}, params={row.get('params')}"
    )
    if row.get("params") and row["params"] > 80000:
        print(f"Warning: {spec.id} params exceed target: {row['params']}")
    return row


def prepare_hierarchical_splits(
    X_raw6: np.ndarray,
    metadata: pd.DataFrame,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    spec: HierarchicalSpec,
    fs: float,
) -> dict[str, Any]:
    if "split" not in metadata:
        raise ValueError("metadata must contain a split column")

    split_values = metadata["split"].to_numpy()
    masks = {name: split_values == name for name in ["train", "val", "test"]}
    for split_name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"No samples found for split={split_name!r}")

    features = compute_imu_features(X_raw6, feature_set=spec.feature_set, fs=fs)
    feature_scaler = fit_feature_scaler(features, metadata, feature_set=spec.feature_set)
    X_scaled = transform_with_feature_scaler(features, feature_scaler)

    plane_y, sagittal_y, plane_mask, sagittal_mask = make_hierarchical_targets(
        y_fall,
        y_direction,
        direction_mask,
    )

    X_summary_scaled = None
    summary_scaler = None
    summary_names: list[str] = []
    if spec.use_summary_branch:
        X_summary, summary_names = direction_summary_features(X_raw6, fs=fs)
        summary_scaler = fit_summary_scaler(X_summary, metadata)
        X_summary_scaled = transform_summary_features(X_summary, summary_scaler)

    def _split(name: str) -> dict[str, np.ndarray]:
        mask = masks[name]
        out: dict[str, np.ndarray] = {
            "X": X_scaled[mask],
            "y_fall": y_fall[mask].astype(np.int64),
            "y_direction": y_direction[mask].astype(np.int64),
            "direction_mask": direction_mask[mask].astype(np.float32),
            "y_plane": plane_y[mask].astype(np.int64),
            "plane_mask": plane_mask[mask].astype(np.float32),
            "y_sagittal": sagittal_y[mask].astype(np.int64),
            "sagittal_mask": sagittal_mask[mask].astype(np.float32),
        }
        if spec.use_summary_branch and X_summary_scaled is not None:
            out["X_summary"] = X_summary_scaled[mask]
        return out

    return {
        "train": _split("train"),
        "val": _split("val"),
        "test": _split("test"),
        "meta_test": metadata.loc[masks["test"]].reset_index(drop=True),
        "feature_scaler": feature_scaler,
        "summary_scaler": summary_scaler,
        "summary_feature_names": summary_names,
    }


def make_hierarchical_targets(
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_fall = np.asarray(y_fall, dtype=np.int64)
    y_direction = np.asarray(y_direction, dtype=np.int64)
    supervised_fall = (np.asarray(direction_mask).astype(bool)) & (y_fall == 1) & (y_direction >= 0)

    y_plane = np.zeros_like(y_direction, dtype=np.int64)
    y_plane[y_direction == 2] = 1

    y_sagittal = np.zeros_like(y_direction, dtype=np.int64)
    y_sagittal[y_direction == 1] = 1

    sagittal_mask = supervised_fall & np.isin(y_direction, [0, 1])
    return y_plane, y_sagittal, supervised_fall.astype(np.float32), sagittal_mask.astype(np.float32)


def compile_hierarchical_model(
    model,
    learning_rate: float,
    fall_class_weights: np.ndarray | list[float] | None,
    plane_class_weights: np.ndarray | list[float],
    sagittal_class_weights: np.ndarray | list[float],
    lambda_plane: float,
    lambda_sagittal: float,
):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "fall_output": make_fall_loss(class_weights=fall_class_weights),
            "plane_output": make_sparse_ce_loss(class_weights=np.asarray(plane_class_weights), name="plane_weighted_ce"),
            "sagittal_output": make_sparse_ce_loss(
                class_weights=np.asarray(sagittal_class_weights),
                name="sagittal_weighted_ce",
            ),
        },
        loss_weights={
            "fall_output": 1.0,
            "plane_output": float(lambda_plane),
            "sagittal_output": float(lambda_sagittal),
        },
        metrics={
            "fall_output": ["accuracy"],
            "plane_output": ["accuracy"],
            "sagittal_output": ["accuracy"],
        },
    )
    return model


def make_model_inputs(split: dict[str, np.ndarray], use_summary_branch: bool):
    if use_summary_branch:
        return {
            "imu_input": np.asarray(split["X"], dtype=np.float32),
            "summary_input": np.asarray(split["X_summary"], dtype=np.float32),
        }
    return np.asarray(split["X"], dtype=np.float32)


def make_hierarchical_tf_dataset(
    split: dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
    use_summary_branch: bool,
):
    import tensorflow as tf

    inputs = make_model_inputs(split, use_summary_branch)
    y_dict = {
        "fall_output": np.asarray(split["y_fall"], dtype=np.int64),
        "plane_output": np.asarray(split["y_plane"], dtype=np.int64),
        "sagittal_output": np.asarray(split["y_sagittal"], dtype=np.int64),
    }
    sw_dict = {
        "fall_output": np.ones_like(split["y_fall"], dtype=np.float32),
        "plane_output": np.asarray(split["plane_mask"], dtype=np.float32),
        "sagittal_output": np.asarray(split["sagittal_mask"], dtype=np.float32),
    }
    ds = tf.data.Dataset.from_tensor_slices((inputs, y_dict, sw_dict))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(split["y_fall"]), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class HierarchicalValidationCallback:
    def __init__(self, val_inputs, val: dict[str, np.ndarray]):
        import tensorflow as tf

        class _Callback(tf.keras.callbacks.Callback):
            def on_epoch_end(callback_self, epoch, logs=None):
                logs = logs if logs is not None else {}
                preds = callback_self.model.predict(val_inputs, verbose=0)
                probs = _normalize_hierarchical_predictions(preds)

                fall_pred = np.argmax(probs["fall_output"], axis=1)
                plane_pred = np.argmax(probs["plane_output"], axis=1)
                sagittal_pred = np.argmax(probs["sagittal_output"], axis=1)
                final_direction_pred = reconstruct_direction_predictions(plane_pred, sagittal_pred)

                logs["val_fall_f1"] = float(
                    f1_score(val["y_fall"], fall_pred, average="binary", zero_division=0)
                )
                plane_mask = val["plane_mask"].astype(bool)
                logs["val_plane_macro_f1"] = _masked_macro_f1(val["y_plane"], plane_pred, plane_mask)
                sagittal_mask = val["sagittal_mask"].astype(bool)
                logs["val_sagittal_macro_f1"] = _masked_macro_f1(
                    val["y_sagittal"],
                    sagittal_pred,
                    sagittal_mask,
                )
                direction_mask = val["direction_mask"].astype(bool) & (val["y_direction"] >= 0) & (val["y_fall"] == 1)
                logs["val_final_direction_macro_f1"] = _masked_macro_f1(
                    val["y_direction"],
                    final_direction_pred,
                    direction_mask,
                )
                print(
                    f" - val_fall_f1: {logs['val_fall_f1']:.4f}"
                    f" - val_plane_macro_f1: {logs['val_plane_macro_f1']:.4f}"
                    f" - val_sagittal_macro_f1: {logs['val_sagittal_macro_f1']:.4f}"
                    f" - val_final_direction_macro_f1: {logs['val_final_direction_macro_f1']:.4f}"
                )

        self.callback = _Callback()


def build_hierarchical_callbacks(run_dir: str | Path, val_inputs, val: dict[str, np.ndarray], patience: int = 14):
    import tensorflow as tf

    run_dir = ensure_dir(run_dir)
    metric_callback = HierarchicalValidationCallback(val_inputs=val_inputs, val=val).callback
    return [
        metric_callback,
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "model_best.keras"),
            monitor="val_final_direction_macro_f1",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_final_direction_macro_f1",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            patience=max(3, patience // 2),
            factor=0.5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
    ]


def evaluate_hierarchical_model(
    model,
    inputs,
    split: dict[str, np.ndarray],
    metadata: pd.DataFrame | None = None,
) -> dict[str, Any]:
    preds = _normalize_hierarchical_predictions(model.predict(inputs, verbose=0))
    return evaluate_hierarchical_predictions(preds, split, metadata=metadata, model=model)


def evaluate_hierarchical_predictions(
    probs: dict[str, np.ndarray],
    split: dict[str, np.ndarray],
    metadata: pd.DataFrame | None = None,
    model=None,
) -> dict[str, Any]:
    fall_pred = np.argmax(probs["fall_output"], axis=1)
    plane_pred = np.argmax(probs["plane_output"], axis=1)
    sagittal_pred = np.argmax(probs["sagittal_output"], axis=1)
    final_direction_pred = reconstruct_direction_predictions(plane_pred, sagittal_pred)

    metrics: dict[str, Any] = {
        "fall": _fall_metrics(split["y_fall"], fall_pred),
        "plane": _multiclass_metrics(
            split["y_plane"],
            plane_pred,
            split["plane_mask"].astype(bool),
            labels=[0, 1],
            target_names=["sagittal", "lateral"],
        ),
        "sagittal": _multiclass_metrics(
            split["y_sagittal"],
            sagittal_pred,
            split["sagittal_mask"].astype(bool),
            labels=[0, 1],
            target_names=["forward", "backward"],
        ),
        "final_direction": _multiclass_metrics(
            split["y_direction"],
            final_direction_pred,
            split["direction_mask"].astype(bool) & (split["y_direction"] >= 0) & (split["y_fall"] == 1),
            labels=[0, 1, 2],
            target_names=["forward", "backward", "lateral"],
        ),
    }
    if model is not None:
        metrics["model_size"] = model_size_summary(model)

    if metadata is not None:
        metrics["per_dataset"] = evaluate_hierarchical_per_dataset(probs, split, metadata)
    return metrics


def evaluate_hierarchical_per_dataset(
    probs: dict[str, np.ndarray],
    split: dict[str, np.ndarray],
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "dataset" not in metadata:
        return out
    meta = metadata.reset_index(drop=True)
    for dataset in sorted(meta["dataset"].dropna().unique()):
        mask = meta["dataset"].to_numpy() == dataset
        if not mask.any():
            continue
        ds_probs = {key: value[mask] for key, value in probs.items()}
        ds_split = {key: value[mask] for key, value in split.items() if isinstance(value, np.ndarray) and len(value) == len(mask)}
        out[str(dataset)] = evaluate_hierarchical_predictions(ds_probs, ds_split, metadata=None, model=None)
    return out


def reconstruct_direction_predictions(plane_pred: np.ndarray, sagittal_pred: np.ndarray) -> np.ndarray:
    plane_pred = np.asarray(plane_pred, dtype=np.int64)
    sagittal_pred = np.asarray(sagittal_pred, dtype=np.int64)
    return np.where(plane_pred == 1, 2, sagittal_pred).astype(np.int64)


def _fall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    metrics = classification_metrics(y_true, y_pred, average="binary")
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["non_fall", "fall"],
        zero_division=0,
        output_dict=True,
    )
    return metrics


def _multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    labels: list[int],
    target_names: list[str],
) -> dict[str, Any]:
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return {"num_supervised": 0}
    yt = np.asarray(y_true, dtype=np.int64)[mask]
    yp = np.asarray(y_pred, dtype=np.int64)[mask]
    return {
        "num_supervised": int(mask.sum()),
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(yt, yp, labels=labels).tolist(),
        "classification_report": classification_report(
            yt,
            yp,
            labels=labels,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        ),
    }


def _masked_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return 0.0
    return float(f1_score(np.asarray(y_true)[mask], np.asarray(y_pred)[mask], average="macro", zero_division=0))


def _normalize_hierarchical_predictions(preds) -> dict[str, np.ndarray]:
    if isinstance(preds, dict):
        return preds
    if isinstance(preds, (list, tuple)) and len(preds) == 3:
        return {
            "fall_output": preds[0],
            "plane_output": preds[1],
            "sagittal_output": preds[2],
        }
    raise ValueError("Expected hierarchical model predictions with fall, plane, and sagittal outputs")


def estimate_hierarchical_latency_ms(model, inputs, warmup: int = 3, repeats: int = 30) -> float | None:
    n = _num_input_samples(inputs)
    if n == 0:
        return None
    sample = _first_input_sample(inputs)
    import time

    for _ in range(warmup):
        model.predict(sample, verbose=0)
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(sample, verbose=0)
    elapsed = time.perf_counter() - start
    return float((elapsed / repeats) * 1000.0)


def _num_input_samples(inputs) -> int:
    if isinstance(inputs, dict):
        first = next(iter(inputs.values()))
        return len(first)
    return len(inputs)


def _first_input_sample(inputs):
    if isinstance(inputs, dict):
        return {key: np.asarray(value[:1], dtype=np.float32) for key, value in inputs.items()}
    return np.asarray(inputs[:1], dtype=np.float32)


def save_hierarchical_confusion_matrix_artifacts(metrics: dict[str, Any], output_dir: str | Path, ablation_id: str) -> None:
    fig_dir = ensure_dir(Path(output_dir) / "figures" / "hierarchical_confusion_matrices" / ablation_id)
    _plot_metric_cm(metrics, "fall", ["non_fall", "fall"], f"{ablation_id} fall", fig_dir / "fall.png")
    _plot_metric_cm(metrics, "plane", ["sagittal", "lateral"], f"{ablation_id} plane", fig_dir / "plane.png")
    _plot_metric_cm(metrics, "sagittal", ["forward", "backward"], f"{ablation_id} sagittal", fig_dir / "sagittal.png")
    _plot_metric_cm(
        metrics,
        "final_direction",
        ["forward", "backward", "lateral"],
        f"{ablation_id} final direction",
        fig_dir / "final_direction.png",
    )
    for dataset, result in metrics.get("per_dataset", {}).items():
        ds_dir = ensure_dir(fig_dir / str(dataset))
        _plot_metric_cm(result, "fall", ["non_fall", "fall"], f"{ablation_id} {dataset} fall", ds_dir / "fall.png")
        _plot_metric_cm(result, "plane", ["sagittal", "lateral"], f"{ablation_id} {dataset} plane", ds_dir / "plane.png")
        _plot_metric_cm(
            result,
            "sagittal",
            ["forward", "backward"],
            f"{ablation_id} {dataset} sagittal",
            ds_dir / "sagittal.png",
        )
        _plot_metric_cm(
            result,
            "final_direction",
            ["forward", "backward", "lateral"],
            f"{ablation_id} {dataset} final direction",
            ds_dir / "final_direction.png",
        )


def _plot_metric_cm(metrics: dict[str, Any], key: str, labels: list[str], title: str, save_path: str | Path) -> None:
    cm = metrics.get(key, {}).get("confusion_matrix")
    if cm is not None:
        plot_confusion_matrix(cm, labels, f"{title} confusion matrix", save_path, show=False)


def flatten_hierarchical_metrics(
    ablation_id: str,
    spec: dict[str, Any],
    metrics: dict[str, Any],
    latency_ms: float | None = None,
) -> dict[str, Any]:
    fall = metrics.get("fall", {})
    plane = metrics.get("plane", {})
    sagittal = metrics.get("sagittal", {})
    final_direction = metrics.get("final_direction", {})
    size = metrics.get("model_size", {})
    row: dict[str, Any] = {
        "id": ablation_id,
        "model": spec.get("model_name"),
        "feature_set": spec.get("feature_set"),
        "use_summary_branch": spec.get("use_summary_branch"),
        "lambda_plane": spec.get("lambda_plane"),
        "lambda_sagittal": spec.get("lambda_sagittal"),
        "fall_loss_weighted": spec.get("fall_loss_weighted"),
        "fall_accuracy": fall.get("accuracy"),
        "fall_precision": fall.get("precision"),
        "fall_recall": fall.get("recall"),
        "fall_f1": fall.get("f1"),
        "plane_num_supervised": plane.get("num_supervised", 0),
        "plane_accuracy": plane.get("accuracy"),
        "plane_macro_f1": plane.get("macro_f1"),
        "sagittal_num_supervised": sagittal.get("num_supervised", 0),
        "sagittal_accuracy": sagittal.get("accuracy"),
        "sagittal_macro_f1": sagittal.get("macro_f1"),
        "final_direction_num_supervised": final_direction.get("num_supervised", 0),
        "final_direction_accuracy": final_direction.get("accuracy"),
        "final_direction_macro_f1": final_direction.get("macro_f1"),
        "params": size.get("params"),
        "fp32_kb": size.get("fp32_kb"),
        "estimated_int8_kb": size.get("estimated_int8_kb"),
        "inference_latency_ms": latency_ms,
    }

    dataset_f1s: list[float] = []
    for dataset, result in metrics.get("per_dataset", {}).items():
        row[f"{dataset}_fall_f1"] = result.get("fall", {}).get("f1")
        row[f"{dataset}_plane_macro_f1"] = result.get("plane", {}).get("macro_f1")
        row[f"{dataset}_sagittal_macro_f1"] = result.get("sagittal", {}).get("macro_f1")
        row[f"{dataset}_final_direction_macro_f1"] = result.get("final_direction", {}).get("macro_f1")
        row[f"{dataset}_final_direction_num_supervised"] = result.get("final_direction", {}).get("num_supervised", 0)
        if row[f"{dataset}_final_direction_macro_f1"] is not None:
            dataset_f1s.append(float(row[f"{dataset}_final_direction_macro_f1"]))
    row["per_dataset_final_direction_macro_f1_std"] = float(np.std(dataset_f1s)) if dataset_f1s else None

    report = final_direction.get("classification_report", {})
    for label in ["forward", "backward", "lateral"]:
        if label in report:
            row[f"final_direction_{label}_precision"] = report[label].get("precision")
            row[f"final_direction_{label}_recall"] = report[label].get("recall")
            row[f"final_direction_{label}_f1"] = report[label].get("f1-score")
    return row


def load_b0_a5wcefw_row(output_dir: str | Path) -> dict[str, Any] | None:
    reports_dir = Path(output_dir) / "reports"
    candidates = [reports_dir / "a5wcefw_after_unit_fix_metrics.csv", reports_dir / "ablation_results.csv"]
    source_row: pd.Series | None = None
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        match = df[df["id"].astype(str).str.upper() == "A5WCEFW"]
        if not match.empty:
            source_row = match.iloc[0]
            break
    if source_row is None:
        return None

    row = source_row.to_dict()
    out: dict[str, Any] = {
        "id": "B0",
        "model": "A5WCEFW",
        "feature_set": row.get("feature_set", "tilt12"),
        "use_summary_branch": False,
        "lambda_plane": None,
        "lambda_sagittal": None,
        "fall_loss_weighted": row.get("fall_loss_weighted"),
        "fall_accuracy": row.get("fall_accuracy"),
        "fall_precision": row.get("fall_precision"),
        "fall_recall": row.get("fall_recall"),
        "fall_f1": row.get("fall_f1"),
        "plane_num_supervised": None,
        "plane_accuracy": None,
        "plane_macro_f1": None,
        "sagittal_num_supervised": None,
        "sagittal_accuracy": None,
        "sagittal_macro_f1": None,
        "final_direction_num_supervised": row.get("direction_num_supervised"),
        "final_direction_accuracy": row.get("direction_accuracy"),
        "final_direction_macro_f1": row.get("direction_macro_f1"),
        "params": row.get("params"),
        "fp32_kb": row.get("fp32_kb"),
        "estimated_int8_kb": row.get("estimated_int8_kb"),
        "inference_latency_ms": row.get("inference_latency_ms"),
        "per_dataset_final_direction_macro_f1_std": None,
        "source": "reused_existing_A5WCEFW_after_unit_fix",
    }
    dataset_f1s: list[float] = []
    for col, value in row.items():
        if col.endswith("_fall_f1"):
            dataset = col.removesuffix("_fall_f1")
            out[f"{dataset}_fall_f1"] = value
        if col.endswith("_direction_macro_f1"):
            dataset = col.removesuffix("_direction_macro_f1")
            out[f"{dataset}_final_direction_macro_f1"] = value
            if pd.notna(value):
                dataset_f1s.append(float(value))
        if col.endswith("_direction_num_supervised"):
            dataset = col.removesuffix("_direction_num_supervised")
            out[f"{dataset}_final_direction_num_supervised"] = value
        if col.startswith("direction_") and col.rsplit("_", 1)[-1] in {"precision", "recall", "f1"}:
            out[f"final_{col}"] = value
    if dataset_f1s:
        out["per_dataset_final_direction_macro_f1_std"] = float(np.std(dataset_f1s))
    return out


def save_b0_confusion_matrices(output_dir: str | Path) -> None:
    metrics_path = Path(output_dir) / "runs" / "A5WCEFW_ds_fall_rd_tilt12" / "metrics.json"
    if not metrics_path.exists():
        return
    metrics = load_json(metrics_path)
    fig_dir = ensure_dir(Path(output_dir) / "figures" / "hierarchical_confusion_matrices" / "B0")
    fall_cm = metrics.get("fall", {}).get("confusion_matrix")
    if fall_cm is not None:
        plot_confusion_matrix(fall_cm, ["non_fall", "fall"], "B0 fall confusion matrix", fig_dir / "fall.png", show=False)
    direction_cm = metrics.get("direction", {}).get("confusion_matrix")
    if direction_cm is not None:
        plot_confusion_matrix(
            direction_cm,
            ["forward", "backward", "lateral"],
            "B0 final direction confusion matrix",
            fig_dir / "final_direction.png",
            show=False,
        )


def save_hierarchical_results(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    reports_dir = ensure_dir(Path(output_dir) / "reports")
    df = pd.DataFrame(rows)
    df.to_csv(reports_dir / "hierarchical_direction_ablation.csv", index=False)
    (reports_dir / "hierarchical_direction_ablation.md").write_text(_hierarchical_markdown(df), encoding="utf-8")


def save_hierarchical_model_selection_report(results: pd.DataFrame, output_dir: str | Path) -> None:
    save_hierarchical_results(results.to_dict("records"), output_dir)


def _hierarchical_markdown(df: pd.DataFrame) -> str:
    lines = [
        "# Hierarchical Direction Ablation",
        "",
        "Selection priority: final direction macro F1, fall F1, per-dataset consistency, then params/latency.",
        "",
    ]
    if df.empty:
        return "\n".join(lines + ["_No rows._", ""]) + "\n"

    ranked = df.copy()
    sort_cols = [col for col in ["final_direction_macro_f1", "fall_f1", "per_dataset_final_direction_macro_f1_std", "params"] if col in ranked]
    ascending = [False, False, True, True][: len(sort_cols)]
    ranked = ranked.sort_values(sort_cols, ascending=ascending, na_position="last") if sort_cols else ranked
    best = ranked.iloc[0].to_dict()
    lines.extend(
        [
            f"Best by implemented ranking: `{best.get('id')}`",
            f"- Final direction macro F1: {_fmt(best.get('final_direction_macro_f1'))}",
            f"- Fall F1: {_fmt(best.get('fall_f1'))}",
            f"- Per-dataset final direction F1 std: {_fmt(best.get('per_dataset_final_direction_macro_f1_std'))}",
            f"- Params: {_fmt(best.get('params'), digits=0)}",
            "",
            "## Ranked Results",
            "",
            _dataframe_to_markdown(ranked),
            "",
        ]
    )
    return "\n".join(lines)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    show_cols = [
        col
        for col in [
            "id",
            "model",
            "use_summary_branch",
            "fall_f1",
            "plane_macro_f1",
            "sagittal_macro_f1",
            "final_direction_macro_f1",
            "final_direction_accuracy",
            "per_dataset_final_direction_macro_f1_std",
            "bits_final_direction_macro_f1",
            "hifd_final_direction_macro_f1",
            "umafall_final_direction_macro_f1",
            "weda_final_direction_macro_f1",
            "params",
            "estimated_int8_kb",
            "inference_latency_ms",
        ]
        if col in df.columns
    ]
    compact = df[show_cols].copy()
    for col in compact.select_dtypes(include=[float]).columns:
        compact[col] = compact[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    header = "| " + " | ".join(compact.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(compact.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in compact.to_numpy()]
    return "\n".join([header, sep] + rows) + "\n"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    if isinstance(value, (int, np.integer)) or digits == 0:
        return f"{float(value):.{digits}f}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)
