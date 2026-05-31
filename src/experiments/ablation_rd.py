from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import DIRECTION_LABEL_MAPPING, FALL_LABEL_MAPPING, ProjectConfig, make_config
from src.data.features import (
    compute_imu_features,
    feature_channel_names,
    fit_feature_scaler,
    inverse_standardize_raw6,
    transform_with_feature_scaler,
)
from src.data.quality import (
    generate_data_quality_report,
    plot_direction_signal_diagnostics,
    run_handcrafted_direction_baseline,
)
from src.models.losses import compute_class_weights
from src.training.augmentation import augment_raw_imu_windows
from src.training.evaluate import (
    estimate_inference_latency_ms,
    evaluate_all,
    flatten_ablation_metrics,
    plot_confusion_matrix,
    save_metrics_json,
)
from src.utils.io import ensure_dir, load_pickle, save_json, save_pickle
from src.utils.seed import set_seed


@dataclass(frozen=True)
class AblationSpec:
    id: str
    model_name: str
    feature_set: str
    direction_loss_type: str = "ce"
    lambda_fall: float = 1.0
    lambda_direction: float = 1.5
    focal_gamma: float = 2.0
    augment: bool = False
    augment_strength: str = "light"
    fall_loss_weighted: bool = False


def default_ablation_specs() -> list[AblationSpec]:
    return [
        AblationSpec("A0", "ds_fall", "raw6", direction_loss_type="ce", lambda_direction=0.5),
        AblationSpec("A1", "ds_fall", "mag_jerk9", direction_loss_type="ce", lambda_direction=0.5),
        AblationSpec("A2", "ds_fall", "roll_pitch11", direction_loss_type="ce", lambda_direction=0.5),
        AblationSpec("A3", "ds_fall", "tilt12", direction_loss_type="ce", lambda_direction=0.5),
        AblationSpec("A4", "ds_fall_rd", "tilt12", direction_loss_type="ce", lambda_direction=0.5),
        AblationSpec("A5", "ds_fall_rd", "tilt12", direction_loss_type="focal", lambda_direction=1.5),
        AblationSpec(
            "A5WCEFW",
            "ds_fall_rd",
            "tilt12",
            direction_loss_type="weighted_ce",
            lambda_direction=1.5,
            fall_loss_weighted=True,
        ),
        AblationSpec("A6", "ds_fall_rd", "tilt12", direction_loss_type="focal", lambda_direction=1.5, augment=True),
    ]


def run_ablation_suite(
    config: ProjectConfig,
    ablation_ids: list[str] | None = None,
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    processed_x_is_normalized: bool = True,
    skip_diagnostics: bool = False,
) -> pd.DataFrame:
    set_seed(config.seed)
    data = load_processed_training_data(config.processed_dir, processed_x_is_normalized=processed_x_is_normalized)
    X_raw6 = data["X_raw6"]
    metadata = data["metadata"]
    y_fall = data["y_fall"]
    y_direction = data["y_direction"]
    direction_mask = data["direction_mask"]

    if not skip_diagnostics:
        print("Generating DS-Fall-RD data quality and direction diagnostics...")
        quality_features = compute_imu_features(X_raw6, feature_set="tilt12", fs=config.model_fs)
        generate_data_quality_report(
            X_raw6,
            quality_features,
            metadata,
            y_fall,
            y_direction,
            direction_mask,
            output_dir=config.output_dir,
            feature_set="tilt12",
            fs=config.model_fs,
        )
        plot_direction_signal_diagnostics(
            X_raw6,
            metadata,
            direction_mask,
            output_dir=config.output_dir,
            fs=config.model_fs,
        )
        baseline = run_handcrafted_direction_baseline(
            X_raw6,
            metadata,
            y_direction,
            direction_mask,
            output_dir=config.output_dir,
            seed=config.seed,
            fs=config.model_fs,
        )
        if baseline.get("warning"):
            print("Handcrafted direction baseline warning:", baseline["warning"])

    specs = default_ablation_specs()
    if ablation_ids:
        wanted = {item.upper() for item in ablation_ids}
        specs = [spec for spec in specs if spec.id.upper() in wanted]
        missing = sorted(wanted - {spec.id.upper() for spec in specs})
        if missing:
            raise ValueError(f"Unknown ablation ids: {missing}")

    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = run_single_ablation(
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
        save_ablation_results(rows, config.output_dir)

    results = pd.DataFrame(rows)
    save_model_selection_report(results, config.output_dir)
    return results


def run_single_ablation(
    spec: AblationSpec,
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
    from src.models.ds_fall import build_model
    from src.training.dataset import make_tf_dataset
    from src.training.train import compile_ds_fall_model

    set_seed(config.seed)
    run_name = f"{spec.id}_{spec.model_name}_{spec.feature_set}"
    run_dir = ensure_dir(config.output_dir / "runs" / run_name)
    print(f"\n=== {spec.id}: model={spec.model_name}, feature_set={spec.feature_set}, loss={spec.direction_loss_type}, augment={spec.augment} ===")

    splits = prepare_feature_splits(
        X_raw6,
        metadata,
        y_fall,
        y_direction,
        direction_mask,
        spec=spec,
        fs=config.model_fs,
        seed=config.seed,
    )
    X_train, yf_train, yd_train, dm_train = splits["train"]
    X_val, yf_val, yd_val, dm_val = splits["val"]
    X_test, yf_test, yd_test, dm_test = splits["test"]
    meta_test = splits["meta_test"]
    scaler = splits["scaler"]

    direction_class_weights = compute_class_weights(yd_train, num_classes=3, mask=dm_train * (yf_train == 1))
    fall_class_weights = compute_class_weights(yf_train, num_classes=2) if spec.fall_loss_weighted else None
    if spec.direction_loss_type == "ce":
        compile_direction_weights = None
    else:
        compile_direction_weights = direction_class_weights

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(
        spec.model_name,
        input_shape=input_shape,
        feature_set=spec.feature_set,
        num_direction_classes=3,
        show_summary=False,
    )
    compile_ds_fall_model(
        model,
        learning_rate=learning_rate,
        direction_loss_type=spec.direction_loss_type,
        lambda_fall=spec.lambda_fall,
        lambda_direction=spec.lambda_direction,
        focal_gamma=spec.focal_gamma,
        direction_class_weights=compile_direction_weights,
        fall_class_weights=fall_class_weights,
    )

    train_ds = make_tf_dataset(X_train, yf_train, yd_train, dm_train, batch_size=batch_size, shuffle=True, seed=config.seed)
    val_ds = make_tf_dataset(X_val, yf_val, yd_val, dm_val, batch_size=batch_size, shuffle=False, seed=config.seed)
    callbacks = build_ablation_callbacks(
        run_dir,
        X_val=X_val,
        y_fall_val=yf_val,
        y_direction_val=yd_val,
        direction_mask_val=dm_val,
        patience=12,
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)

    model.save(run_dir / "model_final.keras")
    save_json(run_dir / "history.json", history.history)
    save_json(run_dir / "feature_config.json", {"feature_set": spec.feature_set, "channel_names": feature_channel_names(spec.feature_set)})
    save_json(run_dir / "model_config.json", asdict(spec) | {"input_shape": list(input_shape), "learning_rate": learning_rate})
    save_json(run_dir / "feature_scaler.json", scaler)
    save_pickle(run_dir / "feature_scaler.pkl", scaler)
    save_json(run_dir / "label_mapping.json", {"fall": FALL_LABEL_MAPPING, "direction": DIRECTION_LABEL_MAPPING})
    save_json(
        run_dir / "class_weights.json",
        {
            "fall": fall_class_weights.tolist() if fall_class_weights is not None else None,
            "direction": direction_class_weights.tolist(),
        },
    )

    metrics = evaluate_all(model, X_test, yf_test, yd_test, dm_test, metadata=meta_test)
    latency_ms = estimate_inference_latency_ms(model, X_test)
    metrics["inference_latency_ms"] = latency_ms
    save_metrics_json(run_dir / "metrics.json", metrics)
    save_per_subject_metrics(metrics, run_dir)
    save_confusion_matrix_artifacts(metrics, config.output_dir, spec.id)

    row = flatten_ablation_metrics(spec.id, asdict(spec), metrics, latency_ms=latency_ms)
    print(
        f"{spec.id} result: fall_f1={row.get('fall_f1')}, "
        f"direction_macro_f1={row.get('direction_macro_f1')}, params={row.get('params')}"
    )
    return row


def prepare_feature_splits(
    X_raw6: np.ndarray,
    metadata: pd.DataFrame,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    spec: AblationSpec,
    fs: float,
    seed: int,
) -> dict[str, Any]:
    if "split" not in metadata:
        raise ValueError("metadata must contain a split column")
    split_values = metadata["split"].to_numpy()
    masks = {name: split_values == name for name in ["train", "val", "test"]}
    for split_name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"No samples found for split={split_name!r}")

    features = compute_imu_features(X_raw6, feature_set=spec.feature_set, fs=fs)
    scaler = fit_feature_scaler(features, metadata, feature_set=spec.feature_set)
    X_scaled = transform_with_feature_scaler(features, scaler)

    X_train = X_scaled[masks["train"]]
    yf_train = y_fall[masks["train"]]
    yd_train = y_direction[masks["train"]]
    dm_train = direction_mask[masks["train"]]

    if spec.augment:
        X_aug_raw = augment_raw_imu_windows(X_raw6[masks["train"]], seed=seed, strength=spec.augment_strength)
        X_aug = transform_with_feature_scaler(
            compute_imu_features(X_aug_raw, feature_set=spec.feature_set, fs=fs),
            scaler,
        )
        X_train = np.concatenate([X_train, X_aug], axis=0)
        yf_train = np.concatenate([yf_train, yf_train], axis=0)
        yd_train = np.concatenate([yd_train, yd_train], axis=0)
        dm_train = np.concatenate([dm_train, dm_train], axis=0)

    return {
        "train": (X_train, yf_train, yd_train, dm_train),
        "val": (X_scaled[masks["val"]], y_fall[masks["val"]], y_direction[masks["val"]], direction_mask[masks["val"]]),
        "test": (X_scaled[masks["test"]], y_fall[masks["test"]], y_direction[masks["test"]], direction_mask[masks["test"]]),
        "meta_test": metadata.loc[masks["test"]].reset_index(drop=True),
        "scaler": scaler,
    }


def load_processed_training_data(
    processed_dir: str | Path,
    processed_x_is_normalized: bool = True,
) -> dict[str, Any]:
    processed_dir = Path(processed_dir)
    required = ["X.npy", "y_fall.npy", "y_direction.npy", "direction_mask.npy", "metadata.csv"]
    missing = [name for name in required if not (processed_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed files in {processed_dir}: {missing}. Run notebook 01 before ablations."
        )

    X_processed = np.load(processed_dir / "X.npy").astype(np.float32)
    metadata = pd.read_csv(processed_dir / "metadata.csv")
    y_fall = np.load(processed_dir / "y_fall.npy").astype(np.int64)
    y_direction = np.load(processed_dir / "y_direction.npy").astype(np.int64)
    direction_mask = np.load(processed_dir / "direction_mask.npy").astype(np.float32)

    scaler_path = processed_dir / "scaler.pkl"
    raw6_scaler = load_pickle(scaler_path) if scaler_path.exists() else None
    if processed_x_is_normalized:
        X_raw6 = inverse_standardize_raw6(X_processed, raw6_scaler)
    else:
        X_raw6 = X_processed

    return {
        "X_processed": X_processed,
        "X_raw6": X_raw6.astype(np.float32),
        "metadata": metadata,
        "y_fall": y_fall,
        "y_direction": y_direction,
        "direction_mask": direction_mask,
        "raw6_scaler": raw6_scaler,
    }


class ValidationMacroF1Callback:
    def __init__(
        self,
        X_val: np.ndarray,
        y_fall_val: np.ndarray,
        y_direction_val: np.ndarray,
        direction_mask_val: np.ndarray,
    ):
        import tensorflow as tf

        class _Callback(tf.keras.callbacks.Callback):
            def on_epoch_end(callback_self, epoch, logs=None):
                logs = logs if logs is not None else {}
                preds = callback_self.model.predict(X_val, verbose=0)
                if isinstance(preds, dict):
                    fall_probs = preds["fall_output"]
                    direction_probs = preds["direction_output"]
                else:
                    fall_probs, direction_probs = preds

                fall_pred = np.argmax(fall_probs, axis=1)
                logs["val_fall_f1"] = float(f1_score(y_fall_val, fall_pred, average="binary", zero_division=0))

                supervised = direction_mask_val.astype(bool) & (y_direction_val >= 0)
                if supervised.any():
                    direction_pred = np.argmax(direction_probs[supervised], axis=1)
                    logs["val_direction_macro_f1"] = float(
                        f1_score(y_direction_val[supervised], direction_pred, average="macro", zero_division=0)
                    )
                else:
                    logs["val_direction_macro_f1"] = 0.0
                print(
                    f" - val_fall_f1: {logs['val_fall_f1']:.4f}"
                    f" - val_direction_macro_f1: {logs['val_direction_macro_f1']:.4f}"
                )

        self.callback = _Callback()


def build_ablation_callbacks(
    run_dir: str | Path,
    X_val: np.ndarray,
    y_fall_val: np.ndarray,
    y_direction_val: np.ndarray,
    direction_mask_val: np.ndarray,
    patience: int = 12,
):
    import tensorflow as tf

    run_dir = ensure_dir(run_dir)
    metric_callback = ValidationMacroF1Callback(
        X_val=X_val,
        y_fall_val=y_fall_val,
        y_direction_val=y_direction_val,
        direction_mask_val=direction_mask_val,
    ).callback
    return [
        metric_callback,
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "model_best.keras"),
            monitor="val_direction_macro_f1",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_direction_macro_f1",
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


def save_confusion_matrix_artifacts(metrics: dict[str, Any], output_dir: str | Path, ablation_id: str) -> None:
    fig_dir = ensure_dir(Path(output_dir) / "figures" / "confusion_matrices" / ablation_id)
    fall_cm = metrics.get("fall", {}).get("confusion_matrix")
    if fall_cm is not None:
        plot_confusion_matrix(fall_cm, ["non_fall", "fall"], f"{ablation_id} fall confusion matrix", fig_dir / "fall.png", show=False)
    direction_cm = metrics.get("direction", {}).get("confusion_matrix")
    if direction_cm is not None:
        plot_confusion_matrix(
            direction_cm,
            ["forward", "backward", "lateral"],
            f"{ablation_id} direction confusion matrix",
            fig_dir / "direction.png",
            show=False,
        )
    for dataset, result in metrics.get("per_dataset", {}).items():
        dataset_dir = ensure_dir(fig_dir / str(dataset))
        ds_fall_cm = result.get("fall", {}).get("confusion_matrix")
        if ds_fall_cm is not None:
            plot_confusion_matrix(
                ds_fall_cm,
                ["non_fall", "fall"],
                f"{ablation_id} {dataset} fall confusion matrix",
                dataset_dir / "fall.png",
                show=False,
            )
        ds_dir_cm = result.get("direction", {}).get("confusion_matrix")
        if ds_dir_cm is not None:
            plot_confusion_matrix(
                ds_dir_cm,
                ["forward", "backward", "lateral"],
                f"{ablation_id} {dataset} direction confusion matrix",
                dataset_dir / "direction.png",
                show=False,
            )


def save_per_subject_metrics(metrics: dict[str, Any], run_dir: str | Path) -> None:
    rows = []
    for subject_key, result in metrics.get("per_subject", {}).items():
        row = {
            "subject": subject_key,
            "num_windows": result.get("num_windows"),
            "fall_f1": result.get("fall", {}).get("f1"),
            "fall_accuracy": result.get("fall", {}).get("accuracy"),
            "direction_num_supervised": result.get("direction", {}).get("num_supervised", 0),
            "direction_accuracy": result.get("direction", {}).get("accuracy"),
            "direction_macro_f1": result.get("direction", {}).get("macro_f1"),
        }
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(Path(run_dir) / "per_subject_metrics.csv", index=False)


def save_ablation_results(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    reports_dir = ensure_dir(Path(output_dir) / "reports")
    df = pd.DataFrame(rows)
    df.to_csv(reports_dir / "ablation_results.csv", index=False)
    (reports_dir / "ablation_results.md").write_text(_dataframe_to_markdown(df), encoding="utf-8")


def save_model_selection_report(results: pd.DataFrame, output_dir: str | Path) -> None:
    reports_dir = ensure_dir(Path(output_dir) / "reports")
    if results.empty:
        return
    sort_cols = [col for col in ["direction_macro_f1", "fall_f1", "params"] if col in results.columns]
    ascending = [False, False, True][: len(sort_cols)]
    ranked = results.sort_values(sort_cols, ascending=ascending, na_position="last") if sort_cols else results
    best = ranked.iloc[0].to_dict()
    lines = [
        "# DS-Fall-RD Model Selection",
        "",
        "Selection priority: direction macro F1, fall F1, per-dataset consistency, then parameter/inference cost.",
        "",
        f"Best ablation by implemented ranking: `{best.get('id')}`",
        f"- Direction macro F1: {best.get('direction_macro_f1')}",
        f"- Fall F1: {best.get('fall_f1')}",
        f"- Params: {best.get('params')}",
        f"- Estimated INT8 KB: {best.get('estimated_int8_kb')}",
        "",
        "## Ranked Results",
        "",
        _dataframe_to_markdown(ranked),
    ]
    (reports_dir / "model_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    show_cols = [
        col
        for col in [
            "id",
            "model",
            "feature_set",
            "direction_loss_type",
            "fall_loss_weighted",
            "augment",
            "fall_f1",
            "direction_macro_f1",
            "direction_accuracy",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DS-Fall-RD ablations A0-A6.")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ablation", nargs="*", default=None, help="Subset of ablation IDs, e.g. A4 A5 A6")
    parser.add_argument("--processed-x-is-raw", action="store_true", help="Use if data/processed/X.npy is already raw6, not normalized.")
    parser.add_argument("--skip-diagnostics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = make_config(args.project_root)
    run_ablation_suite(
        config,
        ablation_ids=args.ablation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        processed_x_is_normalized=not args.processed_x_is_raw,
        skip_diagnostics=args.skip_diagnostics,
    )


if __name__ == "__main__":
    main()
