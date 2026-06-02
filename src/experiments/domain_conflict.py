from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import DIRECTION_LABEL_MAPPING, FALL_LABEL_MAPPING, ProjectConfig
from src.data.features import compute_imu_features, feature_channel_names, transform_with_feature_scaler
from src.experiments.ablation_rd import load_processed_training_data
from src.models.losses import compute_class_weights
from src.reporting.analysis_notes import update_analysis_notes_from_reports
from src.training.dataset import make_tf_dataset
from src.training.evaluate import (
    estimate_inference_latency_ms,
    evaluate_all,
    flatten_ablation_metrics,
    plot_confusion_matrix,
    save_metrics_json,
)
from src.training.train import compile_ds_fall_model
from src.utils.io import ensure_dir, save_json, save_pickle
from src.utils.seed import set_seed


DEFAULT_DATASETS = ["bits", "weda", "hifd", "umafall"]


@dataclass(frozen=True)
class DomainExperimentSpec:
    id: str
    experiment_type: str
    train_datasets: tuple[str, ...]
    test_datasets: tuple[str, ...]
    train_split: str = "train"
    val_split: str = "val"
    test_split: str | None = "test"
    note: str = ""


def dataset_specific_specs(datasets: list[str]) -> list[DomainExperimentSpec]:
    return [
        DomainExperimentSpec(
            id=f"DS_{dataset.upper()}",
            experiment_type="dataset_specific",
            train_datasets=(dataset,),
            test_datasets=(dataset,),
            test_split="test",
            note="Existing split within one dataset.",
        )
        for dataset in datasets
    ]


def train3_test1_specs(datasets: list[str]) -> list[DomainExperimentSpec]:
    out: list[DomainExperimentSpec] = []
    for heldout in datasets:
        train_datasets = tuple(dataset for dataset in datasets if dataset != heldout)
        out.append(
            DomainExperimentSpec(
                id=f"T3_{heldout.upper()}",
                experiment_type="train3_test1",
                train_datasets=train_datasets,
                test_datasets=(heldout,),
                test_split=None,
                note="Held-out test dataset uses all its windows; it is never used for scaler or early stopping.",
            )
        )
    return out


def train1_testothers_specs(datasets: list[str]) -> list[DomainExperimentSpec]:
    out: list[DomainExperimentSpec] = []
    for train_dataset in datasets:
        test_datasets = tuple(dataset for dataset in datasets if dataset != train_dataset)
        out.append(
            DomainExperimentSpec(
                id=f"T1_{train_dataset.upper()}",
                experiment_type="train1_testothers",
                train_datasets=(train_dataset,),
                test_datasets=test_datasets,
                test_split=None,
                note="External test datasets use all their windows.",
            )
        )
    return out


def run_domain_conflict_suite(
    config: ProjectConfig,
    datasets: list[str] | None = None,
    suites: list[str] | None = None,
    epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    processed_x_is_normalized: bool = True,
) -> pd.DataFrame:
    set_seed(config.seed)
    datasets = [dataset.lower() for dataset in (datasets or DEFAULT_DATASETS)]
    suites = [suite.lower() for suite in (suites or ["dataset_specific", "train3_test1"])]

    data = load_processed_training_data(config.processed_dir, processed_x_is_normalized=processed_x_is_normalized)
    X_raw6 = data["X_raw6"]
    metadata = data["metadata"].copy()
    metadata["dataset"] = metadata["dataset"].astype(str).str.lower()
    y_fall = data["y_fall"]
    y_direction = data["y_direction"]
    direction_mask = data["direction_mask"]

    available = set(metadata["dataset"].dropna().unique())
    missing = sorted(set(datasets) - available)
    if missing:
        raise ValueError(f"Datasets not found in processed metadata: {missing}")

    specs: list[DomainExperimentSpec] = []
    if "dataset_specific" in suites:
        specs.extend(dataset_specific_specs(datasets))
    if "train3_test1" in suites:
        specs.extend(train3_test1_specs(datasets))
    if "train1_testothers" in suites:
        specs.extend(train1_testothers_specs(datasets))

    rows: list[dict[str, Any]] = []
    save_domain_conflict_results(rows, config.output_dir)
    for spec in specs:
        row = run_single_domain_experiment(
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
        save_domain_conflict_results(rows, config.output_dir)

    results = pd.DataFrame(rows)
    save_domain_conflict_results(rows, config.output_dir)
    return results


def run_single_domain_experiment(
    spec: DomainExperimentSpec,
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

    set_seed(config.seed)
    run_dir = ensure_dir(config.output_dir / "runs" / f"DOMAIN_{spec.id}_A5WCEFW")
    print(
        f"\n=== {spec.id}: type={spec.experiment_type}, "
        f"train={'+'.join(spec.train_datasets)}, test={'+'.join(spec.test_datasets)} ==="
    )

    splits = prepare_domain_splits(
        X_raw6,
        metadata,
        y_fall,
        y_direction,
        direction_mask,
        spec=spec,
        feature_set="tilt12",
        fs=config.model_fs,
    )
    X_train, yf_train, yd_train, dm_train = splits["train"]
    X_val, yf_val, yd_val, dm_val = splits["val"]
    X_test, yf_test, yd_test, dm_test = splits["test"]
    meta_test = splits["meta_test"]

    direction_class_weights = compute_class_weights(yd_train, num_classes=3, mask=dm_train * (yf_train == 1))
    fall_class_weights = compute_class_weights(yf_train, num_classes=2)

    model = build_model(
        "ds_fall_rd",
        input_shape=(X_train.shape[1], X_train.shape[2]),
        feature_set="tilt12",
        num_direction_classes=3,
        show_summary=False,
    )
    compile_ds_fall_model(
        model,
        learning_rate=learning_rate,
        direction_loss_type="weighted_ce",
        lambda_fall=1.0,
        lambda_direction=1.5,
        focal_gamma=2.0,
        direction_class_weights=direction_class_weights,
        fall_class_weights=fall_class_weights,
    )

    train_ds = make_tf_dataset(X_train, yf_train, yd_train, dm_train, batch_size=batch_size, shuffle=True, seed=config.seed)
    val_ds = make_tf_dataset(X_val, yf_val, yd_val, dm_val, batch_size=batch_size, shuffle=False, seed=config.seed)
    callbacks = build_domain_callbacks(
        run_dir,
        X_val=X_val,
        y_fall_val=yf_val,
        y_direction_val=yd_val,
        direction_mask_val=dm_val,
        patience=20,
        use_early_stopping=spec.experiment_type != "dataset_specific",
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)

    model.save(run_dir / "model_final.keras")
    save_json(run_dir / "history.json", history.history)
    save_json(run_dir / "feature_config.json", {"feature_set": "tilt12", "channel_names": feature_channel_names("tilt12")})
    save_json(run_dir / "model_config.json", asdict(spec) | {"model": "A5WCEFW", "learning_rate": learning_rate})
    save_json(run_dir / "feature_scaler.json", splits["scaler"])
    save_pickle(run_dir / "feature_scaler.pkl", splits["scaler"])
    save_json(run_dir / "label_mapping.json", {"fall": FALL_LABEL_MAPPING, "direction": DIRECTION_LABEL_MAPPING})
    save_json(
        run_dir / "class_weights.json",
        {
            "fall": fall_class_weights.tolist(),
            "direction": direction_class_weights.tolist(),
        },
    )

    metrics = evaluate_all(model, X_test, yf_test, yd_test, dm_test, metadata=meta_test)
    latency_ms = estimate_inference_latency_ms(model, X_test)
    metrics["inference_latency_ms"] = latency_ms
    save_metrics_json(run_dir / "metrics.json", metrics)
    save_domain_confusion_matrices(metrics, config.output_dir, spec.id)

    row = flatten_domain_metrics(spec, metrics, splits, latency_ms=latency_ms)
    print(
        f"{spec.id} result: fall_f1={row.get('fall_f1')}, "
        f"direction_macro_f1={row.get('direction_macro_f1')}, "
        f"direction_n={row.get('direction_num_supervised')}"
    )
    return row


def prepare_domain_splits(
    X_raw6: np.ndarray,
    metadata: pd.DataFrame,
    y_fall: np.ndarray,
    y_direction: np.ndarray,
    direction_mask: np.ndarray,
    spec: DomainExperimentSpec,
    feature_set: str,
    fs: float,
) -> dict[str, Any]:
    if "dataset" not in metadata or "split" not in metadata:
        raise ValueError("metadata must contain dataset and split columns")

    dataset_values = metadata["dataset"].astype(str).str.lower().to_numpy()
    split_values = metadata["split"].astype(str).str.lower().to_numpy()
    train_dataset_mask = np.isin(dataset_values, list(spec.train_datasets))
    test_dataset_mask = np.isin(dataset_values, list(spec.test_datasets))

    train_mask = train_dataset_mask & (split_values == spec.train_split)
    val_mask = train_dataset_mask & (split_values == spec.val_split)
    if spec.test_split is None:
        test_mask = test_dataset_mask
    else:
        test_mask = test_dataset_mask & (split_values == spec.test_split)

    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        if not mask.any():
            raise ValueError(f"{spec.id} has no {name} samples")
    val_supervised = int(((direction_mask > 0) & (y_direction >= 0) & val_mask).sum())
    if val_supervised == 0:
        raise ValueError(f"{spec.id} has no supervised direction samples in val split")

    features = compute_imu_features(X_raw6, feature_set=feature_set, fs=fs)
    scaler = fit_feature_scaler_on_mask(features, train_mask, feature_set=feature_set)
    X_scaled = transform_with_feature_scaler(features, scaler)

    def _split(mask: np.ndarray):
        return (
            X_scaled[mask],
            y_fall[mask].astype(np.int64),
            y_direction[mask].astype(np.int64),
            direction_mask[mask].astype(np.float32),
        )

    return {
        "train": _split(train_mask),
        "val": _split(val_mask),
        "test": _split(test_mask),
        "meta_test": metadata.loc[test_mask].reset_index(drop=True),
        "scaler": scaler,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_train_direction": int(((direction_mask > 0) & (y_direction >= 0) & train_mask).sum()),
        "n_val_direction": int(((direction_mask > 0) & (y_direction >= 0) & val_mask).sum()),
        "n_test_direction": int(((direction_mask > 0) & (y_direction >= 0) & test_mask).sum()),
    }


def fit_feature_scaler_on_mask(
    X_features: np.ndarray,
    train_mask: np.ndarray,
    feature_set: str,
    eps: float = 1e-6,
) -> dict[str, Any]:
    X = np.asarray(X_features, dtype=np.float32)
    train_mask = np.asarray(train_mask).astype(bool)
    if not train_mask.any():
        raise ValueError("Cannot fit scaler without train samples")
    train_x = X[train_mask]
    mean = train_x.mean(axis=(0, 1)).astype(np.float32)
    std = train_x.std(axis=(0, 1)).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    return {
        "feature_set": feature_set,
        "channel_names": feature_channel_names(feature_set),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "fit_scope": "domain_conflict_train_mask_only",
        "eps": float(eps),
    }


class DomainValidationCallback:
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

                logs["val_domain_score"] = logs["val_direction_macro_f1"] + logs["val_fall_f1"]
                print(
                    f" - val_fall_f1: {logs['val_fall_f1']:.4f}"
                    f" - val_direction_macro_f1: {logs['val_direction_macro_f1']:.4f}"
                    f" - val_domain_score: {logs['val_domain_score']:.4f}"
                )

        self.callback = _Callback()


def build_domain_callbacks(
    run_dir: str | Path,
    X_val: np.ndarray,
    y_fall_val: np.ndarray,
    y_direction_val: np.ndarray,
    direction_mask_val: np.ndarray,
    patience: int = 20,
    use_early_stopping: bool = True,
):
    import tensorflow as tf

    run_dir = ensure_dir(run_dir)
    metric_callback = DomainValidationCallback(
        X_val=X_val,
        y_fall_val=y_fall_val,
        y_direction_val=y_direction_val,
        direction_mask_val=direction_mask_val,
    ).callback
    callbacks = [metric_callback]
    if use_early_stopping:
        callbacks.extend(
            [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(run_dir / "model_best.keras"),
                    monitor="val_domain_score",
                    mode="max",
                    save_best_only=True,
                    verbose=1,
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_domain_score",
                    mode="max",
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1,
                ),
            ]
        )
    callbacks.extend(
        [
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
    )
    return callbacks


def flatten_domain_metrics(
    spec: DomainExperimentSpec,
    metrics: dict[str, Any],
    splits: dict[str, Any],
    latency_ms: float | None = None,
) -> dict[str, Any]:
    base = flatten_ablation_metrics(spec.id, {"model_name": "A5WCEFW", "feature_set": "tilt12"}, metrics, latency_ms=latency_ms)
    row: dict[str, Any] = {
        "id": spec.id,
        "experiment_type": spec.experiment_type,
        "model": "A5WCEFW",
        "train_datasets": "+".join(spec.train_datasets),
        "test_datasets": "+".join(spec.test_datasets),
        "test_split": spec.test_split if spec.test_split is not None else "all",
        "n_train": splits["n_train"],
        "n_val": splits["n_val"],
        "n_test": splits["n_test"],
        "n_train_direction": splits["n_train_direction"],
        "n_val_direction": splits["n_val_direction"],
        "n_test_direction": splits["n_test_direction"],
        "note": spec.note,
    }
    row.update(base)
    row["direction_forward_f1"] = _label_f1(metrics, "forward")
    row["direction_backward_f1"] = _label_f1(metrics, "backward")
    row["direction_lateral_f1"] = _label_f1(metrics, "lateral")
    return row


def _label_f1(metrics: dict[str, Any], label: str) -> float | None:
    report = metrics.get("direction", {}).get("classification_report", {})
    if label not in report:
        return None
    return report[label].get("f1-score")


def save_domain_confusion_matrices(metrics: dict[str, Any], output_dir: str | Path, experiment_id: str) -> None:
    fig_dir = ensure_dir(Path(output_dir) / "figures" / "domain_conflict_confusion_matrices" / experiment_id)
    fall_cm = metrics.get("fall", {}).get("confusion_matrix")
    if fall_cm is not None:
        plot_confusion_matrix(fall_cm, ["non_fall", "fall"], f"{experiment_id} fall confusion matrix", fig_dir / "fall.png", show=False)
    direction_cm = metrics.get("direction", {}).get("confusion_matrix")
    if direction_cm is not None:
        plot_confusion_matrix(
            direction_cm,
            ["forward", "backward", "lateral"],
            f"{experiment_id} direction confusion matrix",
            fig_dir / "direction.png",
            show=False,
        )
    for dataset, result in metrics.get("per_dataset", {}).items():
        ds_dir = ensure_dir(fig_dir / str(dataset))
        ds_fall_cm = result.get("fall", {}).get("confusion_matrix")
        if ds_fall_cm is not None:
            plot_confusion_matrix(
                ds_fall_cm,
                ["non_fall", "fall"],
                f"{experiment_id} {dataset} fall confusion matrix",
                ds_dir / "fall.png",
                show=False,
            )
        ds_dir_cm = result.get("direction", {}).get("confusion_matrix")
        if ds_dir_cm is not None:
            plot_confusion_matrix(
                ds_dir_cm,
                ["forward", "backward", "lateral"],
                f"{experiment_id} {dataset} direction confusion matrix",
                ds_dir / "direction.png",
                show=False,
            )


def save_domain_conflict_results(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    reports_dir = ensure_dir(Path(output_dir) / "reports")
    df = pd.DataFrame(rows)
    df.to_csv(reports_dir / "domain_conflict_experiments.csv", index=False)
    (reports_dir / "domain_conflict_experiments.md").write_text(domain_conflict_markdown(df, output_dir), encoding="utf-8")
    update_analysis_notes_from_reports(project_root=Path(output_dir).parent, output_dir=output_dir)


def domain_conflict_markdown(df: pd.DataFrame, output_dir: str | Path) -> str:
    lines = [
        "# Domain Conflict Experiments",
        "",
        "Fixed model: B0 / A5WCEFW = DS-Fall-RD tilt12, task-specific attention, weighted CE fall, weighted CE direction.",
        "Diagnostics use train-only feature scaling per experiment; held-out test datasets are not used for scaler or early stopping.",
        "Dataset-specific runs are evaluated at the final epoch. Train-3-test-1 runs use `val_direction_macro_f1 + val_fall_f1` checkpoint selection.",
        "",
    ]
    if df.empty:
        return "\n".join(lines + ["_No rows yet._", ""]) + "\n"

    ds = df[df["experiment_type"] == "dataset_specific"].copy()
    t3 = df[df["experiment_type"] == "train3_test1"].copy()
    t1 = df[df["experiment_type"] == "train1_testothers"].copy()

    if not ds.empty:
        lines.extend(["## Dataset-Specific Training", "", _table(ds, dataset_specific=True), ""])
    if not t3.empty:
        lines.extend(["## Train-3-Test-1", "", _table(t3), ""])
    if not t1.empty:
        lines.extend(["## Train-1-Test-Others", "", _table(t1), ""])

    lines.extend(["## Diagnosis", ""])
    lines.extend(build_diagnosis_lines(df, output_dir))
    lines.append("")
    return "\n".join(lines)


def build_diagnosis_lines(df: pd.DataFrame, output_dir: str | Path) -> list[str]:
    lines: list[str] = []
    b0_umafall = _load_b0_umafall_direction(output_dir)
    umafall_only = _dataset_specific_direction(df, "umafall")
    if umafall_only is None:
        lines.append("- UMAFall-only direction result is not available yet.")
    elif b0_umafall is None:
        lines.append(f"- UMAFall-only direction macro F1 is {_fmt(umafall_only)}; mixed-training baseline was not found.")
    else:
        delta = umafall_only - b0_umafall
        verdict = "higher than" if delta > 0.05 else "not meaningfully higher than"
        lines.append(
            f"- UMAFall-only direction macro F1 is {_fmt(umafall_only)}, {verdict} mixed-training UMAFall {_fmt(b0_umafall)} "
            f"(delta {_fmt(delta)})."
        )

    t3 = df[df["experiment_type"] == "train3_test1"].copy()
    if not t3.empty and "direction_macro_f1" in t3:
        worst = t3.sort_values("direction_macro_f1", ascending=True, na_position="last").iloc[0]
        best = t3.sort_values("direction_macro_f1", ascending=False, na_position="last").iloc[0]
        lines.append(
            f"- Weakest held-out direction generalization: train {worst.get('train_datasets')} -> test {worst.get('test_datasets')} "
            f"with direction macro F1 {_fmt(worst.get('direction_macro_f1'))}."
        )
        lines.append(
            f"- Strongest held-out direction generalization: train {best.get('train_datasets')} -> test {best.get('test_datasets')} "
            f"with direction macro F1 {_fmt(best.get('direction_macro_f1'))}."
        )

    ds = df[df["experiment_type"] == "dataset_specific"].copy()
    conflict = False
    if not ds.empty and not t3.empty:
        ds_mean = ds["direction_macro_f1"].dropna().mean()
        t3_mean = t3["direction_macro_f1"].dropna().mean()
        conflict = bool(ds_mean - t3_mean > 0.10)
        lines.append(
            f"- Dataset-specific mean direction macro F1 is {_fmt(ds_mean)}; train-3-test-1 mean is {_fmt(t3_mean)}."
        )
    if conflict:
        lines.append("- Evidence supports cross-dataset direction semantic/domain conflict.")
    else:
        lines.append("- Evidence for strong cross-dataset conflict is limited; remaining errors may come from label noise, small direction sample counts, or window/model fit.")

    if umafall_only is not None and b0_umafall is not None:
        if umafall_only > b0_umafall + 0.05:
            lines.append("- UMAFall should be treated as a domain with learnable within-dataset direction but semantic conflict in mixed training.")
        else:
            lines.append("- UMAFall should stay in the realistic multi-dataset setup, but its direction labels/window semantics still need manual protocol review.")

    if not ds.empty:
        clean = ds.sort_values("direction_macro_f1", ascending=False, na_position="last").head(2)
        clean_names = ", ".join(str(v).replace("DS_", "").lower() for v in clean["id"].tolist())
        lines.append(f"- Best candidates for a clean-direction setup from this run: {clean_names}.")
    return lines


def _table(df: pd.DataFrame, dataset_specific: bool = False) -> str:
    cols = [
        "id",
        "train_datasets",
        "test_datasets",
        "n_train_direction",
        "n_test_direction",
        "fall_f1",
        "direction_accuracy",
        "direction_macro_f1",
        "direction_forward_f1",
        "direction_backward_f1",
        "direction_lateral_f1",
    ]
    if dataset_specific:
        cols = [col for col in cols if col != "test_datasets"]
    show = df[[col for col in cols if col in df.columns]].copy()
    for col in show.select_dtypes(include=[float]).columns:
        show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    header = "| " + " | ".join(show.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in show.to_numpy()]
    return "\n".join([header, sep] + rows)


def _dataset_specific_direction(df: pd.DataFrame, dataset: str) -> float | None:
    row = df[(df["experiment_type"] == "dataset_specific") & (df["train_datasets"] == dataset)]
    if row.empty:
        return None
    value = row.iloc[0].get("direction_macro_f1")
    if pd.isna(value):
        return None
    return float(value)


def _load_b0_umafall_direction(output_dir: str | Path) -> float | None:
    path = Path(output_dir) / "reports" / "a5wcefw_after_unit_fix_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    match = df[df["id"].astype(str).str.upper() == "A5WCEFW"]
    if match.empty or "umafall_direction_macro_f1" not in match:
        return None
    value = match.iloc[0]["umafall_direction_macro_f1"]
    if pd.isna(value):
        return None
    return float(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
