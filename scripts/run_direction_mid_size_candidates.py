from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

import run_hybrid_feature_reduction_and_lightweight_direction as hybrid
import run_ml_e1_e7_specialized as mlbase

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf  # noqa: F401

    TENSORFLOW_AVAILABLE = True
    TENSORFLOW_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - environment dependent
    TENSORFLOW_AVAILABLE = False
    TENSORFLOW_IMPORT_ERROR = repr(exc)


DIRECTION_LABELS = ["forward", "backward", "lateral"]
DIR_TO_ID = {name: i for i, name in enumerate(DIRECTION_LABELS)}
BASE_PROTOCOLS = [
    "E1_BITS_TO_BITS",
    "E2_WEDA_TO_WEDA",
    "E3_BITS_WEDA_MIXED",
    "E4_BITS_TO_WEDA",
    "E5_WEDA_TO_BITS",
]
ALL_PROTOCOLS = BASE_PROTOCOLS + ["E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"]
FOCUS_PROTOCOLS = ["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"]


@dataclass
class DirectionRun:
    model: Any
    model_name: str
    params: int
    protocol_id: str
    seed: int
    learning_rate: float
    best_epoch: int
    best_val_macro_f1: float
    best_val_loss: float
    scaler: dict[str, Any]
    complexity: dict[str, Any]
    teacher_note: str = ""
    distill_temperature: float | None = None
    distill_alpha: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mid-size direction edge candidates for DS-Fall-RD Hybrid.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 5e-4])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


if TORCH_AVAILABLE:

    class SepConv1dBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
            super().__init__()
            pad = ((kernel_size - 1) // 2) * dilation
            self.block = nn.Sequential(
                nn.Conv1d(in_ch, in_ch, kernel_size, padding=pad, dilation=dilation, groups=in_ch),
                nn.Conv1d(in_ch, out_ch, 1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.block(x)


    class LDX1WideDSConv(nn.Module):
        def __init__(self, input_channels: int = 12):
            super().__init__()
            self.encoder = nn.Sequential(
                SepConv1dBlock(input_channels, 24, 5),
                SepConv1dBlock(24, 32, 3),
                SepConv1dBlock(32, 48, 3),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(48, 48), nn.ReLU(), nn.Dropout(0.15), nn.Linear(48, 3))

        def forward(self, x):
            return self.head(self.encoder(x.transpose(1, 2)))


    class AttentionPool1d(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.score = nn.Conv1d(channels, 1, 1)

        def forward(self, x):
            alpha = torch.softmax(self.score(x), dim=-1)
            return torch.sum(alpha * x, dim=-1)


    class LDX2WideAttention(nn.Module):
        def __init__(self, input_channels: int = 12):
            super().__init__()
            self.encoder = nn.Sequential(
                SepConv1dBlock(input_channels, 24, 5),
                SepConv1dBlock(24, 32, 3),
                SepConv1dBlock(32, 48, 3),
            )
            self.pool = AttentionPool1d(48)
            self.head = nn.Sequential(nn.Linear(48, 48), nn.ReLU(), nn.Dropout(0.15), nn.Linear(48, 3))

        def forward(self, x):
            h = self.encoder(x.transpose(1, 2))
            return self.head(self.pool(h))


    class DSTCNBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, dilation: int):
            super().__init__()
            self.conv = SepConv1dBlock(in_ch, out_ch, 3, dilation=dilation)
            self.proj = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)

        def forward(self, x):
            return F.relu(self.conv(x) + self.proj(x))


    class LDX3A5Small075(nn.Module):
        def __init__(self):
            super().__init__()
            # tilt12 channel order: ax ay az gx gy gz acc_mag gyro_mag jerk roll pitch tilt_delta
            self.acc_idx = [0, 1, 2, 6, 8, 9, 10, 11]
            self.gyro_idx = [3, 4, 5, 7]
            self.acc_stream = nn.Sequential(
                nn.Conv1d(8, 12, 5, padding=2),
                nn.BatchNorm1d(12),
                nn.ReLU(),
                SepConv1dBlock(12, 18, 5),
                SepConv1dBlock(18, 24, 3),
            )
            self.gyro_stream = nn.Sequential(
                nn.Conv1d(4, 12, 5, padding=2),
                nn.BatchNorm1d(12),
                nn.ReLU(),
                SepConv1dBlock(12, 18, 5),
                SepConv1dBlock(18, 24, 3),
            )
            self.fusion = nn.Sequential(nn.Conv1d(48, 48, 1), nn.BatchNorm1d(48), nn.ReLU())
            self.tcn = nn.Sequential(DSTCNBlock(48, 48, 1), DSTCNBlock(48, 48, 2), DSTCNBlock(48, 72, 4))
            self.pool = AttentionPool1d(72)
            self.head = nn.Sequential(nn.Linear(72, 24), nn.ReLU(), nn.Dropout(0.15), nn.Linear(24, 3))

        def forward(self, x):
            xt = x.transpose(1, 2)
            acc = self.acc_stream(xt[:, self.acc_idx, :])
            gyro = self.gyro_stream(xt[:, self.gyro_idx, :])
            h = self.tcn(self.fusion(torch.cat([acc, gyro], dim=1)))
            return self.head(self.pool(h))


def model_specs(quick: bool = False) -> list[dict[str, Any]]:
    specs = [
        {"model_name": "LDX1_D1_Wide_DSConv", "builder": lambda: LDX1WideDSConv(), "distill": False},
        {"model_name": "LDX2_D1_Wide_Attention", "builder": lambda: LDX2WideAttention(), "distill": False},
        {"model_name": "LDX3_A5_Direction_Small_075", "builder": lambda: LDX3A5Small075(), "distill": False},
        {
            "model_name": "LDX4_D1_Wide_Attention_Distill_T2_A050",
            "builder": lambda: LDX2WideAttention(),
            "distill": True,
            "temperature": 2.0,
            "alpha": 0.5,
        },
        {
            "model_name": "LDX4_D1_Wide_Attention_Distill_T3_A050",
            "builder": lambda: LDX2WideAttention(),
            "distill": True,
            "temperature": 3.0,
            "alpha": 0.5,
        },
        {
            "model_name": "LDX4_D1_Wide_Attention_Distill_T2_A070",
            "builder": lambda: LDX2WideAttention(),
            "distill": True,
            "temperature": 2.0,
            "alpha": 0.7,
        },
    ]
    return specs[:2] if quick else specs


def canonical_train_protocol(protocol_id: str) -> str:
    return "E3_BITS_WEDA_MIXED" if protocol_id in {"E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"} else protocol_id


def direction_mask(df: pd.DataFrame) -> np.ndarray:
    return mlbase.direction_train_mask(df)


def load_data(repo_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = hybrid.load_summary_feature_matrix(repo_root)
    X_temporal, _ = hybrid.load_temporal_data(repo_root, df)
    if X_temporal.shape[1:] != (50, 12):
        raise ValueError(f"Expected tilt12 temporal shape (N,50,12), got {X_temporal.shape}")
    return df, X_temporal.astype(np.float32)


def teacher_probs_for_mask(repo_root: Path, df: pd.DataFrame, protocol_id: str, mask: np.ndarray) -> tuple[np.ndarray, str]:
    rows = df.loc[mask].reset_index(drop=True)
    y = rows["direction_id"].astype(int).to_numpy()
    fallback = np.eye(3, dtype=np.float32)[y]
    pred_path = repo_root / "artifacts" / "experiments_25hz_event" / protocol_id / "predictions.csv"
    if not pred_path.exists():
        return fallback, f"hard_label_fallback_no_a5_prediction_file:{protocol_id}"
    pred = pd.read_csv(pred_path)
    needed = rows[["window_id"]].merge(
        pred[["window_id", "pred_direction", "direction_confidence"]],
        on="window_id",
        how="left",
        validate="one_to_one",
    )
    if needed["pred_direction"].isna().any():
        missing = int(needed["pred_direction"].isna().sum())
        return fallback, f"hard_label_fallback_missing_teacher_rows:{missing}"
    probs = np.full((len(needed), 3), 0.0, dtype=np.float32)
    pred_ids = needed["pred_direction"].astype(str).str.lower().map(DIR_TO_ID).fillna(-1).astype(int).to_numpy()
    conf = needed["direction_confidence"].astype(float).fillna(1.0 / 3.0).clip(1.0 / 3.0, 1.0).to_numpy()
    for i, (cls, c) in enumerate(zip(pred_ids, conf)):
        if cls < 0:
            probs[i] = fallback[i]
        else:
            probs[i] = (1.0 - c) / 2.0
            probs[i, cls] = c
    return probs, f"a5_pred_direction_confidence:{protocol_id}"


def standardize_temporal(X: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mean = X[train_mask].mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = (X[train_mask].std(axis=(0, 1), keepdims=True) + 1e-6).astype(np.float32)
    return ((X[train_mask] - mean) / std).astype(np.float32), ((X[val_mask] - mean) / std).astype(np.float32), {
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }


def compute_params(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def estimate_complexity(model_name: str, params: int) -> dict[str, Any]:
    fp32 = params * 4 / 1024
    int8 = params / 1024
    edge = "high" if params <= 12_000 else "medium-high" if params <= 30_000 else "medium"
    return {
        "estimated_fp32_kb": fp32,
        "estimated_int8_kb": int8,
        "macs_estimate": np.nan,
        "tflite_convert": "not_available_tensorflow_import_failed" if not TENSORFLOW_AVAILABLE else "not_attempted_pytorch_model",
        "int8_tflite_convert": "not_available_tensorflow_import_failed" if not TENSORFLOW_AVAILABLE else "not_attempted_pytorch_model",
        "edge_suitability": edge,
        "complexity_note": f"{model_name}; PyTorch implementation; TensorFlow available={TENSORFLOW_AVAILABLE}",
    }


def train_direction_model(
    repo_root: Path,
    df: pd.DataFrame,
    X_temporal: np.ndarray,
    protocol_id: str,
    spec: dict[str, Any],
    seed: int,
    lr: float,
    epochs: int,
    batch_size: int,
) -> DirectionRun:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available.")
    set_seed(seed)
    train_mask, val_mask, _ = hybrid.get_protocol_splits(df, protocol_id)
    train_mask = train_mask & direction_mask(df)
    val_mask = val_mask & direction_mask(df)
    if int(train_mask.sum()) < 6 or int(val_mask.sum()) < 3:
        raise ValueError(f"Insufficient direction samples for {protocol_id}: train={train_mask.sum()}, val={val_mask.sum()}")
    X_train_np, X_val_np, scaler = standardize_temporal(X_temporal, train_mask, val_mask)
    y_train = df.loc[train_mask, "direction_id"].astype(int).to_numpy()
    y_val = df.loc[val_mask, "direction_id"].astype(int).to_numpy()
    teacher_note = ""
    teacher_train = None
    if spec.get("distill"):
        teacher_train, teacher_note = teacher_probs_for_mask(repo_root, df, protocol_id, train_mask)
    classes = np.array([0, 1, 2])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    model = spec["builder"]()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)
    tensors = [torch.tensor(X_train_np, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)]
    if teacher_train is not None:
        tensors.append(torch.tensor(teacher_train, dtype=torch.float32))
    train_loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True)
    X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    best_state = None
    best_macro = -1.0
    best_loss = math.inf
    best_epoch = 0
    no_improve = 0
    patience = 15
    temp = float(spec.get("temperature", 1.0))
    alpha = float(spec.get("alpha", 0.0))
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            xb, yb = batch[0], batch[1]
            optimizer.zero_grad()
            logits = model(xb)
            ce = criterion(logits, yb)
            if spec.get("distill"):
                tb = batch[2]
                log_student = F.log_softmax(logits / temp, dim=1)
                teacher = F.softmax(torch.log(tb.clamp_min(1e-6)) / temp, dim=1)
                kd = F.kl_div(log_student, teacher, reduction="batchmean") * (temp * temp)
                loss = ce + alpha * kd
            else:
                loss = ce
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = float(criterion(val_logits, y_val_t).item())
            val_pred = torch.argmax(val_logits, dim=1).cpu().numpy()
        val_macro = float(f1_score(y_val, val_pred, labels=[0, 1, 2], average="macro", zero_division=0))
        scheduler.step(val_loss)
        if val_macro > best_macro + 1e-8 or (abs(val_macro - best_macro) <= 1e-8 and val_loss < best_loss):
            best_macro = val_macro
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    params = compute_params(model)
    return DirectionRun(
        model=model,
        model_name=spec["model_name"],
        params=params,
        protocol_id=protocol_id,
        seed=seed,
        learning_rate=lr,
        best_epoch=best_epoch,
        best_val_macro_f1=best_macro,
        best_val_loss=best_loss,
        scaler=scaler,
        complexity=estimate_complexity(spec["model_name"], params),
        teacher_note=teacher_note,
        distill_temperature=spec.get("temperature"),
        distill_alpha=spec.get("alpha"),
    )


def predict_direction(run: DirectionRun, X_temporal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mean = np.asarray(run.scaler["mean"], dtype=np.float32)
    std = np.asarray(run.scaler["std"], dtype=np.float32)
    X = ((X_temporal[mask] - mean) / std).astype(np.float32)
    run.model.eval()
    with torch.no_grad():
        logits = run.model(torch.tensor(X, dtype=torch.float32))
        return torch.argmax(logits, dim=1).cpu().numpy().astype(int)


def evaluate_direction(
    df: pd.DataFrame,
    mask: np.ndarray,
    pred_all: np.ndarray,
    run: DirectionRun,
    run_id: str,
    experiment_id: str,
    a5_direction: pd.DataFrame,
    previous: pd.DataFrame,
) -> dict[str, Any]:
    sub = df.loc[mask].reset_index(drop=True)
    sup = direction_mask(sub)
    y = sub.loc[sup, "direction_id"].astype(int).to_numpy()
    pred = pred_all[sup]
    cm = confusion_matrix(y, pred, labels=[0, 1, 2]) if len(y) else np.zeros((3, 3), dtype=int)
    per = f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0) if len(y) else [np.nan, np.nan, np.nan]
    macro = f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0) if len(y) else np.nan
    a5_rows = a5_direction[a5_direction["experiment_id"].eq(experiment_id)]
    a5_f1 = float(a5_rows.iloc[0]["direction_macro_f1"]) if not a5_rows.empty else np.nan
    d1_f1 = previous_lookup(previous, "D1_Tiny_DSConv_Direction", experiment_id)
    d2_f1 = previous_lookup(previous, "D2_Micro_TCN_Direction", experiment_id)
    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "train_protocol": run.protocol_id,
        "model_name": run.model_name,
        "seed": run.seed,
        "learning_rate": run.learning_rate,
        "best_epoch": run.best_epoch,
        "best_val_macro_f1": run.best_val_macro_f1,
        "best_val_loss": run.best_val_loss,
        "direction_macro_f1": macro,
        "direction_accuracy": accuracy_score(y, pred) if len(y) else np.nan,
        "forward_f1": per[0],
        "backward_f1": per[1],
        "lateral_f1": per[2],
        "direction_n_supervised": len(y),
        "A5_direction_macro_f1": a5_f1,
        "D1_tiny_direction_macro_f1": d1_f1,
        "D2_micro_tcn_direction_macro_f1": d2_f1,
        "delta_vs_A5": macro - a5_f1 if not pd.isna(a5_f1) else np.nan,
        "delta_vs_D1": macro - d1_f1 if not pd.isna(d1_f1) else np.nan,
        "delta_vs_D2": macro - d2_f1 if not pd.isna(d2_f1) else np.nan,
        "params": run.params,
        "estimated_fp32_kb": run.params * 4 / 1024,
        "estimated_int8_kb": run.params / 1024,
        "distill_temperature": run.distill_temperature,
        "distill_alpha": run.distill_alpha,
        "teacher_note": run.teacher_note,
        "cm_00": int(cm[0, 0]),
        "cm_01": int(cm[0, 1]),
        "cm_02": int(cm[0, 2]),
        "cm_10": int(cm[1, 0]),
        "cm_11": int(cm[1, 1]),
        "cm_12": int(cm[1, 2]),
        "cm_20": int(cm[2, 0]),
        "cm_21": int(cm[2, 1]),
        "cm_22": int(cm[2, 2]),
        **run.complexity,
    }


def previous_lookup(previous: pd.DataFrame, model_name: str, experiment_id: str) -> float:
    if previous.empty:
        return math.nan
    rows = previous[(previous["model_name"].eq(model_name)) & (previous["experiment_id"].eq(experiment_id))]
    if rows.empty:
        return math.nan
    rows = rows.sort_values(["direction_macro_f1", "learning_rate"], ascending=[False, True])
    return float(rows.iloc[0]["direction_macro_f1"])


def load_previous_lightweight(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "outputs" / "reports" / "hybrid_feature_reduction_direction_lightweight" / "lightweight_direction_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_fixed_fall_predictions(repo_root: Path, df: pd.DataFrame) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    predictions: dict[str, np.ndarray] = {}
    warnings: list[dict[str, Any]] = []
    model_dir = repo_root / "outputs" / "models" / "hybrid_feature_reduction_direction_lightweight"
    for eval_id in ALL_PROTOCOLS:
        train_protocol = canonical_train_protocol(eval_id)
        pkl = model_dir / f"FALLRED_GradientBoosting_FallNoTiming_Full_132_{train_protocol}.pkl"
        if not pkl.exists():
            warnings.append({"stage": "load_fall", "experiment_id": eval_id, "error": f"missing {pkl}"})
            continue
        import pickle
        import __main__

        # The fall artifacts were produced by an executable script, so pickle
        # records the dataclasses under __main__. Provide compatible aliases
        # instead of retraining or changing the fixed fall branch.
        setattr(__main__, "FallTrained", hybrid.FallTrained)
        setattr(__main__, "FallModelSpec", hybrid.FallModelSpec)

        with pkl.open("rb") as f:
            trained = pickle.load(f)
        _, _, mask = hybrid.get_protocol_splits(df, eval_id)
        X = df.loc[mask, trained.features].astype(float).to_numpy()
        scores = hybrid.fall_scores(trained.model, X)
        predictions[eval_id] = (scores >= trained.threshold).astype(int)
    return predictions, warnings


def evaluate_e2e_direction(
    df: pd.DataFrame,
    mask: np.ndarray,
    fall_pred: np.ndarray,
    direction_pred: np.ndarray,
    run_id: str,
    experiment_id: str,
    train_protocol: str,
) -> dict[str, Any]:
    row = hybrid.evaluate_e2e_direction(
        df=df,
        mask=mask,
        fall_pred=fall_pred,
        direction_pred=direction_pred,
        pipeline=run_id,
        experiment_id=experiment_id,
        train_protocol=train_protocol,
        source="fixed_GB_FallNoTiming_Full_132",
    )
    row["fall_branch"] = "GradientBoosting_FallNoTiming_Full_132"
    return row


def save_direction_cm(row: dict[str, Any], out_dir: Path) -> None:
    ensure_dir(out_dir)
    cm = np.array(
        [
            [row["cm_00"], row["cm_01"], row["cm_02"]],
            [row["cm_10"], row["cm_11"], row["cm_12"]],
            [row["cm_20"], row["cm_21"], row["cm_22"]],
        ],
        dtype=int,
    )
    plt.figure(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=DIRECTION_LABELS, yticklabels=DIRECTION_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{row['model_name']} {row['experiment_id']}")
    plt.tight_layout()
    path = out_dir / f"{safe_name(row['run_id'])}_{row['experiment_id']}_direction.png"
    plt.savefig(path, dpi=140)
    plt.close()


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))


def best_by_protocol(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    rows = []
    for exp, group in results.groupby("experiment_id"):
        rows.append(group.sort_values(["direction_macro_f1", "direction_accuracy", "params"], ascending=[False, False, True]).iloc[0])
    return pd.DataFrame(rows).sort_values("experiment_id")


def build_vs_table(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    focus = results[results["experiment_id"].isin(FOCUS_PROTOCOLS)].copy()
    agg = focus.groupby(["model_name", "seed", "learning_rate", "params"], as_index=False).agg(
        avg_direction_macro_f1=("direction_macro_f1", "mean"),
        e3_direction_macro_f1=("direction_macro_f1", lambda s: s[focus.loc[s.index, "experiment_id"].eq("E3_BITS_WEDA_MIXED")].mean()),
        e6_direction_macro_f1=("direction_macro_f1", lambda s: s[focus.loc[s.index, "experiment_id"].eq("E6_E3_MIXED_TEST_BITS")].mean()),
        e7_direction_macro_f1=("direction_macro_f1", lambda s: s[focus.loc[s.index, "experiment_id"].eq("E7_E3_MIXED_TEST_WEDA")].mean()),
        avg_delta_vs_a5=("delta_vs_A5", "mean"),
        avg_delta_vs_d1=("delta_vs_D1", "mean"),
        avg_delta_vs_d2=("delta_vs_D2", "mean"),
    )
    agg["passes_main_replacement"] = (
        (agg["avg_direction_macro_f1"] >= 0.870)
        & (agg["e3_direction_macro_f1"] >= 0.86)
        & (agg["e6_direction_macro_f1"] >= 0.88)
        & (agg["e7_direction_macro_f1"] >= 0.83)
        & (agg["params"] <= 30_000)
    )
    agg["passes_edge_ablation"] = (
        (agg["avg_direction_macro_f1"] >= 0.850) & (agg["params"] <= 12_000) & (agg["e7_direction_macro_f1"] >= 0.85)
    )
    return agg.sort_values(["avg_direction_macro_f1", "e3_direction_macro_f1", "params"], ascending=[False, False, True])


def make_plots(report_dir: Path, figure_dir: Path) -> None:
    vs_path = report_dir / "direction_mid_size_vs_a5_and_d1.csv"
    results_path = report_dir / "direction_mid_size_results.csv"
    if vs_path.exists():
        vs = pd.read_csv(vs_path)
        if not vs.empty:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(data=vs, x="params", y="avg_direction_macro_f1", hue="model_name", style="learning_rate", s=90)
            plt.axhline(0.8902, color="black", linestyle="--", linewidth=1, label="A5 ref avg")
            plt.axhline(0.850, color="gray", linestyle=":", linewidth=1, label="edge target")
            plt.tight_layout()
            plt.savefig(figure_dir / "params_vs_avg_f1.png", dpi=150)
            plt.close()
    if results_path.exists():
        res = pd.read_csv(results_path)
        focus = res[res["experiment_id"].isin(FOCUS_PROTOCOLS)].copy()
        if not focus.empty:
            best = focus.sort_values(["direction_macro_f1"], ascending=False).groupby(["model_name", "experiment_id"], as_index=False).head(1)
            plt.figure(figsize=(12, 5))
            sns.barplot(data=best, x="experiment_id", y="direction_macro_f1", hue="model_name")
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.savefig(figure_dir / "e3_e6_e7_direction_bar.png", dpi=150)
            plt.close()


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    x = df.head(max_rows).copy()
    for col in x.columns:
        if pd.api.types.is_float_dtype(x[col]):
            x[col] = x[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
        elif pd.api.types.is_bool_dtype(x[col]):
            x[col] = x[col].map(lambda v: "yes" if bool(v) else "no")
        elif pd.api.types.is_integer_dtype(x[col]):
            x[col] = x[col].map(lambda v: "" if pd.isna(v) else str(int(v)))
        else:
            x[col] = x[col].astype(str).replace("nan", "")
    cols = list(x.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in x.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines)


def write_summary(
    report_dir: Path,
    results: pd.DataFrame,
    best: pd.DataFrame,
    vs: pd.DataFrame,
    e2e: pd.DataFrame,
    complexity: pd.DataFrame,
    failed: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Direction Mid-Size Candidates Summary")
    lines.append("")
    lines.append("Benchmark: BITS/WEDA only, 25 Hz, 2-second event-centered windows, temporal `tilt12` input `50 x 12`. Fall branch is fixed to `GradientBoosting + FallNoTiming_Full_132`; no FullTiming and no new dataset.")
    lines.append("")
    lines.append("## Run Status")
    lines.append("")
    lines.append(f"- Direction result rows: `{len(results)}`.")
    lines.append(f"- Failed rows: `{len(failed)}`.")
    lines.append(f"- TensorFlow available: `{TENSORFLOW_AVAILABLE}`; import error: `{TENSORFLOW_IMPORT_ERROR or 'none'}`.")
    lines.append("- TFLite/INT8 export is marked unavailable when TensorFlow import fails; PyTorch `.pt` models are saved.")
    lines.append("- LDX4 distillation limitation: train/val A5 teacher probabilities are not available in saved artifacts, so the script falls back to hard true-label teacher for missing rows and logs this in `teacher_note`.")
    lines.append("")
    lines.append("## Best E3/E6/E7 Average")
    lines.append("")
    show_cols = [
        "model_name",
        "seed",
        "learning_rate",
        "params",
        "avg_direction_macro_f1",
        "e3_direction_macro_f1",
        "e6_direction_macro_f1",
        "e7_direction_macro_f1",
        "avg_delta_vs_a5",
        "avg_delta_vs_d1",
        "passes_main_replacement",
        "passes_edge_ablation",
    ]
    lines.append(md_table(vs[show_cols], 30))
    lines.append("")
    if vs["seed"].nunique() > 1:
        seed_agg = (
            vs.groupby(["model_name", "learning_rate", "params"], as_index=False)
            .agg(
                avg_f1_mean=("avg_direction_macro_f1", "mean"),
                avg_f1_std=("avg_direction_macro_f1", "std"),
                e3_mean=("e3_direction_macro_f1", "mean"),
                e6_mean=("e6_direction_macro_f1", "mean"),
                e7_mean=("e7_direction_macro_f1", "mean"),
                pass_main_count=("passes_main_replacement", "sum"),
                pass_edge_count=("passes_edge_ablation", "sum"),
            )
            .sort_values(["avg_f1_mean", "e3_mean", "params"], ascending=[False, False, True])
        )
        lines.append("## Seed Robustness")
        lines.append("")
        lines.append(md_table(seed_agg, 20))
        lines.append("")
    lines.append("## Best By Protocol")
    lines.append("")
    lines.append(md_table(best[["experiment_id", "model_name", "seed", "learning_rate", "params", "direction_macro_f1", "direction_accuracy", "forward_f1", "backward_f1", "lateral_f1"]], 20))
    lines.append("")
    lines.append("## End-to-End Direction")
    lines.append("")
    if not e2e.empty:
        focus = e2e[e2e["experiment_id"].isin(FOCUS_PROTOCOLS)].copy()
        agg = focus.groupby(["pipeline", "train_protocol"], as_index=False).agg(
            avg_E2E_direction_macro_f1=("E2E_Direction_Macro_F1", "mean"),
            e3_E2E=("E2E_Direction_Macro_F1", lambda s: s[focus.loc[s.index, "experiment_id"].eq("E3_BITS_WEDA_MIXED")].mean()),
            e6_E2E=("E2E_Direction_Macro_F1", lambda s: s[focus.loc[s.index, "experiment_id"].eq("E6_E3_MIXED_TEST_BITS")].mean()),
            e7_E2E=("E2E_Direction_Macro_F1", lambda s: s[focus.loc[s.index, "experiment_id"].eq("E7_E3_MIXED_TEST_WEDA")].mean()),
            coverage=("coverage", "mean"),
        ).sort_values("avg_E2E_direction_macro_f1", ascending=False)
        lines.append(md_table(agg, 20))
    else:
        lines.append("_No E2E rows._")
    lines.append("")
    lines.append("## Complexity")
    lines.append("")
    lines.append(md_table(complexity.drop_duplicates(["model_name", "learning_rate"])[["model_name", "learning_rate", "params", "estimated_fp32_kb", "estimated_int8_kb", "tflite_convert", "int8_tflite_convert", "edge_suitability"]], 20))
    lines.append("")
    lines.append("## Answers")
    lines.append("")
    if vs.empty:
        lines.append("No successful candidate rows were produced.")
    else:
        top = vs.iloc[0]
        robust = None
        if vs["seed"].nunique() > 1:
            robust = (
                vs.groupby(["model_name", "learning_rate", "params"], as_index=False)
                .agg(
                    avg_direction_macro_f1=("avg_direction_macro_f1", "mean"),
                    e3_direction_macro_f1=("e3_direction_macro_f1", "mean"),
                    e6_direction_macro_f1=("e6_direction_macro_f1", "mean"),
                    e7_direction_macro_f1=("e7_direction_macro_f1", "mean"),
                    avg_delta_vs_a5=("avg_delta_vs_a5", "mean"),
                    avg_delta_vs_d1=("avg_delta_vs_d1", "mean"),
                )
                .sort_values(["avg_direction_macro_f1", "e3_direction_macro_f1", "params"], ascending=[False, False, True])
                .iloc[0]
            )
        main = vs[vs["passes_main_replacement"]]
        edge = vs[vs["passes_edge_ablation"]]
        robust_main = pd.DataFrame()
        robust_edge = pd.DataFrame()
        if vs["seed"].nunique() > 1:
            nseed = int(vs["seed"].nunique())
            robust_check = (
                vs.groupby(["model_name", "learning_rate", "params"], as_index=False)
                .agg(
                    avg_mean=("avg_direction_macro_f1", "mean"),
                    e3_mean=("e3_direction_macro_f1", "mean"),
                    e6_mean=("e6_direction_macro_f1", "mean"),
                    e7_mean=("e7_direction_macro_f1", "mean"),
                    pass_main_count=("passes_main_replacement", "sum"),
                    pass_edge_count=("passes_edge_ablation", "sum"),
                )
                .sort_values(["avg_mean", "e3_mean", "params"], ascending=[False, False, True])
            )
            robust_main = robust_check[
                (robust_check["avg_mean"] >= 0.870)
                & (robust_check["e3_mean"] >= 0.86)
                & (robust_check["e6_mean"] >= 0.88)
                & (robust_check["e7_mean"] >= 0.83)
                & (robust_check["params"] <= 30_000)
                & (robust_check["pass_main_count"] == nseed)
            ]
            robust_edge = robust_check[
                (robust_check["avg_mean"] >= 0.850)
                & (robust_check["params"] <= 12_000)
                & (robust_check["e7_mean"] >= 0.85)
                & (robust_check["pass_edge_count"] == nseed)
            ]
        under30 = vs[vs["params"] <= 30_000].sort_values("avg_direction_macro_f1", ascending=False).head(1)
        under12 = vs[vs["params"] <= 12_000].sort_values("avg_direction_macro_f1", ascending=False).head(1)
        e6_best = vs.sort_values("e6_direction_macro_f1", ascending=False).iloc[0]
        e3_best = vs.sort_values("e3_direction_macro_f1", ascending=False).iloc[0]
        e7_best = vs.sort_values("e7_direction_macro_f1", ascending=False).iloc[0]
        attention_plain = compare_attention(vs)
        distill_plain = compare_distill(vs)
        lines.append(f"1. Increasing params above D1 improved over the previous D1 only if `avg_delta_vs_d1` is positive. Best run is `{top['model_name']}` with avg `{top['avg_direction_macro_f1']:.4f}` and delta vs D1 `{top['avg_delta_vs_d1']:.4f}`.")
        if robust is not None:
            lines.append(f"   Across seeds, best mean is `{robust['model_name']}` lr `{robust['learning_rate']}` with mean avg F1 `{robust['avg_direction_macro_f1']:.4f}` and mean delta vs D1 `{robust['avg_delta_vs_d1']:.4f}`.")
        lines.append(f"2. Best params/F1 trade-off: `{top['model_name']}` at `{int(top['params'])}` params, avg F1 `{top['avg_direction_macro_f1']:.4f}`.")
        lines.append(f"3. Closest to A5 under 30k params: `{under30.iloc[0]['model_name']}` with avg F1 `{under30.iloc[0]['avg_direction_macro_f1']:.4f}` and delta vs A5 `{under30.iloc[0]['avg_delta_vs_a5']:.4f}`.")
        lines.append(f"4. Best under 12k params: `{under12.iloc[0]['model_name']}` with avg F1 `{under12.iloc[0]['avg_direction_macro_f1']:.4f}`.")
        lines.append(f"5. Attention pooling effect: {attention_plain}.")
        lines.append(f"6. Distillation effect: {distill_plain}.")
        lines.append(f"7. Best E6 BITS candidate: `{e6_best['model_name']}` with E6 F1 `{e6_best['e6_direction_macro_f1']:.4f}`.")
        lines.append(f"8. Best E3 mixed candidate: `{e3_best['model_name']}` with E3 F1 `{e3_best['e3_direction_macro_f1']:.4f}`.")
        lines.append(f"9. Best E7 WEDA candidate: `{e7_best['model_name']}` with E7 F1 `{e7_best['e7_direction_macro_f1']:.4f}`.")
        if vs["seed"].nunique() > 1:
            if not robust_main.empty:
                lines.append(f"10. Robust multi-seed main replacement candidate exists: `{robust_main.iloc[0]['model_name']}`.")
            elif not main.empty:
                lines.append(f"10. Single-seed main candidate exists (`{main.iloc[0]['model_name']}`), but no candidate passes the main replacement criteria robustly across all seeds. Keep A5 as the main direction expert unless LDX1 stability is improved or a fixed seed protocol is explicitly justified.")
            else:
                lines.append("10. No candidate satisfies all main replacement criteria, so A5 direction should remain the main Hybrid direction expert.")
            if not robust_edge.empty:
                lines.append(f"11. Robust edge ablation candidate: `{robust_edge.iloc[0]['model_name']}`.")
            elif not edge.empty:
                lines.append(f"11. Single-seed edge candidate: `{edge.iloc[0]['model_name']}`. Report it as the best mid-size edge ablation with seed-sensitivity caveat.")
            else:
                lines.append(f"11. No candidate satisfies the formal edge threshold; report `{top['model_name']}` as the best mid-size edge ablation if it improves over D1, otherwise keep D1 as the edge ablation.")
        else:
            if not main.empty:
                lines.append(f"10. Main replacement candidate exists: `{main.iloc[0]['model_name']}`.")
            else:
                lines.append("10. No candidate satisfies all main replacement criteria, so A5 direction should remain the main Hybrid direction expert.")
            if not edge.empty:
                lines.append(f"11. Edge ablation candidate: `{edge.iloc[0]['model_name']}`.")
            else:
                lines.append(f"11. No candidate satisfies the formal edge threshold; report `{top['model_name']}` as the best mid-size edge ablation if it improves over D1, otherwise keep D1 as the edge ablation.")
    lines.append("")
    if not failed.empty:
        lines.append("## Failed Runs")
        lines.append("")
        lines.append(md_table(failed, 50))
        lines.append("")
    lines.append("## Files")
    lines.append("")
    for name in [
        "direction_mid_size_results.csv",
        "direction_mid_size_best_by_protocol.csv",
        "direction_mid_size_vs_a5_and_d1.csv",
        "direction_mid_size_e2e_results.csv",
        "direction_mid_size_complexity.csv",
        "failed_runs.csv",
        "model_configs.json",
    ]:
        lines.append(f"- `{report_dir / name}`")
    (report_dir / "direction_mid_size_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_attention(vs: pd.DataFrame) -> str:
    plain = vs[vs["model_name"].eq("LDX1_D1_Wide_DSConv")]
    att = vs[vs["model_name"].eq("LDX2_D1_Wide_Attention")]
    if plain.empty or att.empty:
        return "not enough rows to compare LDX1 and LDX2"
    plain_best = plain["avg_direction_macro_f1"].max()
    att_best = att["avg_direction_macro_f1"].max()
    delta = att_best - plain_best
    return f"LDX2 attention best avg {att_best:.4f} vs LDX1 GAP best avg {plain_best:.4f}, delta {delta:.4f}"


def compare_distill(vs: pd.DataFrame) -> str:
    att = vs[vs["model_name"].eq("LDX2_D1_Wide_Attention")]
    dist = vs[vs["model_name"].astype(str).str.startswith("LDX4_D1_Wide_Attention_Distill")]
    if att.empty or dist.empty:
        return "not enough rows to compare LDX2 and LDX4"
    att_best = att["avg_direction_macro_f1"].max()
    dist_best = dist["avg_direction_macro_f1"].max()
    delta = dist_best - att_best
    return f"best LDX4 distill avg {dist_best:.4f} vs LDX2 non-distill avg {att_best:.4f}, delta {delta:.4f}; teacher fallback must be considered"


def main() -> None:
    args = parse_args()
    if not args.run_all:
        print("Use --run-all to execute the requested experiment batch.")
        return
    repo_root = Path(args.repo_root).resolve()
    report_dir = ensure_dir(repo_root / "outputs" / "reports" / "direction_mid_size_candidates")
    figure_dir = ensure_dir(repo_root / "outputs" / "figures" / "direction_mid_size_candidates")
    model_dir = ensure_dir(repo_root / "outputs" / "models" / "direction_mid_size_candidates")
    cm_dir = ensure_dir(figure_dir / "confusion_matrices")
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for this script in the current TensorFlow-broken environment.")
    seeds = args.seeds or ([args.seed] if args.seed is not None else [42])
    print("[1/6] Loading BITS/WEDA summary and temporal tilt12 data")
    df, X_temporal = load_data(repo_root)
    a5_fall, a5_direction = mlbase.load_a5_reference_rows()
    previous = load_previous_lightweight(repo_root)
    print("[2/6] Loading fixed GB FallNoTiming_Full_132 predictions for E2E")
    fall_predictions, fall_warnings = load_fixed_fall_predictions(repo_root, df)
    specs = model_specs(args.quick)
    protocols = ["E3_BITS_WEDA_MIXED"] if args.quick else BASE_PROTOCOLS
    lrs = [args.lrs[0]] if args.quick else args.lrs
    results: list[dict[str, Any]] = []
    e2e_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = list(fall_warnings)
    model_configs: list[dict[str, Any]] = []
    trained_cache: dict[tuple[str, str, int, float], DirectionRun] = {}
    print(f"[3/6] Training {len(specs)} candidate specs across protocols={protocols}, seeds={seeds}, lrs={lrs}")
    for spec in specs:
        for seed in seeds:
            for lr in lrs:
                for protocol_id in protocols:
                    try:
                        print(f"[train] {spec['model_name']} protocol={protocol_id} seed={seed} lr={lr}")
                        run = train_direction_model(repo_root, df, X_temporal, protocol_id, spec, seed, lr, args.epochs, args.batch_size)
                        trained_cache[(run.model_name, protocol_id, seed, lr)] = run
                        state_path = model_dir / f"{run.model_name}_{protocol_id}_seed{seed}_lr{lr}.pt"
                        torch.save(
                            {
                                "model_name": run.model_name,
                                "state_dict": run.model.state_dict(),
                                "scaler": run.scaler,
                                "params": run.params,
                                "seed": seed,
                                "learning_rate": lr,
                                "teacher_note": run.teacher_note,
                            },
                            state_path,
                        )
                        complexity_rows.append(
                            {
                                "model_name": run.model_name,
                                "protocol_id": protocol_id,
                                "seed": seed,
                                "learning_rate": lr,
                                "params": run.params,
                                "best_val_macro_f1": run.best_val_macro_f1,
                                "teacher_note": run.teacher_note,
                                **run.complexity,
                            }
                        )
                        train_mask, val_mask, _ = hybrid.get_protocol_splits(df, protocol_id)
                        dm = direction_mask(df)
                        model_configs.append(
                            {
                                "model_name": run.model_name,
                                "protocol_id": protocol_id,
                                "seed": seed,
                                "learning_rate": lr,
                                "batch_size": args.batch_size,
                                "epochs_requested": args.epochs,
                                "best_epoch": run.best_epoch,
                                "params": run.params,
                                "train_direction_count": int((train_mask & dm).sum()),
                                "val_direction_count": int((val_mask & dm).sum()),
                                "class_counts_train": df.loc[train_mask & dm, "direction_id"].value_counts().sort_index().to_dict(),
                                "distill_temperature": run.distill_temperature,
                                "distill_alpha": run.distill_alpha,
                                "teacher_note": run.teacher_note,
                                "state_path": str(state_path),
                            }
                        )
                    except Exception as exc:
                        failed.append(
                            {
                                "stage": "train_direction",
                                "model_name": spec["model_name"],
                                "protocol_id": protocol_id,
                                "seed": seed,
                                "learning_rate": lr,
                                "error": repr(exc),
                            }
                        )
    print("[4/6] Evaluating E1-E7")
    eval_protocols = ["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"] if args.quick else ALL_PROTOCOLS
    for (model_name, protocol_id, seed, lr), run in trained_cache.items():
        for eval_id in eval_protocols:
            if canonical_train_protocol(eval_id) != protocol_id:
                continue
            try:
                _, _, mask = hybrid.get_protocol_splits(df, eval_id)
                pred = predict_direction(run, X_temporal, mask)
                run_id = f"{model_name}_{protocol_id}_seed{seed}_lr{lr}"
                row = evaluate_direction(df, mask, pred, run, run_id, eval_id, a5_direction, previous)
                results.append(row)
                save_direction_cm(row, cm_dir)
                fall_pred = fall_predictions.get(eval_id)
                if fall_pred is not None:
                    e2e_rows.append(evaluate_e2e_direction(df, mask, fall_pred, pred, run_id, eval_id, protocol_id))
            except Exception as exc:
                failed.append(
                    {
                        "stage": "eval_direction",
                        "model_name": model_name,
                        "protocol_id": eval_id,
                        "seed": seed,
                        "learning_rate": lr,
                        "error": repr(exc),
                    }
                )
    print("[5/6] Saving reports and figures")
    results_df = pd.DataFrame(results)
    best_df = best_by_protocol(results_df)
    vs_df = build_vs_table(results_df)
    e2e_df = pd.DataFrame(e2e_rows)
    complexity_df = pd.DataFrame(complexity_rows)
    failed_df = pd.DataFrame(failed)
    if failed_df.empty:
        failed_df = pd.DataFrame(columns=["stage", "model_name", "protocol_id", "seed", "learning_rate", "error"])
    results_df.to_csv(report_dir / "direction_mid_size_results.csv", index=False)
    best_df.to_csv(report_dir / "direction_mid_size_best_by_protocol.csv", index=False)
    vs_df.to_csv(report_dir / "direction_mid_size_vs_a5_and_d1.csv", index=False)
    e2e_df.to_csv(report_dir / "direction_mid_size_e2e_results.csv", index=False)
    complexity_df.to_csv(report_dir / "direction_mid_size_complexity.csv", index=False)
    failed_df.to_csv(report_dir / "failed_runs.csv", index=False)
    (report_dir / "model_configs.json").write_text(
        json.dumps(
            {
                "benchmark": "BITS/WEDA 25Hz event-centered 2s tilt12 50x12",
                "fall_branch": "fixed GradientBoosting FallNoTiming_Full_132",
                "seeds": seeds,
                "learning_rates": lrs,
                "tensorflow_available": TENSORFLOW_AVAILABLE,
                "tensorflow_import_error": TENSORFLOW_IMPORT_ERROR,
                "model_runs": model_configs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    make_plots(report_dir, figure_dir)
    write_summary(report_dir, results_df, best_df, vs_df, e2e_df, complexity_df, failed_df)
    print("[6/6] Done")
    if not vs_df.empty:
        print(vs_df.head(10).to_string(index=False))
    if not failed_df.empty:
        print("Failed/warning rows:")
        print(failed_df.to_string(index=False))


if __name__ == "__main__":
    main()
