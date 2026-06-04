from __future__ import annotations

import argparse
import json
import math
import platform
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

import run_direction_mid_size_candidates as edge
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
BASE_PROTOCOLS = [
    "E1_BITS_TO_BITS",
    "E2_WEDA_TO_WEDA",
    "E3_BITS_WEDA_MIXED",
    "E4_BITS_TO_WEDA",
    "E5_WEDA_TO_BITS",
]
ALL_PROTOCOLS = BASE_PROTOCOLS + ["E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"]
FOCUS_PROTOCOLS = ["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"]
HISTORICAL_A5 = {
    "E3_BITS_WEDA_MIXED": 0.8967,
    "E6_E3_MIXED_TEST_BITS": 0.9350,
    "E7_E3_MIXED_TEST_WEDA": 0.8390,
    "avg": 0.8902,
    "params": 65_959,
}


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
    architecture_note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair multi-seed A5 Direction Only vs LDX1 comparison.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--learning-rates", "--lrs", dest="learning_rates", nargs="+", type=float, default=[1e-3, 5e-4])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
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
        def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1, dropout: float = 0.0):
            super().__init__()
            pad = ((kernel_size - 1) // 2) * dilation
            layers = [
                nn.Conv1d(in_ch, in_ch, kernel_size, padding=pad, dilation=dilation, groups=in_ch),
                nn.Conv1d(in_ch, out_ch, 1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            self.block = nn.Sequential(*layers)

        def forward(self, x):
            return self.block(x)


    class SensorEncoder(nn.Module):
        def __init__(self, input_channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(input_channels, 16, 5, padding=2),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                SepConv1dBlock(16, 24, 5),
                SepConv1dBlock(24, 32, 3),
            )

        def forward(self, x):
            return self.net(x)


    class DSTCNBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, dilation: int, dropout: float = 0.1):
            super().__init__()
            self.sep1 = SepConv1dBlock(in_ch, out_ch, 3, dilation=dilation, dropout=dropout)
            pad = dilation
            self.sep2 = nn.Sequential(
                nn.Conv1d(out_ch, out_ch, 3, padding=pad, dilation=dilation, groups=out_ch),
                nn.Conv1d(out_ch, out_ch, 1),
                nn.BatchNorm1d(out_ch),
            )
            self.proj = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)

        def forward(self, x):
            y = self.sep1(x)
            y = self.sep2(y)
            return F.relu(y + self.proj(x))


    class GatedAttentionPool(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            hidden = max(channels // 2, 1)
            self.hidden = nn.Linear(channels, hidden)
            self.score = nn.Linear(hidden, 1)

        def forward(self, x):
            # x: batch, channels, time
            xt = x.transpose(1, 2)
            h = torch.tanh(self.hidden(xt))
            alpha = torch.softmax(self.score(h), dim=1)
            return torch.sum(alpha * xt, dim=1)


    class A5DirectionOnly(nn.Module):
        def __init__(self):
            super().__init__()
            # tilt12 order: ax ay az gx gy gz acc_mag gyro_mag jerk roll pitch tilt_delta
            self.acc_idx = [0, 1, 2, 6, 8, 9, 10, 11]
            self.gyro_idx = [3, 4, 5, 7]
            self.acc_stream = SensorEncoder(8)
            self.gyro_stream = SensorEncoder(4)
            self.fusion = nn.Sequential(nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
            self.tcn = nn.Sequential(
                DSTCNBlock(64, 64, dilation=1, dropout=0.1),
                DSTCNBlock(64, 64, dilation=2, dropout=0.1),
                DSTCNBlock(64, 96, dilation=4, dropout=0.1),
            )
            self.direction_attention = GatedAttentionPool(96)
            self.direction_head = nn.Sequential(nn.Linear(96, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 3))

        def forward(self, x):
            xt = x.transpose(1, 2)
            acc = self.acc_stream(xt[:, self.acc_idx, :])
            gyro = self.gyro_stream(xt[:, self.gyro_idx, :])
            h = self.fusion(torch.cat([acc, gyro], dim=1))
            h = self.tcn(h)
            return self.direction_head(self.direction_attention(h))


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


def model_specs() -> list[dict[str, Any]]:
    return [
        {
            "model_name": "A5_Direction_Only",
            "builder": lambda: A5DirectionOnly(),
            "architecture_note": "PyTorch direction-only equivalent of DS-Fall-RD A5: dual-stream acc/gyro, fusion 1x1, DS-TCN dilation 1/2/4, direction gated attention, direction dense head; fall branch removed.",
        },
        {
            "model_name": "LDX1_D1_Wide_DSConv",
            "builder": lambda: LDX1WideDSConv(),
            "architecture_note": "PyTorch implementation of prompt LDX1: 3 depthwise-separable Conv1D blocks, global average pooling, Dense(48), Dropout(0.15), Dense(3).",
        },
    ]


def canonical_train_protocol(protocol_id: str) -> str:
    return "E3_BITS_WEDA_MIXED" if protocol_id in {"E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"} else protocol_id


def direction_mask(df: pd.DataFrame) -> np.ndarray:
    return mlbase.direction_train_mask(df)


def load_data(repo_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = hybrid.load_summary_feature_matrix(repo_root)
    X_temporal, _ = hybrid.load_temporal_data(repo_root, df)
    if X_temporal.shape[1:] != (50, 12):
        raise ValueError(f"Expected temporal shape (N,50,12), got {X_temporal.shape}")
    return df, X_temporal.astype(np.float32)


def standardize_temporal(X: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mean = X[train_mask].mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = (X[train_mask].std(axis=(0, 1), keepdims=True) + 1e-6).astype(np.float32)
    return ((X[train_mask] - mean) / std).astype(np.float32), ((X[val_mask] - mean) / std).astype(np.float32), {
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }


def count_params(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def estimate_complexity(model_name: str, params: int, a5_params: int | None = None) -> dict[str, Any]:
    fp32 = params * 4 / 1024
    int8 = params / 1024
    edge = "high" if params <= 12_000 else "medium-high" if params <= 30_000 else "medium"
    reduction = np.nan if not a5_params else 1.0 - params / float(a5_params)
    return {
        "params": params,
        "estimated_fp32_kb": fp32,
        "estimated_int8_kb": int8,
        "params_reduction_vs_A5_Direction_Only": reduction,
        "params_reduction_vs_historical_A5": 1.0 - params / float(HISTORICAL_A5["params"]),
        "edge_suitability": edge,
        "tflite_convert": "not_available_tensorflow_import_failed" if not TENSORFLOW_AVAILABLE else "not_attempted_pytorch_model",
        "int8_tflite_convert": "not_available_tensorflow_import_failed" if not TENSORFLOW_AVAILABLE else "not_attempted_pytorch_model",
        "macs_estimate": np.nan,
        "complexity_note": f"{model_name}; PyTorch implementation; TensorFlow available={TENSORFLOW_AVAILABLE}",
    }


def class_counts_for(df: pd.DataFrame, mask: np.ndarray, protocol_id: str, split_name: str, model_name: str | None = None) -> dict[str, Any]:
    dm = direction_mask(df)
    m = mask & dm
    counts = df.loc[m, "direction_id"].value_counts().reindex([0, 1, 2], fill_value=0)
    return {
        "model_name": model_name or "",
        "protocol_id": protocol_id,
        "split": split_name,
        "n_direction": int(m.sum()),
        "forward": int(counts.loc[0]),
        "backward": int(counts.loc[1]),
        "lateral": int(counts.loc[2]),
    }


def train_direction_model(
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
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1, 2]), y=y_train)
    model = spec["builder"]()
    class_weights = torch.tensor(weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train_np, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
    )
    X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    best_state = None
    best_macro = -1.0
    best_loss = math.inf
    best_epoch = 0
    no_improve = 0
    patience = 15
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(X_val_t)
            val_loss = float(criterion(logits, y_val_t).item())
            val_pred = torch.argmax(logits, dim=1).cpu().numpy()
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
    params = count_params(model)
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
        architecture_note=spec["architecture_note"],
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
) -> dict[str, Any]:
    sub = df.loc[mask].reset_index(drop=True)
    sup = direction_mask(sub)
    y = sub.loc[sup, "direction_id"].astype(int).to_numpy()
    pred = pred_all[sup]
    cm = confusion_matrix(y, pred, labels=[0, 1, 2]) if len(y) else np.zeros((3, 3), dtype=int)
    per = f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0) if len(y) else [np.nan, np.nan, np.nan]
    macro = f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0) if len(y) else np.nan
    hist = HISTORICAL_A5.get(experiment_id, np.nan)
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
        "historical_A5_reference_macro_f1": hist,
        "delta_vs_historical_A5": macro - hist if not pd.isna(hist) else np.nan,
        "params": run.params,
        "estimated_fp32_kb": run.params * 4 / 1024,
        "estimated_int8_kb": run.params / 1024,
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


def evaluate_e2e(
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
    plt.savefig(out_dir / f"{safe_name(row['run_id'])}_{row['experiment_id']}_direction.png", dpi=140)
    plt.close()


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))


def best_by_protocol(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    rows = []
    for keys, group in results.groupby(["experiment_id", "model_name"]):
        rows.append(group.sort_values(["direction_macro_f1", "direction_accuracy"], ascending=[False, False]).iloc[0])
    return pd.DataFrame(rows).sort_values(["experiment_id", "model_name"])


def build_focus_summary(results: pd.DataFrame) -> pd.DataFrame:
    focus = results[results["experiment_id"].isin(FOCUS_PROTOCOLS)].copy()
    if focus.empty:
        return pd.DataFrame()
    agg = focus.groupby(["model_name", "seed", "learning_rate", "params"], as_index=False).agg(
        avg_e3_e6_e7_f1=("direction_macro_f1", "mean"),
        std_e3_e6_e7_f1=("direction_macro_f1", "std"),
        e3_f1=("direction_macro_f1", lambda s: s[focus.loc[s.index, "experiment_id"].eq("E3_BITS_WEDA_MIXED")].mean()),
        e6_f1=("direction_macro_f1", lambda s: s[focus.loc[s.index, "experiment_id"].eq("E6_E3_MIXED_TEST_BITS")].mean()),
        e7_f1=("direction_macro_f1", lambda s: s[focus.loc[s.index, "experiment_id"].eq("E7_E3_MIXED_TEST_WEDA")].mean()),
        direction_accuracy_mean=("direction_accuracy", "mean"),
        delta_vs_historical_A5_mean=("delta_vs_historical_A5", "mean"),
    )
    agg["cv_e3_e6_e7"] = agg["std_e3_e6_e7_f1"] / agg["avg_e3_e6_e7_f1"].replace(0, np.nan)
    return agg.sort_values(["avg_e3_e6_e7_f1", "e3_f1"], ascending=[False, False])


def build_seed_summary(focus_summary: pd.DataFrame, complexity: pd.DataFrame) -> pd.DataFrame:
    if focus_summary.empty:
        return pd.DataFrame()
    agg = focus_summary.groupby(["model_name", "learning_rate", "params"], as_index=False).agg(
        mean_avg_e3_e6_e7_f1=("avg_e3_e6_e7_f1", "mean"),
        std_avg_e3_e6_e7_f1=("avg_e3_e6_e7_f1", "std"),
        mean_e3_f1=("e3_f1", "mean"),
        std_e3_f1=("e3_f1", "std"),
        mean_e6_f1=("e6_f1", "mean"),
        std_e6_f1=("e6_f1", "std"),
        mean_e7_f1=("e7_f1", "mean"),
        std_e7_f1=("e7_f1", "std"),
        mean_cv_e3_e6_e7=("cv_e3_e6_e7", "mean"),
        mean_delta_vs_historical_A5=("delta_vs_historical_A5_mean", "mean"),
    )
    a5_params = agg.loc[agg["model_name"].eq("A5_Direction_Only"), "params"].dropna()
    a5_param = int(a5_params.iloc[0]) if not a5_params.empty else HISTORICAL_A5["params"]
    agg["params_reduction_vs_A5_Direction_Only"] = 1.0 - agg["params"] / float(a5_param)
    agg["estimated_fp32_kb"] = agg["params"] * 4 / 1024
    agg["estimated_int8_kb"] = agg["params"] / 1024
    return agg.sort_values(["mean_avg_e3_e6_e7_f1", "mean_e3_f1"], ascending=[False, False])


def add_pairwise_deltas(seed_summary: pd.DataFrame) -> pd.DataFrame:
    if seed_summary.empty:
        return seed_summary
    rows = []
    for lr, group in seed_summary.groupby("learning_rate"):
        a5 = group[group["model_name"].eq("A5_Direction_Only")]
        ldx = group[group["model_name"].eq("LDX1_D1_Wide_DSConv")]
        for _, row in group.iterrows():
            out = row.to_dict()
            if row["model_name"] == "LDX1_D1_Wide_DSConv" and not a5.empty:
                out["delta_vs_A5_Direction_Only_mean"] = row["mean_avg_e3_e6_e7_f1"] - float(a5.iloc[0]["mean_avg_e3_e6_e7_f1"])
                out["std_delta_vs_A5_Direction_Only"] = row["std_avg_e3_e6_e7_f1"] - float(a5.iloc[0]["std_avg_e3_e6_e7_f1"])
            elif row["model_name"] == "A5_Direction_Only" and not ldx.empty:
                out["delta_vs_LDX1_mean"] = row["mean_avg_e3_e6_e7_f1"] - float(ldx.iloc[0]["mean_avg_e3_e6_e7_f1"])
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["mean_avg_e3_e6_e7_f1", "mean_e3_f1"], ascending=[False, False])


def build_e2e_seed_summary(e2e: pd.DataFrame) -> pd.DataFrame:
    if e2e.empty:
        return pd.DataFrame()
    focus = e2e[e2e["experiment_id"].isin(FOCUS_PROTOCOLS)].copy()
    parsed = focus["pipeline"].str.extract(r"^(?P<model_name>.+)_(?P<protocol>E3_BITS_WEDA_MIXED)_seed(?P<seed>\d+)_lr(?P<lr>[0-9.]+)$")
    focus = pd.concat([focus, parsed], axis=1)
    focus["seed"] = pd.to_numeric(focus["seed"], errors="coerce")
    focus["learning_rate"] = pd.to_numeric(focus["lr"], errors="coerce")
    per_seed = focus.groupby(["model_name", "seed", "learning_rate"], as_index=False).agg(
        avg_E2E_direction_macro_f1=("E2E_Direction_Macro_F1", "mean"),
        e3_E2E=("E2E_Direction_Macro_F1", lambda s: s[focus.loc[s.index, "experiment_id"].eq("E3_BITS_WEDA_MIXED")].mean()),
        e6_E2E=("E2E_Direction_Macro_F1", lambda s: s[focus.loc[s.index, "experiment_id"].eq("E6_E3_MIXED_TEST_BITS")].mean()),
        e7_E2E=("E2E_Direction_Macro_F1", lambda s: s[focus.loc[s.index, "experiment_id"].eq("E7_E3_MIXED_TEST_WEDA")].mean()),
        coverage=("coverage", "mean"),
        correct_rate=("correct_rate", "mean"),
    )
    return per_seed.groupby(["model_name", "learning_rate"], as_index=False).agg(
        mean_avg_E2E_direction_macro_f1=("avg_E2E_direction_macro_f1", "mean"),
        std_avg_E2E_direction_macro_f1=("avg_E2E_direction_macro_f1", "std"),
        mean_e3_E2E=("e3_E2E", "mean"),
        mean_e6_E2E=("e6_E2E", "mean"),
        mean_e7_E2E=("e7_E2E", "mean"),
        mean_coverage=("coverage", "mean"),
        mean_correct_rate=("correct_rate", "mean"),
    ).sort_values("mean_avg_E2E_direction_macro_f1", ascending=False)


def make_plots(report_dir: Path, figure_dir: Path) -> None:
    results_path = report_dir / "a5_vs_ldx1_direction_results.csv"
    seed_path = report_dir / "a5_vs_ldx1_seed_summary.csv"
    if results_path.exists():
        res = pd.read_csv(results_path)
        focus = res[res["experiment_id"].isin(FOCUS_PROTOCOLS)].copy()
        if not focus.empty:
            plt.figure(figsize=(9, 5))
            sns.boxplot(data=focus, x="experiment_id", y="direction_macro_f1", hue="model_name")
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.savefig(figure_dir / "e3_e6_e7_boxplot.png", dpi=150)
            plt.close()
    if seed_path.exists():
        seed = pd.read_csv(seed_path)
        if not seed.empty:
            plt.figure(figsize=(8, 5))
            sns.barplot(data=seed, x="model_name", y="mean_avg_e3_e6_e7_f1", hue="learning_rate")
            plt.ylabel("Mean E3/E6/E7 Direction Macro F1")
            plt.tight_layout()
            plt.savefig(figure_dir / "mean_std_comparison.png", dpi=150)
            plt.close()
            plt.figure(figsize=(7, 5))
            sns.scatterplot(data=seed, x="params", y="mean_avg_e3_e6_e7_f1", hue="model_name", style="learning_rate", s=100)
            plt.tight_layout()
            plt.savefig(figure_dir / "params_vs_mean_f1.png", dpi=150)
            plt.close()


def md_table(df: pd.DataFrame, max_rows: int = 25) -> str:
    if df.empty:
        return "_No rows._"
    x = df.head(max_rows).copy()
    for col in x.columns:
        if pd.api.types.is_float_dtype(x[col]):
            x[col] = x[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
        elif pd.api.types.is_integer_dtype(x[col]):
            x[col] = x[col].map(lambda v: "" if pd.isna(v) else str(int(v)))
        else:
            x[col] = x[col].astype(str).replace("nan", "")
    cols = list(x.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in x.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines)


def write_final_summary(
    report_dir: Path,
    results: pd.DataFrame,
    seed_summary: pd.DataFrame,
    focus_summary: pd.DataFrame,
    best: pd.DataFrame,
    e2e_summary: pd.DataFrame,
    complexity: pd.DataFrame,
    failed: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# A5 Direction Only vs LDX1 Multi-Seed Summary")
    lines.append("")
    lines.append("Benchmark: BITS/WEDA only, 25 Hz, 2-second event-centered windows, temporal `tilt12` input `50 x 12`. Both models are trained direction-only on supervised fall windows. Historical A5 saved predictions are not used as main results.")
    lines.append("")
    lines.append("## Run Status")
    lines.append("")
    lines.append(f"- Direction rows: `{len(results)}`.")
    lines.append(f"- Focus rows E3/E6/E7: `{len(focus_summary)}`.")
    lines.append(f"- Failed rows: `{len(failed)}`.")
    lines.append(f"- TensorFlow available: `{TENSORFLOW_AVAILABLE}`; import error: `{TENSORFLOW_IMPORT_ERROR or 'none'}`.")
    lines.append("- Framework note: TensorFlow import fails in this environment, so both models are implemented and trained in PyTorch for a fair same-framework comparison.")
    lines.append("- A5_Direction_Only note: fall branch is removed; this tests direction-only representation, not the historical multitask A5 artifact.")
    lines.append("")
    lines.append("## Seed Summary")
    lines.append("")
    show = [
        "model_name",
        "learning_rate",
        "params",
        "mean_avg_e3_e6_e7_f1",
        "std_avg_e3_e6_e7_f1",
        "mean_e3_f1",
        "mean_e6_f1",
        "mean_e7_f1",
        "mean_delta_vs_historical_A5",
        "params_reduction_vs_A5_Direction_Only",
        "estimated_int8_kb",
    ]
    lines.append(md_table(seed_summary[show], 20))
    lines.append("")
    lines.append("## Per-Seed E3/E6/E7 Summary")
    lines.append("")
    lines.append(md_table(focus_summary[["model_name", "seed", "learning_rate", "params", "avg_e3_e6_e7_f1", "std_e3_e6_e7_f1", "e3_f1", "e6_f1", "e7_f1", "delta_vs_historical_A5_mean"]], 30))
    lines.append("")
    lines.append("## Best By Protocol")
    lines.append("")
    lines.append(md_table(best[["experiment_id", "model_name", "seed", "learning_rate", "params", "direction_macro_f1", "direction_accuracy", "forward_f1", "backward_f1", "lateral_f1"]], 20))
    lines.append("")
    lines.append("## E2E Direction With Fixed GB Full132 Fall")
    lines.append("")
    lines.append(md_table(e2e_summary, 20))
    lines.append("")
    lines.append("## Complexity")
    lines.append("")
    lines.append(md_table(complexity.drop_duplicates(["model_name", "learning_rate"])[["model_name", "learning_rate", "params", "estimated_fp32_kb", "estimated_int8_kb", "params_reduction_vs_A5_Direction_Only", "params_reduction_vs_historical_A5", "edge_suitability", "tflite_convert", "int8_tflite_convert"]], 20))
    lines.append("")
    lines.append("## Answers")
    lines.append("")
    answer_lines = build_answers(seed_summary, e2e_summary)
    lines.extend(answer_lines)
    lines.append("")
    if not failed.empty:
        lines.append("## Failed Runs")
        lines.append("")
        lines.append(md_table(failed, 50))
        lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for name in [
        "a5_vs_ldx1_direction_results.csv",
        "a5_vs_ldx1_seed_summary.csv",
        "a5_vs_ldx1_best_by_protocol.csv",
        "a5_vs_ldx1_e3_e6_e7_summary.csv",
        "a5_vs_ldx1_e2e_results.csv",
        "a5_vs_ldx1_complexity.csv",
        "class_counts.csv",
        "failed_runs.csv",
        "model_configs.json",
        "environment_info.txt",
    ]:
        lines.append(f"- `{report_dir / name}`")
    (report_dir / "a5_vs_ldx1_final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_lr_row(seed_summary: pd.DataFrame, model_name: str) -> pd.Series | None:
    rows = seed_summary[seed_summary["model_name"].eq(model_name)]
    if rows.empty:
        return None
    return rows.sort_values(["mean_avg_e3_e6_e7_f1", "mean_e3_f1"], ascending=[False, False]).iloc[0]


def build_answers(seed_summary: pd.DataFrame, e2e_summary: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    a5 = best_lr_row(seed_summary, "A5_Direction_Only")
    ldx = best_lr_row(seed_summary, "LDX1_D1_Wide_DSConv")
    if a5 is None or ldx is None:
        return ["Missing A5 or LDX1 rows, cannot answer selection questions."]
    diff = float(ldx["mean_avg_e3_e6_e7_f1"] - a5["mean_avg_e3_e6_e7_f1"])
    std_diff = float(ldx["std_avg_e3_e6_e7_f1"] - a5["std_avg_e3_e6_e7_f1"])
    a5_rows = seed_summary[seed_summary["model_name"].eq("A5_Direction_Only")]
    ldx_rows = seed_summary[seed_summary["model_name"].eq("LDX1_D1_Wide_DSConv")]
    a5_lowest_std = a5_rows.sort_values("std_avg_e3_e6_e7_f1").iloc[0]
    ldx_lowest_std = ldx_rows.sort_values("std_avg_e3_e6_e7_f1").iloc[0]
    e2e_a5 = best_e2e_row(e2e_summary, "A5_Direction_Only")
    e2e_ldx = best_e2e_row(e2e_summary, "LDX1_D1_Wide_DSConv")
    e2e_diff = np.nan
    if e2e_a5 is not None and e2e_ldx is not None:
        e2e_diff = float(e2e_ldx["mean_avg_E2E_direction_macro_f1"] - e2e_a5["mean_avg_E2E_direction_macro_f1"])
    params_reduction = float(ldx["params_reduction_vs_A5_Direction_Only"])
    lines.append(f"1. A5_Direction_Only stability vs LDX1: comparing best-mean configs, A5 std `{a5['std_avg_e3_e6_e7_f1']:.4f}` and LDX1 std `{ldx['std_avg_e3_e6_e7_f1']:.4f}`. The lowest-std config overall is `{ldx_lowest_std['model_name']}` lr `{ldx_lowest_std['learning_rate']}` with std `{ldx_lowest_std['std_avg_e3_e6_e7_f1']:.4f}`.")
    winner_mean = "LDX1" if diff > 0 else "A5_Direction_Only"
    lines.append(f"2. Higher mean E3/E6/E7 Direction Macro F1: `{winner_mean}`. A5 mean `{a5['mean_avg_e3_e6_e7_f1']:.4f}`, LDX1 mean `{ldx['mean_avg_e3_e6_e7_f1']:.4f}`, LDX1-A5 delta `{diff:.4f}`.")
    if float(ldx_lowest_std["std_avg_e3_e6_e7_f1"]) < float(a5_lowest_std["std_avg_e3_e6_e7_f1"]):
        winner_std = "LDX1_D1_Wide_DSConv"
        std_note = f"LDX1 lowest std `{ldx_lowest_std['std_avg_e3_e6_e7_f1']:.4f}` vs A5 lowest std `{a5_lowest_std['std_avg_e3_e6_e7_f1']:.4f}`."
    else:
        winner_std = "A5_Direction_Only"
        std_note = f"A5 lowest std `{a5_lowest_std['std_avg_e3_e6_e7_f1']:.4f}` vs LDX1 lowest std `{ldx_lowest_std['std_avg_e3_e6_e7_f1']:.4f}`."
    lines.append(f"3. Lower std across seeds: `{winner_std}`. {std_note}")
    lines.append(f"4. Better E6 BITS mean: `{'LDX1' if ldx['mean_e6_f1'] > a5['mean_e6_f1'] else 'A5_Direction_Only'}`. A5 `{a5['mean_e6_f1']:.4f}`, LDX1 `{ldx['mean_e6_f1']:.4f}`.")
    lines.append(f"5. Better E7 WEDA mean: `{'LDX1' if ldx['mean_e7_f1'] > a5['mean_e7_f1'] else 'A5_Direction_Only'}`. A5 `{a5['mean_e7_f1']:.4f}`, LDX1 `{ldx['mean_e7_f1']:.4f}`.")
    close = abs(diff) <= 0.02
    if close:
        close_msg = "yes, within +/-0.02"
    elif diff > 0:
        close_msg = f"no; LDX1 is better than A5_Direction_Only by {diff:.4f}, so it is not merely close"
    else:
        close_msg = f"no; LDX1 is lower than A5_Direction_Only by {abs(diff):.4f}"
    lines.append(f"6. Does LDX1 remain close to A5_Direction_Only after fair multi-seed comparison? {close_msg}.")
    lines.append(f"7. LDX1 parameter reduction vs A5_Direction_Only: `{params_reduction * 100:.2f}%`; LDX1 params `{int(ldx['params'])}`, A5 params `{int(a5['params'])}`.")
    if e2e_a5 is not None and e2e_ldx is not None:
        lines.append(f"8. Better E2E direction with fixed GB Full132 fall: `{'LDX1' if e2e_diff > 0 else 'A5_Direction_Only'}`. A5 E2E `{e2e_a5['mean_avg_E2E_direction_macro_f1']:.4f}`, LDX1 E2E `{e2e_ldx['mean_avg_E2E_direction_macro_f1']:.4f}`, delta `{e2e_diff:.4f}`.")
    else:
        lines.append("8. E2E comparison unavailable because E2E rows are missing.")
    a5_under_hist = float(a5["mean_avg_e3_e6_e7_f1"]) < HISTORICAL_A5["avg"]
    lines.append(f"9. Should A5_Direction_Only replace historical A5 reference? `{'no' if a5_under_hist else 'possibly'}`. Historical multitask A5 avg is about `{HISTORICAL_A5['avg']:.4f}`; A5 direction-only mean is `{a5['mean_avg_e3_e6_e7_f1']:.4f}`.")
    ldx_beats_direction_only = diff > 0.02 and params_reduction >= 0.85 and (pd.isna(e2e_diff) or e2e_diff >= -0.02)
    ldx_compact_rule = (close or diff > 0) and abs(float(ldx_lowest_std["std_avg_e3_e6_e7_f1"]) - float(a5_lowest_std["std_avg_e3_e6_e7_f1"])) <= 0.02 and params_reduction >= 0.85 and (pd.isna(e2e_diff) or e2e_diff >= -0.02)
    if ldx_beats_direction_only:
        lines.append("10. LDX1 can replace A5_Direction_Only for compact direction-only deployment: it has higher mean F1, better E2E, and >85% parameter reduction. It should not be claimed to replace the historical multitask A5 reference unless it also beats that reference.")
    elif ldx_compact_rule:
        lines.append("10. LDX1 can replace A5_Direction_Only for compact deployment under the prompt rule: mean is within or above the A5 direction-only result, std is comparable, params reduction is >=85%, and E2E drop is not worse than 0.02.")
    else:
        lines.append("10. LDX1 should not automatically replace A5_Direction_Only under the strict compact rule; use it as compact edge expert unless seed/E2E stability is clearly acceptable.")
    if a5_under_hist:
        main = "historical multitask A5 direction reference"
    elif float(a5["mean_avg_e3_e6_e7_f1"]) >= float(ldx["mean_avg_e3_e6_e7_f1"]) + 0.02:
        main = "A5_Direction_Only"
    elif ldx_compact_rule or ldx_beats_direction_only:
        main = "LDX1_D1_Wide_DSConv for compact main, with A5 as high-capacity reference"
    else:
        main = "A5/historical A5 as main; LDX1 as compact edge ablation"
    lines.append(f"11. Final recommendation: main paper direction expert = `{main}`.")
    lines.append("12. Compact edge direction expert = `LDX1_D1_Wide_DSConv` if its measured mean/std trade-off is acceptable for deployment.")
    lines.append("13. Future work: stabilize LDX1 across seeds or restore TensorFlow/TFLite export to validate real INT8 deployment.")
    return lines


def best_e2e_row(e2e_summary: pd.DataFrame, model_name: str) -> pd.Series | None:
    if e2e_summary.empty:
        return None
    rows = e2e_summary[e2e_summary["model_name"].eq(model_name)]
    if rows.empty:
        return None
    return rows.sort_values("mean_avg_E2E_direction_macro_f1", ascending=False).iloc[0]


def main() -> None:
    args = parse_args()
    if args.full:
        args.run_all = True
    if not args.run_all:
        print("Use --run-all to execute the requested comparison.")
        return
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required because TensorFlow import is unavailable in this environment.")
    repo_root = Path(args.repo_root).resolve()
    report_dir = ensure_dir(repo_root / "outputs" / "reports" / "a5_direction_vs_ldx1_multiseed")
    figure_dir = ensure_dir(repo_root / "outputs" / "figures" / "a5_direction_vs_ldx1_multiseed")
    model_dir = ensure_dir(repo_root / "outputs" / "models" / "a5_direction_vs_ldx1_multiseed")
    cm_dir = ensure_dir(figure_dir / "confusion_matrices")
    seeds = args.seeds[:1] if args.quick else args.seeds
    learning_rates = args.learning_rates[:1] if args.quick else args.learning_rates
    protocols = ["E3_BITS_WEDA_MIXED"] if args.quick else BASE_PROTOCOLS
    eval_protocols = ["E3_BITS_WEDA_MIXED", "E6_E3_MIXED_TEST_BITS", "E7_E3_MIXED_TEST_WEDA"] if args.quick else ALL_PROTOCOLS
    print("[1/6] Loading BITS/WEDA temporal tilt12 data")
    df, X_temporal = load_data(repo_root)
    print(f"Temporal shape: {X_temporal.shape}")
    print("[2/6] Loading fixed GB FallNoTiming_Full_132 predictions")
    fall_predictions, fall_warnings = edge.load_fixed_fall_predictions(repo_root, df)
    results: list[dict[str, Any]] = []
    e2e_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    class_count_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = list(fall_warnings)
    model_configs: list[dict[str, Any]] = []
    trained_cache: dict[tuple[str, str, int, float], DirectionRun] = {}
    specs = model_specs()
    print(f"[3/6] Training models={len(specs)}, protocols={protocols}, seeds={seeds}, lrs={learning_rates}")
    for spec in specs:
        for seed in seeds:
            for lr in learning_rates:
                for protocol_id in protocols:
                    try:
                        train_mask, val_mask, test_mask = hybrid.get_protocol_splits(df, protocol_id)
                        class_count_rows.extend(
                            [
                                class_counts_for(df, train_mask, protocol_id, "train", spec["model_name"]),
                                class_counts_for(df, val_mask, protocol_id, "val", spec["model_name"]),
                                class_counts_for(df, test_mask, protocol_id, "test", spec["model_name"]),
                            ]
                        )
                        print(f"[train] {spec['model_name']} protocol={protocol_id} seed={seed} lr={lr}")
                        run = train_direction_model(df, X_temporal, protocol_id, spec, seed, lr, args.epochs, args.batch_size)
                        trained_cache[(run.model_name, protocol_id, seed, lr)] = run
                        state_path = model_dir / f"{run.model_name}_{protocol_id}_seed{seed}_lr{lr}.pt"
                        torch.save(
                            {
                                "model_name": run.model_name,
                                "state_dict": run.model.state_dict(),
                                "scaler": run.scaler,
                                "params": run.params,
                                "protocol_id": protocol_id,
                                "seed": seed,
                                "learning_rate": lr,
                                "architecture_note": run.architecture_note,
                            },
                            state_path,
                        )
                        complexity_rows.append(
                            {
                                "model_name": run.model_name,
                                "protocol_id": protocol_id,
                                "seed": seed,
                                "learning_rate": lr,
                                "best_val_macro_f1": run.best_val_macro_f1,
                                "best_val_loss": run.best_val_loss,
                                "architecture_note": run.architecture_note,
                                **run.complexity,
                            }
                        )
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
                                "best_val_macro_f1": run.best_val_macro_f1,
                                "params": run.params,
                                "train_direction_count": int((train_mask & dm).sum()),
                                "val_direction_count": int((val_mask & dm).sum()),
                                "test_direction_count": int((test_mask & dm).sum()),
                                "class_weights": compute_class_weight(
                                    class_weight="balanced",
                                    classes=np.array([0, 1, 2]),
                                    y=df.loc[train_mask & dm, "direction_id"].astype(int).to_numpy(),
                                ).tolist(),
                                "state_path": str(state_path),
                                "architecture_note": run.architecture_note,
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
    a5_param = None
    for run in trained_cache.values():
        if run.model_name == "A5_Direction_Only":
            a5_param = run.params
            break
    if a5_param:
        for row in complexity_rows:
            row["params_reduction_vs_A5_Direction_Only"] = 1.0 - row["params"] / float(a5_param)
    print("[4/6] Evaluating E1-E7")
    for (model_name, protocol_id, seed, lr), run in trained_cache.items():
        for eval_id in eval_protocols:
            if canonical_train_protocol(eval_id) != protocol_id:
                continue
            try:
                _, _, mask = hybrid.get_protocol_splits(df, eval_id)
                pred = predict_direction(run, X_temporal, mask)
                run_id = f"{model_name}_{protocol_id}_seed{seed}_lr{lr}"
                row = evaluate_direction(df, mask, pred, run, run_id, eval_id)
                results.append(row)
                save_direction_cm(row, cm_dir)
                fall_pred = fall_predictions.get(eval_id)
                if fall_pred is not None:
                    e2e_rows.append(evaluate_e2e(df, mask, fall_pred, pred, run_id, eval_id, protocol_id))
                print(f"[eval] {run_id} {eval_id} macro_f1={row['direction_macro_f1']:.4f}")
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
    print("[5/6] Saving reports")
    results_df = pd.DataFrame(results)
    best_df = best_by_protocol(results_df)
    focus_summary = build_focus_summary(results_df)
    seed_summary = add_pairwise_deltas(build_seed_summary(focus_summary, pd.DataFrame(complexity_rows)))
    e2e_df = pd.DataFrame(e2e_rows)
    e2e_summary = build_e2e_seed_summary(e2e_df)
    complexity_df = pd.DataFrame(complexity_rows)
    class_counts_df = pd.DataFrame(class_count_rows).drop_duplicates()
    failed_df = pd.DataFrame(failed)
    if failed_df.empty:
        failed_df = pd.DataFrame(columns=["stage", "model_name", "protocol_id", "seed", "learning_rate", "error"])
    results_df.to_csv(report_dir / "a5_vs_ldx1_direction_results.csv", index=False)
    seed_summary.to_csv(report_dir / "a5_vs_ldx1_seed_summary.csv", index=False)
    best_df.to_csv(report_dir / "a5_vs_ldx1_best_by_protocol.csv", index=False)
    focus_summary.to_csv(report_dir / "a5_vs_ldx1_e3_e6_e7_summary.csv", index=False)
    e2e_df.to_csv(report_dir / "a5_vs_ldx1_e2e_results.csv", index=False)
    e2e_summary.to_csv(report_dir / "a5_vs_ldx1_e2e_seed_summary.csv", index=False)
    complexity_df.to_csv(report_dir / "a5_vs_ldx1_complexity.csv", index=False)
    class_counts_df.to_csv(report_dir / "class_counts.csv", index=False)
    failed_df.to_csv(report_dir / "failed_runs.csv", index=False)
    (report_dir / "model_configs.json").write_text(
        json.dumps(
            {
                "benchmark": "BITS/WEDA 25Hz event-centered 2s tilt12 50x12",
                "seeds": seeds,
                "learning_rates": learning_rates,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "fall_branch_for_e2e": "fixed GradientBoosting FallNoTiming_Full_132",
                "tensorflow_available": TENSORFLOW_AVAILABLE,
                "tensorflow_import_error": TENSORFLOW_IMPORT_ERROR,
                "historical_A5_reference": HISTORICAL_A5,
                "model_runs": model_configs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (report_dir / "environment_info.txt").write_text(
        "\n".join(
            [
                f"python={platform.python_version()}",
                f"platform={platform.platform()}",
                f"torch_available={TORCH_AVAILABLE}",
                f"torch_version={getattr(torch, '__version__', 'not_available') if TORCH_AVAILABLE else 'not_available'}",
                f"tensorflow_available={TENSORFLOW_AVAILABLE}",
                f"tensorflow_import_error={TENSORFLOW_IMPORT_ERROR}",
                f"cwd={repo_root}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    make_plots(report_dir, figure_dir)
    write_final_summary(report_dir, results_df, seed_summary, focus_summary, best_df, e2e_summary, complexity_df, failed_df)
    print("[6/6] Done")
    print(seed_summary.to_string(index=False))
    if not failed_df.empty:
        print("Failed/warning rows:")
        print(failed_df.to_string(index=False))


if __name__ == "__main__":
    main()
