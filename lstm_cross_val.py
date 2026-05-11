"""
LSTM transfer learning: pre-train on NASA + Warwick, fine-tune on 2 Oxford cells, test on held-out Oxford.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Inputs: temporal embedding + electro-physical state + degradation velocities (per-battery diffs).
FEATURE_COLUMNS: list[str] = [
    "discharge_cycle",
    "avg_temperature_c",
    "resistance_ratio",
    "nominal_capacity",
    "form_factor",
    "temp_variance",
    "voltage_variance",
    "resistance_velocity",
    "temp_var_velocity",
]

assert "discharge_cycle" in FEATURE_COLUMNS, "Temporal embedding: discharge_cycle must stay in FEATURE_COLUMNS."
assert "resistance_velocity" in FEATURE_COLUMNS and "temp_var_velocity" in FEATURE_COLUMNS

# Columns that must be present in MASTER before velocity features are derived.
_BASE_FEATURE_COLS: list[str] = [
    "avg_temperature_c",
    "resistance_ratio",
    "nominal_capacity",
    "form_factor",
    "temp_variance",
    "voltage_variance",
]
TARGET_COLUMN: str = "soh_percentage"
WINDOW_SIZE: int = 10
TARGET_HORIZON: int = 20
BATCH_SIZE: int = 64

PRETRAIN_EPOCHS: int = 150
PRETRAIN_LR: float = 0.001
FINETUNE_EPOCHS: int = 50
FINETUNE_LR: float = 0.0001

# Phase 22 — LSTM backbone (deep architecture)
LSTM_HIDDEN_SIZE: int = 64
LSTM_NUM_LAYERS: int = 2
LSTM_DROPOUT: float = 0.2


def _add_degradation_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """First differences of physical features within each battery (sorted by discharge_cycle)."""
    out = df.sort_values(["battery_id", "discharge_cycle"], kind="mergesort").copy()
    gb = out.groupby("battery_id", sort=False)
    out["resistance_velocity"] = gb["resistance_ratio"].diff().fillna(0.0)
    out["temp_var_velocity"] = gb["temp_variance"].diff().fillna(0.0)
    return out


def _oxford_adaptation_battery_ids(oxford_df: pd.DataFrame, k: int = 2) -> list[str]:
    """
    Few-shot selection: ``k`` Oxford cells whose **minimum** observed ``soh_percentage`` is lowest
    (deepest degradation trajectory). Tie-break on ``battery_id`` for reproducibility.
    """
    min_soh = (
        oxford_df.groupby("battery_id", sort=False)["soh_percentage"]
        .min()
        .rename("min_soh")
        .reset_index()
    )
    min_soh = min_soh.sort_values(["min_soh", "battery_id"], ascending=[True, True], kind="mergesort")
    return min_soh.head(k)["battery_id"].astype(str).tolist()


def create_sequences(
    df: pd.DataFrame, window_size: int = 10, horizon: int = TARGET_HORIZON
) -> tuple[np.ndarray, np.ndarray]:
    """
    Group by ``battery_id`` and build sliding windows of length ``window_size``.

    **Predictive horizon:** the target is **not** the SoH at the last timestep of the window, but the
    SoH at **row index** ``i + window_size + horizon`` (with default ``horizon=20``, that is
    ``i + window_size + 20``). Sequences that do not have enough future rows are dropped
    (``len(group) < window_size + horizon + 1``).

    **Time-gap protection:** along the row path ``i … i + window_size + horizon``, if any consecutive
    ``discharge_cycle`` step is strictly greater than 1, the window is discarded. Uses ``_dc_orig``
    when present; otherwise ``discharge_cycle``.
    """
    x_list: list[np.ndarray] = []
    y_list: list[float] = []

    dc_col = "_dc_orig" if "_dc_orig" in df.columns else "discharge_cycle"
    # Rows i .. i+window_size-1 are inputs; label at row i + window_size + horizon (20-step lookahead).
    span = window_size + horizon

    for _, group in df.groupby("battery_id", sort=True):
        g = group.sort_values("discharge_cycle", kind="mergesort")
        x = g[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y = g[TARGET_COLUMN].to_numpy(dtype=np.float32)
        dc = g[dc_col].to_numpy(dtype=np.float64)
        if len(g) < span + 1:
            continue
        for i in range(len(g) - span):
            label_idx = i + window_size + horizon
            # Contiguous discharge_cycle from first window row through label row (strict > 1 => discard).
            path_dc = dc[i : label_idx + 1]
            if path_dc.size != span + 1:
                continue
            if np.any(np.diff(path_dc) > 1.0):
                continue
            j = i + window_size
            x_list.append(x[i:j])
            y_list.append(float(y[label_idx]))

    if not x_list:
        return (
            np.empty((0, window_size, len(FEATURE_COLUMNS)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return np.stack(x_list).astype(np.float32), np.asarray(y_list, dtype=np.float32)


class LSTMRegressor(nn.Module):
    """
    Two-layer LSTM regressor with dropout (Phase 22): ``hidden_size=64``, ``num_layers=2``,
    ``dropout=0.2`` between stacked LSTM layers. A linear readout maps the last hidden state to SoH
    (scaled target space during training).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        pred = self.fc(last).squeeze(-1)
        return pred


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    train_mode = optimizer is not None
    if train_mode:
        model.train()
        # Frozen LSTM + training fc: keep LSTM in eval so inter-layer dropout stays off.
        lstm_has_grad = any(p.requires_grad for p in model.lstm.parameters())
        if not lstm_has_grad:
            model.lstm.eval()
    else:
        model.eval()
    total_loss = 0.0
    n = 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            if train_mode:
                optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
    return total_loss / max(n, 1)


def main() -> None:
    root = Path(__file__).resolve().parent
    master_path = root / "output" / "MASTER_processed.csv"
    df = pd.read_csv(master_path).copy()

    src = df["dataset_source"].astype(str).str.upper()

    oxford_df = df[src == "OXFORD"].copy()
    oxford_ids = sorted(oxford_df["battery_id"].dropna().unique().tolist())
    if len(oxford_ids) < 3:
        raise RuntimeError("Need at least 3 Oxford batteries for 2-shot adaptation + held-out test.")
    adapt_ids = _oxford_adaptation_battery_ids(oxford_df, k=2)
    min_soh_by_cell = oxford_df.groupby("battery_id", sort=False)["soh_percentage"].min()
    adapt_min_soh = {bid: float(min_soh_by_cell[bid]) for bid in adapt_ids}

    df_train = pd.concat(
        [
            df[src.isin(("NASA", "WARWICK"))].copy(),
            oxford_df[oxford_df["battery_id"].isin(adapt_ids)].copy(),
        ],
        ignore_index=True,
    )
    df_test = oxford_df[~oxford_df["battery_id"].isin(adapt_ids)].copy()

    required = _BASE_FEATURE_COLS + [TARGET_COLUMN, "battery_id", "discharge_cycle"]
    df_train = df_train.dropna(subset=required)
    df_test = df_test.dropna(subset=required)

    df_train = _add_degradation_velocity(df_train)
    df_test = _add_degradation_velocity(df_test)

    df_train["_dc_orig"] = df_train["discharge_cycle"].astype(np.float64)
    df_test["_dc_orig"] = df_test["discharge_cycle"].astype(np.float64)

    for col in FEATURE_COLUMNS:
        df_train[col] = df_train[col].astype(np.float64)
        df_test[col] = df_test[col].astype(np.float64)

    scaler = MinMaxScaler()
    df_train.loc[:, FEATURE_COLUMNS] = scaler.fit_transform(df_train[FEATURE_COLUMNS])
    df_test.loc[:, FEATURE_COLUMNS] = scaler.transform(df_test[FEATURE_COLUMNS])

    target_scaler = MinMaxScaler()
    df_train.loc[:, [TARGET_COLUMN]] = target_scaler.fit_transform(df_train[[TARGET_COLUMN]])
    df_test.loc[:, [TARGET_COLUMN]] = target_scaler.transform(df_test[[TARGET_COLUMN]])

    src_tr = df_train["dataset_source"].astype(str).str.upper()
    # Phase 1 pool: NASA + WARWICK only (no Oxford in pre-training).
    df_pretrain = df_train[src_tr.isin(("NASA", "WARWICK"))].copy()
    # Phase 2 pool: few-shot Oxford adaptation cells only.
    df_finetune = df_train[(src_tr == "OXFORD") & (df_train["battery_id"].isin(adapt_ids))].copy()

    X_pre_np, y_pre_np = create_sequences(df_pretrain, window_size=WINDOW_SIZE)
    X_ft_np, y_ft_np = create_sequences(df_finetune, window_size=WINDOW_SIZE)
    X_test_np, y_test_np = create_sequences(df_test, window_size=WINDOW_SIZE)

    if X_pre_np.shape[0] == 0:
        raise RuntimeError("No pre-train sequences (NASA + Warwick).")
    if X_ft_np.shape[0] == 0:
        raise RuntimeError("No fine-tune sequences (Oxford adaptation cells).")
    if X_test_np.shape[0] == 0:
        raise RuntimeError("No test sequences.")

    X_pre = torch.tensor(X_pre_np, dtype=torch.float32)
    y_pre = torch.tensor(y_pre_np, dtype=torch.float32)
    X_ft = torch.tensor(X_ft_np, dtype=torch.float32)
    y_ft = torch.tensor(y_ft_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)

    pre_loader = DataLoader(TensorDataset(X_pre, y_pre), batch_size=BATCH_SIZE, shuffle=True)
    ft_loader = DataLoader(TensorDataset(X_ft, y_ft), batch_size=BATCH_SIZE, shuffle=True)

    input_size = len(FEATURE_COLUMNS)
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
    )
    criterion = nn.MSELoss()

    # --- Phase 1: pre-train full model on NASA + Warwick ---
    optimizer_pre = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR)

    print(f"Phase 1 — pre-train (NASA + Warwick only): {len(y_pre_np)} sequences, {PRETRAIN_EPOCHS} epochs, lr={PRETRAIN_LR}")
    for epoch in range(PRETRAIN_EPOCHS):
        avg_loss = _run_epoch(model, pre_loader, criterion, optimizer_pre)
        if epoch == 0 or (epoch + 1) % 30 == 0:
            print(f"  Pre-train epoch {epoch + 1:03d}/{PRETRAIN_EPOCHS} - loss: {avg_loss:.4f}")

    # --- Phase 2: fine-tune readout on 2 Oxford adaptation cells; LSTM frozen ---
    for p in model.lstm.parameters():
        p.requires_grad = False
    model.lstm.eval()
    optimizer_ft = torch.optim.Adam(model.fc.parameters(), lr=FINETUNE_LR)

    print(
        f"\nPhase 2 — fine-tune (frozen LSTM, train fc only): {len(y_ft_np)} sequences, "
        f"{FINETUNE_EPOCHS} epochs, lr={FINETUNE_LR}"
    )
    for epoch in range(FINETUNE_EPOCHS):
        avg_loss = _run_epoch(model, ft_loader, criterion, optimizer_ft)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"  Fine-tune epoch {epoch + 1:02d}/{FINETUNE_EPOCHS} - loss: {avg_loss:.4f}")

    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    y_pred_chunks: list[np.ndarray] = []
    y_true_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in test_loader:
            pb = model(xb).cpu().numpy()
            y_pred_chunks.append(pb.reshape(-1, 1))
            y_true_chunks.append(yb.cpu().numpy().reshape(-1, 1))
    y_pred_scaled = np.concatenate(y_pred_chunks, axis=0)
    y_true_scaled = np.concatenate(y_true_chunks, axis=0)

    y_pred = target_scaler.inverse_transform(y_pred_scaled).ravel()
    y_true = target_scaler.inverse_transform(y_true_scaled).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    print("\nLSTM transfer learning: pre-train NASA+Warwick -> fine-tune 2 Oxford -> test held-out Oxford")
    print(f"  Oxford adaptation batteries (lowest min SoH): {adapt_ids}  min SoH: {adapt_min_soh}")
    print(
        f"  Pre-train sequences: {len(y_pre_np)} | Fine-tune sequences: {len(y_ft_np)} | "
        f"Test sequences: {len(y_test_np)}"
    )
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R^2:  {r2:.4f}")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.15, s=8, edgecolors="none")
    ax.set_xlabel("Actual SoH (Oxford held-out) [%]")
    ax.set_ylabel("Predicted SoH (LSTM transfer) [%]")
    ax.set_title("Pre-train NASA+Warwick, fine-tune 2 Oxford cells, test remaining Oxford")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = x (perfect)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    out_png = root / "lstm_transfer_plot.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to: {out_png.resolve()}")


if __name__ == "__main__":
    main()
