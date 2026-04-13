"""
LSTM few-shot transfer validation: train on NASA + Warwick + 2 Oxford cells, test on remaining Oxford.
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

FEATURE_COLUMNS: list[str] = [
    "avg_temperature_c",
    "resistance_ratio",
    "nominal_capacity",
    "form_factor",
]
TARGET_COLUMN: str = "soh_percentage"
WINDOW_SIZE: int = 10
EPOCHS: int = 30
LEARNING_RATE: float = 0.01


def create_sequences(df: pd.DataFrame, window_size: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Group by battery_id and create rolling windows.
    Target is the last row's soh_percentage inside each window.
    """
    x_list: list[np.ndarray] = []
    y_list: list[float] = []

    for _, group in df.groupby("battery_id", sort=True):
        g = group.sort_values("discharge_cycle", kind="mergesort")
        x = g[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y = g[TARGET_COLUMN].to_numpy(dtype=np.float32)
        if len(g) < window_size:
            continue
        for i in range(len(g) - window_size + 1):
            j = i + window_size
            x_list.append(x[i:j])
            y_list.append(float(y[j - 1]))

    if not x_list:
        return (
            np.empty((0, window_size, len(FEATURE_COLUMNS)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return np.stack(x_list).astype(np.float32), np.asarray(y_list, dtype=np.float32)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 32, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        pred = self.fc(last).squeeze(-1)
        return pred


def main() -> None:
    root = Path(__file__).resolve().parent
    master_path = root / "output" / "MASTER_processed.csv"
    df = pd.read_csv(master_path).copy()

    src = df["dataset_source"].astype(str).str.upper()
    warwick_mask = src == "WARWICK"
    df.loc[warwick_mask, TARGET_COLUMN] = (df.loc[warwick_mask, "capacity_ah"] / 4.85) * 100.0

    oxford_df = df[src == "OXFORD"].copy()
    oxford_ids = sorted(oxford_df["battery_id"].dropna().unique().tolist())
    if len(oxford_ids) < 3:
        raise RuntimeError("Need at least 3 Oxford batteries for 2-shot adaptation + held-out test.")
    adapt_ids = oxford_ids[:2]

    df_train = pd.concat(
        [
            df[src.isin(("NASA", "WARWICK"))].copy(),
            oxford_df[oxford_df["battery_id"].isin(adapt_ids)].copy(),
        ],
        ignore_index=True,
    )
    df_test = oxford_df[~oxford_df["battery_id"].isin(adapt_ids)].copy()

    required = FEATURE_COLUMNS + [TARGET_COLUMN, "battery_id", "discharge_cycle"]
    df_train = df_train.dropna(subset=required)
    df_test = df_test.dropna(subset=required)

    scaler = MinMaxScaler()
    df_train.loc[:, FEATURE_COLUMNS] = scaler.fit_transform(df_train[FEATURE_COLUMNS])
    df_test.loc[:, FEATURE_COLUMNS] = scaler.transform(df_test[FEATURE_COLUMNS])

    target_scaler = MinMaxScaler()
    df_train.loc[:, [TARGET_COLUMN]] = target_scaler.fit_transform(df_train[[TARGET_COLUMN]])
    df_test.loc[:, [TARGET_COLUMN]] = target_scaler.transform(df_test[[TARGET_COLUMN]])

    X_train_np, y_train_np = create_sequences(df_train, window_size=WINDOW_SIZE)
    X_test_np, y_test_np = create_sequences(df_test, window_size=WINDOW_SIZE)

    if X_train_np.shape[0] == 0 or X_test_np.shape[0] == 0:
        raise RuntimeError("No sequences created. Check window_size and input data.")

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)

    model = LSTMRegressor(input_size=4, hidden_size=32, num_layers=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:02d}/{EPOCHS} - loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test).cpu().numpy().reshape(-1, 1)
    y_true_scaled = y_test.cpu().numpy().reshape(-1, 1)

    y_pred = target_scaler.inverse_transform(y_pred_scaled).ravel()
    y_true = target_scaler.inverse_transform(y_true_scaled).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    print("\nLSTM few-shot transfer: train on NASA + Warwick + 2 Oxford cells")
    print(f"  Oxford adaptation batteries: {adapt_ids}")
    print("  Test set: remaining Oxford batteries")
    print(f"  Train sequences: {len(y_train_np)} | Test sequences: {len(y_test_np)}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R^2:  {r2:.4f}")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.15, s=8, edgecolors="none")
    ax.set_xlabel("Actual SoH (Oxford) [%]")
    ax.set_ylabel("Predicted SoH (LSTM few-shot transfer) [%]")
    ax.set_title("LSTM NASA + Warwick + 2 Oxford cells -> held-out Oxford")
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
