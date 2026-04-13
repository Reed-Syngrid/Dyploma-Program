"""
Triple-dataset cross-validation: train Random Forest on NASA + Warwick, evaluate on Oxford.

Warwick SoH is re-scaled to a 4.85 Ah nominal before training so targets align with physical capacity.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURE_COLUMNS: tuple[str, ...] = (
    "avg_temperature_c",
    "resistance_ratio",
)
TARGET_COLUMN: str = "soh_percentage"
# Warwick rows in MASTER use tag ``WARWICK`` (see data_merger.py).
WARWICK_SOURCE: str = "WARWICK"
WARWICK_NOMINAL_AH: float = 4.85


def main() -> None:
    root = Path(__file__).resolve().parent
    master_path = root / "output" / "MASTER_processed.csv"

    df = pd.read_csv(master_path).copy()

    w_mask = df["dataset_source"] == WARWICK_SOURCE
    df.loc[w_mask, "soh_percentage"] = (df.loc[w_mask, "capacity_ah"] / WARWICK_NOMINAL_AH) * 100.0

    df_train = df[df["dataset_source"].isin(("NASA", WARWICK_SOURCE))].copy()
    df_test = df[df["dataset_source"] == "OXFORD"].copy()

    cols = list(FEATURE_COLUMNS) + [TARGET_COLUMN]
    df_train = df_train.dropna(subset=cols)
    df_test = df_test.dropna(subset=cols)

    X_train = df_train.loc[:, list(FEATURE_COLUMNS)]
    y_train = df_train[TARGET_COLUMN].to_numpy()
    X_test = df_test.loc[:, list(FEATURE_COLUMNS)]
    y_test = df_test[TARGET_COLUMN].to_numpy()

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print("Triple cross-dataset validation: train on NASA + Warwick, test on Oxford")
    print(f"  Train rows: {len(df_train)} (NASA + Warwick) | Test rows: {len(df_test)} (Oxford)")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R^2:  {r2:.4f}")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.15, s=8, edgecolors="none")
    ax.set_xlabel("Actual SoH (Oxford) [%]")
    ax.set_ylabel("Predicted SoH (NASA + Warwick train) [%]")
    ax.set_title("NASA + Warwick → Oxford: actual vs predicted SoH (temp + R_ratio)")

    lo = float(min(np.min(y_test), np.min(y_pred)))
    hi = float(max(np.max(y_test), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = x (perfect)")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    out_png = root / "triple_cross_val_plot.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to: {out_png.resolve()}")


if __name__ == "__main__":
    main()
