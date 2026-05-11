"""Random Forest training, regression metrics, and SHAP explanations."""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

FEATURE_COLUMNS: tuple[str, ...] = (
    "discharge_cycle",
    "avg_temperature_c",
)
TARGET_COLUMN: str = "soh_percentage"


@dataclass(frozen=True)
class PipelineResult:
    """Fitted model, held-out predictions, and test features for SHAP."""

    model: RandomForestRegressor
    rmse: float
    r2: float
    mae: float
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: np.ndarray
    y_pred: np.ndarray


def fit_and_evaluate(
    df: pd.DataFrame | None = None,
    *,
    X_train: pd.DataFrame | None = None,
    X_test: pd.DataFrame | None = None,
    y_train: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    groups: pd.Series | np.ndarray | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int | None = 12,
) -> PipelineResult:
    """
    Train a Random Forest on FEATURE_COLUMNS to predict TARGET_COLUMN.

    **Grouped split:** pass ``df``; uses ``GroupShuffleSplit`` on ``battery_id`` (no leakage across cells).

    **Holdout:** pass ``X_train``, ``X_test``, ``y_train``, ``y_test`` (e.g. NASA train / Oxford test).
    """
    if X_train is not None:
        if X_test is None or y_train is None or y_test is None:
            raise ValueError("Holdout mode requires X_train, X_test, y_train, and y_test.")
        if df is not None:
            raise ValueError("Pass either df or pre-split arrays, not both.")
        X_tr = X_train.loc[:, list(FEATURE_COLUMNS)]
        X_te = X_test.loc[:, list(FEATURE_COLUMNS)]
        y_tr = np.asarray(y_train)
        y_te = np.asarray(y_test)
    else:
        if df is None:
            raise TypeError("fit_and_evaluate requires df or X_train/X_test/y_train/y_test.")
        X = df.loc[:, list(FEATURE_COLUMNS)]
        y = df[TARGET_COLUMN].to_numpy()
        grp = groups if groups is not None else df.get("battery_id")
        if grp is None:
            raise ValueError("Grouped split requires battery_id in df or an explicit groups argument.")
        grp_arr = np.asarray(grp)
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=grp_arr))
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_tr = y[train_idx]
        y_te = y[test_idx]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    mse = mean_squared_error(y_te, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_te, y_pred))
    mae = float(mean_absolute_error(y_te, y_pred))

    return PipelineResult(
        model=model,
        rmse=rmse,
        r2=r2,
        mae=mae,
        X_train=X_tr,
        X_test=X_te,
        y_test=y_te,
        y_pred=y_pred,
    )


def format_metrics(result: PipelineResult, *, subtitle: str | None = None) -> str:
    """Human-readable metric block for printing."""
    lines = ["Random Forest - test metrics"]
    if subtitle:
        lines.append(f"  ({subtitle})")
    lines.extend(
        [
            f"  RMSE: {result.rmse:.4f}",
            f"  R^2:  {result.r2:.4f}",
            f"  MAE:  {result.mae:.4f}",
        ]
    )
    return "\n".join(lines)


def compute_shap_values(
    model: RandomForestRegressor,
    X_train: pd.DataFrame,
    X_explain: pd.DataFrame,
    *,
    max_background: int = 300,
) -> np.ndarray:
    """Compute SHAP values for ``X_explain`` (tree explainer, subsampled background)."""
    if list(X_train.columns) != list(FEATURE_COLUMNS):
        raise ValueError("X_train columns must match FEATURE_COLUMNS order")
    if list(X_explain.columns) != list(FEATURE_COLUMNS):
        raise ValueError("X_explain columns must match FEATURE_COLUMNS order")

    rng = np.random.default_rng(42)
    bg = X_train
    if len(bg) > max_background:
        idx = rng.choice(len(bg), size=max_background, replace=False)
        bg = bg.iloc[idx]

    explainer = shap.TreeExplainer(model, data=bg)
    return explainer.shap_values(X_explain)


def shap_summary_plot(
    result: PipelineResult,
    *,
    plot_title: str = "SHAP summary (impact on SoH prediction)",
    save_path: str | None = None,
) -> np.ndarray:
    """
    Draw SHAP summary plot for the test split; return SHAP values array.

    Set env ``SOH_MVP_HEADLESS=1`` to save PNG instead of opening a window
    (default file: ``shap_summary.png``, or ``SOH_MVP_SHAP_PATH``).
    """
    shap_values = compute_shap_values(
        result.model,
        result.X_train,
        result.X_test,
    )

    shap.summary_plot(
        shap_values,
        result.X_test,
        feature_names=list(FEATURE_COLUMNS),
        show=False,
    )
    fig = plt.gcf()
    fig.suptitle(plot_title)
    plt.tight_layout()

    headless = bool(os.environ.get("SOH_MVP_HEADLESS"))
    explicit_path = save_path or os.environ.get("SOH_MVP_SHAP_PATH")
    if headless:
        path = explicit_path or "shap_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"SHAP summary plot saved to: {path}")
        plt.close(fig)
        return shap_values
    if explicit_path:
        plt.savefig(explicit_path, dpi=150, bbox_inches="tight")
    plt.show()
    return shap_values
