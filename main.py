"""
ML entry point: MASTER dataset — train on NASA + Oxford, evaluate on Warwick (holdout).

Requires ``output/MASTER_processed.csv`` (from ``data_merger.py``).
Run from project root: python main.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from model_pipeline import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    fit_and_evaluate,
    format_metrics,
    shap_summary_plot,
)

TRAIN_SOURCES: tuple[str, ...] = ("NASA", "OXFORD")
TEST_SOURCE: str = "WARWICK"


def main() -> None:
    root = Path(__file__).resolve().parent
    master_path = root / "output" / "MASTER_processed.csv"

    df: pd.DataFrame = pd.read_csv(master_path)
    cols_needed = list(FEATURE_COLUMNS) + [TARGET_COLUMN, "dataset_source"]
    df = df[cols_needed].dropna()

    train = df[df["dataset_source"].isin(TRAIN_SOURCES)].copy()
    test = df[df["dataset_source"] == TEST_SOURCE].copy()

    X_train = train.loc[:, list(FEATURE_COLUMNS)]
    y_train = train[TARGET_COLUMN].to_numpy()
    X_test = test.loc[:, list(FEATURE_COLUMNS)]
    y_test = test[TARGET_COLUMN].to_numpy()

    print(f"MASTER_processed.csv (after dropna on features + target): {len(df)} rows")
    print(
        f"  Train (NASA + Oxford): {len(train)} | "
        f"Test (Warwick): {len(test)}"
    )
    print()
    print("Head (first rows of master):")
    print(pd.read_csv(master_path).head())
    print()

    feat_target = df[list(FEATURE_COLUMNS) + [TARGET_COLUMN]]
    corr = feat_target.corr(numeric_only=True)[TARGET_COLUMN].drop(TARGET_COLUMN)
    print("Pearson correlation of model features with target (full master, numeric cols):")
    print(corr.to_string())
    print()

    result = fit_and_evaluate(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        random_state=42,
    )
    print(
        format_metrics(
            result,
            subtitle="train on NASA + Oxford, test on Warwick (cross-dataset holdout)",
        )
    )
    print()

    shap_summary_plot(
        result,
        plot_title="SHAP summary (Warwick test; model trained on NASA + Oxford)",
    )


if __name__ == "__main__":
    main()
