"""
Build ``MASTER_processed.csv`` from NASA, Oxford, and Warwick feature tables.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def merge_datasets(output_dir: str | Path) -> pd.DataFrame:
    """
    Load ``NASA_processed.csv``, ``OXFORD_processed.csv``, and ``WARWICK_processed.csv``,
    tag ``dataset_source``, concatenate, and save ``MASTER_processed.csv``.

    All three inputs use domain-normalized ``resistance_ratio`` (not absolute ohms).
    """
    out = Path(output_dir)
    nasa_path = out / "NASA_processed.csv"
    oxford_path = out / "OXFORD_processed.csv"
    warwick_path = out / "WARWICK_processed.csv"
    master_path = out / "MASTER_processed.csv"

    nasa = pd.read_csv(nasa_path)
    oxford = pd.read_csv(oxford_path)
    warwick = pd.read_csv(warwick_path)

    nasa = nasa.copy()
    oxford = oxford.copy()
    warwick = warwick.copy()
    if "cycle_index" in oxford.columns:
        oxford = oxford.rename(columns={"cycle_index": "discharge_cycle"})
    for label, frame in (("NASA", nasa), ("Oxford", oxford), ("Warwick", warwick)):
        if "internal_resistance_ohm" in frame.columns:
            raise ValueError(f"{label}_processed.csv must use resistance_ratio, not internal_resistance_ohm")
        if "resistance_ratio" not in frame.columns:
            raise ValueError(f"{label}_processed.csv is missing resistance_ratio")
        for req in ("nominal_capacity", "form_factor"):
            if req not in frame.columns:
                raise ValueError(f"{label}_processed.csv is missing {req}")
    nasa["dataset_source"] = "NASA"
    oxford["dataset_source"] = "OXFORD"
    warwick["dataset_source"] = "WARWICK"

    master = pd.concat([nasa, oxford, warwick], ignore_index=True)
    master.to_csv(master_path, index=False)
    return master


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "output"
    df = merge_datasets(out)
    print(f"MASTER_processed.csv rows: {len(df)}")
    print(df["dataset_source"].value_counts().to_string())
    print()
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
