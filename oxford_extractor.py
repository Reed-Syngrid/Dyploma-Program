"""
Oxford Battery Degradation — extract 1C discharge features from ``*.mat`` under ``OXFORD_RAW``.

Loads all ``.mat`` files recursively, reads ``Cell*`` top-level structs, ``cyc*`` cycles,
and ``C1dc`` blocks. ``q`` is treated as cumulative charge in **mAh** (peak-to-peak → Ah);
nominal cell capacity **0.74 Ah** for SoH and DCIR proxy current.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from scipy.io import loadmat

NOMINAL_CAPACITY_AH: float = 0.74
DCIR_PROXY_CURRENT_A: float = 0.74

RESULT_COLUMNS: tuple[str, ...] = (
    "dataset_source",
    "battery_id",
    "cycle_index",
    "avg_temperature_c",
    "resistance_ratio",
    "capacity_ah",
    "soh_percentage",
)

_CYCLE_NAME_RE = re.compile(r"^cyc(\d+)$", re.IGNORECASE)
_CELL_NAME_RE = re.compile(r"^Cell\d+$", re.IGNORECASE)


def _cycle_number_from_field_name(field_name: str) -> int | None:
    m = _CYCLE_NAME_RE.match(field_name.strip())
    if not m:
        return None
    return int(m.group(1))


def _unwrap_scalar_struct(x: Any) -> Any:
    """Unwrap (1,1) and 0-d structured/object arrays to inner scalar / void / mat_struct."""
    cur: Any = x
    for _ in range(64):
        if isinstance(cur, np.ndarray):
            if cur.shape == ():
                cur = cur[()]
                continue
            if cur.shape == (1, 1) and cur.dtype == object:
                cur = cur[0, 0]
                continue
            if cur.size == 1 and (cur.dtype == object or cur.dtype.names):
                cur = cur.flat[0]
                continue
        break
    return cur


def _as_float_vector(x: Any) -> np.ndarray | None:
    try:
        a = np.asarray(x, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if a.size == 0 or not np.any(np.isfinite(a)):
        return None
    return a


def _get_field(container: Any, name: str) -> Any:
    if isinstance(container, np.void) and container.dtype.names and name in container.dtype.names:
        return container[name]
    if hasattr(container, name):
        return getattr(container, name)
    raise KeyError(name)


def _iter_cyc_field_names(cell_obj: Any) -> Iterator[str]:
    u = _unwrap_scalar_struct(cell_obj)
    if isinstance(u, np.void) and u.dtype.names:
        for fn in u.dtype.names:
            if fn.startswith("cyc"):
                yield fn
        return
    fieldnames = getattr(u, "_fieldnames", None)
    if fieldnames:
        for fn in fieldnames:
            if fn.startswith("cyc"):
                yield fn


def _get_c1dc_for_cycle(cell_obj: Any, cyc_name: str) -> Any | None:
    try:
        cyc = _get_field(_unwrap_scalar_struct(cell_obj), cyc_name)
    except KeyError:
        return None
    cyc_u = _unwrap_scalar_struct(cyc)
    try:
        return _get_field(cyc_u, "C1dc")
    except KeyError:
        return None


def _extract_q_T_v(c1dc: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    c = _unwrap_scalar_struct(c1dc)
    try:
        q_raw = _get_field(c, "q")
        T_raw = _get_field(c, "T")
        v_raw = _get_field(c, "v")
    except KeyError:
        return None
    q = _as_float_vector(q_raw)
    T_arr = _as_float_vector(T_raw)
    v = _as_float_vector(v_raw)
    if q is None or T_arr is None or v is None:
        return None
    n = min(q.size, T_arr.size, v.size)
    if n < 6:
        return None
    return q[:n], T_arr[:n], v[:n]


class OxfordBatteryParser:
    """Parse all Oxford ``.mat`` files under a directory into a dense per-battery feature table."""

    def __init__(self, raw_dir: str | Path) -> None:
        self._raw_dir = Path(raw_dir)

    def parse_all(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        mat_paths = sorted(self._raw_dir.rglob("*.mat"))
        for mat_path in mat_paths:
            if not mat_path.is_file():
                continue
            try:
                mat = loadmat(str(mat_path), squeeze_me=True)
            except Exception:
                continue
            for top_key in mat:
                if top_key.startswith("__") or not _CELL_NAME_RE.match(top_key):
                    continue
                battery_id = top_key
                cell_obj = mat[top_key]
                for cyc_name in _iter_cyc_field_names(cell_obj):
                    raw_cycle = _cycle_number_from_field_name(cyc_name)
                    if raw_cycle is None:
                        continue
                    c1dc = _get_c1dc_for_cycle(cell_obj, cyc_name)
                    if c1dc is None:
                        continue
                    extracted = _extract_q_T_v(c1dc)
                    if extracted is None:
                        continue
                    q, T_arr, v = extracted
                    # q is cumulative mAh on C1dc; discharge capacity Ah = |max-min|/1000
                    capacity_ah = float(np.abs(np.ptp(q)) / 1000.0)
                    if not np.isfinite(capacity_ah) or capacity_ah <= 0:
                        continue
                    avg_temperature_c = float(np.mean(T_arr))
                    if not np.isfinite(avg_temperature_c):
                        continue
                    internal_resistance_ohm = float(np.abs(v[0] - v[5]) / DCIR_PROXY_CURRENT_A)
                    if not np.isfinite(internal_resistance_ohm):
                        continue
                    soh_percentage = (capacity_ah / NOMINAL_CAPACITY_AH) * 100.0
                    rows.append(
                        {
                            "battery_id": battery_id,
                            "raw_cycle": raw_cycle,
                            "avg_temperature_c": avg_temperature_c,
                            "internal_resistance_ohm": internal_resistance_ohm,
                            "capacity_ah": capacity_ah,
                            "soh_percentage": soh_percentage,
                        }
                    )

        if not rows:
            return pd.DataFrame(columns=list(RESULT_COLUMNS))

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["battery_id", "raw_cycle"], keep="first")

        dense_parts: list[pd.DataFrame] = []
        feature_cols = [
            "capacity_ah",
            "soh_percentage",
            "avg_temperature_c",
            "internal_resistance_ohm",
        ]
        for bid, g in df.groupby("battery_id", sort=True):
            g = g.sort_values("raw_cycle", kind="mergesort")
            max_cycle = int(g["raw_cycle"].max())
            full_idx = pd.RangeIndex(0, max_cycle + 1)
            wide = g.set_index("raw_cycle")[feature_cols].reindex(full_idx)
            wide = wide.interpolate(method="linear").bfill()
            out = wide.reset_index(names="cycle_index")
            out.insert(0, "battery_id", bid)
            out["dataset_source"] = "Oxford"
            dense_parts.append(out)

        out_df = pd.concat(dense_parts, ignore_index=True)
        out_df = out_df.sort_values(["battery_id", "cycle_index"], kind="mergesort").reset_index(
            drop=True
        )
        r0 = out_df.groupby("battery_id", sort=False)["internal_resistance_ohm"].transform("first")
        out_df["resistance_ratio"] = out_df["internal_resistance_ohm"] / r0.replace(0.0, np.nan)
        out_df = out_df.drop(columns=["internal_resistance_ohm"])
        out_df = out_df[list(RESULT_COLUMNS)]
        return out_df


def main() -> None:
    root = Path(__file__).resolve().parent
    raw = root / "OXFORD_RAW"
    out_dir = root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "OXFORD_processed.csv"

    parser = OxfordBatteryParser(raw)
    df = parser.parse_all()
    df.to_csv(out_csv, index=False)
    print(f"shape: {df.shape}")
    print(df.head().to_string())
    print(f"\nWrote {out_csv.resolve()}")


if __name__ == "__main__":
    main()
