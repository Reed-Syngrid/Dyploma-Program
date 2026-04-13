"""
Extract discharge features from NASA Li-ion `.mat` files for ML (capacity, temperature, Re, SoH).

Nominal capacity is assumed **2.0 Ah** (NASA RW9 dataset convention): ``SoH = (capacity / 2.0) * 100``.
Impedance cycles update **Re** (estimated electrolyte resistance, Ohm); each discharge row uses the
latest **Re** seen so far for that battery, then missing values are filled per battery via ffill/bfill.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

NOMINAL_CAPACITY_AH: float = 2.0

RESULT_COLUMNS: tuple[str, ...] = (
    "battery_id",
    "discharge_cycle",
    "avg_temperature_c",
    "resistance_ratio",
    "capacity_ah",
    "soh_percentage",
)


def _cycle_type_string(raw: np.ndarray) -> str:
    """Normalize cycle ``type`` field to a plain string."""
    x = np.squeeze(raw)
    if x.shape == ():
        item = x.item()
    else:
        item = x.flat[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace").strip()
    if isinstance(item, str):
        return item.strip()
    return str(item).strip()


def _unwrap_to_scalar(obj: object) -> float | None:
    """Unwrap MATLAB-style (1,1) object/float arrays to a single float."""
    cur: object = obj
    for _ in range(32):
        if isinstance(cur, (float, int, np.floating, np.integer)):
            return float(cur)
        if not isinstance(cur, np.ndarray):
            return None
        if cur.size == 0:
            return None
        if cur.dtype == object and cur.shape == (1, 1):
            cur = cur[0, 0]
            continue
        if cur.dtype.names:
            return None
        if np.issubdtype(cur.dtype, np.number):
            val = np.squeeze(cur)
            if np.iscomplexobj(val):
                return float(np.real(val))
            return float(val)
        if cur.dtype == object:
            cur = cur.flat[0]
            continue
        val = np.squeeze(cur)
        if np.iscomplexobj(val):
            return float(np.real(val))
        return float(val)
    return None


def _data_struct_from_cycle(cycle: np.void) -> np.void | None:
    """Return the inner ``data`` struct (numpy.void) or None."""
    try:
        data_wrapped = cycle["data"]
    except (KeyError, ValueError, IndexError, TypeError):
        return None

    if not isinstance(data_wrapped, np.ndarray):
        return None

    ds: object = data_wrapped
    if isinstance(ds, np.ndarray):
        if ds.shape == (1, 1):
            ds = ds[0, 0]
        elif ds.size == 1:
            ds = ds.flat[0]

    if not isinstance(ds, np.void):
        return None
    return ds


def _extract_scalar_field(data_struct: np.void, field: str) -> float | None:
    """Read a numeric scalar from a structured ``data`` void (Capacity, Re, etc.)."""
    names = data_struct.dtype.names
    if names is None or field not in names:
        return None
    return _unwrap_to_scalar(data_struct[field])


def _extract_discharge_capacity_ah(data_struct: np.void) -> float | None:
    """Capacity (Ah) on discharge ``data`` struct."""
    return _extract_scalar_field(data_struct, "Capacity")


def _extract_re_ohm(data_struct: np.void) -> float | None:
    """Estimated electrolyte resistance ``Re`` (Ohm) on impedance ``data`` struct."""
    return _extract_scalar_field(data_struct, "Re")


def _extract_temperature_mean_c(data_struct: np.void) -> float | None:
    """Mean of ``Temperature_measured`` time series (deg C) on discharge ``data`` struct."""
    names = data_struct.dtype.names
    if names is None or "Temperature_measured" not in names:
        return None
    tm = data_struct["Temperature_measured"]
    arr = _unwrap_to_float_array(tm)
    if arr is None or arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
    return float(np.mean(arr))


def _unwrap_to_float_array(obj: object) -> np.ndarray | None:
    """Unwrap nested object arrays to a 1D float array."""
    cur: object = obj
    for _ in range(32):
        if isinstance(cur, np.ndarray):
            if cur.size == 0:
                return None
            if cur.dtype == object and cur.shape == (1, 1):
                cur = cur[0, 0]
                continue
            if cur.dtype == object:
                cur = cur.flat[0]
                continue
            if np.issubdtype(cur.dtype, np.number):
                a = np.asarray(cur, dtype=np.float64).ravel()
                if np.iscomplexobj(a):
                    a = np.real(a)
                return a.astype(np.float64, copy=False)
            return None
        return None
    return None


def _battery_key(mat: dict[str, object], mat_path: Path) -> str:
    """Resolve top-level struct name (prefer filename stem if it matches)."""
    keys = [k for k in mat if not k.startswith("__")]
    stem = mat_path.stem
    if stem in keys:
        return stem
    if len(keys) == 1:
        return keys[0]
    raise ValueError(
        f"{mat_path}: cannot pick battery key; stem={stem!r}, keys={keys!r}"
    )


class NASABatteryParser:
    """Parse NASA PCoE `.mat` battery files into tabular features per discharge cycle."""

    def __init__(self, raw_dir: str | Path) -> None:
        self._raw_dir = Path(raw_dir)

    def parse_all(self) -> pd.DataFrame:
        """
        Walk each ``*.mat`` in order; traverse ``cycle`` chronologically per battery.

        Impedance cycles update the running **Re** value (``internal_resistance_ohm``). Discharge cycles
        emit one row with capacity, mean temperature, latest Re, and SoH. Rows with missing critical
        discharge fields (capacity or temperature mean) are skipped.

        After building the table, ``internal_resistance_ohm`` is forward-filled then back-filled
        **within each** ``battery_id``. Baseline **R0** is the first row's resistance per battery;
        ``resistance_ratio = internal_resistance_ohm / R0``; the absolute ohm column is dropped.
        """
        rows: list[dict[str, float | str | int]] = []
        if not self._raw_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {self._raw_dir.resolve()}")

        for mat_path in sorted(self._raw_dir.glob("*.mat")):
            try:
                mat = loadmat(str(mat_path), struct_as_record=True, squeeze_me=False)
                battery_id = _battery_key(mat, mat_path)
                root = mat[battery_id][0, 0]
                cycle_block = root["cycle"]
                cycle_array = cycle_block[0]
            except (KeyError, ValueError, IndexError, TypeError) as exc:
                print(
                    f"Skip {mat_path.name}: cannot open root/cycle ({exc})",
                    file=sys.stderr,
                )
                continue

            current_resistance = np.nan
            discharge_counter = 0
            n = int(cycle_array.shape[0])

            for i in range(n):
                ci = cycle_array[i]
                if not isinstance(ci, np.void):
                    continue

                try:
                    ctype = _cycle_type_string(ci["type"])
                except (KeyError, ValueError, TypeError, IndexError):
                    continue

                if ctype == "impedance":
                    ds = _data_struct_from_cycle(ci)
                    if ds is None:
                        continue
                    re_val = _extract_re_ohm(ds)
                    if re_val is not None and np.isfinite(re_val):
                        current_resistance = float(re_val)
                    continue

                if ctype != "discharge":
                    continue

                ds = _data_struct_from_cycle(ci)
                if ds is None:
                    continue

                cap = _extract_discharge_capacity_ah(ds)
                avg_temp = _extract_temperature_mean_c(ds)
                if cap is None or avg_temp is None:
                    continue
                if not np.isfinite(cap) or not np.isfinite(avg_temp):
                    continue

                discharge_counter += 1
                soh = (cap / NOMINAL_CAPACITY_AH) * 100.0
                rows.append(
                    {
                        "battery_id": battery_id,
                        "discharge_cycle": discharge_counter,
                        "avg_temperature_c": avg_temp,
                        "internal_resistance_ohm": current_resistance,
                        "capacity_ah": cap,
                        "soh_percentage": soh,
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=list(RESULT_COLUMNS))

        df = df.sort_values(["battery_id", "discharge_cycle"], kind="mergesort").reset_index(
            drop=True
        )
        df["internal_resistance_ohm"] = df.groupby("battery_id", sort=False)[
            "internal_resistance_ohm"
        ].transform(lambda s: s.ffill().bfill())
        r0 = df.groupby("battery_id", sort=False)["internal_resistance_ohm"].transform("first")
        df["resistance_ratio"] = df["internal_resistance_ohm"] / r0.replace(0.0, np.nan)
        df = df.drop(columns=["internal_resistance_ohm"])
        return df[list(RESULT_COLUMNS)]


def main() -> None:
    root = Path(__file__).resolve().parent
    raw = root / "NASA_RAW"
    out_dir = root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "NASA_processed.csv"

    parser = NASABatteryParser(raw)
    df = parser.parse_all()
    print(df.head(5).to_string())
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
