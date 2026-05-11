"""
Warwick raw CSV logs (Maccor-style): discharge cycles, capacity, temperature, DCIR proxy, SoH.

SoH uses **NOMINAL_CAPACITY_META_AH** (4.85 Ah) as the single nominal reference. Resistance is normalized
per ``battery_id`` as ``resistance_ratio`` (first valid R as R0).
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

NOMINAL_CAPACITY_META_AH: float = 4.85
FORM_FACTOR: int = 1
DEFAULT_TEMP_C: float = 25.0
# Drop spurious short discharge pulses; if multiple remain, keep the largest-capacity segment per file.
MIN_SEGMENT_CAPACITY_AH: float = 0.15

RESULT_COLUMNS: tuple[str, ...] = (
    "battery_id",
    "discharge_cycle",
    "avg_temperature_c",
    "resistance_ratio",
    "capacity_ah",
    "nominal_capacity",
    "form_factor",
    "temp_variance",
    "voltage_variance",
    "soh_percentage",
)

_CELL_RE = re.compile(r"Cell(\d+)", re.IGNORECASE)
_CYCLE_RE = re.compile(r"(\d+)cycle", re.IGNORECASE)


def parse_filename(path: Path) -> tuple[str, int | None]:
    """``Cell8_...060cycle`` -> (Cell8, 60); cycle optional."""
    cm = _CELL_RE.search(path.name)
    if not cm:
        return "", None
    battery_id = f"Cell{cm.group(1)}"
    m = _CYCLE_RE.findall(path.name)
    if not m:
        return battery_id, None
    return battery_id, int(m[-1])


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    """One file per (battery_id, filename cycle or path name) — avoids duplicate flat/nested copies."""
    seen: set[tuple[str, str]] = set()
    out: list[Path] = []
    for p in sorted(paths, key=lambda x: str(x).lower()):
        bid, cyc = parse_filename(p)
        if not bid:
            continue
        key = (bid, str(cyc) if cyc is not None else p.name.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _find_header_line_index(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("Step,Status"):
            return i
        if "AhAccu" in line and "Step" in line:
            return i
    return None


def load_warwick_table(path: Path) -> pd.DataFrame | None:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"Skip read error {path.name}: {exc}", file=sys.stderr)
        return None
    lines = raw.splitlines()
    h = _find_header_line_index(lines)
    if h is None:
        print(f"Skip (no header row): {path.name}", file=sys.stderr)
        return None
    body = "\n".join(lines[h:])
    try:
        df = pd.read_csv(StringIO(body), header=0, low_memory=False)
    except (pd.errors.ParserError, ValueError) as exc:
        print(f"Skip parse error {path.name}: {exc}", file=sys.stderr)
        return None
    if df.empty:
        return None
    df.columns = [str(c).strip() for c in df.columns]
    if "Step" in df.columns and len(df) > 0:
        s0 = str(df.iloc[0]["Step"]).strip()
        if s0.startswith("[") or s0 == "[]":
            df = df.iloc[1:].reset_index(drop=True)
    return df


def _col_ci(df: pd.DataFrame, name: str) -> str | None:
    nl = name.lower()
    for c in df.columns:
        if str(c).strip().lower() == nl:
            return c
    return None


def _cycle_index_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).strip().lower().replace(" ", "") == "cycleindex":
            return c
    return None


def _contiguous_discharge_masks(is_dis: pd.Series) -> list[tuple[int, int]]:
    """Inclusive row index ranges where ``is_dis`` is True contiguously."""
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(len(is_dis)):
        if bool(is_dis.iloc[i]):
            if start is None:
                start = i
        else:
            if start is not None:
                ranges.append((start, i - 1))
                start = None
    if start is not None:
        ranges.append((start, len(is_dis) - 1))
    return ranges


def _segment_dataframe(df: pd.DataFrame, i0: int, i1: int) -> pd.DataFrame:
    return df.iloc[i0 : i1 + 1].copy()


def _capacity_from_ah(ah: np.ndarray) -> float | None:
    """Discharge Ah from cumulative ``AhAccu`` in the segment (span / peak-to-peak)."""
    ah = ah[np.isfinite(ah)]
    if ah.size == 0:
        return None
    cap = float(np.ptp(ah))
    if not np.isfinite(cap) or cap <= 0:
        return None
    return cap


def _internal_resistance_ohm(seg: pd.DataFrame, vcol: str, icol: str) -> float:
    if len(seg) < 2:
        return float("nan")
    v = pd.to_numeric(seg[vcol], errors="coerce").to_numpy(dtype=np.float64)
    cur = pd.to_numeric(seg[icol], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(v[0]) or not np.isfinite(v[1]) or not np.isfinite(cur[0]):
        return float("nan")
    i_load = abs(float(cur[0]))
    if i_load < 1e-9:
        return float("nan")
    return float((v[0] - v[1]) / i_load)


def _rows_from_discharge_segments(
    df: pd.DataFrame,
    *,
    filename_cycle: int | None,
) -> list[dict[str, float | int]]:
    vcol = _col_ci(df, "Voltage")
    icol = _col_ci(df, "Current")
    ahcol = _col_ci(df, "AhAccu")
    tcol = _col_ci(df, "LogTemp001")
    if not vcol or not icol or not ahcol or not tcol:
        return []

    cur = pd.to_numeric(df[icol], errors="coerce").fillna(0.0)
    is_dis = cur < 0

    cyc_idx = _cycle_index_column(df)
    blocks: list[tuple[int, int]] = []

    if cyc_idx is not None:
        cyc_num = pd.to_numeric(df[cyc_idx], errors="coerce")
        for cv in sorted(pd.unique(cyc_num.dropna())):
            pos = np.flatnonzero((cyc_num == cv).to_numpy())
            if pos.size == 0:
                continue
            i_lo, i_hi = int(pos[0]), int(pos[-1])
            sub = df.iloc[i_lo : i_hi + 1]
            sub_cur = pd.to_numeric(sub[icol], errors="coerce").fillna(0.0)
            sub_dis = sub_cur < 0
            for a, b in _contiguous_discharge_masks(sub_dis):
                blocks.append((i_lo + a, i_lo + b))
    else:
        for a, b in _contiguous_discharge_masks(is_dis):
            blocks.append((a, b))

    rows: list[dict[str, float | int]] = []
    for seg_i, (a, b) in enumerate(blocks):
        seg = _segment_dataframe(df, a, b)
        if len(seg) < 2:
            continue
        ah = pd.to_numeric(seg[ahcol], errors="coerce").to_numpy(dtype=np.float64)
        cap = _capacity_from_ah(ah)
        if cap is None:
            continue
        tt = pd.to_numeric(seg[tcol], errors="coerce")
        avg_t = float(tt.mean()) if tt.notna().any() else DEFAULT_TEMP_C
        if not np.isfinite(avg_t):
            avg_t = DEFAULT_TEMP_C

        vv_ser = pd.to_numeric(seg[vcol], errors="coerce")
        t_arr = tt.to_numpy(dtype=np.float64)
        v_arr = vv_ser.to_numpy(dtype=np.float64)
        t_arr = t_arr[np.isfinite(t_arr)]
        v_arr = v_arr[np.isfinite(v_arr)]
        if t_arr.size == 0 or v_arr.size == 0:
            continue
        temp_variance = float(np.var(t_arr))
        voltage_variance = float(np.var(v_arr))
        if not np.isfinite(temp_variance) or not np.isfinite(voltage_variance):
            continue

        ir = _internal_resistance_ohm(seg, vcol, icol)
        soh = (cap / NOMINAL_CAPACITY_META_AH) * 100.0

        if filename_cycle is not None:
            dc = filename_cycle if len(blocks) == 1 else filename_cycle * 1000 + seg_i
        else:
            dc = -1

        rows.append(
            {
                "discharge_cycle": int(dc),
                "avg_temperature_c": avg_t,
                "internal_resistance_ohm": ir,
                "capacity_ah": cap,
                "temp_variance": temp_variance,
                "voltage_variance": voltage_variance,
                "soh_percentage": soh,
            }
        )
    return rows


class WarwickBatteryParser:
    """Recursively walk ``WARWICK_RAW`` CSVs and build one feature row per discharge segment."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        root = Path(__file__).resolve().parent
        self._csv_dir = Path(data_dir) if data_dir is not None else root / "data" / "WARWICK_RAW"

    def parse_all(self) -> pd.DataFrame:
        if not self._csv_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {self._csv_dir.resolve()}")

        paths = _dedupe_paths([p for p in self._csv_dir.rglob("*.csv") if p.is_file()])
        rows_out: list[dict[str, float | int]] = []
        dc_fallback: defaultdict[str, int] = defaultdict(int)

        for path in paths:
            bid, fc = parse_filename(path)
            if not bid:
                print(f"Skip (filename pattern): {path.name}", file=sys.stderr)
                continue
            df = load_warwick_table(path)
            if df is None or df.empty:
                continue

            part = _rows_from_discharge_segments(df, filename_cycle=fc)
            part = [r for r in part if r["capacity_ah"] >= MIN_SEGMENT_CAPACITY_AH]
            if not part:
                continue
            if len(part) > 1:
                part = [max(part, key=lambda r: float(r["capacity_ah"]))]
            r0 = part[0]
            if r0["discharge_cycle"] == -1:
                k = dc_fallback[bid]
                r0["discharge_cycle"] = k
                dc_fallback[bid] = k + 1
            elif fc is not None:
                r0["discharge_cycle"] = int(fc)
            rows_out.append({"battery_id": bid, **r0})

        if not rows_out:
            return pd.DataFrame(columns=list(RESULT_COLUMNS))

        out = pd.DataFrame(rows_out)
        out["__cell_num"] = out["battery_id"].str.extract(r"(\d+)", expand=False).astype(int)
        out = out.sort_values(["__cell_num", "battery_id", "discharge_cycle"], kind="mergesort").drop(
            columns=["__cell_num"]
        )

        def _first_valid_r(s: pd.Series) -> float:
            for v in s:
                if pd.notna(v) and np.isfinite(v) and float(v) > 0:
                    return float(v)
            return float("nan")

        r0_map = out.groupby("battery_id", sort=False)["internal_resistance_ohm"].agg(_first_valid_r)
        out["resistance_ratio"] = out["internal_resistance_ohm"] / out["battery_id"].map(r0_map).replace(
            0.0, np.nan
        )
        out = out.drop(columns=["internal_resistance_ohm"])
        out["nominal_capacity"] = NOMINAL_CAPACITY_META_AH
        out["form_factor"] = FORM_FACTOR
        out = out[list(RESULT_COLUMNS)]
        return out.reset_index(drop=True)


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "WARWICK_processed.csv"

    parser = WarwickBatteryParser()
    df = parser.parse_all()
    df.to_csv(out_csv, index=False)
    print(df.head(10).to_string())
    print(f"\nWrote {out_csv.resolve()} ({len(df)} rows)")

    from data_merger import merge_datasets

    master = merge_datasets(out_dir)
    print(f"\nMASTER_processed.csv rows: {len(master)}")
    print(master["dataset_source"].value_counts().to_string())


if __name__ == "__main__":
    main()

