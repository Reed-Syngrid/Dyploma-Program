"""
Explore the first readable Warwick raw file under ``data/WARWICK_RAW`` (recursive):
``.csv``, ``.mat``, or ``.h5``. Maps time, voltage, current, temperature, and capacity-like fields.
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab import MatlabOpaque

# Prefer project layout used by ``warwick_extractor``; allow flat ``WARWICK_RAW`` at repo root.
RAW_DIR_NAMES: tuple[str, ...] = ("data/WARWICK_RAW", "WARWICK_RAW")
DATA_GLOBS: tuple[str, ...] = ("*.csv", "*.mat", "*.h5")

FIELD_HINTS: tuple[str, ...] = (
    "time",
    "volt",
    "current",
    "temp",
    "capac",
    "ah",
    "charge",
    "power",
    "coulomb",
)


def _warwick_raw_root(script_dir: Path) -> Path | None:
    for rel in RAW_DIR_NAMES:
        p = (script_dir / rel).resolve()
        if p.is_dir():
            return p
    return None


def _collect_candidate_files(root: Path) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in DATA_GLOBS:
        for p in root.rglob(pattern):
            if p.is_file():
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    out.append(rp)
    out.sort(key=lambda x: str(x).lower())
    return out


def _find_csv_header_line(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("Step,Status"):
            return i
        if "AhAccu" in line and "Step" in line:
            return i
    return None


def _load_warwick_csv(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    h = _find_csv_header_line(lines)
    if h is not None:
        body = "\n".join(lines[h:])
        df = pd.read_csv(StringIO(body), header=0, low_memory=False)
        if not df.empty and "Step" in df.columns:
            s0 = str(df.iloc[0]["Step"]).strip()
            if s0.startswith("[") or s0 == "[]":
                df = df.iloc[1:].reset_index(drop=True)
        return df
    return pd.read_csv(path, low_memory=False)


def _indent(n: int) -> str:
    return "  " * n


def _unwrap_struct_scalar(arr: np.ndarray) -> np.ndarray | np.void:
    cur: np.ndarray | np.void = arr
    while isinstance(cur, np.ndarray) and cur.dtype.names is not None and cur.shape == (1, 1):
        cur = cur[0, 0]
    return cur


def _field_matches_hint(name: str) -> bool:
    lower = name.lower()
    return any(h in lower for h in FIELD_HINTS)


def describe_mat_fields(obj: object, name: str, level: int, max_depth: int) -> None:
    """Print struct / ndarray tree; mark names that may map to t/V/I/T/Q."""
    pad = _indent(level)
    if level > max_depth:
        print(f"{pad}{name}: … (max depth {max_depth})")
        return

    hint = "  << possible t/V/I/T/capacity" if _field_matches_hint(name) else ""

    if isinstance(obj, MatlabOpaque):
        print(f"{pad}{name}: MatlabOpaque (opaque / table){hint}")
        return

    if isinstance(obj, np.ndarray):
        print(f"{pad}{name}: ndarray shape={obj.shape} dtype={obj.dtype!s}{hint}")
        if not obj.dtype.names:
            return
        print(f"{pad}  dtype.fields: {obj.dtype.names}")
        if level >= max_depth:
            return
        if obj.shape == ():
            inner: np.void | np.ndarray = obj[()]
        else:
            inner = _unwrap_struct_scalar(obj) if obj.size else obj.flat[0]
        if isinstance(inner, np.void) and inner.dtype.names:
            for fn in inner.dtype.names:
                sub = inner[fn]
                describe_mat_fields(sub, f"{name}.{fn}", level + 1, max_depth)
        return

    tname = type(obj).__name__
    if "mat_struct" in tname or hasattr(obj, "_fieldnames"):
        fnames = getattr(obj, "_fieldnames", None) or []
        print(f"{pad}{name}: mat_struct _fieldnames={fnames}{hint}")
        if level >= max_depth:
            return
        for fn in fnames:
            sub = getattr(obj, fn, None)
            st = type(sub).__name__
            shp = getattr(sub, "shape", None) if isinstance(sub, np.ndarray) else None
            extra = "  << possible t/V/I/T/capacity" if _field_matches_hint(fn) else ""
            print(f"{pad}  .{fn}: type={st}" + (f" shape={shp}" if shp is not None else "") + extra)
            describe_mat_fields(sub, f"{name}.{fn}", level + 1, max_depth)
        return

    print(f"{pad}{name}: {type(obj).__name__}{hint}")


def _print_mat_file(path: Path) -> None:
    mat = loadmat(str(path), squeeze_me=True)
    top = [k for k in mat if not k.startswith("__")]
    print("\nTop-level keys (non-meta):")
    for k in sorted(top):
        v = mat[k]
        if isinstance(v, np.ndarray):
            print(f"  {k!r}: ndarray shape={v.shape} dtype={v.dtype!s}")
        else:
            print(f"  {k!r}: {type(v).__name__}")
    print("\nDeep struct walk (hint markers on likely t/V/I/T/capacity names):")
    for k in sorted(top):
        describe_mat_fields(mat[k], k, 0, max_depth=14)
    print(f"\nHint substrings used: {FIELD_HINTS}")


def _print_h5_tree(path: Path) -> None:
    import h5py

    def print_name(name: str, obj: Any) -> None:
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name.split('/')[-1]}: Dataset shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"{indent}{name.split('/')[-1]}: Group")

    with h5py.File(path, "r") as f:
        print("\nHDF5 tree (visit order):")
        f.visititems(print_name)


def _dispatch_file(path: Path) -> None:
    suf = path.suffix.lower()
    if suf == ".csv":
        df = _load_warwick_csv(path)
        print("\n--- pandas.DataFrame ---")
        df.info()
        print("\n--- head() ---")
        print(df.head().to_string())
        return
    if suf == ".mat":
        _print_mat_file(path)
        return
    if suf in (".h5", ".hdf5"):
        _print_h5_tree(path)
        return
    raise ValueError(f"Unsupported extension: {suf}")


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    raw = _warwick_raw_root(root_dir)
    if raw is None:
        print(
            f"No WARWICK_RAW directory found (tried {RAW_DIR_NAMES} under {root_dir})",
            file=sys.stderr,
        )
        sys.exit(1)

    candidates = _collect_candidate_files(raw)
    if not candidates:
        print(f"No .csv / .mat / .h5 files under {raw}", file=sys.stderr)
        sys.exit(1)

    print(f"Search root: {raw}")
    print(f"Candidates (sorted): {len(candidates)} file(s)\n")

    for path in candidates:
        print("=" * 72)
        print(f"Trying: {path.resolve()}")
        try:
            print(f"Loaded file (absolute path): {path.resolve()}")
            _dispatch_file(path)
            print("\nDone.")
            return
        except Exception as exc:
            print(f"  FAILED ({type(exc).__name__}): {exc}", file=sys.stderr)
            continue

    print("All candidate files failed to load.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
