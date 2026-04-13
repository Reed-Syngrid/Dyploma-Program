"""
Inspect the first available Oxford ``.mat`` under ``OXFORD_RAW`` (recursive) to map
time, voltage, current, temperature, and capacity fields.

Uses ``scipy.io.loadmat(..., squeeze_me=True)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.io.matlab import MatlabOpaque

# Prefer this stem if found anywhere under OXFORD_RAW; else use first ``*.mat`` (sorted).
PREFERRED_NAME = "TPG7.0-Cell4.mat"


def _pick_mat_file(oxford_root: Path) -> Path | None:
    """Return preferred file if present, otherwise first ``*.mat`` in sorted order."""
    all_mats = sorted(oxford_root.rglob("*.mat"))
    if not all_mats:
        return None
    for p in all_mats:
        if p.name == PREFERRED_NAME:
            return p
    return all_mats[0]


def _indent(n: int) -> str:
    return "  " * n


def _unwrap_struct_scalar(arr: np.ndarray) -> np.ndarray | np.void:
    cur: np.ndarray | np.void = arr
    while isinstance(cur, np.ndarray) and cur.dtype.names is not None and cur.shape == (1, 1):
        cur = cur[0, 0]
    return cur


def describe_fields(obj: object, name: str, level: int, max_depth: int) -> None:
    """Print field names, types, and shapes for structs and structured arrays."""
    pad = _indent(level)
    if level > max_depth:
        print(f"{pad}{name}: … (max depth {max_depth})")
        return

    if isinstance(obj, MatlabOpaque):
        print(f"{pad}{name}: {type(obj).__name__} (MATLAB opaque / table - columns not available in SciPy)")
        return

    if isinstance(obj, np.ndarray):
        print(f"{pad}{name}: ndarray shape={obj.shape} dtype={obj.dtype!s}")
        if not obj.dtype.names:
            return
        print(f"{pad}  dtype.fields: {obj.dtype.names}")
        if level >= max_depth:
            return
        # Use [()] not .item(): for large structured dtypes, .item() becomes a Python tuple.
        if obj.shape == ():
            inner: np.void | np.ndarray = obj[()]
        else:
            inner = _unwrap_struct_scalar(obj) if obj.size else obj.flat[0]
        if isinstance(inner, np.void) and inner.dtype.names:
            for fn in inner.dtype.names:
                sub = inner[fn]
                describe_fields(sub, f"{name}.{fn}", level + 1, max_depth)
        return

    tname = type(obj).__name__
    if "mat_struct" in tname or hasattr(obj, "_fieldnames"):
        fnames = getattr(obj, "_fieldnames", None) or []
        print(f"{pad}{name}: mat_struct with _fieldnames={fnames}")
        if level >= max_depth:
            return
        for fn in fnames:
            sub = getattr(obj, fn, None)
            st = type(sub).__name__
            shp = getattr(sub, "shape", None) if isinstance(sub, np.ndarray) else None
            print(f"{pad}  .{fn}: type={st}" + (f" shape={shp}" if shp is not None else ""))
            describe_fields(sub, f"{name}.{fn}", level + 1, max_depth)
        return

    print(f"{pad}{name}: {type(obj).__name__}")


def _main_data_key(keys: list[str]) -> str | None:
    """First plausible data variable (skip huge workspace blobs)."""
    for k in sorted(keys):
        if k == "__function_workspace__":
            continue
        return k
    return keys[0] if keys else None


def main() -> None:
    root = Path(__file__).resolve().parent
    oxford_raw = root / "OXFORD_RAW"
    if not oxford_raw.is_dir():
        print(f"Directory not found: {oxford_raw}", file=sys.stderr)
        sys.exit(1)

    fp = _pick_mat_file(oxford_raw)
    if fp is None:
        print(f"No .mat files under {oxford_raw}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded file (absolute path): {fp.resolve()}")
    print(f"(search: {oxford_raw.resolve()} with rglob('*.mat'); prefer name {PREFERRED_NAME})")
    print()

    mat = loadmat(str(fp), squeeze_me=True)

    top = [k for k in mat if not k.startswith("__")]
    print("Top-level keys (non-meta):", top)
    print()

    main_key = _main_data_key(top)
    if main_key is None:
        print("No data keys found.", file=sys.stderr)
        sys.exit(1)

    main_val = mat[main_key]
    print(f"Main data key (first data variable): {main_key!r}")
    print(f"  Python type: {type(main_val).__name__}")
    if isinstance(main_val, np.ndarray):
        print(f"  numpy shape: {main_val.shape}")
    print()
    print("Field tree (map time, voltage, current, temperature, capacity / q):")
    describe_fields(main_val, main_key, 0, max_depth=14)
    print()
    print(
        "If you see only MatlabOpaque, the file stores a MATLAB ``table``. "
        "Export a struct with t, v, i, T, q in MATLAB to read arrays in Python."
    )


if __name__ == "__main__":
    main()
