"""
Inspect NASA Li-ion battery `.mat` files (PCoE / NASA Ames aging dataset layout).

Expected layout (see dataset README): top-level variable named like ``B0005`` contains
field ``cycle`` - a 1xN struct array. Each cycle has ``type``, ``ambient_temperature``,
``time``, and ``data``. The ``data`` struct fields differ by cycle type (charge, discharge,
impedance).

This script loads ``NASA_RAW/B0005.mat``, walks the ``cycle`` array, and for the *first*
charge, discharge, and impedance cycle prints the nested ``numpy`` layout (shapes, dtypes,
structured field names), unwrapping MATLAB-style ``(1,1)`` object arrays as ``[0][0]``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

NASA_RAW_DIR = Path(__file__).resolve().parent / "NASA_RAW"
B0005_PATH = NASA_RAW_DIR / "B0005.mat"

TARGET_TYPES: tuple[str, ...] = ("charge", "discharge", "impedance")


def _cycle_type_string(raw: np.ndarray) -> str:
    """Normalize cycle ``type`` field to a plain string (e.g. ``'discharge'``)."""
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


def _describe_nested(
    label: str,
    obj: object,
    *,
    indent: int = 0,
    max_depth: int = 24,
) -> None:
    """Print shape, dtype, struct field names, and recurse through nesting."""
    pad = "  " * indent
    if indent > max_depth:
        print(f"{pad}{label}: … (max depth {max_depth})")
        return

    if isinstance(obj, np.void):
        names = obj.dtype.names
        print(f"{pad}{label}: numpy.void fields={names}")
        if names is None:
            return
        for fn in names:
            _describe_nested(str(fn), obj[fn], indent=indent + 1, max_depth=max_depth)
        return

    if isinstance(obj, np.ndarray):
        print(f"{pad}{label}: ndarray shape={obj.shape} dtype={obj.dtype!s}")
        if obj.dtype.names:
            print(f"{pad}  structured fields: {obj.dtype.names}")
            for fn in obj.dtype.names:
                _describe_nested(str(fn), obj[fn], indent=indent + 1, max_depth=max_depth)
            return
        if obj.dtype == object and obj.shape == (1, 1):
            inner = obj[0, 0]
            print(f"{pad}  unwrap object ndarray [0][0] -> {type(inner).__name__}")
            _describe_nested(f"{label}[0][0]", inner, indent=indent + 1, max_depth=max_depth)
            return
        if obj.dtype == object and obj.size > 0:
            inner0 = obj.flat[0]
            print(
                f"{pad}  object array: taking flat[0] for preview -> {type(inner0).__name__}"
            )
            _describe_nested(f"{label}.flat[0]", inner0, indent=indent + 1, max_depth=max_depth)
            return
        return

    print(f"{pad}{label}: {type(obj).__name__} {repr(obj)[:200]}")


def _main_data_root(mat: dict[str, object]) -> np.void:
    """Return the primary battery struct (ignore ``__*`` keys)."""
    keys = [k for k in mat if not k.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"Expected one non-meta key, got: {keys!r}")
    cell = mat[keys[0]]
    if not isinstance(cell, np.ndarray) or cell.shape != (1, 1):
        raise ValueError(f"Unexpected wrapper shape for {keys[0]!r}: {type(cell)} {getattr(cell, 'shape', None)}")
    root = cell[0, 0]
    if not isinstance(root, np.void):
        raise ValueError(f"Expected numpy.void at [0][0], got {type(root)}")
    return root


def main() -> None:
    if not B0005_PATH.is_file():
        print(f"Missing file: {B0005_PATH}", file=sys.stderr)
        sys.exit(1)

    mat = loadmat(str(B0005_PATH), struct_as_record=True, squeeze_me=False)

    print(f"Loaded: {B0005_PATH}")
    print("Top-level keys (excluding meta):", [k for k in mat if not k.startswith("__")])
    print()

    root = _main_data_root(mat)
    if root.dtype.names is None or "cycle" not in root.dtype.names:
        raise ValueError(f"Root struct has no 'cycle' field: {root.dtype}")

    cycles = root["cycle"]
    print(
        "Main struct: accessed root['cycle'] - "
        f"ndarray shape={cycles.shape} dtype={cycles.dtype!s}"
    )
    if cycles.dtype.names:
        print("  cycle element fields:", cycles.dtype.names)
    print()

    if cycles.ndim != 2 or cycles.shape[0] != 1:
        raise ValueError(f"Expected cycle array shape (1, N), got {cycles.shape}")

    n = int(cycles.shape[1])
    found: dict[str, int] = {}

    for i in range(n):
        ci = cycles[0, i]
        if not isinstance(ci, np.void):
            raise TypeError(f"Cycle {i} expected numpy.void, got {type(ci)}")
        ctype = _cycle_type_string(ci["type"])
        if ctype not in TARGET_TYPES or ctype in found:
            continue
        found[ctype] = i
        print("=" * 72)
        print(f"First '{ctype}' cycle - index {i} in cycle array [0, {i}]")
        print("-" * 72)
        _describe_nested("cycle", ci, indent=0)
        print()
        if set(found.keys()) == set(TARGET_TYPES):
            break

    missing = [t for t in TARGET_TYPES if t not in found]
    if missing:
        print("WARNING: did not find cycle type(s):", missing, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
