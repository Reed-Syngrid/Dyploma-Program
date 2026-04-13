"""
One-time recursive copy of Warwick Capacity_Check CSV exports into a flat project folder.

Run from project root: python data_teleport.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

SOURCE_DIR = Path(r"C:/Users/parin/Documents/battery_soh_project/WARWICK_RAW/DIB_Data/.csvfiles/Capacity_Check")


def _flatten_unique_name(src: Path, source_root: Path, dest_dir: Path, used: set[str]) -> Path:
    """
    Build a flat destination path under ``dest_dir``.

    If only the basename is used and it is already taken, prepend parent path segments
    joined with underscores (e.g. ``subdir_Cell15_80SOH.csv``). If still colliding, append
    a numeric suffix before ``.csv``.
    """
    rel = src.relative_to(source_root)
    base = rel.name
    candidate = dest_dir / base
    if base not in used and not candidate.exists():
        return candidate

    stem = rel.stem
    suffix = rel.suffix
    parts = rel.parent.parts
    if parts:
        prefixed = "_".join(parts) + "_" + base
        candidate = dest_dir / prefixed
        if prefixed not in used and not candidate.exists():
            return candidate

    n = 2
    while True:
        alt = f"{stem}_{n}{suffix}"
        candidate = dest_dir / alt
        if alt not in used and not candidate.exists():
            return candidate
        n += 1


def main() -> None:
    root = Path(__file__).resolve().parent
    dest_dir = root / "data" / "WARWICK_RAW"

    if not SOURCE_DIR.is_dir():
        print(f"Source directory not found: {SOURCE_DIR}", file=sys.stderr)
        sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(SOURCE_DIR.rglob("*.csv"))
    used_names: set[str] = set()
    copied: list[Path] = []

    for src in csv_files:
        dst = _flatten_unique_name(src, SOURCE_DIR, dest_dir, used_names)
        shutil.copy2(src, dst)
        used_names.add(dst.name)
        copied.append(dst)

    print(f"Destination: {dest_dir.resolve()}")
    print(f"Total CSV files copied: {len(copied)}")
    print("Files:")
    for p in copied:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
