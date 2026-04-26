#!/usr/bin/env python3
"""
Verify LGRC2024 label indexing and (optionally) remap labels to 0-indexed.

Ultralytics YOLO expects 0-indexed class IDs in label files. The LGRC2024
labels appear to use 1-indexed (lowest seen ID is 1). This script:

  1. Scans all label files and reports the ID range.
  2. With --remap, copies the label tree to `LGRC2024 dataset/datasets_yolo/`
     with every class ID decremented by 1, AND copies images into the same
     tree so it's a self-contained training set.
  3. Writes the data.yaml `path:` line you should use for training.

Usage:
    python scripts/prepare_dataset.py                    # report only
    python scripts/prepare_dataset.py --remap            # do the remap
    python scripts/prepare_dataset.py --remap --symlink  # symlink images instead of copying
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from collections import Counter

DEFAULT_SRC = Path(__file__).resolve().parent.parent / "LGRC2024 dataset" / "datasets"
DEFAULT_DST = Path(__file__).resolve().parent.parent / "LGRC2024 dataset" / "datasets_yolo"


def scan(src: Path) -> dict:
    counts = Counter()
    n_files = 0
    for split in ("train", "val"):
        ldir = src / "labels" / split
        if not ldir.exists():
            continue
        for f in ldir.glob("*.txt"):
            n_files += 1
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    cls = int(line.split()[0])
                    counts[cls] += 1
    return {"n_files": n_files, "min": min(counts) if counts else None,
            "max": max(counts) if counts else None, "counts": counts}


def remap(src: Path, dst: Path, symlink: bool = False) -> None:
    """Copy src tree to dst with all class IDs decremented by 1."""
    for split in ("train", "val"):
        # labels (rewritten)
        src_lbl = src / "labels" / split
        dst_lbl = dst / "labels" / split
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for f in src_lbl.glob("*.txt"):
            with open(f) as fh:
                lines = fh.readlines()
            new_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls = int(parts[0]) - 1
                new_lines.append(" ".join([str(cls), *parts[1:]]))
            (dst_lbl / f.name).write_text("\n".join(new_lines) + "\n")

        # images (copied or symlinked)
        src_img = src / "images" / split
        dst_img = dst / "images" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        for f in src_img.iterdir():
            target = dst_img / f.name
            if target.exists():
                continue
            if symlink:
                target.symlink_to(f.resolve())
            else:
                shutil.copy2(f, target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--remap", action="store_true",
                    help="Copy/symlink images and rewrite labels with class IDs decremented by 1.")
    ap.add_argument("--symlink", action="store_true",
                    help="Symlink images instead of copying (saves disk).")
    args = ap.parse_args()

    print(f"Scanning {args.src} ...")
    info = scan(args.src)
    print(f"  Label files: {info['n_files']}")
    print(f"  Class ID range: {info['min']} .. {info['max']}")
    print(f"  Top 10 classes: {info['counts'].most_common(10)}")

    if info["min"] == 0:
        print("\nLabels appear to be 0-indexed already. No remap needed.")
        print(f"Use this in data.yaml:\n  path: {args.src.resolve()}")
        return
    elif info["min"] == 1:
        print("\nLabels appear to be 1-indexed. Remap recommended.")
    else:
        print(f"\nUnusual minimum class ID ({info['min']}). Inspect manually before training.")

    if args.remap:
        print(f"\nRemapping to {args.dst} (this may take a minute) ...")
        remap(args.src, args.dst, symlink=args.symlink)
        print("Done. Use this in data.yaml:")
        print(f"  path: {args.dst.resolve()}")
    else:
        print("\nRe-run with --remap to create a 0-indexed copy.")


if __name__ == "__main__":
    main()
