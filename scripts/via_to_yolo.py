#!/usr/bin/env python3
"""
Convert a VGG VIA (Image Annotator) JSON export into YOLO single-class
labels for the lyric position detector.

VIA JSON layout (the format it exports via Annotation → Export Annotations
(as json)):

    {
      "img_id_or_filename123456": {
        "filename": "15.jpg",
        "size": ...,
        "regions": [
          {
            "shape_attributes": {
              "name": "rect", "x": 855, "y": 670, "width": 75, "height": 70
            },
            "region_attributes": {... optional, may include "char": "嫩"}
          },
          ...
        ],
        ...
      },
      ...
    }

Image dimensions are NOT in the export — we have to open the actual image
file to get them, which is why we need the lyric_dataset/images/<split>/
files to already be in place.

Usage:
    # Convert one VIA JSON, auto-routing each filename to the split where
    # the corresponding image already lives (lyric_dataset/images/<split>/)
    python scripts/via_to_yolo.py via_all.json

    # Force everything into one split
    python scripts/via_to_yolo.py via_train.json --split train

    # Inspect without writing
    python scripts/via_to_yolo.py via_all.json --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
LDIR = ROOT / "lyric_dataset"


def to_yolo(x, y, w, h, W, H) -> str:
    cx = (x + w / 2) / W
    cy = (y + h / 2) / H
    nw = w / W
    nh = h / H
    cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
    nw, nh = max(0.0, min(1.0, nw)), max(0.0, min(1.0, nh))
    return f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def find_split_for(fname: str) -> str | None:
    """Look up which split a given filename was staged into. Returns 'train',
    'val', or None if the image isn't present in either."""
    for split in ("train", "val"):
        if (LDIR / "images" / split / fname).exists():
            return split
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("via_json", help="Path to VIA JSON export")
    ap.add_argument("--split", choices=["train", "val"], default=None,
                    help="Force everything into this split (skip auto-routing)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    via_path = Path(args.via_json)
    if not via_path.exists():
        ap.error(f"{via_path} not found")

    if not args.dry_run:
        for split in ("train", "val"):
            (LDIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    with open(via_path, encoding="utf-8") as f:
        data = json.load(f)

    n_pages = {"train": 0, "val": 0}
    n_boxes = {"train": 0, "val": 0}
    skipped = []
    for entry_id, entry in data.items():
        fname = entry.get("filename")
        if not fname:
            skipped.append((entry_id, "no filename"))
            continue
        page_id = Path(fname).stem
        regions = entry.get("regions", [])
        if not regions:
            skipped.append((page_id, "no regions"))
            continue

        # Auto-route or honor --split
        split = args.split or find_split_for(fname)
        if split is None:
            skipped.append((page_id, "image not staged in train/ or val/"))
            continue
        img_path = LDIR / "images" / split / fname

        with Image.open(img_path) as img:
            W, H = img.size

        lines = []
        for r in regions:
            sa = r.get("shape_attributes", {})
            if sa.get("name") != "rect":
                continue
            try:
                x = float(sa["x"])
                y = float(sa["y"])
                w = float(sa["width"])
                h = float(sa["height"])
            except (KeyError, TypeError, ValueError):
                continue
            lines.append(to_yolo(x, y, w, h, W, H))

        if not lines:
            skipped.append((page_id, "no rect regions"))
            continue

        if args.dry_run:
            print(f"  [DRY] {page_id} → {split}: {len(lines)} boxes")
        else:
            (LDIR / "labels" / split / f"{page_id}.txt").write_text(
                "\n".join(lines) + "\n"
            )
            print(f"  {page_id} → {split}: {len(lines)} boxes")
        n_pages[split] += 1
        n_boxes[split] += len(lines)

    print(f"\nDone. "
          f"train: {n_pages['train']} pages, {n_boxes['train']} boxes  |  "
          f"val: {n_pages['val']} pages, {n_boxes['val']} boxes")
    if skipped:
        print(f"\nSkipped:")
        for pid, why in skipped:
            print(f"  {pid}: {why}")


if __name__ == "__main__":
    main()
