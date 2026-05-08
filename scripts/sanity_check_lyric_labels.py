#!/usr/bin/env python3
"""
Visual sanity check for lyric ground-truth labels.

Overlays the YOLO labels in `lyric_dataset/labels/{train,val}/<page>.txt`
on the corresponding image and saves to `outputs/label_sanity/`. Open the
output PNG and confirm the boxes sit on the lyric calligraphy, not on the
pitch-annotation columns.

Usage:
    # One specific page
    python scripts/sanity_check_lyric_labels.py 137

    # Every page in lyric_dataset
    python scripts/sanity_check_lyric_labels.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
LDIR = ROOT / "lyric_dataset"
OUT = ROOT / "outputs" / "label_sanity"


def overlay(page_id: str) -> Path | None:
    img_p = lbl_p = None
    for split in ("train", "val"):
        cand_img = LDIR / "images" / split / f"{page_id}.jpg"
        cand_lbl = LDIR / "labels" / split / f"{page_id}.txt"
        if cand_img.exists() and cand_lbl.exists():
            img_p, lbl_p = cand_img, cand_lbl
            break
    if img_p is None:
        print(f"  [skip] {page_id}: not in lyric_dataset")
        return None

    img = Image.open(img_p).convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img)
    n = 0
    for line in open(lbl_p):
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = map(float, parts[:5])
        x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
        x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
        d.rectangle([x1, y1, x2, y2], outline="lime", width=4)
        n += 1

    OUT.mkdir(parents=True, exist_ok=True)
    out_p = OUT / f"{page_id}_labels.png"
    img.save(out_p)
    print(f"  {page_id}: {W}x{H}, {n} boxes  →  {out_p.relative_to(ROOT)}")
    return out_p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("page", nargs="?", help="LGRC2024 page id (e.g., 137)")
    ap.add_argument("--all", action="store_true",
                    help="Overlay every page in lyric_dataset")
    args = ap.parse_args()

    if args.all:
        pages = []
        for split in ("train", "val"):
            for f in (LDIR / "labels" / split).glob("*.txt"):
                pages.append(f.stem)
        for p in sorted(pages):
            overlay(p)
    elif args.page:
        overlay(args.page)
    else:
        ap.error("provide a page id or --all")


if __name__ == "__main__":
    main()
