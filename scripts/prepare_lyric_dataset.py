#!/usr/bin/env python3
"""
Convert demo_lyrics/*.json (hand-edited or VLM-bootstrapped) into a single-class
YOLO dataset for training a lyric *detector* (not recognizer).

Output structure (default):
    lyric_dataset/
    ├── images/
    │   ├── train/   # symlinks to LGRC2024 images for pages with annotations
    │   └── val/
    └── labels/
        ├── train/
        │   └── 137.txt   # YOLO format: 0 cx_norm cy_norm w_norm h_norm
        └── val/

Each .txt contains lines of form:  `0 cx cy w h`  (class 0 = lyric, normalized).

The character identities in the JSONs are IGNORED — this dataset is for
*localization only*. That's intentional: lyric character identity is an
open vocabulary and a 28-class-style classifier wouldn't generalize.
A single-class detector learns "what does a lyric character box look like
in this style of page" and generalizes to unseen characters.

Usage:
    # All demo_lyrics JSONs, default split (matches LGRC2024 train/val).
    python scripts/prepare_lyric_dataset.py

    # Custom output dir + skip placeholders (chars marked '？' or '〇').
    python scripts/prepare_lyric_dataset.py --out lyric_data --skip-placeholders

    # Use a holdout for val (e.g. last 20% of pages).
    python scripts/prepare_lyric_dataset.py --val-frac 0.2
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO_LYRICS = ROOT / "demo_lyrics"
LGRC = ROOT / "LGRC2024 dataset" / "datasets" / "images"
DEFAULT_OUT = ROOT / "lyric_dataset"

PLACEHOLDER_CHARS = {"？", "?", "〇"}


def find_image(page_id: str) -> tuple[Path | None, str | None]:
    """Return (image_path, split) or (None, None) if not found."""
    for split in ("train", "val"):
        p = LGRC / split / f"{page_id}.jpg"
        if p.exists():
            return p, split
    return None, None


def to_yolo_line(bbox, page_w: int, page_h: int) -> str:
    """[x1, y1, x2, y2] → '0 cx cy w h' normalized."""
    x1, y1, x2, y2 = bbox
    cx = 0.5 * (x1 + x2) / page_w
    cy = 0.5 * (y1 + y2) / page_h
    w = (x2 - x1) / page_w
    h = (y2 - y1) / page_h
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Output dataset root (default: ./lyric_dataset)")
    ap.add_argument("--val-frac", type=float, default=None,
                    help="Override val fraction (default: use LGRC2024 split)")
    ap.add_argument("--skip-placeholders", action="store_true",
                    help="Skip lyric entries whose char is '？' or '〇' "
                         "(unfilled bootstraps; would inject incorrect boxes)")
    ap.add_argument("--symlink", action="store_true", default=True,
                    help="Symlink images instead of copying (default)")
    ap.add_argument("--copy", dest="symlink", action="store_false",
                    help="Copy images instead of symlinking")
    ap.add_argument("--seed", type=int, default=137)
    args = ap.parse_args()

    json_files = sorted(DEMO_LYRICS.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files in {DEMO_LYRICS}")

    # Override split: use LGRC2024 train/val, OR custom random split
    use_custom_split = args.val_frac is not None
    rng = random.Random(args.seed)

    rows = []
    for jf in json_files:
        page_id = jf.stem
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        img_path, lgrc_split = find_image(page_id)
        if img_path is None:
            print(f"  [skip] no image for page '{page_id}'")
            continue

        boxes = []
        skipped = 0
        for ly in data.get("lyrics", []):
            if args.skip_placeholders and ly.get("char") in PLACEHOLDER_CHARS:
                skipped += 1
                continue
            boxes.append(ly["bbox"])
        if not boxes:
            print(f"  [skip] {page_id}: no usable boxes (skipped {skipped} placeholders)")
            continue

        if use_custom_split:
            split = "val" if rng.random() < args.val_frac else "train"
        else:
            split = lgrc_split

        rows.append({
            "page_id": page_id,
            "img_path": img_path,
            "page_w": data.get("page_w"),
            "page_h": data.get("page_h"),
            "boxes": boxes,
            "split": split,
            "skipped": skipped,
        })

    if not rows:
        raise SystemExit("No annotated pages found.")

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val = sum(1 for r in rows if r["split"] == "val")
    n_boxes = sum(len(r["boxes"]) for r in rows)
    print(f"Found {len(rows)} pages ({n_train} train, {n_val} val), {n_boxes} lyric boxes")

    # Build dataset tree
    out = args.out
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for r in rows:
        # Image (symlink or copy)
        dst_img = out / "images" / r["split"] / r["img_path"].name
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        if args.symlink:
            os.symlink(r["img_path"].resolve(), dst_img)
        else:
            import shutil
            shutil.copy2(r["img_path"], dst_img)
        # Labels
        dst_lbl = out / "labels" / r["split"] / f"{r['page_id']}.txt"
        with open(dst_lbl, "w") as f:
            for bbox in r["boxes"]:
                f.write(to_yolo_line(bbox, r["page_w"], r["page_h"]) + "\n")

    # Write data.yaml
    yaml_path = out / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(
            f"# Auto-generated by scripts/prepare_lyric_dataset.py\n"
            f"path: {out.resolve()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: 1\n"
            f"names:\n"
            f"  0: lyric\n"
        )

    print(f"\nDataset written to {out.resolve()}")
    print(f"  data.yaml: {yaml_path}")
    print(f"  train: {n_train} pages")
    print(f"  val:   {n_val} pages")
    print(f"\nNext: train with `python scripts/train_lyric_detector.py --data {yaml_path}`")


if __name__ == "__main__":
    main()
