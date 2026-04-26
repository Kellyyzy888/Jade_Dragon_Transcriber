#!/usr/bin/env python3
"""
Bootstrap a manual-lyrics JSON for a page by inferring approximate lyric
positions from the gongche label layout.

How it works:
  1. Read the YOLO ground-truth pitch boxes for the page.
  2. Cluster them into vertical columns (using `alignment.cluster_columns`).
  3. Within each column, find vertical gaps between gongche clusters — those
     gaps are where lyric characters sit (suoyi style: notes float above lyric).
  4. Estimate one lyric bbox per gap, with placeholder character "？".
  5. Save to demo_lyrics/<page>.json.

You then open the JSON and replace each "？" with the actual lyric character
read from the page. ~5 minutes per page, vs. measuring pixel coordinates.

Usage:
    python scripts/bootstrap_lyrics.py --page 100
    python scripts/bootstrap_lyrics.py --page 100 --overlay  # also save preview PNG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json

# Ensure the project root is on sys.path so `import jade_dragon` works when
# this script is run as `python scripts/bootstrap_lyrics.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

from jade_dragon.alignment import cluster_columns, yolo_label_to_boxes
from jade_dragon.pitch_map import CLASS_NAMES

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "LGRC2024 dataset" / "datasets"
DEMO_LYRICS = ROOT / "demo_lyrics"


def find_image_and_label(page_id: str) -> tuple[Path, Path]:
    for split in ("train", "val"):
        img = DATASET / "images" / split / f"{page_id}.jpg"
        lbl = DATASET / "labels" / split / f"{page_id}.txt"
        if img.exists() and lbl.exists():
            return img, lbl
    raise FileNotFoundError(f"page {page_id} not found in train or val")


def infer_lyric_boxes(pitch_boxes: list[dict], page_w: int, page_h: int) -> list[dict]:
    """Estimate lyric box positions from gongche layout."""
    columns = cluster_columns(pitch_boxes)
    lyrics: list[dict] = []

    for col_idx, col in enumerate(columns):
        if not col:
            continue
        # Sort by center-y
        col_sorted = sorted(col, key=lambda b: 0.5 * (b["bbox"][1] + b["bbox"][3]))

        # Estimate lyric size from gongche size: lyrics are typically 2-3x bigger
        med_h = sorted([b["bbox"][3] - b["bbox"][1] for b in col_sorted])[len(col_sorted) // 2]
        med_w = sorted([b["bbox"][2] - b["bbox"][0] for b in col_sorted])[len(col_sorted) // 2]
        lyric_w = int(med_w * 2.5)
        lyric_h = int(med_h * 2.5)

        # Approximate lyric x: just left of the gongche column (since suoyi puts
        # gongche to the upper-right of its lyric).
        col_xs = [0.5 * (b["bbox"][0] + b["bbox"][2]) for b in col_sorted]
        col_cx = sum(col_xs) / len(col_xs)
        lyric_cx = col_cx - lyric_w * 0.5  # shift left

        # Find vertical gaps in the gongche cluster sequence — these are
        # candidate lyric positions.
        # Heuristic: walk down the column, group consecutive gongche into a
        # cluster (notes within `med_h * 1.5` of each other), then the END of
        # each cluster is approximately above a lyric.
        clusters: list[list[dict]] = []
        cur: list[dict] = []
        prev_cy = None
        gap_thresh = med_h * 2.0
        for b in col_sorted:
            cy = 0.5 * (b["bbox"][1] + b["bbox"][3])
            if prev_cy is not None and (cy - prev_cy) > gap_thresh:
                if cur:
                    clusters.append(cur)
                cur = [b]
            else:
                cur.append(b)
            prev_cy = cy
        if cur:
            clusters.append(cur)

        # Each cluster's bottom edge is approximately where the lyric sits.
        for cluster in clusters:
            bottoms = [b["bbox"][3] for b in cluster]
            cluster_bottom = max(bottoms)
            # lyric below the cluster
            ly_top = cluster_bottom + med_h * 0.2
            ly_bot = ly_top + lyric_h
            ly_x1 = lyric_cx - lyric_w / 2
            ly_x2 = lyric_cx + lyric_w / 2
            lyrics.append({
                "bbox": [
                    max(0, float(ly_x1)),
                    max(0, float(ly_top)),
                    min(page_w, float(ly_x2)),
                    min(page_h, float(ly_bot)),
                ],
                "char": "？",
                "_column": col_idx,
            })
    return lyrics


def overlay_preview(image_path: Path, lyrics: list[dict], pitches: list[dict], out_path: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for p in pitches:
        x1, y1, x2, y2 = p["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    for ly in lyrics:
        x1, y1, x2, y2 = ly["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=4)
    img.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", required=True)
    ap.add_argument("--out", default=None, help="Output JSON path (default: demo_lyrics/<page>.json)")
    ap.add_argument("--overlay", action="store_true",
                    help="Also save a preview PNG with red=pitches, blue=lyrics")
    args = ap.parse_args()

    img_path, lbl_path = find_image_and_label(args.page)
    img = Image.open(img_path)
    page_w, page_h = img.size

    all_boxes = yolo_label_to_boxes(str(lbl_path), page_w, page_h, CLASS_NAMES)
    pitch_boxes = [b for b in all_boxes if b["kind"] == "pitch"]

    lyrics = infer_lyric_boxes(pitch_boxes, page_w, page_h)
    # strip private fields before saving
    payload = {
        "page_w": page_w,
        "page_h": page_h,
        "lyrics": [{"bbox": ly["bbox"], "char": ly["char"]} for ly in lyrics],
    }

    out = Path(args.out) if args.out else DEMO_LYRICS / f"{args.page}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(lyrics)} placeholder lyrics to {out}")
    print(f"Edit the file and replace each '？' with the actual lyric character.")

    if args.overlay:
        overlay_path = out.with_suffix(".preview.png")
        overlay_preview(img_path, lyrics, pitch_boxes, overlay_path)
        print(f"Preview saved to {overlay_path}")


if __name__ == "__main__":
    main()
