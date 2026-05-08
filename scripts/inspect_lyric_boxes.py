"""
Render the lyric YOLO boxes on a page so we can see exactly what got detected.
Each box gets numbered + colored by column. Useful for diagnosing
"chars came back padded with ？" cases.

Usage:
    python3 inspect_lyric_boxes.py lyric_dataset/images/val/137.jpg
    python3 inspect_lyric_boxes.py lyric_dataset/images/val/137.jpg --conf 0.55
    python3 inspect_lyric_boxes.py lyric_dataset/images/val/137.jpg --conf 0.55 --weights checkpoints/lyric_yolov8m_best.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from jade_dragon.lyric_ocr import LyricDetector
from jade_dragon.alignment import cluster_columns


COLORS = [
    (255,  60,  60),   # red    — column 0
    (60,  170, 255),   # blue   — column 1
    (60,  200,  90),   # green  — column 2
    (255, 180,  40),   # orange — column 3
    (200,  60, 255),   # purple — column 4
]


def main(image_path: str, weights: str, conf: float) -> None:
    image_path = Path(image_path)
    # Encode conf in the output name so multiple sweeps don't clobber each other.
    suffix = f"_lyric_boxes_conf{int(conf*100):03d}.jpg"
    out_path = image_path.parent / f"{image_path.stem}{suffix}"

    detector = LyricDetector(weights, conf=conf)
    boxes = detector.detect(image_path)
    print(f"\n[detector] conf={conf}, weights={weights}")
    print(f"[detector] {len(boxes)} total boxes before column clustering\n")

    columns = cluster_columns(boxes, column_eps_factor=1.5)

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 24,
        )
    except OSError:
        font = ImageFont.load_default()

    for col_idx, col in enumerate(columns):
        col_sorted = sorted(col, key=lambda b: 0.5 * (b["bbox"][1] + b["bbox"][3]))
        color = COLORS[col_idx % len(COLORS)]
        print(f"Column {col_idx}: {len(col_sorted)} boxes")
        for i, b in enumerate(col_sorted):
            x1, y1, x2, y2 = b["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            label = f"c{col_idx}#{i} {b['score']:.2f}"
            draw.rectangle([x1, y1 - 28, x1 + 130, y1], fill=color)
            draw.text((x1 + 4, y1 - 26), label, fill=(255, 255, 255), font=font)
            print(f"  c{col_idx}#{i:2d}: bbox={tuple(round(v,1) for v in b['bbox'])} score={b['score']:.3f}")

    img.save(out_path, quality=92)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image_path")
    ap.add_argument("--weights", default="checkpoints/lyric_yolov8m_best.pt")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="YOLO confidence threshold (default 0.25 for diagnostic)")
    args = ap.parse_args()
    main(args.image_path, args.weights, args.conf)
