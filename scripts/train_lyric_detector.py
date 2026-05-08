#!/usr/bin/env python3
"""
Train a single-class YOLO detector for lyric character bounding boxes.

Why a separate detector (not a 29-class joint with pitches): lyrics are an
open vocabulary (~3000+ unique characters across Lilu Qupu), while pitches
are a closed 28-class vocabulary. A joint classifier head wouldn't
generalize to unseen lyric characters; a single-class detector learns the
visual properties of "lyric character bbox" and generalizes cleanly.

Recognition (which character is in each box) is handled separately by the
VLM in `jade_dragon.lyric_ocr.vlm_recognize_page`.

Settings mirror `train_pitch_detector.py`: rotation/flip augmentation
disabled (gongche pages have a fixed reading orientation), mild HSV jitter
for paper-color robustness.

Usage:
    python scripts/train_lyric_detector.py
    python scripts/train_lyric_detector.py --data lyric_dataset/data.yaml --epochs 100
    python scripts/train_lyric_detector.py --resume runs/detect/lyric_yolov8m/weights/last.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="lyric_dataset/data.yaml",
                    help="Path to the lyric data.yaml (built by prepare_lyric_dataset.py)")
    ap.add_argument("--model", default="yolov8m.pt",
                    help="Base weights — yolov8m.pt for from-ImageNet, or "
                         "checkpoints/lgrc2024_yolov8m_best.pt for warm-starting "
                         "from the pitch detector's backbone.")
    ap.add_argument("--epochs", type=int, default=80,
                    help="80 is comfortable for a single-class detector with "
                         "30-50 train pages; bump if you have more annotation.")
    ap.add_argument("--patience", type=int, default=20,
                    help="Early stopping patience. Pitch detector overfit past "
                         "epoch ~30 with patience=50; tighter is better.")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--name", default="lyric_yolov8m",
                    help="Run name under runs/detect/")
    ap.add_argument("--device", default="0")
    ap.add_argument("--resume", default=None,
                    help="Path to a last.pt to resume training from")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=args.data,
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            imgsz=args.imgsz,
            lr0=args.lr0,
            optimizer=args.optimizer,
            name=args.name,
            device=args.device,
            # Augmentation policy mirrors train_pitch_detector.py:
            # no rotation/flip (gongche reading order is fixed), no mosaic/mixup
            # (would corrupt page-level structure), keep mild HSV jitter.
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            flipud=0.0,
            fliplr=0.0,
            mosaic=0.0,
            mixup=0.0,
            hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        )

    print(f"\nDone. Best weights at: runs/detect/{args.name}/weights/best.pt")
    print(f"Copy to checkpoints/ for use in the pipeline:")
    print(f"  cp runs/detect/{args.name}/weights/best.pt "
          f"checkpoints/{args.name}_best.pt")


if __name__ == "__main__":
    main()
