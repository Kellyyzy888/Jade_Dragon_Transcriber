#!/usr/bin/env python3
"""
Train YOLOv8m on the LGRC2024 dataset for gongche pitch detection.

Reproduces the LGRC2024 paper's baseline (~78% accuracy / ~74% mAP50) before
attention-mechanism additions. Use as a starting checkpoint for downstream
alignment + transcription work.

Usage:
    python scripts/train_pitch_detector.py --data configs/lgrc2024.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/lgrc2024.yaml", help="Path to data.yaml")
    ap.add_argument("--model", default="yolov8m.pt", help="Base model (downloaded if missing)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--name", default="lgrc2024_yolov8m", help="Run name under runs/detect/")
    ap.add_argument("--device", default="0", help='"0" for GPU 0, "cpu" for CPU')
    args = ap.parse_args()

    from ultralytics import YOLO  # imported here so script is light when not training
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        optimizer=args.optimizer,
        name=args.name,
        device=args.device,
        # Disable rotation/flip augmentations — they break gongche reading order.
        # Keep mosaic/mixup off since they corrupt page-level structure too.
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        # Keep mild HSV jitter for paper-color robustness
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
    )

    print(f"\nDone. Best weights at: runs/detect/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
