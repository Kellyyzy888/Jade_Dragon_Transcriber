"""
Pitch (gongche character) detector — thin wrapper around Ultralytics YOLOv8.

Loads a trained checkpoint and produces Box dicts compatible with
`alignment.align_page`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from .pitch_map import CLASS_NAMES, is_pitch_class, is_key_signature_class


class PitchDetector:
    def __init__(
        self,
        weights: Union[str, Path],
        *,
        conf: float = 0.25,
        iou: float = 0.5,
        device: str = "auto",
    ):
        from ultralytics import YOLO  # imported lazily so the rest of the package
                                       # doesn't pull torch when only using GT labels
        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        self.device = device

    def detect(self, image_path: Union[str, Path]) -> list[dict]:
        """
        Run detection on a single image. Returns Box dicts with
        kind ∈ {"pitch", "key"} and label = class name.
        """
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf,
            iou=self.iou,
            device=None if self.device == "auto" else self.device,
            verbose=False,
        )
        boxes_out: list[dict] = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            scores = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, s in zip(xyxy, cls, scores):
                label = CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else f"class-{c}"
                if is_pitch_class(label):
                    kind = "pitch"
                elif is_key_signature_class(label):
                    kind = "key"
                else:
                    kind = "other"
                boxes_out.append({
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "kind": kind,
                    "label": label,
                    "score": float(s),
                })
        return boxes_out


def infer_key_class(boxes: list[dict]) -> str | None:
    """Return the key class label (highest-confidence keySignature-* box), or None."""
    keys = [b for b in boxes if b.get("kind") == "key"]
    if not keys:
        return None
    keys.sort(key=lambda b: b.get("score", 0.0), reverse=True)
    return keys[0]["label"]
