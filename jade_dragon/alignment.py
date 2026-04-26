"""
Geometric alignment of lyric characters with their gongche pitch clusters.

Suoyi-style convention (the only style in LGRC2024):
    - Lyrics flow top-to-bottom in vertical columns.
    - Columns are read right-to-left.
    - Each lyric character's gongche pitches sit *above* it (between the previous
      lyric character and this one) and slightly to the right.
    - Within a cluster, pitches are read top-to-bottom, then left-to-right for
      side-by-side notes (rare — most clusters are vertically stacked).

This module is a pure-geometric heuristic. It does NOT learn alignment;
that's a Stage-2 ML upgrade. The heuristic gets you to a working MVP and a
solid baseline to compare a learned aligner against.

Key data structure — a Box:
    {
        "bbox": (x1, y1, x2, y2),   # pixel coordinates
        "kind": "lyric" | "pitch" | "key",
        "label": str,                # lyric character or pitch class name
        "score": float,              # detector confidence (optional)
    }

Output:
    [
        {
            "lyric": Box,            # the lyric box
            "pitches": [Box, ...],   # pitches assigned, in reading order
            "column": int,           # column index (0 = rightmost)
        },
        ...
    ]
"""

from __future__ import annotations

from typing import Iterable
import numpy as np

Box = dict


def _cx(b: Box) -> float:
    x1, _, x2, _ = b["bbox"]
    return 0.5 * (x1 + x2)


def _cy(b: Box) -> float:
    _, y1, _, y2 = b["bbox"]
    return 0.5 * (y1 + y2)


def _height(b: Box) -> float:
    _, y1, _, y2 = b["bbox"]
    return y2 - y1


def _width(b: Box) -> float:
    x1, _, x2, _ = b["bbox"]
    return x2 - x1


# -------------------------------------------------------------------------------
# Column clustering
# -------------------------------------------------------------------------------

def cluster_columns(
    boxes: list[Box],
    *,
    column_eps_factor: float = 1.5,
) -> list[list[Box]]:
    """
    Cluster boxes into vertical columns by horizontal position of their centers.

    `column_eps_factor`: a box is added to a column if its center-x is within
    `column_eps_factor * median_lyric_width` of an existing column's centroid.
    Returns columns sorted right-to-left (Kunqu reading order).

    Robust to mixed lyric/pitch boxes — uses lyric widths to set the scale, since
    lyric characters are bigger and more uniform than gongche characters.
    """
    if not boxes:
        return []

    lyric_widths = [_width(b) for b in boxes if b.get("kind") == "lyric"]
    if lyric_widths:
        scale = float(np.median(lyric_widths))
    else:
        # No lyrics — fall back to median width of all boxes
        scale = float(np.median([_width(b) for b in boxes]))
    eps = column_eps_factor * scale

    # Sort by center-x descending (right-to-left) and greedy cluster
    sorted_boxes = sorted(boxes, key=_cx, reverse=True)
    columns: list[list[Box]] = []
    centroids: list[float] = []
    for b in sorted_boxes:
        cx = _cx(b)
        # find nearest existing column centroid
        if centroids:
            distances = [abs(cx - c) for c in centroids]
            nearest = int(np.argmin(distances))
            if distances[nearest] <= eps:
                columns[nearest].append(b)
                # recompute centroid as mean of lyric centers (more stable)
                lyric_cxs = [_cx(x) for x in columns[nearest] if x.get("kind") == "lyric"]
                centroids[nearest] = float(np.mean(lyric_cxs)) if lyric_cxs else float(np.mean(
                    [_cx(x) for x in columns[nearest]]
                ))
                continue
        columns.append([b])
        centroids.append(cx)

    # Sort right-to-left explicitly (already mostly there from the greedy pass)
    order = np.argsort(centroids)[::-1]
    columns = [columns[i] for i in order]
    return columns


# -------------------------------------------------------------------------------
# Within-column lyric ↔ pitch alignment
# -------------------------------------------------------------------------------

def align_column(
    column: list[Box],
    *,
    horizontal_tolerance: float = 2.0,
) -> list[dict]:
    """
    Within a single column, group pitches with their lyric characters.

    Algorithm:
      1. Split the column's boxes into `lyrics` and `pitches`.
      2. Sort lyrics top-to-bottom by center-y.
      3. For each pitch, find the lyric whose center-y is just BELOW the pitch's
         center-y (suoyi: notes float above their lyric).
         - If no lyric is below (pitch is below the bottommost lyric), assign
           to the bottommost lyric.
      4. Within each lyric's cluster, sort pitches top-to-bottom, then left-to-right.

    `horizontal_tolerance`: pitches more than `tol * median_pitch_width` away
    from the column's lyric centroid horizontally are dropped (likely a mis-cluster).
    """
    lyrics = [b for b in column if b.get("kind") == "lyric"]
    pitches = [b for b in column if b.get("kind") == "pitch"]

    if not lyrics:
        # Edge case: pitches without any lyric in the column. Return a single
        # synthetic group so we don't lose them.
        return [{"lyric": None, "pitches": _sort_cluster(pitches), "column": None}]

    lyrics = sorted(lyrics, key=_cy)
    lyric_cys = [_cy(l) for l in lyrics]

    # Optional horizontal filtering — drop pitches far from the column's lyric x-centroid
    if pitches:
        col_cx = float(np.mean([_cx(l) for l in lyrics]))
        pitch_widths = [_width(p) for p in pitches] or [10.0]
        max_dx = horizontal_tolerance * float(np.median(pitch_widths)) + max(_width(l) for l in lyrics)
        pitches = [p for p in pitches if abs(_cx(p) - col_cx) <= max_dx]

    # Assign each pitch to a lyric
    clusters: dict[int, list[Box]] = {i: [] for i in range(len(lyrics))}
    for p in pitches:
        py = _cy(p)
        # find first lyric whose cy >= py (i.e., next lyric below the pitch)
        target = None
        for i, ly in enumerate(lyric_cys):
            if ly >= py:
                target = i
                break
        if target is None:
            # pitch is below the bottommost lyric — assign to last
            target = len(lyrics) - 1
        clusters[target].append(p)

    out: list[dict] = []
    for i, ly in enumerate(lyrics):
        out.append({
            "lyric": ly,
            "pitches": _sort_cluster(clusters[i]),
            "column": None,  # filled in by caller
        })
    return out


def _sort_cluster(pitches: list[Box]) -> list[Box]:
    """Reading order within a single melisma cluster: top-to-bottom, then left-to-right."""
    if not pitches:
        return []
    # Use a coarse vertical bucket so side-by-side notes at the same y are sorted L-R.
    heights = [_height(p) for p in pitches]
    bucket = float(np.median(heights)) * 0.6 if heights else 5.0
    return sorted(pitches, key=lambda p: (round(_cy(p) / bucket) * bucket, _cx(p)))


# -------------------------------------------------------------------------------
# Top-level: full page alignment
# -------------------------------------------------------------------------------

def align_page(
    lyric_boxes: list[Box],
    pitch_boxes: list[Box],
    *,
    column_eps_factor: float = 1.5,
) -> list[dict]:
    """
    Align an entire page: cluster into columns, align within each column.
    Returns a flat list in reading order (rightmost column first, top-to-bottom within).
    """
    boxes: list[Box] = list(lyric_boxes) + list(pitch_boxes)
    if not boxes:
        return []

    columns = cluster_columns(boxes, column_eps_factor=column_eps_factor)

    aligned: list[dict] = []
    for col_idx, col in enumerate(columns):
        col_aligned = align_column(col)
        for item in col_aligned:
            item["column"] = col_idx
            aligned.append(item)
    return aligned


# -------------------------------------------------------------------------------
# Convenience: build Box dicts from YOLO label files (for ground-truth eval)
# -------------------------------------------------------------------------------

def yolo_label_to_boxes(
    label_path: str,
    image_width: int,
    image_height: int,
    class_names: list[str],
) -> list[Box]:
    """
    Read a YOLO label file (class cx cy w h, normalized) and return Box dicts.

    Boxes are tagged "kind=pitch" if the class is a note-* or keySignature-*
    (LGRC2024 has no lyric class). For end-to-end demos using GT labels for
    pitches, you supply lyric boxes separately.
    """
    boxes: list[Box] = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:5])
            x1 = (cx - w / 2) * image_width
            y1 = (cy - h / 2) * image_height
            x2 = (cx + w / 2) * image_width
            y2 = (cy + h / 2) * image_height
            label = class_names[cls] if 0 <= cls < len(class_names) else f"class-{cls}"
            kind = "key" if label.startswith("keySignature-") else "pitch"
            boxes.append({
                "bbox": (x1, y1, x2, y2),
                "kind": kind,
                "label": label,
                "score": 1.0,
            })
    return boxes


# ---- Smoke test ---------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic mini-test: one column with 2 lyrics, each with a small cluster of pitches
    lyrics = [
        {"bbox": (100, 100, 140, 150), "kind": "lyric", "label": "謁"},
        {"bbox": (100, 220, 140, 270), "kind": "lyric", "label": "金"},
    ]
    pitches = [
        # cluster above 謁 (y < 100)
        {"bbox": (155, 30, 175, 50), "kind": "pitch", "label": "note-C4"},
        {"bbox": (155, 60, 175, 80), "kind": "pitch", "label": "note-D4"},
        # cluster between 謁 and 金 (y between 150 and 220)
        {"bbox": (155, 160, 175, 180), "kind": "pitch", "label": "note-E4"},
        {"bbox": (155, 190, 175, 210), "kind": "pitch", "label": "note-F4"},
    ]
    aligned = align_page(lyrics, pitches)
    assert len(aligned) == 2
    assert aligned[0]["lyric"]["label"] == "謁"
    assert [p["label"] for p in aligned[0]["pitches"]] == ["note-C4", "note-D4"]
    assert aligned[1]["lyric"]["label"] == "金"
    assert [p["label"] for p in aligned[1]["pitches"]] == ["note-E4", "note-F4"]
    print("alignment.py smoke tests passed")
