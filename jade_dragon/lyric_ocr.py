"""
Lyric character detection + recognition.

Three backends:

  1. easyocr (default, recommended on macOS) — uses PyTorch (already installed
     for the YOLO pipeline). Reliable on Apple Silicon. `pip install easyocr`.
  2. paddleocr — fast on Linux GPU, but has known hangs/threading issues on
     Mac arm64. Kept for completeness.
  3. manual — hand-edited JSON file (`load_manual_lyrics`). Fallback for
     pages where OCR fails or for very high-stakes evaluation.

A page has many text regions: lyrics (large), stage directions, key-name
labels, page numbers, source attribution. We want only the lyric characters.
The cheap filter that mostly works: lyrics are the LARGEST text on a page,
so we keep boxes whose height is ≥ `size_filter_ratio` × median height.
That drops most of the small metadata text reliably.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union


def detect_and_recognize(
    image_path: Union[str, Path],
    *,
    backend: str = "easyocr",
    size_filter_ratio: float = 0.6,
    language_codes: Optional[list[str]] = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Detect and recognize lyric characters on a page.

    Returns Box dicts with kind="lyric", compatible with `alignment.align_page`.

    Parameters
    ----------
    backend : "easyocr" (default) | "paddleocr"
    size_filter_ratio : keep boxes with height ≥ ratio * median height. 0.6 is a
        good default — drops small metadata text (page numbers, stage directions)
        while keeping the large lyric characters and the title.
    language_codes : passed to the backend. EasyOCR defaults to
        ["ch_tra", "en"] (Traditional Chinese + safety net).
    """
    if backend == "easyocr":
        return _detect_easyocr(
            image_path,
            language_codes=language_codes,
            size_filter_ratio=size_filter_ratio,
            verbose=verbose,
        )
    elif backend == "paddleocr":
        return _detect_paddleocr(
            image_path,
            language_codes=language_codes,
            size_filter_ratio=size_filter_ratio,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'easyocr' or 'paddleocr'.")


# ---- EasyOCR backend ---------------------------------------------------------


_EASYOCR_READER_CACHE: dict[tuple[str, ...], "object"] = {}


def _get_easyocr_reader(language_codes: tuple[str, ...]):
    """Cache the Reader so we don't reload the (~150 MB) models per call."""
    if language_codes not in _EASYOCR_READER_CACHE:
        import easyocr
        _EASYOCR_READER_CACHE[language_codes] = easyocr.Reader(
            list(language_codes), gpu=False, verbose=False,
        )
    return _EASYOCR_READER_CACHE[language_codes]


def _detect_easyocr(
    image_path: Union[str, Path],
    *,
    language_codes: Optional[list[str]] = None,
    size_filter_ratio: float = 0.6,
    verbose: bool = True,
) -> list[dict]:
    """EasyOCR backend. Lazy-imports easyocr so the module loads even without it."""
    if language_codes is None:
        # Traditional Chinese (Lilu Qupu uses traditional characters).
        # Adding 'en' is a no-op safety net for any Latin-script noise.
        language_codes = ["ch_tra", "en"]

    reader = _get_easyocr_reader(tuple(language_codes))

    # readtext returns: list of (bbox, text, confidence)
    # bbox is a list of 4 points [[x,y], [x,y], [x,y], [x,y]]
    raw = reader.readtext(str(image_path))

    boxes: list[dict] = []
    for poly, text, score in raw:
        text = (text or "").strip()
        if not text:
            continue
        # Drop non-CJK detections (Latin punctuation, numbers, etc.)
        if not any("\u4e00" <= c <= "\u9fff" for c in text):
            continue

        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x1, y1, x2, y2 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))

        if len(text) == 1:
            boxes.append({
                "bbox": (x1, y1, x2, y2),
                "kind": "lyric",
                "label": text,
                "score": float(score),
            })
        else:
            # Multi-char detection — naive split into vertical slices.
            # Lyrics are vertical (top-to-bottom), so this is appropriate.
            n = len(text)
            h = (y2 - y1) / n
            for i, c in enumerate(text):
                ya = y1 + i * h
                yb = y1 + (i + 1) * h
                boxes.append({
                    "bbox": (x1, ya, x2, yb),
                    "kind": "lyric",
                    "label": c,
                    "score": float(score),
                })

    # Size filter: drop small text (metadata), keep large text (lyrics)
    if size_filter_ratio > 0 and boxes:
        heights = sorted([b["bbox"][3] - b["bbox"][1] for b in boxes])
        median_h = heights[len(heights) // 2]
        threshold = size_filter_ratio * median_h
        before = len(boxes)
        boxes = [b for b in boxes if (b["bbox"][3] - b["bbox"][1]) >= threshold]
        if verbose:
            print(f"[easyocr] size filter: {before} → {len(boxes)} boxes "
                  f"(median height={median_h:.0f}, threshold={threshold:.0f})")

    return boxes


# ---- PaddleOCR backend -------------------------------------------------------


def _detect_paddleocr(
    image_path: Union[str, Path],
    *,
    language_codes: Optional[list[str]] = None,
    size_filter_ratio: float = 0.6,
    verbose: bool = True,
) -> list[dict]:
    """
    PaddleOCR backend. Known to hang on Mac arm64 — use easyocr instead.
    Kept here for Linux GPU users.
    """
    from paddleocr import PaddleOCR  # heavy lazy import

    lang = (language_codes or ["ch"])[0]
    ocr = PaddleOCR(use_angle_cls=False, lang=lang, show_log=False)
    result = ocr.ocr(str(image_path), cls=False)

    boxes: list[dict] = []
    if not result or not result[0]:
        return boxes

    for line in result[0]:
        poly, (text, score) = line
        text = (text or "").strip()
        if not text:
            continue
        if not any("\u4e00" <= c <= "\u9fff" for c in text):
            continue

        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x1, y1, x2, y2 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))

        if len(text) == 1:
            boxes.append({
                "bbox": (x1, y1, x2, y2),
                "kind": "lyric",
                "label": text,
                "score": float(score),
            })
        else:
            n = len(text)
            h = (y2 - y1) / n
            for i, c in enumerate(text):
                ya = y1 + i * h
                yb = y1 + (i + 1) * h
                boxes.append({
                    "bbox": (x1, ya, x2, yb),
                    "kind": "lyric",
                    "label": c,
                    "score": float(score),
                })

    if size_filter_ratio > 0 and boxes:
        heights = sorted([b["bbox"][3] - b["bbox"][1] for b in boxes])
        median_h = heights[len(heights) // 2]
        threshold = size_filter_ratio * median_h
        before = len(boxes)
        boxes = [b for b in boxes if (b["bbox"][3] - b["bbox"][1]) >= threshold]
        if verbose:
            print(f"[paddleocr] size filter: {before} → {len(boxes)} boxes "
                  f"(median height={median_h:.0f}, threshold={threshold:.0f})")

    return boxes


# ---- Manual JSON I/O ---------------------------------------------------------


def load_manual_lyrics(json_path: Union[str, Path]) -> list[dict]:
    """
    Load hand-edited lyric annotations from a JSON file.

    Format:
        {
          "page_w": 2480, "page_h": 3508,
          "lyrics": [
            {"bbox": [x1, y1, x2, y2], "char": "謁"},
            ...
          ]
        }
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        {
            "bbox": tuple(item["bbox"]),
            "kind": "lyric",
            "label": item["char"],
            "score": 1.0,
        }
        for item in data.get("lyrics", [])
    ]


def save_manual_lyrics(
    boxes: list[dict],
    json_path: Union[str, Path],
    *,
    page_w: Optional[int] = None,
    page_h: Optional[int] = None,
) -> None:
    """Inverse of `load_manual_lyrics`."""
    payload = {
        "page_w": page_w,
        "page_h": page_h,
        "lyrics": [
            {"bbox": list(b["bbox"]), "char": b["label"]}
            for b in boxes
            if b.get("kind") == "lyric"
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
