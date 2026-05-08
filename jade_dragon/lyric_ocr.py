"""
Lyric character detection + recognition.

Four backends:

  1. easyocr — PyTorch-based, reliable on Apple Silicon. Fails on Lilu Qupu's
     calligraphic kaishu in practice (treats lyrics as decoration). `pip install easyocr`.
  2. paddleocr — fast on Linux GPU, but has known hangs/threading issues on
     Mac arm64. Kept for completeness; same calligraphy weakness as easyocr.
  3. vlm (recommended for calligraphic pages) — sends per-column crops to a
     vision-language model (Anthropic Claude by default). Position geometry
     comes from `vlm_fill_lyrics`, which expects bootstrap boxes computed
     from gongche column layout. ~$0.01 / page on Sonnet vision.
  4. manual — hand-edited JSON file (`load_manual_lyrics`). Fallback for
     pages where OCR fails or for very high-stakes evaluation.

Standard OCR backends (easyocr/paddleocr) try to find AND recognize text on
their own. The VLM backend only does recognition — positions are taken from
the existing bootstrap geometry in `scripts/bootstrap_lyrics.py`. This split
plays to each tool's strength: gongche layout is rigid and geometric (cheap
to extract), but the lyric *characters* are hand-calligraphed kaishu (which
specialized CJK OCR fails on but VLMs handle gracefully).

A page has many text regions: lyrics (large), stage directions, key-name
labels, page numbers, source attribution. We want only the lyric characters.
The cheap filter that mostly works: lyrics are the LARGEST text on a page,
so we keep boxes whose height is ≥ `size_filter_ratio` × median height.
That drops most of the small metadata text reliably.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Optional, Union


def detect_and_recognize(
    image_path: Union[str, Path],
    *,
    backend: str = "easyocr",
    size_filter_ratio: float = 0.6,
    language_codes: Optional[list[str]] = None,
    lyric_weights: Optional[Union[str, Path]] = None,
    conf: float = 0.25,
    iou: float = 0.5,
    column_eps_factor: float = 1.5,
    vlm_model: Optional[str] = None,
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Detect and recognize lyric characters on a page.

    Returns Box dicts with kind="lyric", compatible with `alignment.align_page`.

    Parameters
    ----------
    backend : "easyocr" (default) | "paddleocr" | "yolo_vlm"
        - "easyocr"/"paddleocr": single-stage CJK OCR. Tends to fail on
          calligraphic kaishu but works for cleaner printed pages.
        - "yolo_vlm": two-stage trained YOLO single-class detector for bboxes
          + Anthropic Claude vision for character recognition. The recommended
          path for Lilu Qupu since the calligraphy defeats CJK OCR. Requires
          `lyric_weights` and `ANTHROPIC_API_KEY`.
    size_filter_ratio : keep boxes with height ≥ ratio * median height. 0.6 is a
        good default — drops small metadata text (page numbers, stage directions)
        while keeping the large lyric characters and the title. Only applies
        to easyocr/paddleocr; yolo_vlm doesn't need it because the detector
        was trained to find only lyric boxes.
    language_codes : passed to easyocr/paddleocr. Defaults to ["ch_tra", "en"].
    lyric_weights, conf, iou, column_eps_factor, vlm_model, api_key :
        forwarded to `detect_and_recognize_lyrics` when backend == "yolo_vlm".
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
    elif backend == "yolo_vlm":
        if lyric_weights is None:
            raise ValueError(
                "backend='yolo_vlm' requires `lyric_weights=...` (path to "
                "the trained lyric YOLO checkpoint, e.g. "
                "checkpoints/lyric_yolov8m_best.pt)."
            )
        kwargs: dict = {
            "conf": conf,
            "iou": iou,
            "column_eps_factor": column_eps_factor,
            "api_key": api_key,
            "verbose": verbose,
        }
        if vlm_model is not None:
            kwargs["model"] = vlm_model
        return detect_and_recognize_lyrics(
            image_path,
            lyric_weights,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Use 'easyocr', 'paddleocr', or 'yolo_vlm'."
        )


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


# ---- YOLO lyric detector + VLM recognizer ------------------------------------
#
# This is the architectural successor to the geometry-only `vlm_recognize_page`:
# real lyric bboxes from a trained single-class YOLO, characters from the VLM.
#
# Why single class: lyric character identity is open-vocabulary
# (~3000+ unique characters in Lilu Qupu); a YOLO classifier head wouldn't
# generalize to unseen characters. A single-class detector learns "what does a
# lyric character bbox look like in this style of page" — which generalizes.
# Recognition (which character) happens at the VLM step.

# Defined here (rather than later) because LyricDetector + helpers reference it.
_DEFAULT_VLM_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"  # Chinese-native, strong CJK OCR & calligraphy


# A "VLM client" is a small dict carrying everything needed to make a chat call.
#   {"backend": "anthropic", "client": Anthropic(...), "model_default": str}
#   {"backend": "openrouter", "api_key": str, "model_default": str}
# We wrap so call sites stay uniform via _vlm_chat(...).

def _make_vlm_client(api_key: Optional[str] = None) -> dict:
    """
    Pick a VLM backend and construct a thin client descriptor.

    Routing logic, in priority order:
      1. Explicit `api_key=...` → Anthropic direct.
      2. OPENROUTER_API_KEY env → OpenRouter (HTTP, OpenAI-compatible).
         Recommended when direct Anthropic access isn't available.
      3. ANTHROPIC_API_KEY env → Anthropic direct.
      4. Otherwise → RuntimeError.

    Returns a dict (not an SDK object directly) so that `_vlm_chat` can
    dispatch on backend. The model name in `_DEFAULT_VLM_MODEL` is used
    as-is for Anthropic. For OpenRouter we prepend "anthropic/" if the
    caller doesn't already.
    """
    if api_key:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "Anthropic backend requires the `anthropic` package. "
                "Install with: pip install anthropic"
            ) from e
        return {
            "backend": "anthropic",
            "client": Anthropic(api_key=api_key),
        }

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return {
            "backend": "openrouter",
            "api_key": openrouter_key,
        }

    direct_key = os.environ.get("ANTHROPIC_API_KEY")
    if direct_key:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "Anthropic backend requires the `anthropic` package. "
                "Install with: pip install anthropic"
            ) from e
        return {
            "backend": "anthropic",
            "client": Anthropic(api_key=direct_key),
        }

    raise RuntimeError(
        "No VLM API key found. Set OPENROUTER_API_KEY for OpenRouter access, "
        "or ANTHROPIC_API_KEY for direct Anthropic access."
    )


def _vlm_chat(
    vlm_client: dict,
    *,
    model: str,
    prompt: str,
    image_b64_png: str,
    max_tokens: int = 512,
) -> str:
    """
    One-shot vision+text call. Returns plain text response.

    Dispatches on `vlm_client["backend"]`:
      - "anthropic": uses the Anthropic SDK's messages.create.
      - "openrouter": uses OpenRouter's OpenAI-compatible /chat/completions
        endpoint via httpx. Model names are auto-prefixed with "anthropic/"
        if the caller passes a bare Claude name.
    """
    backend = vlm_client["backend"]

    if backend == "anthropic":
        client = vlm_client["client"]
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64_png,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return "".join(
            block.text for block in resp.content
            if getattr(block, "type", None) == "text"
        )

    if backend == "openrouter":
        # httpx is a transitive dep of the anthropic SDK, so always available.
        import httpx

        # If the caller passes a bare Claude name (no provider prefix),
        # qualify it for OpenRouter. Any name that already contains "/"
        # passes through unchanged (e.g. "qwen/qwen3-vl-235b-a22b-instruct",
        # "google/gemini-2.5-flash", "openai/gpt-4o").
        or_model = model
        if "/" not in or_model and or_model.startswith("claude-"):
            or_model = f"anthropic/{or_model}"

        body = {
            "model": or_model,
            "max_tokens": max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64_png}",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        headers = {
            "Authorization": f"Bearer {vlm_client['api_key']}",
            "Content-Type": "application/json",
            # Optional but recommended by OpenRouter so usage attributes correctly:
            "HTTP-Referer": "https://github.com/jade-dragon",
            "X-Title": "JadeDragon Transcriber",
        }
        with httpx.Client(timeout=60.0) as http:
            r = http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=body,
                headers=headers,
            )
        if r.status_code >= 400:
            raise RuntimeError(
                f"OpenRouter HTTP {r.status_code}: {r.text[:500]}"
            )
        data = r.json()
        # OpenAI-compatible: choices[0].message.content is a string.
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"OpenRouter response missing choices[0].message.content: "
                f"{str(data)[:500]}"
            ) from e

    raise RuntimeError(f"Unknown VLM backend: {backend!r}")





class LyricDetector:
    """Thin wrapper around a single-class YOLO trained for lyric bboxes.

    Returns Box dicts with kind="lyric", label=None, score=detector_conf.
    Recognition fills in `label` later (see `detect_and_recognize_lyrics`).
    """

    def __init__(
        self,
        weights: Union[str, Path],
        *,
        conf: float = 0.25,
        iou: float = 0.5,
        device: str = "auto",
    ):
        from ultralytics import YOLO  # lazy import — keep package light
        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou
        self.device = device

    def detect(self, image_path: Union[str, Path]) -> list[dict]:
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
            scores = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), s in zip(xyxy, scores):
                boxes_out.append({
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "kind": "lyric",
                    "label": None,    # filled in by recognition step
                    "score": float(s),
                })
        return boxes_out


def detect_and_recognize_lyrics(
    image_path: Union[str, Path],
    lyric_weights: Union[str, Path],
    *,
    conf: float = 0.25,
    iou: float = 0.5,
    column_eps_factor: float = 1.5,
    model: str = _DEFAULT_VLM_MODEL,
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> list[dict]:
    """End-to-end lyric pipeline: YOLO bbox detection + per-column VLM recognition.

    Stages:
      1. Run the lyric YOLO on the page → real lyric bboxes (no characters).
      2. Cluster bboxes into columns by x-center.
      3. For each column, crop a vertical strip and ask the VLM to read all
         characters top-to-bottom. The VLM's char count must match the YOLO
         detection count; if not, we pad/truncate and log a warning.
      4. Assign characters to bboxes in y-order within each column.

    Returns Box dicts ready to be passed to `alignment.align_page`.
    """
    from PIL import Image

    client = _make_vlm_client(api_key)

    # ---- Stage 1: detect ------------------------------------------------------
    detector = LyricDetector(lyric_weights, conf=conf, iou=iou)
    bboxes = detector.detect(image_path)
    if verbose:
        print(f"[lyric-yolo] {len(bboxes)} lyric boxes "
              f"(conf range {min((b['score'] for b in bboxes), default=0):.2f}"
              f"..{max((b['score'] for b in bboxes), default=0):.2f})")
    if not bboxes:
        return []

    # ---- Stage 2: column-cluster the lyric bboxes -----------------------------
    from .alignment import cluster_columns
    columns = cluster_columns(bboxes, column_eps_factor=column_eps_factor)
    if verbose:
        print(f"[lyric-yolo] {len(columns)} columns, "
              f"sizes={[len(c) for c in columns]}")

    # ---- Stage 3: VLM recognition per column ----------------------------------
    page_image = Image.open(image_path).convert("RGB")
    out: list[dict] = []
    for col_idx, col in enumerate(columns):
        col_sorted = sorted(col, key=lambda b: 0.5 * (b["bbox"][1] + b["bbox"][3]))
        crop = _column_crop(page_image, col_sorted)
        if verbose:
            print(f"[lyric-yolo] column {col_idx}: {len(col_sorted)} boxes, "
                  f"crop {crop.size[0]}×{crop.size[1]}")

        chars = _vlm_recognize_column(
            crop, len(col_sorted), client=client, model=model,
        )
        if verbose:
            print(f"[lyric-yolo] column {col_idx} chars: {''.join(chars)}")

        # Stage 4: zip chars into the bboxes top-to-bottom
        for box, ch in zip(col_sorted, chars):
            new_box = dict(box)
            new_box["label"] = ch
            new_box["_column"] = col_idx
            new_box["score"] = box.get("score", 1.0) if ch != "？" else 0.0
            out.append(new_box)

    return out


# ---- VLM (Anthropic Claude vision) backend -----------------------------------
#
# Strategy: positions come from gongche column geometry (cheap, rigid), chars
# come from the VLM (handles calligraphic CJK that EasyOCR/PaddleOCR fail on).
#
# This section pre-dates `LyricDetector` above and is kept for the geometry-
# only path (no trained lyric detector required). Once a lyric YOLO is in
# checkpoints/, prefer `detect_and_recognize_lyrics`.


def _png_b64(image, max_dim: int = 1568) -> str:
    """Encode a PIL image as base64 PNG, downsizing if needed."""
    from PIL import Image  # local to keep import lazy

    if max(image.size) > max_dim:
        scale = max_dim / max(image.size)
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _column_crop(
    page_image,
    boxes_in_column: list[dict],
    *,
    pad_frac_x: float = 0.4,
    pad_frac_top: float = 0.25,
    pad_frac_bot: float = 0.10,
):
    """Crop a vertical strip containing a single lyric column.

    Asymmetric vertical padding: lyrics often extend ABOVE the topmost
    detected gongche (the first lyric character may have little or no
    gongche above it because the melody starts at that lyric), so we
    pad the top generously. The bottom edge of the gongche cluster is
    typically tight against the last lyric in the column, so a smaller
    bottom pad is fine.
    """
    xs = [b["bbox"][0] for b in boxes_in_column] + [b["bbox"][2] for b in boxes_in_column]
    ys = [b["bbox"][1] for b in boxes_in_column] + [b["bbox"][3] for b in boxes_in_column]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    w, h = x2 - x1, y2 - y1
    pw = pad_frac_x * w
    px1 = max(0, int(x1 - pw))
    py1 = max(0, int(y1 - pad_frac_top * h))
    px2 = min(page_image.size[0], int(x2 + pw))
    py2 = min(page_image.size[1], int(y2 + pad_frac_bot * h))
    return page_image.crop((px1, py1, px2, py2))


def _parse_vlm_chars(reply: str, expected_n: int) -> list[str]:
    """
    Extract `expected_n` Han characters from a VLM reply, in order.

    The prompt asks for one char per line, but VLMs sometimes add commentary,
    spaces, or punctuation. Strategy: pull every CJK Unified-Ideograph char
    in order and pad/truncate to `expected_n`.
    """
    # 一-鿿 = CJK Unified Ideographs (covers traditional + simplified)
    chars = re.findall(r"[一-鿿]", reply)
    if len(chars) >= expected_n:
        return chars[:expected_n]
    # pad with placeholder so caller can see which positions are unfilled
    return chars + ["？"] * (expected_n - len(chars))


def _vlm_recognize_column(
    column_crop_image,
    n_chars_expected: int,
    *,
    client,
    model: str,
    max_retries: int = 2,
) -> list[str]:
    """
    Send one column crop to Claude and ask for the `n_chars_expected` lyric
    characters in top-to-bottom order. Returns a list of length `n_chars_expected`
    (padded with '？' if Claude returned fewer chars).
    """
    img_b64 = _png_b64(column_crop_image)

    prompt = (
        f"This is a vertical column from a Kunqu opera gongche-notation page "
        f"(handwritten kaishu calligraphy). Read ONLY the large lyric characters "
        f"in this column, top-to-bottom. There should be exactly {n_chars_expected} "
        f"characters.\n\n"
        f"Output rules:\n"
        f"- One Chinese character per line.\n"
        f"- No commentary, no Pinyin, no English.\n"
        f"- Use traditional characters (do not simplify).\n"
        f"- If you cannot read a character, output 〇 (U+3007) for that line.\n"
        f"- Ignore the smaller gongche notation symbols (合 四 一 上 尺 工 凡 六 五 乙 "
        f"and their octave-up forms). Those are not lyrics.\n"
    )

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            text = _vlm_chat(
                client,
                model=model,
                prompt=prompt,
                image_b64_png=img_b64,
                max_tokens=512,
            )
            return _parse_vlm_chars(text, n_chars_expected)
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt == max_retries:
                raise
    raise RuntimeError(f"unreachable; last error: {last_error}")


def _vlm_recognize_column_unbounded(
    column_crop_image,
    *,
    client,
    model: str,
    max_retries: int = 2,
) -> list[str]:
    """
    Like `_vlm_recognize_column`, but doesn't pin a character count up front.
    Used when the count itself is what we want to discover.
    """
    img_b64 = _png_b64(column_crop_image)
    prompt = (
        "This is a vertical column from a Kunqu opera gongche-notation page "
        "(handwritten kaishu calligraphy). Read EVERY large lyric character "
        "in this column, top-to-bottom.\n\n"
        "Output rules:\n"
        "- One Chinese character per line.\n"
        "- No commentary, no Pinyin, no English, no Roman numerals.\n"
        "- Use traditional characters (do not simplify).\n"
        "- If you cannot read a character, output 〇 (U+3007) for that line.\n"
        "- Ignore the smaller gongche notation symbols (合 四 一 上 尺 工 凡 六 五 乙 "
        "and their octave-up forms with the 亻 prefix). Those are NOT lyrics.\n"
        "- Ignore any small annotations, beat marks, or punctuation.\n"
        "- Do not invent characters; do not pad. Stop when there are no more lyrics."
    )

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            text = _vlm_chat(
                client,
                model=model,
                prompt=prompt,
                image_b64_png=img_b64,
                max_tokens=512,
            )
            chars = re.findall(r"[一-鿿〇]", text)
            return chars
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt == max_retries:
                raise
    raise RuntimeError(f"unreachable; last error: {last_error}")


def vlm_recognize_page(
    image_path: Union[str, Path],
    pitch_boxes: list[dict],
    *,
    column_eps_factor: float = 2.5,
    model: str = _DEFAULT_VLM_MODEL,
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Recognize lyric characters on a page directly from pitch column geometry.

    Pipeline:
      1. Cluster pitch boxes into columns by x-center.
      2. For each column, crop a vertical strip covering all gongche.
      3. Send the crop to a VLM, ask for all lyric characters top-to-bottom.
      4. Place each character at an evenly-spaced y position across the
         column's vertical extent. The exact bbox is a synthetic rectangle
         centered slightly left of the gongche column (suoyi style).

    The VLM determines the character *count* per column. We don't try to
    guess the count from gongche gaps because gongche-within-melisma spacing
    is the same as gongche-between-melisma spacing — that signal isn't there.

    Returns a list of Box dicts with kind="lyric", `_column` set to column
    index, and `label` set to the recognized character (or 〇 if the VLM
    couldn't read it).

    Raises:
        RuntimeError: if no API key is found.
        ImportError: if `anthropic` is not installed.
    """
    from PIL import Image

    client = _make_vlm_client(api_key)

    # Local import to avoid circular: alignment imports nothing from us.
    from .alignment import cluster_columns

    columns = cluster_columns(pitch_boxes, column_eps_factor=column_eps_factor)
    if verbose:
        print(f"[vlm] {len(columns)} columns, sizes={[len(c) for c in columns]}")

    page_image = Image.open(image_path).convert("RGB")
    out: list[dict] = []

    for col_idx, col in enumerate(columns):
        if not col:
            continue
        crop = _column_crop(page_image, col)
        if verbose:
            print(f"[vlm] column {col_idx}: {len(col)} pitches, "
                  f"crop {crop.size[0]}×{crop.size[1]}")
        chars = _vlm_recognize_column_unbounded(crop, client=client, model=model)
        if verbose:
            print(f"[vlm] column {col_idx}: returned {len(chars)} chars: {''.join(chars)}")

        # Synthesize bounding boxes: equally spaced down the column's y-extent.
        # x-position is intentionally aligned with the gongche column rather
        # than shifted left to the "true" lyric position. Reason: the
        # gongche x-range can be wide (octave-up pitches with 亻 prefix sit
        # well to the left of regular gongche), so any offset formula is
        # brittle. What downstream alignment actually needs is for lyric and
        # gongche to cluster into the same column — `cluster_columns` does
        # that on x-center within an eps tolerance, and aligning to the
        # gongche centroid is the most robust choice.
        if not chars:
            continue
        ys = [b["bbox"][1] for b in col] + [b["bbox"][3] for b in col]
        y_top, y_bot = min(ys), max(ys)
        # Pitch-mode x-center: median of pitch x-centers is more robust to
        # octave-up outliers than the mean of all x corners.
        pitch_cxs = sorted(
            0.5 * (b["bbox"][0] + b["bbox"][2]) for b in col
        )
        gongche_cx = pitch_cxs[len(pitch_cxs) // 2]

        # Lyric characters are roughly 2-2.5x bigger than gongche; estimate.
        pitch_widths = [b["bbox"][2] - b["bbox"][0] for b in col]
        med_pw = sorted(pitch_widths)[len(pitch_widths) // 2]
        lyric_size = med_pw * 2.5

        lyric_cx = gongche_cx

        n = len(chars)
        # Place chars at evenly spaced y centers from y_top to y_bot
        for i, ch in enumerate(chars):
            cy = y_top + (i + 0.5) * (y_bot - y_top) / n
            x1 = lyric_cx - lyric_size / 2
            x2 = lyric_cx + lyric_size / 2
            y1 = cy - lyric_size / 2
            y2 = cy + lyric_size / 2
            out.append({
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "kind": "lyric",
                "label": ch,
                "score": 1.0 if ch != "〇" else 0.0,
                "_column": col_idx,
            })
    return out


def vlm_fill_lyrics(
    image_path: Union[str, Path],
    bootstrap_boxes: list[dict],
    *,
    model: str = _DEFAULT_VLM_MODEL,
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Legacy entry: take pre-computed bootstrap boxes and fill chars per column.

    Use this if you already have hand-tuned lyric box positions and just want
    Claude to fill in the characters. For most cases, `vlm_recognize_page`
    is the better entry point because it lets the VLM determine the char count.
    """
    from PIL import Image

    client = _make_vlm_client(api_key)

    by_col: dict[int, list[dict]] = {}
    for b in bootstrap_boxes:
        col = b.get("_column", -1)
        by_col.setdefault(col, []).append(b)

    page_image = Image.open(image_path).convert("RGB")
    out: list[dict] = []

    for col_idx in sorted(by_col.keys()):
        col_boxes = sorted(by_col[col_idx], key=lambda b: b["bbox"][1])
        if not col_boxes:
            continue
        crop = _column_crop(page_image, col_boxes)
        if verbose:
            print(f"[vlm] column {col_idx}: {len(col_boxes)} expected chars, "
                  f"crop {crop.size[0]}×{crop.size[1]}")
        chars = _vlm_recognize_column(
            crop, len(col_boxes), client=client, model=model,
        )
        for box, ch in zip(col_boxes, chars):
            new_box = dict(box)
            new_box["label"] = ch
            new_box["kind"] = "lyric"
            new_box["score"] = 1.0 if ch != "？" else 0.0
            out.append(new_box)
        if verbose:
            print(f"[vlm] column {col_idx}: " + "".join(chars))
    return out


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
def _vlm_recognize_box(
    page_image,
    bbox: tuple,
    *,
    client,
    model: str,
    pad: int = 12,
    max_retries: int = 2,
) -> str:
    """
    Crop a single lyric bbox (with a small pad) and ask the VLM what character
    is inside. Returns a single Han char, or '〇' if the VLM declined.
    """
    x1, y1, x2, y2 = bbox
    pw, ph = page_image.size
    cx1 = max(0, int(x1 - pad))
    cy1 = max(0, int(y1 - pad))
    cx2 = min(pw, int(x2 + pad))
    cy2 = min(ph, int(y2 + pad))
    crop = page_image.crop((cx1, cy1, cx2, cy2))
    img_b64 = _png_b64(crop)

    prompt = (
        "This image shows a single Chinese character from a Kunqu opera lyric, "
        "handwritten in kaishu calligraphy on a Lilu Qupu page.\n\n"
        "Output rules:\n"
        "- Output ONLY the single Chinese character that you see.\n"
        "- No Pinyin, no English, no commentary, no quotes, no punctuation.\n"
        "- Use traditional characters (do not simplify).\n"
        "- If you genuinely cannot read it, output 〇 (U+3007).\n"
    )

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            text = _vlm_chat(
                client,
                model=model,
                prompt=prompt,
                image_b64_png=img_b64,
                max_tokens=16,
            )
            chars = _parse_vlm_chars(text, expected_n=1)
            return chars[0]
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt == max_retries:
                raise
    raise RuntimeError(f"unreachable; last error: {last_error}")


def vlm_fill_lyrics_per_box(
    image_path: Union[str, Path],
    lyric_boxes: list[dict],
    *,
    model: str = _DEFAULT_VLM_MODEL,
    api_key: Optional[str] = None,
    pad: int = 12,
    verbose: bool = True,
) -> list[dict]:
    """
    Per-box VLM recognition. Sends each lyric bbox individually to the VLM,
    so non-lyric text in the column gap (stage directions, attributions, etc.)
    cannot leak into the recognized output.

    Trade-off: N API calls instead of 1-per-column. Roughly 5-10x slower than
    `vlm_fill_lyrics`, but eliminates count-mismatch and column-leak issues.

    Each input box should be a Box dict (as returned by LyricDetector.detect).
    Returns a new list of boxes with `label` set to the recognized character.
    """
    from PIL import Image

    client = _make_vlm_client(api_key)
    page_image = Image.open(image_path).convert("RGB")
    out: list[dict] = []

    # Sort by column then top-to-bottom for nicer verbose output. We preserve
    # this order in the returned list, which matches what callers tend to want.
    def _sort_key(b):
        cx = 0.5 * (b["bbox"][0] + b["bbox"][2])
        cy = 0.5 * (b["bbox"][1] + b["bbox"][3])
        # Right-to-left columns (Kunqu reading order), top-to-bottom within col.
        return (-cx, cy)

    sorted_boxes = sorted(lyric_boxes, key=_sort_key)
    n = len(sorted_boxes)

    if verbose:
        print(f"[vlm/per_box] recognizing {n} boxes one at a time")

    for i, b in enumerate(sorted_boxes, 1):
        ch = _vlm_recognize_box(
            page_image, b["bbox"],
            client=client, model=model, pad=pad,
        )
        new_box = dict(b)
        new_box["label"] = ch
        new_box["kind"] = "lyric"
        new_box["score"] = 1.0 if ch != "〇" else 0.0
        out.append(new_box)
        if verbose:
            print(f"[vlm/per_box] {i:3d}/{n}: {ch}")

    return out
