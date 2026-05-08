#!/usr/bin/env python3
"""
Dual-YOLO end-to-end demo: detect pitches AND lyric bounding boxes with two
trained YOLO checkpoints, align them column-by-column, emit MusicXML + MIDI.

By default, characters are placeholders ("L1", "L2", ...). Pass --vlm to call
the VLM (Qwen3-VL via OpenRouter, or Anthropic Claude) and fill in real chars.

Usage:
    # Detect-only, fast: placeholder labels
    python scripts/demo_dual_yolo.py --page 137

    # Full pipeline: YOLO detection → per-box VLM recognition → MusicXML
    python scripts/demo_dual_yolo.py --page 137 --lyric-conf 0.60 --vlm

    # Score against ground truth
    python scripts/demo_dual_yolo.py --page 137 --lyric-conf 0.60 --vlm \
        --eval-against demo_lyrics/137.json

    # Use the older column-strip strategy (faster but leaks non-lyric text)
    python scripts/demo_dual_yolo.py --page 137 --lyric-conf 0.60 --vlm \
        --vlm-strategy column
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

from jade_dragon.alignment import align_page, cluster_columns
from jade_dragon.audio_render import musicxml_to_midi
from jade_dragon.lyric_ocr import (
    LyricDetector, vlm_fill_lyrics, vlm_fill_lyrics_per_box,
)
from jade_dragon.musicxml_emit import write_musicxml
from jade_dragon.pitch_detector import PitchDetector, infer_key_class

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "LGRC2024 dataset" / "datasets"
DEFAULT_OUT = ROOT / "outputs" / "dual_yolo"
DEFAULT_PITCH_W = ROOT / "checkpoints" / "lgrc2024_yolov8m_best.pt"
DEFAULT_LYRIC_W = ROOT / "checkpoints" / "lyric_yolov8m_best.pt"
DEFAULT_VLM_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"


def find_lgrc_image(page_id: str) -> Path:
    for split in ("train", "val"):
        img = DATASET / "images" / split / f"{page_id}.jpg"
        if img.exists():
            return img
    # Also check lyric_dataset (where the val val subset images may live).
    for split in ("train", "val"):
        img = ROOT / "lyric_dataset" / "images" / split / f"{page_id}.jpg"
        if img.exists():
            return img
    raise FileNotFoundError(f"page {page_id} not in any known location")


def assign_columns(lyric_boxes: list[dict], column_eps_factor: float = 1.5) -> list[dict]:
    """
    Cluster lyric boxes into columns and tag each with its column index.
    Required by `vlm_fill_lyrics` (column-strip strategy).
    """
    columns = cluster_columns(lyric_boxes, column_eps_factor=column_eps_factor)
    out: list[dict] = []
    for col_idx, col in enumerate(columns):
        for b in col:
            b2 = dict(b)
            b2["_column"] = col_idx
            out.append(b2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_argument_group("page source")
    src.add_argument("--page", help="LGRC2024 page id (e.g., 137)")
    src.add_argument("--image", help="Path to page image (overrides --page)")
    ap.add_argument("--pitch-weights", default=str(DEFAULT_PITCH_W))
    ap.add_argument("--lyric-weights", default=str(DEFAULT_LYRIC_W))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--pitch-conf", type=float, default=0.25)
    ap.add_argument("--lyric-conf", type=float, default=0.25)
    ap.add_argument("--tempo-bpm", type=int, default=80)
    ap.add_argument("--work-title", default="Gongche transcription (dual-YOLO)")
    # VLM character recognition
    ap.add_argument("--vlm", action="store_true",
                    help="Run VLM character recognition on detected lyric boxes")
    ap.add_argument("--vlm-strategy", choices=["per_box", "column"],
                    default="per_box",
                    help="per_box: one VLM call per bbox (slower, more accurate, "
                         "no leak from non-lyric text). column: one call per "
                         "column-strip (faster, can leak adjacent text).")
    ap.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL,
                    help=f"VLM model id (default: {DEFAULT_VLM_MODEL})")
    ap.add_argument("--api-key", default=None,
                    help="Override OPENROUTER_API_KEY / ANTHROPIC_API_KEY")
    ap.add_argument("--column-eps-factor", type=float, default=1.5)
    ap.add_argument("--eval-against", default=None,
                    help="Reference demo_lyrics JSON; report char accuracy "
                         "(requires --vlm)")
    args = ap.parse_args()

    if not args.page and not args.image:
        ap.error("provide --page or --image")
    if args.eval_against and not args.vlm:
        ap.error("--eval-against requires --vlm (placeholder labels can't be scored)")

    img_path = Path(args.image) if args.image else find_lgrc_image(args.page)
    page_id = img_path.stem
    out_dir = Path(args.out_dir) if args.out_dir else (DEFAULT_OUT / page_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    print(f"[page] {img_path.name}  ({W}×{H})")

    # ---- Stage 1: pitches ----------------------------------------------------
    t0 = time.time()
    print(f"[pitch] loading {args.pitch_weights}")
    pitch_det = PitchDetector(args.pitch_weights, conf=args.pitch_conf)
    all_p_boxes = pitch_det.detect(str(img_path))
    pitches = [b for b in all_p_boxes if b["kind"] == "pitch"]
    keys = [b for b in all_p_boxes if b["kind"] == "key"]
    key_class = infer_key_class(all_p_boxes)
    print(f"[pitch] {len(pitches)} pitches, {len(keys)} key-sig; "
          f"key={key_class}  ({time.time() - t0:.1f}s)")

    # ---- Stage 2: lyric bboxes -----------------------------------------------
    t0 = time.time()
    print(f"[lyric] loading {args.lyric_weights}")
    lyric_det = LyricDetector(args.lyric_weights, conf=args.lyric_conf)
    raw_lyrics = lyric_det.detect(str(img_path))
    print(f"[lyric] {len(raw_lyrics)} lyric boxes  ({time.time() - t0:.1f}s)")

    # ---- Stage 2b: VLM character recognition (optional) ----------------------
    if args.vlm:
        t0 = time.time()
        if args.vlm_strategy == "per_box":
            print(f"[vlm] strategy=per_box, model={args.vlm_model}, "
                  f"{len(raw_lyrics)} calls")
            recognized = vlm_fill_lyrics_per_box(
                str(img_path), raw_lyrics,
                model=args.vlm_model,
                api_key=args.api_key,
                verbose=True,
            )
        else:
            boxes_with_col = assign_columns(raw_lyrics, args.column_eps_factor)
            print(f"[vlm] strategy=column, model={args.vlm_model}, "
                  f"{len({b['_column'] for b in boxes_with_col})} columns")
            recognized = vlm_fill_lyrics(
                str(img_path), boxes_with_col,
                model=args.vlm_model,
                api_key=args.api_key,
                verbose=True,
            )
        lyric_boxes = [{**b, "label": b.get("label", "？")} for b in recognized]
        print(f"[vlm] {len(lyric_boxes)} chars in {time.time() - t0:.1f}s")
    else:
        lyric_boxes = []
        for i, b in enumerate(raw_lyrics):
            b = dict(b)
            b["label"] = f"L{i + 1}"
            lyric_boxes.append(b)

    # ---- Stage 3: align + MusicXML + MIDI ------------------------------------
    aligned = align_page(lyric_boxes, pitches)
    n_with_pitches = sum(1 for e in aligned if e["pitches"])
    n_assigned = sum(len(e["pitches"]) for e in aligned)
    print(f"[align] {len(aligned)} lyric entries, {n_with_pitches} with pitches, "
          f"{n_assigned}/{len(pitches)} pitches assigned")

    xml_path = out_dir / f"{page_id}.musicxml"
    midi_path = out_dir / f"{page_id}.mid"
    write_musicxml(
        aligned, str(xml_path),
        key_class=key_class,
        tempo_bpm=args.tempo_bpm,
        work_title=args.work_title,
    )
    musicxml_to_midi(str(xml_path), str(midi_path))
    print(f"[out] {xml_path}")
    print(f"[out] {midi_path}")

    # ---- Stage 4: overlay PNG ------------------------------------------------
    overlay = img.copy()
    d = ImageDraw.Draw(overlay)
    for p in pitches:
        d.rectangle(p["bbox"], outline="red", width=2)
    for l in lyric_boxes:
        d.rectangle(l["bbox"], outline="blue", width=4)
    overlay_path = out_dir / f"{page_id}.overlay.png"
    overlay.save(overlay_path)
    print(f"[out] {overlay_path}")

    # ---- Stage 5: alignment dump ---------------------------------------------
    align_dump = []
    for e in aligned:
        align_dump.append({
            "column": e["column"],
            "lyric_label": (e["lyric"] or {}).get("label"),
            "lyric_bbox": (e["lyric"] or {}).get("bbox"),
            "pitches": [
                {"label": p["label"], "bbox": p["bbox"]}
                for p in e["pitches"]
            ],
        })
    align_path = out_dir / f"{page_id}.alignment.json"
    with open(align_path, "w", encoding="utf-8") as f:
        json.dump(align_dump, f, ensure_ascii=False, indent=2)
    print(f"[out] {align_path}")

    # ---- Stage 6: demo_lyrics-style JSON -------------------------------------
    if args.vlm:
        lyrics_json_path = out_dir / f"{page_id}.lyrics.json"
        payload = {
            "page_w": W,
            "page_h": H,
            "_source": (
                f"YOLO+VLM via {args.vlm_model} "
                f"({args.vlm_strategy}) (demo_dual_yolo.py)"
            ),
            "lyrics": [
                {"bbox": list(b["bbox"]), "char": b["label"]}
                for b in lyric_boxes
            ],
        }
        with open(lyrics_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[out] {lyrics_json_path}")

    # ---- Stage 7: evaluation -------------------------------------------------
    if args.eval_against:
        from scripts.recognize_lyrics_vlm import char_accuracy, load_reference_lyrics
        ref = load_reference_lyrics(Path(args.eval_against))
        pred = [
            {"bbox": tuple(b["bbox"]), "label": b["label"]}
            for b in lyric_boxes
        ]
        report = char_accuracy(pred, ref)
        print()
        print("=== Character accuracy ===")
        print(f"  predicted: {report['n_pred']}, reference: {report['n_ref']}")
        print(f"  matched:   {report['n_matched']}")
        print(f"  correct:   {report['n_correct']}")
        if report['n_ref']:
            print(f"  accuracy:  {report['char_accuracy']:.3f}  "
                  f"(correct / reference)")
        if report['n_pred']:
            print(f"  precision: {report['char_precision']:.3f}  "
                  f"(correct / predicted)")
        if report['misread']:
            print(f"  misreads ({len(report['misread'])}):")
            for pred_ch, ref_ch in report['misread'][:20]:
                print(f"    {pred_ch}  →  {ref_ch}  (predicted → expected)")
            if len(report['misread']) > 20:
                print(f"    ... and {len(report['misread']) - 20} more")
        if report.get('unmatched_pred'):
            print(f"  unmatched_pred ({len(report['unmatched_pred'])}): "
                  f"{''.join(report['unmatched_pred'][:30])}")
        if report.get('unmatched_ref'):
            print(f"  unmatched_ref ({len(report['unmatched_ref'])}): "
                  f"{''.join(report['unmatched_ref'][:30])}")
        if report.get('per_col'):
            print(f"  per-column:")
            for c in report['per_col']:
                print(f"    col {c['col']:2d}: ref={c['ref']}, pred={c['pred']}, "
                      f"correct={c['correct']}, misread={c['misread']}")

    note = f" with VLM ({args.vlm_strategy})" if args.vlm else " (placeholder labels)"
    print(f"\nDone{note}. Open the .musicxml in MuseScore or Verovio.")


if __name__ == "__main__":
    main()
