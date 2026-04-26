#!/usr/bin/env python3
"""
Day-3 MVP demo: gongche page → MusicXML + MIDI (+ optional WAV).

No model training required. Uses LGRC2024 ground-truth pitch labels for one
page + a hand-edited lyric JSON. Proves the alignment + MusicXML + audio
path end-to-end, before any ML work.

Usage:
    python scripts/demo_one_page.py --page 100
    python scripts/demo_one_page.py --page 100 --wav      # also render WAV (needs fluidsynth)
    python scripts/demo_one_page.py --image path/to/img.jpg --label path/to/label.txt --lyrics path/to/lyrics.json
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jade_dragon.pipeline import transcribe_with_gt

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "LGRC2024 dataset" / "datasets"
DEMO_LYRICS = ROOT / "demo_lyrics"
OUT_DIR = ROOT / "outputs" / "demo"


def resolve_page(page_id: str) -> tuple[Path, Path, Path]:
    """Find image + label + manual-lyric JSON for a given LGRC2024 page id."""
    img = None
    lbl = None
    for split in ("train", "val"):
        candidate_img = DATASET / "images" / split / f"{page_id}.jpg"
        candidate_lbl = DATASET / "labels" / split / f"{page_id}.txt"
        if candidate_img.exists() and candidate_lbl.exists():
            img, lbl = candidate_img, candidate_lbl
            break
    if img is None:
        raise FileNotFoundError(f"Could not find image+label for page id {page_id}")

    lyrics = DEMO_LYRICS / f"{page_id}.json"
    if not lyrics.exists():
        raise FileNotFoundError(
            f"No manual lyric file at {lyrics}. Create one with the schema in "
            "jade_dragon/lyric_ocr.py:load_manual_lyrics."
        )
    return img, lbl, lyrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", help="LGRC2024 page id (e.g., 100)")
    ap.add_argument("--image", help="Path to image (overrides --page)")
    ap.add_argument("--label", help="Path to YOLO label file (overrides --page)")
    ap.add_argument("--lyrics", help="Path to manual lyric JSON (overrides --page)")
    ap.add_argument("--out", default=str(OUT_DIR), help="Output directory")
    ap.add_argument("--wav", action="store_true", help="Render WAV via fluidsynth")
    ap.add_argument("--soundfont", default=None, help="Path to .sf2 SoundFont (optional)")
    ap.add_argument("--tempo", type=int, default=80)
    ap.add_argument("--title", default="Gongche transcription demo")
    args = ap.parse_args()

    if args.image and args.label and args.lyrics:
        img, lbl, lyrics = Path(args.image), Path(args.label), Path(args.lyrics)
    elif args.page:
        img, lbl, lyrics = resolve_page(args.page)
    else:
        ap.error("Either --page or all of --image/--label/--lyrics must be provided")

    print(f"Image:    {img}")
    print(f"Labels:   {lbl}")
    print(f"Lyrics:   {lyrics}")
    print(f"Output:   {args.out}")
    print()

    result = transcribe_with_gt(
        image_path=img,
        gt_label_path=lbl,
        manual_lyrics_path=lyrics,
        out_dir=args.out,
        work_title=args.title,
        tempo_bpm=args.tempo,
        render_wav=args.wav,
        soundfont_path=args.soundfont,
    )

    print(f"Detected key:     {result['key']}")
    print(f"Lyric chars:      {result['n_lyrics']}")
    print(f"Total notes:      {result['n_pitches']}")
    print(f"MusicXML:         {result['musicxml']}")
    print(f"MIDI:             {result['midi']}")
    if result["wav"]:
        print(f"WAV:              {result['wav']}")

    # Write a small JSON of the alignment for inspection
    alignment_dump = Path(result["musicxml"]).with_suffix(".alignment.json")
    serializable = [
        {
            "lyric": (e["lyric"] or {}).get("label"),
            "pitches": [p["label"] for p in e["pitches"]],
            "column": e["column"],
        }
        for e in result["aligned"]
    ]
    with open(alignment_dump, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"Alignment dump:   {alignment_dump}")


if __name__ == "__main__":
    main()
