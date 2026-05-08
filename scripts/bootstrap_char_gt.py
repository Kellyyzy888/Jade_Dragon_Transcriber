#!/usr/bin/env python3
"""
Bootstrap character ground-truth files for val pages.

For each page in --pages:
  1. Runs YOLO+VLM (per-box) at conf=0.55 to get predicted (bbox, char) pairs.
  2. Writes the result to demo_lyrics/<page>.json in the canonical format.
  3. Prints a side-by-side "review checklist" showing every char.

You then open each demo_lyrics/<page>.json next to the page image and
hand-correct any wrong chars. Qwen is typically ~95% per-box accurate, so
expect to correct 1-3 chars per page.

After hand-correction, demo_lyrics/<page>.json becomes the character GT
for that page, usable with --eval-against in demo_dual_yolo.py.

Usage:
    PYTHONPATH=. python3 scripts/bootstrap_char_gt.py --pages 172 186 107
"""

from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO_LYRICS = ROOT / "demo_lyrics"
DUAL_YOLO_OUT = ROOT / "outputs" / "dual_yolo"


def run_one_page(page_id: str, conf: float = 0.55) -> Path | None:
    """Run the full YOLO+VLM pipeline on a single page. Returns the lyrics.json path."""
    print(f"\n{'='*70}")
    print(f"Page {page_id}: running YOLO+VLM at conf={conf}")
    print(f"{'='*70}")

    cmd = [
        sys.executable, "scripts/demo_dual_yolo.py",
        "--page", page_id,
        "--lyric-conf", str(conf),
        "--vlm",
    ]
    env_pythonpath = str(ROOT)
    result = subprocess.run(
        cmd, cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": env_pythonpath},
    )
    if result.returncode != 0:
        print(f"[!!] Pipeline failed for {page_id}")
        return None

    out_path = DUAL_YOLO_OUT / page_id / f"{page_id}.lyrics.json"
    if not out_path.exists():
        print(f"[!!] Expected output not found: {out_path}")
        return None
    return out_path


def stage_for_correction(pred_path: Path, page_id: str) -> Path:
    """
    Copy the prediction lyrics.json into demo_lyrics/<page>.json, updating
    the _source field to mark it as a draft for hand-correction.
    """
    DEMO_LYRICS.mkdir(parents=True, exist_ok=True)
    dst = DEMO_LYRICS / f"{page_id}.json"

    # Don't clobber existing GT
    if dst.exists():
        print(f"[skip-stage] {dst} already exists; not overwriting")
        return dst

    data = json.loads(pred_path.read_text())
    data["_source"] = (
        f"DRAFT — bootstrapped from YOLO+VLM (Qwen3-VL) on val/{page_id}.jpg. "
        f"HAND-CORRECT EACH CHAR before using as ground truth."
    )

    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"[stage] wrote draft to {dst}")
    return dst


def print_review_checklist(staged_path: Path, page_id: str) -> None:
    """Print a per-char review summary so the user can quickly audit."""
    data = json.loads(staged_path.read_text())
    lyrics = data["lyrics"]

    # Group by approximate column (x-center bin)
    by_col: dict[int, list[tuple[int, dict]]] = {}
    for i, item in enumerate(lyrics):
        cx = 0.5 * (item["bbox"][0] + item["bbox"][2])
        col = int(cx // 100) * 100
        by_col.setdefault(col, []).append((i, item))

    print(f"\n[review] Page {page_id}: {len(lyrics)} chars across "
          f"{len(by_col)} columns (sorted by x-center bin)")

    # Print right-to-left (Kunqu reading order)
    for col_x in sorted(by_col.keys(), reverse=True):
        items = sorted(by_col[col_x], key=lambda t: t[1]["bbox"][1])
        chars = "".join(t[1]["char"] for t in items)
        print(f"  col x≈{col_x:>5}: {chars}  ({len(items)} chars)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", nargs="+", required=True,
                    help="LGRC2024 page IDs to bootstrap (e.g. 172 186 107)")
    ap.add_argument("--conf", type=float, default=0.55,
                    help="YOLO lyric-detector confidence threshold")
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="Skip the YOLO+VLM run; use cached lyrics.json from "
                         "outputs/dual_yolo/<page>/<page>.lyrics.json")
    args = ap.parse_args()

    print(f"\nBootstrapping char GT for {len(args.pages)} pages: {args.pages}")
    print(f"Strategy: per-box VLM at conf={args.conf}")
    if not args.skip_pipeline:
        print(f"Estimated cost: ~$0.03/page in OpenRouter Qwen3-VL calls")
        print(f"Estimated time: ~90s/page")

    summary = []
    t_start = time.time()

    for page_id in args.pages:
        if args.skip_pipeline:
            pred_path = DUAL_YOLO_OUT / page_id / f"{page_id}.lyrics.json"
            if not pred_path.exists():
                print(f"[!!] No cached output for page {page_id}: {pred_path}")
                continue
        else:
            pred_path = run_one_page(page_id, conf=args.conf)
            if pred_path is None:
                continue

        staged = stage_for_correction(pred_path, page_id)
        print_review_checklist(staged, page_id)
        summary.append((page_id, staged))

    print(f"\n{'='*70}")
    print(f"Done. Processed {len(summary)} pages in {time.time() - t_start:.0f}s")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"  1. Open each draft side-by-side with the page image. Page images:")
    for page_id, _ in summary:
        candidates = [
            ROOT / "lyric_dataset" / "images" / "val" / f"{page_id}.jpg",
            ROOT / "LGRC2024 dataset" / "datasets" / "images" / "val" / f"{page_id}.jpg",
            ROOT / "LGRC2024 dataset" / "datasets" / "images" / "train" / f"{page_id}.jpg",
        ]
        img = next((c for c in candidates if c.exists()), None)
        print(f"     page {page_id}: {img}")
    print(f"\n  2. Hand-correct each char in:")
    for page_id, dst in summary:
        print(f"     {dst}")
    print(f"\n  3. Update _source field to remove 'DRAFT' marker once you trust it.")
    print(f"\n  4. Run final eval:")
    pages_str = " ".join(p for p, _ in summary)
    print(f"     for p in {pages_str}; do")
    print(f"       PYTHONPATH=. python3 scripts/demo_dual_yolo.py \\")
    print(f"         --page \"$p\" --lyric-conf {args.conf} --vlm \\")
    print(f"         --eval-against \"demo_lyrics/$p.json\"")
    print(f"     done")


if __name__ == "__main__":
    main()
