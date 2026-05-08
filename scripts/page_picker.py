"""
Suggest which unlabeled pages to label next, prioritizing pages where
the current lyric detector is most uncertain (= where new labels help most).

Usage:
    PYTHONPATH=. python3 scripts/page_picker.py \
        --pool "LGRC2024 dataset/datasets/images/train" "LGRC2024 dataset/datasets/images/val" \
        --already-labeled lyric_dataset/images/train lyric_dataset/images/val \
        --weights checkpoints/lyric_yolov8m_best.pt \
        --num-hard 15 \
        --num-random 15 \
        --out picked_pages.txt

How "hardness" is scored:
- Run the lyric YOLO at conf=0.1 on each page (low threshold so we see uncertain dets).
- Count detections in the "uncertainty band" 0.1 <= conf <= 0.5.
- More uncertain detections = the model is confused = labeling here helps more.

Pages with zero high-confidence detections (i.e. no lyric content) are EXCLUDED
from both hard and random picks — labeling pages with nothing to label is wasted effort.
"""
from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path

from ultralytics import YOLO


def find_originals(folder: Path) -> list[Path]:
    """Originals = .jpg files whose stem is a single integer (no underscore)."""
    out = []
    for p in sorted(folder.glob("*.jpg")):
        if p.stem.isdigit():
            out.append(p)
    return out


def score_page(model: YOLO, image_path: Path,
               low: float = 0.1, high: float = 0.5,
               min_conf_for_relevance: float = 0.4) -> tuple[int, int]:
    """
    Return (hardness, num_confident_dets).
    - hardness: detections with conf in [low, high] — the uncertainty band.
    - num_confident_dets: detections with conf >= min_conf_for_relevance.
                          If 0, the page probably has no lyric content at all
                          (e.g. pure gongche page, cover, blank).
    """
    res = model(str(image_path), conf=low, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return 0, 0
    confs = res.boxes.conf.cpu().numpy()
    hardness = int(((confs >= low) & (confs <= high)).sum())
    confident = int((confs >= min_conf_for_relevance).sum())
    return hardness, confident


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, nargs="+", required=True,
                    help="Directories with candidate pages (originals only)")
    ap.add_argument("--already-labeled", type=Path, nargs="+", required=True,
                    help="Image directories whose pages already have YOLO labels")
    ap.add_argument("--labels-root", type=Path, default=Path("lyric_dataset/labels"),
                    help="Root where labels/{train,val}/<stem>.txt live")
    ap.add_argument("--weights", type=Path, required=True,
                    help="Current lyric YOLO weights")
    ap.add_argument("--num-hard", type=int, default=15)
    ap.add_argument("--num-random", type=int, default=15)
    ap.add_argument("--min-confident-dets", type=int, default=3,
                    help="Skip pages with fewer than this many high-conf lyric detections "
                         "(probably means no lyric content)")
    ap.add_argument("--out", type=Path, default=Path("picked_pages.txt"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Build candidate pool from --pool dirs.
    pool: list[Path] = []
    seen_stems: set[str] = set()
    for d in args.pool:
        if not d.exists():
            print(f"[picker] WARNING: pool dir doesn't exist: {d}")
            continue
        for p in find_originals(d):
            if p.stem not in seen_stems:
                pool.append(p)
                seen_stems.add(p.stem)
    print(f"[picker] {len(pool)} unique originals in pool")

    # Find labeled stems by checking labels/{train,val}/<stem>.txt
    labels_train = args.labels_root / "train"
    labels_val = args.labels_root / "val"
    labeled_stems: set[str] = set()
    for ld in (labels_train, labels_val):
        if ld.exists():
            for txt in ld.glob("*.txt"):
                if txt.stem.isdigit():
                    labeled_stems.add(txt.stem)
    print(f"[picker] {len(labeled_stems)} pages already labeled: "
          f"{sorted(labeled_stems, key=int)}")

    candidates = [p for p in pool if p.stem not in labeled_stems]
    print(f"[picker] {len(candidates)} unlabeled candidates")

    if not candidates:
        print("[picker] nothing to pick — pool exhausted.")
        sys.exit(0)

    # Score everything in the pool, then filter out pages with no lyric content.
    print(f"[picker] scoring with {args.weights} (this can take a while)...")
    model = YOLO(str(args.weights))
    scored = []
    for i, p in enumerate(candidates):
        hardness, confident = score_page(model, p)
        scored.append((p, hardness, confident))
        if (i + 1) % 10 == 0 or i == len(candidates) - 1:
            print(f"[picker]   scored {i+1}/{len(candidates)}")

    # Filter: must have at least N confident detections (otherwise no lyrics).
    relevant = [(p, h, c) for p, h, c in scored
                if c >= args.min_confident_dets]
    skipped = len(scored) - len(relevant)
    print(f"[picker] {len(relevant)} pages with lyric content "
          f"(skipped {skipped} pages with <{args.min_confident_dets} confident dets)")

    if not relevant:
        print("[picker] no pages with lyric content found — try lowering --min-confident-dets.")
        sys.exit(1)

    # Sort by hardness desc.
    relevant.sort(key=lambda x: -x[1])

    n_total_wanted = args.num_hard + args.num_random
    if len(relevant) <= n_total_wanted:
        print(f"[picker] only {len(relevant)} relevant candidates — picking all of them.")
        picks = [p for p, _, _ in relevant]
        print()
        print("[picker] All picks (sorted by hardness):")
        for p, h, c in relevant:
            print(f"  {p.name:>12}  hardness={h:>3}  confident_dets={c:>3}")
    else:
        hard = relevant[:args.num_hard]
        rest = relevant[args.num_hard:]
        rng = random.Random(args.seed)
        random_picks = rng.sample(rest, min(args.num_random, len(rest)))
        picks = [p for p, _, _ in hard] + [p for p, _, _ in random_picks]

        print()
        print("=" * 60)
        print(f"[picker] HARD picks (model is uncertain — labels help most):")
        for p, h, c in hard:
            print(f"  {p.name:>12}  hardness={h:>3}  confident_dets={c:>3}")
        print()
        print(f"[picker] RANDOM picks (variety, all have lyric content):")
        for p, h, c in random_picks:
            print(f"  {p.name:>12}  hardness={h:>3}  confident_dets={c:>3}")

    args.out.write_text("\n".join(str(p.absolute()) for p in picks) + "\n")
    print()
    print(f"[picker] wrote {len(picks)} picks to {args.out}")
    print(f"[picker] next: python3 scripts/stage_labeling_batch.py {args.out}")


if __name__ == "__main__":
    main()
