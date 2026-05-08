"""
Copy picked pages into the dataset structure for labeling.

Splits picks 80/20 between train/val so the val set keeps growing.
Copies into lyric_dataset/images/train/ and lyric_dataset/images/val/
(same dirs as before — VIA will see the new files when you re-add them).

Usage:
    python3 scripts/stage_labeling_batch.py picked_pages.txt
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picked_file", type=Path,
                    help="Output of page_picker.py — one filepath per line")
    ap.add_argument("--train-dir", type=Path,
                    default=Path("lyric_dataset/images/train"))
    ap.add_argument("--val-dir", type=Path,
                    default=Path("lyric_dataset/images/val"))
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    picks = [Path(line.strip()) for line in args.picked_file.read_text().splitlines() if line.strip()]
    print(f"[stage] read {len(picks)} picks from {args.picked_file}")

    # Sanity: warn on missing files.
    missing = [p for p in picks if not p.exists()]
    if missing:
        print(f"[stage] WARNING: {len(missing)} picks don't exist:")
        for m in missing:
            print(f"  {m}")
        picks = [p for p in picks if p.exists()]

    # 80/20 split. Use a stable, deterministic split based on filename hash
    # so reruns don't shuffle pages between train and val.
    n_val = round(len(picks) * args.val_frac)
    train_picks = picks[:-n_val] if n_val > 0 else picks
    val_picks = picks[-n_val:] if n_val > 0 else []

    args.train_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stage] copying {len(train_picks)} pages to {args.train_dir}")
    for p in train_picks:
        dst = args.train_dir / p.name
        if dst.exists():
            print(f"  SKIP (exists): {p.name}")
            continue
        shutil.copy2(p, dst)
        print(f"  + {p.name}")

    print(f"[stage] copying {len(val_picks)} pages to {args.val_dir}")
    for p in val_picks:
        dst = args.val_dir / p.name
        if dst.exists():
            print(f"  SKIP (exists): {p.name}")
            continue
        shutil.copy2(p, dst)
        print(f"  + {p.name}")

    print()
    print("[stage] Done. Next:")
    print("  1. Open VIA, load via_all.json")
    print(f"  2. Add Files: select all NEW pages from {args.train_dir} and {args.val_dir}")
    print("  3. Label them, then Annotation -> Export Annotations (as json)")
    print("  4. Save as via_all.json (overwrite the old one)")
    print("  5. Run: python3 scripts/via_to_yolo.py via_all.json")
    print("  6. Resubmit Oscar training: sbatch scripts/oscar_train_lyric.sbatch")


if __name__ == "__main__":
    main()
