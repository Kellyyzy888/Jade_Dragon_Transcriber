#!/usr/bin/env python3
"""
End-to-end val sweep: run the YOLO+VLM pipeline on every val page, score the
ones with char ground truth, and produce a summary table for the report.

For pages with demo_lyrics/<page>.json: full char-accuracy eval.
For pages without GT: pipeline-completeness check (does it run cleanly?).

Usage:
    PYTHONPATH=. python3 scripts/val_sweep.py
    PYTHONPATH=. python3 scripts/val_sweep.py --conf 0.55 --skip-pipeline   # use cached
    PYTHONPATH=. python3 scripts/val_sweep.py --pages 137 172 107  # subset
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VAL_IMG_DIR_PRIMARY = ROOT / "lyric_dataset" / "images" / "val"
DUAL_OUT = ROOT / "outputs" / "dual_yolo"
DEMO = ROOT / "demo_lyrics"


def list_val_pages() -> list[str]:
    if not VAL_IMG_DIR_PRIMARY.exists():
        return []
    pages = []
    for p in sorted(VAL_IMG_DIR_PRIMARY.glob("*.jpg")):
        stem = p.stem
        # Skip diagnostic overlays like "137_lyric_boxes"
        if "_" in stem:
            continue
        pages.append(stem)
    return pages


def has_char_gt(page: str) -> bool:
    return (DEMO / f"{page}.json").exists()


def run_one_page(page: str, conf: float) -> bool:
    """Invoke demo_dual_yolo.py for a single page. Returns success flag."""
    import subprocess
    cmd = [
        sys.executable, "scripts/demo_dual_yolo.py",
        "--page", page,
        "--lyric-conf", str(conf),
        "--vlm",
    ]
    if has_char_gt(page):
        cmd += ["--eval-against", str(DEMO / f"{page}.json")]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    print(f"\n  → running on page {page}...")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    dt = time.time() - t0
    print(f"  → page {page} done in {dt:.0f}s (rc={result.returncode})")
    return result.returncode == 0


def score_one_page(page: str) -> dict:
    """Score the cached lyrics.json for `page` against demo_lyrics GT."""
    from scripts.recognize_lyrics_vlm import char_accuracy, load_reference_lyrics

    pred_path = DUAL_OUT / page / f"{page}.lyrics.json"
    gt_path = DEMO / f"{page}.json"
    if not (pred_path.exists() and gt_path.exists()):
        return {}

    pred_data = json.load(open(pred_path))
    pred = [
        {"bbox": tuple(l["bbox"]), "label": l["char"]}
        for l in pred_data["lyrics"]
    ]
    ref = load_reference_lyrics(gt_path)
    return char_accuracy(pred, ref)


def summarize(pages: list[str], scored: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print("VAL SWEEP SUMMARY")
    print("=" * 78)

    print("\n[A] Pipeline completeness (all val pages)")
    print("-" * 78)
    print(f"{'Page':>6}  {'YOLO boxes':>10}  {'VLM chars':>10}  {'GT?':>4}")
    for page in pages:
        lyrics_path = DUAL_OUT / page / f"{page}.lyrics.json"
        if lyrics_path.exists():
            d = json.load(open(lyrics_path))
            n = len(d["lyrics"])
            print(f"{page:>6}  {n:>10}  {n:>10}  {'✓' if has_char_gt(page) else '–':>4}")
        else:
            print(f"{page:>6}  {'(missing)':>10}  {'(missing)':>10}  "
                  f"{'✓' if has_char_gt(page) else '–':>4}")

    if scored:
        print("\n[B] Character accuracy on GT-bearing pages")
        print("-" * 78)
        print(f"{'Page':>6}  {'pred':>5}  {'ref':>5}  {'corr':>5}  "
              f"{'mis':>4}  {'extra':>5}  {'missed':>6}  {'acc':>6}  {'prec':>6}")
        weighted_correct = 0
        weighted_ref = 0
        weighted_pred = 0
        for page, r in scored.items():
            n_unmatched_pred = len(r['unmatched_pred'])
            n_unmatched_ref = len(r['unmatched_ref'])
            print(f"{page:>6}  {r['n_pred']:>5}  {r['n_ref']:>5}  "
                  f"{r['n_correct']:>5}  {len(r['misread']):>4}  "
                  f"{n_unmatched_pred:>5}  {n_unmatched_ref:>6}  "
                  f"{r['char_accuracy']:>6.3f}  {r['char_precision']:>6.3f}")
            weighted_correct += r['n_correct']
            weighted_ref += r['n_ref']
            weighted_pred += r['n_pred']
        print("-" * 78)
        if weighted_ref:
            agg_acc = weighted_correct / weighted_ref
            agg_prec = weighted_correct / weighted_pred if weighted_pred else 0
            print(f"{'TOTAL':>6}  {weighted_pred:>5}  {weighted_ref:>5}  "
                  f"{weighted_correct:>5}  {'-':>4}  {'-':>5}  {'-':>6}  "
                  f"{agg_acc:>6.3f}  {agg_prec:>6.3f}")

        print("\n[C] Misreads per GT page (Qwen recognition errors)")
        print("-" * 78)
        for page, r in scored.items():
            if r['misread']:
                misreads_str = ", ".join(f"{p}→{q}" for p, q in r['misread'])
                print(f"  {page}: {misreads_str}")
            else:
                print(f"  {page}: none")

        print("\n[D] Detection misses (chars in GT but not detected by YOLO)")
        print("-" * 78)
        for page, r in scored.items():
            if r['unmatched_ref']:
                print(f"  {page}: {''.join(r['unmatched_ref'])}")
            else:
                print(f"  {page}: none")

        print("\n[E] Detection extras (boxes detected but not in GT)")
        print("-" * 78)
        for page, r in scored.items():
            if r['unmatched_pred']:
                print(f"  {page}: {''.join(r['unmatched_pred'])}")
            else:
                print(f"  {page}: none")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", nargs="*", default=None,
                    help="Specific pages to run; default = all val pages")
    ap.add_argument("--conf", type=float, default=0.55,
                    help="YOLO lyric-detector confidence threshold")
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="Skip running the pipeline; use cached outputs")
    ap.add_argument("--out", default="outputs/val_sweep_summary.json",
                    help="Where to write the summary JSON")
    args = ap.parse_args()

    pages = args.pages if args.pages else list_val_pages()
    if not pages:
        print("[!!] no val pages found")
        return

    print(f"[sweep] {len(pages)} pages: {' '.join(pages)}")
    print(f"[sweep] conf={args.conf}, skip_pipeline={args.skip_pipeline}")
    gt_pages = [p for p in pages if has_char_gt(p)]
    print(f"[sweep] {len(gt_pages)} have char GT: {' '.join(gt_pages)}")

    t_start = time.time()

    if not args.skip_pipeline:
        for i, page in enumerate(pages, 1):
            print(f"\n[{i}/{len(pages)}] page {page}")
            run_one_page(page, args.conf)

    # Score all GT-bearing pages from cached output
    scored: dict[str, dict] = {}
    for page in gt_pages:
        r = score_one_page(page)
        if r:
            scored[page] = r

    summarize(pages, scored)

    # Persist a JSON summary
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "config": {
            "conf": args.conf,
            "pages": pages,
            "gt_pages": gt_pages,
        },
        "results": {
            page: {
                "n_pred": r["n_pred"],
                "n_ref": r["n_ref"],
                "n_correct": r["n_correct"],
                "char_accuracy": r["char_accuracy"],
                "char_precision": r["char_precision"],
                "misreads": [list(m) for m in r["misread"]],
                "unmatched_pred": r["unmatched_pred"],
                "unmatched_ref": r["unmatched_ref"],
            }
            for page, r in scored.items()
        },
        "elapsed_seconds": time.time() - t_start,
    }
    with open(out_path, "w") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"\n[sweep] wrote {out_path}")
    print(f"[sweep] total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
