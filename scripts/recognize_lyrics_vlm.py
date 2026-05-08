#!/usr/bin/env python3
"""
Bootstrap lyric positions from gongche layout, then fill characters via a VLM.

This is the automated successor to `bootstrap_lyrics.py`'s "edit ？ by hand"
step. Geometry comes from the same column-clustering logic; characters come
from Anthropic Claude's vision model.

Usage:
    # Use LGRC2024 ground-truth labels (no detector inference)
    python scripts/recognize_lyrics_vlm.py --page 137

    # Use the trained YOLO detector on an arbitrary image
    python scripts/recognize_lyrics_vlm.py \
        --image path/to/page.jpg \
        --weights checkpoints/lgrc2024_yolov8m_best.pt \
        --out demo_lyrics/mypage.json

    # Compare to a ground-truth JSON to compute character accuracy
    python scripts/recognize_lyrics_vlm.py --page 137 \
        --eval-against demo_lyrics/137.json

Requires `ANTHROPIC_API_KEY` in your environment, or `--api-key`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

# Reuse the bootstrap geometry — single source of truth for column layout.
from scripts.bootstrap_lyrics import infer_lyric_boxes, find_image_and_label

from jade_dragon.alignment import yolo_label_to_boxes
from jade_dragon.lyric_ocr import vlm_fill_lyrics, vlm_recognize_page
from jade_dragon.pitch_map import CLASS_NAMES

ROOT = Path(__file__).resolve().parent.parent
DEMO_LYRICS = ROOT / "demo_lyrics"


# -- pitch-box sources ---------------------------------------------------------


def pitches_from_gt_label(label_path: Path, page_w: int, page_h: int) -> list[dict]:
    boxes = yolo_label_to_boxes(str(label_path), page_w, page_h, CLASS_NAMES)
    return [b for b in boxes if b["kind"] == "pitch"]


def pitches_from_detector(image_path: Path, weights: Path, conf: float = 0.25) -> list[dict]:
    from jade_dragon.pitch_detector import PitchDetector
    det = PitchDetector(weights, conf=conf)
    boxes = det.detect(str(image_path))
    return [b for b in boxes if b["kind"] == "pitch"]


# -- evaluation against a ground-truth JSON ------------------------------------


def _column_index(box: dict, col_xs: list[float], eps: float) -> int:
    """Pick the closest column-x; -1 if outside eps."""
    cx = 0.5 * (box["bbox"][0] + box["bbox"][2])
    if not col_xs:
        return -1
    dists = [abs(cx - x) for x in col_xs]
    j = min(range(len(col_xs)), key=lambda i: dists[i])
    return j if dists[j] <= eps else -1


def char_accuracy(predicted: list[dict], reference: list[dict]) -> dict:
    """
    Column-aware character accuracy with sequence alignment.

    Why sequence alignment vs simple zip:
      - Zip pairs pred[i] with ref[i] within each column. A single missed
        detection (pred shorter than ref by 1) cascades into apparent misreads
        for every later position.
      - Sequence alignment (via difflib.SequenceMatcher) finds the optimal
        alignment allowing insertions/deletions, so a missed detection shows
        as one deletion instead of cascading.
      - This matches standard OCR evaluation practice (CER, edit distance).

    Strategy:
      1. Cluster predictions and references by x-center into columns.
      2. Sort columns right-to-left.
      3. Drop pred columns that don't fit (smallest count first — likely
         titles, attributions, stray detections).
      4. For each matched column, run sequence alignment between pred and
         ref char sequences (top-to-bottom). Count exact matches.
    """
    from difflib import SequenceMatcher

    if not reference:
        return {
            "n_pred": len(predicted), "n_ref": 0, "n_matched": 0,
            "n_correct": 0, "char_accuracy": 0.0, "char_precision": 0.0,
            "misread": [], "unmatched_pred": [b["label"] for b in predicted],
            "unmatched_ref": [], "per_col": [],
        }

    def cluster_x(boxes: list[dict], eps: float = 80.0) -> list[list[dict]]:
        if not boxes:
            return []
        ranked = sorted(
            boxes,
            key=lambda b: 0.5 * (b["bbox"][0] + b["bbox"][2]),
            reverse=True,
        )
        cols: list[list[dict]] = []
        cur: list[dict] = []
        cur_cx = None
        for b in ranked:
            cx = 0.5 * (b["bbox"][0] + b["bbox"][2])
            if cur_cx is None or abs(cx - cur_cx) <= eps:
                cur.append(b)
                cs = [0.5 * (x["bbox"][0] + x["bbox"][2]) for x in cur]
                cur_cx = sum(cs) / len(cs)
            else:
                cols.append(cur)
                cur = [b]
                cur_cx = cx
        if cur:
            cols.append(cur)
        return cols

    def align_column(pred_chars: list[str], ref_chars: list[str]):
        """
        Return (n_correct, n_matched_pairs, misreads, unmatched_pred, unmatched_ref).
        Uses SequenceMatcher's longest-common-subsequence-based opcodes.
        """
        sm = SequenceMatcher(a=pred_chars, b=ref_chars, autojunk=False)
        correct = 0
        misreads: list[tuple[str, str]] = []
        unmatched_p: list[str] = []
        unmatched_r: list[str] = []
        n_matched = 0

        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                # pred[i1:i2] == ref[j1:j2], char-for-char match
                correct += i2 - i1
                n_matched += i2 - i1
            elif op == "replace":
                # Mismatched substring of equal-or-different lengths
                # Pair up to min length as misreads, treat extras as unmatched.
                p_seg = pred_chars[i1:i2]
                r_seg = ref_chars[j1:j2]
                k = min(len(p_seg), len(r_seg))
                for x in range(k):
                    misreads.append((p_seg[x], r_seg[x]))
                    n_matched += 1
                if len(p_seg) > k:
                    unmatched_p.extend(p_seg[k:])
                if len(r_seg) > k:
                    unmatched_r.extend(r_seg[k:])
            elif op == "delete":
                # pred has extras not in ref
                unmatched_p.extend(pred_chars[i1:i2])
            elif op == "insert":
                # ref has chars pred missed
                unmatched_r.extend(ref_chars[j1:j2])

        return correct, n_matched, misreads, unmatched_p, unmatched_r

    pred_cols_all = cluster_x(predicted)
    ref_cols_all = cluster_x(reference)

    misread: list[tuple[str, str]] = []
    unmatched_pred: list[str] = []
    unmatched_ref: list[str] = []

    # Drop smallest pred cols if we have more pred cols than ref cols
    pred_cols = list(pred_cols_all)
    if len(pred_cols) > len(ref_cols_all):
        sizes = [(i, len(c)) for i, c in enumerate(pred_cols)]
        sorted_by_size = sorted(sizes, key=lambda t: t[1], reverse=True)
        keep_idx = {i for i, _ in sorted_by_size[:len(ref_cols_all)]}
        kept = []
        for i, col in enumerate(pred_cols):
            if i in keep_idx:
                kept.append(col)
            else:
                unmatched_pred.extend(b["label"] for b in col)
        pred_cols = kept

    ref_cols = list(ref_cols_all)
    if len(ref_cols) > len(pred_cols):
        diff = len(ref_cols) - len(pred_cols)
        for col in ref_cols[-diff:]:
            unmatched_ref.extend(b["label"] for b in col)
        ref_cols = ref_cols[:-diff]

    n_pair = min(len(pred_cols), len(ref_cols))
    correct = 0
    n_matched = 0
    per_col_report = []

    for ci in range(n_pair):
        pred_col_sorted = sorted(
            pred_cols[ci],
            key=lambda b: 0.5 * (b["bbox"][1] + b["bbox"][3]),
        )
        ref_col_sorted = sorted(
            ref_cols[ci],
            key=lambda b: 0.5 * (b["bbox"][1] + b["bbox"][3]),
        )
        p_chars = [b["label"] for b in pred_col_sorted]
        r_chars = [b["label"] for b in ref_col_sorted]

        col_correct, col_matched, col_misread, col_unmatched_p, col_unmatched_r = \
            align_column(p_chars, r_chars)

        correct += col_correct
        n_matched += col_matched
        misread.extend(col_misread)
        unmatched_pred.extend(col_unmatched_p)
        unmatched_ref.extend(col_unmatched_r)

        per_col_report.append({
            "col": ci,
            "ref": len(ref_col_sorted),
            "pred": len(pred_col_sorted),
            "correct": col_correct,
            "misread": len(col_misread),
            "missed_in_pred": len(col_unmatched_r),  # ref chars not aligned to any pred
            "extra_in_pred": len(col_unmatched_p),   # pred chars not aligned to any ref
        })

    return {
        "n_pred": len(predicted),
        "n_ref": len(reference),
        "n_matched": n_matched,
        "n_correct": correct,
        "char_accuracy": correct / len(reference),
        "char_precision": (correct / len(predicted)) if predicted else 0.0,
        "misread": misread,
        "unmatched_pred": unmatched_pred,
        "unmatched_ref": unmatched_ref,
        "per_col": per_col_report,
    }

def load_reference_lyrics(json_path: Path) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        {"bbox": tuple(item["bbox"]), "label": item["char"]}
        for item in data.get("lyrics", [])
    ]


# -- main ----------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_argument_group("page source (pick one)")
    src.add_argument("--page", help="LGRC2024 page id (uses GT labels)")
    src.add_argument("--image", help="Path to a page image (uses --weights)")
    ap.add_argument("--label", help="YOLO label file (only with --image, optional)")
    ap.add_argument("--weights", default="checkpoints/lgrc2024_yolov8m_best.pt",
                    help="YOLO weights (used when no GT label is available)")
    ap.add_argument("--out", default=None, help="Output JSON path")
    ap.add_argument("--overlay", action="store_true",
                    help="Save a preview PNG with predicted boxes + chars")
    ap.add_argument("--model", default="claude-sonnet-4-5",
                    help="Anthropic model id (vision-capable)")
    ap.add_argument("--api-key", default=None,
                    help="Override ANTHROPIC_API_KEY env var")
    ap.add_argument("--eval-against", default=None,
                    help="Path to a reference demo_lyrics JSON; report char accuracy")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip the VLM call; emit JSON with '？' placeholders. "
                         "Useful for testing geometry / output paths.")
    ap.add_argument("--column-eps-factor", type=float, default=2.5,
                    help="Column-clustering tolerance multiplier "
                         "(see scripts/bootstrap_lyrics.py)")
    ap.add_argument("--mode", choices=["auto", "bootstrap"], default="auto",
                    help="auto: VLM determines char count per column "
                         "(default; works without prior knowledge of lyric "
                         "count). bootstrap: use bootstrap_lyrics gap-finding "
                         "to pre-determine count.")
    args = ap.parse_args()

    if not args.page and not args.image:
        ap.error("provide either --page or --image")

    # Resolve image + pitch boxes
    if args.page:
        img_path, lbl_path = find_image_and_label(args.page)
        page_id = args.page
    else:
        img_path = Path(args.image)
        lbl_path = Path(args.label) if args.label else None
        page_id = img_path.stem

    img = Image.open(img_path)
    page_w, page_h = img.size

    if lbl_path and lbl_path.exists():
        print(f"[geometry] using GT labels: {lbl_path}")
        pitches = pitches_from_gt_label(lbl_path, page_w, page_h)
    else:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            ap.error(f"no GT label available and weights not found: {weights_path}")
        print(f"[geometry] running detector: {weights_path}")
        pitches = pitches_from_detector(img_path, weights_path)
    print(f"[geometry] {len(pitches)} pitch boxes")

    if args.mode == "bootstrap":
        # Geometric gap-finding determines positions AND count; VLM only
        # supplies characters. Useful when you trust the geometry more than
        # the VLM's char-count-from-image judgement.
        bootstrap = infer_lyric_boxes(
            pitches, page_w, page_h,
            column_eps_factor=args.column_eps_factor,
        )
        print(f"[geometry] inferred {len(bootstrap)} lyric positions across "
              f"{len({b['_column'] for b in bootstrap})} columns")
        if args.dry_run:
            print("[vlm] dry-run: skipping API call")
            filled = [
                {**b, "label": "？", "kind": "lyric", "score": 0.0}
                for b in bootstrap
            ]
        else:
            filled = vlm_fill_lyrics(
                img_path, bootstrap,
                model=args.model, api_key=args.api_key,
            )
    else:
        # Default: column geometry only, VLM determines char count + chars.
        if args.dry_run:
            from jade_dragon.alignment import cluster_columns
            cols = cluster_columns(pitches, column_eps_factor=args.column_eps_factor)
            print(f"[geometry] {len(cols)} columns, sizes={[len(c) for c in cols]}")
            print("[vlm] dry-run: skipping API call")
            filled = []
            # Synthesize one ？ per column at the column's vertical midpoint
            # just so downstream code has something to work with.
            for ci, col in enumerate(cols):
                if not col:
                    continue
                xs = [b["bbox"][0] for b in col] + [b["bbox"][2] for b in col]
                ys = [b["bbox"][1] for b in col] + [b["bbox"][3] for b in col]
                cx = sum(xs) / len(xs)
                cy = (min(ys) + max(ys)) / 2
                filled.append({
                    "bbox": (cx - 30, cy - 30, cx + 30, cy + 30),
                    "kind": "lyric", "label": "？", "score": 0.0,
                    "_column": ci,
                })
        else:
            filled = vlm_recognize_page(
                img_path, pitches,
                column_eps_factor=args.column_eps_factor,
                model=args.model, api_key=args.api_key,
            )

    # Save in demo_lyrics JSON format
    out_path = Path(args.out) if args.out else DEMO_LYRICS / f"{page_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "page_w": page_w,
        "page_h": page_h,
        "_source": f"VLM-recognized via {args.model} (jade_dragon.lyric_ocr.vlm_fill_lyrics)",
        "lyrics": [
            {"bbox": list(b["bbox"]), "char": b["label"]}
            for b in filled
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[out] wrote {len(filled)} lyrics to {out_path}")

    # Overlay
    if args.overlay:
        overlay_path = out_path.with_suffix(".preview.png")
        canvas = img.convert("RGB").copy()
        d = ImageDraw.Draw(canvas)
        for b in filled:
            x1, y1, x2, y2 = b["bbox"]
            d.rectangle([x1, y1, x2, y2], outline="blue", width=4)
        canvas.save(overlay_path)
        print(f"[overlay] {overlay_path}")

    # Eval
    if args.eval_against:
        ref = load_reference_lyrics(Path(args.eval_against))
        pred = [{"bbox": tuple(b["bbox"]), "label": b["label"]} for b in filled]
        report = char_accuracy(pred, ref)
        print()
        print("=" * 60)
        print(f"Character accuracy vs {args.eval_against}")
        print("=" * 60)
        print(f"  predicted: {report['n_pred']}, reference: {report['n_ref']}, "
              f"matched: {report['n_matched']}")
        print(f"  correct (of matched): {report['n_correct']}")
        print(f"  recall:    {report['char_accuracy']:.1%}")
        print(f"  precision: {report['char_precision']:.1%}")
        for c in report.get("per_col", []):
            print(f"  col {c['col']}: pred={c['pred']:>2}, ref={c['ref']:>2}, "
                  f"correct={c['correct']:>2}, misread={c['misread']}")
        if report["misread"]:
            misread_str = ", ".join(f"{p}→{r}" for p, r in report["misread"][:20])
            print(f"  misreads ({len(report['misread'])}): {misread_str}"
                  + ("..." if len(report["misread"]) > 20 else ""))
        if report["unmatched_pred"]:
            print(f"  unmatched predictions: {report['unmatched_pred']}")
        if report["unmatched_ref"]:
            print(f"  unmatched references:  {report['unmatched_ref']}")


if __name__ == "__main__":
    main()
