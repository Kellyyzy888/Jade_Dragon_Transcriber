"""
Merge a new VIA JSON export into the existing via_all.json, preserving
all annotations from both. Pages that exist in both (same filename) are
overwritten by the NEW file (assumes the new file is the latest truth).

Usage:
    python3 scripts/merge_via_json.py \
        via_all.json \
        ~/Downloads/via_project_8May2026_15h11m_json.json \
        --out via_all.json
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path


def get_metadata(data: dict) -> dict:
    """VIA exports come in two shapes — handle both."""
    if "_via_img_metadata" in data:
        return data["_via_img_metadata"]
    return data


def set_metadata(data: dict, meta: dict) -> dict:
    """Put metadata back into the same shape as the input data."""
    if "_via_img_metadata" in data:
        data["_via_img_metadata"] = meta
        return data
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path, help="The existing via_all.json (kept as base)")
    ap.add_argument("new", type=Path, help="The new VIA export to merge in")
    ap.add_argument("--out", type=Path, required=True, help="Output path (can be same as base)")
    args = ap.parse_args()

    base_data = json.loads(args.base.read_text())
    new_data = json.loads(args.new.read_text())

    base_meta = get_metadata(base_data)
    new_meta = get_metadata(new_data)

    # Match by filename, not by VIA's opaque keys (which include file size).
    base_by_fname = {v["filename"]: (k, v) for k, v in base_meta.items()}
    new_by_fname = {v["filename"]: (k, v) for k, v in new_meta.items()}

    print(f"[merge] base has {len(base_by_fname)} pages")
    print(f"[merge] new has  {len(new_by_fname)} pages")

    # Merge: for each filename, prefer new if present, else keep base.
    overlap = set(base_by_fname) & set(new_by_fname)
    only_base = set(base_by_fname) - set(new_by_fname)
    only_new = set(new_by_fname) - set(base_by_fname)

    print(f"[merge]   {len(overlap)} pages in both (new wins)")
    print(f"[merge]   {len(only_base)} pages only in base (kept)")
    print(f"[merge]   {len(only_new)} pages only in new (added)")

    merged = {}
    # Use base entries where new doesn't override.
    for fname in only_base:
        k, v = base_by_fname[fname]
        merged[k] = v
    # Use new entries (covers overlap + only_new).
    for fname, (k, v) in new_by_fname.items():
        merged[k] = v

    print(f"[merge] result: {len(merged)} total pages")

    # Backup the base before overwriting (safety only — won't leave .bak around)
    if args.out == args.base:
        # If overwriting in place, dump to a temp first then move.
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        out_data = set_metadata(base_data, merged)
        tmp.write_text(json.dumps(out_data))
        shutil.move(str(tmp), str(args.out))
    else:
        out_data = set_metadata(base_data, merged)
        args.out.write_text(json.dumps(out_data))

    print(f"[merge] wrote {args.out}")
    total_boxes = sum(len(v.get("regions", [])) for v in merged.values())
    print(f"[merge] total boxes across {len(merged)} pages: {total_boxes}")


if __name__ == "__main__":
    main()
