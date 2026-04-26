# Manual lyric annotations

Hand-edited lyric bounding boxes for selected LGRC2024 pages, used by
`scripts/demo_one_page.py` before lyric OCR is wired up.

## Schema

```json
{
  "page_w": 2480,
  "page_h": 3508,
  "lyrics": [
    {"bbox": [x1, y1, x2, y2], "char": "謁"},
    {"bbox": [x1, y1, x2, y2], "char": "金"}
  ]
}
```

Coordinates are in **pixels** of the original page image (top-left origin, +y down).
Lyrics should be listed in any order — `alignment.align_page` will sort them by
column and then top-to-bottom within each column.

## How to make one

1. Open the page image (e.g., `LGRC2024 dataset/datasets/images/train/100.jpg`)
   in any image editor or [LabelImg](https://github.com/HumanSignal/labelImg).
2. Draw a bounding box around each large Chinese lyric character.
3. Note the pixel coordinates (most editors show them on hover).
4. Type the character. Save as `<page_id>.json` here.

## Sample

`100.json` is included as a worked example for page 100. It is not
ground-truth — it's an approximation good enough to demonstrate the pipeline.
