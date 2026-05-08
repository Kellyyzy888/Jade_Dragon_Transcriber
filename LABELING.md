# Labeling workflow for JadeDragon

This project has **two distinct labeling artifacts**:

1. **YOLO bbox labels** in `lyric_dataset/labels/{train,val}/<page>.txt` — used to train the lyric detector. Contains positions only, no character identities.
2. **Character ground truth** in `demo_lyrics/<page>.json` — used to evaluate end-to-end character accuracy. Contains positions AND the actual Chinese characters.

You usually only produce the first; the second is bootstrapped from the YOLO+VLM pipeline output and hand-corrected for a small number of pages.

The bbox labels are produced by **VIA** (VGG Image Annotator) running offline from `via.html`. The cumulative VIA project file is `via_all.json` at the project root; a converter (`scripts/via_to_yolo.py`) translates it into per-page YOLO labels.

**Augmented variants (`<id>_1.jpg`, `<id>_2.jpg`, …) are excluded.** Only original LGRC2024 / Lilu Qupu pages (`<id>.jpg`, no underscore suffix) are used.

---

## Part 1: Labeling new pages with VIA

This is the main workflow you'll run when expanding the training set.

### Step 1: Pick pages worth labeling (active learning)

`scripts/page_picker.py` finds unlabeled originals in your image pool, runs the current detector at low confidence on each, and ranks them by "hardness" (count of detections in the uncertain conf band [0.1, 0.5]). It picks 15 hard + 15 random pages by default.

```bash
PYTHONPATH=. python3 scripts/page_picker.py \
    --pool "LGRC2024 dataset/datasets/images/train" "LGRC2024 dataset/datasets/images/val" \
    --already-labeled lyric_dataset/images/train lyric_dataset/images/val \
    --weights checkpoints/lyric_yolov8m_best.pt \
    --num-hard 15 --num-random 15 \
    --out picked_pages.txt
```

Why pick "hard" pages: the detector's confident wins teach it nothing new, but pages where it's uncertain (lots of low-confidence detections) are exactly where new training data has the most marginal value. Random picks supply layout variety so the active-learning loop doesn't drift toward a single page style.

The output `picked_pages.txt` is a flat list of file paths to label.

### Step 2: Stage the picks into the dataset directories

`scripts/stage_labeling_batch.py` copies the picked pages into `lyric_dataset/images/{train,val}/` with an 80/20 split. Skips files that are already there.

```bash
python3 scripts/stage_labeling_batch.py picked_pages.txt
```

After this you'll have ~24 new train images + ~6 new val images, ready to label.

### Step 3: Label in VIA

Open `via.html` in your browser (no install needed — it's a single self-contained HTML file). Then:

1. **Project → Load** → select `via_all.json`. This restores all previously-labeled pages with their bboxes.
2. **Add Files** in the left panel → navigate to `lyric_dataset/images/train/` and ⌘-click each new train file. Click Open.
3. **Add Files** again → `lyric_dataset/images/val/` → ⌘-click each new val file.
4. Press **N** to step through pages. Pages with boxes already drawn = skip. Pages blank = label.

For each lyric character on a blank page:
- Click and drag a tight rectangle around the character.
- Boxes should enclose ONLY the lyric character, not the gongche annotations beside it.
- Skip stage directions (small text like 旦上唱, 介, 唱, 白, 介白), titles (right-side e.g. 尋夢), and attributions (left-side e.g. 粟蘆曲譜, 牡丹亭). Only label the large kaishu lyric calligraphy that has gongche pitch marks beside it.

When done with all 30 new pages: **Annotation → Export Annotations (as json)** → save it somewhere temporary like `~/Downloads/via_new.json`.

### Step 4: Merge the new export into the cumulative project

VIA's "export" produces only the pages currently loaded. To preserve the labels you've accumulated across batches, merge:

```bash
mv ~/Downloads/via_project_*.json ./via_new.json
python3 scripts/merge_via_json.py via_all.json via_new.json --out via_all.json
```

The merger is filename-keyed: pages in both files are overwritten by the new file (latest wins); pages only in the base are preserved. Expect output like:

```
[merge] base has 21 pages
[merge] new has  30 pages
[merge]   0 pages in both (new wins)
[merge]   21 pages only in base (kept)
[merge]   30 pages only in new (added)
[merge] result: 51 total pages
[merge] total boxes across 51 pages: 2029
```

### Step 5: Convert to YOLO format

```bash
python3 scripts/via_to_yolo.py via_all.json
```

This regenerates every `.txt` in `lyric_dataset/labels/{train,val}/` from the merged VIA JSON. Auto-routes each page to train or val based on which `lyric_dataset/images/` subfolder the corresponding `.jpg` lives in. Cleanup tip: the converter writes a stray `classes.txt` into each labels dir — harmless but you can delete it (the class is already defined in `data.yaml`).

### Step 6: Sanity-check before training

```bash
PYTHONPATH=. python3 scripts/sanity_check_lyric_labels.py --pages 14 79 140 178
```

Open `outputs/label_sanity/{14,79,140,178}_labels.png`. Confirm boxes sit tightly on lyric characters and not on gongche annotations.

### Step 7: Train

**Locally** (slow on Mac, ~3-5 hours for 51 pages):

```bash
PYTHONPATH=. python3 scripts/train_lyric_detector.py --data lyric_dataset/data.yaml
```

**On Oscar** (recommended, ~10-15 min on L40S):

```bash
rsync -avh --delete lyric_dataset/ oscar:/oscar/data/class/csci1430/students/zyang188/JadeDragon/lyric_dataset/
ssh oscar "cd /oscar/data/class/csci1430/students/zyang188/JadeDragon && sbatch scripts/oscar_train_lyric.sbatch"
ssh oscar "squeue -u zyang188"   # watch for job start
```

The sbatch script automatically copies new weights to `checkpoints/lyric_yolov8m_best.pt` on Oscar after training. Pull them down with:

```bash
scp oscar:/oscar/data/class/csci1430/students/zyang188/JadeDragon/checkpoints/lyric_yolov8m_best.pt checkpoints/lyric_yolov8m_best.pt
```

### Step 8: Re-run the pipeline + check results

```bash
PYTHONPATH=. python3 scripts/inspect_lyric_boxes.py "lyric_dataset/images/val/137.jpg"
PYTHONPATH=. python3 scripts/demo_dual_yolo.py --page 137 --lyric-conf 0.55 --vlm
```

The inspect script writes `<page>_lyric_boxes.jpg` showing colored boxes per detected column with confidence scores. Use it to spot regressions on pages you'd previously seen working.

### `data.yaml` portability gotcha

Ultralytics resolves `path:` in `data.yaml` relative to the **current working directory**, not relative to the YAML file. To make training work from any CWD, set an absolute path:

```yaml
# lyric_dataset/data.yaml on Oscar
path: /oscar/data/class/csci1430/students/zyang188/JadeDragon/lyric_dataset
train: images/train
val: images/val
nc: 1
names:
  0: lyric
```

The Mac copy can use a different absolute path (or relative if you always train from project root). Keep them out of sync — one absolute path per environment.

---

## Part 2: Character ground truth (`demo_lyrics/<page>.json`)

Used **only for end-to-end evaluation**. Most pages don't need this; only the val pages you want to score against.

### Schema

```json
{
  "page_w": 1653,
  "page_h": 2338,
  "_source": "Lilu Qupu val/137.jpg (Xun Meng from The Peony Pavilion), hand-verified",
  "lyrics": [
    {"bbox": [x1, y1, x2, y2], "char": "嫩"},
    {"bbox": [x1, y1, x2, y2], "char": "畫"},
    ...
  ]
}
```

- `page_w`, `page_h` — pixel dimensions from `Image.open(path).size`.
- `_source` — provenance string. Mark as DRAFT if not yet verified.
- `lyrics` — flat list. Order doesn't strictly matter for scoring (the scorer clusters by x-center into columns then sorts by y), but reading order (right column top-to-bottom, then next column leftward) is conventional.
- `bbox` — page pixel coordinates (NOT normalized).
- `char` — the actual Chinese character at that bbox.

### Bootstrap workflow (much faster than typing from scratch)

`scripts/bootstrap_char_gt.py` runs the YOLO+VLM pipeline at conf=0.55 on the pages you specify and writes draft `demo_lyrics/<page>.json` files containing whatever Qwen3-VL produced. You then hand-correct the few wrong characters.

```bash
PYTHONPATH=. python3 scripts/bootstrap_char_gt.py --pages 172 186 107
```

Per-page, expect:
- ~95-98% of detected boxes will have the correct character already (Qwen3-VL is very strong on classical Chinese calligraphy).
- ~15-30% of actual lyric characters will be missing entirely (YOLO didn't detect them). These need to be added by hand if you want full GT — or accepted as detection misses if you only want to measure recognition quality on the detected subset.

### Hand-correction in your editor

Open `demo_lyrics/<page>.json` next to the page image (the originals live at `lyric_dataset/images/val/<page>.jpg` or `LGRC2024 dataset/datasets/images/val/<page>.jpg`). For each entry:

1. Glance at the page region defined by `bbox: [x1, y1, x2, y2]`.
2. If `char` matches what's drawn there, leave it.
3. If it's wrong, fix the `"char"` field. Common Qwen3-VL ambiguities:
   - 嬾 vs 嫩 (visually identical in this calligraphy)
   - 茶 vs 荼 (one stroke difference at top)
   - 慵 vs 慷 (similar 心 component)
4. Look for `〇` or `？` placeholders — those are guaranteed errors that need human input.
5. When done, change `_source` from "DRAFT" to a verified provenance string like `"Lilu Qupu val/172.jpg, hand-verified"`.

### Scoring against GT

```bash
PYTHONPATH=. python3 scripts/demo_dual_yolo.py \
    --page 137 \
    --lyric-conf 0.55 \
    --vlm \
    --eval-against demo_lyrics/137.json
```

Prints sequence-aligned character accuracy. The scorer uses `difflib.SequenceMatcher` (which finds the longest common subsequence) so a single missed detection counts as one deletion rather than cascading misalignments.

For aggregate eval across all val pages:

```bash
PYTHONPATH=. python3 scripts/val_sweep.py
```

For pages without GT, you can also use `scripts/manual_tally.py` to log hand-counted accuracy:

```bash
python3 scripts/manual_tally.py 137 28 34
python3 scripts/manual_tally.py show
```

---

## Tips for fast labeling

**Pick batches with layout variety.** Mix LGRC2024 (clean, single-column lyrics) with Lilu Qupu (137-194 range, dense dual-column gongche+lyric) so the detector sees both distributions. The page picker's "hard" picks tend toward Lilu Qupu, the random picks balance it.

**Keep box sizes uniform within a column.** Lyric characters are roughly the same size on a page. If your boxes vary widely in size, you've probably accidentally boxed a gongche annotation as a lyric.

**Don't relabel pages already in `via_all.json`.** VIA preserves them; you only need to label the new ones. After "Add Files" in step 3, the old labeled pages will appear in the file panel with their boxes already drawn — skip those.

**Check `data.yaml` after every dataset change.** Run a quick `head lyric_dataset/data.yaml` and confirm the path is absolute and points to the right location for the machine you're training on.

**On the gongche-vs-lyric distinction.** Lyric characters are the LARGE kaishu chars in their own vertical column. Gongche pitch chars are SMALLER (about half the size) and sit beside the lyric column with octave-modifier prefixes (亻 for upper octave, etc.). When in doubt, look for the size: if it's the size of the BIG characters, label it; if it's a small character with hooks/decorations, skip it.
