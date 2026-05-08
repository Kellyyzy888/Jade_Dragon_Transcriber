# JadeDragon Transcriber

Optical music recognition for **Kunqu opera gongche notation (工尺谱)** — turn a scanned page of handwritten gongche into a MusicXML score with pitches and lyrics aligned, plus playable MIDI.

This is the deep-learning successor to Chen Genfang's 2014 EURASIP system, framed as an [AMNLT](https://arxiv.org/abs/2412.04217)-style aligned music-and-lyrics transcription task for non-Western vocal scores.

---

## What this does

**Input:** a JPG of a Kunqu gongche page (e.g. from *Lilu Qupu* 栗廬曲譜).

**Output:** a MusicXML 4.0 file with:
- Notes for each detected gongche pitch character
- Real Chinese lyric characters attached as melismas (one syllable held over multiple notes)
- Key signature inferred from the page
- Plus a MIDI rendering for audio playback

**Pipeline stages:**

```
   page.jpg
       │
       ├──> YOLOv8m pitch detector (28 classes)  ──> pitch + key-sig boxes
       │
       ├──> YOLOv8m lyric detector (1 class)     ──> bbox per lyric character
       │       │
       │       └──> Qwen3-VL per-box recognition ──> Chinese character per box
       │
       ├──> Geometric column-based alignment     ──> (lyric, [pitches]) tuples
       │
       └──> MusicXML emitter                     ──> .musicxml + .mid
```

The pipeline is fully automated end-to-end. Two trained YOLO checkpoints handle detection (pitches and lyric positions); a vision-language model handles character recognition only on the bboxes the lyric detector found, so non-lyric text (stage directions, attributions) is excluded by construction.

## Current results

**Pitch detector** — YOLOv8m on LGRC2024 (1496 train / 638 val):

| Metric | This work | LGRC2024 paper baseline |
|---|---|---|
| mAP50 | **~81%** (peak) | 74.3% |
| mAP50-95 | ~45% | not reported |
| Training | 110 epochs on Brown Oscar A40 (~1 h) | 150 epochs |

The +7 mAP50 over the paper's YOLOv8m baseline comes from disabling rotation/flip augmentation — rotated pages aren't part of the real test distribution and the inductive bias hurts gongche reading order.

**Lyric detector** — YOLOv8m, single-class, trained on 51 hand-labeled pages (40 train / 11 val) with 2029 lyric bboxes:

| Metric | First training (21 pages) | Final (51 pages) |
|---|---|---|
| mAP50 | 0.904 | **0.993** |
| mAP50-95 | 0.443 | **0.503** |
| Precision | 0.862 | **0.972** |
| Recall | 0.892 | **0.980** |

The expansion from 21 to 51 pages, with deliberate coverage of dense Lilu Qupu layouts, eliminated false positives where the detector previously fired on small gongche annotation characters.

**End-to-end character recognition** — manual evaluation on all 11 val pages:

| Page set | Pages | Total chars | Correct | Accuracy |
|---|---|---|---|---|
| LGRC2024-style (clean layout) | 5 | 188 | 159 | 85% |
| Lilu Qupu (dense, multi-column) | 6 | 225 | 171 | 76% |
| **Aggregate** | **11** | **413** | **330** | **80%** |

Where errors come from:
- ~5% Qwen recognition errors (mostly calligraphic ambiguities like 嬾/嫩, 茶/荼)
- ~15% YOLO detection misses (top/bottom of dense columns, chars adjacent to stage directions)

The pitch detection accuracy and the per-box VLM recognition rate are both ~95%+; the bottleneck is detector recall on the densest Lilu Qupu pages.

## Project layout

```
JadeDragon Transcriber/
├── LGRC2024 dataset/          # YOLO-format pitch dataset (gitignored)
├── lyric_dataset/             # YOLO-format lyric-bbox dataset
│   ├── images/{train,val}/    # 40 train + 11 val originals
│   ├── labels/{train,val}/    # YOLO-format box labels (single class)
│   └── data.yaml
├── jade_dragon/               # core package
│   ├── pitch_map.py           # gongche character ↔ Western pitch + key sig
│   ├── pitch_detector.py      # ultralytics wrapper for pitch inference
│   ├── lyric_ocr.py           # LyricDetector + per-box VLM recognition
│   ├── alignment.py           # geometric lyric ↔ pitch grouping
│   ├── musicxml_emit.py       # tuples → MusicXML 4.0 with lyric melismas
│   ├── audio_render.py        # MusicXML → MIDI → WAV (via fluidsynth)
│   └── pipeline.py            # end-to-end glue
├── scripts/
│   ├── demo_dual_yolo.py      # main entry point: dual-detector + VLM pipeline
│   ├── train_pitch_detector.py
│   ├── train_lyric_detector.py
│   ├── via_to_yolo.py         # VIA bbox JSON → YOLO labels
│   ├── merge_via_json.py      # merge VIA exports into via_all.json
│   ├── page_picker.py         # active learning: pick hardest unlabeled pages
│   ├── stage_labeling_batch.py# stage picked pages into train/val 80/20
│   ├── inspect_lyric_boxes.py # diagnostic: render predicted boxes on a page
│   ├── sanity_check_lyric_labels.py  # render ground-truth boxes on a page
│   ├── bootstrap_char_gt.py   # bootstrap char GT from YOLO+VLM output
│   ├── val_sweep.py           # run pipeline on full val set + accuracy
│   ├── manual_tally.py        # log hand-counted per-page accuracy
│   ├── recognize_lyrics_vlm.py # legacy: geometry-derived lyric VLM (column-strip)
│   ├── oscar_train.sbatch     # SLURM batch: pitch detector
│   └── oscar_train_lyric.sbatch # SLURM batch: lyric detector
├── notebooks/
│   ├── 01_train_pitch_detector.ipynb
│   └── 02_demo_one_page_e2e.ipynb
├── configs/
│   ├── lgrc2024.yaml          # pitch YOLO data.yaml
│   └── lyric.yaml             # lyric YOLO data.yaml
├── demo_lyrics/               # per-page character ground truth (JSON)
├── via_all.json               # cumulative VIA project (all labeled pages)
├── via.html                   # offline VIA labeler
├── outputs/                   # generated .musicxml / .mid / overlays
├── checkpoints/               # trained model weights (gitignored)
├── OSCAR.md                   # SLURM workflow on Brown Oscar
├── LABELING.md                # how to label new pages
├── environment.yml            # conda env spec (Linux/Oscar)
├── requirements.txt           # pip deps (Mac/local)
└── README.md
```

## Quick start

### Run the full pipeline on one page

If both checkpoints are present (`checkpoints/lgrc2024_yolov8m_best.pt` and `checkpoints/lyric_yolov8m_best.pt`):

```bash
pip3 install --user --break-system-packages ultralytics music21 Pillow numpy httpx
export OPENROUTER_API_KEY=sk-or-...      # or ANTHROPIC_API_KEY=sk-ant-...

PYTHONPATH=. python3 scripts/demo_dual_yolo.py \
    --page 137 \
    --lyric-conf 0.55 \
    --vlm
```

This produces in `outputs/dual_yolo/137/`:
- `137.musicxml` — score with pitches + lyric chars
- `137.mid` — playable MIDI
- `137.overlay.png` — visual sanity check (red = pitch boxes, blue = lyric boxes)
- `137.alignment.json` — column-grouped pitch+lyric tuples
- `137.lyrics.json` — recognized lyrics in `demo_lyrics/` format

Without `--vlm`, the script runs detection only and uses placeholder labels (`L1`, `L2`, ...) — useful when you want to inspect detector output without paying for VLM calls.

### Score a page against ground truth

```bash
PYTHONPATH=. python3 scripts/demo_dual_yolo.py \
    --page 137 \
    --lyric-conf 0.55 \
    --vlm \
    --eval-against demo_lyrics/137.json
```

Prints character accuracy with sequence-aligned matching (Needleman-Wunsch via `difflib.SequenceMatcher`), so a single missed detection counts as one deletion rather than cascading into apparent misreads on every subsequent character.

### Run on all val pages

```bash
PYTHONPATH=. python3 scripts/val_sweep.py
```

Runs the pipeline on every page in `lyric_dataset/images/val/`, scores those with character GT in `demo_lyrics/`, prints a 5-section summary, and writes `outputs/val_sweep_summary.json`.

### Train detectors from scratch

**On Brown Oscar (recommended):** see [`OSCAR.md`](OSCAR.md). Roughly:

```bash
# Pitch detector (~1 h on A40)
sbatch scripts/oscar_train.sbatch

# Lyric detector (~10-15 min on L40S given 51-page dataset)
sbatch scripts/oscar_train_lyric.sbatch
```

**On Mac/Colab:** the notebooks in `notebooks/` mirror the sbatch flow.

### Add new pages to the lyric training set

See [`LABELING.md`](LABELING.md). The full workflow is:

```bash
# 1. Active-learning pick: 15 hardest + 15 random unlabeled pages
PYTHONPATH=. python3 scripts/page_picker.py \
    --pool "LGRC2024 dataset/datasets/images/train" "LGRC2024 dataset/datasets/images/val" \
    --already-labeled lyric_dataset/images/train lyric_dataset/images/val \
    --weights checkpoints/lyric_yolov8m_best.pt \
    --out picked_pages.txt

# 2. Stage them 80/20 into lyric_dataset/images/{train,val}/
python3 scripts/stage_labeling_batch.py picked_pages.txt

# 3. Label in VIA (open via.html offline, draw boxes, export JSON)

# 4. Merge new export into via_all.json
python3 scripts/merge_via_json.py via_all.json via_new.json --out via_all.json

# 5. Convert to YOLO format
python3 scripts/via_to_yolo.py via_all.json

# 6. Ship to Oscar + retrain
rsync -avh --delete lyric_dataset/ oscar:/oscar/.../JadeDragon/lyric_dataset/
ssh oscar "cd /oscar/.../JadeDragon && sbatch scripts/oscar_train_lyric.sbatch"
```

## Method notes

**Two-stage detection-then-recognition.** Lyric character identity is an open vocabulary (~3000+ unique characters across Lilu Qupu); a YOLO classifier head wouldn't generalize to unseen characters. A `lyric`-only detector learns the visual properties of "lyric character bbox in this calligraphy style" and generalizes to any character. Recognition (which character) is handled separately by Qwen3-VL on per-box crops, so the model only sees one character per call and can't be confused by adjacent stage directions.

**Per-box vs column-strip VLM.** The `--vlm-strategy column` option uses one VLM call per gongche column (faster, ~1/10th the cost) but pulls in any non-lyric text in the column gap. The default `per_box` strategy makes one call per detected bbox (slower but isolates each character). On page 137, per-box recognition is ~95-98% accurate; column-strip leaks stage-direction characters into the output.

**Movable-do solfège.** Gongche labels assume 上 = C4 (C-major reference). At MusicXML emission we transpose all notes by the actual key signature (e.g., 小工調 = D major → +2 semitones). See `jade_dragon/pitch_map.py`.

**Suoyi-style alignment.** Lyrics flow top-to-bottom in vertical columns; gongche characters sit in the upper-right region of each lyric character. Within a column, each pitch is assigned to the lyric whose center-y is just below the pitch's center-y. Cross-column ordering is right-to-left (Kunqu reading order). Pure geometric heuristic — no learned alignment yet.

**Rhythm deferred.** Every detected pitch becomes one quarter note at 80 BPM, with bar lines every 4 quarter notes. Audibly mechanical, structurally correct. Real rhythm reconstruction needs detection of ban (。, downbeat) and yan (、, subdivision) marks — a future module.

**Why YOLO instead of OCR for pitches.** Standard OCR engines aren't trained on the octave-modified gongche forms (亻上, comma-decorated forms) and return text strings rather than the semantic pitch classes we need. Object detection with a 28-class custom scheme is the right tool.

## What's missing / open research

1. **Detector recall on dense Lilu Qupu pages.** mAP50 of 0.993 is measured on val pages similar to training distribution. End-to-end character accuracy drops from ~85% on clean LGRC2024-style pages to ~75% on dense Lilu Qupu pages where lyric chars sit tight against gongche columns and stage directions. More labeled Lilu Qupu pages would help.

2. **Rhythm reconstruction.** Detect ban/yan beat marks → infer durations + bar boundaries. The largest correctness gap remaining after lyric recognition.

3. **Generalization beyond Lilu Qupu.** Untested on Yuzhu, Yizi, Nanyin gongche, or the *Jicheng Qupu* corpus (32 volumes). Now that lyric OCR is automated, scaling eval to other corpora is much cheaper.

4. **Synthetic pretraining.** Gongche encoding rules are deterministic; lyric corpora are abundant. Procedural rendering of synthetic pages → unlimited paired (image, MusicXML) data.

## References

- LGRC2024 paper — He, Zhang, Zhang, Hu, *Electronics* **14**(14), 2802 (2025). [Link](https://www.mdpi.com/2079-9292/14/14/2802)
- Chen, G. (2014). An optical music recognition system for traditional Chinese Kunqu Opera scores. *EURASIP J. Audio Speech Music Process.* [Link](https://link.springer.com/article/10.1186/1687-4722-2014-7)
- Ríos-Vila, A. et al. (2024). Aligned Music Notation and Lyrics Transcription. [arXiv:2412.04217](https://arxiv.org/abs/2412.04217)
- Qwen3-VL technical report — Alibaba (2024)
- MusicXML 4.0 lyric element — [w3.org spec](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/lyric/)
- Gongche notation overview — [Wikipedia](https://en.wikipedia.org/wiki/Gongche_notation)
