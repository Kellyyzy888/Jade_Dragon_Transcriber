# JadeDragon Transcriber

Optical music recognition for **Kunqu opera gongche notation (工尺谱)** — turn a scanned page of handwritten gongche into a MusicXML score with pitches and lyrics aligned, plus playable MIDI.

This is the deep-learning successor to Chen Genfang's 2014 EURASIP system, framed as an [AMNLT](https://arxiv.org/abs/2412.04217)-style aligned music-and-lyrics transcription task for non-Western vocal scores.

---

## What this does

**Input:** a JPG of a Kunqu gongche page (e.g. from *Lilu Qupu* 栗廬曲譜).

**Output:** a MusicXML 4.0 file with:
- Notes for each detected gongche pitch character
- Lyrics attached as melismas (one syllable held over multiple notes)
- Key signature inferred from the page
- Plus a MIDI rendering for audio playback

**Pipeline stages:**

```
   page.jpg
       │
       ├──> YOLOv8m pitch detector  ──> 28-class boxes (pitches + keys)
       │
       ├──> Lyric character detector ──> CJK boxes + recognized chars
       │       (currently manual JSON; OCR options under research)
       │
       ├──> Geometric column-based alignment ──> (lyric, [pitches]) tuples
       │
       └──> MusicXML emitter ──> .musicxml + .mid + (.wav)
```

## Current results

Trained YOLOv8m on the LGRC2024 dataset (1496 train / 638 val):

| Metric | This work | LGRC2024 paper baseline |
|---|---|---|
| mAP50 | **~81%** (peak) | 74.3% |
| mAP50-95 | ~45% | not reported |
| Training | 110 epochs on Brown Oscar A40 (~1 h) | 150 epochs |

The +7 mAP50 over the paper's YOLOv8m baseline comes from disabling rotation/flip augmentation — rotated pages aren't part of the real test distribution and the inductive bias hurts gongche reading order.

End-to-end pipeline confirmed working on `val/137.jpg` (*Xun Meng* 尋夢 from *The Peony Pavilion*, 六字調 F major, 34 lyric chars + 89 detected pitches).

## Project layout

```
JadeDragon Transcriber/
├── LGRC2024 dataset/          # YOLO-format dataset (untouched, gitignored)
├── jade_dragon/               # core package
│   ├── pitch_map.py           # gongche character ↔ Western pitch + key sig
│   ├── alignment.py           # geometric lyric ↔ pitch grouping
│   ├── musicxml_emit.py       # tuples → MusicXML 4.0 with melisma lyrics
│   ├── audio_render.py        # MusicXML → MIDI → WAV (via fluidsynth)
│   ├── pitch_detector.py      # ultralytics YOLO wrapper for inference
│   ├── lyric_ocr.py           # EasyOCR + PaddleOCR backends + manual fallback
│   └── pipeline.py            # end-to-end glue
├── scripts/
│   ├── prepare_dataset.py     # remap labels to 0-indexed for ultralytics
│   ├── train_pitch_detector.py
│   ├── bootstrap_lyrics.py    # auto-suggest lyric box positions per page
│   ├── demo_one_page.py       # no-training MVP demo
│   ├── oscar_train.sbatch     # SLURM batch — full 150-epoch training
│   └── oscar_smoke.sbatch     # SLURM batch — 5-epoch smoke test
├── notebooks/
│   ├── 01_train_pitch_detector.ipynb     # Colab training (alternative to Oscar)
│   └── 02_demo_one_page_e2e.ipynb        # full pipeline demo
├── configs/
│   └── lgrc2024.yaml          # YOLO data.yaml
├── demo_lyrics/               # hand-transcribed lyric JSONs
├── outputs/                   # generated .musicxml / .mid / .wav
├── checkpoints/               # trained model weights (gitignored)
├── OSCAR.md                   # SLURM workflow on Brown Oscar
├── environment.yml            # conda env spec (Linux/Oscar)
├── requirements.txt           # pip deps (Mac/local)
└── README.md
```

## Quick start

### Try the demo on one page (no training needed)

If you have the trained weights at `checkpoints/lgrc2024_yolov8m_best.pt`:

```bash
pip3 install --user --break-system-packages ultralytics music21 Pillow numpy

python3 -c "
import sys; sys.path.insert(0, '.')
from jade_dragon.pipeline import transcribe_with_detector
result = transcribe_with_detector(
    image_path='LGRC2024 dataset/datasets/images/val/137.jpg',
    weights='checkpoints/lgrc2024_yolov8m_best.pt',
    out_dir='outputs/mvp_137',
    use_ocr=False,
    manual_lyrics_path='demo_lyrics/137.json',
)
print(result)
"
```

This produces `outputs/mvp_137/137.musicxml` and `137.mid`.

### Train the pitch detector from scratch

**On Brown Oscar (recommended):** see [`OSCAR.md`](OSCAR.md) for the full SLURM workflow. Roughly:

```bash
# Install miniforge and build the env
bash <(curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh) -b -p $HOME/miniforge3
source ~/.bashrc
conda env create -f environment.yml
conda activate jadedragon

# Submit smoke test (5 epochs, gpu-debug queue)
sbatch scripts/oscar_smoke.sbatch

# Submit full training (150 epochs, gpu queue, ~1 h on A40)
sbatch scripts/oscar_train.sbatch
```

**On Colab (alternative):** open `notebooks/01_train_pitch_detector.ipynb`, mount Drive, run all cells.

### Add a new page

1. Place the page image at any path; let's say `mypage.jpg`.
2. Generate placeholder lyric boxes:
   ```bash
   python3 scripts/bootstrap_lyrics.py --page mypage --overlay
   ```
3. Open `demo_lyrics/mypage.json` and `demo_lyrics/mypage.preview.png` side-by-side. Replace each `？` with the actual lyric character. Adjust box positions if needed (the bootstrap is approximate).
4. Run the pipeline as in the quick start above.

## Method notes

**Movable-do solfège.** Gongche labels assume 上 = C4 (C-major reference). At MusicXML emission we transpose all notes by the actual key signature (e.g., 小工調 = D major → +2 semitones). See `jade_dragon/pitch_map.py`.

**Suoyi-style alignment.** Lyrics flow top-to-bottom in vertical columns; gongche characters sit in the upper-right region of each lyric character. Within a column, each pitch is assigned to the lyric whose center-y is just below the pitch's center-y. Cross-column ordering is right-to-left (Kunqu reading order). Pure geometric heuristic — no learned alignment yet.

**Rhythm deferred.** Every detected pitch becomes one quarter note at 80 BPM, with bar lines every 4 quarter notes. Audibly mechanical, structurally correct. Real rhythm reconstruction needs detection of ban (。, downbeat) and yan (、, subdivision) marks — a future module.

**Why YOLO instead of OCR for pitches.** Standard OCR engines aren't trained on the octave-modified gongche forms (亻上, comma-decorated forms) and return text strings rather than the semantic pitch classes we need. Object detection with a 28-class custom scheme is the right tool.

## What's missing / open research

1. **OCR for calligraphic Chinese lyrics.** EasyOCR/PaddleOCR fail on Lilu Qupu's handwritten kaishu. Manual JSON is the current fallback. Options to explore: `ocrmac` (Apple Vision), VLM API (Claude/Gemini), fine-tune a recognition head on annotated lyric crops.
2. **Rhythm reconstruction.** Detect ban/yan beat marks → infer durations + bar boundaries. Next module after lyric OCR is automated.
3. **Generalization beyond Lilu Qupu.** Trained on one corpus in suoyi style. Untested on Yuzhu, Yizi, Nanyin gongche, or the *Jicheng Qupu* corpus (32 volumes).
4. **Synthetic pretraining.** Gongche encoding rules are deterministic; lyric corpora are abundant. Procedural rendering of synthetic pages → unlimited paired (image, MusicXML) data.

## References

- LGRC2024 paper — He, Zhang, Zhang, Hu, *Electronics* **14**(14), 2802 (2025). [Link](https://www.mdpi.com/2079-9292/14/14/2802)
- Chen, G. (2014). An optical music recognition system for traditional Chinese Kunqu Opera scores. *EURASIP J. Audio Speech Music Process.* [Link](https://link.springer.com/article/10.1186/1687-4722-2014-7)
- Ríos-Vila, A. et al. (2024). Aligned Music Notation and Lyrics Transcription. [arXiv:2412.04217](https://arxiv.org/abs/2412.04217)
- MusicXML 4.0 lyric element — [w3.org spec](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/lyric/)
- Gongche notation overview — [Wikipedia](https://en.wikipedia.org/wiki/Gongche_notation)
