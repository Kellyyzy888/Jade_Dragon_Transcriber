"""
End-to-end pipeline: image → MusicXML + MIDI + (optional) WAV.

Two entry points:
  - `transcribe_with_detector(image, weights, ...)` — full automatic pipeline
    using a trained YOLO detector for pitches and PaddleOCR for lyrics.
  - `transcribe_with_gt(image, gt_label_path, manual_lyrics_path, ...)` — uses
    LGRC2024 ground-truth labels for pitches + manually-supplied lyrics.
    This is the no-training MVP path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
from PIL import Image

from .alignment import align_page, yolo_label_to_boxes
from .musicxml_emit import write_musicxml
from .audio_render import musicxml_to_midi, midi_to_wav
from .pitch_map import CLASS_NAMES
from .pitch_detector import infer_key_class
from .lyric_ocr import load_manual_lyrics, detect_and_recognize


def transcribe_with_gt(
    image_path: Union[str, Path],
    gt_label_path: Union[str, Path],
    manual_lyrics_path: Union[str, Path],
    *,
    out_dir: Union[str, Path],
    work_title: str = "Gongche transcription",
    tempo_bpm: int = 80,
    render_wav: bool = False,
    soundfont_path: Optional[str] = None,
) -> dict:
    """No-training MVP path: pitches from GT YOLO labels, lyrics from manual JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path)
    w, h = img.size

    pitch_and_key_boxes = yolo_label_to_boxes(
        str(gt_label_path), w, h, CLASS_NAMES,
    )
    lyric_boxes = load_manual_lyrics(manual_lyrics_path)
    key_class = infer_key_class(pitch_and_key_boxes)

    aligned = align_page(lyric_boxes, [b for b in pitch_and_key_boxes if b["kind"] == "pitch"])

    stem = Path(image_path).stem
    xml_path = out_dir / f"{stem}.musicxml"
    midi_path = out_dir / f"{stem}.mid"
    wav_path = out_dir / f"{stem}.wav"

    write_musicxml(
        aligned,
        str(xml_path),
        key_class=key_class,
        tempo_bpm=tempo_bpm,
        work_title=work_title,
    )
    musicxml_to_midi(str(xml_path), str(midi_path))
    if render_wav:
        midi_to_wav(str(midi_path), str(wav_path), soundfont_path=soundfont_path)

    return {
        "musicxml": str(xml_path),
        "midi": str(midi_path),
        "wav": str(wav_path) if render_wav else None,
        "key": key_class,
        "n_lyrics": len(lyric_boxes),
        "n_pitches": sum(len(e["pitches"]) for e in aligned),
        "aligned": aligned,
    }


def transcribe_with_detector(
    image_path: Union[str, Path],
    weights: Union[str, Path],
    *,
    out_dir: Union[str, Path],
    use_ocr: bool = True,
    ocr_backend: str = "easyocr",          # "easyocr" | "paddleocr" | "yolo_vlm"
    lyric_weights: Optional[Union[str, Path]] = None,
    manual_lyrics_path: Optional[Union[str, Path]] = None,
    work_title: str = "Gongche transcription",
    tempo_bpm: int = 80,
    render_wav: bool = False,
    soundfont_path: Optional[str] = None,
    # Back-compat: old callers used `use_paddle_ocr=True/False`.
    use_paddle_ocr: Optional[bool] = None,
) -> dict:
    """Full pipeline: trained YOLO for pitches + lyrics-source for lyrics → MusicXML/MIDI/WAV.

    Lyric source is determined by, in priority order:
      - `manual_lyrics_path` if given (hand-edited JSON)
      - the OCR backend chosen by `ocr_backend` if `use_ocr=True`:
        * "easyocr" / "paddleocr": single-stage CJK OCR (limited on calligraphy).
        * "yolo_vlm": trained lyric YOLO (boxes) + Claude vision (characters).
          Requires `lyric_weights` and ANTHROPIC_API_KEY in env.
      - empty list (no lyrics in output) otherwise
    """
    from .pitch_detector import PitchDetector

    # Back-compat shim
    if use_paddle_ocr is not None:
        use_ocr = use_paddle_ocr

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = PitchDetector(weights)
    pitch_and_key_boxes = detector.detect(image_path)
    key_class = infer_key_class(pitch_and_key_boxes)
    pitch_boxes = [b for b in pitch_and_key_boxes if b["kind"] == "pitch"]

    if manual_lyrics_path is not None:
        lyric_boxes = load_manual_lyrics(manual_lyrics_path)
    elif use_ocr:
        lyric_boxes = detect_and_recognize(
            image_path,
            backend=ocr_backend,
            lyric_weights=lyric_weights,
        )
    else:
        lyric_boxes = []

    aligned = align_page(lyric_boxes, pitch_boxes)

    stem = Path(image_path).stem
    xml_path = out_dir / f"{stem}.musicxml"
    midi_path = out_dir / f"{stem}.mid"
    wav_path = out_dir / f"{stem}.wav"

    write_musicxml(
        aligned,
        str(xml_path),
        key_class=key_class,
        tempo_bpm=tempo_bpm,
        work_title=work_title,
    )
    musicxml_to_midi(str(xml_path), str(midi_path))
    if render_wav:
        midi_to_wav(str(midi_path), str(wav_path), soundfont_path=soundfont_path)

    return {
        "musicxml": str(xml_path),
        "midi": str(midi_path),
        "wav": str(wav_path) if render_wav else None,
        "key": key_class,
        "n_lyrics": len(lyric_boxes),
        "n_pitches": sum(len(e["pitches"]) for e in aligned),
        "aligned": aligned,
    }
