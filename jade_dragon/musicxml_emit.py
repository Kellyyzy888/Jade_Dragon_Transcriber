"""
Convert (lyric, [pitch_class, ...]) tuples into MusicXML 4.0 with melisma lyrics.

MVP rhythm policy
-----------------
Every pitch becomes one quarter note. Bar lines fall every 4 quarter notes
regardless of musical phrase boundaries. This is metrically wrong but produces
a schema-valid, audible MusicXML. When `rhythm_parse.py` lands later, only
this module needs to change (the rest of the pipeline is rhythm-agnostic).

Key signature handling
----------------------
The pitch labels assume 上 = C4 (C-major reference). If the page declares a
key (e.g., 小工調 = D major), we transpose all notes by the appropriate
interval so the resulting audio is in the correct sounding key.

Melisma representation
----------------------
For a lyric with N pitches:
    - N == 1: single note carries `<lyric><syllabic>single</syllabic><text>X</text></lyric>`
    - N >= 2: first note has `<syllabic>begin</syllabic><text>X</text>`,
              middle notes have `<syllabic>middle</syllabic>` with no text,
              last note has `<syllabic>end</syllabic>` with no text.
              An `<extend/>` element is added to all but the first note.

This is the AMNLT-compliant schema and what MuseScore / Finale will render
correctly. It's verbose but correct; we build it directly rather than relying
on music21's lyric defaults (which are less reliable for melismas).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Optional

from .pitch_map import (
    Pitch,
    class_name_to_pitch,
    transpose_for_key,
    music21_key_string,
)

DEFAULT_TEMPO = 80
DEFAULT_DIVISIONS = 4   # divisions per quarter note (4 = 16th-note resolution)


def _key_to_fifths(key_class: Optional[str]) -> int:
    """MusicXML <fifths> for a major key."""
    fifths = {
        "keySignature-C":  0,
        "keySignature-G":  1,
        "keySignature-D":  2,
        "keySignature-A":  3,
        "keySignature-F": -1,
        "keySignature-bB": -2,
        "keySignature-bE": -3,
    }
    return fifths.get(key_class or "keySignature-C", 0)


def _emit_note(
    parent: ET.Element,
    pitch: Pitch,
    duration_divisions: int,
    *,
    lyric_text: Optional[str] = None,
    syllabic: Optional[str] = None,   # "single" | "begin" | "middle" | "end" | None
    extend: bool = False,
) -> None:
    """Append a <note> with optional lyric to `parent` (a <measure>)."""
    note_el = ET.SubElement(parent, "note")
    pitch_el = ET.SubElement(note_el, "pitch")
    ET.SubElement(pitch_el, "step").text = pitch.step
    if pitch.alter != 0:
        ET.SubElement(pitch_el, "alter").text = str(pitch.alter)
    ET.SubElement(pitch_el, "octave").text = str(pitch.octave)
    ET.SubElement(note_el, "duration").text = str(duration_divisions)
    ET.SubElement(note_el, "voice").text = "1"
    ET.SubElement(note_el, "type").text = "quarter"

    if syllabic is not None:
        lyric_el = ET.SubElement(note_el, "lyric", number="1")
        ET.SubElement(lyric_el, "syllabic").text = syllabic
        if lyric_text is not None:
            ET.SubElement(lyric_el, "text").text = lyric_text
        if extend:
            ET.SubElement(lyric_el, "extend", type="continue" if syllabic == "middle" else "stop")
    elif extend:
        # extend without syllabic — used for melisma continuations
        lyric_el = ET.SubElement(note_el, "lyric", number="1")
        ET.SubElement(lyric_el, "extend", type="stop")


def _emit_rest(parent: ET.Element, duration_divisions: int) -> None:
    note_el = ET.SubElement(parent, "note")
    ET.SubElement(note_el, "rest")
    ET.SubElement(note_el, "duration").text = str(duration_divisions)
    ET.SubElement(note_el, "voice").text = "1"
    ET.SubElement(note_el, "type").text = "quarter"


def aligned_to_musicxml(
    aligned: list[dict],
    *,
    key_class: Optional[str] = None,
    tempo_bpm: int = DEFAULT_TEMPO,
    work_title: str = "Gongche transcription",
    composer: str = "JadeDragon Transcriber",
) -> str:
    """
    Convert aligned [(lyric, [pitches]), ...] tuples into a MusicXML 4.0 string.

    `aligned` is the output of `alignment.align_page` — a list of dicts with
    `lyric` (Box or None) and `pitches` (list of Box).

    Returns: MusicXML as a pretty-printed string.
    """
    # ---- Score skeleton --------------------------------------------------------
    score = ET.Element("score-partwise", version="4.0")
    work = ET.SubElement(score, "work")
    ET.SubElement(work, "work-title").text = work_title
    identification = ET.SubElement(score, "identification")
    creator = ET.SubElement(identification, "creator", type="composer")
    creator.text = composer

    part_list = ET.SubElement(score, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Voice"

    part = ET.SubElement(score, "part", id="P1")

    # ---- Build a flat list of (Pitch, lyric_text, melisma_position) tuples -----
    # melisma_position: "single" | "begin" | "middle" | "end" | None (for keys/skipped)
    flat: list[tuple[Pitch, Optional[str], Optional[str]]] = []
    for entry in aligned:
        lyric = entry.get("lyric")
        pitches = entry.get("pitches", [])
        lyric_text = lyric.get("label") if lyric else None
        n = len(pitches)
        if n == 0:
            continue
        for i, pbox in enumerate(pitches):
            class_name = pbox.get("label", "")
            pitch = class_name_to_pitch(class_name)
            if pitch is None:
                continue
            if key_class is not None:
                pitch = transpose_for_key(pitch, key_class)
            if n == 1:
                pos = "single" if lyric_text else None
            elif i == 0:
                pos = "begin" if lyric_text else None
            elif i == n - 1:
                pos = "end" if lyric_text else None
            else:
                pos = "middle" if lyric_text else None
            flat.append((pitch, lyric_text if i == 0 else None, pos))

    if not flat:
        flat = []  # empty score is valid; will produce a 1-measure rest

    # ---- Emit measures (4/4, quarter notes, 4 notes per bar) -------------------
    notes_per_measure = 4
    measure_idx = 1
    cur_measure: Optional[ET.Element] = None
    notes_in_cur = 0

    def open_measure(idx: int, with_attributes: bool) -> ET.Element:
        m = ET.SubElement(part, "measure", number=str(idx))
        if with_attributes:
            attr = ET.SubElement(m, "attributes")
            ET.SubElement(attr, "divisions").text = str(DEFAULT_DIVISIONS)
            key_el = ET.SubElement(attr, "key")
            ET.SubElement(key_el, "fifths").text = str(_key_to_fifths(key_class))
            ET.SubElement(key_el, "mode").text = "major"
            time_el = ET.SubElement(attr, "time")
            ET.SubElement(time_el, "beats").text = "4"
            ET.SubElement(time_el, "beat-type").text = "4"
            clef = ET.SubElement(attr, "clef")
            ET.SubElement(clef, "sign").text = "G"
            ET.SubElement(clef, "line").text = "2"
            # Tempo direction
            direction = ET.SubElement(m, "direction", placement="above")
            dt = ET.SubElement(direction, "direction-type")
            metro = ET.SubElement(dt, "metronome")
            ET.SubElement(metro, "beat-unit").text = "quarter"
            ET.SubElement(metro, "per-minute").text = str(tempo_bpm)
            ET.SubElement(direction, "sound", tempo=str(tempo_bpm))
        return m

    for pitch, lyric_text, pos in flat:
        if cur_measure is None or notes_in_cur >= notes_per_measure:
            cur_measure = open_measure(measure_idx, with_attributes=(measure_idx == 1))
            measure_idx += 1
            notes_in_cur = 0
        extend = pos in ("middle", "end")
        _emit_note(
            cur_measure,
            pitch,
            duration_divisions=DEFAULT_DIVISIONS,
            lyric_text=lyric_text,
            syllabic=pos,
            extend=extend,
        )
        notes_in_cur += 1

    # Pad final measure with rests
    if cur_measure is not None and notes_in_cur < notes_per_measure:
        for _ in range(notes_per_measure - notes_in_cur):
            _emit_rest(cur_measure, DEFAULT_DIVISIONS)
    elif cur_measure is None:
        # No notes at all — emit one empty measure
        cur_measure = open_measure(1, with_attributes=True)
        for _ in range(notes_per_measure):
            _emit_rest(cur_measure, DEFAULT_DIVISIONS)

    # ---- Pretty-print ----------------------------------------------------------
    rough = ET.tostring(score, encoding="utf-8", xml_declaration=False)
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="UTF-8").decode("utf-8")
    # minidom adds its own declaration; ensure DOCTYPE for MusicXML
    body = pretty.split("?>", 1)[1] if pretty.startswith("<?xml") else pretty
    header = (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        '<!DOCTYPE score-partwise PUBLIC '
        '"-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n'
    )
    return header + body.lstrip()


def write_musicxml(aligned: list[dict], output_path: str, **kwargs) -> None:
    xml = aligned_to_musicxml(aligned, **kwargs)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml)


# ---- Smoke test ---------------------------------------------------------------

if __name__ == "__main__":
    aligned = [
        {
            "lyric": {"bbox": (0, 0, 1, 1), "kind": "lyric", "label": "謁"},
            "pitches": [
                {"bbox": (0, 0, 1, 1), "kind": "pitch", "label": "note-C4"},
                {"bbox": (0, 0, 1, 1), "kind": "pitch", "label": "note-D4"},
                {"bbox": (0, 0, 1, 1), "kind": "pitch", "label": "note-E4"},
            ],
            "column": 0,
        },
        {
            "lyric": {"bbox": (0, 0, 1, 1), "kind": "lyric", "label": "金"},
            "pitches": [
                {"bbox": (0, 0, 1, 1), "kind": "pitch", "label": "note-G4"},
            ],
            "column": 0,
        },
    ]
    xml = aligned_to_musicxml(aligned, key_class="keySignature-D")
    assert "<step>" in xml
    assert "謁" in xml
    assert "金" in xml
    assert "<syllabic>begin</syllabic>" in xml
    assert "<syllabic>single</syllabic>" in xml
    print("musicxml_emit.py smoke tests passed")
    print(xml[:500] + "\n...")
