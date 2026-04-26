"""
Mapping between gongche notation and Western pitches.

The LGRC2024 class scheme has 28 classes (0-indexed after `prepare_dataset.py`):
  0–6   : keySignature-{bB, C, D, bE, F, G, A}
  7–20  : note-{C3 .. B4}     (10 base gongche characters in C-major reference,
                               extended with note-C3..F3 placeholders)
  21–27 : note-{C5 .. B5}     (octave-up versions: 亻上, 亻尺, ...)

The 10 base gongche characters and their canonical pitches (when 上 = C4) are:

    合 (Hé)     — G3        ← class 11
    四 (Sì)     — A3        ← class 12
    一 (Yī)     — B3        ← class 13
    上 (Shàng)  — C4        ← class 14   (middle C in C-major reference)
    尺 (Chǐ)    — D4        ← class 15
    工 (Gōng)   — E4        ← class 16
    凡 (Fán)    — F4        ← class 17
    六 (Liù)    — G4        ← class 18
    五 (Wǔ)     — A4        ← class 19
    乙 (Yǐ)     — B4        ← class 20

Octave-up forms (亻 prefix) shift +1 octave, classes 21–27 (note-C5 .. note-B5).

The labels are *movable-do*: in 小工調 (D major), 上 sounds as D4, not C4.
`transpose_for_key()` returns the semitone shift to apply when emitting MusicXML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ---- Class index → label string ------------------------------------------------

CLASS_NAMES: list[str] = [
    "keySignature-bB",  # 0
    "keySignature-C",   # 1
    "keySignature-D",   # 2  (most common in Kunqu)
    "keySignature-bE",  # 3
    "keySignature-F",   # 4
    "keySignature-G",   # 5
    "keySignature-A",   # 6
    "note-C3",          # 7
    "note-D3",          # 8
    "note-E3",          # 9
    "note-F3",          # 10
    "note-G3",          # 11  合
    "note-A3",          # 12  四
    "note-B3",          # 13  一
    "note-C4",          # 14  上
    "note-D4",          # 15  尺
    "note-E4",          # 16  工
    "note-F4",          # 17  凡
    "note-G4",          # 18  六
    "note-A4",          # 19  五
    "note-B4",          # 20  乙
    "note-C5",          # 21  亻上
    "note-D5",          # 22  亻尺
    "note-E5",          # 23  亻工
    "note-F5",          # 24  亻凡
    "note-G5",          # 25  亻六
    "note-A5",          # 26  亻五
    "note-B5",          # 27  亻乙
]
assert len(CLASS_NAMES) == 28

NAME_TO_CLASS: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

# ---- Gongche character → class label ------------------------------------------

GONGCHE_CHAR_TO_CLASS: dict[str, str] = {
    # Base characters (C-major reference, octave 3-4)
    "合": "note-G3",
    "四": "note-A3",
    "一": "note-B3",
    "上": "note-C4",
    "尺": "note-D4",
    "工": "note-E4",
    "凡": "note-F4",
    "六": "note-G4",
    "五": "note-A4",
    "乙": "note-B4",
    # Octave-up (亻 prefix). Note: 合, 四, 一 typically don't get 亻 prefix in
    # standard practice (they're already in lower octave), but included for completeness.
    "亻上": "note-C5",
    "亻尺": "note-D5",
    "亻工": "note-E5",
    "亻凡": "note-F5",
    "亻六": "note-G5",
    "亻五": "note-A5",
    "亻乙": "note-B5",
}

# ---- Key signature mappings ----------------------------------------------------

# Gongche key name → music21 Key string
KEY_TO_MUSIC21: dict[str, str] = {
    "keySignature-bB": "B-",   # 上字调 — B♭ major
    "keySignature-C":  "C",    # 尺字调
    "keySignature-D":  "D",    # 小工调 (most common)
    "keySignature-bE": "E-",   # 凡字调
    "keySignature-F":  "F",    # 六字调
    "keySignature-G":  "G",    # 五字调
    "keySignature-A":  "A",    # 乙字调
}

# Reference key (the pitch labels assume 上 = C4). Transposition shifts the
# entire melody up by `key_to_semitone(key) - key_to_semitone(C)` semitones.
REFERENCE_KEY: str = "keySignature-C"

KEY_TO_SEMITONE: dict[str, int] = {
    "keySignature-bB": -2,   # B♭
    "keySignature-C":  0,    # C  ← reference
    "keySignature-D":  2,    # D
    "keySignature-bE": 3,    # E♭
    "keySignature-F":  5,    # F
    "keySignature-G":  7,    # G
    "keySignature-A":  9,    # A
}


# ---- Public API ----------------------------------------------------------------

@dataclass
class Pitch:
    """A pitch with optional alteration. `step` is one of C D E F G A B."""
    step: str
    octave: int
    alter: int = 0  # +1 = sharp, -1 = flat

    def to_music21_string(self) -> str:
        """Return e.g. 'C4', 'B-3' (B♭3), 'F#4'."""
        accidental = ""
        if self.alter == -1:
            accidental = "-"
        elif self.alter == 1:
            accidental = "#"
        return f"{self.step}{accidental}{self.octave}"


def class_name_to_pitch(class_name: str) -> Optional[Pitch]:
    """Parse a `note-X#` style class name into a Pitch. Returns None for non-note classes."""
    if not class_name.startswith("note-"):
        return None
    label = class_name[len("note-"):]
    # label is like "C4", "Bb3" etc. Our scheme doesn't use sharps/flats in note labels —
    # only the key signature provides accidentals.
    if len(label) < 2:
        return None
    step = label[0].upper()
    octave = int(label[1:])
    return Pitch(step=step, octave=octave, alter=0)


def class_index_to_pitch(class_idx: int) -> Optional[Pitch]:
    if 0 <= class_idx < len(CLASS_NAMES):
        return class_name_to_pitch(CLASS_NAMES[class_idx])
    return None


FLAT_KEYS = {"keySignature-bB", "keySignature-bE", "keySignature-F"}


def transpose_for_key(pitch: Pitch, key_class_name: str) -> Pitch:
    """
    Adjust a pitch from C-major reference to its sounding pitch in the given gongche key.

    Movable-do: the gongche character 上 always means *do*, but *do* sounds
    different depending on the key. In 小工調 (D major), 上 = D4, not C4.
    Enharmonic spelling follows the key signature (flats for flat keys, sharps
    for sharp keys).
    """
    if key_class_name not in KEY_TO_SEMITONE:
        return pitch  # unknown key, leave unchanged
    semitones = KEY_TO_SEMITONE[key_class_name]
    prefer = "flat" if key_class_name in FLAT_KEYS else "sharp"
    return _shift_pitch_semitones(pitch, semitones, prefer=prefer)


_CHROMATIC_SHARP = [
    ("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
    ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0),
]
_CHROMATIC_FLAT = [
    ("C", 0), ("D", -1), ("D", 0), ("E", -1), ("E", 0), ("F", 0),
    ("G", -1), ("G", 0), ("A", -1), ("A", 0), ("B", -1), ("B", 0),
]


def _shift_pitch_semitones(pitch: Pitch, semitones: int, prefer: str = "sharp") -> Pitch:
    """Shift a pitch up/down by N semitones. `prefer` ∈ {'sharp', 'flat'}."""
    step_to_semi = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    chromatic = _CHROMATIC_FLAT if prefer == "flat" else _CHROMATIC_SHARP
    cur_semi = step_to_semi[pitch.step] + pitch.alter
    total_semi = pitch.octave * 12 + cur_semi + semitones
    new_octave = total_semi // 12
    new_within = total_semi % 12
    new_step, new_alter = chromatic[new_within]
    return Pitch(step=new_step, octave=new_octave, alter=new_alter)


def music21_key_string(key_class_name: str) -> str:
    """Return e.g. 'D' for keySignature-D, 'B-' for keySignature-bB."""
    return KEY_TO_MUSIC21.get(key_class_name, "C")


def is_pitch_class(class_name: str) -> bool:
    return class_name.startswith("note-")


def is_key_signature_class(class_name: str) -> bool:
    return class_name.startswith("keySignature-")


# ---- Tiny smoke test (run `python -m jade_dragon.pitch_map`) -------------------

if __name__ == "__main__":
    # 上 = C4 in C-major reference
    p = class_name_to_pitch("note-C4")
    assert p is not None and p.to_music21_string() == "C4"
    # 上 in 小工調 (D major) sounds as D4
    transposed = transpose_for_key(p, "keySignature-D")
    assert transposed.to_music21_string() == "D4", transposed
    # 尺 in 小工調 sounds as E4
    p2 = class_name_to_pitch("note-D4")
    transposed2 = transpose_for_key(p2, "keySignature-D")
    assert transposed2.to_music21_string() == "E4", transposed2
    # 上 in 上字调 (B♭) sounds as B♭3
    transposed3 = transpose_for_key(p, "keySignature-bB")
    assert transposed3.to_music21_string() == "B-3", transposed3
    print("pitch_map.py smoke tests passed")
