"""
MusicXML → MIDI → WAV rendering.

Two backends:
  - music21 : converts MusicXML to MIDI in pure Python (always available).
  - fluidsynth : renders MIDI to WAV via a SoundFont (.sf2). System install required:
        macOS:  brew install fluid-synth
        Linux:  apt-get install fluidsynth
    Plus a SoundFont — `FluidR3_GM.sf2` is the standard freely-distributed choice.

If you only need to *listen* to the result (not produce a WAV file), opening the
.mid file in any DAW or even in macOS QuickTime (which has GM playback) works fine.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Optional


def musicxml_to_midi(xml_path: str, midi_path: str) -> None:
    """Convert MusicXML to MIDI using music21."""
    from music21 import converter
    score = converter.parse(xml_path)
    score.write("midi", fp=midi_path)


def midi_to_wav(
    midi_path: str,
    wav_path: str,
    *,
    soundfont_path: Optional[str] = None,
    sample_rate: int = 44100,
) -> None:
    """
    Render a MIDI file to WAV using fluidsynth.

    `soundfont_path`: path to a .sf2 file. If None, looks for
    `~/.fluidsynth/default_sound_font.sf2` or `/usr/share/sounds/sf2/FluidR3_GM.sf2`.
    """
    sf2 = soundfont_path or _default_soundfont()
    if sf2 is None or not Path(sf2).exists():
        raise FileNotFoundError(
            "No SoundFont found. Pass `soundfont_path=...` explicitly. "
            "Free option: download FluidR3_GM.sf2 (~140 MB) from "
            "https://member.keymusician.com/Member/FluidR3_GM/index.aspx "
            "or any GM SoundFont."
        )
    cmd = [
        "fluidsynth", "-ni",
        sf2,
        midi_path,
        "-F", wav_path,
        "-r", str(sample_rate),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"fluidsynth failed (exit {result.returncode}):\n{result.stderr}"
        )


def musicxml_to_wav(
    xml_path: str,
    wav_path: str,
    *,
    midi_path: Optional[str] = None,
    soundfont_path: Optional[str] = None,
) -> str:
    """
    Convenience: MusicXML → MIDI → WAV in one call.
    Returns the MIDI path (in case the caller wants to keep it).
    """
    if midi_path is None:
        midi_path = str(Path(wav_path).with_suffix(".mid"))
    musicxml_to_midi(xml_path, midi_path)
    midi_to_wav(midi_path, wav_path, soundfont_path=soundfont_path)
    return midi_path


def _default_soundfont() -> Optional[str]:
    candidates = [
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",                 # Debian/Ubuntu
        "/usr/share/soundfonts/FluidR3_GM.sf2",
        str(Path.home() / ".fluidsynth/default_sound_font.sf2"),
        "/opt/homebrew/share/fluid-synth/sf2/FluidR3_GM.sf2",   # macOS arm64
        "/usr/local/share/fluid-synth/sf2/FluidR3_GM.sf2",      # macOS intel
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


# ---- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("xml", help="Input MusicXML file")
    parser.add_argument("--midi", default=None, help="Output MIDI path")
    parser.add_argument("--wav", default=None, help="Output WAV path (requires fluidsynth)")
    parser.add_argument("--soundfont", default=None, help="Path to .sf2 SoundFont")
    args = parser.parse_args()

    midi_out = args.midi or str(Path(args.xml).with_suffix(".mid"))
    musicxml_to_midi(args.xml, midi_out)
    print(f"MIDI written to {midi_out}")

    if args.wav:
        midi_to_wav(midi_out, args.wav, soundfont_path=args.soundfont)
        print(f"WAV written to {args.wav}")
