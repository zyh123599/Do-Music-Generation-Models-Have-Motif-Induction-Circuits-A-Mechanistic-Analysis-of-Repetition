"""MIDI construction (pretty_midi) and FluidSynth rendering to 32 kHz WAV.

Rendering resolution order: ``fluidsynth`` CLI -> ``pyfluidsynth`` ->
RuntimeError with an install hint. SoundFont resolution order: explicit arg
-> ``$MOTIF_SF2`` -> common system paths. All heavy imports are lazy.
"""
from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
import shutil
import subprocess
import typing as tp

import numpy as np

from .motif import Motif

__all__ = ["build_midi", "render_midi", "resolve_soundfont", "peak_normalize"]

logger = logging.getLogger(__name__)

_SF2_GLOBS = (
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/*.sf2",
    "/usr/share/soundfonts/*.sf2",
)


def resolve_soundfont(soundfont: tp.Optional[str] = None) -> str:
    """Locate a GM SoundFont (arg > $MOTIF_SF2 > system paths)."""
    candidates: tp.List[str] = []
    if soundfont:
        candidates.append(str(soundfont))
    env = os.environ.get("MOTIF_SF2")
    if env:
        candidates.append(env)
    for pattern in _SF2_GLOBS:
        candidates.extend(sorted(glob.glob(pattern)))
    for c in candidates:
        if Path(c).is_file():
            return c
    raise RuntimeError(
        "No GM SoundFont found. Install one (e.g. "
        "`apt install fluid-soundfont-gm`) or set MOTIF_SF2=/path/to/GM.sf2")


def build_midi(sections: tp.Sequence[tp.Tuple[Motif, float, int]], bpm: float,
               out_path: tp.Union[str, Path], velocity: int = 96,
               velocity_jitter: tp.Optional[np.random.Generator] = None
               ) -> None:
    """Write a MIDI file from (motif, start_beat, GM program) sections.

    Each section gets its own instrument track with its own program so S4
    (timbre-change) stimuli can switch instruments mid-file. Velocities are
    ``velocity`` +- deterministic jitter of up to 3 when a generator is given
    (the render-seed variation of the stimulus set).

    Parameters
    ----------
    sections : sequence of (Motif, start_beat, program)
    bpm : float
        Fixed tempo of the whole file.
    out_path : path
        ``.mid`` output path (parent dirs created).
    velocity : int
        Base MIDI velocity (1-127).
    velocity_jitter : np.random.Generator, optional
        Jitter source; drawn per note in section order (deterministic).
    """
    import pretty_midi  # lazy

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    spb = 60.0 / float(bpm)
    for motif, start_beat, program in sections:
        inst = pretty_midi.Instrument(program=int(program))
        for pitch, onset, dur in zip(motif.pitches, motif.onsets_beats,
                                     motif.durations_beats):
            vel = int(velocity)
            if velocity_jitter is not None:
                vel += int(velocity_jitter.integers(-3, 4))
            vel = int(np.clip(vel, 1, 127))
            start = (float(start_beat) + float(onset)) * spb
            end = start + float(dur) * spb
            inst.notes.append(pretty_midi.Note(
                velocity=vel, pitch=int(pitch), start=start, end=end))
        pm.instruments.append(inst)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))


def peak_normalize(wav: np.ndarray, headroom_db: float = 1.0) -> np.ndarray:
    """Scale so the peak sits ``headroom_db`` below full scale (silence-safe)."""
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak <= 0:
        return wav
    target = 10.0 ** (-headroom_db / 20.0)
    return (wav * (target / peak)).astype(np.float32)


def _render_cli(midi_path: Path, sf2: str, sr: int, gain: float,
                tmp_wav: Path) -> bool:
    exe = shutil.which("fluidsynth")
    if not exe:
        return False
    cmd = [exe, "-ni", "-r", str(sr), "-g", f"{gain:.3f}",
           "-F", str(tmp_wav), sf2, str(midi_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"fluidsynth failed: {res.stderr[-500:]}")
    return True


def _render_pyfluidsynth(midi_path: Path, sf2: str, sr: int, gain: float,
                         tmp_wav: Path) -> bool:
    try:
        import midi2audio  # thin pyfluidsynth wrapper
    except ImportError:
        return False
    midi2audio.FluidSynth(sound_font=sf2, sample_rate=sr).midi_to_audio(
        str(midi_path), str(tmp_wav))
    _ = gain  # midi2audio has no gain control; normalization handles level
    return True


def render_midi(midi_path: tp.Union[str, Path], wav_path: tp.Union[str, Path],
                sr: int = 32000, soundfont: tp.Optional[str] = None,
                gain: float = 0.7, duration_s: float = 10.0) -> None:
    """Render a MIDI file to a mono float32 WAV of exactly ``duration_s``.

    Post-processing: mono mix-down, resample to ``sr`` when the renderer
    disagrees, peak-normalize to -1 dBFS, trim/zero-pad to ``duration_s``.

    Raises
    ------
    RuntimeError
        When neither the fluidsynth CLI nor pyfluidsynth is available
        (install hint included), or when rendering fails.
    """
    import soundfile as sf  # lazy

    midi_path, wav_path = Path(midi_path), Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf2 = resolve_soundfont(soundfont)
    tmp = wav_path.with_suffix(".tmp.wav")
    try:
        ok = _render_cli(midi_path, sf2, sr, gain, tmp)
        if not ok:
            ok = _render_pyfluidsynth(midi_path, sf2, sr, gain, tmp)
        if not ok:
            raise RuntimeError(
                "No MIDI renderer available. Install fluidsynth "
                "(`apt install fluidsynth fluid-soundfont-gm`) or "
                "`pip install midi2audio`")
        wav, got_sr = sf.read(tmp, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if got_sr != sr:
            from scipy.signal import resample_poly
            g = int(np.gcd(int(got_sr), int(sr)))
            wav = resample_poly(wav, sr // g, got_sr // g).astype(np.float32)
        wav = peak_normalize(wav, headroom_db=1.0)
        n = int(round(duration_s * sr))
        wav = wav[:n] if len(wav) >= n else np.pad(wav, (0, n - len(wav)))
        sf.write(wav_path, wav.astype(np.float32), sr, subtype="FLOAT")
    finally:
        tmp.unlink(missing_ok=True)
    logger.debug("rendered %s -> %s", midi_path.name, wav_path.name)
