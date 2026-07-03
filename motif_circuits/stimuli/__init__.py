"""Synthetic stimulus set (S1-S6) and real validation set (S7)."""
from .motif import (Motif, SCALES, sample_motif, transpose, rhythm_variation,
                    rhythm_scramble_pitches, sample_gap_material,
                    shares_interval_ngrams, motif_frames, compute_phi)
from .build import (StimulusRecord, load_manifest, save_manifest,
                    build_category, build_all, CATEGORIES, PROGRAMS)
from .midi_render import build_midi, render_midi, resolve_soundfont
from .real import convert_s7, load_s7_csv

__all__ = [
    "Motif", "SCALES", "sample_motif", "transpose", "rhythm_variation",
    "rhythm_scramble_pitches", "sample_gap_material",
    "shares_interval_ngrams", "motif_frames", "compute_phi",
    "StimulusRecord", "load_manifest", "save_manifest", "build_category",
    "build_all", "CATEGORIES", "PROGRAMS",
    "build_midi", "render_midi", "resolve_soundfont",
    "convert_s7", "load_s7_csv",
]
