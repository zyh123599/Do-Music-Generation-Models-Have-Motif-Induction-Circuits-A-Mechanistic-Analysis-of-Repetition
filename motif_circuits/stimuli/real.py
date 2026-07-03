"""S7 real-music validation set: loaders and manifest conversion.

The S7 set provides external validity (research plan §4, risk R3): real
riff/theme-recurrence excerpts on which the screening score RANKING and
ablation TRENDS must replicate (no ground-truth phi — alignment comes from
``analysis.alignment`` at analysis time).

Expected external layout (user-provided; audio is never downloaded here)::

    <s7_root>/
      riffs.csv          # one row per excerpt, columns below
      audio/<id>.wav     # 32 kHz mono (or convertible) excerpts, ~10 s

``riffs.csv`` columns (header required):
    id, wav_path, a_start_s, a_end_s, ap_start_s, ap_end_s[, semitones]

- ``a_*`` mark the motif segment A, ``ap_*`` its recurrence A' (seconds).
- ``semitones`` (optional): annotated transposition of A' vs A.

POP909: use its phrase-structure annotations to cut recurrence pairs, then
list them in ``riffs.csv``. SALAMI / HookTheory sections can be converted the
same way — the CSV is the single integration point.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
import typing as tp

from .build import (DURATION_S, FRAME_RATE, SAMPLE_RATE, StimulusRecord,
                    save_manifest)

__all__ = ["load_s7_csv", "convert_s7"]

logger = logging.getLogger(__name__)

_REQUIRED = ("id", "wav_path", "a_start_s", "a_end_s", "ap_start_s", "ap_end_s")


def _to_frames(seconds: float) -> int:
    return int(round(float(seconds) * FRAME_RATE))


def load_s7_csv(csv_path: tp.Union[str, Path]) -> tp.List[dict]:
    """Parse ``riffs.csv`` rows with validation and clear errors."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{csv_path} not found — create it per the layout documented in "
            "motif_circuits/stimuli/real.py (id, wav_path, a_start_s, "
            "a_end_s, ap_start_s, ap_end_s[, semitones])")
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        missing = [c for c in _REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path} misses required columns: {missing}")
        for i, row in enumerate(reader):
            try:
                for c in _REQUIRED[2:]:
                    row[c] = float(row[c])
            except ValueError as e:
                raise ValueError(f"{csv_path} row {i + 2}: {e}") from e
            if not (row["a_start_s"] < row["a_end_s"] <= row["ap_start_s"]
                    < row["ap_end_s"]):
                raise ValueError(
                    f"{csv_path} row {i + 2} ({row['id']}): need "
                    "a_start < a_end <= ap_start < ap_end")
            rows.append(row)
    return rows


def convert_s7(s7_root: tp.Union[str, Path],
               out_dir: tp.Union[str, Path]) -> tp.List[StimulusRecord]:
    """Convert an S7 annotation CSV into a standard stimulus manifest.

    Audio files are referenced in place (``wav_path`` may be absolute or
    relative to ``s7_root``); no audio is copied or re-encoded here — run
    ``scripts/02_encode_stimuli.py`` on the output directory as usual.
    """
    s7_root, out_dir = Path(s7_root), Path(out_dir)
    records = []
    for row in load_s7_csv(s7_root / "riffs.csv"):
        wav = Path(row["wav_path"])
        wav_abs = wav if wav.is_absolute() else s7_root / wav
        if not wav_abs.is_file():
            raise FileNotFoundError(f"S7 audio missing: {wav_abs}")
        a = [_to_frames(row["a_start_s"]), _to_frames(row["a_end_s"])]
        ap = [_to_frames(row["ap_start_s"]), _to_frames(row["ap_end_s"])]
        transform = {"type": "real"}
        if row.get("semitones") not in (None, ""):
            transform["semitones"] = int(float(row["semitones"]))
        records.append(StimulusRecord(
            id=str(row["id"]), category="S7", sr=SAMPLE_RATE,
            frame_rate=FRAME_RATE, bpm=0.0, key_root=-1, scale="unknown",
            program=-1, velocity=-1, seed=-1, render_seed=0,
            motif={}, transform=transform,
            segments={"A": a, "G": [a[1], ap[0]], "A_prime": ap},
            phi=None, wav_path=str(wav_abs), midi_path="",
            pair_id=None, duration_s=DURATION_S))
    save_manifest(records, out_dir / "manifest.jsonl")
    logger.info("converted %d S7 excerpts", len(records))
    return records
