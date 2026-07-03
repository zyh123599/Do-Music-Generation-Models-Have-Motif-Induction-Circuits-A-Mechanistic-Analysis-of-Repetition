"""Symbolic motif sampling and transforms (research plan §4).

A :class:`Motif` is a monophonic note list in beat time (onsets/durations in
beats, MIDI pitches). All transforms are pure functions of a
``numpy.random.Generator`` so stimulus construction is fully deterministic.

Rhythm grid: 0.25-beat resolution. Note durations never cross the next onset
(monophonic, non-overlapping), which keeps the ground-truth phi alignment
piecewise monotone.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import typing as tp

import numpy as np

__all__ = ["Motif", "SCALES", "sample_motif", "transpose", "rhythm_variation",
           "rhythm_scramble_pitches", "sample_gap_material",
           "shares_interval_ngrams", "motif_frames", "compute_phi"]

SCALES: tp.Dict[str, tp.Tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
}

GRID = 0.25  # beat quantization of onsets/durations
_PITCH_LO, _PITCH_HI = 48, 84  # C3..C6 melodic register


@dataclass(frozen=True)
class Motif:
    """A monophonic phrase in beat time."""

    pitches: tp.Tuple[int, ...]
    onsets_beats: tp.Tuple[float, ...]
    durations_beats: tp.Tuple[float, ...]

    def __post_init__(self):
        n = len(self.pitches)
        if not (n == len(self.onsets_beats) == len(self.durations_beats)):
            raise ValueError("pitches/onsets/durations length mismatch")
        if n == 0:
            raise ValueError("empty motif")
        o = np.asarray(self.onsets_beats)
        if np.any(np.diff(o) <= 0):
            raise ValueError("onsets must be strictly increasing")
        if np.any(np.asarray(self.durations_beats) <= 0):
            raise ValueError("durations must be positive")
        object.__setattr__(self, "pitches", tuple(int(p) for p in self.pitches))
        object.__setattr__(self, "onsets_beats",
                           tuple(float(x) for x in self.onsets_beats))
        object.__setattr__(self, "durations_beats",
                           tuple(float(x) for x in self.durations_beats))

    @property
    def n_notes(self) -> int:
        return len(self.pitches)

    @property
    def total_beats(self) -> float:
        return self.onsets_beats[-1] + self.durations_beats[-1]

    def to_json(self) -> dict:
        return {"pitches": list(self.pitches),
                "onsets_beats": list(self.onsets_beats),
                "durations_beats": list(self.durations_beats)}

    @classmethod
    def from_json(cls, d: dict) -> "Motif":
        return cls(tuple(d["pitches"]), tuple(d["onsets_beats"]),
                   tuple(d["durations_beats"]))


def scale_pitches(root: int, scale_type: str,
                  lo: int = _PITCH_LO, hi: int = _PITCH_HI) -> np.ndarray:
    """All MIDI pitches of a scale within ``[lo, hi]`` (ascending)."""
    if scale_type not in SCALES:
        raise ValueError(f"unknown scale {scale_type!r}")
    degrees = SCALES[scale_type]
    return np.asarray([p for p in range(lo, hi + 1)
                       if (p - root) % 12 in degrees], dtype=np.int64)


def _clip_durations(onsets: np.ndarray, durations: np.ndarray,
                    end_beat: float) -> np.ndarray:
    """Keep the phrase monophonic: durations end at the next onset / end."""
    next_onset = np.append(onsets[1:], end_beat)
    return np.minimum(durations, next_onset - onsets)


def sample_motif(rng: np.random.Generator, root: int, scale_type: str,
                 n_notes_range: tp.Tuple[int, int] = (4, 12),
                 n_bars: int = 1, beats_per_bar: int = 4) -> Motif:
    """Sample a motif A: 4-12 scale notes on a 0.25-beat grid over n bars.

    Pitches follow a bounded random walk over scale degrees (steps in
    [-3, 3] degrees, register clamped), which yields singable, motif-like
    contours instead of white-noise pitch sequences.
    """
    lo_n, hi_n = n_notes_range
    total = float(n_bars * beats_per_bar)
    n_slots = int(round(total / GRID))
    n_notes = int(rng.integers(lo_n, min(hi_n, n_slots) + 1))
    slots = np.sort(rng.choice(n_slots, size=n_notes, replace=False))
    if slots[0] != 0:  # anchor the motif at its first beat
        slots[0] = 0
        slots = np.unique(slots)
        while slots.size < n_notes:  # re-add clashed slots deterministically
            extra = rng.integers(1, n_slots)
            slots = np.unique(np.append(slots, extra))
    onsets = slots.astype(np.float64) * GRID
    durations = np.diff(np.append(onsets, total))
    sustain = rng.uniform(0.6, 1.0, size=n_notes)
    durations = np.maximum(durations * sustain, GRID / 2)
    durations = _clip_durations(onsets, durations, total)

    pitches_pool = scale_pitches(root, scale_type)
    center = int(rng.integers(len(pitches_pool) // 3, 2 * len(pitches_pool) // 3))
    idx = [center]
    for _ in range(n_notes - 1):
        step = int(rng.integers(-3, 4))
        idx.append(int(np.clip(idx[-1] + step, 0, len(pitches_pool) - 1)))
    pitches = pitches_pool[np.asarray(idx)]
    return Motif(tuple(pitches), tuple(onsets), tuple(durations))


def transpose(motif: Motif, semitones: int) -> Motif:
    """Transpose all pitches by ``semitones`` (rhythm untouched)."""
    return replace(motif,
                   pitches=tuple(int(p) + int(semitones) for p in motif.pitches))


def rhythm_variation(motif: Motif, rng: np.random.Generator,
                     mode: str = "syncopate") -> Motif:
    """Rewrite the rhythm while preserving the exact pitch sequence (S3).

    Modes
    -----
    ``augment``   : all onsets/durations doubled (half tempo feel).
    ``diminish``  : all onsets/durations halved (double tempo feel).
    ``syncopate`` : every other movable onset shifted by +-0.25 beats
                    (order preserved; first onset fixed; durations re-clipped).
    """
    onsets = np.asarray(motif.onsets_beats)
    durations = np.asarray(motif.durations_beats)
    if mode == "augment":
        onsets, durations = onsets * 2.0, durations * 2.0
    elif mode == "diminish":
        onsets, durations = onsets * 0.5, durations * 0.5
        durations = np.maximum(durations, GRID / 2)
    elif mode == "syncopate":
        onsets = onsets.copy()
        for i in range(1, len(onsets)):
            if i % 2 == 0:
                continue
            shift = float(rng.choice([-GRID, GRID]))
            lo = onsets[i - 1] + GRID / 2
            hi = onsets[i + 1] - GRID / 2 if i + 1 < len(onsets) else np.inf
            onsets[i] = float(np.clip(onsets[i] + shift, lo, hi))
    else:
        raise ValueError(f"unknown rhythm variation mode {mode!r}")
    end = float(onsets[-1] + durations[-1])
    durations = _clip_durations(onsets, durations, end)
    return Motif(motif.pitches, tuple(onsets), tuple(durations))


def rhythm_scramble_pitches(motif: Motif, rng: np.random.Generator,
                            root: int, scale_type: str,
                            min_changed: float = 0.8) -> Motif:
    """S5 negative control: keep A's rhythm skeleton, resample pitches.

    Pitches are drawn uniformly from the scale (same register pool) with the
    constraint that at least ``min_changed`` of the notes differ from A.
    """
    pool = scale_pitches(root, scale_type)
    n = motif.n_notes
    need = int(np.ceil(min_changed * n))
    for _ in range(64):
        pitches = pool[rng.integers(0, len(pool), size=n)]
        if int(np.sum(pitches != np.asarray(motif.pitches))) >= need:
            return Motif(tuple(pitches), motif.onsets_beats,
                         motif.durations_beats)
    # Deterministic fallback: force-change the first `need` notes.
    pitches = np.asarray(motif.pitches).copy()
    for i in range(need):
        choices = pool[pool != pitches[i]]
        pitches[i] = choices[int(rng.integers(0, len(choices)))]
    return Motif(tuple(pitches), motif.onsets_beats, motif.durations_beats)


def _interval_ngrams(pitches: tp.Sequence[int], n: int) -> tp.Set[tp.Tuple[int, ...]]:
    iv = np.diff(np.asarray(pitches))
    return {tuple(iv[i:i + n]) for i in range(len(iv) - n + 1)}


def shares_interval_ngrams(a: tp.Sequence[int], b: tp.Sequence[int],
                           n: int = 3) -> bool:
    """True when the two pitch sequences share any pitch-interval n-gram.

    This is the transposition-invariant "contains A's material" check used
    to keep the gap G free of motif content.
    """
    ga, gb = _interval_ngrams(a, n), _interval_ngrams(b, n)
    return bool(ga & gb)


def sample_gap_material(rng: np.random.Generator, root: int, scale_type: str,
                        n_beats: float, avoid: tp.Optional[Motif] = None,
                        max_tries: int = 32) -> Motif:
    """Random gap phrase G that avoids the motif's interval material.

    Rejection-samples phrases until none of A's pitch-interval 3-grams
    appears in G (transposition-invariant avoidance); after ``max_tries``
    the last candidate is returned with a warning-free best effort (the
    probability of exhaustion is negligible for 3-grams over a 7-note scale).
    """
    n_bars = max(1, int(round(n_beats / 4)))
    candidate = None
    for _ in range(max_tries):
        candidate = sample_motif(rng, root, scale_type,
                                 n_notes_range=(4, 10), n_bars=n_bars)
        if avoid is None or not shares_interval_ngrams(avoid.pitches,
                                                       candidate.pitches):
            return candidate
    return candidate  # best effort (see docstring)


# ----------------------------------------------------------------------
# frame-level ground truth
# ----------------------------------------------------------------------
def motif_frames(motif: Motif, bpm: float, frame_rate: int,
                 offset_s: float) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Note (onset, end) positions in absolute frames.

    Parameters
    ----------
    motif : Motif
    bpm : float
        Tempo (beats per minute).
    frame_rate : int
        Frames per second (50 for EnCodec).
    offset_s : float
        Absolute start time of the motif's beat 0, in seconds.

    Returns
    -------
    (onset_frames, end_frames) : np.ndarray int64
        Per note; ``end = onset + duration`` (monophonic, non-overlapping).
    """
    spb = 60.0 / bpm
    on = np.asarray(motif.onsets_beats) * spb + offset_s
    en = on + np.asarray(motif.durations_beats) * spb
    return (np.round(on * frame_rate).astype(np.int64),
            np.round(en * frame_rate).astype(np.int64))


def compute_phi(motif_a: Motif, motif_ap: Motif, bpm: float, frame_rate: int,
                offset_a_s: float, offset_ap_s: float
                ) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Ground-truth frame alignment phi: A' frames -> A frames.

    Uses the note-level correspondence (note i of A' <-> note i of A, true
    for every stimulus transform) with a piecewise-linear time map inside
    each note: A' frames within note i's span map proportionally onto A note
    i's span. Only frames WITHIN A' notes are covered (rests are skipped).

    Returns
    -------
    (a_prime_frames, a_frames) : np.ndarray int64
        Equal length; ``a_prime_frames`` strictly increasing.
    """
    if motif_a.n_notes != motif_ap.n_notes:
        raise ValueError("phi needs equal note counts (index correspondence)")
    on_a, en_a = motif_frames(motif_a, bpm, frame_rate, offset_a_s)
    on_p, en_p = motif_frames(motif_ap, bpm, frame_rate, offset_ap_s)
    ap_out: tp.List[int] = []
    a_out: tp.List[int] = []
    for i in range(motif_a.n_notes):
        span_p = max(int(en_p[i]) - int(on_p[i]), 1)
        span_a = max(int(en_a[i]) - int(on_a[i]), 1)
        for f in range(int(on_p[i]), int(on_p[i]) + span_p):
            rel = (f - int(on_p[i])) / span_p
            t_a = int(on_a[i]) + int(np.floor(rel * span_a))
            if ap_out and f <= ap_out[-1]:
                continue  # guard against rounding collisions between notes
            ap_out.append(f)
            a_out.append(t_a)
    return (np.asarray(ap_out, dtype=np.int64),
            np.asarray(a_out, dtype=np.int64))
