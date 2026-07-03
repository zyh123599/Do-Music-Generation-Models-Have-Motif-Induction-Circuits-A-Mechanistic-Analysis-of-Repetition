"""Stimulus-set construction: S1-S6 categories, manifests, patching pairs.

Sample structure (10 s @ 32 kHz = 500 frames @ 50 Hz)::

    [0.2 s lead-in] A (motif, 1-2 bars) | G (gap) | A' (transform) [>=0.25 s tail]

Category transforms (research plan §4): S1 exact, S2 transpose ±{3,5,7},
S3 rhythm variation (pitch sequence kept), S4 timbre change (same notes,
different GM program), S5 rhythm-preserved pitch scramble (negative control),
S6 unrelated phrase (no repetition; null corpus + patching partners).

Every S1/S2/S3 sample gets a paired corrupted sample (same A + G, unrelated
continuation) written into the S6 directory as ``S6pair_<clean_id>`` and
referenced by ``pair_id`` — the clean/corrupted pairs of the activation
patching protocol (research plan §6.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import typing as tp

import numpy as np

from .motif import (Motif, compute_phi, rhythm_scramble_pitches,
                    rhythm_variation, sample_gap_material, sample_motif,
                    transpose)
from .midi_render import build_midi, render_midi

__all__ = ["StimulusRecord", "load_manifest", "save_manifest",
           "build_category", "build_all", "CATEGORIES", "PROGRAMS"]

logger = logging.getLogger(__name__)

CATEGORIES = ("S1", "S2", "S3", "S4", "S5", "S6")
#: GM programs: piano, e-piano, guitar, bass, strings, brass, synth lead.
PROGRAMS = (0, 4, 25, 33, 48, 61, 80)
TRANSPOSES = (-7, -5, -3, 3, 5, 7)
RHYTHM_MODES = ("augment", "diminish", "syncopate")

DURATION_S = 10.0
LEAD_IN_S = 0.2
TAIL_S = 0.25
FRAME_RATE = 50
SAMPLE_RATE = 32000


@dataclass
class StimulusRecord:
    """One manifest line (see docs/interfaces.md §1)."""

    id: str
    category: str
    sr: int
    frame_rate: int
    bpm: float
    key_root: int
    scale: str
    program: int
    velocity: int
    seed: int
    render_seed: int
    motif: dict
    transform: dict
    segments: tp.Dict[str, tp.List[int]]
    phi: tp.Optional[tp.Dict[str, tp.List[int]]]
    wav_path: str
    midi_path: str
    pair_id: tp.Optional[str] = None
    duration_s: float = DURATION_S

    def to_json(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_json(cls, d: dict) -> "StimulusRecord":
        return cls(**d)

    @property
    def phi_arrays(self) -> tp.Optional[tp.Tuple[np.ndarray, np.ndarray]]:
        if self.phi is None:
            return None
        return (np.asarray(self.phi["a_prime_frames"], dtype=np.int64),
                np.asarray(self.phi["a_frames"], dtype=np.int64))


def save_manifest(records: tp.Sequence[StimulusRecord],
                  path: tp.Union[str, Path]) -> None:
    """Write one JSON record per line (jsonl)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r.to_json()) + "\n")


def load_manifest(path: tp.Union[str, Path]) -> tp.List[StimulusRecord]:
    """Read a jsonl manifest back into records."""
    out = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(StimulusRecord.from_json(json.loads(line)))
    return out


# ----------------------------------------------------------------------
# layout planning
# ----------------------------------------------------------------------
@dataclass
class _Layout:
    bpm: float
    n_bars: int
    gap_beats: float
    rhythm_mode: str  # only used by S3
    a_start_s: float
    g_start_s: float
    ap_start_s: float
    ap_motif_scale: float  # A' duration multiplier (S3 augment/diminish)


def _plan_layout(rng: np.random.Generator, category: str,
                 bpm_range: tp.Tuple[float, float]) -> _Layout:
    """Choose bpm/bars/gap so A + G + A' (+ tail) fits in 10 s.

    Tries 2 bars first, then 1 bar; for S3 the rhythm mode is chosen among
    the ones that fit (augment doubles A' duration). Falls back to faster
    bpm resampling when nothing fits (rare at 80 BPM + 2 bars).
    """
    budget = DURATION_S - LEAD_IN_S - TAIL_S
    for _ in range(64):
        bpm = float(rng.uniform(*bpm_range))
        spb = 60.0 / bpm
        modes = list(RHYTHM_MODES) if category == "S3" else ["syncopate"]
        rng.shuffle(modes)
        for n_bars in (2, 1):
            beats_a = 4.0 * n_bars
            for mode in modes:
                scale_ap = {"augment": 2.0, "diminish": 0.5}.get(
                    mode if category == "S3" else "", 1.0)
                gap_beats = float(rng.uniform(2.0, 6.0))
                need = (beats_a + gap_beats + beats_a * scale_ap) * spb
                if need <= budget:
                    a_start = LEAD_IN_S
                    g_start = a_start + beats_a * spb
                    ap_start = g_start + gap_beats * spb
                    return _Layout(bpm, n_bars, gap_beats, mode, a_start,
                                   g_start, ap_start, scale_ap)
    raise RuntimeError("could not fit stimulus layout into 10 s")


def _seconds_to_frames(s: float) -> int:
    return int(round(s * FRAME_RATE))


def _segments(layout: _Layout, spb: float) -> tp.Dict[str, tp.List[int]]:
    beats_a = 4.0 * layout.n_bars
    a0 = _seconds_to_frames(layout.a_start_s)
    g0 = _seconds_to_frames(layout.g_start_s)
    p0 = _seconds_to_frames(layout.ap_start_s)
    p1 = _seconds_to_frames(layout.ap_start_s
                            + beats_a * layout.ap_motif_scale * spb)
    total = _seconds_to_frames(DURATION_S)
    return {"A": [a0, g0], "G": [g0, p0], "A_prime": [p0, min(p1, total)]}


# ----------------------------------------------------------------------
# per-category construction
# ----------------------------------------------------------------------
def _make_ap(category: str, motif_a: Motif, layout: _Layout,
             rng: np.random.Generator, key_root: int, scale: str
             ) -> tp.Tuple[Motif, dict]:
    """A' motif + transform metadata for a category."""
    if category == "S1":
        return motif_a, {"type": "exact"}
    if category == "S2":
        k = int(rng.choice(TRANSPOSES))
        return transpose(motif_a, k), {"type": "transpose", "semitones": k}
    if category == "S3":
        mode = layout.rhythm_mode
        return (rhythm_variation(motif_a, rng, mode),
                {"type": "rhythm", "mode": mode})
    if category == "S4":
        return motif_a, {"type": "timbre"}  # program_b filled by caller
    if category == "S5":
        return (rhythm_scramble_pitches(motif_a, rng, key_root, scale),
                {"type": "rhythm_scramble"})
    if category == "S6":
        beats = 4.0 * layout.n_bars * layout.ap_motif_scale
        return (sample_gap_material(rng, key_root, scale, beats,
                                    avoid=motif_a),
                {"type": "none"})
    raise ValueError(f"unknown category {category!r}")


def _build_one(category: str, index: int, cfg: dict, out_dir: Path,
               rng: np.random.Generator, render: bool,
               pair_out_dir: tp.Optional[Path]
               ) -> tp.Tuple[tp.List[StimulusRecord], tp.List[StimulusRecord]]:
    """Build one logical sample (all render seeds); returns (records, pairs)."""
    seed = int(rng.integers(0, 2 ** 31 - 1))
    srng = np.random.default_rng(seed)

    key_root = int(srng.integers(0, 12))
    scale = str(srng.choice(["major", "minor"]))
    program = int(srng.choice(cfg.get("programs", PROGRAMS)))
    velocity = int(srng.integers(70, 111))
    layout = _plan_layout(srng, category, tuple(cfg.get("bpm", (80.0, 140.0))))
    spb = 60.0 / layout.bpm

    motif_a = sample_motif(srng, key_root, scale, n_bars=layout.n_bars)
    motif_ap, transform = _make_ap(category, motif_a, layout, srng,
                                   key_root, scale)
    program_ap = program
    if category == "S4":
        others = [p for p in cfg.get("programs", PROGRAMS) if p != program]
        program_ap = int(srng.choice(others))
        transform["program_b"] = program_ap
    gap_motif = sample_gap_material(srng, key_root, scale, layout.gap_beats,
                                    avoid=motif_a)
    segments = _segments(layout, spb)

    phi = None
    if category != "S6":
        ap_f, a_f = compute_phi(motif_a, motif_ap, layout.bpm, FRAME_RATE,
                                layout.a_start_s, layout.ap_start_s)
        total = _seconds_to_frames(DURATION_S)
        keep = ap_f < total
        phi = {"a_prime_frames": ap_f[keep].tolist(),
               "a_frames": a_f[keep].tolist()}

    sections = [(motif_a, layout.a_start_s / spb, program),
                (gap_motif, layout.g_start_s / spb, program),
                (motif_ap, layout.ap_start_s / spb, program_ap)]

    # optional patching partner: same A + G, unrelated continuation
    pair_needed = category in ("S1", "S2", "S3") and pair_out_dir is not None
    pair_sections = None
    if pair_needed:
        beats_ap = 4.0 * layout.n_bars * layout.ap_motif_scale
        unrelated = sample_gap_material(srng, key_root, scale, beats_ap,
                                        avoid=motif_a)
        pair_sections = [(motif_a, layout.a_start_s / spb, program),
                         (gap_motif, layout.g_start_s / spb, program),
                         (unrelated, layout.ap_start_s / spb, program)]

    records: tp.List[StimulusRecord] = []
    pairs: tp.List[StimulusRecord] = []
    n_render_seeds = int(cfg.get("render_seeds", 3))
    for j in range(n_render_seeds):
        sample_id = f"{category}_{index:06d}_r{j}"
        pair_id = f"S6pair_{sample_id}" if pair_needed else None
        jitter = np.random.default_rng((seed, j))
        gain = 0.7 * float(1.0 + 0.05 * (jitter.uniform(-1, 1)))

        def _emit(sid: str, secs, cat: str, tform: dict, phi_: tp.Optional[dict],
                  dest: Path, pid: tp.Optional[str]) -> StimulusRecord:
            midi_rel = f"midi/{sid}.mid"
            wav_rel = f"audio/{sid}.wav"
            build_midi(secs, layout.bpm, dest / midi_rel, velocity=velocity,
                       velocity_jitter=np.random.default_rng((seed, j, 7)))
            if render:
                render_midi(dest / midi_rel, dest / wav_rel, sr=SAMPLE_RATE,
                            soundfont=cfg.get("soundfont"), gain=gain,
                            duration_s=DURATION_S)
            return StimulusRecord(
                id=sid, category=cat, sr=SAMPLE_RATE, frame_rate=FRAME_RATE,
                bpm=layout.bpm, key_root=key_root, scale=scale,
                program=program, velocity=velocity, seed=seed, render_seed=j,
                motif=motif_a.to_json(), transform=tform, segments=segments,
                phi=phi_, wav_path=wav_rel, midi_path=midi_rel, pair_id=pid)

        records.append(_emit(sample_id, sections, category, transform, phi,
                             out_dir, pair_id))
        if pair_needed:
            pairs.append(_emit(pair_id, pair_sections, "S6",
                               {"type": "none", "pair_of": sample_id}, None,
                               pair_out_dir, sample_id))
    return records, pairs


def build_category(cat: str, cfg: dict, out_dir: tp.Union[str, Path],
                   rng: np.random.Generator, render: bool = True,
                   pair_out_dir: tp.Optional[tp.Union[str, Path]] = None
                   ) -> tp.Tuple[tp.List[StimulusRecord], tp.List[StimulusRecord]]:
    """Build one category; returns (records, patching-pair records).

    Pair records (S1/S2/S3 partners) are written under ``pair_out_dir`` (the
    S6 directory) but NOT added to any manifest here — ``build_all`` merges
    them into the S6 manifest.
    """
    if cat not in CATEGORIES:
        raise ValueError(f"unknown category {cat!r}")
    out_dir = Path(out_dir)
    pair_dir = Path(pair_out_dir) if pair_out_dir is not None else None
    n = int(cfg.get("n_per_category", 100))
    records, pairs = [], []
    for i in range(n):
        recs, prs = _build_one(cat, i, cfg, out_dir, rng, render, pair_dir)
        records.extend(recs)
        pairs.extend(prs)
    save_manifest(records, out_dir / "manifest.jsonl")
    (out_dir / "meta.json").write_text(json.dumps(
        {"category": cat, "n_logical": n,
         "render_seeds": int(cfg.get("render_seeds", 3)),
         "rendered": bool(render), "config": {k: v for k, v in cfg.items()
                                              if k != "soundfont" or v}},
        indent=2, default=str))
    logger.info("built %s: %d records (%d pairs)", cat, len(records),
                len(pairs))
    return records, pairs


def build_all(cfg: dict, out_root: tp.Union[str, Path],
              render: bool = True) -> tp.Dict[str, int]:
    """Build every configured category under ``out_root`` (data/stimuli).

    S1/S2/S3 patching partners are written into the S6 directory and
    appended to the S6 manifest. Returns ``category -> record count``.
    """
    out_root = Path(out_root)
    categories = list(cfg.get("categories", CATEGORIES))
    master = np.random.default_rng(int(cfg.get("seed", 0)))
    # one child seed per category, drawn in fixed order for determinism
    child_seeds = {c: int(master.integers(0, 2 ** 31 - 1)) for c in CATEGORIES}
    s6_dir = out_root / "S6"
    all_pairs: tp.List[StimulusRecord] = []
    counts: tp.Dict[str, int] = {}
    for cat in categories:
        if cat == "S6":
            continue
        recs, pairs = build_category(
            cat, cfg, out_root / cat, np.random.default_rng(child_seeds[cat]),
            render=render, pair_out_dir=s6_dir)
        all_pairs.extend(pairs)
        counts[cat] = len(recs)
    if "S6" in categories:
        recs, _ = build_category(
            "S6", cfg, s6_dir, np.random.default_rng(child_seeds["S6"]),
            render=render, pair_out_dir=None)
        merged = recs + all_pairs
        save_manifest(merged, s6_dir / "manifest.jsonl")
        counts["S6"] = len(merged)
    elif all_pairs:
        s6_dir.mkdir(parents=True, exist_ok=True)
        save_manifest(all_pairs, s6_dir / "manifest.jsonl")
        counts["S6"] = len(all_pairs)
    logger.info("stimulus set complete: %s", counts)
    return counts
