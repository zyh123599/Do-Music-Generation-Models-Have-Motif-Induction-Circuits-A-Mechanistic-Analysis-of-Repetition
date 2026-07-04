"""Shared script plumbing: argparse, config, paths, metric aggregation.

Heavy imports (torch/audiocraft/librosa/matplotlib/soundfile) stay inside
functions so `--help` and unit tests never touch them.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import typing as tp

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motif_circuits.utils.config import load_config  # noqa: E402

logger = logging.getLogger("motif_circuits.scripts")


def make_parser(description: str, default_config: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / default_config),
                   help="YAML config (merged onto configs/default.yaml)")
    p.add_argument("--override", action="append", default=[],
                   metavar="KEY.PATH=VALUE",
                   help="config override, repeatable (YAML-typed values)")
    p.add_argument("--force", action="store_true",
                   help="recompute outputs that already exist")
    return p


def setup(args: argparse.Namespace) -> dict:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    cfg = load_config(args.config, args.override)
    logger.info("config: %s (+%d overrides)", args.config, len(args.override))
    return cfg


def data_root(cfg: dict) -> Path:
    p = Path(cfg["paths"]["data_root"])
    return p if p.is_absolute() else REPO_ROOT / p


def results_root(cfg: dict) -> Path:
    p = Path(cfg["paths"]["results_root"])
    base = p if p.is_absolute() else REPO_ROOT / p
    return base / cfg["model"]["size"]


def stimuli_dir(cfg: dict, category: str) -> Path:
    return data_root(cfg) / "stimuli" / category


def load_codes_np(category_dir: Path, sample_id: str) -> np.ndarray:
    """Load one sample's EnCodec codes ``[K, T]`` int (from scripts/02)."""
    path = category_dir / "codes" / f"{sample_id}.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing — run scripts/02_encode_stimuli.py first")
    return np.load(path)


def load_ranked_heads(screening_dir: Path,
                      allow_fallback: bool = False) -> tp.List[tp.Tuple[int, int]]:
    """Ranked heads for interventions: verified candidates, else fallback.

    Reads ``candidates.json`` (scripts/04). When it is EMPTY:

    * ``allow_fallback=False`` (default; real experiments): abort with an
      actionable message — causal claims require statistically verified
      candidate heads.
    * ``allow_fallback=True`` (pilot/plumbing runs only): fall back to the
      full excess ranking (``ranked_all.json``) with periodic heads removed,
      and warn loudly. Downstream numbers are then only good for validating
      that the pipeline runs, not for any scientific conclusion.
    """
    from motif_circuits.utils.io import load_json

    cands = load_json(screening_dir / "candidates.json")
    if cands:
        return [(int(c["layer"]), int(c["head"])) for c in cands]
    if not allow_fallback:
        raise SystemExit(
            f"{screening_dir / 'candidates.json'} is empty: no heads passed "
            "the significance criteria. Rerun scripts/03+04 at larger sample "
            "scale, set the explicit head list in the config, or — for "
            "pilot/plumbing runs ONLY — override "
            "<section>.allow_ranking_fallback=true")
    ranked_path = screening_dir / "ranked_all.json"
    if not ranked_path.is_file():
        raise SystemExit(f"{ranked_path} missing — rerun scripts/04 "
                         "(older runs predate the fallback ranking)")
    def _finite(x) -> bool:  # JSON stores NaN as null -> None
        return isinstance(x, (int, float)) and np.isfinite(x)

    entries = [e for e in load_json(ranked_path)
               if not e.get("periodic") and _finite(e.get("excess"))]
    logger.warning(
        "candidates.json is EMPTY — falling back to the raw excess ranking "
        "(%d non-periodic heads). Pipeline-validation mode: downstream "
        "results are NOT scientifically meaningful.", len(entries))
    return [(int(e["layer"]), int(e["head"])) for e in entries]


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 rng: tp.Optional[np.random.Generator] = None
                 ) -> tp.Tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap CI over 1-D values (NaN-dropped)."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    boots = np.array([values[rng.integers(0, len(values), len(values))].mean()
                      for _ in range(n_boot)])
    return (float(values.mean()),
            float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


def evaluate_generation_sample(wav: np.ndarray, sr: int, codes: np.ndarray,
                               cfg_metrics: dict,
                               prompt_a_chroma: tp.Optional[np.ndarray] = None,
                               prompt_frames: int = 0) -> dict:
    """Structural metrics for one generated sample (shared by scripts 07/08).

    Parameters
    ----------
    wav : np.ndarray
        Mono waveform of the full generation (prompt included).
    sr : int
    codes : np.ndarray
        ``[K, T]`` frame-aligned codes of the generation.
    cfg_metrics : dict
        ``{stripe_min_lag, stripe_max_lag, taus?}``.
    prompt_a_chroma : np.ndarray, optional
        Chroma of the motif-A part of the prompt; enables recurrence scoring.
    prompt_frames : int
        Continuation starts at this frame (prompt excluded from recurrence).
    """
    from motif_circuits.metrics import (chroma_ssm, stripe_energy,
                                        foote_novelty, novelty_contrast,
                                        loop_score, motif_recurrence)
    from motif_circuits.utils.chroma import chroma_features

    chroma = chroma_features(np.asarray(wav).reshape(-1), sr)
    ssm = chroma_ssm(chroma)
    out: dict = {
        "stripe_energy": float(stripe_energy(
            ssm, int(cfg_metrics["stripe_min_lag"]),
            int(cfg_metrics["stripe_max_lag"]))),
        "novelty_contrast": float(novelty_contrast(foote_novelty(ssm))),
    }
    lr = loop_score(codes, np.asarray(wav).reshape(-1), sr)
    out.update({"ngram_rate": lr.ngram_rate, "autocorr_peak": lr.autocorr_peak,
                "is_loop": bool(lr.is_loop)})
    if prompt_a_chroma is not None and prompt_a_chroma.shape[0] > 0:
        cont = chroma[prompt_frames:]
        if cont.shape[0] >= prompt_a_chroma.shape[0]:
            taus = np.asarray(cfg_metrics.get(
                "taus", [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]))
            rec = motif_recurrence(prompt_a_chroma, cont, taus=taus)
            out["recurrence_best_corr"] = rec.best_corr
            out["recurrence_hits"] = {str(k): bool(v)
                                      for k, v in rec.hits.items()}
    return out
