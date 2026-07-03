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
