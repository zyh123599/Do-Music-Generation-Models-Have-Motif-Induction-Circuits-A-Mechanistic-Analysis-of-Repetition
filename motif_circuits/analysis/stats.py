"""Permutation tests, phi permutations and multiple-comparison correction.

Candidate-head criteria (research plan section 5.0): a head must (a) score
significantly above its lag null under a permutation test with BH-FDR
correction across all heads, and (b) NOT be significant on the S5 negative
control. The permutation family used throughout is *global lag shifts* of the
alignment phi: the a_frames vector is circularly shifted within the A segment
while a_prime_frames stays fixed, which preserves the pair count, the query
positions and the within-segment lag distribution's support.
"""
from __future__ import annotations

import logging
import typing as tp

import numpy as np

__all__ = ["permutation_test", "permuted_phis", "benjamini_hochberg",
           "head_table"]

logger = logging.getLogger(__name__)

PhiTuple = tp.Tuple[np.ndarray, np.ndarray]


def _phi_arrays(phi) -> PhiTuple:
    if hasattr(phi, "a_prime_frames") and hasattr(phi, "a_frames"):
        ap, a = phi.a_prime_frames, phi.a_frames
    else:
        ap, a = phi
    return (np.asarray(ap, dtype=np.int64), np.asarray(a, dtype=np.int64))


def permutation_test(observed: np.ndarray,
                     null_samples: np.ndarray) -> np.ndarray:
    """One-sided permutation p-values per head.

    Parameters
    ----------
    observed : np.ndarray
        Observed statistic per head, ``[H]``.
    null_samples : np.ndarray
        Statistic under the permutation null, ``[n_perm, H]``.

    Returns
    -------
    np.ndarray
        ``[H]`` p-values ``(1 + #{null >= observed}) / (n_perm + 1)``
        (add-one correction; never exactly 0). NaN observed values or NaN
        null columns yield NaN p-values (NaN comparisons count as
        non-exceedances, so heads with partially-NaN nulls still get a
        conservative finite p when ``observed`` is finite).
    """
    observed = np.asarray(observed, dtype=np.float64)
    null_samples = np.asarray(null_samples, dtype=np.float64)
    if null_samples.ndim != 2 or observed.ndim != 1 \
            or null_samples.shape[1] != observed.shape[0]:
        raise ValueError("expected observed [H] and null_samples [n_perm, H]")
    n_perm = null_samples.shape[0]
    with np.errstate(invalid="ignore"):
        exceed = (null_samples >= observed[None, :]).sum(axis=0)
    p = (1.0 + exceed) / (n_perm + 1.0)
    p = np.where(np.isfinite(observed), p, np.nan)
    return p


def permuted_phis(phi, rng: np.random.Generator, n: int,
                  t_min: int, t_max: int) -> tp.Iterator[PhiTuple]:
    """Random global lag shifts of an alignment (the permutation family).

    Each permutation shifts the whole ``a_frames`` vector by a random nonzero
    offset, wrapping within the half-open A segment ``[t_min, t_max)``::

        a' = t_min + (a - t_min + delta) % (t_max - t_min)

    ``a_prime_frames`` stays fixed, so the permuted alignment has the same
    query frames and pair count but a destroyed A' -> A correspondence.
    Note the wrap makes ``a_frames`` piecewise monotone (one wrap point).

    Parameters
    ----------
    phi : AlignResult or (a_prime_frames, a_frames)
        The true alignment.
    rng : np.random.Generator
        Source of randomness.
    n : int
        Number of permutations to yield.
    t_min, t_max : int
        Half-open frame bounds of the A segment.

    Yields
    ------
    (a_prime_frames, shifted_a_frames)
        ``n`` permuted alignments.
    """
    ap, a = _phi_arrays(phi)
    span = int(t_max) - int(t_min)
    if span < 2:
        raise ValueError(f"A segment [{t_min}, {t_max}) too short to permute")
    if np.any((a < t_min) | (a >= t_max)):
        raise ValueError("a_frames fall outside the given A segment bounds")
    for _ in range(int(n)):
        delta = int(rng.integers(1, span))  # nonzero shift
        shifted = t_min + (a - t_min + delta) % span
        yield ap, shifted


def benjamini_hochberg(pvals: np.ndarray,
                       q: float = 0.05) -> tp.Tuple[np.ndarray, float]:
    """Benjamini-Hochberg FDR control.

    Parameters
    ----------
    pvals : np.ndarray
        Flat array of p-values; NaNs are excluded from the procedure and
        marked non-significant.
    q : float
        Target false-discovery rate.

    Returns
    -------
    (mask, threshold)
        Boolean array of the same shape (True = rejected null = significant)
        and the p-value threshold actually applied (0.0 when nothing is
        rejected).
    """
    pvals = np.asarray(pvals, dtype=np.float64)
    flat = pvals.reshape(-1)
    finite = np.isfinite(flat)
    m = int(finite.sum())
    mask = np.zeros(flat.shape, dtype=bool)
    threshold = 0.0
    if m > 0:
        p_f = flat[finite]
        order = np.argsort(p_f)
        ranked = p_f[order]
        crit = q * (np.arange(1, m + 1) / m)
        below = np.flatnonzero(ranked <= crit)
        if below.size:
            k = below[-1]
            threshold = float(ranked[k])
            sig_f = p_f <= threshold
            mask[np.flatnonzero(finite)] = sig_f
    logger.debug("BH-FDR: %d/%d significant at q=%.3g (threshold %.4g)",
                 int(mask.sum()), m, q, threshold)
    return mask.reshape(pvals.shape), threshold


def head_table(scores: tp.Dict[str, np.ndarray]) -> tp.List[dict]:
    """Flatten per-head [L, H] score arrays into a JSON-friendly table.

    Parameters
    ----------
    scores : dict[str, np.ndarray]
        Named score arrays, all ``[L, H]`` (bool arrays allowed).

    Returns
    -------
    list of dict
        One dict per (layer, head): ``{'layer', 'head', <name>: value, ...}``,
        row-major order. NaNs become ``None`` for clean JSON.
    """
    if not scores:
        return []
    shapes = {np.asarray(v).shape for v in scores.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError(f"all score arrays must share one [L, H] shape, got {shapes}")
    L, H = next(iter(shapes))

    def _cell(v):
        if isinstance(v, (np.bool_, bool)):
            return bool(v)
        v = float(v)
        return v if np.isfinite(v) else None

    table = []
    for layer in range(L):
        for head in range(H):
            row: dict = {"layer": layer, "head": head}
            for name, arr in scores.items():
                row[name] = _cell(np.asarray(arr)[layer, head])
            table.append(row)
    return table
