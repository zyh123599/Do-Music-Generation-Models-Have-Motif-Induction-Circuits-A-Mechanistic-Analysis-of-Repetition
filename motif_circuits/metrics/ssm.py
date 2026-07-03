"""Chroma self-similarity metrics: SSM, lag stripes and Foote novelty.

Output-side structural-repetition measures (research plan section 7,
"structural repetition score"). Everything here operates on frame-level
chroma features ``[T, 12]`` (50 Hz frames by repo convention) and is pure
NumPy so it imports everywhere.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "chroma_ssm",
    "lag_profile",
    "stripe_energy",
    "foote_novelty",
    "novelty_contrast",
]

_EPS = 1e-12


def _normalize_rows(x: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """L2-normalize rows; all-zero rows stay zero instead of dividing by 0."""
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, eps)


def chroma_ssm(chroma: np.ndarray) -> np.ndarray:
    """Cosine self-similarity matrix of a chroma sequence.

    Parameters
    ----------
    chroma : np.ndarray
        ``[T, 12]`` chroma features (need not be pre-normalized).

    Returns
    -------
    np.ndarray
        ``[T, T]`` cosine self-similarity. Rows of ``chroma`` are
        L2-normalized with a zero-guard: an all-zero (silent) frame has
        similarity 0 with every frame, including itself.
    """
    chroma = np.asarray(chroma, dtype=np.float64)
    if chroma.ndim != 2:
        raise ValueError(f"chroma must be 2-D [T, 12], got shape {chroma.shape}")
    normed = _normalize_rows(chroma)
    return normed @ normed.T


def lag_profile(ssm: np.ndarray) -> np.ndarray:
    """Mean sub-diagonal similarity per lag.

    Parameters
    ----------
    ssm : np.ndarray
        ``[T, T]`` self-similarity matrix.

    Returns
    -------
    np.ndarray
        ``[T]`` array where entry ``l`` is ``mean_t ssm[t, t - l]`` over the
        valid queries ``t in [l, T)``. Entry 0 is the diagonal mean (1.0 for
        non-silent, normalized chroma). Large lags average few (t, t-l)
        pairs and are correspondingly noisy.
    """
    ssm = np.asarray(ssm, dtype=np.float64)
    if ssm.ndim != 2 or ssm.shape[0] != ssm.shape[1]:
        raise ValueError(f"ssm must be square [T, T], got shape {ssm.shape}")
    T = ssm.shape[0]
    profile = np.empty(T, dtype=np.float64)
    for lag in range(T):
        profile[lag] = np.mean(np.diagonal(ssm, offset=-lag))
    return profile


def stripe_energy(ssm: np.ndarray, min_lag: int, max_lag: int) -> float:
    """Off-diagonal stripe energy: lag-domain peak prominence of the SSM.

    The lag profile is lightly smoothed with a width-3 moving average and the
    stripe energy is ``max - median`` of the smoothed profile over the
    inclusive lag range ``[min_lag, max_lag]``. A strong repetition stripe at
    some lag yields a peak well above the profile's typical (median) level.

    Comparability across durations
    ------------------------------
    Each lag-profile entry is a *mean* of cosine similarities (bounded in
    [-1, 1]), so the profile — and hence ``max - median`` — does not scale
    with clip length; scores are comparable across generations of different
    durations as long as the same frame-lag range ``[min_lag, max_lag]`` is
    used. Longer clips only reduce estimator variance (more (t, t-l) pairs
    per lag). Keep ``max_lag`` well below ``T`` so every lag in the range
    still averages many pairs.

    Parameters
    ----------
    ssm : np.ndarray
        ``[T, T]`` self-similarity matrix (see :func:`chroma_ssm`).
    min_lag, max_lag : int
        Inclusive lag range (frames) to search. Use ``min_lag >= 2`` so the
        smoothed diagonal (lag 0) does not leak into the range.

    Returns
    -------
    float
        Peak prominence ``max - median`` of the smoothed lag profile within
        the range. Near 0 for structureless audio.
    """
    profile = lag_profile(ssm)
    T = profile.shape[0]
    if min_lag < 0 or min_lag > max_lag:
        raise ValueError(f"need 0 <= min_lag <= max_lag, got [{min_lag}, {max_lag}]")
    max_lag = min(max_lag, T - 1)
    if min_lag > max_lag:
        raise ValueError(f"min_lag={min_lag} exceeds largest available lag {T - 1}")
    # Light smoothing (moving average, width 3) to absorb +-1-frame jitter of
    # the repetition lag without erasing the peak.
    smoothed = np.convolve(profile, np.ones(3) / 3.0, mode="same")
    window = smoothed[min_lag:max_lag + 1]
    return float(np.max(window) - np.median(window))


def foote_novelty(ssm: np.ndarray, kernel_size: int = 16) -> np.ndarray:
    """Foote novelty curve via a Gaussian-tapered checkerboard kernel.

    A checkerboard kernel (positive on the past-past / future-future
    quadrants, negative on the cross quadrants, Gaussian taper away from the
    center) is slid along the main diagonal of the SSM. The response is high
    where two internally homogeneous but mutually dissimilar blocks meet,
    i.e. at segment boundaries.

    Parameters
    ----------
    ssm : np.ndarray
        ``[T, T]`` self-similarity matrix.
    kernel_size : int
        Full side length of the checkerboard kernel in frames (default 16 =
        320 ms at 50 Hz). Even sizes place the checkerboard crossing exactly
        between two frames, so a boundary starting at frame ``b`` peaks at
        ``novelty[b]``.

    Returns
    -------
    np.ndarray
        ``[T]`` novelty curve. Edges are handled by replicate-padding the
        SSM, so homogeneous edges score ~0 rather than spiking. The kernel
        is normalized by its total absolute weight, making values comparable
        across kernel sizes (roughly within [-1, 1] for a cosine SSM).
    """
    ssm = np.asarray(ssm, dtype=np.float64)
    if ssm.ndim != 2 or ssm.shape[0] != ssm.shape[1]:
        raise ValueError(f"ssm must be square [T, T], got shape {ssm.shape}")
    if kernel_size < 2:
        raise ValueError("kernel_size must be >= 2")
    T = ssm.shape[0]
    L = int(kernel_size)
    # Centered coordinates; for even L there is no zero entry and the sign
    # crossing sits between the two central frames.
    idx = np.arange(L) - (L - 1) / 2.0
    sign = np.sign(np.outer(idx, idx))          # +1 on diagonal quadrants
    taper1d = np.exp(-(idx ** 2) / (2.0 * (L / 4.0) ** 2))
    kernel = sign * np.outer(taper1d, taper1d)
    kernel /= np.sum(np.abs(kernel))
    m = L // 2
    padded = np.pad(ssm, ((m, L - m), (m, L - m)), mode="edge")
    novelty = np.empty(T, dtype=np.float64)
    for t in range(T):
        novelty[t] = np.sum(kernel * padded[t:t + L, t:t + L])
    return novelty


def novelty_contrast(novelty: np.ndarray, eps: float = 1e-8) -> float:
    """Peak-to-median contrast of a novelty curve.

    Summarizes how clearly boundaries stand out from the baseline novelty
    level: ``max(novelty) / max(median(novelty), eps)``. The median is
    floored at ``eps`` so flat or negative-median curves do not divide by
    zero or flip sign; a flat all-zero curve scores 0.

    Parameters
    ----------
    novelty : np.ndarray
        ``[T]`` novelty curve from :func:`foote_novelty`.
    eps : float
        Guard floor for the median denominator.

    Returns
    -------
    float
        Peak / median contrast (dimensionless, larger = clearer structure).
    """
    novelty = np.asarray(novelty, dtype=np.float64)
    if novelty.size == 0:
        return 0.0
    peak = float(np.max(novelty))
    med = float(np.median(novelty))
    return peak / max(med, eps)
