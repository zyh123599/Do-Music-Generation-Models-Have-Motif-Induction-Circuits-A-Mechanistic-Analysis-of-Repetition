"""Frame-level alignment phi between a motif segment A and its recurrence A'.

All aligners return an :class:`AlignResult` whose ``a_prime_frames[i]``
(absolute frame in A') corresponds to ``a_frames[i]`` (absolute frame in A).
Segments are half-open frame intervals ``(start, end)`` in absolute 50 Hz
frame coordinates, matching the repo-wide convention.

Feature arrays may be passed either as full-track features (sliced here with
the segment bounds) or as pre-sliced per-segment features whose length equals
the segment length.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import typing as tp

import numpy as np
import scipy.signal

__all__ = ["AlignResult", "align_identity", "align_transposition_invariant",
           "align_dtw"]

logger = logging.getLogger(__name__)

Segment = tp.Tuple[int, int]


@dataclass
class AlignResult:
    """Frame alignment between A' and A.

    Attributes
    ----------
    a_prime_frames : np.ndarray
        Absolute frame indices in the A' segment, ``int64 [N]``.
    a_frames : np.ndarray
        Corresponding absolute frame indices in the A segment, ``int64 [N]``.
    meta : dict
        Method-specific metadata (e.g. ``shift``, ``corr``, ``path_cost``).
    """

    a_prime_frames: np.ndarray
    a_frames: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.a_prime_frames = np.asarray(self.a_prime_frames, dtype=np.int64)
        self.a_frames = np.asarray(self.a_frames, dtype=np.int64)
        if self.a_prime_frames.shape != self.a_frames.shape:
            raise ValueError("a_prime_frames and a_frames must have equal length")


def _check_segment(seg: Segment) -> tp.Tuple[int, int]:
    s, e = int(seg[0]), int(seg[1])
    if e <= s or s < 0:
        raise ValueError(f"invalid half-open segment {seg}")
    return s, e


def _segment_features(feat: np.ndarray, seg: Segment, name: str) -> np.ndarray:
    """Slice full-track features to a segment, or pass through pre-sliced ones.

    ``feat`` is treated as segment-local when its length equals the segment
    length; otherwise it must cover the segment (``len >= seg end``).
    """
    s, e = _check_segment(seg)
    feat = np.asarray(feat, dtype=np.float64)
    if feat.ndim != 2:
        raise ValueError(f"{name} must be [T, d]")
    if feat.shape[0] == e - s:
        return feat
    if feat.shape[0] >= e:
        return feat[s:e]
    raise ValueError(
        f"{name} has {feat.shape[0]} frames; expected the segment length "
        f"{e - s} or full-track coverage >= {e}")


def _l2_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.where(norms > eps, x / np.maximum(norms, eps), 0.0)


def align_identity(seg_a: Segment, seg_ap: Segment) -> AlignResult:
    """Linear (identity-like) alignment between two segments.

    Equal-length segments map frame-for-frame; unequal lengths are mapped
    proportionally (endpoints to endpoints, rounded linear interpolation).

    Parameters
    ----------
    seg_a, seg_ap : tuple of int
        Half-open ``(start, end)`` frame intervals of A and A'.

    Returns
    -------
    AlignResult
        One entry per A' frame; ``meta = {'method': 'identity'}``.
    """
    a0, a1 = _check_segment(seg_a)
    ap0, ap1 = _check_segment(seg_ap)
    n_a, n_ap = a1 - a0, ap1 - ap0
    ap_frames = np.arange(ap0, ap1, dtype=np.int64)
    if n_ap == 1:
        a_local = np.zeros(1)
    else:
        a_local = np.arange(n_ap) * (n_a - 1) / (n_ap - 1)
    a_frames = a0 + np.round(a_local).astype(np.int64)
    return AlignResult(ap_frames, a_frames, {"method": "identity"})


def align_transposition_invariant(chroma_a: np.ndarray, chroma_ap: np.ndarray,
                                  seg_a: Segment, seg_ap: Segment) -> AlignResult:
    """Transposition-invariant chroma alignment (S2 stimuli).

    For each of the 12 circular pitch-class shifts of the A' chroma, the
    shifted A' features are cross-correlated with the A features over time
    (FFT-based, rows L2-normalized, correlation normalized by the overlap
    length); the best ``(shift, lag)`` pair defines a rigid frame map.

    Sign convention (tested): ``meta['shift']`` is the signed number of
    semitones, folded to ``[-5, 6]``, by which **A' is ABOVE A**. I.e. if A'
    plays the motif transposed up by ``k`` semitones, ``meta['shift'] == k``;
    equivalently ``np.roll(chroma_ap, -k, axis=1)`` matches ``chroma_a``.

    Lags whose A/A' overlap is shorter than half the shorter segment are
    excluded to avoid spurious short-overlap peaks.

    Parameters
    ----------
    chroma_a, chroma_ap : np.ndarray
        ``[T, 12]`` chroma of segment A / A' (segment-local or full-track).
    seg_a, seg_ap : tuple of int
        Half-open frame intervals of A and A'.

    Returns
    -------
    AlignResult
        One entry per A' frame; mapped A frames are clipped to the A segment.
        ``meta = {'shift': int, 'corr': float, 'lag': int}``; ``corr`` is the
        peak mean per-frame inner product of unit rows (in [0, 1] up to
        numerical noise).
    """
    a0, a1 = _check_segment(seg_a)
    ap0, ap1 = _check_segment(seg_ap)
    X = _l2_rows(_segment_features(chroma_a, seg_a, "chroma_a"))
    Y = _l2_rows(_segment_features(chroma_ap, seg_ap, "chroma_ap"))
    if X.shape[1] != 12 or Y.shape[1] != 12:
        raise ValueError("chroma features must have 12 bins")
    n_a, n_ap = X.shape[0], Y.shape[0]
    lags = np.arange(-(n_ap - 1), n_a)  # corr(l) = sum_i Y[i] . X[i + l]
    overlap = (np.minimum(n_ap, n_a - lags)
               - np.maximum(0, -lags)).clip(min=0).astype(np.float64)
    min_overlap = max(1, min(n_a, n_ap) // 2)
    usable = overlap >= min_overlap

    best = (-np.inf, 0, 0)  # (norm corr, raw shift, lag)
    for k in range(12):
        Yk = np.roll(Y, -k, axis=1)
        # Correlation over time: conv[j] = sum_i Y_k[i] * X[i + (j - n_ap + 1)]
        conv = scipy.signal.fftconvolve(X, Yk[::-1], axes=0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            norm = np.where(usable, conv / np.maximum(overlap, 1.0), -np.inf)
        j = int(np.argmax(norm))
        if norm[j] > best[0]:
            best = (float(norm[j]), k, int(lags[j]))
    corr, shift_raw, lag = best
    shift_signed = ((shift_raw + 5) % 12) - 5  # fold to [-5, 6]

    ap_frames = np.arange(ap0, ap1, dtype=np.int64)
    a_local = np.clip(np.arange(n_ap) + lag, 0, n_a - 1)
    a_frames = a0 + a_local.astype(np.int64)
    logger.debug("transposition alignment: shift=%d lag=%d corr=%.3f",
                 shift_signed, lag, corr)
    return AlignResult(ap_frames, a_frames,
                       {"shift": int(shift_signed), "corr": corr,
                        "lag": int(lag), "method": "transposition_invariant"})


def _cosine_cost(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Pairwise cosine distance ``1 - cos``; zero rows get cost 1."""
    An = _l2_rows(A, eps)
    Bn = _l2_rows(B, eps)
    return 1.0 - An @ Bn.T


def _dtw_path(cost: np.ndarray) -> tp.Tuple[np.ndarray, float]:
    """DTW with steps (1,1), (1,0), (0,1).

    Returns the optimal path as an ``[P, 2]`` int array of ``(i, j)`` index
    pairs from ``(0, 0)`` to ``(n-1, m-1)`` and the accumulated path cost.
    """
    n, m = cost.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        row = cost[i - 1]
        for j in range(1, m + 1):
            D[i, j] = row[j - 1] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    path = [(n - 1, m - 1)]
    i, j = n, m
    while (i, j) != (1, 1):
        # Prefer the diagonal on ties for a maximally smooth path.
        choices = ((D[i - 1, j - 1], i - 1, j - 1),
                   (D[i - 1, j], i - 1, j),
                   (D[i, j - 1], i, j - 1))
        _, i, j = min(choices, key=lambda c: c[0])
        path.append((i - 1, j - 1))
    path.reverse()
    return np.asarray(path, dtype=np.int64), float(D[n, m])


def align_dtw(feat_a: np.ndarray, feat_ap: np.ndarray,
              seg_a: Segment, seg_ap: Segment) -> AlignResult:
    """DTW alignment on joint features (chroma + onset column; S3 stimuli).

    Cosine distance is used as the local cost; the DP allows steps
    (1,1), (1,0), (0,1) in O(n*m). The warping path is converted to a
    monotone per-A'-frame map by taking the median matched A frame for each
    A' frame.

    Parameters
    ----------
    feat_a, feat_ap : np.ndarray
        ``[T, d]`` features of segment A / A' (segment-local or full-track),
        typically ``[chroma | onset_env[:, None]]``.
    seg_a, seg_ap : tuple of int
        Half-open frame intervals of A and A'.

    Returns
    -------
    AlignResult
        One entry per A' frame; ``meta = {'path_cost': float,
        'path_cost_per_step': float, 'path_length': int}``.
    """
    a0, _ = _check_segment(seg_a)
    ap0, ap1 = _check_segment(seg_ap)
    A = _segment_features(feat_a, seg_a, "feat_a")
    B = _segment_features(feat_ap, seg_ap, "feat_ap")
    if A.shape[1] != B.shape[1]:
        raise ValueError("feat_a and feat_ap must share the feature dimension")
    path, path_cost = _dtw_path(_cosine_cost(A, B))
    m = B.shape[0]
    a_frames = np.empty(m, dtype=np.int64)
    for j in range(m):
        matched = path[path[:, 1] == j, 0]
        a_frames[j] = a0 + int(round(float(np.median(matched))))
    ap_frames = np.arange(ap0, ap1, dtype=np.int64)
    meta = {"path_cost": path_cost,
            "path_cost_per_step": path_cost / len(path),
            "path_length": int(len(path)),
            "method": "dtw"}
    return AlignResult(ap_frames, a_frames, meta)
