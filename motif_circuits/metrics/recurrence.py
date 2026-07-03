"""Motif recurrence rate: does the free continuation bring motif A back?

Research plan section 7, "motif recurrence rate": given a prompt containing
motif A, count the fraction of generated continuations that contain a
segment whose transposition-invariant chroma correlation with A exceeds a
threshold tau, with a sensitivity sweep over tau in [0.6, 0.9].

Pure NumPy; chroma extraction lives in ``motif_circuits.utils.chroma``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import typing as tp

import numpy as np

__all__ = ["RecurrenceResult", "motif_recurrence", "recurrence_rate"]

_EPS = 1e-12
#: Number of decimals used to canonicalize tau dict keys (np.arange floats
#: like 0.8000000000000002 become the exact literal 0.8).
_TAU_DECIMALS = 6


def _tau_key(tau: float) -> float:
    """Canonical dict key for a threshold (rounds away float-arange noise)."""
    return round(float(tau), _TAU_DECIMALS)


def _normalize_rows(x: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """L2-normalize rows with a zero-guard (silent frames stay zero)."""
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, eps)


@dataclass
class RecurrenceResult:
    """Result of scanning one continuation for recurrences of motif A.

    Attributes
    ----------
    best_corr : float
        Maximum transposition-invariant correlation over all windows/shifts.
    best_frame : int
        Start frame of the best window, in *continuation* coordinates
        (frame 0 = first continuation frame; the prompt is not included).
    best_shift : int
        Circular chroma-bin shift (0..11) attaining ``best_corr``: the
        recurrence appears transposed up by ``best_shift`` semitones
        (mod 12) relative to A.
    hits : dict[float, bool]
        ``tau -> (best_corr > tau)`` for each requested threshold. Keys are
        rounded to 6 decimals so e.g. ``hits[0.8]`` works with
        ``np.arange``-generated taus.
    corr_curve : np.ndarray
        ``[n_windows]`` best-over-shift correlation per window; window ``i``
        starts at continuation frame ``i * hop``.
    """

    best_corr: float
    best_frame: int
    best_shift: int
    hits: tp.Dict[float, bool] = field(default_factory=dict)
    corr_curve: np.ndarray = field(default_factory=lambda: np.zeros(0))


def motif_recurrence(
    prompt_chroma_A: np.ndarray,
    cont_chroma: np.ndarray,
    taus: np.ndarray = np.arange(0.6, 0.91, 0.05),
    hop: int = 1,
) -> RecurrenceResult:
    """Scan a continuation for transposition-invariant recurrences of motif A.

    Windows of length ``Ta`` (the motif length) slide over the continuation.
    For each window the correlation with A is the *maximum over the 12
    circular chroma-bin shifts* of the mean per-frame cosine similarity —
    i.e. exact repeats and transposed repeats score identically. Silent
    (all-zero) frames contribute 0 similarity.

    Parameters
    ----------
    prompt_chroma_A : np.ndarray
        ``[Ta, 12]`` chroma of the motif segment A only (cut from the
        prompt; not the whole prompt).
    cont_chroma : np.ndarray
        ``[Tc, 12]`` chroma of the *continuation only*. The caller must
        exclude the prompt frames — otherwise the prompt's own copy of A
        trivially matches. ``best_frame`` is therefore relative to the
        continuation start.
    taus : np.ndarray
        Detection thresholds; ``hits[tau] = best_corr > tau`` (strict).
        Default sweeps 0.6..0.9 in steps of 0.05 per the research plan.
    hop : int
        Window stride in frames (1 = every frame).

    Returns
    -------
    RecurrenceResult
        See the dataclass docs. The interface-contract dict
        "tau -> any window with corr > tau" is the ``hits`` field.
    """
    A = _normalize_rows(prompt_chroma_A)
    C = _normalize_rows(cont_chroma)
    if A.ndim != 2 or A.shape[1] != 12:
        raise ValueError(f"prompt_chroma_A must be [Ta, 12], got {A.shape}")
    if C.ndim != 2 or C.shape[1] != 12:
        raise ValueError(f"cont_chroma must be [Tc, 12], got {C.shape}")
    if hop < 1:
        raise ValueError("hop must be >= 1")
    Ta, Tc = A.shape[0], C.shape[0]
    if Ta == 0:
        raise ValueError("motif A is empty")
    if Tc < Ta:
        raise ValueError(f"continuation ({Tc} frames) shorter than motif ({Ta} frames)")

    # [12, Ta, 12]: A transposed up by r semitones = chroma rolled by +r bins.
    shifted_A = np.stack([np.roll(A, r, axis=1) for r in range(12)])
    # [n_windows, Ta, 12] sliding windows over the continuation (view).
    windows = np.lib.stride_tricks.sliding_window_view(C, (Ta, 12))[::hop, 0]
    # corr[w, r] = mean_t <shifted_A[r, t], window_w[t]>
    corr = np.einsum("wtb,rtb->wr", windows, shifted_A) / Ta

    corr_curve = corr.max(axis=1)
    w_best, r_best = np.unravel_index(int(np.argmax(corr)), corr.shape)
    best_corr = float(corr[w_best, r_best])
    taus = np.atleast_1d(np.asarray(taus, dtype=np.float64))
    hits = {_tau_key(tau): bool(best_corr > tau) for tau in taus}
    return RecurrenceResult(
        best_corr=best_corr,
        best_frame=int(w_best * hop),
        best_shift=int(r_best),
        hits=hits,
        corr_curve=corr_curve,
    )


def recurrence_rate(
    results: tp.Sequence[RecurrenceResult],
    taus: np.ndarray = np.arange(0.6, 0.91, 0.05),
) -> tp.Dict[float, float]:
    """Fraction of samples with a recurrence hit, per threshold.

    Parameters
    ----------
    results : sequence of RecurrenceResult
        One result per generated continuation.
    taus : np.ndarray
        Thresholds to report (the sensitivity curve of the research plan).

    Returns
    -------
    dict[float, float]
        ``tau -> fraction of samples with hits[tau]``. Keys rounded to 6
        decimals as in :func:`motif_recurrence`. Thresholds missing from a
        result's ``hits`` fall back to ``best_corr > tau``.
    """
    taus = np.atleast_1d(np.asarray(taus, dtype=np.float64))
    out: tp.Dict[float, float] = {}
    for tau in taus:
        key = _tau_key(tau)
        if len(results) == 0:
            out[key] = 0.0
            continue
        flags = [
            bool(r.hits[key]) if key in r.hits else bool(r.best_corr > tau)
            for r in results
        ]
        out[key] = float(np.mean(flags))
    return out
