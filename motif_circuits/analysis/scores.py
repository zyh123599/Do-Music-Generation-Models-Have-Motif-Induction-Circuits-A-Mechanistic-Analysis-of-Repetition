"""Per-head induction scores (single sample) and their null baselines.

Score/null split: ``*_score`` functions return the RAW mean attention on the
induction targets; the matching ``*_null`` functions evaluate the lag null
model at exactly the same alignment pairs. The plan's
``IS = score - null`` (research plan section 5) is formed by the caller, so
both terms stay available for permutation testing.

``phi`` may be either an :class:`~motif_circuits.analysis.alignment.AlignResult`
or a plain ``(a_prime_frames, a_frames)`` tuple of integer arrays.
"""
from __future__ import annotations

import logging
import typing as tp

import numpy as np

__all__ = ["token_induction_score", "token_induction_null",
           "motif_induction_score", "motif_induction_null",
           "dla_copy_score"]

logger = logging.getLogger(__name__)

PhiLike = tp.Union[tp.Tuple[np.ndarray, np.ndarray], "AlignResult"]  # noqa: F821


def _phi_arrays(phi: PhiLike) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Unpack phi into (a_prime_frames, a_frames) int64 arrays."""
    if hasattr(phi, "a_prime_frames") and hasattr(phi, "a_frames"):
        ap, a = phi.a_prime_frames, phi.a_frames
    else:
        ap, a = phi
    ap = np.asarray(ap, dtype=np.int64)
    a = np.asarray(a, dtype=np.int64)
    if ap.shape != a.shape or ap.ndim != 1:
        raise ValueError("phi arrays must be 1-D and of equal length")
    return ap, a


# ----------------------------------------------------------------------
# token-level induction (step coordinates)
# ----------------------------------------------------------------------
def token_induction_score(step_attn: np.ndarray, phi: PhiLike, dm,
                          T: int, codebook: int) -> np.ndarray:
    """Raw token-level induction score per head (one codebook).

    For every aligned pair ``(t_ap, t_a)``, the query is the step of frame
    ``t_ap`` and the key is the step of the SUCCESSOR frame ``t_a + 1``
    (attending to the token that *followed* the previous occurrence =
    induction), both in the given codebook. Pairs are kept only when both
    steps are inside the attention matrix, ``t_a + 1 < T``, and the key step
    is strictly before the query step (causality guard; also drops the
    degenerate self-attention case ``t_ap == t_a + 1``).

    Parameters
    ----------
    step_attn : np.ndarray
        Step-level attention ``[H, S, S]`` (query steps x key steps).
    phi : AlignResult or (a_prime_frames, a_frames)
        Frame alignment A' -> A.
    dm : motif_circuits.delay_map.DelayMap
        Frame<->step mapping.
    T : int
        Number of frames in the sample.
    codebook : int
        Codebook index for both query and key steps.

    Returns
    -------
    np.ndarray
        ``[H]`` mean attention over valid pairs (NaN-aware); all-NaN if no
        pair is valid.
    """
    step_attn = np.asarray(step_attn, dtype=np.float64)
    if step_attn.ndim != 3 or step_attn.shape[1] != step_attn.shape[2]:
        raise ValueError("step_attn must be [H, S, S]")
    H, S, _ = step_attn.shape
    ap, a = _phi_arrays(phi)
    qs = dm.step(ap, codebook)
    ts = dm.step(a + 1, codebook)
    valid = (qs < S) & (ts < S) & (a + 1 < T) & (ts < qs)
    n_valid = int(valid.sum())
    if n_valid == 0:
        logger.warning("token_induction_score: no valid (query, key) pairs")
        return np.full(H, np.nan)
    vals = step_attn[:, qs[valid], ts[valid]]  # [H, n_valid]
    with np.errstate(invalid="ignore"):
        return np.nanmean(vals, axis=1)


def token_induction_null(null_model, phi: PhiLike) -> np.ndarray:
    """Lag-null baseline matching :func:`token_induction_score`.

    For equal-codebook (query, key) steps, the step lag equals the FRAME lag
    ``l = t_ap - (t_a + 1)`` exactly: ``step(t_ap, k) - step(t_a + 1, k) =
    t_ap - t_a - 1`` since the per-codebook delay and special offset cancel.
    The null is therefore the frame-lag spectrum evaluated at ``l``, averaged
    over pairs with ``l >= 1`` (the score's causality guard in frame terms).
    The trailing-edge step/frame bounds of the score cannot be applied here
    (the null has no S/T) — they only trim a few end-of-segment pairs.

    Parameters
    ----------
    null_model : motif_circuits.analysis.null_model.NullModel
        Fitted lag null model.
    phi : AlignResult or (a_prime_frames, a_frames)
        Frame alignment A' -> A.

    Returns
    -------
    np.ndarray
        ``[H]`` mean null attention over pairs; all-NaN if no valid pair.
    """
    ap, a = _phi_arrays(phi)
    lags = ap - (a + 1)
    lags = lags[lags >= 1]
    if lags.size == 0:
        return np.full(null_model.n_heads, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(null_model.null_at_lags(lags), axis=1)


# ----------------------------------------------------------------------
# motif-level induction (frame coordinates)
# ----------------------------------------------------------------------
def _window_keys(t_ap: int, t_a: int, window: int) -> np.ndarray:
    """Causal key-frame window ``[t_a+1-window, t_a+1+window] ∩ [0, t_ap-1]``."""
    lo = max(0, t_a + 1 - window)
    hi = min(t_ap - 1, t_a + 1 + window)
    if hi < lo:
        return np.empty(0, dtype=np.int64)
    return np.arange(lo, hi + 1, dtype=np.int64)


def motif_induction_score(frame_attn: np.ndarray, phi: PhiLike,
                          window: int = 2) -> np.ndarray:
    """Raw motif-level induction score per head.

    For every aligned pair ``(t_ap, t_a)``, sums the frame-level attention
    from query frame ``t_ap`` over the key-frame window
    ``[t_a+1-window, t_a+1+window]`` intersected with the causal range
    ``[0, t_ap - 1]``, then averages over pairs. Pairs with an empty window,
    with ``t_ap`` outside ``[0, T)``, or with an all-NaN window are skipped
    (NaN-aware mean over pairs).

    Parameters
    ----------
    frame_attn : np.ndarray
        Frame-coordinate attention ``[H, T, T]`` (e.g. from
        ``DelayMap.aggregate_frames``; may contain NaN).
    phi : AlignResult or (a_prime_frames, a_frames)
        Frame alignment A' -> A.
    window : int, optional
        Tolerance window w in frames (default +-2 = 40 ms at 50 Hz).

    Returns
    -------
    np.ndarray
        ``[H]`` mean windowed attention mass over valid pairs.
    """
    frame_attn = np.asarray(frame_attn, dtype=np.float64)
    if frame_attn.ndim != 3 or frame_attn.shape[1] != frame_attn.shape[2]:
        raise ValueError("frame_attn must be [H, T, T]")
    H, T, _ = frame_attn.shape
    ap, a = _phi_arrays(phi)
    per_pair = []
    for t_ap, t_a in zip(ap, a):
        if not 0 <= t_ap < T:
            continue
        keys = _window_keys(int(t_ap), int(t_a), window)
        keys = keys[keys < T]
        if keys.size == 0:
            continue
        vals = frame_attn[:, t_ap, keys]  # [H, |W|]
        finite = np.isfinite(vals)
        s = np.where(finite.any(axis=1),
                     np.nansum(np.where(finite, vals, 0.0), axis=1), np.nan)
        per_pair.append(s)
    if not per_pair:
        logger.warning("motif_induction_score: no valid pairs")
        return np.full(H, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.stack(per_pair, axis=1), axis=1)


def motif_induction_null(null_model, phi: PhiLike,
                         window: int = 2) -> np.ndarray:
    """Lag-null baseline matching :func:`motif_induction_score`.

    For each pair, sums the null spectrum over the SAME window key frames as
    the score (translated to lags ``t_ap - s``, all >= 1 by construction of
    the causal window), then averages over pairs.

    Parameters
    ----------
    null_model : motif_circuits.analysis.null_model.NullModel
        Fitted lag null model.
    phi : AlignResult or (a_prime_frames, a_frames)
        Frame alignment A' -> A.
    window : int, optional
        Same tolerance window as used for the score.

    Returns
    -------
    np.ndarray
        ``[H]`` mean windowed null mass over valid pairs.
    """
    ap, a = _phi_arrays(phi)
    per_pair = []
    for t_ap, t_a in zip(ap, a):
        if t_ap < 0:
            continue
        keys = _window_keys(int(t_ap), int(t_a), window)
        if keys.size == 0:
            continue
        lags = int(t_ap) - keys
        per_pair.append(null_model.null_at_lags(lags).sum(axis=1))
    if not per_pair:
        return np.full(null_model.n_heads, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.stack(per_pair, axis=1), axis=1)


# ----------------------------------------------------------------------
# direct logit attribution (copy score, token level)
# ----------------------------------------------------------------------
def dla_copy_score(z_head: np.ndarray, out_proj_weight: np.ndarray,
                   head_index: int, ln_gain: np.ndarray, ln_scale: float,
                   linear_k_weight: np.ndarray, true_token: int) -> float:
    """Direct logit attribution of one head's output to a true token's logit.

    Computes the head's contribution to the residual stream through
    ``out_proj`` (its head-contiguous column block), pushes it through a
    LINEARIZED final LayerNorm, and projects onto the true token's unembedding
    row of the per-codebook readout::

        contribution = W_k[true_token] . (ln_gain * (W_O[:, h*d:(h+1)*d] @ z)) / ln_scale

    LN-linearization caveat: LayerNorm is nonlinear; we freeze the
    normalization scale at ``ln_scale`` (the std computed on the FULL residual
    at this position) and neglect both (a) the head's effect on that scale and
    (b) the mean-subtraction of the contribution. This is the standard DLA
    approximation — accurate when the head's write is small relative to the
    residual norm; treat scores as first-order attributions, not exact
    logit differences.

    Parameters
    ----------
    z_head : np.ndarray
        Head output before out_proj, ``[d_head]`` (one position).
    out_proj_weight : np.ndarray
        Attention output projection weight ``[D, D]`` (torch layout:
        ``out = W @ x``; head h reads columns ``[h*d_head, (h+1)*d_head)``).
    head_index : int
        Head index h.
    ln_gain : np.ndarray
        Final LayerNorm (out_norm) gain ``[D]``.
    ln_scale : float
        Frozen LN denominator at this position: ``std(resid) = sqrt(var + eps)``.
    linear_k_weight : np.ndarray
        Per-codebook readout weight ``[card, D]`` (no bias in MusicGen).
    true_token : int
        Token id whose logit is attributed.

    Returns
    -------
    float
        The head's linearized contribution to the true token's logit.
    """
    z_head = np.asarray(z_head, dtype=np.float64).reshape(-1)
    W_O = np.asarray(out_proj_weight, dtype=np.float64)
    ln_gain = np.asarray(ln_gain, dtype=np.float64).reshape(-1)
    W_k = np.asarray(linear_k_weight, dtype=np.float64)
    D = W_O.shape[0]
    d_head = z_head.shape[0]
    lo, hi = head_index * d_head, (head_index + 1) * d_head
    if hi > W_O.shape[1]:
        raise ValueError(f"head {head_index} slice [{lo}:{hi}] exceeds "
                         f"out_proj width {W_O.shape[1]}")
    if ln_gain.shape[0] != D or W_k.shape[1] != D:
        raise ValueError("ln_gain / linear_k_weight incompatible with D")
    if not np.isfinite(ln_scale) or ln_scale <= 0:
        raise ValueError("ln_scale must be a positive finite float")
    resid_write = W_O[:, lo:hi] @ z_head              # [D]
    ln_out = ln_gain * resid_write / float(ln_scale)  # linearized LN
    return float(W_k[int(true_token)] @ ln_out)
