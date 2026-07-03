"""Activation-patching protocol (research plan section 6.1).

Denoising direction: activations captured in the CLEAN forward (prelude A + G
followed by the true recurrence A') are patched into the CORRUPTED forward
(same prelude, unrelated continuation), and we measure how much of the clean
continuation's log-probability is recovered.

Metric (normative): ``Delta`` is the mean teacher-forcing log-probability of
the CLEAN continuation's tokens over the first ``L`` frames of the A' segment
(all requested codebooks, valid-mask aware). ``Delta_clean`` evaluates them
under the clean context, ``Delta_corr`` under the corrupted context (positions
align 1:1 because the pair shares the prelude), ``Delta_patched`` under the
corrupted context with clean head outputs injected at the A' steps. Recovery::

    R = (Delta_patched - Delta_corr) / (Delta_clean - Delta_corr)

Torch and ``motif_circuits.model`` (which requires torch) are imported lazily
inside functions so this module stays importable on NumPy-only machines; the
pure arithmetic (:func:`recovery_rate`) is CPU-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import typing as tp

import numpy as np

__all__ = ["PatchTargets", "PatchResult", "recovery_rate",
           "run_patching_pair", "mean_head_outputs", "head_accumulation_curve"]

logger = logging.getLogger(__name__)

_DEGENERATE_EPS = 1e-6


@dataclass
class PatchTargets:
    """What to patch and where to measure.

    Attributes
    ----------
    layer_heads : list of (layer, head)
        Heads whose clean outputs are injected.
    frames : np.ndarray
        Absolute frame indices of the A' segment (patch site; the metric uses
        its first ``L`` frames).
    codebooks : list of int
        Codebooks included in the log-prob metric (patching itself always
        targets the steps of ALL codebooks at the given frames).
    """

    layer_heads: tp.List[tp.Tuple[int, int]]
    frames: np.ndarray
    codebooks: tp.List[int] = field(default_factory=lambda: [0, 1, 2, 3])

    def __post_init__(self):
        self.frames = np.asarray(self.frames, dtype=np.int64)
        if self.frames.ndim != 1 or self.frames.size == 0:
            raise ValueError("frames must be a non-empty 1-D array")


@dataclass
class PatchResult:
    """Outcome of one clean/corrupted patching pair."""

    delta_clean: float
    delta_corr: float
    delta_patched: float
    recovery: float
    degenerate: bool
    per_codebook: tp.Dict[int, tp.Dict[str, float]]
    n_valid: int


def recovery_rate(delta_clean: float, delta_corr: float, delta_patched: float,
                  eps: float = _DEGENERATE_EPS) -> tp.Tuple[float, bool]:
    """Recovery ``R = (patched - corr) / (clean - corr)`` with a guard.

    Returns ``(nan, True)`` when the clean/corrupted gap is smaller than
    ``eps`` (the pair carries no signal to recover) or any input is not
    finite; otherwise ``(R, False)``.
    """
    vals = (delta_clean, delta_corr, delta_patched)
    if not all(np.isfinite(v) for v in vals):
        return float("nan"), True
    gap = delta_clean - delta_corr
    if abs(gap) < eps:
        return float("nan"), True
    return float((delta_patched - delta_corr) / gap), False


def _metric_frames(targets: PatchTargets, L: int) -> np.ndarray:
    """First ``L`` frames of the (sorted) A' patch-target frames."""
    return np.sort(targets.frames)[: int(L)]


def _delta(tf_result, clean_codes, frames: np.ndarray,
           codebooks: tp.Sequence[int]) -> tp.Tuple[float, tp.Dict[int, float], int]:
    """Mean log-prob of the clean tokens at the metric positions.

    Parameters
    ----------
    tf_result : motif_circuits.model.TFResult
        Teacher-forcing result of the evaluated forward (any context).
    clean_codes : torch.Tensor
        ``[1, K, T]`` clean codes whose tokens are scored.
    frames : np.ndarray
        Metric frame indices.
    codebooks : sequence of int
        Codebooks to include.

    Returns
    -------
    (delta, per_codebook, n_valid)
    """
    from motif_circuits.model import logprobs_of  # lazy: torch

    lp = logprobs_of(tf_result.logits, clean_codes)      # [1, K, T] cpu float32
    lp = lp[0].numpy()
    mask = np.asarray(tf_result.mask[0], dtype=bool)      # [K, T]
    T = lp.shape[1]
    frames = frames[(frames >= 0) & (frames < T)]
    per_cb: tp.Dict[int, float] = {}
    total, n_valid = 0.0, 0
    for k in codebooks:
        m = mask[k, frames]
        vals = lp[k, frames][m]
        per_cb[int(k)] = float(np.mean(vals)) if vals.size else float("nan")
        total += float(vals.sum())
        n_valid += int(vals.size)
    delta = total / n_valid if n_valid else float("nan")
    return delta, per_cb, n_valid


def run_patching_pair(model, clean_codes, corrupted_codes,
                      targets: PatchTargets, ct=None, L: int = 25) -> PatchResult:
    """Run the full clean -> corrupted denoising-patch protocol for one pair.

    Parameters
    ----------
    model : MusicGen
        Loaded model (see ``motif_circuits.model.load_musicgen``).
    clean_codes, corrupted_codes : torch.Tensor
        ``[1, K, T]`` EnCodec codes sharing the prelude (A + G) and differing
        in the continuation (true A' vs unrelated phrase). Equal T required.
    targets : PatchTargets
        Heads to patch, A' frames, metric codebooks.
    ct : ConditionTensors, optional
        Precomputed (null) condition tensors; built internally when None.
    L : int
        Number of leading A' frames in the metric (default 25 = 0.5 s).

    Returns
    -------
    PatchResult
        Deltas, recovery R (guarded), per-codebook breakdown of the three
        deltas, and the number of valid metric positions.
    """
    import torch  # lazy
    from motif_circuits.model import (HeadIntervention, HeadInterventions,
                                      HeadOutputCapture, delay_map_for,
                                      null_condition_tensors,
                                      teacher_forcing_logprobs)

    if clean_codes.shape != corrupted_codes.shape:
        raise ValueError("clean and corrupted codes must have equal shape "
                         f"({tuple(clean_codes.shape)} vs {tuple(corrupted_codes.shape)})")
    dm = delay_map_for(model)
    T = int(clean_codes.shape[-1])
    S = dm.seq_len(T, keep_only_valid_steps=True)
    if ct is None:
        ct = null_condition_tensors(model, int(clean_codes.shape[0]))

    layers = sorted({l for l, _ in targets.layer_heads})
    metric_frames = _metric_frames(targets, L)

    # (a) clean forward: capture per-head outputs + clean metric.
    with HeadOutputCapture(model, layers=layers) as cap:
        tf_clean = teacher_forcing_logprobs(model, clean_codes, condition_tensors=ct)
    d_clean, cb_clean, n_valid = _delta(tf_clean, clean_codes, metric_frames,
                                        targets.codebooks)

    # (b) corrupted forward, no intervention: baseline metric on clean tokens.
    tf_corr = teacher_forcing_logprobs(model, corrupted_codes, condition_tensors=ct)
    d_corr, cb_corr, _ = _delta(tf_corr, clean_codes, metric_frames,
                                targets.codebooks)

    # (c) corrupted forward with clean head outputs injected at the A' steps.
    steps = dm.steps_for_frames(targets.frames.tolist(), S=S)
    interventions = []
    for layer, head in targets.layer_heads:
        z = cap.z[layer]  # [B, S, H, d_head] (cpu, original dtype)
        interventions.append(HeadIntervention(
            layer=layer, head=head, mode="patch",
            value=z[0, :, head, :], steps=steps))
    with HeadInterventions(model, interventions):
        tf_patched = teacher_forcing_logprobs(model, corrupted_codes,
                                              condition_tensors=ct)
    d_patched, cb_patched, _ = _delta(tf_patched, clean_codes, metric_frames,
                                      targets.codebooks)

    r, degenerate = recovery_rate(d_clean, d_corr, d_patched)
    per_cb = {k: {"clean": cb_clean.get(k, float("nan")),
                  "corr": cb_corr.get(k, float("nan")),
                  "patched": cb_patched.get(k, float("nan"))}
              for k in map(int, targets.codebooks)}
    logger.debug("patching pair: clean=%.4f corr=%.4f patched=%.4f R=%.3f%s",
                 d_clean, d_corr, d_patched, r,
                 " (degenerate)" if degenerate else "")
    return PatchResult(d_clean, d_corr, d_patched, r, degenerate,
                       per_cb, n_valid)


def mean_head_outputs(model, codes_iter: tp.Iterable,
                      layers: tp.Sequence[int]) -> tp.Dict[tp.Tuple[int, int], np.ndarray]:
    """Mean per-head output vectors over an S6 corpus (mean-ablation values).

    Streaming accumulation over samples and time — memory stays bounded by
    one sample's capture.

    Parameters
    ----------
    model : MusicGen
    codes_iter : iterable of torch.Tensor
        ``[1, K, T]`` code tensors (e.g. loaded S6 samples).
    layers : sequence of int
        Layers whose heads are accumulated.

    Returns
    -------
    dict[(layer, head)] -> np.ndarray
        Float32 mean vector ``[d_head]`` per head.
    """
    from motif_circuits.model import (HeadOutputCapture, model_geometry,
                                      null_condition_tensors,
                                      teacher_forcing_logprobs)

    geom = model_geometry(model)
    layers = list(layers)
    sums = {(l, h): np.zeros(geom.d_head, dtype=np.float64)
            for l in layers for h in range(geom.n_heads)}
    count = 0
    ct = None
    ct_batch = -1
    for codes in codes_iter:
        batch = int(codes.shape[0])
        if ct is None or ct_batch != batch:
            ct = null_condition_tensors(model, batch)
            ct_batch = batch
        with HeadOutputCapture(model, layers=layers) as cap:
            teacher_forcing_logprobs(model, codes, condition_tensors=ct)
        for l in layers:
            z = cap.z[l]
            if hasattr(z, "float"):  # torch tensor -> numpy
                z = z.float().numpy()
            z = np.asarray(z, dtype=np.float64)  # [B, S, H, d]
            for h in range(geom.n_heads):
                sums[(l, h)] += z[:, :, h, :].reshape(-1, geom.d_head).sum(axis=0)
        b, s = cap.z[layers[0]].shape[:2]
        count += int(b) * int(s)
    if count == 0:
        raise ValueError("codes_iter yielded no samples")
    return {k: (v / count).astype(np.float32) for k, v in sums.items()}


def head_accumulation_curve(model, ranked_heads: tp.Sequence[tp.Tuple[int, int]],
                            pairs_iter: tp.Iterable,
                            ks: tp.Sequence[int] = (1, 2, 4, 8, 16, 32),
                            codebooks: tp.Sequence[int] = (0, 1, 2, 3),
                            ct=None, L: int = 25) -> tp.Dict[int, float]:
    """Mean recovery R as a function of the number of patched top heads.

    Parameters
    ----------
    model : MusicGen
    ranked_heads : sequence of (layer, head)
        Candidate heads, best first (from the screening ranking).
    pairs_iter : iterable of (clean_codes, corrupted_codes, frames)
        Patching pairs; ``frames`` is the A'-segment frame array. The
        iterable is materialized (list) so every K sees the same pairs.
    ks : sequence of int
        Head-set sizes to evaluate (clipped to ``len(ranked_heads)``).
    codebooks, ct, L
        Forwarded to :func:`run_patching_pair`.

    Returns
    -------
    dict[int, float]
        ``K -> mean recovery`` over non-degenerate pairs (NaN when every
        pair is degenerate).
    """
    pairs = list(pairs_iter)
    if not pairs:
        raise ValueError("pairs_iter yielded no pairs")
    curve: tp.Dict[int, float] = {}
    for k in ks:
        k_eff = min(int(k), len(ranked_heads))
        if k_eff == 0:
            continue
        rs = []
        for clean_codes, corrupted_codes, frames in pairs:
            targets = PatchTargets(layer_heads=list(ranked_heads[:k_eff]),
                                   frames=np.asarray(frames),
                                   codebooks=list(codebooks))
            res = run_patching_pair(model, clean_codes, corrupted_codes,
                                    targets, ct=ct, L=L)
            if not res.degenerate:
                rs.append(res.recovery)
        curve[int(k)] = float(np.mean(rs)) if rs else float("nan")
        logger.info("accumulation curve: K=%d -> R=%.3f over %d/%d pairs",
                    k, curve[int(k)], len(rs), len(pairs))
    return curve
