"""Teacher-forcing evaluation: audio <-> codes and true-token log-probs.

The teacher-forcing entry point is audiocraft's
``LMModel.compute_predictions`` (logits already re-aligned to frames; the
delay interleaving happens inside — see ``docs/audiocraft_api_notes.md`` §2).
Invalid tail positions (frames whose delayed steps fall beyond the valid
layout) carry NaN logits upstream; here they are masked and zeroed so NaNs
never leak into downstream statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import typing as tp

import numpy as np
import torch

from .loader import _autocast, _get_lm, null_condition_tensors

__all__ = ["TFResult", "encode_audio", "decode_codes",
           "teacher_forcing_logprobs", "logprobs_of"]

logger = logging.getLogger(__name__)


@dataclass
class TFResult:
    """Teacher-forcing forward result (all on CPU).

    Attributes
    ----------
    logits : torch.Tensor
        ``[B, K, T, card]`` float32; NaN at invalid positions replaced by 0
        (use ``mask``!).
    logprob_true : torch.Tensor
        ``[B, K, T]`` float32 log-probability of the codes that were fed;
        0.0 where ``mask`` is False.
    mask : torch.Tensor
        ``[B, K, T]`` bool; True where the position is a valid prediction.
    """

    logits: torch.Tensor
    logprob_true: torch.Tensor
    mask: torch.Tensor


def _wav_to_tensor(wav, device) -> torch.Tensor:
    """Coerce [T] / [C, T] / [B, C, T] audio into a float32 [B, C, T] tensor."""
    t = torch.as_tensor(np.asarray(wav) if isinstance(wav, np.ndarray) else wav,
                        dtype=torch.float32)
    if t.dim() == 1:
        t = t[None, None, :]
    elif t.dim() == 2:
        t = t[None, :, :]
    elif t.dim() != 3:
        raise ValueError(f"wav must be [T], [C, T] or [B, C, T]; got {tuple(t.shape)}")
    return t.to(device)


def encode_audio(model, wav, sr: int) -> torch.Tensor:
    """EnCodec-encode audio into codes ``[B, K, T]`` (long, on model device).

    Parameters
    ----------
    model : MusicGen
    wav : np.ndarray | torch.Tensor
        ``[T]``, ``[C, T]`` or ``[B, C, T]`` waveform.
    sr : int
        Sample rate of ``wav``; converted to the model's rate/channels.
    """
    from audiocraft.data.audio_utils import convert_audio  # lazy

    t = _wav_to_tensor(wav, model.device)
    t = convert_audio(t, sr, model.sample_rate, model.audio_channels)
    with torch.no_grad():
        codes, scale = model.compression_model.encode(t)
    assert scale is None, "unexpected scaled EnCodec model"
    return codes


def decode_codes(model, codes: torch.Tensor) -> np.ndarray:
    """Decode codes ``[B, K, T]`` to waveform ``[B, C, T_samples]`` float32."""
    with torch.no_grad():
        wav = model.compression_model.decode(codes.to(model.device), None)
    return wav.float().cpu().numpy()


def logprobs_of(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Log-probabilities of arbitrary token ids under given logits.

    Needed by activation patching, where the CLEAN continuation's tokens are
    scored under the CORRUPTED context's logits.

    Parameters
    ----------
    logits : torch.Tensor
        ``[B, K, T, card]`` (NaN-free; see :class:`TFResult`).
    tokens : torch.Tensor
        ``[B, K, T]`` long token ids.

    Returns
    -------
    torch.Tensor
        ``[B, K, T]`` float32 log-probabilities (garbage at positions that
        were masked upstream — apply the TFResult mask).
    """
    logp = torch.log_softmax(logits.float(), dim=-1)
    idx = tokens.to(device=logp.device, dtype=torch.long).unsqueeze(-1)
    return logp.gather(-1, idx).squeeze(-1)


def teacher_forcing_logprobs(model, codes: torch.Tensor,
                             condition_tensors: tp.Optional[dict] = None
                             ) -> TFResult:
    """Teacher-forcing forward returning per-position true-token log-probs.

    Runs ``lm.compute_predictions`` under the model's autocast and no_grad.
    When ``condition_tensors`` is None, null (unconditional) conditions are
    built internally for the batch.

    Parameters
    ----------
    model : MusicGen
    codes : torch.Tensor
        ``[B, K, T]`` long codes to evaluate.
    condition_tensors : ConditionTensors, optional
        Precomputed conditioning (from ``null_condition_tensors`` /
        ``text_condition_tensors``); prefer precomputing when evaluating
        many samples.
    """
    lm = _get_lm(model)
    codes = codes.to(next(lm.parameters()).device)
    if condition_tensors is None:
        condition_tensors = null_condition_tensors(model, int(codes.shape[0]))
    with torch.no_grad(), _autocast(model):
        out = lm.compute_predictions(codes, [], condition_tensors=condition_tensors)
    logits = out.logits.detach().float().cpu()      # [B, K, T, card]
    mask = out.mask.detach().bool().cpu()           # [B, K, T]
    nan_free = torch.where(torch.isnan(logits), torch.zeros(()), logits)
    lp = logprobs_of(nan_free, codes.detach().cpu())
    lp = torch.where(mask, lp, torch.zeros(()))
    if bool((~mask).any()):
        logger.debug("teacher forcing: %d invalid tail positions masked",
                     int((~mask).sum()))
    return TFResult(logits=nan_free, logprob_true=lp, mask=mask)
