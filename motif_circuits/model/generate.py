"""Free generation with optional per-head interventions.

Mirrors ``MusicGen._generate_tokens``'s simple (single-window) path but calls
``lm.generate`` directly so :class:`HeadInterventions` hooks stay in control
of the whole autoregressive loop (step counters reset per run).
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import typing as tp

import numpy as np
import torch

from .hooks import HeadIntervention, HeadInterventions
from .loader import _autocast
from .teacher_forcing import _wav_to_tensor

__all__ = ["GenResult", "generate_with_interventions"]

logger = logging.getLogger(__name__)


@dataclass
class GenResult:
    """One generation batch.

    Attributes
    ----------
    wav : np.ndarray
        ``[B, C, T_samples]`` float32 decoded audio.
    codes : np.ndarray
        ``[B, K, T]`` int64 frame-aligned EnCodec codes (prompt included).
    sr : int
        Sample rate of ``wav``.
    """

    wav: np.ndarray
    codes: np.ndarray
    sr: int


def generate_with_interventions(
        model,
        interventions: tp.Optional[tp.Sequence[HeadIntervention]] = None,
        *,
        descriptions: tp.Optional[tp.Sequence[tp.Optional[str]]] = None,
        prompt_wav=None,
        prompt_sr: tp.Optional[int] = None,
        num_samples: int = 1,
        duration: float = 10.0,
        seed: int = 0,
        progress: bool = False,
        **gen_overrides) -> GenResult:
    """Generate audio under (optional) head interventions, deterministically.

    Parameters
    ----------
    model : MusicGen
        Loaded model; its ``generation_params`` (set via
        ``set_generation_params``) are used, overridable per call.
    interventions : sequence of HeadIntervention, optional
        Head edits active for the whole generation (both CFG halves).
    descriptions : sequence of str or None, optional
        Text conditions; ``None`` entries (or omitting the argument) give
        unconditional samples. Length defines the batch when given.
    prompt_wav : array-like, optional
        Audio prompt for continuation, ``[T]``/``[C,T]``/``[B,C,T]`` at
        ``prompt_sr``. The prompt is re-encoded and INCLUDED in the output.
    prompt_sr : int, optional
        Sample rate of ``prompt_wav`` (required with a prompt).
    num_samples : int
        Batch size when neither descriptions nor prompt fix it.
    duration : float
        Total output duration in seconds (must exceed the prompt duration;
        must be <= the model's max single-window duration, 30 s).
    seed : int
        Seeds torch CPU+CUDA RNGs for reproducible sampling.
    progress : bool
        Print generation progress (audiocraft callback).
    **gen_overrides
        Overrides merged into the model's generation params (e.g.
        ``top_k=0, temp=0.9``).

    Returns
    -------
    GenResult
    """
    from audiocraft.data.audio_utils import convert_audio        # lazy
    from audiocraft.modules.conditioners import ConditioningAttributes

    frame_rate = float(model.frame_rate)
    max_gen_len = int(duration * frame_rate)

    # --- batch bookkeeping -------------------------------------------------
    if descriptions is not None:
        batch = len(descriptions)
    elif prompt_wav is not None:
        t = _wav_to_tensor(prompt_wav, "cpu")
        batch = int(t.shape[0])
    else:
        batch = int(num_samples)
    descs = list(descriptions) if descriptions is not None else [None] * batch

    attributes = [ConditioningAttributes(text={"description": d}) for d in descs]

    prompt_tokens = None
    if prompt_wav is not None:
        if prompt_sr is None:
            raise ValueError("prompt_sr is required with prompt_wav")
        t = _wav_to_tensor(prompt_wav, model.device)
        if t.shape[0] == 1 and batch > 1:
            t = t.expand(batch, -1, -1)
        t = convert_audio(t, prompt_sr, model.sample_rate, model.audio_channels)
        with torch.no_grad():
            prompt_tokens, scale = model.compression_model.encode(t)
        assert scale is None
        if prompt_tokens.shape[-1] >= max_gen_len:
            raise ValueError(
                f"prompt ({prompt_tokens.shape[-1]} frames) must be shorter "
                f"than duration*frame_rate ({max_gen_len} frames); increase "
                "duration or trim the prompt")

    if duration > float(getattr(model, "max_duration", 30.0)):
        raise ValueError(f"duration {duration}s exceeds the single-window "
                         f"maximum {model.max_duration}s (extend-stride "
                         "generation is out of scope for circuit analysis)")

    params = dict(model.generation_params)
    params.update(gen_overrides)
    if params.get("two_step_cfg"):
        raise ValueError(
            "two_step_cfg runs two forwards per generation step, which would "
            "desynchronize the interventions' absolute-step counters; use the "
            "default single-pass CFG")

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    callback = None
    if progress:
        def callback(i, n):  # noqa: ANN001 - audiocraft signature
            print(f"  gen {i:5d}/{n:5d}", end="\r")

    manager = HeadInterventions(model, list(interventions or []))
    with torch.no_grad(), _autocast(model), manager:
        tokens = model.lm.generate(prompt_tokens, attributes,
                                   callback=callback,
                                   max_gen_len=max_gen_len, **params)

    with torch.no_grad():
        wav = model.compression_model.decode(tokens, None).float().cpu().numpy()
    logger.info("generated %d sample(s), %d frames, interventions=%d",
                tokens.shape[0], tokens.shape[-1],
                len(list(interventions or [])))
    return GenResult(wav=wav, codes=tokens.cpu().numpy(),
                     sr=int(model.sample_rate))
