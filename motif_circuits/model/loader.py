"""Model loading and basic introspection for MusicGen (audiocraft 1.3.0).

All audiocraft imports are lazy so this module stays importable on machines
without audiocraft (e.g. the CPU test box). See ``docs/interfaces.md`` section
3 and ``docs/audiocraft_api_notes.md`` for the verified upstream semantics
this module relies on.
"""
from __future__ import annotations

import contextlib
import logging
import typing as tp
from dataclasses import dataclass

import torch

from ..delay_map import DelayMap

logger = logging.getLogger(__name__)

__all__ = [
    "ModelGeometry",
    "load_musicgen",
    "model_geometry",
    "delay_map_for",
    "null_condition_tensors",
    "text_condition_tensors",
]

_SIZES = ("small", "medium", "large")


@dataclass(frozen=True)
class ModelGeometry:
    """Static geometry of a MusicGen model (see interfaces.md section 3)."""

    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    n_q: int
    card: int
    frame_rate: int
    sample_rate: int


def _get_lm(model) -> tp.Any:
    """Return the LM module from a MusicGen wrapper, or the object itself."""
    return model.lm if hasattr(model, "lm") else model


def _autocast(model) -> tp.ContextManager:
    """Return the model's autocast context, or a null context for mocks."""
    ac = getattr(model, "autocast", None)
    return ac if ac is not None else contextlib.nullcontext()


def load_musicgen(size: str = "small", device: str = "cuda") -> "tp.Any":
    """Load a released text-to-music MusicGen model and validate assumptions.

    Parameters
    ----------
    size : str
        One of ``{"small", "medium", "large"}``; maps to
        ``facebook/musicgen-<size>``.
    device : str
        Torch device string, e.g. ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    MusicGen
        The audiocraft ``MusicGen`` wrapper with
        ``set_generation_params(duration=10.0)`` applied as a sane default.

    Raises
    ------
    ValueError
        If ``size`` is not a known model size.
    AssertionError
        If the loaded model violates the assumptions this codebase is built
        on (delayed pattern with delays [0,1,2,3]; text-only model with no
        prepend conditioning).
    """
    if size not in _SIZES:
        raise ValueError(f"size must be one of {_SIZES}, got {size!r}")
    from audiocraft.models import MusicGen  # lazy: heavy dependency
    from audiocraft.modules.codebooks_patterns import DelayedPatternProvider

    name = f"facebook/musicgen-{size}"
    logger.info("Loading %s on %s", name, device)
    model = MusicGen.get_pretrained(name, device=device)

    provider = model.lm.pattern_provider
    assert isinstance(provider, DelayedPatternProvider), (
        f"expected DelayedPatternProvider, got {type(provider).__name__}; "
        "this codebase only supports the released delayed-pattern MusicGen models")
    assert list(provider.delays) == [0, 1, 2, 3], (
        f"expected delays [0, 1, 2, 3], got {list(provider.delays)}")
    assert len(model.lm.fuser.fuse2cond["prepend"]) == 0, (
        "expected a text-only model with no prepend conditioning "
        "(sequence step indices would be shifted otherwise)")

    model.lm.eval()
    model.compression_model.eval()
    model.set_generation_params(duration=10.0)
    logger.info("Loaded %s: %s", name, model_geometry(model))
    return model


def model_geometry(model) -> ModelGeometry:
    """Extract the static geometry of a (possibly mocked) MusicGen model.

    Accepts either the ``MusicGen`` wrapper (has ``.lm``) or a bare LM-like
    module (has ``.transformer.layers``). ``frame_rate``/``sample_rate`` are
    read from the wrapper and default to MusicGen's 50 Hz / 32 kHz when the
    attributes are missing (bare LM mocks).
    """
    lm = _get_lm(model)
    layers = lm.transformer.layers
    self_attn = layers[0].self_attn
    n_heads = int(self_attn.num_heads)
    d_model = int(self_attn.embed_dim)
    n_q = getattr(lm, "num_codebooks", None)
    if n_q is None:
        n_q = lm.n_q
    return ModelGeometry(
        n_layers=len(layers),
        n_heads=n_heads,
        d_model=d_model,
        d_head=d_model // n_heads,
        n_q=int(n_q),
        card=int(lm.card),
        frame_rate=int(round(float(getattr(model, "frame_rate", 50)))),
        sample_rate=int(getattr(model, "sample_rate", 32000)),
    )


def delay_map_for(model) -> DelayMap:
    """Build the :class:`DelayMap` matching the model's pattern provider."""
    lm = _get_lm(model)
    return DelayMap.from_audiocraft(lm.pattern_provider)


def text_condition_tensors(model, descriptions: tp.List[tp.Optional[str]]) -> tp.Dict[str, tp.Any]:
    """Precompute condition tensors for a batch of text descriptions.

    Parameters
    ----------
    model : MusicGen
        Loaded MusicGen wrapper.
    descriptions : list of str or None
        One entry per batch item; ``None`` entries produce the null (dropped)
        condition for that item, mirroring
        ``ClassifierFreeGuidanceDropout(p=1.0)``.

    Returns
    -------
    ConditionTensors
        The dict produced by ``lm.condition_provider`` — pass it as
        ``condition_tensors`` to ``lm.compute_predictions`` / ``lm.forward``.
    """
    from audiocraft.modules.conditioners import ConditioningAttributes  # lazy

    lm = _get_lm(model)
    attributes = [ConditioningAttributes(text={"description": d}) for d in descriptions]
    tokenized = lm.condition_provider.tokenize(attributes)
    with torch.no_grad(), _autocast(model):
        return lm.condition_provider(tokenized)


def null_condition_tensors(model, batch_size: int) -> tp.Dict[str, tp.Any]:
    """Null (unconditional) condition tensors for ``batch_size`` items.

    Equivalent to conditioning attributes after
    ``ClassifierFreeGuidanceDropout(p=1.0)``: every description is ``None``.
    """
    return text_condition_tensors(model, [None] * batch_size)
