"""Forward hooks for attention capture and per-head interventions.

Everything here works on audiocraft's ``StreamingMultiheadAttention`` custom
path (the one used by all released MusicGen models — ``memory_efficient=True``
implies ``self.custom``; see ``docs/audiocraft_api_notes.md`` §3):

* the module input (``query`` = ``norm1(x)``) together with the module's own
  packed ``in_proj_weight`` fully determines q/k, so per-head attention
  probabilities can be recomputed *exactly* even though the model's
  memory-efficient kernel never materializes them;
* the input of ``self_attn.out_proj`` is ``[B, S, H * d_head]`` with
  head-contiguous channel slices ``[h*d_head, (h+1)*d_head)`` — the single
  hook point for per-head output capture, ablation, scaling and patching.

All hook managers are context managers that always remove their hooks on
exit and accept either a ``MusicGen`` wrapper (``model.lm``) or a bare
LM-like module (``.transformer.layers``) so unit tests can use light mocks.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import typing as tp

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["AttentionCapture", "HeadOutputCapture", "HeadIntervention",
           "HeadInterventions", "resolve_layers"]

logger = logging.getLogger(__name__)


def _get_lm(model) -> tp.Any:
    return model.lm if hasattr(model, "lm") else model


def resolve_layers(model, layers: tp.Optional[tp.Sequence[int]] = None
                   ) -> tp.Dict[int, tp.Any]:
    """Map layer index -> transformer layer module (all layers when None)."""
    all_layers = list(_get_lm(model).transformer.layers)
    if layers is None:
        layers = range(len(all_layers))
    out = {}
    for idx in layers:
        if not 0 <= idx < len(all_layers):
            raise ValueError(f"layer index {idx} out of range "
                             f"[0, {len(all_layers)})")
        out[int(idx)] = all_layers[idx]
    return out


def _check_supported_attention(attn: tp.Any) -> None:
    """Refuse module variants whose q/k recomputation would be wrong."""
    if getattr(attn, "rope", None) is not None:
        raise RuntimeError("RoPE attention is not supported by the hooks "
                           "(recomputed q/k would miss the rotation)")
    if getattr(attn, "qk_layer_norm", False):
        raise RuntimeError("qk_layer_norm attention is not supported")
    if getattr(attn, "kv_repeat", 1) != 1:
        raise RuntimeError("kv_repeat != 1 is not supported")
    if getattr(attn, "cross_attention", False):
        raise RuntimeError("hook attached to a cross-attention module; "
                           "only self-attention is supported")


def _head_geometry(attn: tp.Any) -> tp.Tuple[int, int]:
    """(num_heads, d_head) of a StreamingMultiheadAttention-like module."""
    H = int(attn.num_heads)
    D = int(attn.embed_dim)
    return H, D // H


class AttentionCapture:
    """Recompute and store per-head self-attention probabilities.

    Teacher-forcing only: the capture raises if the module holds streaming
    state (during autoregressive generation q/k of past steps are not part of
    the current call's input, so a per-call recomputation would be wrong).

    Parameters
    ----------
    model : MusicGen | LM-like
        Model whose ``transformer.layers[i].self_attn`` get hooked.
    layers : sequence of int, optional
        Layer indices to capture (default: all).
    dtype : numpy dtype, optional
        Storage dtype for full attention matrices (default float16; the
        softmax itself is always computed in float32).
    reduce : callable, optional
        ``reduce(layer_index, attn_f32) -> Any`` where ``attn_f32`` is the
        ``[B, H, S, S]`` float32 NumPy array. When given, only its return
        value is kept (``.reduced[layer]``) and the raw attention is
        discarded immediately — the memory-bounded path used by the
        screening script.

    Attributes
    ----------
    attention : dict[int, np.ndarray]
        ``layer -> [B, H, S, S]`` (only when ``reduce`` is None). If the
        model is called several times under one capture, the LAST call wins.
    reduced : dict[int, Any]
        ``layer -> reduce(...)`` return value (only when ``reduce`` given).
    """

    def __init__(self, model, layers: tp.Optional[tp.Sequence[int]] = None,
                 dtype=np.float16,
                 reduce: tp.Optional[tp.Callable[[int, np.ndarray], tp.Any]] = None):
        self._layers = resolve_layers(model, layers)
        self._dtype = np.dtype(dtype)
        self._reduce = reduce
        self._handles: tp.List[tp.Any] = []
        self.attention: tp.Dict[int, np.ndarray] = {}
        self.reduced: tp.Dict[int, tp.Any] = {}

    # ------------------------------------------------------------------
    def _make_hook(self, layer_idx: int, attn_module) -> tp.Callable:
        def hook(module, args):
            if getattr(module, "_streaming_state", None):
                raise RuntimeError(
                    "AttentionCapture used during streaming generation; "
                    "it only supports teacher-forcing forwards")
            query = args[0]
            probs = self._recompute(module, query)
            arr = probs.numpy()
            if self._reduce is not None:
                self.reduced[layer_idx] = self._reduce(layer_idx, arr)
            else:
                self.attention[layer_idx] = arr.astype(self._dtype)
            return None

        return hook

    @staticmethod
    def _recompute(module, query: torch.Tensor) -> torch.Tensor:
        """Per-head causal softmax probabilities, float32 CPU ``[B,H,S,S]``.

        Mirrors the custom self-attention path of audiocraft's
        ``StreamingMultiheadAttention``: packed qkv projection with channel
        order ``(3, H, d_head)`` and scores ``q @ k^T / sqrt(d_head)`` under
        a causal mask. The model runs the softmax in its autocast dtype
        (float16 on GPU); we upcast to float32, which changes probabilities
        only at fp16 rounding level.
        """
        _check_supported_attention(module)
        H, d_head = _head_geometry(module)
        D = int(module.embed_dim)
        # The forward usually runs under the model's autocast; disable it so
        # the recomputation genuinely happens in float32.
        with torch.no_grad(), torch.autocast(device_type=query.device.type,
                                             enabled=False):
            x = query.detach().float()
            w = module.in_proj_weight.detach().float()
            b = module.in_proj_bias
            b = b.detach().float() if b is not None else None
            projected = F.linear(x, w, b)
            q = projected[..., :D]
            k = projected[..., D:2 * D]
            B, S, _ = q.shape
            q = q.view(B, S, H, d_head).transpose(1, 2)  # [B,H,S,d]
            k = k.view(B, S, H, d_head).transpose(1, 2)
            scores = q @ k.transpose(-1, -2) / math.sqrt(d_head)
            causal = torch.ones(S, S, dtype=torch.bool,
                                device=scores.device).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
            return torch.softmax(scores, dim=-1).cpu()

    # ------------------------------------------------------------------
    def __enter__(self) -> "AttentionCapture":
        self.attention.clear()
        self.reduced.clear()
        for idx, layer in self._layers.items():
            attn = layer.self_attn
            _check_supported_attention(attn)
            self._handles.append(
                attn.register_forward_pre_hook(self._make_hook(idx, attn)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


class HeadOutputCapture:
    """Capture per-head attention outputs (out_proj inputs) per layer.

    Attributes
    ----------
    z : dict[int, torch.Tensor]
        ``layer -> [B, S, H, d_head]`` on CPU, in the runtime dtype (float16
        under autocast). If the model runs several forwards under one
        capture, the LAST forward wins.
    """

    def __init__(self, model, layers: tp.Optional[tp.Sequence[int]] = None):
        self._layers = resolve_layers(model, layers)
        self._handles: tp.List[tp.Any] = []
        self.z: tp.Dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int, n_heads: int) -> tp.Callable:
        def hook(module, args):
            x = args[0]                     # [B, S, H * d_head]
            B, S, D = x.shape
            d_head = D // n_heads
            self.z[layer_idx] = (x.detach()
                                 .reshape(B, S, n_heads, d_head)
                                 .cpu())
            return None

        return hook

    def __enter__(self) -> "HeadOutputCapture":
        self.z.clear()
        for idx, layer in self._layers.items():
            attn = layer.self_attn
            H, _ = _head_geometry(attn)
            self._handles.append(
                attn.out_proj.register_forward_pre_hook(
                    self._make_hook(idx, H)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


@dataclass
class HeadIntervention:
    """One per-head edit of the attention output.

    Attributes
    ----------
    layer, head : int
        Target head.
    mode : str
        ``'zero'`` | ``'mean'`` | ``'scale'`` | ``'patch'``.
    value : float | torch.Tensor | np.ndarray, optional
        ``'scale'``: multiplicative gamma (float).
        ``'mean'``: replacement vector ``[d_head]``.
        ``'patch'``: cached activations ``[S_total, d_head]`` (broadcast over
        the batch — including a CFG-doubled batch) or ``[B, S_total, d_head]``
        (must match the runtime batch, or be tiled 2x for CFG). Absolute
        sequence steps index into ``S_total``; steps beyond it are left
        untouched.
    steps : np.ndarray, optional
        Absolute sequence steps to affect (``None`` = every step).
    """

    layer: int
    head: int
    mode: str
    value: tp.Any = None
    steps: tp.Optional[np.ndarray] = None

    def __post_init__(self):
        if self.mode not in ("zero", "mean", "scale", "patch"):
            raise ValueError(f"unknown intervention mode {self.mode!r}")
        if self.mode == "scale" and not isinstance(self.value, (int, float)):
            raise ValueError("'scale' needs a float gamma value")
        if self.mode in ("mean", "patch") and self.value is None:
            raise ValueError(f"'{self.mode}' needs a value tensor")
        if self.steps is not None:
            self.steps = np.asarray(self.steps, dtype=np.int64)


class HeadInterventions:
    """Apply a list of :class:`HeadIntervention` via out_proj pre-hooks.

    Step tracking: each hooked layer keeps a counter of absolute sequence
    steps, advanced by ``x.shape[1]`` per out_proj call. In teacher forcing a
    single call covers steps ``0..S-1``; in streaming generation the first
    call may cover several steps (special token + audio-prompt steps) and
    every later call covers one — the counter handles both. Entering the
    context (or calling :meth:`reset`) zeroes the counters, so create/enter a
    fresh context (or ``reset()``) per generation run.

    CFG note: interventions are applied to ALL batch rows, so a CFG-doubled
    ``[cond; uncond]`` batch gets the same edit in both halves.
    """

    def __init__(self, model, interventions: tp.Sequence[HeadIntervention]):
        self._interventions = list(interventions)
        by_layer: tp.Dict[int, tp.List[HeadIntervention]] = {}
        for iv in self._interventions:
            by_layer.setdefault(int(iv.layer), []).append(iv)
        self._by_layer = by_layer
        self._layers = resolve_layers(model, sorted(by_layer)) if by_layer else {}
        self._offsets: tp.Dict[int, int] = {}
        self._handles: tp.List[tp.Any] = []
        # validate head indices eagerly
        for idx, layer in self._layers.items():
            H, d_head = _head_geometry(layer.self_attn)
            for iv in by_layer[idx]:
                if not 0 <= iv.head < H:
                    raise ValueError(f"head {iv.head} out of range for layer "
                                     f"{idx} with {H} heads")

    def reset(self) -> None:
        """Zero the absolute-step counters (call between generation runs)."""
        self._offsets = {idx: 0 for idx in self._layers}

    # ------------------------------------------------------------------
    @staticmethod
    def _as_tensor(value, x: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(np.asarray(value) if isinstance(value, np.ndarray)
                            else value)
        return t.to(device=x.device, dtype=x.dtype)

    def _apply(self, iv: HeadIntervention, x: torch.Tensor,
               abs_steps: torch.Tensor, d_head: int) -> None:
        """Mutate ``x[:, sel, head_slice]`` in place according to ``iv``."""
        B, T, _ = x.shape
        sl = slice(iv.head * d_head, (iv.head + 1) * d_head)
        if iv.steps is None:
            sel = torch.ones(T, dtype=torch.bool, device=x.device)
        else:
            steps_t = torch.as_tensor(iv.steps, device=x.device)
            sel = torch.isin(abs_steps, steps_t)
        if iv.mode == "patch":
            val = self._as_tensor(iv.value, x)
            s_total = val.shape[-2]
            sel = sel & (abs_steps < s_total)
        if not bool(sel.any()):
            return
        if iv.mode == "zero":
            x[:, sel, sl] = 0
        elif iv.mode == "scale":
            x[:, sel, sl] *= float(iv.value)
        elif iv.mode == "mean":
            vec = self._as_tensor(iv.value, x).reshape(-1)
            if vec.numel() != d_head:
                raise ValueError(f"'mean' vector has {vec.numel()} dims, "
                                 f"expected d_head={d_head}")
            x[:, sel, sl] = vec
        elif iv.mode == "patch":
            val = self._as_tensor(iv.value, x)
            pos = abs_steps[sel]
            if val.ndim == 2:                       # [S_total, d] -> broadcast
                x[:, sel, sl] = val[pos][None, :, :]
            elif val.ndim == 3:                     # [B_v, S_total, d]
                if val.shape[0] == B:
                    x[:, sel, sl] = val[:, pos, :]
                elif B == 2 * val.shape[0]:         # CFG-doubled batch
                    x[:, sel, sl] = val[:, pos, :].repeat(2, 1, 1)
                else:
                    raise ValueError(
                        f"patch batch {val.shape[0]} incompatible with "
                        f"runtime batch {B}")
            else:
                raise ValueError("patch value must be [S, d] or [B, S, d]")

    def _make_hook(self, layer_idx: int, d_head: int) -> tp.Callable:
        def hook(module, args):
            x = args[0]
            B, T, _ = x.shape
            offset = self._offsets[layer_idx]
            self._offsets[layer_idx] = offset + T
            abs_steps = torch.arange(offset, offset + T, device=x.device)
            x = x.clone()
            for iv in self._by_layer[layer_idx]:
                self._apply(iv, x, abs_steps, d_head)
            return (x,) + tuple(args[1:])

        return hook

    # ------------------------------------------------------------------
    def __enter__(self) -> "HeadInterventions":
        self.reset()
        for idx, layer in self._layers.items():
            _, d_head = _head_geometry(layer.self_attn)
            self._handles.append(
                layer.self_attn.out_proj.register_forward_pre_hook(
                    self._make_hook(idx, d_head)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
