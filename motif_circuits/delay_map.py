"""Delay-aware frame<->step coordinate mapping for MusicGen-style delayed
codebook interleaving.

MusicGen generates EnCodec tokens with a delayed pattern over ``n_q``
codebooks: the token of *frame* ``t`` (50 Hz audio frame) for codebook ``q``
appears at transformer *sequence step*::

    s = special_offset + t + delays[q]        (MusicGen: 1 + t + q)

The initial step ``s = 0`` holds the special (start) token. All induction
scores, attention slicing and activation-patching targets in this repository
are defined on the *frame* axis and converted to the step axis through this
module. See ``docs/audiocraft_api_notes.md`` for the verified upstream
semantics.

The implementation is intentionally NumPy-only so that it can be unit-tested
without torch/audiocraft; ``verify_against_audiocraft`` cross-checks it
against the real ``DelayedPatternProvider`` when audiocraft is importable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import typing as tp

import numpy as np

__all__ = ["DelayMap", "MUSICGEN_DELAY_MAP"]


@dataclass(frozen=True)
class DelayMap:
    """Bidirectional frame<->step mapping for a delayed codebook pattern.

    Args:
        n_q: number of codebooks (MusicGen: 4).
        delays: per-codebook delay in steps (MusicGen: (0, 1, 2, 3)).
        special_offset: number of leading special-token steps (MusicGen: 1).
    """

    n_q: int = 4
    delays: tp.Tuple[int, ...] = (0, 1, 2, 3)
    special_offset: int = 1

    def __post_init__(self):
        if len(self.delays) != self.n_q:
            raise ValueError(f"delays must have length n_q={self.n_q}")
        if list(self.delays) != sorted(self.delays):
            raise ValueError("delays must be sorted ascending (audiocraft invariant)")
        if self.special_offset < 0:
            raise ValueError("special_offset must be >= 0")

    # ------------------------------------------------------------------
    # scalar / vectorized coordinate transforms
    # ------------------------------------------------------------------
    @property
    def max_delay(self) -> int:
        return max(self.delays)

    def step(self, t: tp.Union[int, np.ndarray], q: int) -> tp.Union[int, np.ndarray]:
        """Sequence step holding the token of frame ``t`` for codebook ``q``.

        Vectorized over ``t`` (accepts scalars or arrays; negative frames are
        rejected).
        """
        self._check_q(q)
        t_arr = np.asarray(t)
        if np.any(t_arr < 0):
            raise ValueError("frame index must be >= 0")
        out = t_arr + self.delays[q] + self.special_offset
        return int(out) if np.isscalar(t) or t_arr.ndim == 0 else out

    def frame(self, s: tp.Union[int, np.ndarray], q: int) -> tp.Union[int, np.ndarray]:
        """Frame whose codebook-``q`` token sits at step ``s``.

        Returns -1 where the step does not hold a codebook-``q`` token
        (special-token steps / the initial delay ramp).
        """
        self._check_q(q)
        s_arr = np.asarray(s)
        out = s_arr - self.delays[q] - self.special_offset
        out = np.where(s_arr < self.special_offset, -1, out)
        out = np.where(out < 0, -1, out)
        return int(out) if np.isscalar(s) or s_arr.ndim == 0 else out

    def coords_at_step(self, s: int, T: tp.Optional[int] = None) -> tp.List[tp.Tuple[int, int]]:
        """All ``(t, q)`` token coordinates present at sequence step ``s``.

        If ``T`` is given, frames ``t >= T`` are excluded (the trailing ramp of
        a length-``T`` sequence).
        """
        coords = []
        for q in range(self.n_q):
            t = s - self.delays[q] - self.special_offset
            if s >= self.special_offset and t >= 0 and (T is None or t < T):
                coords.append((t, q))
        return coords

    # ------------------------------------------------------------------
    # sequence lengths & validity (mirrors audiocraft Pattern semantics)
    # ------------------------------------------------------------------
    def seq_len(self, T: int, keep_only_valid_steps: bool = False) -> int:
        """Length S of the interleaved sequence for T frames.

        ``keep_only_valid_steps=False`` (generation): full layout including the
        trailing delay ramp. ``True`` (audiocraft ``compute_predictions``):
        truncated to steps whose content is fully defined.
        """
        full = self.special_offset + T + self.max_delay
        if keep_only_valid_steps:
            return full - self.max_delay  # == special_offset + T
        return full

    def is_valid_step(self, t: int, q: int, T: int, keep_only_valid_steps: bool = True) -> bool:
        """Whether the token (t, q) survives in a length-T teacher-forcing
        sequence (it is dropped when its step falls beyond the valid layout)."""
        if t < 0 or t >= T:
            return False
        s = self.step(t, q)
        return s < self.seq_len(T, keep_only_valid_steps=keep_only_valid_steps)

    def step_index_matrix(self, T: int) -> np.ndarray:
        """[n_q, T] matrix of sequence steps: entry (q, t) = step(t, q)."""
        t = np.arange(T)
        return np.stack([t + self.delays[q] + self.special_offset for q in range(self.n_q)])

    # ------------------------------------------------------------------
    # attention slicing / frame aggregation
    # ------------------------------------------------------------------
    def slice_attention(self, attn: np.ndarray, T: int, q_query: int, q_key: int,
                        fill: float = np.nan) -> np.ndarray:
        """Extract the frame-coordinate attention block for a codebook pair.

        Args:
            attn: step-level attention weights ``[..., S_q, S_k]`` (query steps
                x key steps; the last two axes must cover the steps of the
                requested tokens).
            T: number of frames.
            q_query, q_key: codebook of the query / key tokens.
            fill: value used where the query or key step is out of range
                (e.g. truncated valid layout).
        Returns:
            ``[..., T, T]`` array: entry (tq, tk) = attention from the step
            holding (tq, q_query) to the step holding (tk, q_key).
        """
        S_q, S_k = attn.shape[-2], attn.shape[-1]
        qs = self.step_index_matrix(T)[q_query]   # [T]
        ks = self.step_index_matrix(T)[q_key]     # [T]
        q_ok = qs < S_q
        k_ok = ks < S_k
        out = np.full(attn.shape[:-2] + (T, T), fill, dtype=attn.dtype)
        qi = qs[q_ok]
        ki = ks[k_ok]
        block = attn[..., qi[:, None], ki[None, :]]
        out[..., np.flatnonzero(q_ok)[:, None], np.flatnonzero(k_ok)[None, :]] = block
        return out

    def aggregate_frames(self, attn: np.ndarray, T: int,
                         query_codebooks: tp.Optional[tp.Sequence[int]] = None,
                         key_codebooks: tp.Optional[tp.Sequence[int]] = None) -> np.ndarray:
        """Frame-level attention: sum over key codebooks, mean over query codebooks.

        For a query token, its attention distribution sums to 1 over *steps*;
        summing the steps of a key frame's codebooks yields the total mass on
        that key frame, and averaging over the query token's codebooks yields a
        per-frame-pair summary. Entries where a query step is missing are NaN.

        Returns ``[..., T, T]``.
        """
        if query_codebooks is None:
            query_codebooks = range(self.n_q)
        if key_codebooks is None:
            key_codebooks = range(self.n_q)
        acc = None
        for qq in query_codebooks:
            per_query = None
            for qk in key_codebooks:
                block = self.slice_attention(attn, T, qq, qk, fill=np.nan)
                per_query = block if per_query is None else per_query + block
            acc = per_query if acc is None else acc + per_query
        return acc / len(list(query_codebooks))

    # ------------------------------------------------------------------
    # patch-target helpers
    # ------------------------------------------------------------------
    def steps_for_frames(self, frames: tp.Sequence[int],
                         codebooks: tp.Optional[tp.Sequence[int]] = None,
                         S: tp.Optional[int] = None) -> np.ndarray:
        """Flat, sorted, unique array of steps holding the given frames'
        tokens (optionally restricted to ``codebooks`` and to steps < S)."""
        if codebooks is None:
            codebooks = range(self.n_q)
        steps = []
        for q in codebooks:
            for t in frames:
                s = self.step(int(t), q)
                if S is None or s < S:
                    steps.append(s)
        return np.unique(np.asarray(steps, dtype=np.int64))

    # ------------------------------------------------------------------
    # audiocraft interop
    # ------------------------------------------------------------------
    @classmethod
    def from_audiocraft(cls, pattern_provider) -> "DelayMap":
        """Build a DelayMap from an audiocraft ``DelayedPatternProvider``.

        Raises if the provider is not a plain delayed pattern (flatten_first /
        empty_initial are unsupported by this analysis codebase).
        """
        if getattr(pattern_provider, "flatten_first", 0):
            raise ValueError("flatten_first != 0 is not supported")
        empty_initial = getattr(pattern_provider, "empty_initial", 0)
        if empty_initial < 0:
            raise ValueError("empty_initial < 0 (no special token) is not supported")
        return cls(n_q=pattern_provider.n_q,
                   delays=tuple(pattern_provider.delays),
                   special_offset=1 + empty_initial)

    def verify_against_audiocraft(self, pattern_provider, T: int = 37) -> None:
        """Assert exact equivalence with an audiocraft pattern for length T.

        Note: audiocraft's raw ``Pattern.layout`` keeps "ghost" coords with
        ``t >= T`` in the trailing delay ramp and only filters them at scatter
        time (``coords.t < timesteps``); we compare against the filtered view,
        which is what determines actual token placement and masking.
        """
        pattern = pattern_provider.get_pattern(T)
        layout = pattern.layout
        assert len(layout) == self.seq_len(T), \
            f"layout length {len(layout)} != {self.seq_len(T)}"
        for s, coords in enumerate(layout):
            ours = self.coords_at_step(s, T=T)
            theirs = sorted((c.t, c.q) for c in coords if c.t < T)
            assert sorted(ours) == theirs, f"step {s}: {ours} != {theirs}"
        # valid layout truncation semantics
        assert len(pattern.valid_layout) == self.seq_len(T, keep_only_valid_steps=True)

    def _check_q(self, q: int) -> None:
        if not 0 <= q < self.n_q:
            raise ValueError(f"codebook index {q} out of range [0, {self.n_q})")


#: The mapping used by all released MusicGen text-to-music models.
MUSICGEN_DELAY_MAP = DelayMap(n_q=4, delays=(0, 1, 2, 3), special_offset=1)
