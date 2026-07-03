"""Loop-artifact detection: token n-gram repetition x envelope autocorrelation.

Research plan section 7, "loop artifact rate": a generation counts as a
degenerate loop only when BOTH the token n-gram repetition rate over the
final seconds AND the audio-envelope autocorrelation collapse fire (joint
criterion — either signal alone is common in healthy, merely repetitive
music).

Pure NumPy; ``codes`` follow the repo convention ``[K, T]`` int EnCodec
codes at 50 Hz, ``wav`` is mono float audio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["token_ngram_repetition", "audio_autocorr_collapse", "LoopResult", "loop_score"]

#: RMS envelope rate (Hz) used by audio_autocorr_collapse; matches the 50 Hz
#: EnCodec frame rate so envelope lags are directly comparable to frame lags.
ENV_RATE_HZ = 50.0
_EPS = 1e-12


def token_ngram_repetition(
    codes: np.ndarray,
    n: int = 4,
    tail_frames: int = 250,
    codebooks: str = "first",
) -> float:
    """Fraction of repeated token n-grams in the tail of a generation.

    Over the last ``tail_frames`` frames, every run of ``n`` consecutive
    frame symbols is hashed; the score is the fraction of runs whose tuple
    already occurred at an earlier position *within the tail*. A perfectly
    tiled tail scores close to 1 (only each distinct n-gram's first
    occurrence is new); random codes score near 0.

    Parameters
    ----------
    codes : np.ndarray
        ``[K, T]`` integer EnCodec codes (frame-aligned, not delay-interleaved).
    n : int
        n-gram length in frames (default 4 = 80 ms at 50 Hz).
    tail_frames : int
        Number of trailing frames to analyze (default 250 = 5 s at 50 Hz).
        Shorter sequences use all available frames.
    codebooks : {'first', 'all'}
        'first': a frame's symbol is its codebook-0 token (row 0).
        'all': a frame's symbol stacks all K codebook tokens, so a repeat
        requires every codebook to match.

    Returns
    -------
    float
        Repeated-n-gram fraction in [0, 1]; 0.0 when the tail is too short
        to form at least one n-gram.
    """
    codes = np.asarray(codes)
    if codes.ndim != 2:
        raise ValueError(f"codes must be [K, T], got shape {codes.shape}")
    if n < 1:
        raise ValueError("n must be >= 1")
    if tail_frames < 1:
        raise ValueError("tail_frames must be >= 1")
    tail = codes[:, -tail_frames:]
    if codebooks == "first":
        symbols = [int(v) for v in tail[0]]
    elif codebooks == "all":
        symbols = [tuple(int(v) for v in tail[:, t]) for t in range(tail.shape[1])]
    else:
        raise ValueError(f"codebooks must be 'first' or 'all', got {codebooks!r}")
    total = len(symbols) - n + 1
    if total <= 0:
        return 0.0
    seen = set()
    repeated = 0
    for i in range(total):
        gram = tuple(symbols[i:i + n])
        if gram in seen:
            repeated += 1
        else:
            seen.add(gram)
    return repeated / total


def _as_mono(wav: np.ndarray) -> np.ndarray:
    """Coerce audio to a 1-D mono float array (averages channel axes)."""
    wav = np.asarray(wav, dtype=np.float64).squeeze()
    if wav.ndim == 2:
        # Assume the shorter axis is channels ([C, N] or [N, C]).
        wav = wav.mean(axis=int(np.argmax(wav.shape) != 1))
    if wav.ndim != 1:
        raise ValueError(f"wav must be mono-reducible, got shape {np.asarray(wav).shape}")
    return wav


def audio_autocorr_collapse(
    wav: np.ndarray,
    sr: int,
    tail_s: float = 5.0,
    min_lag_s: float = 0.2,
    max_lag_s: float = 2.5,
) -> "tuple[float, float]":
    """Envelope-autocorrelation collapse score for the tail of a generation.

    The last ``tail_s`` seconds are reduced to a 50 Hz RMS envelope
    (non-overlapping ``sr // 50``-sample windows). The envelope is
    mean-subtracted and variance-normalized, and its *biased* normalized
    autocorrelation ``ac[l] = (1/N) * sum_t e[t] e[t+l] / var`` is scanned
    over lags between ``min_lag_s`` and ``max_lag_s`` (0.2–2.5 s: shorter
    lags are note-level periodicity, longer ones exceed the tail). A near-1
    peak means the tail's dynamics are an (almost) exact short loop. Note
    the biased normalization shrinks the achievable peak by ``(N - l) / N``
    at lag ``l``, penalizing loops long relative to the tail.

    Parameters
    ----------
    wav : np.ndarray
        Mono audio (``[T]``, ``[1, T]`` etc. are squeezed; channels averaged).
    sr : int
        Sample rate in Hz.
    tail_s : float
        Length of the analyzed tail in seconds (default 5.0).
    min_lag_s, max_lag_s : float
        Lag search range in seconds.

    Returns
    -------
    (float, float)
        ``(peak, lag_s)``: maximum autocorrelation in the lag range and the
        lag (seconds) attaining it. ``(0.0, 0.0)`` when the tail is too
        short or the envelope is constant (zero variance).
    """
    if sr <= 0:
        raise ValueError("sr must be positive")
    wav = _as_mono(wav)
    n_tail = int(round(tail_s * sr))
    x = wav[-n_tail:]
    hop = max(int(round(sr / ENV_RATE_HZ)), 1)
    n_frames = len(x) // hop
    min_lag = int(np.ceil(min_lag_s * ENV_RATE_HZ))
    max_lag = int(np.floor(max_lag_s * ENV_RATE_HZ))
    if n_frames <= min_lag:
        return 0.0, 0.0
    env = np.sqrt(np.mean(x[: n_frames * hop].reshape(n_frames, hop) ** 2, axis=1))
    e = env - env.mean()
    var = float(np.mean(e ** 2))
    if var <= _EPS:
        return 0.0, 0.0
    N = len(e)
    ac = np.correlate(e, e, mode="full")[N - 1:] / (N * var)  # biased, ac[0] = 1
    max_lag = min(max_lag, N - 1)
    if min_lag > max_lag:
        return 0.0, 0.0
    seg = ac[min_lag:max_lag + 1]
    k = int(np.argmax(seg))
    return float(seg[k]), float((min_lag + k) / ENV_RATE_HZ)


@dataclass
class LoopResult:
    """Joint loop-artifact verdict for one generation.

    Attributes
    ----------
    ngram_rate : float
        Token n-gram repetition rate over the tail (:func:`token_ngram_repetition`).
    autocorr_peak : float
        Envelope autocorrelation peak over the tail (:func:`audio_autocorr_collapse`).
    is_loop : bool
        Joint criterion: both signals above their thresholds.
    autocorr_lag_s : float
        Lag (seconds) of the autocorrelation peak (0.0 if degenerate).
    """

    ngram_rate: float
    autocorr_peak: float
    is_loop: bool
    autocorr_lag_s: float = 0.0


def loop_score(
    codes: np.ndarray,
    wav: np.ndarray,
    sr: int,
    ngram_thresh: float = 0.8,
    autocorr_thresh: float = 0.9,
) -> LoopResult:
    """Classify a generation as a loop artifact (joint criterion).

    Per the research plan, a sample is a loop only if BOTH conditions fire
    on the final 5 seconds: token n-gram repetition rate ``> ngram_thresh``
    (codebook 0, n=4, 250 frames) AND envelope autocorrelation peak
    ``> autocorr_thresh``. Requiring both avoids flagging healthy repetitive
    music (which repeats tokens but keeps evolving dynamics) or steady tones
    (periodic envelope, varied tokens).

    Parameters
    ----------
    codes : np.ndarray
        ``[K, T]`` frame-aligned EnCodec codes of the generation.
    wav : np.ndarray
        Mono audio of the same generation.
    sr : int
        Sample rate of ``wav``.
    ngram_thresh, autocorr_thresh : float
        Strict lower thresholds for the two signals (defaults 0.8 / 0.9).

    Returns
    -------
    LoopResult
        Component scores plus the joint ``is_loop`` verdict.
    """
    ngram_rate = token_ngram_repetition(codes, n=4, tail_frames=250, codebooks="first")
    autocorr_peak, lag_s = audio_autocorr_collapse(wav, sr, tail_s=5.0)
    is_loop = bool(ngram_rate > ngram_thresh and autocorr_peak > autocorr_thresh)
    return LoopResult(
        ngram_rate=float(ngram_rate),
        autocorr_peak=float(autocorr_peak),
        is_loop=is_loop,
        autocorr_lag_s=float(lag_s),
    )
