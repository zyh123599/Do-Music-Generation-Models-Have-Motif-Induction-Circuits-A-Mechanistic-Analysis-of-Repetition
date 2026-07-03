"""Chroma and onset-envelope features on the 50 Hz EnCodec frame grid.

Features are computed with ``hop = sr // frame_rate`` (640 samples at
32 kHz / 50 Hz) so that feature frame ``t`` lines up with EnCodec frame ``t``.
``librosa`` is imported lazily; a pure scipy fallback (``method='internal'``)
implements the same contract, including the pitch-class convention
(chroma bin 0 = C, A440 reference), so both methods agree on the per-frame
argmax pitch class for tonal signals.
"""
from __future__ import annotations

import logging
import typing as tp

import numpy as np
import scipy.signal

__all__ = ["chroma_features", "onset_envelope"]

logger = logging.getLogger(__name__)

_DEFAULT_N_FFT = 2048


def _n_frames(n_samples: int, hop: int) -> int:
    """Target number of feature frames for a signal of ``n_samples``."""
    return int(round(n_samples / hop))


def _fit_length(feat: np.ndarray, T: int) -> np.ndarray:
    """Trim or zero-pad ``feat`` along axis 0 to exactly ``T`` frames."""
    if feat.shape[0] == T:
        return feat
    if feat.shape[0] > T:
        return feat[:T]
    pad = [(0, T - feat.shape[0])] + [(0, 0)] * (feat.ndim - 1)
    return np.pad(feat, pad, mode="constant")


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize rows; all-zero rows are left as zeros (zero-row guard)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.where(norms > eps, x / np.maximum(norms, eps), 0.0)


def _stft_magnitude(wav: np.ndarray, sr: int, hop: int,
                    n_fft: int = _DEFAULT_N_FFT) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Centered STFT magnitude via scipy.

    Returns
    -------
    freqs : np.ndarray
        Bin center frequencies in Hz, shape ``[n_fft // 2 + 1]``.
    mag : np.ndarray
        Magnitude spectrogram, shape ``[n_frames, n_fft // 2 + 1]``
        (time-major). ``boundary='zeros'`` centers window ``t`` on sample
        ``t * hop``, matching librosa ``center=True``.
    """
    freqs, _, Z = scipy.signal.stft(
        wav.astype(np.float64), fs=sr, window="hann", nperseg=n_fft,
        noverlap=n_fft - hop, boundary="zeros", padded=True)
    return freqs, np.abs(Z).T


def _internal_chroma(wav: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """Scipy-based chroma: fold STFT bins onto 12 pitch classes (A440).

    Each positive-frequency bin is assigned to its nearest equal-tempered
    semitone (MIDI ``69 + 12*log2(f/440)``) and folded to a pitch class with
    the librosa convention (bin 0 = C, since MIDI 60 % 12 == 0). Magnitudes
    are accumulated per class.
    """
    freqs, mag = _stft_magnitude(wav, sr, hop)
    # Skip DC and sub-audible bins that map to meaningless pitch classes.
    valid = freqs > 20.0
    midi = 69.0 + 12.0 * np.log2(freqs[valid] / 440.0)
    pcs = np.mod(np.round(midi).astype(int), 12)
    chroma = np.zeros((mag.shape[0], 12), dtype=np.float64)
    np.add.at(chroma.T, pcs, mag[:, valid].T)
    return chroma


def _librosa_chroma(wav: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """Librosa chroma_stft (time-major). ``tuning=0`` pins the A440 reference
    so the pitch-class convention matches the internal fallback."""
    import librosa

    c = librosa.feature.chroma_stft(
        y=wav.astype(np.float32), sr=sr, hop_length=hop,
        n_fft=_DEFAULT_N_FFT, center=True, tuning=0.0)
    return np.asarray(c, dtype=np.float64).T


def chroma_features(wav: np.ndarray, sr: int, frame_rate: int = 50,
                    method: str = "auto") -> np.ndarray:
    """Chromagram on the EnCodec frame grid.

    Parameters
    ----------
    wav : np.ndarray
        Mono waveform ``[n_samples]``.
    sr : int
        Sample rate in Hz.
    frame_rate : int, optional
        Target frame rate; ``hop = sr // frame_rate`` (640 at 32 kHz).
    method : {'auto', 'librosa', 'internal'}, optional
        'auto' uses librosa when importable and falls back to the internal
        scipy implementation; the other values force one backend.

    Returns
    -------
    np.ndarray
        ``[T, 12]`` float64 chroma, rows L2-normalized (all-zero rows stay
        zero), ``T = round(len(wav) / hop)``. Bin 0 = pitch class C.
    """
    wav = np.asarray(wav, dtype=np.float64).reshape(-1)
    hop = sr // frame_rate
    T = _n_frames(len(wav), hop)
    if method not in ("auto", "librosa", "internal"):
        raise ValueError(f"unknown chroma method: {method!r}")
    chroma = None
    if method in ("auto", "librosa"):
        try:
            chroma = _librosa_chroma(wav, sr, hop)
        except ImportError:
            if method == "librosa":
                raise
            logger.info("librosa unavailable; using internal chroma fallback")
    if chroma is None:
        chroma = _internal_chroma(wav, sr, hop)
    return _l2_normalize_rows(_fit_length(chroma, T))


def onset_envelope(wav: np.ndarray, sr: int, frame_rate: int = 50) -> np.ndarray:
    """Onset-strength envelope on the EnCodec frame grid, scaled to [0, 1].

    Uses ``librosa.onset.onset_strength`` when available; otherwise spectral
    flux (positive first difference of the log-magnitude STFT summed over
    frequency). Length is ``T = round(len(wav) / hop)``; the maximum is
    normalized to 1 (an all-zero envelope stays zero).

    Parameters
    ----------
    wav : np.ndarray
        Mono waveform ``[n_samples]``.
    sr : int
        Sample rate in Hz.
    frame_rate : int, optional
        Target frame rate; ``hop = sr // frame_rate``.

    Returns
    -------
    np.ndarray
        ``[T]`` float64 envelope in [0, 1].
    """
    wav = np.asarray(wav, dtype=np.float64).reshape(-1)
    hop = sr // frame_rate
    T = _n_frames(len(wav), hop)
    env: tp.Optional[np.ndarray] = None
    try:
        import librosa

        env = np.asarray(
            librosa.onset.onset_strength(y=wav.astype(np.float32), sr=sr,
                                         hop_length=hop),
            dtype=np.float64)
    except ImportError:
        logger.info("librosa unavailable; using spectral-flux onset fallback")
    if env is None:
        _, mag = _stft_magnitude(wav, sr, hop)
        log_mag = np.log1p(mag)
        flux = np.maximum(np.diff(log_mag, axis=0), 0.0).sum(axis=1)
        env = np.concatenate([[0.0], flux])
    env = np.maximum(_fit_length(env, T), 0.0)
    peak = env.max()
    if peak > 0:
        env = env / peak
    return env
