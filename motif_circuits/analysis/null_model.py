"""Lag-spectrum null model estimated on the S6 (no-repetition) corpus.

The null model captures each head's *content-independent* tendency to attend
at a fixed frame lag (beat/bar periodic heads, positional decays). All
induction scores are compared against ``Null_h(t) = P_h(l_t)`` with
``l_t = t - (phi(t) + 1)`` per the research plan (section 5.0).

Frame-coordinate attention ``[H, T, T]`` comes from
``DelayMap.aggregate_frames``; entries can be NaN (missing query steps), so
all statistics here are NaN-aware.
"""
from __future__ import annotations

import logging
import typing as tp

import numpy as np

__all__ = ["lag_spectrum", "NullModel"]

logger = logging.getLogger(__name__)


def lag_spectrum(frame_attn: np.ndarray, max_lag: int) -> np.ndarray:
    """Per-head attention lag spectrum ``P_h(l) = E_t[attn[h, t, t-l]]``.

    Parameters
    ----------
    frame_attn : np.ndarray
        Frame-coordinate attention ``[H, T, T]`` (may contain NaN).
    max_lag : int
        Largest lag (inclusive); lags run ``0..max_lag``.

    Returns
    -------
    np.ndarray
        ``[H, max_lag + 1]`` float64; entry ``(h, l)`` is the NaN-mean of
        ``frame_attn[h, t, t - l]`` over ``t >= l``. Lags with no finite
        entries (or ``l >= T``) are NaN.
    """
    frame_attn = np.asarray(frame_attn, dtype=np.float64)
    if frame_attn.ndim != 3 or frame_attn.shape[1] != frame_attn.shape[2]:
        raise ValueError("frame_attn must be [H, T, T]")
    H, T, _ = frame_attn.shape
    out = np.full((H, max_lag + 1), np.nan)
    for l in range(min(max_lag, T - 1) + 1):
        diag = np.diagonal(frame_attn, offset=-l, axis1=1, axis2=2)  # [H, T-l]
        with np.errstate(invalid="ignore"):
            finite = np.isfinite(diag).any(axis=1)
        out[finite, l] = np.nanmean(diag[finite], axis=1)
    return out


class NullModel:
    """Head-wise lag-spectrum null, averaged over an S6 corpus.

    Attributes
    ----------
    spectrum : np.ndarray or None
        ``[H, max_lag + 1]`` mean lag spectrum after :meth:`fit`/:meth:`load`.
    """

    def __init__(self, spectrum: tp.Optional[np.ndarray] = None):
        self.spectrum: tp.Optional[np.ndarray] = None
        if spectrum is not None:
            self.spectrum = np.asarray(spectrum, dtype=np.float64)
            if self.spectrum.ndim != 2:
                raise ValueError("spectrum must be [H, max_lag + 1]")

    # ------------------------------------------------------------------
    @property
    def max_lag(self) -> int:
        self._require_fit()
        return self.spectrum.shape[1] - 1

    @property
    def n_heads(self) -> int:
        self._require_fit()
        return self.spectrum.shape[0]

    def _require_fit(self) -> None:
        if self.spectrum is None:
            raise RuntimeError("NullModel is not fitted; call fit() or load()")

    # ------------------------------------------------------------------
    def fit(self, spectra: tp.Sequence[np.ndarray]) -> "NullModel":
        """Average per-sample lag spectra (NaN-aware) into the null spectrum.

        Parameters
        ----------
        spectra : sequence of np.ndarray
            Per-sample ``[H, max_lag + 1]`` spectra (from :func:`lag_spectrum`
            on S6 samples), all with identical shape.

        Returns
        -------
        NullModel
            ``self``, with ``.spectrum`` set to the NaN-mean over samples.
        """
        if len(spectra) == 0:
            raise ValueError("need at least one spectrum to fit")
        stack = np.stack([np.asarray(s, dtype=np.float64) for s in spectra])
        with np.errstate(invalid="ignore"):
            self.spectrum = np.nanmean(stack, axis=0)
        logger.info("NullModel fitted on %d spectra: H=%d, max_lag=%d",
                    len(spectra), self.n_heads, self.max_lag)
        return self

    def null_at_lags(self, lags: np.ndarray) -> np.ndarray:
        """Null attention values at the requested frame lags.

        Lags are clipped into ``[0, max_lag]``: negative lags read the lag-0
        value and lags beyond the fitted range read the ``max_lag`` value
        (documented boundary behavior — callers should keep requested lags
        within the fitted range for exact nulls).

        Parameters
        ----------
        lags : np.ndarray
            Integer frame lags ``[N]``.

        Returns
        -------
        np.ndarray
            ``[H, N]`` null values.
        """
        self._require_fit()
        lags = np.clip(np.asarray(lags, dtype=np.int64), 0, self.max_lag)
        return self.spectrum[:, lags]

    # ------------------------------------------------------------------
    def periodic_heads(self, beat_lag: float, bar_lag: float,
                       z_thresh: float = 4.0
                       ) -> tp.Tuple[np.ndarray, dict]:
        """Flag heads with peaks at beat/bar-multiple lags (periodic heads B).

        For every integer multiple ``m * beat_lag`` and ``m * bar_lag``
        (rounded to the nearest frame, ``m >= 1``) within the fitted range,
        the spectrum maximum over a +-1 frame tolerance window is compared to
        the head's own robust baseline ``median + z_thresh * MAD`` (raw MAD,
        both NaN-aware). A head is periodic if ANY tested multiple exceeds
        the threshold.

        Parameters
        ----------
        beat_lag, bar_lag : float
            Beat / bar period in frames (e.g. 25 and 100 at 120 BPM, 50 Hz).
        z_thresh : float, optional
            Robust z threshold.

        Returns
        -------
        periodic : np.ndarray
            Boolean ``[H]`` mask.
        details : dict
            ``{'beat_lags', 'bar_lags'}``: tested (rounded) lags;
            ``{'z_beat', 'z_bar'}``: robust z at each tested lag ``[H, M]``;
            ``{'median', 'mad', 'z_max'}``: per-head baseline stats ``[H]``.
        """
        self._require_fit()
        spec = self.spectrum
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(spec, axis=1)
            mad = np.nanmedian(np.abs(spec - med[:, None]), axis=1)
        safe_mad = np.maximum(mad, 1e-12)

        def window_max(lag: int) -> np.ndarray:
            lo = max(0, lag - 1)
            hi = min(self.max_lag, lag + 1)
            with np.errstate(invalid="ignore"):
                return np.nanmax(spec[:, lo:hi + 1], axis=1)

        def multiples(period: float) -> np.ndarray:
            if period <= 0:
                return np.empty(0, dtype=np.int64)
            ms = np.arange(1, int(np.floor(self.max_lag / period)) + 1)
            lags = np.unique(np.round(ms * period).astype(np.int64))
            return lags[(lags >= 1) & (lags <= self.max_lag)]

        details: dict = {"median": med, "mad": mad, "z_thresh": z_thresh}
        periodic = np.zeros(self.n_heads, dtype=bool)
        z_all = []
        for name, period in (("beat", beat_lag), ("bar", bar_lag)):
            lags = multiples(period)
            if lags.size:
                vals = np.stack([window_max(int(l)) for l in lags], axis=1)
                z = (vals - med[:, None]) / safe_mad[:, None]
            else:
                z = np.empty((self.n_heads, 0))
            details[f"{name}_lags"] = lags
            details[f"z_{name}"] = z
            if z.size:
                with np.errstate(invalid="ignore"):
                    periodic |= np.nanmax(z, axis=1) > z_thresh
                z_all.append(z)
        details["z_max"] = (np.nanmax(np.concatenate(z_all, axis=1), axis=1)
                            if z_all else np.full(self.n_heads, np.nan))
        return periodic, details

    # ------------------------------------------------------------------
    def save(self, path) -> None:
        """Save the fitted spectrum to an ``.npz`` file."""
        self._require_fit()
        np.savez(path, spectrum=self.spectrum)

    @classmethod
    def load(cls, path) -> "NullModel":
        """Load a :class:`NullModel` saved by :meth:`save`."""
        with np.load(path) as data:
            return cls(spectrum=data["spectrum"])
