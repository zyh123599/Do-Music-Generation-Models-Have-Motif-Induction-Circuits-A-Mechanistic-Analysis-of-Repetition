"""Tests for chroma features and the three phi aligners."""
import numpy as np
import pytest

from motif_circuits.analysis.alignment import (align_identity,
                                               align_transposition_invariant,
                                               align_dtw)
from motif_circuits.utils.chroma import chroma_features, onset_envelope


def one_hot_chroma(pitch_classes, frames_per_note=5):
    """[T, 12] one-hot chroma from a pitch-class sequence."""
    rows = []
    for pc in pitch_classes:
        row = np.zeros(12)
        row[pc % 12] = 1.0
        rows.extend([row] * frames_per_note)
    return np.asarray(rows)


class TestIdentity:
    def test_equal_length(self):
        res = align_identity((10, 20), (110, 120))
        np.testing.assert_array_equal(res.a_prime_frames, np.arange(110, 120))
        np.testing.assert_array_equal(res.a_frames, np.arange(10, 20))

    def test_proportional(self):
        res = align_identity((0, 5), (100, 110))
        assert res.a_frames[0] == 0 and res.a_frames[-1] == 4
        assert np.all(np.diff(res.a_frames) >= 0)


class TestTranspositionInvariant:
    @pytest.mark.parametrize("k", [3, 5, 7, -3, -5])
    def test_recovers_shift_and_lag(self, k):
        motif = [0, 4, 7, 4, 0, 9, 7, 4]
        a = one_hot_chroma(motif)
        ap = one_hot_chroma([p + k for p in motif])
        Ta = a.shape[0]
        res = align_transposition_invariant(a, ap, (0, Ta), (200, 200 + Ta))
        assert res.meta["shift"] == ((k + 5) % 12) - 5
        assert res.meta["corr"] > 0.99
        # equal-length exact repeat at lag 0 -> frame-for-frame map
        np.testing.assert_array_equal(res.a_frames, np.arange(Ta))
        np.testing.assert_array_equal(res.a_prime_frames, np.arange(200, 200 + Ta))

    def test_sign_convention(self):
        """A' above A by +4 semitones => shift == +4 (not -4)."""
        a = one_hot_chroma([0, 0, 7, 7])
        ap = one_hot_chroma([4, 4, 11, 11])
        res = align_transposition_invariant(a, ap, (0, 20), (0, 20))
        assert res.meta["shift"] == 4

    def test_full_track_slicing(self):
        motif = [2, 5, 9, 5]
        full = np.zeros((100, 12))
        full[10:30] = one_hot_chroma(motif)
        full[60:80] = one_hot_chroma([p + 5 for p in motif])
        res = align_transposition_invariant(full, full, (10, 30), (60, 80))
        assert res.meta["shift"] == 5


class TestDTW:
    def test_time_stretch(self):
        """A' = A with doubled durations -> map slope ~1/2."""
        motif = [0, 4, 7, 11, 2, 5]
        a = one_hot_chroma(motif, frames_per_note=4)
        ap = one_hot_chroma(motif, frames_per_note=8)
        na, nap = a.shape[0], ap.shape[0]
        res = align_dtw(a, ap, (0, na), (100, 100 + nap))
        assert res.a_frames[0] == 0
        assert res.a_frames[-1] == na - 1
        # slope of the map ~ na/nap = 0.5
        slope = np.polyfit(np.arange(nap), res.a_frames, 1)[0]
        assert 0.35 < slope < 0.65
        assert np.all(np.diff(res.a_frames) >= 0)

    def test_identity_case(self):
        a = one_hot_chroma([0, 3, 6, 9])
        res = align_dtw(a, a, (0, 20), (50, 70))
        np.testing.assert_array_equal(res.a_frames, np.arange(20))
        assert res.meta["path_cost"] < 1e-9


class TestChroma:
    @pytest.mark.parametrize("method", ["internal", "librosa"])
    def test_pure_tone_pitch_class(self, method):
        if method == "librosa":
            pytest.importorskip("librosa")
        sr = 32000
        t = np.arange(sr) / sr
        wav = np.sin(2 * np.pi * 440.0 * t)  # A4 -> pitch class 9
        c = chroma_features(wav, sr, method=method)
        assert c.shape == (50, 12)
        mid = c[10:40]
        assert (mid.argmax(axis=1) == 9).mean() > 0.9

    def test_methods_agree_on_argmax(self):
        pytest.importorskip("librosa")
        sr = 32000
        t = np.arange(sr) / sr
        wav = np.sin(2 * np.pi * 261.63 * t)  # C4 -> class 0
        a = chroma_features(wav, sr, method="internal")[10:40].argmax(axis=1)
        b = chroma_features(wav, sr, method="librosa")[10:40].argmax(axis=1)
        assert (a == b).mean() > 0.9

    def test_onset_envelope_shape_and_range(self):
        sr = 32000
        rng = np.random.default_rng(0)
        wav = np.zeros(sr)
        wav[sr // 2:] = rng.standard_normal(sr - sr // 2) * 0.5  # onset mid-way
        env = onset_envelope(wav, sr)
        assert env.shape == (50,)
        assert env.min() >= 0.0 and env.max() == pytest.approx(1.0)
        assert env[23:28].max() > 0.5  # peak near the onset
