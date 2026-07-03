"""Tests for output-side repetition metrics (SSM, recurrence, loop)."""
import numpy as np
import pytest

from motif_circuits.metrics import (chroma_ssm, lag_profile, stripe_energy,
                                    foote_novelty, novelty_contrast,
                                    motif_recurrence, recurrence_rate,
                                    token_ngram_repetition,
                                    audio_autocorr_collapse, loop_score)


def one_hot_chroma(pitch_classes, frames_per_note=5):
    rows = []
    for pc in pitch_classes:
        row = np.zeros(12)
        row[pc % 12] = 1.0
        rows.extend([row] * frames_per_note)
    return np.asarray(rows)


def random_chroma(T, rng):
    c = rng.random((T, 12))
    return c / np.linalg.norm(c, axis=1, keepdims=True)


class TestSSM:
    def test_periodic_pattern_peaks_at_period(self):
        motif = [0, 4, 7, 11]
        block = one_hot_chroma(motif, frames_per_note=3)  # period 12 frames
        chroma = np.tile(block, (8, 1))
        ssm = chroma_ssm(chroma)
        prof = lag_profile(ssm)
        period = block.shape[0]
        # profile at the period ~1, elsewhere low
        assert prof[period] == pytest.approx(1.0)
        assert stripe_energy(ssm, 2, 40) > 0.5

    def test_random_chroma_low_energy(self):
        rng = np.random.default_rng(0)
        ssm = chroma_ssm(random_chroma(200, rng))
        assert stripe_energy(ssm, 2, 100) < 0.15

    def test_silent_rows_guarded(self):
        chroma = np.zeros((10, 12))
        ssm = chroma_ssm(chroma)
        assert np.all(ssm == 0.0)

    def test_novelty_peaks_at_boundary(self):
        a = one_hot_chroma([0], frames_per_note=40)
        b = one_hot_chroma([6], frames_per_note=40)
        ssm = chroma_ssm(np.vstack([a, b]))
        nov = foote_novelty(ssm, kernel_size=16)
        assert abs(int(np.argmax(nov)) - 40) <= 2
        assert novelty_contrast(nov) > 3.0

    def test_novelty_flat_input(self):
        ssm = chroma_ssm(one_hot_chroma([3], frames_per_note=60))
        nov = foote_novelty(ssm, kernel_size=8)
        assert np.max(np.abs(nov)) < 1e-6


class TestRecurrence:
    def test_planted_transposed_recurrence(self):
        motif = [0, 4, 7, 4, 9, 7]
        A = one_hot_chroma(motif)
        rng = np.random.default_rng(1)
        cont = random_chroma(200, rng) * 0.5
        offset = 120
        planted = one_hot_chroma([p + 4 for p in motif])
        cont[offset:offset + planted.shape[0]] = planted
        res = motif_recurrence(A, cont, taus=np.arange(0.6, 0.91, 0.05))
        assert res.best_shift == 4
        assert abs(res.best_frame - offset) <= 2
        assert res.hits[0.8] is True
        assert res.best_corr > 0.95

    def test_unrelated_low(self):
        A = one_hot_chroma([0, 4, 7, 11])
        rng = np.random.default_rng(2)
        res = motif_recurrence(A, random_chroma(150, rng))
        assert res.best_corr < 0.6
        assert res.hits[0.8] is False

    def test_recurrence_rate_curve(self):
        A = one_hot_chroma([0, 5])
        hit = motif_recurrence(A, np.tile(A, (5, 1)))
        rng = np.random.default_rng(3)
        miss = motif_recurrence(A, random_chroma(60, rng))
        rate = recurrence_rate([hit, miss], taus=np.array([0.8]))
        assert rate[0.8] == pytest.approx(0.5)

    def test_validation(self):
        with pytest.raises(ValueError):
            motif_recurrence(np.zeros((10, 12)), np.zeros((5, 12)))  # cont < motif


class TestLoop:
    def test_tiled_codes_high(self):
        block = np.arange(40).reshape(4, 10)
        codes = np.tile(block, (1, 30))  # perfect 10-frame loop
        assert token_ngram_repetition(codes) > 0.9

    def test_random_codes_low(self):
        rng = np.random.default_rng(0)
        codes = rng.integers(0, 2048, size=(4, 300))
        assert token_ngram_repetition(codes) < 0.05

    def test_all_codebooks_stricter(self):
        rng = np.random.default_rng(1)
        codes = rng.integers(0, 2048, size=(4, 300))
        codes[0] = np.tile(codes[0, :10], 30)  # only codebook 0 loops
        assert token_ngram_repetition(codes, codebooks="first") > 0.9
        assert token_ngram_repetition(codes, codebooks="all") < 0.05

    def test_autocorr_periodic_envelope(self):
        sr = 32000
        t = np.arange(6 * sr) / sr
        carrier = np.sin(2 * np.pi * 220 * t)
        gate = (np.sin(2 * np.pi * t / 1.0) > 0).astype(float)  # 1 s period
        peak, lag = audio_autocorr_collapse(carrier * gate, sr)
        assert peak > 0.85
        assert abs(lag - 1.0) < 0.1

    def test_autocorr_noise_low(self):
        rng = np.random.default_rng(4)
        peak, _ = audio_autocorr_collapse(rng.standard_normal(6 * 32000), 32000)
        assert peak < 0.5

    def test_joint_criterion(self):
        sr = 32000
        t = np.arange(6 * sr) / sr
        loop_wav = np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * t) > 0)
        rng = np.random.default_rng(5)
        noise_wav = rng.standard_normal(6 * sr) * 0.1
        looped_codes = np.tile(np.arange(40).reshape(4, 10), (1, 30))
        random_codes = rng.integers(0, 2048, size=(4, 300))
        assert loop_score(looped_codes, loop_wav, sr).is_loop is True
        # only one criterion firing -> not a loop
        assert loop_score(looped_codes, noise_wav, sr).is_loop is False
        assert loop_score(random_codes, loop_wav, sr).is_loop is False

    def test_quality_wrappers_error_message(self):
        from motif_circuits.metrics import fad_score, clap_score
        for fn in (fad_score, clap_score):
            try:
                fn("a", "b")
            except RuntimeError as e:
                assert "pip install" in str(e)
            except TypeError:
                pass  # signature differences are fine; only the hint matters
