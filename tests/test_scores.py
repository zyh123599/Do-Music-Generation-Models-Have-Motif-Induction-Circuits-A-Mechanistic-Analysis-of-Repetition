"""Tests for the lag null model and the three-tier induction scores."""
import numpy as np
import pytest

from motif_circuits.delay_map import MUSICGEN_DELAY_MAP as DM
from motif_circuits.analysis.null_model import NullModel, lag_spectrum
from motif_circuits.analysis.scores import (token_induction_score,
                                            token_induction_null,
                                            motif_induction_score,
                                            motif_induction_null,
                                            dla_copy_score)


def make_phi(a_start=0, ap_start=60, n=20):
    ap = np.arange(ap_start, ap_start + n)
    a = np.arange(a_start, a_start + n)
    return ap, a


class TestLagSpectrum:
    def test_hand_built(self):
        T = 6
        attn = np.zeros((1, T, T))
        for t in range(T):
            for l in range(t + 1):
                attn[0, t, t - l] = l  # value == lag
        spec = lag_spectrum(attn, max_lag=4)
        np.testing.assert_allclose(spec[0, :5], np.arange(5.0))

    def test_nan_awareness(self):
        attn = np.full((1, 4, 4), np.nan)
        attn[0, 2, 1] = 0.5  # only one finite lag-1 entry
        spec = lag_spectrum(attn, max_lag=3)
        assert spec[0, 1] == pytest.approx(0.5)
        assert np.isnan(spec[0, 0])


class TestNullModel:
    def test_fit_and_lookup(self):
        s1 = np.tile(np.arange(5.0), (2, 1))
        s2 = np.tile(np.arange(5.0) + 2, (2, 1))
        nm = NullModel().fit([s1, s2])
        np.testing.assert_allclose(nm.spectrum[0], np.arange(5.0) + 1)
        np.testing.assert_allclose(nm.null_at_lags(np.array([0, 4, 99]))[0],
                                   [1.0, 5.0, 5.0])  # clip documented

    def test_periodic_detection(self):
        max_lag = 200
        flat = np.full(max_lag + 1, 0.01)
        comb = flat.copy()
        for m in range(1, 9):
            comb[25 * m] = 0.2  # beat comb at lag 25
        nm = NullModel(spectrum=np.stack([comb, flat]))
        periodic, details = nm.periodic_heads(beat_lag=25.0, bar_lag=100.0)
        assert periodic[0] and not periodic[1]
        assert details["z_max"][0] > details["z_max"][1]

    def test_periodic_non_integer_beat(self):
        max_lag = 150
        spec = np.full(max_lag + 1, 0.01)
        # 110 bpm at 50 Hz -> beat lag 27.27; peaks at rounded multiples
        for m in range(1, 5):
            spec[int(round(27.27 * m))] = 0.3
        nm = NullModel(spectrum=spec[None, :])
        periodic, _ = nm.periodic_heads(beat_lag=27.27, bar_lag=4 * 27.27)
        assert periodic[0]

    def test_save_load_roundtrip(self, tmp_path):
        nm = NullModel(spectrum=np.random.default_rng(0).random((3, 10)))
        p = tmp_path / "null.npz"
        nm.save(p)
        np.testing.assert_array_equal(NullModel.load(p).spectrum, nm.spectrum)


class TestTokenInduction:
    def test_planted_head(self):
        T, H, k = 90, 2, 0
        ap, a = make_phi(a_start=0, ap_start=60, n=20)
        S = DM.seq_len(T, keep_only_valid_steps=True)
        attn = np.zeros((H, S, S))
        # head 0: delta mass on the induction target step(t_a + 1, k)
        for t_ap, t_a in zip(ap, a):
            attn[0, DM.step(t_ap, k), DM.step(t_a + 1, k)] = 1.0
        # head 1: uniform causal attention
        for s in range(S):
            attn[1, s, : s + 1] = 1.0 / (s + 1)
        score = token_induction_score(attn, (ap, a), DM, T, codebook=k)
        assert score[0] == pytest.approx(1.0)
        assert score[1] < 0.05

    def test_causality_guard_and_null(self):
        # phi mapping ap -> ap - 1 makes target == query step -> excluded
        ap = np.arange(10, 20)
        a = ap - 1
        S = DM.seq_len(30, keep_only_valid_steps=True)
        attn = np.zeros((1, S, S))
        score = token_induction_score(attn, (ap, a), DM, 30, codebook=0)
        assert np.isnan(score[0])
        nm = NullModel(spectrum=np.arange(50.0)[None, :])
        ap2, a2 = make_phi(0, 60, 5)
        null = token_induction_null(nm, (ap2, a2))
        # lag = 60 - 1 = 59 -> clipped to 49 -> value 49
        assert null[0] == pytest.approx(49.0)


class TestMotifInduction:
    def test_planted_window_mass(self):
        T, H, w = 90, 2, 2
        ap, a = make_phi(a_start=0, ap_start=60, n=15)
        attn = np.zeros((H, T, T))
        for t_ap, t_a in zip(ap, a):
            attn[0, t_ap, t_a + 1] = 0.6          # exact successor
            attn[0, t_ap, max(t_a, 0)] = 0.2      # inside +-2 window
        for t in range(T):
            attn[1, t, : max(t, 1)] = 1.0 / max(t, 1)
        score = motif_induction_score(attn, (ap, a), window=w)
        assert score[0] == pytest.approx(0.8)
        assert score[1] < 0.1

    def test_window_causal_clip(self):
        """Keys at/after the query frame are excluded from the window."""
        T = 20
        attn = np.zeros((1, T, T))
        ap = np.array([5])
        a = np.array([4])  # window around 5 = frames 3..7, causal keeps 3..4
        attn[0, 5, 3] = 0.3
        attn[0, 5, 4] = 0.2
        attn[0, 5, 5] = 100.0  # must NOT count (non-causal)
        score = motif_induction_score(attn, (ap, a), window=2)
        assert score[0] == pytest.approx(0.5)

    def test_null_matches_window(self):
        nm = NullModel(spectrum=np.ones((1, 100)) * 0.01)
        # a starts at 2 so every pair's +-2 window around t_a+1 stays >= 0
        ap, a = make_phi(2, 60, 10)
        null = motif_induction_null(nm, (ap, a), window=2)
        # 5 window keys per pair, all causal at lag ~57 -> 5 * 0.01
        assert null[0] == pytest.approx(0.05)

    def test_nan_tolerance(self):
        T = 50
        attn = np.full((1, T, T), np.nan)
        ap, a = make_phi(0, 30, 5)
        for t_ap, t_a in zip(ap, a):
            attn[0, t_ap, t_a + 1] = 0.4  # only target finite
        score = motif_induction_score(attn, (ap, a), window=2)
        assert score[0] == pytest.approx(0.4)


class TestDLACopyScore:
    def test_hand_built(self):
        D, d_head, card = 8, 4, 6
        z = np.array([1.0, 0.0, -1.0, 2.0])
        W_O = np.zeros((D, D))
        W_O[:, 4:8] = np.eye(D)[:, :4] * 2.0  # head 1 columns
        ln_gain = np.full(D, 3.0)
        W_k = np.zeros((card, D))
        W_k[2, :4] = 1.0
        # resid = W_O[:, 4:8] @ z = 2*[1,0,-1,2,0,0,0,0]
        # ln_out = 3 * resid / 1.5 ; logit[2] = sum first 4 = 2*(1+0-1+2)*2 = 8
        val = dla_copy_score(z, W_O, head_index=1, ln_gain=ln_gain,
                             ln_scale=1.5, linear_k_weight=W_k, true_token=2)
        assert val == pytest.approx(2.0 * (1 + 0 - 1 + 2) * 3.0 / 1.5)

    def test_guards(self):
        with pytest.raises(ValueError):
            dla_copy_score(np.ones(4), np.eye(8), 2, np.ones(8), 1.0,
                           np.ones((5, 8)), 0)  # head slice out of range
        with pytest.raises(ValueError):
            dla_copy_score(np.ones(4), np.eye(8), 0, np.ones(8), 0.0,
                           np.ones((5, 8)), 0)  # bad ln_scale
