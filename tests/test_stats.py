"""Tests for permutation statistics, BH-FDR and the recovery-rate guard."""
import numpy as np
import pytest

from motif_circuits.analysis.stats import (permutation_test, permuted_phis,
                                           benjamini_hochberg, head_table)
from motif_circuits.analysis.patching import PatchTargets, recovery_rate


class TestPermutationTest:
    def test_extremes(self):
        obs = np.array([10.0, -10.0])
        null = np.random.default_rng(0).normal(size=(999, 2))
        p = permutation_test(obs, null)
        assert p[0] == pytest.approx(1.0 / 1000)
        assert p[1] == pytest.approx(1.0)

    def test_calibration_under_null(self):
        """When observed is drawn from the null, p is ~Uniform(0,1)."""
        rng = np.random.default_rng(1)
        n_sim, n_perm = 200, 199
        ps = []
        for _ in range(n_sim):
            null = rng.normal(size=(n_perm, 1))
            obs = rng.normal(size=1)
            ps.append(permutation_test(obs, null)[0])
        ps = np.asarray(ps)
        assert 0.4 < ps.mean() < 0.6
        assert (ps < 0.05).mean() < 0.12  # ~5% expected

    def test_nan_observed(self):
        p = permutation_test(np.array([np.nan]), np.zeros((10, 1)))
        assert np.isnan(p[0])


class TestPermutedPhis:
    def test_properties(self):
        rng = np.random.default_rng(0)
        ap = np.arange(60, 80)
        a = np.arange(5, 25)
        t_min, t_max = 0, 40
        perms = list(permuted_phis((ap, a), rng, 50, t_min, t_max))
        assert len(perms) == 50
        for ap2, a2 in perms:
            np.testing.assert_array_equal(ap2, ap)   # queries unchanged
            assert a2.shape == a.shape
            assert a2.min() >= t_min and a2.max() < t_max
            assert not np.array_equal(a2, a)          # nonzero shift
        # shifts should differ across permutations
        deltas = {int((a2 - a)[0] % (t_max - t_min)) for _, a2 in perms}
        assert len(deltas) > 10

    def test_rejects_bad_bounds(self):
        with pytest.raises(ValueError):
            list(permuted_phis((np.array([1]), np.array([5])),
                               np.random.default_rng(0), 1, 0, 3))


class TestBenjaminiHochberg:
    def test_textbook_example(self):
        p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
        mask, thr = benjamini_hochberg(p, q=0.05)
        # BH at q=.05: largest k with p_(k) <= k/8*.05 is k=5 (0.042 <= 0.03125? no)
        # sorted crits: .00625,.0125,.01875,.025,.03125,.0375,.04375,.05
        # p_(1)=.001<=.00625 ok; p_(2)=.008<=.0125 ok; p_(3)=.039>.01875 ...
        assert thr == pytest.approx(0.008)
        assert mask.sum() == 2
        assert mask[0] and mask[1]

    def test_none_significant(self):
        mask, thr = benjamini_hochberg(np.array([0.5, 0.9]), q=0.05)
        assert not mask.any() and thr == 0.0

    def test_nan_and_shape_preserved(self):
        p = np.array([[0.001, np.nan], [0.9, 0.002]])
        mask, _ = benjamini_hochberg(p, q=0.05)
        assert mask.shape == p.shape
        assert not mask[0, 1]
        assert mask[0, 0] and mask[1, 1]


class TestHeadTable:
    def test_flattening(self):
        scores = {"is": np.array([[1.0, np.nan]]),
                  "periodic": np.array([[True, False]])}
        table = head_table(scores)
        assert table[0] == {"layer": 0, "head": 0, "is": 1.0, "periodic": True}
        assert table[1]["is"] is None
        assert table[1]["periodic"] is False

    def test_shape_mismatch(self):
        with pytest.raises(ValueError):
            head_table({"a": np.zeros((2, 3)), "b": np.zeros((3, 2))})


class TestRecovery:
    def test_formula(self):
        r, deg = recovery_rate(-1.0, -3.0, -2.0)
        assert r == pytest.approx(0.5) and not deg

    def test_degenerate_gap(self):
        r, deg = recovery_rate(-2.0, -2.0 + 1e-9, -1.0)
        assert deg and np.isnan(r)

    def test_nonfinite(self):
        r, deg = recovery_rate(np.nan, -3.0, -2.0)
        assert deg and np.isnan(r)

    def test_patch_targets_validation(self):
        with pytest.raises(ValueError):
            PatchTargets(layer_heads=[(0, 0)], frames=np.zeros((2, 2)))
        t = PatchTargets(layer_heads=[(0, 0)], frames=[5, 6, 7])
        assert t.frames.dtype == np.int64
        assert t.codebooks == [0, 1, 2, 3]
