"""Unit tests for the delay-aware frame<->step mapping.

The cross-check against audiocraft's real ``DelayedPatternProvider`` uses the
vendored copy in ``tests/vendor_codebooks_patterns.py`` (MIT-licensed, from
audiocraft 1.3.0) so it runs wherever torch is available, even without a full
audiocraft install. On the server, ``scripts/00_smoke_test.py`` re-runs the
check against the installed audiocraft.
"""
import numpy as np
import pytest

from motif_circuits.delay_map import DelayMap, MUSICGEN_DELAY_MAP

torch = pytest.importorskip("torch", reason="pattern cross-check needs torch")


def _vendor_provider(n_q=4, delays=None):
    from tests.vendor_codebooks_patterns import DelayedPatternProvider
    return DelayedPatternProvider(n_q=n_q, delays=delays)


class TestCoordinates:
    def test_musicgen_step_formula(self):
        dm = MUSICGEN_DELAY_MAP
        assert dm.step(0, 0) == 1
        assert dm.step(0, 3) == 4
        assert dm.step(10, 2) == 13
        np.testing.assert_array_equal(dm.step(np.array([0, 5]), 1), [2, 7])

    def test_frame_inverse(self):
        dm = MUSICGEN_DELAY_MAP
        for q in range(4):
            for t in [0, 1, 17, 499]:
                assert dm.frame(dm.step(t, q), q) == t
        # special token step and ramp map to -1
        assert dm.frame(0, 0) == -1
        assert dm.frame(1, 1) == -1  # step 1 has no codebook-1 token
        assert dm.frame(3, 3) == -1

    def test_coords_at_step(self):
        dm = MUSICGEN_DELAY_MAP
        assert dm.coords_at_step(0) == []
        assert dm.coords_at_step(1) == [(0, 0)]
        assert dm.coords_at_step(2) == [(1, 0), (0, 1)]
        assert dm.coords_at_step(5) == [(4, 0), (3, 1), (2, 2), (1, 3)]
        # trailing ramp of a T=4 sequence
        assert dm.coords_at_step(5, T=4) == [(3, 1), (2, 2), (1, 3)]

    def test_seq_len_and_validity(self):
        dm = MUSICGEN_DELAY_MAP
        T = 10
        assert dm.seq_len(T) == 1 + T + 3
        assert dm.seq_len(T, keep_only_valid_steps=True) == 1 + T
        # with valid-only layout, codebook q loses its last q frames
        for q in range(4):
            for t in range(T):
                expected = t <= T - 1 - q
                assert dm.is_valid_step(t, q, T) == expected, (t, q)

    def test_rejects_bad_input(self):
        dm = MUSICGEN_DELAY_MAP
        with pytest.raises(ValueError):
            dm.step(-1, 0)
        with pytest.raises(ValueError):
            dm.step(0, 4)
        with pytest.raises(ValueError):
            DelayMap(n_q=2, delays=(1, 0))


class TestAudiocraftEquivalence:
    @pytest.mark.parametrize("T", [1, 4, 37, 128])
    def test_musicgen_pattern(self, T):
        provider = _vendor_provider(n_q=4)
        MUSICGEN_DELAY_MAP.verify_against_audiocraft(provider, T=T)

    def test_custom_delays(self):
        provider = _vendor_provider(n_q=3, delays=[0, 2, 4])
        dm = DelayMap.from_audiocraft(provider)
        assert dm.delays == (0, 2, 4)
        dm.verify_against_audiocraft(provider, T=23)

    def test_build_sequence_positions(self):
        """Token placed by audiocraft's build_pattern_sequence lands exactly at
        our predicted step."""
        provider = _vendor_provider(n_q=4)
        T = 12
        codes = torch.arange(4 * T).reshape(1, 4, T)
        pattern = provider.get_pattern(T)
        seq, _, mask = pattern.build_pattern_sequence(codes, special_token=9999)
        dm = MUSICGEN_DELAY_MAP
        for q in range(4):
            for t in range(T):
                s = dm.step(t, q)
                assert seq[0, q, s].item() == codes[0, q, t].item()
                assert bool(mask[q, s])
        # special-token step
        assert (seq[0, :, 0] == 9999).all()

    def test_valid_steps_match_compute_predictions_mask(self):
        """keep_only_valid_steps semantics agree with the reverted mask."""
        provider = _vendor_provider(n_q=4)
        T = 12
        codes = torch.zeros(1, 4, T, dtype=torch.long)
        pattern = provider.get_pattern(T)
        seq, _, _ = pattern.build_pattern_sequence(codes, special_token=2048,
                                                   keep_only_valid_steps=True)
        assert seq.shape[-1] == MUSICGEN_DELAY_MAP.seq_len(T, keep_only_valid_steps=True)
        logits = torch.zeros(1, 5, 4, seq.shape[-1])
        _, _, logit_mask = pattern.revert_pattern_logits(
            logits, float('nan'), keep_only_valid_steps=True)
        dm = MUSICGEN_DELAY_MAP
        for q in range(4):
            for t in range(T):
                assert bool(logit_mask[q, t]) == dm.is_valid_step(t, q, T), (t, q)


class TestAttentionSlicing:
    def _synthetic_attention(self, dm, T):
        """Attention where A[s_q, s_k] encodes (s_q, s_k) uniquely."""
        S = dm.seq_len(T)
        a = np.zeros((S, S))
        for sq in range(S):
            for sk in range(sq + 1):
                a[sq, sk] = sq * 1000 + sk
        return a

    def test_slice_attention_targets_right_steps(self):
        dm = MUSICGEN_DELAY_MAP
        T = 8
        a = self._synthetic_attention(dm, T)
        block = dm.slice_attention(a, T, q_query=2, q_key=0)
        for tq in range(T):
            for tk in range(T):
                sq, sk = dm.step(tq, 2), dm.step(tk, 0)
                expected = a[sq, sk]
                assert block[tq, tk] == expected

    def test_slice_attention_fills_out_of_range(self):
        dm = MUSICGEN_DELAY_MAP
        T = 8
        S_valid = dm.seq_len(T, keep_only_valid_steps=True)
        a = self._synthetic_attention(dm, T)[:S_valid, :S_valid]
        block = dm.slice_attention(a, T, q_query=3, q_key=3)
        # last 3 frames of codebook 3 fall beyond the valid layout
        assert np.isnan(block[T - 1, 0])
        assert not np.isnan(block[T - 4, 0])

    def test_aggregate_frames_sums_key_mass(self):
        dm = MUSICGEN_DELAY_MAP
        T = 6
        S = dm.seq_len(T)
        rng = np.random.default_rng(0)
        # random causal attention rows, normalized
        a = np.tril(rng.random((S, S))) + 1e-9
        a /= a.sum(axis=-1, keepdims=True)
        agg = dm.aggregate_frames(a, T, query_codebooks=[0], key_codebooks=[0, 1, 2, 3])
        tq, tk = 5, 2
        sq = dm.step(tq, 0)
        expected = sum(a[sq, dm.step(tk, q)] for q in range(4))
        assert np.isclose(agg[tq, tk], expected)

    def test_steps_for_frames(self):
        dm = MUSICGEN_DELAY_MAP
        steps = dm.steps_for_frames([0, 1], codebooks=[0, 1])
        np.testing.assert_array_equal(steps, [1, 2, 3])
        steps = dm.steps_for_frames([5], S=7)
        np.testing.assert_array_equal(steps, [6])
