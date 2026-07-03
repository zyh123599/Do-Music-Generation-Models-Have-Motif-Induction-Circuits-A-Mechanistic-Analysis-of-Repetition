"""Tests for the synthetic stimulus set (no rendering required)."""
import numpy as np
import pytest

from motif_circuits.stimuli import (Motif, sample_motif, transpose,
                                    rhythm_variation, rhythm_scramble_pitches,
                                    sample_gap_material,
                                    shares_interval_ngrams, motif_frames,
                                    compute_phi, build_category, build_all,
                                    load_manifest, StimulusRecord, CATEGORIES)
from motif_circuits.stimuli.motif import scale_pitches


RNG = lambda s=0: np.random.default_rng(s)  # noqa: E731


class TestMotifSampling:
    def test_ranges_and_determinism(self):
        for seed in range(20):
            m = sample_motif(RNG(seed), root=3, scale_type="minor", n_bars=2)
            assert 4 <= m.n_notes <= 12
            assert m.onsets_beats[0] == 0.0
            assert m.total_beats <= 8.0 + 1e-9
            pool = set(scale_pitches(3, "minor").tolist())
            assert set(m.pitches) <= pool
        a = sample_motif(RNG(7), 0, "major")
        b = sample_motif(RNG(7), 0, "major")
        assert a == b

    def test_transpose(self):
        m = sample_motif(RNG(1), 0, "major")
        t = transpose(m, 5)
        assert tuple(p + 5 for p in m.pitches) == t.pitches
        assert t.onsets_beats == m.onsets_beats

    def test_rhythm_variation_preserves_pitches(self):
        m = sample_motif(RNG(2), 5, "minor")
        for mode in ("augment", "diminish", "syncopate"):
            v = rhythm_variation(m, RNG(3), mode)
            assert v.pitches == m.pitches
            if mode == "augment":
                np.testing.assert_allclose(v.onsets_beats,
                                           np.asarray(m.onsets_beats) * 2)
            elif mode == "diminish":
                np.testing.assert_allclose(v.onsets_beats,
                                           np.asarray(m.onsets_beats) * 0.5)
            else:
                assert v.onsets_beats != m.onsets_beats or m.n_notes <= 2
            assert np.all(np.diff(v.onsets_beats) > 0)

    def test_rhythm_scramble(self):
        m = sample_motif(RNG(4), 7, "major", n_bars=2)
        s = rhythm_scramble_pitches(m, RNG(5), 7, "major")
        assert s.onsets_beats == m.onsets_beats
        assert s.durations_beats == m.durations_beats
        changed = np.mean(np.asarray(s.pitches) != np.asarray(m.pitches))
        assert changed >= 0.8
        pool = set(scale_pitches(7, "major").tolist())
        assert set(s.pitches) <= pool

    def test_gap_avoids_motif_material(self):
        for seed in range(10):
            m = sample_motif(RNG(seed), 2, "minor", n_bars=1)
            g = sample_gap_material(RNG(seed + 100), 2, "minor", 4.0, avoid=m)
            assert not shares_interval_ngrams(m.pitches, g.pitches)

    def test_interval_ngram_check_is_transposition_invariant(self):
        a = [60, 64, 67, 65]
        assert shares_interval_ngrams(a, [p + 6 for p in a])


class TestPhi:
    def test_exact_repeat_constant_offset(self):
        """2-note motif at 120 bpm: A at 1.0 s, A' at 5.0 s -> +200 frames."""
        m = Motif((60, 64), (0.0, 1.0), (1.0, 1.0))  # 2 notes, 1 beat each
        ap_f, a_f = compute_phi(m, m, bpm=120, frame_rate=50,
                                offset_a_s=1.0, offset_ap_s=5.0)
        # at 120 bpm, 1 beat = 0.5 s = 25 frames; notes span frames
        # A: [50, 75) and [75, 100); A': [250, 275) and [275, 300)
        np.testing.assert_array_equal(ap_f, np.arange(250, 300))
        np.testing.assert_array_equal(a_f, ap_f - 200)

    def test_augmentation_piecewise_linear(self):
        """A' with doubled durations: two A' frames map to one A frame."""
        m = Motif((60, 64), (0.0, 1.0), (1.0, 1.0))
        ap = Motif((60, 64), (0.0, 2.0), (2.0, 2.0))
        ap_f, a_f = compute_phi(m, ap, bpm=120, frame_rate=50,
                                offset_a_s=0.0, offset_ap_s=4.0)
        # A note 0: frames [0, 25); A' note 0: frames [200, 250)
        assert ap_f[0] == 200 and a_f[0] == 0
        # halfway through A' note 0 -> halfway through A note 0
        i = np.flatnonzero(ap_f == 225)[0]
        assert a_f[i] == 12  # floor(25 * 25/50)
        # strictly monotone queries, non-decreasing targets
        assert np.all(np.diff(ap_f) > 0)
        assert np.all(np.diff(a_f) >= 0)

    def test_in_note_coverage_only(self):
        m = Motif((60, 62), (0.0, 2.0), (0.5, 0.5))  # rests between notes
        ap_f, _ = compute_phi(m, m, bpm=120, frame_rate=50,
                              offset_a_s=0.0, offset_ap_s=2.0)
        # note 0 spans frames [100, 112.5) -> 12 frames; rest until 150
        assert 113 not in ap_f.tolist() and 120 not in ap_f.tolist()


class TestBuild:
    @pytest.mark.parametrize("cat", CATEGORIES)
    def test_build_category_no_render(self, tmp_path, cat):
        cfg = {"n_per_category": 3, "render_seeds": 2, "seed": 0}
        recs, pairs = build_category(cat, cfg, tmp_path / cat, RNG(11),
                                     render=False,
                                     pair_out_dir=tmp_path / "S6")
        assert len(recs) == 6  # 3 logical x 2 render seeds
        loaded = load_manifest(tmp_path / cat / "manifest.jsonl")
        assert len(loaded) == len(recs)
        for r in loaded:
            assert isinstance(r, StimulusRecord)
            for name in ("A", "G", "A_prime"):
                s, e = r.segments[name]
                assert 0 <= s < e <= 500
            assert r.segments["A"][1] == r.segments["G"][0]
            assert r.segments["G"][1] == r.segments["A_prime"][0]
            assert (tmp_path / cat / r.midi_path).is_file()
            if cat == "S6":
                assert r.phi is None
            else:
                ap_f, a_f = r.phi_arrays
                assert np.all(np.diff(ap_f) > 0)
                a0, a1 = r.segments["A"]
                p0, p1 = r.segments["A_prime"]
                assert a_f.min() >= a0 - 1 and a_f.max() <= a1 + 1
                assert ap_f.min() >= p0 - 1 and ap_f.max() <= p1 + 1
            if cat in ("S1", "S2", "S3"):
                assert r.pair_id == f"S6pair_{r.id}"
        if cat in ("S1", "S2", "S3"):
            assert len(pairs) == 6
            for p in pairs:
                assert p.category == "S6" and p.phi is None
                assert (tmp_path / "S6" / p.midi_path).is_file()
        else:
            assert pairs == []

    def test_s2_records_transpose(self, tmp_path):
        recs, _ = build_category("S2", {"n_per_category": 5, "render_seeds": 1},
                                 tmp_path / "S2", RNG(3), render=False)
        for r in recs:
            assert r.transform["type"] == "transpose"
            assert abs(r.transform["semitones"]) in (3, 5, 7)

    def test_s4_changes_program(self, tmp_path):
        recs, _ = build_category("S4", {"n_per_category": 5, "render_seeds": 1},
                                 tmp_path / "S4", RNG(4), render=False)
        for r in recs:
            assert r.transform["program_b"] != r.program

    def test_midi_determinism(self, tmp_path):
        cfg = {"n_per_category": 2, "render_seeds": 1, "seed": 9}
        build_category("S1", cfg, tmp_path / "a", RNG(42), render=False)
        build_category("S1", cfg, tmp_path / "b", RNG(42), render=False)
        for mid in sorted((tmp_path / "a" / "midi").glob("*.mid")):
            other = tmp_path / "b" / "midi" / mid.name
            assert mid.read_bytes() == other.read_bytes()

    def test_build_all_merges_pairs_into_s6(self, tmp_path):
        cfg = {"n_per_category": 2, "render_seeds": 1,
               "categories": ["S1", "S6"], "seed": 1}
        counts = build_all(cfg, tmp_path, render=False)
        s6 = load_manifest(tmp_path / "S6" / "manifest.jsonl")
        ids = {r.id for r in s6}
        assert counts["S6"] == len(s6)
        assert any(i.startswith("S6pair_S1_") for i in ids)
        assert any(i.startswith("S6_") for i in ids)
        # pair prelude identity: same seed-derived A motif as the clean sample
        s1 = load_manifest(tmp_path / "S1" / "manifest.jsonl")
        pair_by_id = {r.id: r for r in s6}
        for r in s1:
            p = pair_by_id[r.pair_id]
            assert p.motif == r.motif
            assert p.segments == r.segments
            assert p.bpm == r.bpm and p.program == r.program
