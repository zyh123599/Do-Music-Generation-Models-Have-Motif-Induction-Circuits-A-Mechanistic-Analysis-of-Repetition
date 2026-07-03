"""Tests for config/io utils and script hygiene (imports, --help)."""
import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from motif_circuits.utils.config import apply_override, deep_merge, load_config
from motif_circuits.utils.io import (load_json, load_npz, save_json, save_npz,
                                     write_run_json, git_revision)

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((REPO / "scripts").glob("[0-9]*.py"))
HEAVY = {"torch", "audiocraft", "librosa", "matplotlib", "pandas",
         "soundfile", "pretty_midi", "tqdm"}


class TestConfig:
    def test_deep_merge(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        out = deep_merge(base, {"a": {"b": 9}, "e": 4})
        assert out == {"a": {"b": 9, "c": 2}, "d": 3, "e": 4}
        assert base["a"]["b"] == 1  # no mutation

    def test_typed_overrides(self):
        cfg = {"a": {"b": 0}}
        apply_override(cfg, "a.b=3")
        assert cfg["a"]["b"] == 3 and isinstance(cfg["a"]["b"], int)
        apply_override(cfg, "x=true")
        assert cfg["x"] is True
        apply_override(cfg, "y=[1,2]")
        assert cfg["y"] == [1, 2]
        apply_override(cfg, "new.deep.key=hello")
        assert cfg["new"]["deep"]["key"] == "hello"
        with pytest.raises(ValueError):
            apply_override(cfg, "no_equals_sign")

    def test_load_config_merges_defaults_and_overrides(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("model: {size: large}\nfoo: 1\n")
        cfg = load_config(p, overrides=["foo=2", "model.device=cpu"])
        assert cfg["model"]["size"] == "large"     # file overrides default
        assert cfg["model"]["device"] == "cpu"     # CLI overrides file
        assert cfg["paths"]["data_root"] == "data"  # default preserved
        assert cfg["foo"] == 2


class TestIO:
    def test_npz_roundtrip_with_meta(self, tmp_path):
        arrays = {"a": np.arange(6).reshape(2, 3),
                  "b": np.ones(4, dtype=np.float16)}
        save_npz(tmp_path / "x.npz", arrays, meta={"k": "v", "n": 3})
        loaded, meta = load_npz(tmp_path / "x.npz")
        np.testing.assert_array_equal(loaded["a"], arrays["a"])
        assert loaded["b"].dtype == np.float16
        assert meta == {"k": "v", "n": 3}

    def test_json_roundtrip_numpy_safe(self, tmp_path):
        save_json(tmp_path / "x.json",
                  {"a": np.float64(1.5), "b": np.arange(3), "c": np.int32(7)})
        assert load_json(tmp_path / "x.json") == {"a": 1.5, "b": [0, 1, 2],
                                                  "c": 7}

    def test_run_json(self, tmp_path):
        write_run_json(tmp_path, {"seed": 1}, {"stage": "test"})
        data = load_json(tmp_path / "run.json")
        assert data["config"] == {"seed": 1}
        assert data["stage"] == "test"
        assert "timestamp" in data and "git_revision" in data

    def test_git_revision_in_repo(self):
        assert git_revision(REPO) != ""


class TestScriptHygiene:
    def test_scripts_exist(self):
        names = [p.name for p in SCRIPTS]
        for i in range(9):
            assert any(n.startswith(f"0{i}_") for n in names), names

    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
    def test_no_toplevel_heavy_imports(self, script):
        tree = ast.parse(script.read_text())
        offenders = []
        for node in tree.body:  # module level only
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                root = m.split(".")[0]
                if root in HEAVY or m.startswith("motif_circuits.model") \
                        or m.startswith("motif_circuits.analysis.patching"):
                    offenders.append(m)
        assert not offenders, f"{script.name}: top-level heavy imports {offenders}"

    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
    def test_help_runs_without_heavy_deps(self, script):
        res = subprocess.run([sys.executable, str(script), "--help"],
                             capture_output=True, text=True, cwd=REPO,
                             timeout=60)
        assert res.returncode == 0, res.stderr[-800:]
        assert "--config" in res.stdout and "--override" in res.stdout


def _import_script(name):
    import importlib.util

    scripts_dir = str(REPO / "scripts")
    if scripts_dir not in sys.path:  # scripts resolve `_common` relative
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestStatsScriptHelpers:
    """Lock the profile-based permutation statistics of scripts/04."""

    def _setup(self):
        mod = _import_script("04_stats_heatmaps.py")
        n_lags, L, H = 60, 1, 2
        profile = np.full((1, L, H, n_lags), 0.01)
        profile[0, 0, 0, 30] = 0.5          # head 0: bump exactly at lag 30
        ap = np.arange(40, 50)
        a = ap - 31                          # pair lag to successor = 30
        return mod, profile, [(ap, a)]

    def test_observed_hits_planted_bump(self):
        mod, profile, phis = self._setup()
        obs = mod.observed_scores_from_profiles(profile, phis, window=2)
        # head 0 window sum ~ 0.5 + 4 * 0.01; head 1 ~ 5 * 0.01
        assert obs[0, 0] == pytest.approx(0.54, abs=1e-6)
        assert obs[0, 1] == pytest.approx(0.05, abs=1e-6)

    def test_permutation_null_below_observed(self):
        mod, profile, phis = self._setup()
        rng = np.random.default_rng(0)
        null = mod.permuted_scores_from_profiles(
            profile, phis, [(9, 19)], window=2, n_perm=200, rng=rng)
        obs = mod.observed_scores_from_profiles(profile, phis, window=2)
        # shifted phis almost never land on the bump for head 0
        assert (null[:, 0, 0] >= obs[0, 0]).mean() < 0.2
        # head 1 (flat profile) is exchangeable: null ~ observed
        assert np.nanmedian(null[:, 0, 1]) == pytest.approx(0.05, abs=0.01)

    def test_all_out_of_range_pairs_are_nan(self):
        mod = _import_script("04_stats_heatmaps.py")
        profile = np.full((1, 2, 10), 0.01)
        s = mod._profile_score(profile, np.array([5]), np.array([50]),
                               window=1)
        assert np.isnan(s).all()
