#!/usr/bin/env python3
"""Permutation statistics, FDR correction, candidate ranking and heatmaps.

Permutation null (research plan §5.0): for each of ``n_perm`` global lag
shifts of phi (the same family as ``analysis.stats.permuted_phis``), the
motif score is recomputed from the saved A'-restricted lag profiles: a
shifted phi turns each aligned pair into a (query, lag') pair, and the score
is the window sum of the profile at lag'. This treats attention as
exchangeable across A' query frames — exactly the approximation the lag null
model itself makes — and requires no model re-run.

Candidate criteria: BH-FDR significant (q) on ALL configured positive
categories AND not significant on the S5 negative control AND not flagged
periodic. Output: ranked ``candidates.json`` + per-category heatmaps.
"""
from __future__ import annotations

import sys

import numpy as np

from _common import make_parser, results_root, setup, stimuli_dir


def _profile_score(profile: np.ndarray, ap: np.ndarray, a: np.ndarray,
                   window: int) -> np.ndarray:
    """Window-sum motif score of one sample from its lag profile.

    Approximates the exact per-pair score by treating attention as
    exchangeable across the A' query frames (the same approximation the lag
    null model makes): each pair contributes the profile summed over its
    window's lags. Pairs whose whole window is out of range are skipped
    (NOT counted as zero — nansum over an all-NaN window would deflate the
    statistic).
    """
    L, H, n_lags = profile.shape
    lags = ap[:, None] - (a[:, None] + 1
                          + np.arange(-window, window + 1)[None, :])
    valid = (lags >= 1) & (lags < n_lags)          # causality + range, [P, W]
    has_valid = valid.any(axis=1)                  # [P]
    if not has_valid.any():
        return np.full((L, H), np.nan)
    flat = np.clip(lags, 0, n_lags - 1)
    vals = profile[:, :, flat]                     # [L, H, P, W]
    vals = np.where(valid[None, None], vals, np.nan)
    with np.errstate(invalid="ignore"):
        per_pair = np.nansum(vals, axis=-1)        # [L, H, P]
        per_pair = np.where(has_valid[None, None], per_pair, np.nan)
        return np.nanmean(per_pair, axis=-1)       # [L, H]


def observed_scores_from_profiles(profiles: np.ndarray, phis,
                                  window: int) -> np.ndarray:
    """Profile-based OBSERVED statistic ``[L, H]`` (mean over samples).

    Must live in the same approximation family as the permutation null so
    the two are comparable — the exact per-pair scores from scripts/03 are
    kept for RANKING, but p-values use this statistic.
    """
    N = profiles.shape[0]
    prof64 = profiles.astype(np.float64)
    acc, cnt = 0.0, 0
    for n in range(N):
        ap, a = phis[n]
        s = _profile_score(prof64[n], ap, a, window)
        good = np.isfinite(s)
        acc = acc + np.where(good, s, 0.0)
        cnt = cnt + good.astype(np.int64)
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def permuted_scores_from_profiles(profiles: np.ndarray, phis, seg_bounds,
                                  window: int, n_perm: int,
                                  rng: np.random.Generator) -> np.ndarray:
    """Permutation-null motif scores from per-sample lag profiles.

    Parameters
    ----------
    profiles : np.ndarray
        ``[N, L, H, max_lag + 1]`` A'-restricted lag profiles (float16 ok).
    phis : list of (a_prime_frames, a_frames)
        Ground-truth alignment per sample.
    seg_bounds : list of (t_min, t_max)
        A-segment bounds per sample (shift wrap range).
    window : int
        Motif-score tolerance window (+-w).
    n_perm : int
    rng : np.random.Generator

    Returns
    -------
    np.ndarray
        ``[n_perm, L, H]`` null scores (mean over samples and pairs).
    """
    from motif_circuits.analysis.stats import permuted_phis

    N, L, H, _ = profiles.shape
    out = np.zeros((n_perm, L, H))
    counts = np.zeros((n_perm, L, H))
    prof64 = profiles.astype(np.float64)
    for n in range(N):
        ap, a = phis[n]
        t_min, t_max = seg_bounds[n]
        perms = permuted_phis((ap, a), rng, n_perm, t_min, t_max)
        for p, (ap_p, a_p) in enumerate(perms):
            score = _profile_score(prof64[n], ap_p, a_p, window)
            good = np.isfinite(score)
            out[p][good] += score[good]
            counts[p][good] += 1
    return np.where(counts > 0, out / np.maximum(counts, 1), np.nan)


def main() -> int:  # noqa: PLR0915
    args = make_parser(__doc__, "screening.yaml").parse_args()
    cfg = setup(args)

    from motif_circuits.analysis.stats import (benjamini_hochberg,
                                               head_table, permutation_test)
    from motif_circuits.stimuli import load_manifest
    from motif_circuits.utils.io import load_npz, save_json, write_run_json

    scfg, stats_cfg = cfg["screening"], cfg["stats"]
    window = int(scfg["window"])
    out_dir = results_root(cfg) / "screening"
    arrays_null, _ = load_npz(out_dir / "null_model.npz")
    periodic = arrays_null["periodic_mask"]           # [L, H]
    layers = list(arrays_null["layers"])
    rng = np.random.default_rng(int(cfg.get("seed", 42)))

    categories = list(dict.fromkeys(
        list(stats_cfg["positive_categories"])
        + [stats_cfg["negative_category"]]))
    results: dict = {}
    for cat in categories:
        arrays, meta = load_npz(out_dir / cat / "persample_scores.npz")
        excess = arrays["is_motif"] - arrays["null_motif"]   # [N, L, H]
        observed = np.nanmean(excess, axis=0)                # [L, H]
        manifest = {r.id: r for r in load_manifest(
            stimuli_dir(cfg, cat) / "manifest.jsonl")}
        ids = [str(s) for s in arrays["sample_ids"]]
        phis, bounds = [], []
        for sid in ids:
            r = manifest[sid]
            phis.append(r.phi_arrays)
            bounds.append(tuple(r.segments["A"]))
        null_scores = permuted_scores_from_profiles(
            arrays["aprime_lag_profiles"], phis, bounds, window,
            int(stats_cfg["n_perm"]), rng)
        # observed statistic in the SAME exchangeability approximation as the
        # null (profile-based); the exact excess is kept for ranking/plots
        observed_prof = observed_scores_from_profiles(
            arrays["aprime_lag_profiles"], phis, window)
        p = permutation_test(
            observed_prof.reshape(-1),
            null_scores.reshape(null_scores.shape[0], -1))
        L, H = observed.shape
        pvals = p.reshape(L, H)
        sig, thr = benjamini_hochberg(pvals, q=float(stats_cfg["fdr_q"]))
        results[cat] = {"observed": observed, "pvals": pvals, "sig": sig,
                        "threshold": thr}
        print(f"{cat}: {int(sig.sum())}/{L * H} heads significant "
              f"(BH q={stats_cfg['fdr_q']}, thr={thr:.4g})")

    pos = list(stats_cfg["positive_categories"])
    neg = stats_cfg["negative_category"]
    cand_mask = np.ones_like(periodic, dtype=bool)
    for cat in pos:
        cand_mask &= results[cat]["sig"]
    cand_mask &= ~results[neg]["sig"]
    cand_mask &= ~periodic
    rank_score = np.mean([results[c]["observed"] for c in pos], axis=0)
    n_cand = int(cand_mask.sum())
    L, H = cand_mask.shape
    # full ranking over ALL heads (candidates first by construction of the
    # sort key); ranked_all.json powers the pilot-only fallback of 05/06/08
    sort_key = np.where(np.isfinite(rank_score), -rank_score, np.inf)
    all_entries = []
    for flat in np.argsort(sort_key, axis=None):
        li, h = divmod(int(flat), H)
        entry = {"layer": int(layers[li]), "head": int(h),
                 "excess": float(rank_score[li, h]),
                 "periodic": bool(periodic[li, h]),
                 "candidate": bool(cand_mask[li, h])}
        for cat in categories:
            entry[f"p_{cat.lower()}"] = float(results[cat]["pvals"][li, h])
            entry[f"excess_{cat.lower()}"] = float(
                results[cat]["observed"][li, h])
        all_entries.append(entry)
    top_k = min(n_cand, int(stats_cfg.get("top_k_report", 64)))
    candidates = [e for e in all_entries if e["candidate"]][:top_k]
    save_json(out_dir / "candidates.json", candidates)
    save_json(out_dir / "ranked_all.json", all_entries)
    save_json(out_dir / "stats_summary.json", {
        "n_candidates": n_cand,
        "n_periodic": int(periodic.sum()),
        "per_category": {c: {"n_sig": int(results[c]["sig"].sum()),
                             "bh_threshold": results[c]["threshold"]}
                         for c in categories},
        "head_table": head_table({
            "excess": rank_score, "candidate": cand_mask,
            "periodic": periodic,
            **{f"p_{c.lower()}": results[c]["pvals"] for c in categories}}),
    })
    print(f"candidates: {n_cand} heads -> {out_dir / 'candidates.json'}")

    # -------------------------------------------------------------- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for cat in categories:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        im0 = axes[0].imshow(results[cat]["observed"], aspect="auto",
                             cmap="magma")
        axes[0].set(title=f"{cat}: motif IS excess", xlabel="head",
                    ylabel="layer")
        fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(-np.log10(results[cat]["pvals"]), aspect="auto",
                             cmap="viridis")
        axes[1].set(title=f"{cat}: -log10 p", xlabel="head", ylabel="layer")
        fig.colorbar(im1, ax=axes[1])
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap_{cat}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    x = results[pos[0]]["observed"].reshape(-1)
    y = results[neg]["observed"].reshape(-1)
    colors = np.where(cand_mask.reshape(-1), "tab:red",
                      np.where(periodic.reshape(-1), "tab:blue", "0.7"))
    ax.scatter(x, y, s=8, c=colors, alpha=0.8)
    ax.set(xlabel=f"{pos[0]} excess", ylabel=f"{neg} excess",
           title="candidates (red) vs periodic (blue)")
    fig.tight_layout()
    fig.savefig(out_dir / f"scatter_{pos[0]}_vs_{neg}.png", dpi=150)
    plt.close(fig)

    write_run_json(out_dir, cfg, {"stage": "stats"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
