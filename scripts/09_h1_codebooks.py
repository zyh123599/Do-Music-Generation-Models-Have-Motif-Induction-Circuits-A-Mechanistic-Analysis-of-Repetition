#!/usr/bin/env python3
"""H1 codebook-division-of-labor test (research plan §2 H1, §5.2).

Uses the per-codebook token induction scores saved by scripts/03 pass B on
S1 (exact repeat) and S4 (same notes, different instrument):

* H1(i)  coarse codebooks carry the induction signal:
         ``IS_tok^(0,1) >> IS_tok^(2,3)`` on S1 (paired Wilcoxon over the
         top heads).
* H1(ii) timbre change spares the coarse signal and collapses the fine one:
         retention ratio ``S4/S1`` per codebook — expected near 1 for k0/k1
         and clearly below 1 for k2/k3.

Outputs ``screening/h1_codebooks.json`` + heatmap/scatter PNGs. Pure CPU;
run any time after scripts/03 covered S1 and S4.
"""
from __future__ import annotations

import sys

import numpy as np

from _common import make_parser, results_root, setup


def h1_summary(tok_s1: dict, tok_s4: dict, top_n: int = 32) -> dict:
    """H1 statistics from per-codebook mean score maps.

    Parameters
    ----------
    tok_s1, tok_s4 : dict[int, np.ndarray]
        ``codebook -> [L, H]`` mean token-IS maps for S1 / S4.
    top_n : int
        Number of heads (ranked by S1 coarse score) entering the paired
        statistics.

    Returns
    -------
    dict
        ``top_heads``, per-codebook means over the top heads, coarse/fine
        contrasts (H1 i) and S4/S1 retention ratios (H1 ii) with Wilcoxon
        p-values where applicable.
    """
    from scipy import stats as sstats

    coarse_s1 = (tok_s1[0] + tok_s1[1]) / 2.0
    fine_s1 = (tok_s1[2] + tok_s1[3]) / 2.0
    coarse_s4 = (tok_s4[0] + tok_s4[1]) / 2.0
    fine_s4 = (tok_s4[2] + tok_s4[3]) / 2.0

    L, H = coarse_s1.shape
    key = np.where(np.isfinite(coarse_s1), coarse_s1, -np.inf)
    order = np.argsort(key, axis=None)[::-1][: int(top_n)]
    li, hi = np.unravel_index(order, (L, H))
    top_heads = [[int(l), int(h)] for l, h in zip(li, hi)]

    def sel(m: np.ndarray) -> np.ndarray:
        return m[li, hi]

    c1, f1 = sel(coarse_s1), sel(fine_s1)
    c4, f4 = sel(coarse_s4), sel(fine_s4)
    ok_i = np.isfinite(c1) & np.isfinite(f1)
    wil_i = (sstats.wilcoxon(c1[ok_i], f1[ok_i], alternative="greater")
             if int(ok_i.sum()) >= 5 else None)

    eps = 1e-9
    ret_coarse = c4 / np.maximum(c1, eps)
    ret_fine = f4 / np.maximum(f1, eps)
    ok_ii = np.isfinite(ret_coarse) & np.isfinite(ret_fine)
    wil_ii = (sstats.wilcoxon(ret_coarse[ok_ii], ret_fine[ok_ii],
                              alternative="greater")
              if int(ok_ii.sum()) >= 5 else None)

    def _f(x) -> float:
        return float(x)

    return {
        "top_n": int(top_n),
        "top_heads": top_heads,
        "per_codebook_mean_top": {
            "S1": {f"k{k}": _f(np.nanmean(sel(tok_s1[k]))) for k in range(4)},
            "S4": {f"k{k}": _f(np.nanmean(sel(tok_s4[k]))) for k in range(4)},
        },
        "h1_i_coarse_vs_fine_S1": {
            "coarse_mean": _f(np.nanmean(c1)), "fine_mean": _f(np.nanmean(f1)),
            "ratio": _f(np.nanmean(c1) / max(np.nanmean(f1), eps)),
            "wilcoxon_p": _f(wil_i.pvalue) if wil_i else None,
            "n": int(ok_i.sum()),
        },
        "h1_ii_retention_S4_over_S1": {
            "coarse_median": _f(np.nanmedian(ret_coarse[ok_ii])),
            "fine_median": _f(np.nanmedian(ret_fine[ok_ii])),
            "wilcoxon_p": _f(wil_ii.pvalue) if wil_ii else None,
            "n": int(ok_ii.sum()),
        },
    }


def main() -> int:
    parser = make_parser(__doc__, "screening.yaml")
    parser.add_argument("--top-n", type=int, default=32,
                        help="heads (by S1 coarse score) in the paired stats")
    args = parser.parse_args()
    cfg = setup(args)

    from motif_circuits.utils.io import load_npz, save_json, write_run_json

    out_dir = results_root(cfg) / "screening"
    tok = {}
    for cat in ("S1", "S4"):
        path = out_dir / cat / "persample_scores.npz"
        if not path.is_file():
            print(f"{path} missing — run scripts/03 with {cat} in "
                  "screening.categories first")
            return 1
        arrays, _ = load_npz(path)
        missing = [k for k in range(4) if f"is_tok_k{k}" not in arrays]
        if missing:
            print(f"{cat}: is_tok_k{missing} absent — ensure {cat} is in "
                  "screening.token_categories and rerun scripts/03")
            return 1
        tok[cat] = {k: np.nanmean(arrays[f"is_tok_k{k}"], axis=0)
                    for k in range(4)}

    summary = h1_summary(tok["S1"], tok["S4"], top_n=args.top_n)
    save_json(out_dir / "h1_codebooks.json", summary)
    print("H1(i)  S1 coarse/fine ratio:",
          f"{summary['h1_i_coarse_vs_fine_S1']['ratio']:.2f}",
          f"(p={summary['h1_i_coarse_vs_fine_S1']['wilcoxon_p']})")
    print("H1(ii) retention coarse vs fine:",
          f"{summary['h1_ii_retention_S4_over_S1']['coarse_median']:.2f}",
          "vs",
          f"{summary['h1_ii_retention_S4_over_S1']['fine_median']:.2f}",
          f"(p={summary['h1_ii_retention_S4_over_S1']['wilcoxon_p']})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey=True)
    vmax = max(np.nanmax(m) for maps in tok.values() for m in maps.values())
    for row, cat in enumerate(("S1", "S4")):
        for k in range(4):
            im = axes[row, k].imshow(tok[cat][k], aspect="auto", cmap="magma",
                                     vmin=0, vmax=vmax)
            axes[row, k].set_title(f"{cat} IS_tok k{k}")
            if k == 0:
                axes[row, k].set_ylabel("layer")
            axes[row, k].set_xlabel("head")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig.suptitle("H1: per-codebook token induction (S1 exact vs S4 timbre)")
    fig.savefig(out_dir / "h1_codebooks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (name, ks) in zip(axes, (("coarse (k0+k1)/2", (0, 1)),
                                     ("fine (k2+k3)/2", (2, 3)))):
        x = ((tok["S1"][ks[0]] + tok["S1"][ks[1]]) / 2).reshape(-1)
        y = ((tok["S4"][ks[0]] + tok["S4"][ks[1]]) / 2).reshape(-1)
        ax.scatter(x, y, s=6, alpha=0.6)
        lim = np.nanmax(np.concatenate([x, y])) * 1.05
        ax.plot([0, lim], [0, lim], "0.7", ls="--")
        ax.set(xlabel="S1 score", ylabel="S4 score", title=name,
               xlim=(0, lim), ylim=(0, lim))
    fig.suptitle("H1(ii): timbre-change retention per head")
    fig.tight_layout()
    fig.savefig(out_dir / "h1_retention_scatter.png", dpi=150)
    plt.close(fig)

    write_run_json(out_dir, cfg, {"stage": "h1-codebooks"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
