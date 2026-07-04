#!/usr/bin/env python3
"""Activation patching: recovery curves for candidate vs control head sets.

Pairs come from the stimulus manifests (clean S1/S2 sample + its
``S6pair_*`` partner sharing the A+G prelude). Conditions: top-K candidate
heads, random-K heads, periodic-K heads (research plan §6.2/§6.3). Also fits
mean-ablation vectors on an S6 subset (consumed by scripts/06) and the
head-accumulation curve over the candidate ranking (§6.1).
"""
from __future__ import annotations

import sys

import numpy as np

from _common import (load_codes_np, load_ranked_heads, make_parser,
                     results_root, setup, stimuli_dir)


def main() -> int:  # noqa: PLR0915
    args = make_parser(__doc__, "patching.yaml").parse_args()
    cfg = setup(args)

    import torch
    from tqdm import tqdm

    from motif_circuits.analysis.patching import (PatchTargets,
                                                  head_accumulation_curve,
                                                  mean_head_outputs,
                                                  run_patching_pair)
    from motif_circuits.model import (load_musicgen, model_geometry,
                                      null_condition_tensors)
    from motif_circuits.stimuli import load_manifest
    from motif_circuits.utils.io import (load_json, load_npz, save_json,
                                         save_npz, write_run_json)

    pcfg = cfg["patching"]
    out_dir = results_root(cfg) / "patching"
    out_dir.mkdir(parents=True, exist_ok=True)
    screening_dir = results_root(cfg) / "screening"

    model = load_musicgen(cfg["model"]["size"], cfg["model"]["device"])
    geom = model_geometry(model)
    ct = null_condition_tensors(model, 1)
    rng = np.random.default_rng(int(pcfg.get("control_seed", 0)))

    # ------------------------------------------------------ head sets
    if pcfg.get("heads"):
        ranked = [(int(l), int(h)) for l, h in pcfg["heads"]]
    else:
        ranked = load_ranked_heads(
            screening_dir,
            allow_fallback=bool(pcfg.get("allow_ranking_fallback", False)))

    arrays_null, _ = load_npz(screening_dir / "null_model.npz")
    periodic_mask = arrays_null["periodic_mask"]
    layer_ids = list(arrays_null["layers"])
    periodic_heads = [(int(layer_ids[li]), int(h))
                      for li, h in zip(*np.where(periodic_mask))]
    all_heads = [(l, h) for l in range(geom.n_layers)
                 for h in range(geom.n_heads)]
    non_candidates = [x for x in all_heads if x not in set(ranked)]

    # ------------------------------------------------------ pairs
    pairs = []
    s6_dir = stimuli_dir(cfg, "S6")
    for cat in pcfg["pair_categories"]:
        cat_dir = stimuli_dir(cfg, cat)
        for r in load_manifest(cat_dir / "manifest.jsonl"):
            if not r.pair_id:
                continue
            try:
                clean = load_codes_np(cat_dir, r.id)
                corr = load_codes_np(s6_dir, r.pair_id)
            except FileNotFoundError:
                continue
            T = min(clean.shape[-1], corr.shape[-1])
            p0, p1 = r.segments["A_prime"]
            frames = np.arange(p0, min(p1, T))
            pairs.append((r.id, clean[:, :T], corr[:, :T], frames))
    max_pairs = pcfg.get("max_pairs")
    if max_pairs:
        pairs = pairs[: int(max_pairs)]
    assert pairs, "no clean/corrupted pairs found — check scripts/01+02 output"
    print(f"{len(pairs)} patching pairs, {len(ranked)} ranked heads")

    def to_t(arr):
        return torch.as_tensor(arr, dtype=torch.long,
                               device=model.device)[None]

    # ------------------------------------------------------ mean vectors
    mean_path = out_dir / "mean_vectors.npz"
    if args.force or not mean_path.is_file():
        s6_records = [r for r in load_manifest(s6_dir / "manifest.jsonl")
                      if r.id.startswith("S6_")][: int(pcfg["n_mean_samples"])]
        codes_iter = (to_t(load_codes_np(s6_dir, r.id)) for r in s6_records)
        means = mean_head_outputs(model, tqdm(list(codes_iter),
                                              desc="mean vectors"),
                                  layers=list(range(geom.n_layers)))
        save_npz(mean_path, {f"{l}_{h}": v for (l, h), v in means.items()},
                 meta={"n_samples": len(s6_records)})
        print(f"mean-ablation vectors -> {mean_path}")

    # ------------------------------------------------------ conditions
    L_metric = int(pcfg["metric_frames"])
    codebooks = [int(k) for k in pcfg["codebooks"]]
    rows = []
    for K in [int(k) for k in pcfg["ks"]]:
        conditions = {
            f"candidates_top{K}": ranked[:K],
            f"random_{K}": [non_candidates[i] for i in
                            rng.choice(len(non_candidates), size=K,
                                       replace=False)],
            f"periodic_{K}": periodic_heads[:K],
        }
        for cond, heads in conditions.items():
            if not heads:
                print(f"skip {cond}: no heads available")
                continue
            for pid, clean, corr, frames in tqdm(pairs, desc=cond):
                res = run_patching_pair(
                    model, to_t(clean), to_t(corr),
                    PatchTargets(layer_heads=list(heads), frames=frames,
                                 codebooks=codebooks),
                    ct=ct, L=L_metric)
                rows.append({
                    "pair": pid, "condition": cond, "K": K,
                    "recovery": res.recovery, "degenerate": res.degenerate,
                    "delta_clean": res.delta_clean,
                    "delta_corr": res.delta_corr,
                    "delta_patched": res.delta_patched,
                    "n_valid": res.n_valid,
                    "per_codebook": res.per_codebook})
    import json

    with (out_dir / "recovery.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # ------------------------------------------------------ curve
    curve_pairs = [(to_t(c), to_t(x), fr) for _, c, x, fr in pairs]
    curve = head_accumulation_curve(
        model, ranked, curve_pairs,
        ks=[int(k) for k in pcfg["curve_ks"]],
        codebooks=codebooks, ct=ct, L=L_metric)
    save_npz(out_dir / "curves.npz",
             {"ks": np.asarray(sorted(curve)),
              "recovery": np.asarray([curve[k] for k in sorted(curve)])},
             meta={"n_pairs": len(pairs)})

    summary = {}
    for row in rows:
        if not row["degenerate"]:
            summary.setdefault(row["condition"], []).append(row["recovery"])
    summary_mean = {c: float(np.mean(v)) for c, v in summary.items()}
    save_json(out_dir / "summary.json",
              {"mean_recovery": summary_mean, "curve": curve,
               "n_pairs": len(pairs)})
    print("mean recovery per condition:", summary_mean)
    print("accumulation curve:", curve)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ks = sorted(curve)
    ax.plot(ks, [curve[k] for k in ks], "o-", label="top-K candidates")
    for prefix, style in (("random", "s--"), ("periodic", "^:")):
        xs = sorted({r["K"] for r in rows if r["condition"].startswith(prefix)})
        ys = [np.mean([r["recovery"] for r in rows
                       if r["condition"] == f"{prefix}_{k}"
                       and not r["degenerate"]]) for k in xs]
        if xs:
            ax.plot(xs, ys, style, label=f"{prefix}-K control")
    ax.set(xlabel="patched heads K", ylabel="recovery R", xscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "recovery_curve.png", dpi=150)
    plt.close(fig)

    write_run_json(out_dir, cfg, {"stage": "patching"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
