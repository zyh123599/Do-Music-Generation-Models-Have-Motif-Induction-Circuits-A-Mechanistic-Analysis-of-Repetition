#!/usr/bin/env python3
"""Output-side metrics over a generation directory (research plan §7).

Walks ``results/<model>/<target>/<condition>/<mode>/`` (default target:
``ablation``; scripts/08 output uses the same layout), computes per sample:
stripe energy, novelty contrast, loop score, and — for continuations — the
motif-recurrence hit per tau against the prompt's motif-A chroma. Aggregates
mean + bootstrap CI per condition and writes ``metrics.json`` + comparison
plots.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from _common import (bootstrap_ci, evaluate_generation_sample, make_parser,
                     results_root, setup, stimuli_dir)


def main() -> int:  # noqa: PLR0915
    parser = make_parser(__doc__, "knob.yaml")
    parser.add_argument("--target", default="ablation",
                        help="results subdirectory to evaluate")
    args = parser.parse_args()
    cfg = setup(args)

    import soundfile as sf
    from tqdm import tqdm

    from motif_circuits.metrics import recurrence_rate, RecurrenceResult
    from motif_circuits.stimuli import load_manifest
    from motif_circuits.utils.chroma import chroma_features
    from motif_circuits.utils.io import save_json, write_run_json

    mcfg = cfg["metrics"]
    taus = [float(t) for t in cfg.get("knob", {}).get(
        "taus", [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])]
    mcfg = dict(mcfg, taus=taus)
    target_dir = results_root(cfg) / args.target
    assert target_dir.is_dir(), f"{target_dir} not found"

    # cache prompt-A chroma per prompt id
    prompt_chroma_cache: dict = {}

    def prompt_a_chroma(row) -> np.ndarray:
        pid = row["prompt_id"]
        if pid not in prompt_chroma_cache:
            s1_dir = stimuli_dir(cfg, "S1")
            rec = next(r for r in load_manifest(s1_dir / "manifest.jsonl")
                       if r.id == pid)
            wav, sr = sf.read(s1_dir / rec.wav_path, dtype="float32",
                              always_2d=True)
            chroma = chroma_features(wav.mean(axis=1), sr)
            a0, a1 = rec.segments["A"]
            a1 = min(a1, int(row["prompt_frames"]))  # motif part heard
            prompt_chroma_cache[pid] = chroma[a0:a1]
        return prompt_chroma_cache[pid]

    per_condition: dict = {}
    for manifest_path in sorted(target_dir.glob("*/*/generation_manifest.jsonl")):
        gen_dir = manifest_path.parent
        rows = [json.loads(l) for l in manifest_path.read_text().splitlines()
                if l.strip()]
        if not rows:
            continue
        cond = rows[0]["condition"]
        mode = rows[0]["mode"]
        key = f"{cond}/{mode}"
        samples = []
        for row in tqdm(rows, desc=key):
            wav, sr = sf.read(gen_dir / row["wav_path"], dtype="float32",
                              always_2d=True)
            codes = np.load(gen_dir / row["codes_path"])
            pa = None
            if mode == "continuation" and row.get("prompt_id"):
                pa = prompt_a_chroma(row)
            m = evaluate_generation_sample(
                wav.mean(axis=1), sr, codes, mcfg,
                prompt_a_chroma=pa,
                prompt_frames=int(row.get("prompt_frames", 0)))
            m["id"] = row["id"]
            samples.append(m)
        agg: dict = {"n": len(samples)}
        rng = np.random.default_rng(0)
        for name in ("stripe_energy", "novelty_contrast", "ngram_rate",
                     "autocorr_peak", "recurrence_best_corr"):
            vals = np.asarray([s[name] for s in samples if name in s],
                              dtype=np.float64)
            if vals.size:
                mean, lo, hi = bootstrap_ci(
                    vals, n_boot=int(mcfg["n_bootstrap"]), rng=rng)
                agg[name] = {"mean": mean, "ci": [lo, hi],
                             "n": int(vals.size)}
        agg["loop_rate"] = float(np.mean([s["is_loop"] for s in samples]))
        if any("recurrence_hits" in s for s in samples):
            results = [RecurrenceResult(
                best_corr=s.get("recurrence_best_corr", 0.0),
                best_frame=0, best_shift=0,
                hits={float(k): v for k, v in s["recurrence_hits"].items()},
                corr_curve=np.zeros(0))
                for s in samples if "recurrence_hits" in s]
            agg["recurrence_rate"] = {
                str(k): v for k, v in
                recurrence_rate(results, taus=np.asarray(taus)).items()}
        per_condition[key] = {"aggregate": agg, "samples": samples}
        print(f"{key}: loop_rate={agg['loop_rate']:.2f} "
              f"stripe={agg.get('stripe_energy', {}).get('mean', float('nan')):.3f}")

    out_path = target_dir / "metrics.json"
    save_json(out_path, per_condition)
    print(f"metrics -> {out_path}")

    # -------------------------------------------------------------- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = sorted(per_condition)
    for metric in ("stripe_energy", "novelty_contrast"):
        pts = [(k, per_condition[k]["aggregate"].get(metric))
               for k in keys]
        pts = [(k, v) for k, v in pts if v]
        if not pts:
            continue
        fig, ax = plt.subplots(figsize=(max(6, len(pts) * 1.2), 4))
        xs = np.arange(len(pts))
        means = [v["mean"] for _, v in pts]
        errs = np.array([[v["mean"] - v["ci"][0] for _, v in pts],
                         [v["ci"][1] - v["mean"] for _, v in pts]])
        ax.bar(xs, means, yerr=errs, capsize=3)
        ax.set_xticks(xs, [k for k, _ in pts], rotation=30, ha="right")
        ax.set(ylabel=metric, title=f"{metric} per condition (95% CI)")
        fig.tight_layout()
        fig.savefig(target_dir / f"bars_{metric}.png", dpi=150)
        plt.close(fig)

    rec_keys = [k for k in keys
                if "recurrence_rate" in per_condition[k]["aggregate"]]
    if rec_keys:
        fig, ax = plt.subplots(figsize=(6, 4))
        for k in rec_keys:
            rr = per_condition[k]["aggregate"]["recurrence_rate"]
            ts = sorted(float(t) for t in rr)
            ax.plot(ts, [rr[str(t)] for t in ts], "o-", label=k)
        ax.set(xlabel="tau", ylabel="motif recurrence rate",
               title="recurrence-rate sensitivity")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(target_dir / "recurrence_vs_tau.png", dpi=150)
        plt.close(fig)

    write_run_json(target_dir, cfg, {"stage": "output-metrics"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
