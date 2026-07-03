#!/usr/bin/env python3
"""Motif Recurrence Knob evaluation (research plan §8).

T1 (motif-recurrence boost): gamma sweep on the top-K candidate heads with
FIXED seeds/prompts across gammas -> recurrence rate, stripe energy and loop
rate vs gamma (the trade-off curve).

T2 (loop repair): baseline (gamma=1) samples flagged ``is_loop`` are
regenerated with the SAME seed/prompt under the suppression gamma; reports
the repair rate and metric deltas.

Output layout matches scripts/06 (per-gamma condition dirs), so
scripts/07_output_metrics.py --target knob works on it too.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from _common import (evaluate_generation_sample, make_parser, results_root,
                     setup, stimuli_dir)


def main() -> int:  # noqa: PLR0915
    args = make_parser(__doc__, "knob.yaml").parse_args()
    cfg = setup(args)

    import soundfile as sf
    from tqdm import tqdm

    from motif_circuits.knob import MotifRecurrenceKnob
    from motif_circuits.model import (generate_with_interventions,
                                      load_musicgen, model_geometry)
    from motif_circuits.stimuli import load_manifest
    from motif_circuits.utils.chroma import chroma_features
    from motif_circuits.utils.io import save_json, save_npz, write_run_json

    kcfg, mcfg = cfg["knob"], dict(cfg["metrics"])
    taus = [float(t) for t in kcfg["taus"]]
    mcfg["taus"] = taus
    out_root = results_root(cfg) / "knob"
    out_root.mkdir(parents=True, exist_ok=True)
    candidates_json = results_root(cfg) / "screening" / "candidates.json"

    model = load_musicgen(cfg["model"]["size"], cfg["model"]["device"])
    geom = model_geometry(model)
    duration = float(kcfg["duration_s"])
    base_seed = int(cfg.get("seed", 42))
    top_k = int(kcfg["top_k"])
    n_samples = int(kcfg["n_samples"])

    # ------------------------------------------------------ prompts (S1)
    s1_dir = stimuli_dir(cfg, "S1")
    recs = [r for r in load_manifest(s1_dir / "manifest.jsonl")
            if r.render_seed == 0 and (s1_dir / r.wav_path).is_file()]
    prompts = []
    prompt_frames = int(float(kcfg["prompt_s"]) * geom.frame_rate)
    for r in recs[: int(kcfg["n_prompts"])]:
        wav, sr = sf.read(s1_dir / r.wav_path, dtype="float32",
                          always_2d=True)
        n = int(float(kcfg["prompt_s"]) * sr)
        chroma = chroma_features(wav.mean(axis=1), sr)
        a0, a1 = r.segments["A"]
        prompts.append({"id": r.id, "wav": wav.mean(axis=1)[:n], "sr": sr,
                        "a_chroma": chroma[a0:min(a1, prompt_frames)]})
    assert prompts, "knob eval needs rendered S1 stimuli"

    def run_one(gamma: float, seed: int, prompt) -> dict:
        ivs = (MotifRecurrenceKnob.from_candidates(
            candidates_json, top_k=top_k, gamma=gamma).interventions()
            if gamma != 1.0 else None)
        res = generate_with_interventions(
            model, ivs, prompt_wav=prompt["wav"], prompt_sr=prompt["sr"],
            num_samples=1, duration=duration, seed=seed)
        m = evaluate_generation_sample(
            res.wav[0, 0], res.sr, res.codes[0], mcfg,
            prompt_a_chroma=prompt["a_chroma"], prompt_frames=prompt_frames)
        m["seed"] = seed
        m["prompt_id"] = prompt["id"]
        return m, res

    # ------------------------------------------------------ T1: sweep
    gammas = [float(g) for g in kcfg["gammas"]]
    sweep_path = out_root / "sweep.json"
    sweep: dict = {}
    if sweep_path.is_file() and not args.force:
        sweep = json.loads(sweep_path.read_text())
        print(f"loaded existing sweep ({len(sweep)} gammas)")
    for gamma in gammas:
        gkey = f"{gamma:g}"
        if gkey in sweep:
            continue
        gen_dir = out_root / f"gamma_{gkey}" / "continuation"
        gen_dir.mkdir(parents=True, exist_ok=True)
        samples, rows = [], []
        for i in tqdm(range(n_samples), desc=f"gamma={gkey}"):
            seed = base_seed + i
            prompt = prompts[i % len(prompts)]
            m, res = run_one(gamma, seed, prompt)
            sid = f"knob_g{gkey}_{i:04d}"
            sf.write(gen_dir / f"{sid}.wav", res.wav[0, 0], res.sr,
                     subtype="FLOAT")
            np.save(gen_dir / f"{sid}.npy", res.codes[0].astype(np.int16))
            m["id"] = sid
            samples.append(m)
            rows.append({"id": sid, "condition": f"gamma_{gkey}",
                         "mode": "continuation", "seed": seed,
                         "wav_path": f"{sid}.wav", "codes_path": f"{sid}.npy",
                         "prompt_id": prompt["id"],
                         "prompt_frames": prompt_frames,
                         "prompt_segments": None})
        with (gen_dir / "generation_manifest.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        rate = {str(t): float(np.mean(
            [s["recurrence_hits"].get(str(t), False) for s in samples
             if "recurrence_hits" in s])) for t in taus}
        sweep[gkey] = {
            "recurrence_rate": rate,
            "stripe_energy": float(np.nanmean(
                [s["stripe_energy"] for s in samples])),
            "loop_rate": float(np.mean([s["is_loop"] for s in samples])),
            "samples": samples,
        }
        save_json(sweep_path, sweep)

    save_npz(out_root / "tradeoff.npz", {
        "gammas": np.asarray(gammas),
        "recurrence_08": np.asarray(
            [sweep[f"{g:g}"]["recurrence_rate"].get("0.8", np.nan)
             for g in gammas]),
        "stripe": np.asarray([sweep[f"{g:g}"]["stripe_energy"]
                              for g in gammas]),
        "loop_rate": np.asarray([sweep[f"{g:g}"]["loop_rate"]
                                 for g in gammas])},
        meta={"top_k": top_k, "n_samples": n_samples})

    # ------------------------------------------------------ T2: repair
    repair_gamma = float(kcfg["repair_gamma"])
    baseline = sweep.get("1", sweep.get("1.0"))
    assert baseline is not None, "gamma=1.0 must be in the sweep for T2"
    looped = [s for s in baseline["samples"] if s["is_loop"]]
    repairs = []
    for s in tqdm(looped, desc="T2 repair"):
        prompt = next(p for p in prompts if p["id"] == s["prompt_id"])
        m, _ = run_one(repair_gamma, s["seed"], prompt)
        repairs.append({"seed": s["seed"], "before": s, "after": m,
                        "repaired": bool(not m["is_loop"])})
    repair_report = {
        "n_loops_at_gamma1": len(looped),
        "loop_rate_at_gamma1": baseline["loop_rate"],
        "repair_gamma": repair_gamma,
        "repair_rate": (float(np.mean([r["repaired"] for r in repairs]))
                        if repairs else None),
        "stripe_delta_mean": (float(np.mean(
            [r["after"]["stripe_energy"] - r["before"]["stripe_energy"]
             for r in repairs])) if repairs else None),
        "cases": repairs,
    }
    save_json(out_root / "repair_report.json", repair_report)
    print(f"T2: {len(looped)} loops at gamma=1, repair rate "
          f"{repair_report['repair_rate']}")

    # -------------------------------------------------------------- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    xs = gammas
    axes[0].plot(xs, [sweep[f"{g:g}"]["recurrence_rate"].get("0.8", np.nan)
                      for g in xs], "o-")
    axes[0].set(xlabel="gamma", ylabel="recurrence rate (tau=0.8)")
    axes[1].plot(xs, [sweep[f"{g:g}"]["stripe_energy"] for g in xs], "o-")
    axes[1].set(xlabel="gamma", ylabel="stripe energy")
    axes[2].plot(xs, [sweep[f"{g:g}"]["loop_rate"] for g in xs], "o-")
    axes[2].set(xlabel="gamma", ylabel="loop rate")
    for ax in axes:
        ax.axvline(1.0, color="0.7", ls="--")
    fig.suptitle(f"Motif Recurrence Knob trade-off (top-{top_k} heads)")
    fig.tight_layout()
    fig.savefig(out_root / "tradeoff.png", dpi=150)
    plt.close(fig)

    write_run_json(out_root, cfg, {"stage": "knob"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
