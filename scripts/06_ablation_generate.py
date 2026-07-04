#!/usr/bin/env python3
"""Free generation under head ablations (research plan §6.2).

Conditions: baseline / mean-ablate top-K candidates / random-K / periodic-K
(controls use the same K, fixed seed). Modes: unconditional and motif-prompt
continuation (prompt = first seconds of S1 stimuli). Outputs WAV + codes +
``generation_manifest.jsonl`` per condition x mode under
``results/<model>/ablation/``. Mean vectors come from scripts/05 (computed
here when missing).
"""
from __future__ import annotations

import json
import sys

import numpy as np

from _common import (load_ranked_heads, make_parser, results_root, setup,
                     stimuli_dir)


def main() -> int:  # noqa: PLR0915
    args = make_parser(__doc__, "ablation.yaml").parse_args()
    cfg = setup(args)

    import soundfile as sf
    import torch
    from tqdm import tqdm

    from motif_circuits.model import (HeadIntervention,
                                      generate_with_interventions,
                                      load_musicgen, model_geometry)
    from motif_circuits.stimuli import load_manifest
    from motif_circuits.utils.io import load_npz, save_npz, write_run_json

    acfg = cfg["ablation"]
    K = int(acfg["k"])
    out_root = results_root(cfg) / "ablation"
    screening_dir = results_root(cfg) / "screening"
    patch_dir = results_root(cfg) / "patching"

    model = load_musicgen(cfg["model"]["size"], cfg["model"]["device"])
    geom = model_geometry(model)
    rng = np.random.default_rng(int(acfg.get("control_seed", 0)))

    # ------------------------------------------------------ head sets
    ranked = load_ranked_heads(
        screening_dir,
        allow_fallback=bool(acfg.get("allow_ranking_fallback", False)))
    arrays_null, _ = load_npz(screening_dir / "null_model.npz")
    layer_ids = list(arrays_null["layers"])
    periodic_heads = [(int(layer_ids[li]), int(h)) for li, h in
                      zip(*np.where(arrays_null["periodic_mask"]))]
    all_heads = [(l, h) for l in range(geom.n_layers)
                 for h in range(geom.n_heads)]
    non_candidates = [x for x in all_heads if x not in set(ranked)]

    mean_path = patch_dir / "mean_vectors.npz"
    if not mean_path.is_file():
        print(f"{mean_path} missing — run scripts/05_patching.py first "
              "(it fits the mean-ablation vectors)")
        return 1
    mean_arrays, _ = load_npz(mean_path)

    def mean_interventions(heads):
        ivs = []
        for l, h in heads:
            key = f"{l}_{h}"
            if key not in mean_arrays:
                raise KeyError(f"mean vector missing for head {key}")
            ivs.append(HeadIntervention(layer=l, head=h, mode="mean",
                                        value=mean_arrays[key]))
        return ivs

    conditions = {
        "baseline": [],
        "ablate_candidates": mean_interventions(ranked[:K]),
        "ablate_random": mean_interventions(
            [non_candidates[i] for i in rng.choice(
                len(non_candidates), size=K, replace=False)]),
        "ablate_periodic": mean_interventions(periodic_heads[:K]),
    }
    conditions = {c: iv for c, iv in conditions.items()
                  if c in acfg["conditions"]}

    # ------------------------------------------------------ prompts
    prompt_frames = int(float(acfg["prompt_s"]) * geom.frame_rate)
    prompts = []
    if "continuation" in acfg["modes"]:
        s1_dir = stimuli_dir(cfg, "S1")
        recs = [r for r in load_manifest(s1_dir / "manifest.jsonl")
                if r.render_seed == 0 and (s1_dir / r.wav_path).is_file()]
        for r in recs[: int(acfg["n_prompts"])]:
            wav, sr = sf.read(s1_dir / r.wav_path, dtype="float32",
                              always_2d=True)
            n = int(float(acfg["prompt_s"]) * sr)
            prompts.append({"id": r.id, "wav": wav.mean(axis=1)[:n], "sr": sr,
                            "segments": r.segments})
        assert prompts, "continuation mode needs rendered S1 stimuli"

    n_samples = int(acfg["n_samples"])
    n_seeds = int(acfg.get("n_seeds", 1))
    duration = float(acfg["duration_s"])
    base_seed = int(cfg.get("seed", 42))
    desc = acfg.get("descriptions")

    for cond, interventions in conditions.items():
        for mode in acfg["modes"]:
            out_dir = out_root / cond / mode
            manifest_path = out_dir / "generation_manifest.jsonl"
            if manifest_path.is_file() and not args.force:
                print(f"skip {cond}/{mode} (exists)")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            jobs = [(s, i) for s in range(n_seeds) for i in range(n_samples)]
            for s, i in tqdm(jobs, desc=f"{cond}/{mode}"):
                seed = base_seed + 1000 * s + i
                prompt = prompts[i % len(prompts)] if mode == "continuation" \
                    else None
                res = generate_with_interventions(
                    model, interventions,
                    descriptions=[desc] if desc else None,
                    prompt_wav=prompt["wav"] if prompt else None,
                    prompt_sr=prompt["sr"] if prompt else None,
                    num_samples=1, duration=duration, seed=seed)
                sid = f"{cond}_{mode}_s{s}_{i:04d}"
                sf.write(out_dir / f"{sid}.wav", res.wav[0, 0], res.sr,
                         subtype="FLOAT")
                np.save(out_dir / f"{sid}.npy",
                        res.codes[0].astype(np.int16))
                rows.append({
                    "id": sid, "condition": cond, "mode": mode, "seed": seed,
                    "wav_path": f"{sid}.wav", "codes_path": f"{sid}.npy",
                    "prompt_id": prompt["id"] if prompt else None,
                    "prompt_frames": prompt_frames if prompt else 0,
                    "prompt_segments": prompt["segments"] if prompt else None,
                })
            with manifest_path.open("w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            print(f"{cond}/{mode}: {len(rows)} generations")

    save_npz(out_root / "conditions.npz",
             {"k": np.asarray([K])},
             meta={c: [(l, h) for (l, h) in
                       [(iv.layer, iv.head) for iv in ivs]]
                   for c, ivs in conditions.items()})
    write_run_json(out_root, cfg, {"stage": "ablation-generate"})
    _ = torch
    return 0


if __name__ == "__main__":
    sys.exit(main())
