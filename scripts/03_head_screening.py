#!/usr/bin/env python3
"""Full-head screening: lag null model (S6) + three-tier induction scores.

Pass A (S6 corpus, ``S6_*`` ids only — patching partners excluded):
    teacher-forcing attention -> frame aggregation -> per-head lag spectra
    -> per-layer NullModel + periodic-head votes (beat/bar lags from each
    sample's BPM). Output: ``screening/null_model.npz``.

Pass B (S1-S5): per sample, per head: motif induction score (ground-truth
phi from the manifest), its lag-null baseline, per-codebook token induction
scores (token categories only), and the A'-restricted per-head lag profile.
The lag profile is what makes permutation statistics (scripts/04) computable
WITHOUT re-running the model: the permutation family is global lag shifts of
phi, and a globally-shifted phi's motif score equals a windowed sum of this
profile at the shifted lags (attention treated as exchangeable across the A'
query frames — the same approximation as the null model itself).

Memory: hooks reduce each layer's [1, H, S, S] float32 attention (~16-32 MB
at S≈501) to per-head statistics immediately; nothing bigger is retained.
"""
from __future__ import annotations

import sys

import numpy as np

from _common import (data_root, load_codes_np, make_parser, results_root,
                     setup, stimuli_dir)


def restricted_lag_profile(frame_attn: np.ndarray, query_frames: np.ndarray,
                           max_lag: int) -> np.ndarray:
    """Mean attention per lag over the given query frames only.

    Parameters
    ----------
    frame_attn : np.ndarray
        ``[H, T, T]`` frame-aggregated attention (NaN-padded).
    query_frames : np.ndarray
        Query (A'-segment) frame indices.
    max_lag : int

    Returns
    -------
    np.ndarray
        ``[H, max_lag + 1]``; NaN where no (query, lag) pair is in range.
    """
    H, T, _ = frame_attn.shape
    q = query_frames[(query_frames >= 0) & (query_frames < T)]
    out = np.full((H, max_lag + 1), np.nan)
    for lag in range(max_lag + 1):
        keys = q - lag
        ok = keys >= 0
        if not ok.any():
            continue
        vals = frame_attn[:, q[ok], keys[ok]]
        finite = np.isfinite(vals).any(axis=1)
        with np.errstate(invalid="ignore"):
            out[finite, lag] = np.nanmean(vals[finite], axis=1)
    return out


def main() -> int:  # noqa: PLR0915 - linear pipeline script
    parser = make_parser(__doc__, "screening.yaml")
    parser.add_argument("--pass", dest="which", choices=["A", "B", "both"],
                        default="both", help="run null-model pass, score pass, or both")
    args = parser.parse_args()
    cfg = setup(args)

    import torch
    from tqdm import tqdm

    from motif_circuits.analysis.alignment import align_transposition_invariant
    from motif_circuits.analysis.null_model import NullModel, lag_spectrum
    from motif_circuits.analysis.scores import (motif_induction_null,
                                                motif_induction_score,
                                                token_induction_score)
    from motif_circuits.model import (AttentionCapture, delay_map_for,
                                      load_musicgen, model_geometry,
                                      null_condition_tensors,
                                      teacher_forcing_logprobs)
    from motif_circuits.stimuli import load_manifest
    from motif_circuits.utils.io import load_npz, save_npz, write_run_json

    scfg = cfg["screening"]
    max_lag = int(scfg["max_lag"])
    window = int(scfg["window"])
    out_dir = results_root(cfg) / "screening"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_musicgen(cfg["model"]["size"], cfg["model"]["device"])
    geom = model_geometry(model)
    dm = delay_map_for(model)
    ct = null_condition_tensors(model, 1)
    layers = (list(range(geom.n_layers)) if scfg.get("layers") is None
              else [int(x) for x in scfg["layers"]])
    L, H = len(layers), geom.n_heads

    def capture_forward(codes_np: np.ndarray, reduce_fn):
        codes = torch.as_tensor(codes_np, dtype=torch.long,
                                device=model.device)[None]
        with AttentionCapture(model, layers=layers, reduce=reduce_fn) as cap:
            teacher_forcing_logprobs(model, codes, condition_tensors=ct)
        return cap.reduced

    def limited(records):
        cap_n = scfg.get("max_samples")
        return records if cap_n is None else records[: int(cap_n)]

    null_path = out_dir / "null_model.npz"

    # ------------------------------------------------------------ pass A
    if args.which in ("A", "both") and (args.force or not null_path.is_file()):
        s6_dir = stimuli_dir(cfg, "S6")
        records = [r for r in load_manifest(s6_dir / "manifest.jsonl")
                   if r.id.startswith("S6_")]
        records = limited(records)
        assert records, "no S6 records — run scripts/01 + 02 first"
        spectra_sum = np.zeros((L, H, max_lag + 1))
        spectra_cnt = np.zeros((L, H, max_lag + 1))
        periodic_votes = np.zeros((L, H))
        for r in tqdm(records, desc="pass A (S6 null)"):
            codes_np = load_codes_np(s6_dir, r.id)
            T = codes_np.shape[-1]

            def reduce_fn(layer, attn, _T=T):
                fa = dm.aggregate_frames(attn[0].astype(np.float64), _T)
                return lag_spectrum(fa, max_lag)

            reduced = capture_forward(codes_np, reduce_fn)
            beat_lag = geom.frame_rate * 60.0 / r.bpm
            for li, layer in enumerate(layers):
                spec = reduced[layer]
                finite = np.isfinite(spec)
                spectra_sum[li][finite] += spec[finite]
                spectra_cnt[li][finite] += 1
                per, _ = NullModel(spectrum=spec).periodic_heads(
                    beat_lag, 4.0 * beat_lag, z_thresh=float(scfg["periodic_z"]))
                periodic_votes[li] += per
        with np.errstate(invalid="ignore"):
            spectra = np.where(spectra_cnt > 0, spectra_sum / spectra_cnt,
                               np.nan)
        periodic_mask = periodic_votes / len(records) > float(
            scfg["periodic_vote"])
        save_npz(null_path,
                 {"spectra": spectra.astype(np.float32),
                  "periodic_mask": periodic_mask,
                  "periodic_votes": periodic_votes,
                  "layers": np.asarray(layers)},
                 meta={"n_samples": len(records), "max_lag": max_lag,
                       "model": cfg["model"]["size"]})
        print(f"pass A: {len(records)} samples; periodic heads: "
              f"{int(periodic_mask.sum())}/{L * H}")
    elif args.which in ("A", "both"):
        print(f"pass A: {null_path} exists (use --force to redo)")

    if args.which == "A":
        write_run_json(out_dir, cfg, {"stage": "screening-A"})
        return 0

    # ------------------------------------------------------------ pass B
    arrays_null, _ = load_npz(null_path)
    assert list(arrays_null["layers"]) == layers, \
        "layer subset differs from the fitted null model; rerun pass A"
    null_models = {layer: NullModel(spectrum=arrays_null["spectra"][li])
                   for li, layer in enumerate(layers)}

    token_cats = set(scfg.get("token_categories", []))
    for cat in scfg["categories"]:
        cat_dir = stimuli_dir(cfg, cat)
        out_npz = out_dir / cat / "persample_scores.npz"
        if out_npz.is_file() and not args.force:
            print(f"{cat}: {out_npz} exists (use --force)")
            continue
        records = [r for r in load_manifest(cat_dir / "manifest.jsonl")
                   if r.phi is not None]
        records = limited(records)
        if not records:
            print(f"{cat}: no scored records, skipping")
            continue
        N = len(records)
        is_motif = np.full((N, L, H), np.nan)
        null_motif = np.full((N, L, H), np.nan)
        profiles = np.full((N, L, H, max_lag + 1), np.nan, dtype=np.float16)
        tok = ({k: np.full((N, L, H), np.nan) for k in range(geom.n_q)}
               if cat in token_cats else {})
        for n, r in enumerate(tqdm(records, desc=f"pass B ({cat})")):
            codes_np = load_codes_np(cat_dir, r.id)
            T = codes_np.shape[-1]
            phi = r.phi_arrays
            ap_frames = phi[0]

            def reduce_fn(layer, attn, _T=T, _phi=phi, _ap=ap_frames,
                          _cat=cat):
                a = attn[0].astype(np.float64)      # [H, S, S]
                fa = dm.aggregate_frames(a, _T)     # [H, T, T]
                out = {
                    "is_motif": motif_induction_score(fa, _phi, window=window),
                    "profile": restricted_lag_profile(fa, _ap, max_lag),
                }
                if _cat in token_cats:
                    for k in range(geom.n_q):
                        out[f"tok{k}"] = token_induction_score(
                            a, _phi, dm, _T, codebook=k)
                return out

            reduced = capture_forward(codes_np, reduce_fn)
            for li, layer in enumerate(layers):
                red = reduced[layer]
                is_motif[n, li] = red["is_motif"]
                null_motif[n, li] = motif_induction_null(
                    null_models[layer], phi, window=window)
                profiles[n, li] = red["profile"].astype(np.float16)
                for k in tok:
                    tok[k][n, li] = red[f"tok{k}"]

        arrays = {"is_motif": is_motif, "null_motif": null_motif,
                  "aprime_lag_profiles": profiles,
                  "sample_ids": np.asarray([r.id for r in records])}
        for k, v in tok.items():
            arrays[f"is_tok_k{k}"] = v
        save_npz(out_npz, arrays, meta={
            "category": cat, "n": N, "window": window, "max_lag": max_lag,
            "layers": layers, "model": cfg["model"]["size"]})
        summary = {"is_motif_mean": np.nanmean(is_motif, axis=0),
                   "null_motif_mean": np.nanmean(null_motif, axis=0),
                   "excess_mean": np.nanmean(is_motif - null_motif, axis=0)}
        for k, v in tok.items():
            summary[f"is_tok_k{k}_mean"] = np.nanmean(v, axis=0)
        save_npz(out_dir / cat / "summary.npz", summary,
                 meta={"category": cat, "n": N})
        print(f"{cat}: N={N}; max excess "
              f"{np.nanmax(summary['excess_mean']):.4f}")

    # ------------------------------------------------ S2 data sanity check
    n_verify = int(scfg.get("n_verify", 0))
    if n_verify and "S2" in scfg["categories"]:
        import soundfile as sf

        from motif_circuits.utils.chroma import chroma_features

        s2_dir = stimuli_dir(cfg, "S2")
        recs = [r for r in load_manifest(s2_dir / "manifest.jsonl")
                if (s2_dir / r.wav_path).is_file()][:n_verify]
        mismatches = 0
        for r in recs:
            wav, sr = sf.read(s2_dir / r.wav_path, dtype="float32",
                              always_2d=True)
            chroma = chroma_features(wav.mean(axis=1), sr,
                                     frame_rate=geom.frame_rate)
            res = align_transposition_invariant(
                chroma, chroma, tuple(r.segments["A"]),
                tuple(r.segments["A_prime"]))
            expected = ((int(r.transform["semitones"]) + 5) % 12) - 5
            if res.meta["shift"] != expected:
                mismatches += 1
        if recs:
            print(f"S2 sanity: {mismatches}/{len(recs)} chroma-alignment "
                  f"shift mismatches (ground-truth phi is used regardless)")

    write_run_json(out_dir, cfg, {"stage": "screening"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
