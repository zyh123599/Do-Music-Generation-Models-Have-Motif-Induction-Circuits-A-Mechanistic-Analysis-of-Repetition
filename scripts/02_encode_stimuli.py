#!/usr/bin/env python3
"""EnCodec-encode all stimulus WAVs into ``codes/<id>.npy`` (int16 [4, T]).

Requires the rendered stimulus set (scripts/01) and a GPU model load; the
compression model alone is used (no LM forward).
"""
from __future__ import annotations

import sys

import numpy as np

from _common import data_root, make_parser, setup


def main() -> int:
    args = make_parser(__doc__, "stimuli.yaml").parse_args()
    cfg = setup(args)

    import soundfile as sf
    import torch
    from tqdm import tqdm

    from motif_circuits.model import encode_audio, load_musicgen
    from motif_circuits.stimuli import load_manifest
    from motif_circuits.utils.io import save_json, write_run_json

    model = load_musicgen(cfg["model"]["size"], cfg["model"]["device"])
    batch_size = int(cfg.get("encode", {}).get("batch_size", 16))
    categories = list(cfg["stimuli"]["categories"])

    for cat in categories:
        cat_dir = data_root(cfg) / "stimuli" / cat
        manifest = cat_dir / "manifest.jsonl"
        if not manifest.is_file():
            print(f"skip {cat}: no manifest at {manifest}")
            continue
        records = load_manifest(manifest)
        codes_dir = cat_dir / "codes"
        codes_dir.mkdir(parents=True, exist_ok=True)
        lengths = []
        todo = [r for r in records
                if args.force or not (codes_dir / f"{r.id}.npy").is_file()]
        for i in tqdm(range(0, len(todo), batch_size), desc=f"encode {cat}"):
            chunk = todo[i:i + batch_size]
            wavs = []
            for r in chunk:
                wav_path = cat_dir / r.wav_path
                if not wav_path.is_file():
                    wav_path = r.wav_path  # S7 absolute paths
                wav, sr = sf.read(wav_path, dtype="float32", always_2d=True)
                wavs.append((wav.mean(axis=1), sr))
            srs = {sr for _, sr in wavs}
            assert len(srs) == 1, f"mixed sample rates in batch: {srs}"
            batch = np.stack([w for w, _ in wavs])[:, None, :]
            codes = encode_audio(model, batch, srs.pop())
            for r, c in zip(chunk, codes):
                arr = c.cpu().numpy().astype(np.int16)
                np.save(codes_dir / f"{r.id}.npy", arr)
                lengths.append(arr.shape[-1])
        if lengths:
            lengths = np.asarray(lengths)
            if np.any(np.abs(lengths - 500) > 2):
                print(f"WARNING {cat}: {int(np.sum(np.abs(lengths-500)>2))} "
                      f"samples with T outside 500+-2 "
                      f"(range {lengths.min()}..{lengths.max()})")
            save_json(cat_dir / "codes_meta.json", {
                "model_size": cfg["model"]["size"], "n": int(len(lengths)),
                "T_min": int(lengths.min()), "T_max": int(lengths.max()),
                "T_median": float(np.median(lengths))})
        print(f"{cat}: encoded {len(todo)} new / {len(records)} total")
        _ = torch  # keep torch referenced (loaded model lifetime)
    write_run_json(data_root(cfg) / "stimuli", cfg, {"stage": "encode"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
