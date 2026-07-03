#!/usr/bin/env python3
"""GPU smoke test (<2 min): validates every assumption the pipeline rests on.

Steps (each printed PASS/FAIL; exits non-zero on any failure):
  1. model load + geometry
  2. DelayMap equivalence with the model's own pattern provider
  3. EnCodec encode/decode roundtrip shapes
  4. teacher-forcing log-probs finite under the validity mask
  5. AttentionCapture rows sum to 1
  6. zero-ablating one head changes the logits
  7. generation with and without a scale intervention
"""
from __future__ import annotations

import sys
import traceback

import numpy as np

from _common import make_parser, setup


def main() -> int:
    args = make_parser(__doc__, "default.yaml").parse_args()
    cfg = setup(args)

    import torch

    from motif_circuits.model import (AttentionCapture, HeadIntervention,
                                      HeadInterventions, decode_codes,
                                      delay_map_for, encode_audio,
                                      generate_with_interventions,
                                      load_musicgen, model_geometry,
                                      null_condition_tensors,
                                      teacher_forcing_logprobs)

    failures = []

    def check(name, fn):
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001 - smoke test reports all
            failures.append(name)
            print(f"FAIL  {name}: {e}")
            traceback.print_exc()

    state = {}

    def load():
        state["model"] = load_musicgen(cfg["model"]["size"],
                                       cfg["model"]["device"])
        state["geom"] = model_geometry(state["model"])
        print(f"      geometry: {state['geom']}")

    check("1 model load", load)
    if "model" not in state:
        print("model load failed; aborting")
        return 1
    model, geom = state["model"], state["geom"]

    def delay_check():
        dm = delay_map_for(model)
        dm.verify_against_audiocraft(model.lm.pattern_provider, T=37)
        dm.verify_against_audiocraft(model.lm.pattern_provider, T=128)
        state["dm"] = dm

    check("2 delay pattern equivalence", delay_check)

    def roundtrip():
        sr = geom.sample_rate
        t = np.arange(sr) / sr
        wav = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        codes = encode_audio(model, wav, sr)
        assert codes.shape[:2] == (1, geom.n_q), codes.shape
        assert abs(codes.shape[-1] - geom.frame_rate) <= 2, codes.shape
        out = decode_codes(model, codes)
        assert out.ndim == 3 and out.shape[-1] > 0
        state["codes_1s"] = codes

    check("3 encode/decode roundtrip", roundtrip)

    def tf_check():
        torch.manual_seed(0)
        codes = torch.randint(0, geom.card, (1, geom.n_q, 100),
                              device=model.device)
        ct = null_condition_tensors(model, 1)
        res = teacher_forcing_logprobs(model, codes, condition_tensors=ct)
        assert res.logprob_true.shape == (1, geom.n_q, 100)
        vals = res.logprob_true[res.mask]
        assert torch.isfinite(vals).all(), "non-finite masked logprobs"
        assert bool(res.mask.any()) and bool((~res.mask).any())
        state["codes_tf"] = codes
        state["ct"] = ct
        state["tf"] = res

    check("4 teacher-forcing logprobs", tf_check)

    def attn_check():
        with AttentionCapture(model, layers=[0, 1]) as cap:
            teacher_forcing_logprobs(model, state["codes_tf"],
                                     condition_tensors=state["ct"])
        for layer in (0, 1):
            a = cap.attention[layer].astype(np.float32)
            assert a.shape[1] == geom.n_heads
            np.testing.assert_allclose(a.sum(-1), 1.0, atol=2e-2)

    check("5 attention rows sum to 1", attn_check)

    def ablate_check():
        base = state["tf"].logits
        iv = [HeadIntervention(layer=0, head=0, mode="zero")]
        with HeadInterventions(model, iv):
            res = teacher_forcing_logprobs(model, state["codes_tf"],
                                           condition_tensors=state["ct"])
        diff = (res.logits - base).abs().max().item()
        assert diff > 0, "zero-ablation left logits unchanged"
        print(f"      max |dlogit| = {diff:.4f}")

    check("6 head ablation changes logits", ablate_check)

    def gen_check():
        base = generate_with_interventions(model, None, num_samples=1,
                                           duration=2.0, seed=0)
        iv = [HeadIntervention(layer=0, head=0, mode="scale", value=2.0)]
        mod = generate_with_interventions(model, iv, num_samples=1,
                                          duration=2.0, seed=0)
        for r in (base, mod):
            assert r.wav.ndim == 3 and r.codes.shape[1] == geom.n_q
        assert not np.array_equal(base.codes, mod.codes), \
            "intervention did not affect generation"

    check("7 generation +- intervention", gen_check)

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
