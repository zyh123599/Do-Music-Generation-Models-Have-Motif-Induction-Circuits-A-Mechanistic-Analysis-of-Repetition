"""Mock-based tests for attention capture and head interventions.

The mock reproduces audiocraft's ``StreamingMultiheadAttention`` custom path
exactly (packed qkv projection with ``(3, H, d)`` channel order, causal SDPA,
head-contiguous out_proj input), so everything the hooks assume about the
real model is exercised here without audiocraft.
"""
import json
import math
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from motif_circuits.model import (AttentionCapture, HeadOutputCapture,
                                  HeadIntervention, HeadInterventions,
                                  model_geometry)
from motif_circuits.knob import MotifRecurrenceKnob

D_MODEL, N_HEADS, N_LAYERS = 32, 4, 3
D_HEAD = D_MODEL // N_HEADS


class MockStreamingMHA(nn.Module):
    """Faithful mock of the custom self-attention path (audiocraft 1.3.0)."""

    def __init__(self, d_model=D_MODEL, num_heads=N_HEADS):
        super().__init__()
        in_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.in_proj_weight = in_proj.weight
        self.in_proj_bias = in_proj.bias
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.embed_dim = d_model
        self.num_heads = num_heads
        self.causal = True
        self.rope = None
        self.qk_layer_norm = False
        self.kv_repeat = 1
        self.cross_attention = False
        self._streaming_state = {}

    def forward(self, query, key, value, **kwargs):
        assert query is key and key is value, "specialized self-attn path"
        proj = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        B, T, _ = proj.shape
        H, d = self.num_heads, self.embed_dim // self.num_heads
        # rearrange "b t (p h d) -> p b h t d"
        packed = proj.view(B, T, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = packed[0], packed[1], packed[2]
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x.transpose(1, 2).reshape(B, T, H * d)
        return self.out_proj(x), None


class MockLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.self_attn = MockStreamingMHA()

    def forward(self, x):
        h = self.norm1(x)  # upstream passes the SAME tensor as q, k and v
        return x + self.self_attn(h, h, h)[0]


def make_mock_model(n_layers=N_LAYERS, seed=0):
    torch.manual_seed(seed)
    layers = nn.ModuleList([MockLayer() for _ in range(n_layers)])

    class MockTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = layers

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    transformer = MockTransformer()
    lm = types.SimpleNamespace(transformer=transformer, n_q=4, card=2048)
    return types.SimpleNamespace(lm=lm, transformer=transformer)


def run_forward(model, B=2, S=11, seed=1):
    torch.manual_seed(seed)
    x = torch.randn(B, S, D_MODEL)
    with torch.no_grad():
        return model.lm.transformer(x), x


def reference_attention(attn: MockStreamingMHA, query: torch.Tensor):
    """Direct softmax(qk/sqrt d) with causal mask — independent of the hook."""
    proj = F.linear(query, attn.in_proj_weight, attn.in_proj_bias)
    B, S, _ = proj.shape
    packed = proj.view(B, S, 3, N_HEADS, D_HEAD).permute(2, 0, 3, 1, 4)
    q, k, v = packed[0], packed[1], packed[2]
    scores = q @ k.transpose(-1, -2) / math.sqrt(D_HEAD)
    mask = torch.ones(S, S, dtype=torch.bool).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, -1), v


class TestAttentionCapture:
    def test_matches_reference_and_rows_sum_to_one(self):
        model = make_mock_model()
        with AttentionCapture(model, layers=[0, 2]) as cap:
            run_forward(model)
        assert set(cap.attention) == {0, 2}
        # recompute the layer-0 reference from the actual layer input
        _, x = run_forward(model)
        query = model.lm.transformer.layers[0].norm1(x)
        ref, _ = reference_attention(model.lm.transformer.layers[0].self_attn,
                                     query)
        got = cap.attention[0].astype(np.float32)
        np.testing.assert_allclose(got, ref.detach().numpy(), atol=2e-3)
        np.testing.assert_allclose(got.sum(-1), 1.0, atol=2e-3)

    def test_reconstructs_module_output(self):
        """probs @ v must equal the module's own pre-out_proj activations."""
        model = make_mock_model()
        with AttentionCapture(model, layers=[1], dtype=np.float32) as ac, \
                HeadOutputCapture(model, layers=[1]) as zc:
            run_forward(model)
        probs = torch.from_numpy(ac.attention[1].astype(np.float32))
        # recompute v exactly as the module does
        _, x = run_forward(model)
        layer = model.lm.transformer.layers[1]
        # layer input is x after layers 0 already applied residuals
        with torch.no_grad():
            h = x
            h = model.lm.transformer.layers[0](h)
            query = layer.norm1(h)
        _, v = reference_attention(layer.self_attn, query)
        recon = (probs @ v).transpose(1, 2).detach()  # [B, S, H, d]
        z = zc.z[1].float()
        np.testing.assert_allclose(recon.numpy(), z.numpy(), atol=1e-4)

    def test_reduce_path_discards_raw(self):
        model = make_mock_model()
        seen = {}

        def reduce(layer, attn):
            seen[layer] = attn.shape
            return float(attn.mean())

        with AttentionCapture(model, layers=[0], reduce=reduce) as cap:
            run_forward(model, B=1, S=7)
        assert cap.attention == {}
        assert 0 in cap.reduced and isinstance(cap.reduced[0], float)
        assert seen[0] == (1, N_HEADS, 7, 7)

    def test_streaming_state_rejected(self):
        model = make_mock_model()
        model.lm.transformer.layers[0].self_attn._streaming_state = {
            "past_keys": torch.zeros(1)}
        with AttentionCapture(model, layers=[0]):
            with pytest.raises(RuntimeError, match="streaming"):
                run_forward(model)

    def test_unsupported_variants_rejected(self):
        model = make_mock_model()
        model.lm.transformer.layers[0].self_attn.rope = object()
        with pytest.raises(RuntimeError, match="RoPE"):
            AttentionCapture(model, layers=[0]).__enter__()

    def test_hooks_removed_on_exit(self):
        model = make_mock_model()
        out_before, _ = run_forward(model)
        with AttentionCapture(model):
            run_forward(model)
        out_after, _ = run_forward(model)
        assert not model.lm.transformer.layers[0].self_attn._forward_pre_hooks
        torch.testing.assert_close(out_before, out_after)


class TestHeadOutputCapture:
    def test_shape_and_slicing(self):
        model = make_mock_model()
        with HeadOutputCapture(model, layers=[0]) as cap:
            run_forward(model, B=2, S=9)
        z = cap.z[0]
        assert z.shape == (2, 9, N_HEADS, D_HEAD)

    def test_planted_pattern(self):
        """With out_proj-input == v and a diagonal in_proj, slices are exact."""
        model = make_mock_model(n_layers=1)
        attn = model.lm.transformer.layers[0].self_attn
        with torch.no_grad():
            attn.in_proj_weight.zero_()
            attn.in_proj_bias.zero_()
            # v-projection = identity; q,k zero -> uniform causal attention
            attn.in_proj_weight[2 * D_MODEL:, :] = torch.eye(D_MODEL)
        x = torch.zeros(1, 3, D_MODEL)
        x[0, 0, D_HEAD:2 * D_HEAD] = 7.0  # only head 1 channels at step 0
        # call the attention module directly with q=k=v
        with HeadOutputCapture(model, layers=[0]) as cap:
            with torch.no_grad():
                attn(x, x, x)
        z = cap.z[0].float()
        # uniform causal attention: step 0 output = v[0] = x[0] (identity v)
        np.testing.assert_allclose(z[0, 0, 1].numpy(), np.full(D_HEAD, 7.0),
                                   atol=1e-5)
        np.testing.assert_allclose(z[0, 0, 0].numpy(), np.zeros(D_HEAD),
                                   atol=1e-5)


class TestHeadInterventions:
    def _z_under(self, model, interventions, B=1, S=6, seed=3):
        with HeadInterventions(model, interventions), \
                HeadOutputCapture(model, layers=[0, 1, 2]) as cap:
            run_forward(model, B=B, S=S, seed=seed)
        return cap.z

    def test_zero_exact_channels(self):
        model = make_mock_model()
        base = self._z_under(model, [])
        z = self._z_under(model, [HeadIntervention(1, 2, "zero")])
        assert torch.all(z[1][:, :, 2, :] == 0)
        torch.testing.assert_close(z[1][:, :, 0, :], base[1][:, :, 0, :])
        torch.testing.assert_close(z[0], base[0])
        # NOTE: layer 2 differs because layer 1's residual changed upstream.

    def test_capture_order_sees_intervention(self):
        """HeadOutputCapture registered after interventions sees edited x."""
        model = make_mock_model()
        z = self._z_under(model, [HeadIntervention(0, 0, "scale", value=0.0)])
        assert torch.all(z[0][:, :, 0, :] == 0)

    def test_scale(self):
        model = make_mock_model()
        base = self._z_under(model, [])
        z = self._z_under(model, [HeadIntervention(0, 1, "scale", value=2.0)])
        torch.testing.assert_close(z[0][:, :, 1, :],
                                   2.0 * base[0][:, :, 1, :])

    def test_mean_replacement(self):
        model = make_mock_model()
        vec = torch.full((D_HEAD,), 0.25)
        z = self._z_under(model, [HeadIntervention(0, 3, "mean", value=vec)])
        assert torch.all(z[0][:, :, 3, :] == 0.25)

    def test_patch_full_and_batched(self):
        model = make_mock_model()
        S = 6
        val = torch.arange(S * D_HEAD, dtype=torch.float32).reshape(S, D_HEAD)
        z = self._z_under(model, [HeadIntervention(0, 0, "patch", value=val)],
                          B=2, S=S)
        for b in range(2):
            torch.testing.assert_close(z[0][b, :, 0, :], val)
        # CFG-style: patch batch 1, runtime batch 2 via [B_v, S, d]
        z = self._z_under(model,
                          [HeadIntervention(0, 0, "patch", value=val[None])],
                          B=2, S=S)
        torch.testing.assert_close(z[0][1, :, 0, :], val)

    def test_step_restriction(self):
        model = make_mock_model()
        base = self._z_under(model, [])
        z = self._z_under(model, [HeadIntervention(
            0, 2, "zero", steps=np.array([1, 3]))])
        assert torch.all(z[0][:, 1, 2, :] == 0)
        assert torch.all(z[0][:, 3, 2, :] == 0)
        torch.testing.assert_close(z[0][:, 0, 2, :], base[0][:, 0, 2, :])
        torch.testing.assert_close(z[0][:, 2, 2, :], base[0][:, 2, 2, :])

    def test_streaming_step_counter(self):
        """Calls with T=2,1,1 map to absolute steps {0,1},{2},{3}."""
        model = make_mock_model(n_layers=1)
        layer = model.lm.transformer.layers[0]
        iv = HeadIntervention(0, 0, "zero", steps=np.array([2]))
        outs = []
        with HeadInterventions(model, [iv]), \
                HeadOutputCapture(model, layers=[0]) as cap:
            torch.manual_seed(0)
            for T in (2, 1, 1):
                x = torch.randn(1, T, D_MODEL)
                with torch.no_grad():
                    layer.self_attn(x, x, x)
                outs.append(cap.z[0].clone())
        # absolute step 2 = SECOND call (first covered steps 0-1) -> zeroed
        assert torch.all(outs[1][:, :, 0, :] == 0)
        assert not torch.all(outs[0][:, :, 0, :] == 0)
        assert not torch.all(outs[2][:, :, 0, :] == 0)

    def test_reset_between_runs(self):
        model = make_mock_model(n_layers=1)
        layer = model.lm.transformer.layers[0]
        iv = HeadIntervention(0, 0, "zero", steps=np.array([0]))
        mgr = HeadInterventions(model, [iv])
        with mgr, HeadOutputCapture(model, layers=[0]) as cap:
            x = torch.randn(1, 1, D_MODEL)
            with torch.no_grad():
                layer.self_attn(x, x, x)
            first = cap.z[0].clone()
            mgr.reset()
            with torch.no_grad():
                layer.self_attn(x, x, x)
            second = cap.z[0].clone()
        assert torch.all(first[:, :, 0, :] == 0)
        assert torch.all(second[:, :, 0, :] == 0)  # step 0 again after reset

    def test_validation(self):
        model = make_mock_model()
        with pytest.raises(ValueError):
            HeadIntervention(0, 0, "warp")
        with pytest.raises(ValueError):
            HeadIntervention(0, 0, "scale", value=None)
        with pytest.raises(ValueError):
            HeadInterventions(model, [HeadIntervention(0, 99, "zero")])

    def test_hooks_removed_and_noop_without_interventions(self):
        model = make_mock_model()
        out_before, _ = run_forward(model)
        with HeadInterventions(model, []):
            out_during, _ = run_forward(model)
        out_after, _ = run_forward(model)
        torch.testing.assert_close(out_before, out_during)
        torch.testing.assert_close(out_before, out_after)


class TestGeometryAndKnob:
    def test_model_geometry_on_mock(self):
        geom = model_geometry(make_mock_model())
        assert (geom.n_layers, geom.n_heads, geom.d_model, geom.d_head) == \
            (N_LAYERS, N_HEADS, D_MODEL, D_HEAD)
        assert geom.n_q == 4 and geom.card == 2048

    def test_knob_from_candidates(self, tmp_path):
        cands = [{"layer": 5, "head": 2, "excess": 0.3},
                 {"layer": 1, "head": 7, "excess": 0.2},
                 {"layer": 9, "head": 0, "excess": 0.1}]
        p = tmp_path / "candidates.json"
        p.write_text(json.dumps(cands))
        knob = MotifRecurrenceKnob.from_candidates(p, top_k=2, gamma=1.5)
        ivs = knob.interventions()
        assert [(iv.layer, iv.head) for iv in ivs] == [(5, 2), (1, 7)]
        assert all(iv.mode == "scale" and iv.value == 1.5 for iv in ivs)

    def test_knob_presets(self):
        with pytest.raises(ValueError):
            MotifRecurrenceKnob.suppress([(0, 0)], gamma=1.5)
        with pytest.raises(ValueError):
            MotifRecurrenceKnob.enhance([(0, 0)], gamma=0.5)
        assert MotifRecurrenceKnob.suppress([(0, 0)]).gamma == 0.5
