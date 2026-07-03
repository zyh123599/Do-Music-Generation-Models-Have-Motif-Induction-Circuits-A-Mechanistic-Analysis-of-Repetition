# AudioCraft 1.3.0 API notes (verified against source)

These facts were extracted by reading the audiocraft 1.3.0 source
(`pip download --no-deps audiocraft==1.3.0`). All code in this repository is
written against them. If you upgrade audiocraft, re-verify with
`scripts/00_smoke_test.py`.

## 1. Delay pattern (`audiocraft/modules/codebooks_patterns.py`)

- MusicGen uses `DelayedPatternProvider` with `n_q=4` and default
  `delays=[0, 1, 2, 3]`, `flatten_first=0`, `empty_initial=0`.
- `get_pattern(T).layout` is a list of length `1 + T + max_delay`.
  `layout[0] == []` (the special-token step). For sequence step `s >= 1`,
  `layout[s]` contains `LayoutCoord(t=s - 1 - delay[q], q)` for every codebook
  `q` with `0 <= s - 1 - delay[q] < T` (upper bound `T` only matters for the
  trailing ramp).
- **Therefore: the token of frame `t`, codebook `q` sits at sequence step
  `s = 1 + t + delay[q] = 1 + t + q`** (MusicGen defaults). The special-token
  offset `s0 = 1`.
- `pattern.valid_layout` truncates to `len(layout) - max_delay` steps, i.e.
  `T + 1` steps (indices `0..T`). With `keep_only_valid_steps=True`
  (default in `compute_predictions`) the model sequence has `S = T + 1` steps
  and the tokens of codebook `q` for frames `t > T - 1 - q` are replaced by the
  special token; the returned `mask` marks them invalid.
- `pattern.build_pattern_sequence(z[B,K,T], special_token, keep_only_valid_steps)`
  -> `(values[B,K,S], indexes[K,S], mask[K,S])`.
- `pattern.revert_pattern_logits(logits[B,card,K,S], float('nan'), keep_only_valid_steps)`
  drops the first (special) step via `is_model_output=True`, i.e. logits at
  sequence step `s` predict the tokens at step `s+1`; after revert, `logits[..., t]`
  is the prediction *for* frame `t` (no extra shift needed).
- `pattern.get_first_step_with_timesteps(t)` = first step containing frame `t`
  = `1 + t` (codebook 0).

## 2. LM (`audiocraft/models/lm.py`)

- `model.lm` is `LMModel`; `card = 2048`, `special_token_id = card = 2048`,
  `n_q = 4`, embeddings `self.emb[k]` (`ScaledEmbedding(card+1, dim)`), output
  heads `self.linears[k]: Linear(dim, card, bias=False)` (bias_proj false in
  config), `self.out_norm = LayerNorm(dim)` (because `norm_first: true`).
- `LMModel.forward(sequence[B,K,S], conditions, condition_tensors)` returns
  logits `[B, K, S, card]`. Input embedding = `sum_k emb[k](sequence[:,k])`;
  positional sin embedding is added *inside* `StreamingTransformer.forward`.
- Text conditioning enters via **cross-attention** (fuser 'cross'); nothing is
  prepended to the sequence for text-only MusicGen models, so sequence step
  indices are NOT shifted by conditioning. (`fuser.fuse2cond['prepend']` is
  empty for text-only models — asserted in our loader.)
- `LMModel.compute_predictions(codes[B,K,T], conditions=[], condition_tensors=ct)`
  -> `LMOutput(logits[B,K,T,card], mask[B,K,T])`, already re-aligned to frames:
  `logits[b,k,t]` is the model's prediction for `codes[b,k,t]`. **This is the
  teacher-forcing entry point.** Internally uses `keep_only_valid_steps=True`
  so the transformer sees `S = T + 1` steps.
- CFG in `_sample_next_token`: one forward with batch doubled as
  `[conditional; unconditional]` (conditional half FIRST). For unconditional
  generation (`conditions=[]`) `cfg_conditions == {}` and the batch is not
  doubled. `two_step_cfg` is false for released models.
- `LMModel.generate` loop: builds the full delayed `gen_sequence [B,K,S_total]`
  (`S_total = 1 + max_gen_len + max_delay`), then iterates
  `offset in range(start_offset_sequence, S_total)` in streaming mode.
  The FIRST streaming forward receives steps `[0, start_offset_sequence)`
  (special token step + any audio-prompt steps; length >= 1), every later
  forward receives exactly 1 step. `start_offset_sequence = 1 + prompt_frames`.
  => hooks that track the current step during generation must add
  `query.shape[1]` per call, not assume 1.

## 3. Attention (`audiocraft/modules/transformer.py`)

- Released MusicGen LM config (`config/model/lm/musicgen_lm.yaml`):
  `memory_efficient: true`, `custom: false` (but `_is_custom(custom, memory_efficient)`
  makes `self.custom = True`), `attention_as_float32: false`,
  `positional_embedding: sin`, `qk_layer_norm: false`, `kv_repeat: 1`,
  `cross_attention: true`, `causal: true`, `past_context: null`,
  `norm_first: true`, bias_proj false. NO RoPE.
- Layer structure (`StreamingTransformerLayer`, pre-norm):
  `x = x + self_attn(norm1(x))`; then `x = x + cross_attention(norm_cross(x), text)`;
  then `x = x + ff(norm2(x))`. Self-attention module: `layer.self_attn`
  (`StreamingMultiheadAttention`), cross attention: `layer.cross_attention`.
- Projections (custom path, self-attention): single packed
  `projected = F.linear(x, in_proj_weight, in_proj_bias)`; channel order is
  `[q(all heads) | k(all heads) | v(all heads)]`, each block `h*d_head`
  channels with **head-major contiguous slices** (rearrange
  `"b t (p h d) -> ..."`, p=3). So
  `q = projected[..., :D]`, `k = projected[..., D:2D]`, `v = projected[..., 2D:]`
  and head `h` of q occupies channels `[h*d_head, (h+1)*d_head)` of its block.
- Attention math: scores = `q @ k^T / sqrt(d_head)` + causal mask
  (`softmax` over keys `j <= i`). With `memory_efficient=True` and torch
  backend it calls `F.scaled_dot_product_attention(q, k, v, is_causal=True)`
  where q,k,v are `[B, H, T, d_head]` (time_dim=2). Attention weights are
  NEVER materialized by audiocraft — we recompute them in our hooks from the
  captured module input.
- The module input to `self_attn` (i.e. `query`) is `norm1(x)`; recomputing
  q/k from it with `in_proj_weight` reproduces the model's own q/k exactly
  (no qk layer norm, no rope for MusicGen).
- Attention output before out_proj: rearranged `"b h t d -> b t (h d)"` so the
  **input of `self_attn.out_proj` has head-contiguous channel slices
  `[h*d_head, (h+1)*d_head)`** -> hook point for per-head output capture,
  ablation, scaling and patching.
- Streaming state: `self_attn._streaming_state['past_keys']` `[B, H, T_past, d]`
  (time_dim=2). During streaming, `_complete_kv` concatenates past K/V. The
  transformer-level `_streaming_state['offsets']` tracks absolute position for
  the sin positional embedding.
- `attention_as_float32=False`; models run under `self.autocast`
  (float16 on CUDA) inside `MusicGen`. Our recomputation should upcast q,k to
  float32 for the softmax to be numerically faithful-enough for analysis.

## 4. MusicGen API (`audiocraft/models/musicgen.py`, `genmodel.py`)

- `MusicGen.get_pretrained('facebook/musicgen-small'|'-medium'|'-large', device=...)`.
- `model.compression_model.encode(wav[B,1,T_samples])` -> `(codes[B,K,T], scale=None)`;
  `wav` must be 32 kHz mono (`model.sample_rate == 32000`,
  `model.frame_rate == 50`); use `audiocraft.data.audio_utils.convert_audio`.
- `model.lm` = LMModel. `model.set_generation_params(duration=…, use_sampling=True,
  top_k=250, top_p=0.0, temperature=1.0, cfg_coef=3.0, two_step_cfg=False,
  extend_stride=18)`; `model.generation_params` is the kwargs dict passed to
  `lm.generate`.
- `model.generate_unconditional(num_samples, return_tokens=True)`;
  `model.generate(descriptions, return_tokens=True)`;
  `model.generate_continuation(prompt_wav[B,1,T], prompt_sample_rate,
  descriptions=None, return_tokens=True)`. All wrap `lm.generate` inside
  `self.autocast`.
- Text attributes: `ConditioningAttributes(text={'description': desc_or_None})`.
  Null conditions = `ClassifierFreeGuidanceDropout(p=1.0)(conditions)`.
  Precompute condition tensors with
  `lm.condition_provider(lm.condition_provider.tokenize(attrs))`.
- Model scales: small = 24 layers x 16 heads, d=1024 (d_head 64);
  medium = 48 x 24, d=1536 (d_head 64); large = 48 x 32, d=2048 (d_head 64).

## 5. Gotchas encoded in this repo

- **Frame->step**: `step(t, q) = 1 + t + q` (never forget the +1).
- Teacher forcing with `compute_predictions` gives `S = T + 1` transformer
  steps: step 0 = special token, step `1+t+q` = frame t codebook q (invalid
  tail masked).
- During free generation with CFG the batch is doubled `[cond; uncond]`;
  head interventions must be applied to BOTH halves (same behavior), and
  attention captures must select the right half.
- The first streaming forward covers multiple steps when an audio prompt is
  used.
- `lm.generate` asserts `prompt_frames < max_gen_len`.
- Everything (embeddings sum, out_norm, per-codebook linears) works on the
  *sequence step* axis; our analysis converts to the frame axis via
  `motif_circuits.delay_map`.
