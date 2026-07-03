# Cross-module interface contract

Every module in `motif_circuits/` MUST follow the signatures and data formats
in this document. Scripts glue modules together only through these surfaces.
Coordinate conventions: **frames** are 50 Hz EnCodec frames; **steps** are
transformer sequence steps; conversion only via `motif_circuits.delay_map.DelayMap`
(default instance `MUSICGEN_DELAY_MAP`). Frame intervals are half-open
`[start, end)`.

## 0. Repo layout & conventions

```
data/stimuli/<CAT>/audio/<id>.wav        # 32 kHz mono float32 WAV
data/stimuli/<CAT>/midi/<id>.mid
data/stimuli/<CAT>/codes/<id>.npy        # int16 [K=4, T] EnCodec codes
data/stimuli/<CAT>/manifest.jsonl        # one StimulusRecord per line
data/stimuli/<CAT>/meta.json             # generation config snapshot
results/<model>/screening/...            # model in {small, medium, large}
results/<model>/patching/...
results/<model>/ablation/...
results/<model>/knob/...
```

- Heavy deps (`torch`, `audiocraft`, `librosa`, `pretty_midi`) are imported
  lazily inside functions/classes that need them, or guarded, so that pure
  NumPy modules stay importable everywhere. `numpy`/`scipy` may be imported
  at module top.
- All randomness goes through `numpy.random.default_rng(seed)`; seeds come
  from configs. No global seeding.
- All public functions carry numpy-style docstrings; comments in English.

## 1. StimulusRecord (manifest.jsonl line)

```json
{
  "id": "S2_000123_r0",
  "category": "S1|S2|S3|S4|S5|S6",
  "sr": 32000, "frame_rate": 50,
  "bpm": 120.0, "key_root": 4, "scale": "minor", "program": 25,
  "velocity": 96, "seed": 123, "render_seed": 0,
  "motif": {"pitches": [64, 67, ...], "onsets_beats": [0.0, 0.5, ...],
             "durations_beats": [0.5, ...]},
  "transform": {"type": "exact|transpose|rhythm|timbre|rhythm_scramble|none",
                 "semitones": 5, "program_b": 40},
  "segments": {"A": [f0, f1], "G": [f0, f1], "A_prime": [f0, f1]},
  "phi": {"a_prime_frames": [int...], "a_frames": [int...]},
  "wav_path": "audio/S2_000123_r0.wav",
  "midi_path": "midi/S2_000123_r0.mid",
  "pair_id": "S6_..." | null,
  "duration_s": 10.0
}
```

- `phi` is the ground-truth frame alignment: `phi.a_prime_frames[i]` in A'
  corresponds to `phi.a_frames[i]` in A. For S6 `phi` is null. For S5 `phi`
  maps rhythm-skeleton onsets (used only to *test* that candidate heads do NOT
  fire).
- `pair_id` links an S1/S2/S3 sample to its patching partner (same prelude
  A+G, unrelated continuation) stored in S6.

Python-side dataclass: `motif_circuits.stimuli.StimulusRecord`
(`from_json/to_json`, plus `load_manifest(path) -> list[StimulusRecord]`,
`save_manifest(records, path)` in `motif_circuits.stimuli.build`; re-exported
from the *package* `motif_circuits.stimuli`).

## 2. Stimuli (`motif_circuits/stimuli/`)

```python
# motif.py  (pure numpy — no rendering deps)
@dataclass Motif: pitches: list[int]; onsets_beats: list[float]; durations_beats: list[float]
sample_motif(rng, scale_root, scale_type, n_notes_range=(4,12), n_bars=(1,2)) -> Motif
transpose(motif, semitones) -> Motif
rhythm_variation(motif, rng, mode="augment|diminish|syncopate") -> Motif   # pitch contour preserved
rhythm_scramble_pitches(motif, rng, scale_root, scale_type) -> Motif       # S5: rhythm kept, pitches resampled
sample_gap_material(rng, scale_root, scale_type, n_beats, avoid: Motif) -> Motif
motif_frames(motif, bpm, frame_rate, offset_s) -> (onset_frames, dur_frames)

# midi_render.py
build_midi(sections: list[tuple[Motif, float, int]], bpm, out_path) -> None
    # sections: (material, start_beat, program)
render_midi(midi_path, wav_path, sr=32000, soundfont=None, gain=0.7) -> None
    # FluidSynth CLI first, pyfluidsynth fallback; raises RuntimeError with
    # install hint if neither is available
peak_normalize(wav, headroom_db=1.0) -> wav

# build.py
build_category(cat: str, cfg: dict, out_dir: Path, rng) -> list[StimulusRecord]
build_all(cfg: dict, out_root: Path) -> None
compute_phi(record_fields...) -> (a_prime_frames, a_frames)   # ground truth from MIDI timing
```

S-category construction (10 s @ 50 Hz = 500 frames): A (1–2 bars) + G (random
gap, no A material) + A′. Randomize key, bpm (80–140), program, velocity.
≥3 render seeds per logical sample -> distinct `render_seed` (sets FluidSynth
gain jitter ±5% & velocity jitter ±3, so EnCodec sees different signals).

## 3. Model access (`motif_circuits/model/`)

```python
# loader.py
load_musicgen(size: str = "small", device: str = "cuda") -> "MusicGen"
    # asserts: DelayedPatternProvider, delays==[0,1,2,3], no prepend-fuser
model_geometry(model) -> ModelGeometry  # dataclass: n_layers, n_heads, d_model, d_head, n_q, card, frame_rate, sample_rate
null_condition_tensors(model, batch_size) -> ConditionTensors
text_condition_tensors(model, descriptions: list[str|None]) -> ConditionTensors
delay_map_for(model) -> DelayMap      # DelayMap.from_audiocraft(model.lm.pattern_provider)

# teacher_forcing.py
encode_audio(model, wav: np.ndarray | torch.Tensor, sr: int) -> torch.Tensor  # [1,K,T] long
decode_codes(model, codes) -> np.ndarray  # [1, T_samples] float
teacher_forcing_logprobs(model, codes, condition_tensors=None) -> TFResult
    # TFResult: logits [B,K,T,card] (cpu float32), logprob_true [B,K,T], mask [B,K,T]
    # runs lm.compute_predictions under model.autocast, no_grad

# hooks.py  — all context managers, all reset streaming-safe
class AttentionCapture:
    """Recompute per-head self-attention probs from captured self_attn inputs.
    AttentionCapture(lm, layers: list[int] | None, dtype=np.float16,
                     reduce: Callable[[int, np.ndarray], Any] | None = None)
    After forward: .attention[layer] -> [B, H, S, S] numpy (if reduce is None);
    if reduce is given, it is called per layer with the [B,H,S,S] float32 array
    and .reduced[layer] stores its return value (attention is not kept).
    """
class HeadOutputCapture:
    """Capture out_proj inputs per layer: .z[layer] -> [B, S, H, d_head] (cpu torch)."""
@dataclass HeadIntervention:
    layer: int; head: int
    mode: str            # 'zero' | 'mean' | 'scale' | 'patch'
    value: float | torch.Tensor | None = None
        # 'scale': gamma float; 'mean': [d_head] tensor; 'patch': [S, d_head] or [B, S, d_head]
    steps: np.ndarray | None = None   # absolute sequence steps to affect; None = all
class HeadInterventions:
    """Context manager applying a list of HeadIntervention via out_proj pre-hooks.
    Tracks absolute step offset per layer across streaming calls (+= T per call).
    reset() must be called between generation runs (context entry auto-resets).
    Applies to the full batch (both CFG halves)."""

# generate.py
generate_with_interventions(model, interventions: list[HeadIntervention] | None,
    *, descriptions=None, prompt_wav=None, prompt_sr=None, num_samples=1,
    duration=10.0, seed=0, **gen_params) -> GenResult
    # GenResult: wav [B, 1, T_samples] numpy float32, codes [B,K,T] numpy, sr
```

## 4. Analysis (`motif_circuits/analysis/`)

```python
# ../utils/chroma.py
chroma_features(wav, sr, frame_rate=50, method="auto") -> np.ndarray  # [T, 12], L2-normalized rows; hop = sr//frame_rate
onset_envelope(wav, sr, frame_rate=50) -> np.ndarray                  # [T]

# alignment.py — all return AlignResult(a_prime_frames, a_frames, meta: dict)
align_identity(seg_a: tuple[int,int], seg_ap: tuple[int,int]) -> AlignResult
align_transposition_invariant(chroma_a, chroma_ap, seg_a, seg_ap) -> AlignResult
    # circular cross-correlation over 12 bins; meta['shift'] = estimated semitones
align_dtw(feat_a, feat_ap, seg_a, seg_ap) -> AlignResult
    # feat = [chroma | onset_env]; scipy-based DTW, meta['path_cost']

# null_model.py
lag_spectrum(frame_attn: np.ndarray, max_lag: int) -> np.ndarray
    # frame_attn [H, T, T] -> P_h(l) [H, max_lag+1]; mean over valid (t, t-l)
class NullModel:
    fit(spectra: list[np.ndarray]) -> NullModel        # average over S6 corpus
    null_at_lags(lags: np.ndarray) -> np.ndarray        # [H, len(lags)]
    periodic_heads(beat_lag, bar_lag, z_thresh=4.0) -> np.ndarray  # bool [H] (per layer caller-side)

# scores.py — all operate on ONE sample and return per-head arrays
token_induction_score(step_attn, phi, dm, T, codebook) -> np.ndarray        # [H]; attn to step(phi(t)+1, k)
motif_induction_score(frame_attn, phi, window=2) -> np.ndarray              # [H]; mass in +-window around phi(t)+1
copy_score_substitution(...): implemented in patching.py (needs model)
# each *_score has a matching `..._null(spectrum, phi, ...)` -> [H] baseline

# stats.py (pure numpy)
permutation_test(observed: np.ndarray, null_samples: np.ndarray) -> p (one-sided)
permuted_phis(phi, rng, n, t_min, t_max) -> iterator of shifted/shuffled phi
benjamini_hochberg(pvals: np.ndarray, q=0.05) -> bool mask
head_table(scores: dict[str, np.ndarray]) -> structured summary for npz/json

# patching.py
@dataclass PatchTargets: layer_heads: list[tuple[int,int]]; frames: np.ndarray; codebooks: list[int]
run_patching_pair(model, clean_codes, corrupted_codes, targets, ct, L=25) -> PatchResult
    # denoising direction: clean activations -> corrupted forward
    # PatchResult: delta_clean, delta_corr, delta_patched, recovery R, per-codebook logprobs
head_accumulation_curve(model, ranked_heads, pair_iter, ...) -> curve arrays
```

## 5. Metrics (`motif_circuits/metrics/`)

```python
ssm.chroma_ssm(chroma) -> [T, T]; ssm.stripe_energy(ssm, min_lag, max_lag) -> float
ssm.foote_novelty(ssm, kernel_size) -> [T]
recurrence.motif_recurrence(prompt_chroma_A, cont_chroma, taus=np.arange(0.6, 0.91, 0.05))
    -> dict tau -> bool (any window with transposition-invariant corr > tau)
loop.loop_score(codes, wav, sr) -> LoopResult(ngram_rate, autocorr_peak, is_loop)
    # is_loop = ngram_rate > .8 AND autocorr collapse, on final 5 s
quality.fad_placeholder / clap_score — optional deps, raise with install hint
```

## 6. Script contracts (`scripts/`)

Every script: `python scripts/XX_name.py --config configs/foo.yaml [--override key=value ...]`,
writes into `results/...`/`data/...` as per §0, logs to stdout + a `run.json`
(config snapshot, git rev, timestamps) next to its outputs. Scripts import
ONLY public APIs above. `00_smoke_test.py` must run in <2 min on GPU server
and validate: model load, DelayMap.verify_against_audiocraft, encode/decode
roundtrip, AttentionCapture rows sum to 1, HeadIntervention zero-ablation
changes logits, generation with and without interventions.

## 7. Score definitions (normative, from the research plan)

- `Null_h(t) = P_h(l_t)`, `l_t = t - (phi(t)+1)`.
- `IS_tok^(k)(h) = E_t[ alpha_h(step(t,k) -> step(phi(t)+1, k)) ] - Null_h`.
- `IS_motif(h) = E_t[ sum_{s in W(phi(t)+1, w)} alpha_frame_h(t -> s) ] - Null_h`,
  w = ±2 frames, alpha_frame aggregated per §3 DelayMap.aggregate_frames.
- Candidate head criteria: (a) FDR-corrected permutation p < .05 vs null AND
  (b) NOT significant on S5.
- Patching recovery `R = (Delta_patched - Delta_corr) / (Delta_clean - Delta_corr)`,
  Delta = mean log-prob of true tokens over first L=25 frames of A′.
