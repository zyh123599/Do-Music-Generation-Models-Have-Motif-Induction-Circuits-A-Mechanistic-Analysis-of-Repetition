"""MusicGen access layer: loading, hooks, teacher forcing, generation.

Importing this package requires torch; audiocraft itself is only needed at
call time (lazy imports), so hook classes are testable with mocks.
"""
from .loader import (ModelGeometry, load_musicgen, model_geometry,
                     delay_map_for, null_condition_tensors,
                     text_condition_tensors)
from .hooks import (AttentionCapture, HeadOutputCapture, HeadIntervention,
                    HeadInterventions, resolve_layers)
from .teacher_forcing import (TFResult, encode_audio, decode_codes,
                              teacher_forcing_logprobs, logprobs_of)
from .generate import GenResult, generate_with_interventions

__all__ = [
    "ModelGeometry", "load_musicgen", "model_geometry", "delay_map_for",
    "null_condition_tensors", "text_condition_tensors",
    "AttentionCapture", "HeadOutputCapture", "HeadIntervention",
    "HeadInterventions", "resolve_layers",
    "TFResult", "encode_audio", "decode_codes", "teacher_forcing_logprobs",
    "logprobs_of",
    "GenResult", "generate_with_interventions",
]
