"""Motif induction circuit analysis for autoregressive music transformers.

Subpackages import heavy dependencies (torch/audiocraft/librosa) lazily; the
top-level package only exposes the NumPy-based coordinate core.
"""
from .delay_map import DelayMap, MUSICGEN_DELAY_MAP

__version__ = "0.1.0"
__all__ = ["DelayMap", "MUSICGEN_DELAY_MAP", "__version__"]
