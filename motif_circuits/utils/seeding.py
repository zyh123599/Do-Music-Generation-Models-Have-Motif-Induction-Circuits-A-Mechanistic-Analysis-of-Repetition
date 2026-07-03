"""Deterministic seeding across numpy and (when present) torch."""
from __future__ import annotations

import logging

import numpy as np

__all__ = ["seed_everything"]

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> np.random.Generator:
    """Seed torch (CPU+CUDA, when importable) and return a numpy Generator.

    Library code receives the returned generator explicitly; the global torch
    seeds cover sampling inside audiocraft's generation loop.
    """
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        logger.debug("torch not installed; numpy-only seeding")
    return np.random.default_rng(int(seed))
