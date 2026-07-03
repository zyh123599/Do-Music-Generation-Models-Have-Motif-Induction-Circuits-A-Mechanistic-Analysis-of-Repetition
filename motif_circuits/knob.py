"""Motif Recurrence Knob: a training-free repetition dial (research plan §8).

Scales the outputs of the candidate motif-induction heads by a factor gamma
(> 1 amplifies motif recurrence, < 1 suppresses it / repairs loops). This is
a thin, audiocraft-free wrapper producing :class:`HeadIntervention` lists;
apply them via ``generate_with_interventions`` or ``HeadInterventions``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import typing as tp

from .model.hooks import HeadIntervention

__all__ = ["MotifRecurrenceKnob"]

logger = logging.getLogger(__name__)


class MotifRecurrenceKnob:
    """Gamma-scaling knob over a set of candidate heads.

    Parameters
    ----------
    heads : list of (layer, head)
        Heads to scale (typically the top-K screening candidates).
    gamma : float
        Output scale; 1.0 is a no-op, 0.0 ablates the heads.
    """

    def __init__(self, heads: tp.Sequence[tp.Tuple[int, int]], gamma: float):
        if not heads:
            raise ValueError("need at least one head")
        self.heads = [(int(l), int(h)) for l, h in heads]
        self.gamma = float(gamma)

    def interventions(self) -> tp.List[HeadIntervention]:
        """The knob as a list of 'scale' interventions (all steps)."""
        return [HeadIntervention(layer=l, head=h, mode="scale",
                                 value=self.gamma)
                for l, h in self.heads]

    # ------------------------------------------------------------------
    @classmethod
    def from_candidates(cls, candidates_json: tp.Union[str, Path],
                        top_k: int, gamma: float) -> "MotifRecurrenceKnob":
        """Build a knob from the ranked ``candidates.json`` of scripts/04.

        The file holds a ranked (best-first) list of dicts with at least
        ``layer`` and ``head`` keys; the top ``top_k`` entries are used.
        """
        path = Path(candidates_json)
        entries = json.loads(path.read_text())
        if isinstance(entries, dict):          # allow {"candidates": [...]}
            entries = entries.get("candidates", [])
        if not entries:
            raise ValueError(f"no candidate heads in {path}")
        heads = [(int(e["layer"]), int(e["head"])) for e in entries[:top_k]]
        logger.info("Knob over %d heads (gamma=%.2f): %s",
                    len(heads), gamma, heads)
        return cls(heads, gamma)

    @classmethod
    def suppress(cls, heads: tp.Sequence[tp.Tuple[int, int]],
                 gamma: float = 0.5) -> "MotifRecurrenceKnob":
        """Suppression preset (loop repair, T2)."""
        if not 0.0 <= gamma < 1.0:
            raise ValueError("suppress expects gamma in [0, 1)")
        return cls(heads, gamma)

    @classmethod
    def enhance(cls, heads: tp.Sequence[tp.Tuple[int, int]],
                gamma: float = 2.0) -> "MotifRecurrenceKnob":
        """Enhancement preset (motif-recurrence boost, T1)."""
        if gamma <= 1.0:
            raise ValueError("enhance expects gamma > 1")
        return cls(heads, gamma)
