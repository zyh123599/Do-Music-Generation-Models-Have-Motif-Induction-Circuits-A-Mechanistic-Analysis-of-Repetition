"""Output-side repetition metrics for free generation (research plan §7)."""
from .ssm import (chroma_ssm, lag_profile, stripe_energy, foote_novelty,
                  novelty_contrast)
from .recurrence import RecurrenceResult, motif_recurrence, recurrence_rate
from .loop import (LoopResult, loop_score, token_ngram_repetition,
                   audio_autocorr_collapse)
from .quality import fad_score, clap_score, compute_quality_report

__all__ = [
    "chroma_ssm", "lag_profile", "stripe_energy", "foote_novelty",
    "novelty_contrast",
    "RecurrenceResult", "motif_recurrence", "recurrence_rate",
    "LoopResult", "loop_score", "token_ngram_repetition",
    "audio_autocorr_collapse",
    "fad_score", "clap_score", "compute_quality_report",
]
