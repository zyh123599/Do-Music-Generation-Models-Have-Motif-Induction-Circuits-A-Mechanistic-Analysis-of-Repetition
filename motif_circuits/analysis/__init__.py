"""Circuit-analysis toolbox: alignment, null model, scores, stats, patching.

``patching`` is not imported eagerly (it lazily needs torch at call time but
keeping the top-level import light lets NumPy-only machines use the rest).
"""
from .alignment import (AlignResult, align_identity,
                        align_transposition_invariant, align_dtw)
from .null_model import NullModel, lag_spectrum
from .scores import (token_induction_score, token_induction_null,
                     motif_induction_score, motif_induction_null,
                     dla_copy_score)
from .stats import (permutation_test, permuted_phis, benjamini_hochberg,
                    head_table)
from .patching import PatchTargets, PatchResult, recovery_rate

__all__ = [
    "AlignResult", "align_identity", "align_transposition_invariant",
    "align_dtw",
    "NullModel", "lag_spectrum",
    "token_induction_score", "token_induction_null",
    "motif_induction_score", "motif_induction_null", "dla_copy_score",
    "permutation_test", "permuted_phis", "benjamini_hochberg", "head_table",
    "PatchTargets", "PatchResult", "recovery_rate",
]
