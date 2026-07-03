"""Audio quality / prompt adherence wrappers: FAD (fadtk) and CLAP score.

Research plan section 7, "quality and prompt adherence": mechanism
interventions must not buy repetition control at the cost of audio quality.
Both metrics depend on heavy optional packages that are only installed on
the GPU server; import failures raise ``RuntimeError`` with an exact
``pip install ...`` hint so screening scripts fail loudly and actionably.
The happy paths are exercised on the GPU server, not in CPU-only CI.
"""
from __future__ import annotations

import logging
import typing as tp

import numpy as np

__all__ = ["fad_score", "fad_placeholder", "clap_score", "compute_quality_report"]

logger = logging.getLogger(__name__)

_FADTK_HINT = (
    "fadtk is required for FAD computation but is not installed. "
    "Install it on the GPU server with: pip install fadtk"
)
_CLAP_HINT = (
    "laion_clap is required for CLAP score computation but is not installed. "
    "Install it on the GPU server with: pip install laion_clap"
)


def fad_score(
    dir_a: str,
    dir_b: str,
    model_name: str = "clap-laion-music",
    workers: int = 8,
) -> float:
    """Frechet Audio Distance between two directories of audio files.

    Thin wrapper over ``fadtk`` (Gui et al. 2024 music-adapted FAD/FD).
    ``dir_a`` is typically the reference set (e.g. baseline generations or a
    real-music corpus) and ``dir_b`` the intervention set.

    Parameters
    ----------
    dir_a, dir_b : str
        Directories of audio files to compare.
    model_name : str
        fadtk embedding model name (default the music-adapted CLAP model).
    workers : int
        Parallel audio-loading workers.

    Returns
    -------
    float
        FAD between the two sets (lower = more similar).

    Raises
    ------
    RuntimeError
        If ``fadtk`` is not installed (with a pip install hint).
    """
    try:
        from fadtk.fad import FrechetAudioDistance
        from fadtk.model_loader import get_all_models
    except ImportError as exc:  # pragma: no cover - exact path exercised in tests
        raise RuntimeError(_FADTK_HINT) from exc

    models = {m.name: m for m in get_all_models()}
    if model_name not in models:
        raise ValueError(
            f"unknown fadtk model {model_name!r}; available: {sorted(models)}"
        )
    fad = FrechetAudioDistance(models[model_name], audio_load_worker=workers, load_model=True)
    fad.cache_embedding_files(dir_a, models[model_name], workers)
    fad.cache_embedding_files(dir_b, models[model_name], workers)
    return float(fad.score(dir_a, dir_b))


#: Interface-contract alias (interfaces.md section 5 names this
#: ``quality.fad_placeholder``); identical to :func:`fad_score`.
fad_placeholder = fad_score


def clap_score(
    audio_paths: tp.Sequence[str],
    descriptions: tp.Sequence[str],
    model_kwargs: tp.Optional[dict] = None,
) -> float:
    """Mean CLAP audio-text cosine similarity over (audio, description) pairs.

    Thin wrapper over ``laion_clap``. Used to check prompt adherence of
    intervened generations against their text prompts.

    Parameters
    ----------
    audio_paths : sequence of str
        Paths to audio files (one per sample).
    descriptions : sequence of str
        Text prompts, aligned with ``audio_paths``.
    model_kwargs : dict, optional
        Forwarded to ``laion_clap.CLAP_Module`` (e.g. ``enable_fusion``).

    Returns
    -------
    float
        Mean cosine similarity between paired audio and text embeddings.

    Raises
    ------
    RuntimeError
        If ``laion_clap`` is not installed (with a pip install hint).
    """
    try:
        import laion_clap
    except ImportError as exc:  # pragma: no cover - exact path exercised in tests
        raise RuntimeError(_CLAP_HINT) from exc

    if len(audio_paths) != len(descriptions):
        raise ValueError(
            f"got {len(audio_paths)} audio files but {len(descriptions)} descriptions"
        )
    if len(audio_paths) == 0:
        raise ValueError("need at least one (audio, description) pair")

    model = laion_clap.CLAP_Module(**(model_kwargs or {}))
    model.load_ckpt()
    audio_emb = np.asarray(
        model.get_audio_embedding_from_filelist(list(audio_paths), use_tensor=False)
    )
    text_emb = np.asarray(model.get_text_embedding(list(descriptions), use_tensor=False))
    audio_emb /= np.maximum(np.linalg.norm(audio_emb, axis=1, keepdims=True), 1e-12)
    text_emb /= np.maximum(np.linalg.norm(text_emb, axis=1, keepdims=True), 1e-12)
    return float(np.mean(np.sum(audio_emb * text_emb, axis=1)))


def compute_quality_report(
    dir_a: str,
    dir_b: str,
    descriptions: tp.Optional[tp.Sequence[str]] = None,
    audio_paths: tp.Optional[tp.Sequence[str]] = None,
) -> dict:
    """Best-effort quality report combining FAD and (optionally) CLAP.

    Orchestrator stub: runs each available metric, records ``None`` plus the
    error message for metrics whose optional dependency is missing instead
    of aborting, so screening scripts always produce a report.

    Parameters
    ----------
    dir_a, dir_b : str
        Reference / candidate audio directories for FAD.
    descriptions : sequence of str, optional
        Text prompts for the CLAP score; CLAP is skipped when omitted.
    audio_paths : sequence of str, optional
        Audio files paired with ``descriptions`` (required for CLAP).

    Returns
    -------
    dict
        ``{"fad": float | None, "clap": float | None, "errors": dict[str, str]}``.
    """
    report: dict = {"fad": None, "clap": None, "errors": {}}
    try:
        report["fad"] = fad_score(dir_a, dir_b)
    except RuntimeError as exc:
        logger.warning("FAD unavailable: %s", exc)
        report["errors"]["fad"] = str(exc)
    if descriptions is not None and audio_paths is not None:
        try:
            report["clap"] = clap_score(audio_paths, descriptions)
        except RuntimeError as exc:
            logger.warning("CLAP unavailable: %s", exc)
            report["errors"]["clap"] = str(exc)
    return report
