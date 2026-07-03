"""Result I/O: atomic npz/json writes and run.json provenance stamps."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import typing as tp

import numpy as np

__all__ = ["atomic_write_bytes", "save_npz", "load_npz", "save_json",
           "load_json", "write_run_json", "git_revision"]

logger = logging.getLogger(__name__)


def atomic_write_bytes(path: tp.Union[str, Path], data: bytes) -> None:
    """Write bytes via a same-directory temp file + rename (crash-safe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_npz(path: tp.Union[str, Path], arrays: tp.Dict[str, np.ndarray],
             meta: tp.Optional[dict] = None) -> None:
    """Atomically save named arrays (+ optional JSON meta under ``_meta``)."""
    import io as _io

    buf = _io.BytesIO()
    payload = dict(arrays)
    if meta is not None:
        payload["_meta"] = np.frombuffer(
            json.dumps(meta, default=str).encode(), dtype=np.uint8)
    np.savez_compressed(buf, **payload)
    atomic_write_bytes(path, buf.getvalue())


def load_npz(path: tp.Union[str, Path]
             ) -> tp.Tuple[tp.Dict[str, np.ndarray], tp.Optional[dict]]:
    """Load arrays saved by :func:`save_npz`; returns (arrays, meta)."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != "_meta"}
        meta = None
        if "_meta" in data.files:
            meta = json.loads(bytes(data["_meta"].tobytes()).decode())
    return arrays, meta


def save_json(path: tp.Union[str, Path], obj: tp.Any, indent: int = 2) -> None:
    """Atomically write pretty JSON (numpy scalars handled)."""

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    atomic_write_bytes(path, json.dumps(obj, indent=indent,
                                        default=_default).encode())


def load_json(path: tp.Union[str, Path]) -> tp.Any:
    return json.loads(Path(path).read_text())


def git_revision(repo_root: tp.Optional[tp.Union[str, Path]] = None) -> str:
    """Current git rev (short) or 'unknown' outside a repo / without git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root or Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_run_json(out_dir: tp.Union[str, Path], config: dict,
                   extra: tp.Optional[dict] = None) -> None:
    """Stamp an output directory with config snapshot + provenance."""
    import datetime

    payload = {
        "config": config,
        "git_revision": git_revision(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    save_json(Path(out_dir) / "run.json", payload)
