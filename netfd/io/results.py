"""Helpers for writing experiment outputs (JSON summaries, npz blobs)."""

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np


def ensure_outdir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(payload: Dict[str, Any], path: str, indent: int = 2) -> None:
    """Save a dict as JSON, converting numpy types to plain Python."""
    def _convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(payload, f, indent=indent, default=_convert)


def save_npz(path: str, **arrays) -> None:
    """Save arrays as compressed npz."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **arrays)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    """Load an npz file into a regular dict."""
    return dict(np.load(path, allow_pickle=True))
