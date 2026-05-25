"""
Shared helpers for the experiment scripts.

Every experiment script accepts a single positional argument: the YAML
config path. `--override key=value` allows quick scalar overrides without
editing the YAML.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# CLI parser shared across experiments
# ---------------------------------------------------------------------------

def make_parser(default_config: str, description: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", "-c", default=default_config,
        help=f"Path to experiment YAML (default: {default_config}).",
    )
    parser.add_argument(
        "--outdir", "-o", default=None,
        help="Override the outdir specified in the YAML.",
    )
    parser.add_argument(
        "--override", "-O", action="append", default=[],
        metavar="key.path=value",
        help=("Override a scalar config value. Can be repeated. "
              "Example: --override ppo.total_timesteps=10000"),
    )
    return parser


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> None:
    """Apply `key.path=value` strings to `cfg` in place.

    Value parsing tries int, float, bool, then leaves it as a string.
    """
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"--override expects key=value, got {spec!r}")
        key_path, raw = spec.split("=", 1)
        keys = key_path.split(".")

        # Parse value
        parsed: Any = raw
        for caster in (int, float):
            try:
                parsed = caster(raw)
                break
            except ValueError:
                continue
        else:
            if raw.lower() in ("true", "false"):
                parsed = raw.lower() == "true"

        # Walk into nested dicts and set.
        node = cfg
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                raise KeyError(f"--override path not found: {key_path}")
            node = node[k]
        node[keys[-1]] = parsed


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def banner(title: str, lines: List[str] = None) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)
    if lines:
        for ln in lines:
            print(f"  {ln}")
        print("-" * 72)


def section(title: str) -> None:
    print()
    print(f"[{title}]")


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

def resolve_path(path: str, project_root: Path = None) -> Path:
    """Resolve a relative path against the project root if not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    return (project_root / p).resolve()
