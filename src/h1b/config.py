"""Load params.yaml and resolve data paths relative to the repository root."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# src/h1b/config.py -> repo root is two levels above the package directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS = REPO_ROOT / "params.yaml"


@dataclass(frozen=True)
class Paths:
    raw: Path
    interim: Path
    processed: Path
    holdout: Path
    reports: Path


def load_params(path: str | Path | None = None) -> dict[str, Any]:
    """Read the pipeline config. H1B_PARAMS overrides the default location (used by tests/Airflow)."""
    path = Path(path or os.environ.get("H1B_PARAMS", DEFAULT_PARAMS))
    with path.open() as f:
        return yaml.safe_load(f)


def get_paths(params: dict[str, Any], root: Path = REPO_ROOT) -> Paths:
    """Turn the relative directories in params['data'] into absolute paths, creating them if needed."""
    d = params["data"]
    paths = Paths(
        raw=root / d["raw_dir"],
        interim=root / d["interim_dir"],
        processed=root / d["processed_dir"],
        holdout=root / d["holdout_dir"],
        reports=root / d["reports_dir"],
    )
    for p in vars(paths).values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
