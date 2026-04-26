from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_BRANCH_RANGES = {
    "d0_pt": {"x_min": 0.0, "x_max": 20.0, "y_min": 0.0, "y_max": 0.95},
    "q2": {"x_min": -5.0, "x_max": 15.0, "y_min": 0.14, "y_max": 0.27},
    "mmiss2": {"x_min": -5.0, "x_max": 15.0, "y_min": 0.0, "y_max": 1.0},
    "el": {"x_min": 0.0, "x_max": 4.0, "y_min": 0.1, "y_max": 0.45},
    "nspdhits": {"x_min": 0.0, "x_max": 700.0, "y_min": 0.0, "y_max": 0.32},
}


@dataclass(frozen=True)
class BranchConfig:
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    bins: int


@dataclass(frozen=True)
class BootstrapConfig:
    input_path: Path
    tag: str
    branches: tuple[BranchConfig, ...]
    n_bootstraps: int = 100
    train_fraction: float = 0.5
    seed: int = 12345
    tree: str = "DecayTree"
    source_tree: str = "TupleB0/DecayTree"
    year: str = "2016"
    ntrees: int = 300
    max_depth: int = 4
    two_dimensional_bins: int = 12


def _coalesce(raw, *names, default=None):
    for name in names:
        if name in raw:
            return raw[name]
    return default


def _branch_config(name, ranges, default_bins):
    merged = dict(DEFAULT_BRANCH_RANGES.get(name, {}))
    merged.update(ranges.get(name, {}))
    missing = [key for key in ("x_min", "x_max", "y_min", "y_max") if key not in merged]
    if missing:
        raise ValueError(f"Missing plot range keys for {name}: {', '.join(missing)}")
    return BranchConfig(
        name=name,
        x_min=float(merged["x_min"]),
        x_max=float(merged["x_max"]),
        y_min=float(merged["y_min"]),
        y_max=float(merged["y_max"]),
        bins=int(merged.get("bins", default_bins)),
    )


def load_config(path):
    config_path = Path(path)
    with config_path.open() as config_file:
        raw = yaml.safe_load(config_file) or {}

    if "input" not in raw:
        raise ValueError(f"{config_path} must define an input path")

    branch_names = tuple(raw.get("branches", ("d0_pt",)))
    if not branch_names:
        raise ValueError("At least one plotting branch is required")

    default_bins = int(raw.get("num_bins", raw.get("bins", 20)))
    ranges = _coalesce(raw, "plot_ranges", "plot-ranges", default={})
    branches = tuple(_branch_config(name, ranges, default_bins) for name in branch_names)

    return BootstrapConfig(
        input_path=Path(raw["input"]).expanduser(),
        tag=str(raw.get("tag", "l0hadron")),
        branches=branches,
        n_bootstraps=int(_coalesce(raw, "n_bootstraps", "num_subsets", "num-subsets", default=100)),
        train_fraction=float(_coalesce(raw, "train_fraction", "train_frac", "train-frac", default=0.5)),
        seed=int(raw.get("seed", 12345)),
        tree=str(raw.get("tree", "DecayTree")),
        source_tree=str(raw.get("source_tree", raw.get("source-tree", "TupleB0/DecayTree"))),
        year=str(raw.get("year", "2016")),
        ntrees=int(raw.get("ntrees", 300)),
        max_depth=int(raw.get("max_depth", raw.get("max-depth", 4))),
        two_dimensional_bins=int(raw.get("two_dimensional_bins", raw.get("two-dimensional-bins", 12))),
    )
