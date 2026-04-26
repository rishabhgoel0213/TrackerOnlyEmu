#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BOOTSTRAP_DIR = SCRIPT_DIR / "bootstrap_l0hadron"
sys.path.insert(0, str(BOOTSTRAP_DIR))

STEPS = (
    "all",
    "prepare",
    "central",
    "xgb-weights",
    "train-test",
    "real-validation",
    "plot",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bootstrap uncertainty pipeline for Run 2 RDX L0Hadron XGB emulation."
    )
    parser.add_argument("config", help="YAML config describing the input sample and plot ranges.")
    parser.add_argument(
        "-s",
        "--step",
        choices=STEPS,
        default="all",
        help="Run one step, or the full pipeline. Defaults to all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate outputs even when resumable files already exist.",
    )
    parser.add_argument(
        "--plot-kind",
        action="append",
        choices=("xgb-weights", "train-test", "real-validation", "ratio", "2d"),
        help="Limit --step plot to one plot family. Repeat for multiple families.",
    )
    parser.add_argument(
        "--plot-branch",
        action="append",
        help="Limit --step plot to one branch. Repeat for multiple branches.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from l0bootstrap.config import load_config
    from l0bootstrap.pipeline import Pipeline

    config = load_config(args.config)
    Pipeline(
        config,
        force=args.force,
        plot_kinds=args.plot_kind,
        plot_branches=args.plot_branch,
    ).run(args.step)


if __name__ == "__main__":
    main()
