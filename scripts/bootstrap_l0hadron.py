#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
BOOTSTRAP_DIR = SCRIPT_DIR / "bootstrap_l0hadron"
GEN_DIR = REPO_ROOT / "gen"

CREATE_SUBSETS_SRC = BOOTSTRAP_DIR / "createSubsets.cpp"
CREATE_SUBSETS_BIN = BOOTSTRAP_DIR / "createSubsets"
GENERATE_PLOTS_SRC = BOOTSTRAP_DIR / "generatePlots.cpp"
GENERATE_PLOTS_BIN = BOOTSTRAP_DIR / "generatePlots"

RUN_SCRIPT = SCRIPT_DIR / "run2-rdx-l0_hadron_tos.py"
DEFAULT_PLOT_NAME = "efficiency_plot_combined.png"
DEFAULT_NUM_SUBSETS = 100
DEFAULT_TRAIN_FRAC = 0.5
DEFAULT_TAG = "run"
CPP_STD = "c++17"
PIPELINE_STEPS = ["subset", "train", "test", "plot"]


def run(cmd: list[str] | str, cwd: Path = REPO_ROOT) -> None:
    if isinstance(cmd, str):
        print(f"+ {cmd}")
        subprocess.run(cmd, cwd=str(cwd), shell=True, check=True)
        return
    print("+ " + " ".join(shlex.quote(token) for token in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def get_tagged_run_dir(tag: str) -> Path:
    run_dir = GEN_DIR / tag
    if not run_dir.exists():
        return run_dir
    suffix = 2
    while (GEN_DIR / f"{tag}-{suffix}").exists():
        suffix += 1
    return GEN_DIR / f"{tag}-{suffix}"


def build_root_cpp(src: Path, out: Path) -> None:
    src_q = shlex.quote(str(src))
    out_q = shlex.quote(str(out))
    cmd = (
        f"g++ -fdiagnostics-color=always -g $(root-config --cflags) "
        f"-std={CPP_STD} {src_q} -o {out_q} "
        "$(root-config --libs) -lstdc++fs"
    )
    run(cmd)


def run_subset_step(
    input_path: str | None, num_subsets: int, train_frac: float, subset_dir: Path
) -> None:
    subset_dir.mkdir(parents=True, exist_ok=True)
    build_root_cpp(CREATE_SUBSETS_SRC, CREATE_SUBSETS_BIN)
    resolved_input = input_path or str(REPO_ROOT / "samples" / "run2-rdx-sample.root")
    cmd = [
        str(CREATE_SUBSETS_BIN),
        resolved_input,
        str(num_subsets),
        str(train_frac),
        str(subset_dir),
    ]
    run(cmd)


def run_train_step(run_dir: Path, subset_dir: Path) -> None:
    for input_file in sorted(subset_dir.glob("train_subset_*.root")):
        base = input_file.stem
        output_file = run_dir / f"{base}_trained_output.root"
        pickle_file = run_dir / f"{base}_xgb.pickle"
        run(
            [
                sys.executable,
                str(RUN_SCRIPT),
                str(input_file),
                str(output_file),
                "--tree",
                "DecayTree",
                "--dump",
                str(pickle_file),
            ]
        )


def run_test_step(run_dir: Path, subset_dir: Path) -> None:
    for input_file in sorted(subset_dir.glob("test_subset_*.root")):
        base = input_file.stem
        suffix = base[len("test_subset_"):] if base.startswith("test_subset_") else base
        train_base = f"train_subset_{suffix}"
        output_file = run_dir / f"{base}_output.root"
        model_file = run_dir / f"{train_base}_xgb.pickle"
        run(
            [
                sys.executable,
                str(RUN_SCRIPT),
                str(input_file),
                str(output_file),
                "--tree",
                "DecayTree",
                "--load",
                str(model_file),
                "--debug",
            ]
        )


def run_plot_step(
    run_dir: Path,
    plot_dir: Path,
    plot_name: str,
    branches: list[str],
    plot_ranges: dict[str, dict[str, float]],
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    build_root_cpp(GENERATE_PLOTS_SRC, GENERATE_PLOTS_BIN)
    branch_specs = []
    for branch in branches:
        ranges = plot_ranges[branch]
        branch_specs.append(
            f"{branch}:{ranges['x_min']}:{ranges['x_max']}:{ranges['y_min']}:{ranges['y_max']}"
        )
    run(
        [
            str(GENERATE_PLOTS_BIN),
            plot_name,
            "--gen-dir",
            str(run_dir),
            "--plot-dir",
            str(plot_dir),
            *branch_specs,
        ]
    )


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap pipeline for L0Hadron."
    )
    parser.add_argument("config", help="Path to YAML config.")
    parser.add_argument(
        "-s",
        "--step",
        choices=PIPELINE_STEPS,
        help="Run exactly one step. If omitted, runs the full pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    input_path = config.get("input")
    plot_name = config.get("plot_name", config.get("plot-name", DEFAULT_PLOT_NAME))
    branches = config.get("branches", ["d0_pt"])
    num_subsets = config.get("num_subsets", config.get("num-subsets", DEFAULT_NUM_SUBSETS))
    train_frac = config.get("train_frac", config.get("train-frac", DEFAULT_TRAIN_FRAC))
    plot_ranges = config.get("plot_ranges", config.get("plot-ranges", {}))
    tag = config.get("tag", DEFAULT_TAG)
    requested_step = args.step

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    if requested_step is None:
        steps_to_run = PIPELINE_STEPS
        run_dir = get_tagged_run_dir(tag)
    elif requested_step == "subset":
        steps_to_run = [requested_step]
        run_dir = get_tagged_run_dir(tag)
    else:
        steps_to_run = [requested_step]
        run_dir = GEN_DIR / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    subset_dir = run_dir / "subsets"
    plot_dir = run_dir / "plots"
    print(f"Run directory: {run_dir}")

    if "subset" in steps_to_run:
        run_subset_step(input_path, num_subsets, train_frac, subset_dir)
    if "train" in steps_to_run:
        run_train_step(run_dir, subset_dir)
    if "test" in steps_to_run:
        run_test_step(run_dir, subset_dir)
    if "plot" in steps_to_run:
        run_plot_step(run_dir, plot_dir, plot_name, branches, plot_ranges)


if __name__ == "__main__":
    main()
