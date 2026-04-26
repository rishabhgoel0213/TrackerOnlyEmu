import pickle
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from TrackerOnlyEmu.emulation.run2_rdx import XGB_TRAIN_BRANCHES

from .paths import REPO_ROOT, RUN_SCRIPT
from .root_io import TARGET_BRANCH, rdf_arrays


def _format_cmd(cmd):
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_command(cmd):
    print("+ " + _format_cmd(cmd))
    subprocess.run([str(part) for part in cmd], cwd=REPO_ROOT, check=True)


def train_xgb(input_path, tree_name, output_model, ntrees, max_depth, force=False):
    output_model = Path(output_model)
    if output_model.exists() and not force:
        print(f"skip existing model: {output_model}")
        return

    arrays = rdf_arrays(input_path, tree_name, (*XGB_TRAIN_BRANCHES, TARGET_BRANCH))
    features = np.array([arrays[name] for name in XGB_TRAIN_BRANCHES]).T
    target = np.asarray(arrays[TARGET_BRANCH], dtype=int)

    model = XGBClassifier(
        n_estimators=ntrees,
        max_depth=max_depth,
        use_label_encoder=False,
        eval_metric="mlogloss",
        reg_lambda=0.5,
    )
    print(f"training XGB model: {output_model}")
    model.fit(features, target)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    with output_model.open("wb") as model_file:
        pickle.dump(model, model_file)


def apply_xgb(input_path, tree_name, model_path, output_path, year, force=False):
    output_path = Path(output_path)
    if output_path.exists() and not force:
        print(f"skip existing output: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            RUN_SCRIPT,
            input_path,
            output_path,
            "--tree",
            tree_name,
            "--year",
            year,
            "--load",
            model_path,
            "--debug",
        ]
    )
