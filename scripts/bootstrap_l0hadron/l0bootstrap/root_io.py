from pathlib import Path

import numpy as np
import ROOT

from TrackerOnlyEmu.emulation.run2_rdx import XGB_TRAIN_BRANCHES


ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.PyConfig.DisableRootLogon = True
ROOT.gROOT.SetBatch(True)

TARGET_BRANCH = "d0_L0HadronDecision_TOS"
EXTRA_SOURCE_BRANCHES = (
    "runNumber",
    "eventNumber",
    "NumSPDHits",
    "FitVar_q2",
    "FitVar_Mmiss2",
    "FitVar_El",
)

DERIVED_BRANCHES = {
    "d0_l0_hadron_tos": ("d0_L0HadronDecision_TOS", 1.0),
    "d0_pt": ("d0_PT", 1.0e-3),
    "k_pt": ("k_PT", 1.0e-3),
    "pi_pt": ("pi_PT", 1.0e-3),
    "d0_p": ("d0_P", 1.0e-3),
    "k_p": ("k_P", 1.0e-3),
    "pi_p": ("pi_P", 1.0e-3),
    "nspdhits": ("NumSPDHits", 1.0),
    "q2": ("FitVar_q2", 1.0e-6),
    "mmiss2": ("FitVar_Mmiss2", 1.0e-6),
    "el": ("FitVar_El", 1.0e-3),
}


def required_source_branches():
    return tuple(dict.fromkeys((*XGB_TRAIN_BRANCHES, TARGET_BRANCH, *EXTRA_SOURCE_BRANCHES)))


def input_files(input_path):
    path = Path(input_path)
    if path.is_dir():
        files = []
        for item in sorted(path.glob("*.root")):
            name = item.name
            if "aux_hammer" in name or "aux_trk" in name:
                continue
            files.append(item)
        return files
    return [path]


def make_chain(files, tree_name):
    chain = ROOT.TChain(tree_name)
    for path in files:
        chain.Add(str(path))
    return chain


def validate_source(chain, branches):
    missing = [branch for branch in branches if not chain.GetBranch(branch)]
    if missing:
        raise RuntimeError("Input tree is missing required branches: " + ", ".join(missing))
    if chain.GetEntries() <= 0:
        raise RuntimeError("Input tree has no entries")


def _set_active_branches(tree, branch_names):
    tree.SetBranchStatus("*", 0)
    for branch in branch_names:
        if tree.GetBranch(branch):
            tree.SetBranchStatus(branch, 1)


def copy_entries(source_tree, entry_indexes, output_path, output_tree_name, branch_names):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _set_active_branches(source_tree, branch_names)

    output_file = ROOT.TFile(str(output_path), "RECREATE")
    output_file.cd()
    output_tree = source_tree.CloneTree(0)
    output_tree.SetName(output_tree_name)

    for entry in sorted(int(value) for value in entry_indexes):
        source_tree.GetEntry(entry)
        output_tree.Fill()

    output_tree.Write()
    output_file.Close()


def write_initial_split(input_path, source_tree, output_tree, train_fraction, seed, train_path, test_path):
    files = input_files(input_path)
    chain = make_chain(files, source_tree)
    branches = required_source_branches()
    validate_source(chain, branches)

    n_entries = int(chain.GetEntries())
    rng = np.random.default_rng(seed)
    indexes = rng.permutation(n_entries)
    n_train = int(n_entries * train_fraction)

    copy_entries(chain, indexes[:n_train], train_path, output_tree, branches)
    copy_entries(chain, indexes[n_train:], test_path, output_tree, branches)
    return n_train, n_entries - n_train


def write_bootstrap_sample(source_path, tree_name, output_path, seed):
    source_file = ROOT.TFile.Open(str(source_path), "READ")
    source_tree = source_file.Get(tree_name)
    if not source_tree:
        raise RuntimeError(f"Could not read {tree_name} from {source_path}")
    n_entries = int(source_tree.GetEntries())
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, n_entries, size=n_entries)
    copy_entries(source_tree, indexes, output_path, tree_name, required_source_branches())
    source_file.Close()


def rdf_arrays(path, tree_name, branches):
    frame = ROOT.RDataFrame(tree_name, str(path))
    return frame.AsNumpy(columns=list(branches))


def read_branch_array(path, tree_name, branch):
    frame = ROOT.RDataFrame(tree_name, str(path))
    column_names = set(str(name) for name in frame.GetColumnNames())
    if branch in column_names:
        return np.asarray(frame.AsNumpy(columns=[branch])[branch], dtype=float)
    if branch in DERIVED_BRANCHES:
        raw_branch, scale = DERIVED_BRANCHES[branch]
        if raw_branch not in column_names:
            raise RuntimeError(f"{path} does not contain {branch} or source branch {raw_branch}")
        values = frame.AsNumpy(columns=[raw_branch])[raw_branch]
        return np.asarray(values, dtype=float) * scale
    raise RuntimeError(f"{path} does not contain branch {branch}")
