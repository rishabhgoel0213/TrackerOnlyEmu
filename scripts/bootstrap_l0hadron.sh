#!/usr/bin/env bash
# Usage:
#   scripts/bootstrap_l0hadron.sh [options]
# Flags:
#   -i, --input PATH        Input ROOT file/path used by the subset step.
#   -p, --plot-name NAME    Base filename for output plot(s).
#   -b, --branches ...      Branch names to use for plotting bins.
#   -s, --step ...          Steps to run (subset/train/test/plot), order ignored.
#   -h, --help              Print CLI help and exit.
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BOOTSTRAP_DIR="${SCRIPT_DIR}/bootstrap_l0hadron"

GEN_DIR="${REPO_ROOT}/gen"
SUBSETS_DIR="${BOOTSTRAP_DIR}/subsets"
PLOTS_DIR="${BOOTSTRAP_DIR}/plots"

CREATE_SUBSETS_SRC="${BOOTSTRAP_DIR}/createSubsets.cpp"
CREATE_SUBSETS_BIN="${BOOTSTRAP_DIR}/createSubsets"
GENERATE_PLOTS_SRC="${BOOTSTRAP_DIR}/generatePlots.cpp"
GENERATE_PLOTS_BIN="${BOOTSTRAP_DIR}/generatePlots"

TRAIN_SCRIPT="${BOOTSTRAP_DIR}/train_all.sh"
TEST_SCRIPT="${BOOTSTRAP_DIR}/test_all.sh"
DEFAULT_FINAL_PLOT_NAME="efficiency_plot_combined.png"
DEFAULT_CPP_STD="c++14"

build_root_cpp() {
  local src="$1"
  local out="$2"
  local cpp_std="${BOOTSTRAP_CPP_STD:-${DEFAULT_CPP_STD}}"
  local compile_cmd

  compile_cmd="root_cflags=\$(root-config --cflags | sed -E 's/(^| )-std=[^ ]+//g'); g++ -fdiagnostics-color=always -g \${root_cflags} -std=${cpp_std} \"${src}\" -o \"${out}\" \$(root-config --libs) -lstdc++fs"
  nix-shell --pure -p root gcc --run "${compile_cmd}"
}

run_in_python_root_shell() {
  local cmd="$1"
  local run_cmd
  run_cmd="unset PYTHONHOME VIRTUAL_ENV CONDA_PREFIX; root_lib=\$(root-config --libdir); export PYTHONNOUSERSITE=1; export PYTHONPATH=\"\${root_lib}:${REPO_ROOT}:\${PYTHONPATH:-}\"; export LD_LIBRARY_PATH=\"\${root_lib}:\${LD_LIBRARY_PATH:-}\"; ${cmd}"
  nix-shell --pure -p root gcc python3 python3Packages.numpy python3Packages.scikit-learn python3Packages.xgboost \
    --run "${run_cmd}"
}

cleanup_intermediate() {
  local -a files=(
    "${PLOTS_DIR}"/test_subset_*_eff.png
  )

  if ((${#files[@]})); then
    rm -f -- "${files[@]}"
  fi
}

require_glob() {
  local pattern="$1"
  local what="$2"
  local dir="${pattern%/*}"
  local name="${pattern##*/}"
  if [[ "${dir}" == "${pattern}" ]]; then
    dir="."
  fi
  if [[ ! -d "${dir}" ]] || ! find "${dir}" -maxdepth 1 -type f -name "${name}" -print -quit | grep -q .; then
    echo "Required files missing for ${what}: ${pattern}" >&2
    exit 1
  fi
}

clear_plot_outputs() {
  local final_plot_name="$1"
  shift
  local -a bin_branches=("$@")
  rm -f -- "${PLOTS_DIR}/${DEFAULT_FINAL_PLOT_NAME}" "${PLOTS_DIR}/${final_plot_name}"
  for branch in "${bin_branches[@]}"; do
    rm -f -- "${PLOTS_DIR}/${branch}_${final_plot_name}"
  done
}

run_subset_step() {
  local input_path="$1"
  rm -f -- "${SUBSETS_DIR}"/train_subset_*.root "${SUBSETS_DIR}"/test_subset_*.root
  rm -f -- "${GEN_DIR}"/train_subset_*_trained_output.root "${GEN_DIR}"/train_subset_*_xgb.pickle
  rm -f -- "${GEN_DIR}"/test_subset_*_output.root

  build_root_cpp "${CREATE_SUBSETS_SRC}" "${CREATE_SUBSETS_BIN}"
  if [[ -n "${input_path}" ]]; then
    "${CREATE_SUBSETS_BIN}" "${input_path}"
  else
    "${CREATE_SUBSETS_BIN}"
  fi

  require_glob "${SUBSETS_DIR}/train_subset_*.root" "subset step"
  require_glob "${SUBSETS_DIR}/test_subset_*.root" "subset step"
}

run_train_step() {
  require_glob "${SUBSETS_DIR}/train_subset_*.root" "train step"
  rm -f -- "${GEN_DIR}"/train_subset_*_trained_output.root "${GEN_DIR}"/train_subset_*_xgb.pickle
  run_in_python_root_shell "bash -e \"${TRAIN_SCRIPT}\""
  require_glob "${GEN_DIR}/train_subset_*_xgb.pickle" "train step"
}

run_test_step() {
  require_glob "${SUBSETS_DIR}/test_subset_*.root" "test step"
  require_glob "${GEN_DIR}/train_subset_*_xgb.pickle" "test step"
  rm -f -- "${GEN_DIR}"/test_subset_*_output.root
  run_in_python_root_shell "bash -e \"${TEST_SCRIPT}\""
  require_glob "${GEN_DIR}/test_subset_*_output.root" "test step"
}

run_plot_step() {
  local final_plot_name="$1"
  shift
  local -a bin_branches=("$@")

  require_glob "${GEN_DIR}/test_subset_*_output.root" "plot step"
  clear_plot_outputs "${final_plot_name}" "${bin_branches[@]}"
  build_root_cpp "${GENERATE_PLOTS_SRC}" "${GENERATE_PLOTS_BIN}"
  "${GENERATE_PLOTS_BIN}" "${final_plot_name}" "${bin_branches[@]}"
  for branch in "${bin_branches[@]}"; do
    if [[ ! -f "${PLOTS_DIR}/${branch}_${final_plot_name}" ]]; then
      echo "Final plot was not produced at ${PLOTS_DIR}/${branch}_${final_plot_name}" >&2
      exit 1
    fi
  done
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -i, --input PATH           Input ROOT file for subset creation.
  -p, --plot-name NAME       Base name for output plot files.
  -b, --branches LIST...     Space-separated branch list (can be repeated).
  -s, --step STEP...         Run selected steps (input order ignored): subset train test plot.
  -h, --help                 Show this help message.

Examples:
  $(basename "$0")
  $(basename "$0") -i /path/to/input.root -p efficiency_plot_2018.png
  $(basename "$0") -b d0_pt k_pt -b pi_pt
  $(basename "$0") -s plot train test
EOF
}

main() {
  local input_path=""
  local final_plot_name="${DEFAULT_FINAL_PLOT_NAME}"
  local -a bin_branches=()
  local -a steps=()
  while (($#)); do
    case "$1" in
      -i|--input)
        if (($# < 2)); then
          echo "Missing value for $1" >&2
          usage >&2
          exit 1
        fi
        input_path="$2"
        shift 2
        ;;
      -p|--plot-name)
        if (($# < 2)); then
          echo "Missing value for $1" >&2
          usage >&2
          exit 1
        fi
        final_plot_name="$2"
        shift 2
        ;;
      -b|--branches)
        if (($# < 2)); then
          echo "Missing value for $1" >&2
          usage >&2
          exit 1
        fi
        shift
        while (($#)) && [[ "$1" != -* ]]; do
          bin_branches+=("$1")
          shift
        done
        ;;
      -s|--step)
        if (($# < 2)); then
          echo "Missing value for $1" >&2
          usage >&2
          exit 1
        fi
        shift
        while (($#)) && [[ "$1" != -* ]]; do
          case "$1" in
            subset|train|test|plot)
              steps+=("$1")
              ;;
            *)
              echo "Unknown step: $1" >&2
              usage >&2
              exit 1
              ;;
          esac
          shift
        done
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        if (($#)); then
          echo "Unexpected positional arguments: $*" >&2
          usage >&2
          exit 1
        fi
        break
        ;;
      -*)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 1
        ;;
      *)
        echo "Unexpected positional argument: $1" >&2
        usage >&2
        exit 1
        ;;
    esac
  done

  if ((${#bin_branches[@]} == 0)); then
    bin_branches=("d0_pt")
  fi

  if ((${#steps[@]} == 0)); then
    steps=("subset" "train" "test" "plot")
  fi

  local run_subset=0
  local run_train=0
  local run_test=0
  local run_plot=0
  for step in "${steps[@]}"; do
    case "${step}" in
      subset) run_subset=1 ;;
      train) run_train=1 ;;
      test) run_test=1 ;;
      plot) run_plot=1 ;;
    esac
  done

  cd -- "${REPO_ROOT}"
  mkdir -p -- "${GEN_DIR}" "${SUBSETS_DIR}" "${PLOTS_DIR}"

  cleanup_intermediate
  if ((run_subset)); then
    run_subset_step "${input_path}"
  fi
  if ((run_train)); then
    run_train_step
  fi
  if ((run_test)); then
    run_test_step
  fi
  if ((run_plot)); then
    run_plot_step "${final_plot_name}" "${bin_branches[@]}"
  fi
}

main "$@"
