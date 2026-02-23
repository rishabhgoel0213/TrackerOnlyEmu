#!/usr/bin/env bash
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
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    nix-shell -p root gcc python3 python3Packages.numpy python3Packages.scikit-learn python3Packages.xgboost \
    --run "${cmd}"
}

cleanup_intermediate() {
  local -a files=(
    "${SUBSETS_DIR}"/train_subset_*.root
    "${SUBSETS_DIR}"/test_subset_*.root
    "${GEN_DIR}"/train_subset_*_trained_output.root
    "${GEN_DIR}"/train_subset_*_xgb.pickle
    "${PLOTS_DIR}"/test_subset_*_eff.png
  )

  if ((${#files[@]})); then
    rm -f -- "${files[@]}"
  fi
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -i, --input PATH           Input ROOT file for subset creation.
  -p, --plot-name NAME       Base name for output plot files.
  -b, --branches LIST...     Space-separated branch list (can be repeated).
  -h, --help                 Show this help message.

Examples:
  $(basename "$0")
  $(basename "$0") -i /path/to/input.root -p efficiency_plot_2018.png
  $(basename "$0") -b d0_pt k_pt -b pi_pt
EOF
}

main() {
  local input_path=""
  local final_plot_name="${DEFAULT_FINAL_PLOT_NAME}"
  local -a bin_branches=()
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

  cd -- "${REPO_ROOT}"
  mkdir -p -- "${GEN_DIR}" "${SUBSETS_DIR}" "${PLOTS_DIR}"

  cleanup_intermediate
  rm -f -- "${GEN_DIR}"/test_subset_*_output.root "${PLOTS_DIR}/${DEFAULT_FINAL_PLOT_NAME}" "${PLOTS_DIR}/${final_plot_name}"
  for branch in "${bin_branches[@]}"; do
    rm -f -- "${PLOTS_DIR}/${branch}_${final_plot_name}"
  done

  build_root_cpp "${CREATE_SUBSETS_SRC}" "${CREATE_SUBSETS_BIN}"
  if [[ -n "${input_path}" ]]; then
    "${CREATE_SUBSETS_BIN}" "${input_path}"
  else
    "${CREATE_SUBSETS_BIN}"
  fi

  run_in_python_root_shell "bash -e \"${TRAIN_SCRIPT}\""
  run_in_python_root_shell "bash -e \"${TEST_SCRIPT}\""

  build_root_cpp "${GENERATE_PLOTS_SRC}" "${GENERATE_PLOTS_BIN}"
  "${GENERATE_PLOTS_BIN}" "${final_plot_name}" "${bin_branches[@]}"

  cleanup_intermediate

  if ! compgen -G "${GEN_DIR}/test_subset_*_output.root" > /dev/null; then
    echo "No final test output ROOT files were produced in ${GEN_DIR}" >&2
    exit 1
  fi

  for branch in "${bin_branches[@]}"; do
    if [[ ! -f "${PLOTS_DIR}/${branch}_${final_plot_name}" ]]; then
      echo "Final plot was not produced at ${PLOTS_DIR}/${branch}_${final_plot_name}" >&2
      exit 1
    fi
  done
}

main "$@"
