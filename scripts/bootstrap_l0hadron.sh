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

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -i, --input PATH           Input ROOT file for subset creation.
  -p, --plot-name NAME       Base name for output plot files.
  -b, --branches LIST...     Space-separated branch list (can be repeated).
  -s, --skip STEP...         Skip pipeline steps: subsets train test plots.
  -h, --help                 Show this help message.

Examples:
  $(basename "$0")
  $(basename "$0") -i /path/to/input.root -p efficiency_plot_2018.png
  $(basename "$0") -b d0_pt k_pt -b pi_pt
  $(basename "$0") -s subsets train
EOF
}

main() {
  local input_path=""
  local final_plot_name="${DEFAULT_FINAL_PLOT_NAME}"
  local -a bin_branches=()
  local skip_subsets=0
  local skip_train=0
  local skip_test=0
  local skip_plots=0
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
      -s|--skip)
        if (($# < 2)); then
          echo "Missing value for $1" >&2
          usage >&2
          exit 1
        fi
        shift
        while (($#)) && [[ "$1" != -* ]]; do
          case "$1" in
            subsets) skip_subsets=1 ;;
            train) skip_train=1 ;;
            test) skip_test=1 ;;
            plots) skip_plots=1 ;;
            *)
              echo "Unknown skip step: $1" >&2
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

  local run_subsets=1
  local run_train=1
  local run_test=1
  local run_plots=1
  ((skip_subsets)) && run_subsets=0
  ((skip_train)) && run_train=0
  ((skip_test)) && run_test=0
  ((skip_plots)) && run_plots=0

  cd -- "${REPO_ROOT}"
  mkdir -p -- "${GEN_DIR}" "${SUBSETS_DIR}" "${PLOTS_DIR}"

  cleanup_intermediate

  if ((run_subsets)); then
    rm -f -- "${SUBSETS_DIR}"/train_subset_*.root "${SUBSETS_DIR}"/test_subset_*.root
    rm -f -- "${GEN_DIR}"/train_subset_*_trained_output.root "${GEN_DIR}"/train_subset_*_xgb.pickle
    rm -f -- "${GEN_DIR}"/test_subset_*_output.root
    rm -f -- "${PLOTS_DIR}/${DEFAULT_FINAL_PLOT_NAME}" "${PLOTS_DIR}/${final_plot_name}"
    for branch in "${bin_branches[@]}"; do
      rm -f -- "${PLOTS_DIR}/${branch}_${final_plot_name}"
    done

    build_root_cpp "${CREATE_SUBSETS_SRC}" "${CREATE_SUBSETS_BIN}"
    if [[ -n "${input_path}" ]]; then
      "${CREATE_SUBSETS_BIN}" "${input_path}"
    else
      "${CREATE_SUBSETS_BIN}"
    fi
  fi

  if ((run_train)); then
    rm -f -- "${GEN_DIR}"/train_subset_*_trained_output.root "${GEN_DIR}"/train_subset_*_xgb.pickle
    run_in_python_root_shell "bash -e \"${TRAIN_SCRIPT}\""
  fi

  if ((run_test)); then
    rm -f -- "${GEN_DIR}"/test_subset_*_output.root
    run_in_python_root_shell "bash -e \"${TEST_SCRIPT}\""
  fi

  if ((run_plots)); then
    rm -f -- "${PLOTS_DIR}/${DEFAULT_FINAL_PLOT_NAME}" "${PLOTS_DIR}/${final_plot_name}"
    for branch in "${bin_branches[@]}"; do
      rm -f -- "${PLOTS_DIR}/${branch}_${final_plot_name}"
    done
    build_root_cpp "${GENERATE_PLOTS_SRC}" "${GENERATE_PLOTS_BIN}"
    "${GENERATE_PLOTS_BIN}" "${final_plot_name}" "${bin_branches[@]}"
  fi

  if ((run_test || run_plots)); then
    if ! compgen -G "${GEN_DIR}/test_subset_*_output.root" > /dev/null; then
      echo "No final test output ROOT files were produced in ${GEN_DIR}" >&2
      exit 1
    fi
  fi

  if ((run_plots)); then
    for branch in "${bin_branches[@]}"; do
      if [[ ! -f "${PLOTS_DIR}/${branch}_${final_plot_name}" ]]; then
        echo "Final plot was not produced at ${PLOTS_DIR}/${branch}_${final_plot_name}" >&2
        exit 1
      fi
    done
  fi
}

main "$@"
