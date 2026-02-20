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

build_root_cpp() {
  local src="$1"
  local out="$2"
  local compile_cmd

  compile_cmd="g++ -fdiagnostics-color=always -g \$(root-config --cflags) \"${src}\" -o \"${out}\" \$(root-config --libs) -lstdc++fs"
  nix-shell -p root gcc --run "${compile_cmd}"
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

main() {
  local input_path="${1:-}"
  local final_plot_name="${2:-${DEFAULT_FINAL_PLOT_NAME}}"
  local default_final_plot="${PLOTS_DIR}/${DEFAULT_FINAL_PLOT_NAME}"
  local final_plot="${PLOTS_DIR}/${final_plot_name}"

  cd -- "${REPO_ROOT}"
  mkdir -p -- "${GEN_DIR}" "${SUBSETS_DIR}" "${PLOTS_DIR}"

  cleanup_intermediate
  rm -f -- "${GEN_DIR}"/test_subset_*_output.root "${default_final_plot}" "${final_plot}"

  build_root_cpp "${CREATE_SUBSETS_SRC}" "${CREATE_SUBSETS_BIN}"
  if [[ -n "${input_path}" ]]; then
    "${CREATE_SUBSETS_BIN}" "${input_path}"
  else
    "${CREATE_SUBSETS_BIN}"
  fi

  run_in_python_root_shell "bash -e \"${TRAIN_SCRIPT}\""
  run_in_python_root_shell "bash -e \"${TEST_SCRIPT}\""

  build_root_cpp "${GENERATE_PLOTS_SRC}" "${GENERATE_PLOTS_BIN}"
  "${GENERATE_PLOTS_BIN}" "${final_plot_name}"

  cleanup_intermediate

  if ! compgen -G "${GEN_DIR}/test_subset_*_output.root" > /dev/null; then
    echo "No final test output ROOT files were produced in ${GEN_DIR}" >&2
    exit 1
  fi

  if [[ ! -f "${final_plot}" ]]; then
    echo "Final plot was not produced at ${final_plot}" >&2
    exit 1
  fi
}

main "$@"
