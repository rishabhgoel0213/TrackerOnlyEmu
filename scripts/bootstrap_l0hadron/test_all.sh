shopt -s nullglob
mkdir -p gen

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

for f in ${SCRIPT_DIR}/subsets/test_subset_*.root; do
  base="$(basename "${f%.root}")"
  train_base="train_subset_${base#test_subset_}"
  ${SCRIPT_DIR}/../run2-rdx-l0_hadron_tos.py "$f" "${SCRIPT_DIR}/../../gen/${base}_output.root" --tree "DecayTree" --load "${SCRIPT_DIR}/../../gen/${train_base}_xgb.pickle" --debug
done
