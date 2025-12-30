shopt -s nullglob
mkdir -p gen

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

for f in ${SCRIPT_DIR}/subsets/train_subset_*.root; do
  base="$(basename "${f%.root}")"
  ${SCRIPT_DIR}/../run2-rdx-l0_hadron_tos.py "$f" "${SCRIPT_DIR}/../../gen/${base}_trained_output.root" --tree "DecayTree" --dump "${SCRIPT_DIR}/../../gen/${base}_xgb.pickle"
done
