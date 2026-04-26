from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = SCRIPT_DIR.parent
BOOTSTRAP_DIR = SCRIPT_DIR / "bootstrap_l0hadron"
GEN_DIR = REPO_ROOT / "gen"
RUN_SCRIPT = SCRIPT_DIR / "run2-rdx-l0_hadron_tos.py"


class RunPaths:
    def __init__(self, tag):
        self.root = GEN_DIR / tag
        self.data = self.root / "data"
        self.bootstrap_inputs = self.root / "bootstrap_inputs"
        self.models = self.root / "models"
        self.outputs = self.root / "outputs"
        self.plots = self.root / "plots"

        self.train = self.data / "train.root"
        self.test = self.data / "test.root"
        self.central_model = self.models / "central_xgb.pickle"
        self.central_output = self.outputs / "central_test.root"

        self.xgb_weight_models = self.models / "xgb_weight_bootstrap"
        self.train_test_models = self.models / "train_test_bootstrap"
        self.xgb_weight_outputs = self.outputs / "xgb_weight_bootstrap"
        self.train_test_outputs = self.outputs / "train_test_bootstrap"

    def mkdirs(self):
        for path in (
            self.data,
            self.bootstrap_inputs,
            self.models,
            self.outputs,
            self.plots,
            self.xgb_weight_models,
            self.train_test_models,
            self.xgb_weight_outputs,
            self.train_test_outputs,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def bootstrap_train(self, index):
        return self.bootstrap_inputs / f"train_bootstrap_{index:04d}.root"

    def bootstrap_test(self, index):
        return self.bootstrap_inputs / f"test_bootstrap_{index:04d}.root"

    def xgb_weight_model(self, index):
        return self.xgb_weight_models / f"xgb_weight_{index:04d}.pickle"

    def xgb_weight_output(self, index):
        return self.xgb_weight_outputs / f"test_xgb_weight_{index:04d}.root"

    def train_test_model(self, index):
        return self.train_test_models / f"train_test_{index:04d}.pickle"

    def train_test_output(self, index):
        return self.train_test_outputs / f"test_train_test_{index:04d}.root"
