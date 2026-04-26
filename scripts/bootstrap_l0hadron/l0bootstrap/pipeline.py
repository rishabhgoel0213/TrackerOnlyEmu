from pathlib import Path

from .model import apply_xgb, train_xgb
from .paths import RunPaths
from .plotting import efficiency2d_plot, efficiency_plot, ratio_plot
from .root_io import write_bootstrap_sample, write_initial_split
from .stats import (
    bootstrap_efficiencies,
    bootstrap_errors,
    efficiency,
    efficiency2d,
    ratio,
)


STEPS = (
    "all",
    "prepare",
    "central",
    "xgb-weights",
    "train-test",
    "real-validation",
    "plot",
)


class Pipeline:
    def __init__(self, config, force=False, plot_kinds=None, plot_branches=None):
        self.config = config
        self.force = force
        self.plot_kinds = set(plot_kinds or ())
        self.plot_branches = set(plot_branches or ())
        self.paths = RunPaths(config.tag)
        self.paths.mkdirs()

    def run(self, step):
        print(f"Run directory: {self.paths.root}")
        if step in ("all", "prepare"):
            self.prepare()
        if step in ("all", "central"):
            self.central()
        if step in ("all", "xgb-weights"):
            self.xgb_weight_bootstrap()
        if step in ("all", "train-test"):
            self.train_test_bootstrap()
        if step in ("all", "real-validation"):
            self.real_validation_inputs()
        if step in ("all", "plot"):
            self.plot()

    def prepare(self):
        if self.paths.train.exists() and self.paths.test.exists() and not self.force:
            print("skip existing fixed train/test split")
        else:
            n_train, n_test = write_initial_split(
                self.config.input_path,
                self.config.source_tree,
                self.config.tree,
                self.config.train_fraction,
                self.config.seed,
                self.paths.train,
                self.paths.test,
            )
            print(f"fixed split: {n_train} train entries, {n_test} test entries")

        self._write_bootstrap_train_samples()
        self.real_validation_inputs()

    def central(self):
        self._require(self.paths.train, "fixed train split")
        self._require(self.paths.test, "fixed test split")
        train_xgb(
            self.paths.train,
            self.config.tree,
            self.paths.central_model,
            self.config.ntrees,
            self.config.max_depth,
            self.force,
        )
        apply_xgb(
            self.paths.test,
            self.config.tree,
            self.paths.central_model,
            self.paths.central_output,
            self.config.year,
            self.force,
        )

    def xgb_weight_bootstrap(self):
        self._require(self.paths.train, "fixed train split")
        self._require(self.paths.test, "fixed test split")
        self._write_bootstrap_train_samples()
        for index in self._bootstrap_range():
            train_file = self.paths.bootstrap_train(index)
            model_file = self.paths.xgb_weight_model(index)
            output_file = self.paths.xgb_weight_output(index)
            train_xgb(
                train_file,
                self.config.tree,
                model_file,
                self.config.ntrees,
                self.config.max_depth,
                self.force,
            )
            apply_xgb(
                self.paths.test,
                self.config.tree,
                model_file,
                output_file,
                self.config.year,
                self.force,
            )

    def train_test_bootstrap(self):
        self._require(self.paths.train, "fixed train split")
        self._require(self.paths.test, "fixed test split")
        self._write_bootstrap_train_samples()
        self.real_validation_inputs()
        for index in self._bootstrap_range():
            train_file = self.paths.bootstrap_train(index)
            test_file = self.paths.bootstrap_test(index)
            model_file = self.paths.train_test_model(index)
            output_file = self.paths.train_test_output(index)
            train_xgb(
                train_file,
                self.config.tree,
                model_file,
                self.config.ntrees,
                self.config.max_depth,
                self.force,
            )
            apply_xgb(
                test_file,
                self.config.tree,
                model_file,
                output_file,
                self.config.year,
                self.force,
            )

    def real_validation_inputs(self):
        self._require(self.paths.test, "fixed test split")
        for index in self._bootstrap_range():
            test_file = self.paths.bootstrap_test(index)
            if test_file.exists() and not self.force:
                continue
            write_bootstrap_sample(
                self.paths.test,
                self.config.tree,
                test_file,
                self.config.seed + 20_000 + index,
            )

    def plot(self):
        self._require(self.paths.central_output, "central output")
        xgb_weight_outputs = self._existing(self.paths.xgb_weight_output)
        train_test_outputs = self._existing(self.paths.train_test_output)
        test_bootstraps = self._existing(self.paths.bootstrap_test)

        for branch in self.config.branches:
            if not self._plot_branch_enabled(branch.name):
                continue
            central_real = efficiency(
                self.paths.central_output,
                self.config.tree,
                branch,
                "d0_l0_hadron_tos",
                clopper_pearson=True,
            )
            central_emu = efficiency(
                self.paths.central_output,
                self.config.tree,
                branch,
                "d0_l0_hadron_tos_emu_xgb",
            )

            if self._plot_kind_enabled("xgb-weights") and xgb_weight_outputs:
                _, xgb_weight_error = bootstrap_errors(
                    bootstrap_efficiencies(
                        xgb_weight_outputs,
                        self.config.tree,
                        branch,
                        "d0_l0_hadron_tos_emu_xgb",
                    )
                )
                efficiency_plot(
                    self.paths.plots / f"{branch.name}_xgb_weight_uncertainty.png",
                    "XGB weight bootstrap",
                    branch,
                    central_real,
                    central_emu,
                    emu_error=xgb_weight_error,
                )

            if (self._plot_kind_enabled("train-test") or self._plot_kind_enabled("ratio")) and train_test_outputs:
                train_test_mean, train_test_error = bootstrap_errors(
                    bootstrap_efficiencies(
                        train_test_outputs,
                        self.config.tree,
                        branch,
                        "d0_l0_hadron_tos_emu_xgb",
                    )
                )
                if self._plot_kind_enabled("train-test"):
                    efficiency_plot(
                        self.paths.plots / f"{branch.name}_train_test_uncertainty.png",
                        "Train and test bootstrap",
                        branch,
                        central_real,
                        central_emu,
                        emu_error=train_test_error,
                    )

                if branch.name == "d0_pt" and self._plot_kind_enabled("ratio"):
                    ratio_plot(
                        self.paths.plots / "d0_pt_train_test_bootstrap_ratio.png",
                        branch,
                        central_emu.x,
                        ratio(train_test_mean, central_emu.y),
                    )

            if self._plot_kind_enabled("real-validation") and test_bootstraps:
                _, real_validation_error = bootstrap_errors(
                    bootstrap_efficiencies(
                        test_bootstraps,
                        self.config.tree,
                        branch,
                        "d0_l0_hadron_tos",
                    )
                )
                efficiency_plot(
                    self.paths.plots / f"{branch.name}_real_validation_uncertainty.png",
                    "Real-response validation bootstrap",
                    branch,
                    real_eff=central_real,
                    real_error=real_validation_error,
                )

        if self._plot_kind_enabled("2d"):
            self._plot_2d_central_values()

    def _plot_2d_central_values(self):
        branches = {branch.name: branch for branch in self.config.branches}
        d0_pt = branches.get("d0_pt")
        if d0_pt is None:
            return
        for branch in self.config.branches:
            if branch.name == "d0_pt":
                continue
            if not self._plot_branch_enabled(branch.name):
                continue
            real_eff = efficiency2d(
                self.paths.central_output,
                self.config.tree,
                d0_pt,
                branch,
                "d0_l0_hadron_tos",
                self.config.two_dimensional_bins,
            )
            emu_eff = efficiency2d(
                self.paths.central_output,
                self.config.tree,
                d0_pt,
                branch,
                "d0_l0_hadron_tos_emu_xgb",
                self.config.two_dimensional_bins,
            )
            efficiency2d_plot(
                self.paths.plots / f"d0_pt_vs_{branch.name}_central_efficiency_2d.png",
                d0_pt,
                branch,
                real_eff,
                emu_eff,
            )

    def _plot_kind_enabled(self, name):
        return not self.plot_kinds or name in self.plot_kinds

    def _plot_branch_enabled(self, name):
        return not self.plot_branches or name in self.plot_branches

    def _write_bootstrap_train_samples(self):
        for index in self._bootstrap_range():
            train_file = self.paths.bootstrap_train(index)
            if train_file.exists() and not self.force:
                continue
            write_bootstrap_sample(
                self.paths.train,
                self.config.tree,
                train_file,
                self.config.seed + 10_000 + index,
            )

    def _bootstrap_range(self):
        return range(1, self.config.n_bootstraps + 1)

    def _existing(self, path_func):
        return [path_func(index) for index in self._bootstrap_range() if path_func(index).exists()]

    def _require(self, path, label):
        if not Path(path).exists():
            raise RuntimeError(f"Missing {label}: {path}. Run the needed step first.")
