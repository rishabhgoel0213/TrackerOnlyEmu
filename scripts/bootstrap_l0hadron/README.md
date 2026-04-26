# L0Hadron bootstrap pipeline

This directory contains the support code for `../bootstrap_l0hadron.py`.

The pipeline now follows the ordering from the 2026-04-23 meeting notes:

1. Build one fixed reduced `train.root` and `test.root` split from the input sample.
2. Train one central XGB model on the full fixed train sample and apply it to the fixed test sample.
3. Bootstrap the train sample only, train one XGB per resample, and apply each model to the fixed test sample. These plots estimate the uncertainty from XGB weights.
4. Bootstrap both train and test samples, train/apply each pair, and plot the combined train+test statistical uncertainty.
5. Bootstrap the fixed test sample without XGB and plot the real-response validation uncertainty.
6. Produce central-value two-dimensional efficiency plots in `d0_pt` versus each other configured variable, plus a `d0_pt` ratio plot comparing the train+test bootstrap mean to the central emulated efficiency.

Typical use:

```sh
./scripts/bootstrap_l0hadron.py scripts/bootstrap_l0hadron/configs/2016_cfg.yml
```

Individual resumable steps are available with `--step prepare`, `--step central`, `--step xgb-weights`, `--step train-test`, `--step real-validation`, and `--step plot`. Existing outputs are reused unless `--force` is passed.

Plot regeneration can be narrowed with `--plot-kind` and `--plot-branch`.
For example, to regenerate only the `d0_pt` versus `nspdhits` color plot:

```sh
./scripts/bootstrap_l0hadron.py scripts/bootstrap_l0hadron/configs/2016_cfg.yml --step plot --plot-kind 2d --plot-branch nspdhits
```
