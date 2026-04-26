from dataclasses import dataclass

import numpy as np
import ROOT

from .root_io import read_branch_array


CL68 = 0.6827


@dataclass(frozen=True)
class Efficiency:
    x: np.ndarray
    xerr: np.ndarray
    y: np.ndarray
    yerr_low: np.ndarray
    yerr_high: np.ndarray
    numerator: np.ndarray
    denominator: np.ndarray


@dataclass(frozen=True)
class Efficiency2D:
    values: np.ndarray
    denominator: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray


def _edges(branch):
    return np.linspace(branch.x_min, branch.x_max, branch.bins + 1)


def _centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def _efficiency_from_arrays(x_values, weights, branch, clopper_pearson=False):
    edges = _edges(branch)
    denominator, _ = np.histogram(x_values, bins=edges)
    numerator, _ = np.histogram(x_values, bins=edges, weights=weights)

    y = np.full(branch.bins, np.nan)
    good = denominator > 0
    y[good] = numerator[good] / denominator[good]

    yerr_low = np.zeros(branch.bins)
    yerr_high = np.zeros(branch.bins)
    if clopper_pearson:
        for index, (passed, total, eff) in enumerate(zip(numerator, denominator, y)):
            if total <= 0 or not np.isfinite(eff):
                continue
            n_total = int(round(total))
            n_passed = min(n_total, max(0, int(round(passed))))
            low = ROOT.TEfficiency.ClopperPearson(n_total, n_passed, CL68, False)
            high = ROOT.TEfficiency.ClopperPearson(n_total, n_passed, CL68, True)
            yerr_low[index] = max(0.0, eff - low)
            yerr_high[index] = max(0.0, high - eff)

    return Efficiency(
        x=_centers(edges),
        xerr=np.zeros(branch.bins),
        y=y,
        yerr_low=yerr_low,
        yerr_high=yerr_high,
        numerator=numerator,
        denominator=denominator,
    )


def efficiency(path, tree_name, branch, value_branch, clopper_pearson=False):
    x_values = read_branch_array(path, tree_name, branch.name)
    weights = read_branch_array(path, tree_name, value_branch)
    return _efficiency_from_arrays(x_values, weights, branch, clopper_pearson)


def bootstrap_efficiencies(paths, tree_name, branch, value_branch):
    values = []
    for path in paths:
        values.append(efficiency(path, tree_name, branch, value_branch).y)
    if not values:
        return np.empty((0, branch.bins))
    return np.vstack(values)


def bootstrap_errors(samples):
    if samples.size == 0:
        return np.array([]), np.array([])
    mean = np.nanmean(samples, axis=0)
    if samples.shape[0] <= 1:
        return mean, np.zeros(samples.shape[1])
    return mean, np.nanstd(samples, axis=0, ddof=1)


def ratio(numerator, denominator):
    out = np.full_like(numerator, np.nan, dtype=float)
    good = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    out[good] = numerator[good] / denominator[good]
    return out


def efficiency2d(path, tree_name, x_branch, y_branch, value_branch, bins):
    x_values = read_branch_array(path, tree_name, x_branch.name)
    y_values = read_branch_array(path, tree_name, y_branch.name)
    weights = read_branch_array(path, tree_name, value_branch)
    x_edges = np.linspace(x_branch.x_min, x_branch.x_max, bins + 1)
    y_edges = np.linspace(y_branch.x_min, y_branch.x_max, bins + 1)

    numerator, _, _ = np.histogram2d(x_values, y_values, bins=(x_edges, y_edges), weights=weights)
    denominator, _, _ = np.histogram2d(x_values, y_values, bins=(x_edges, y_edges))
    eff = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
    return Efficiency2D(
        values=eff,
        denominator=denominator,
        x_edges=x_edges,
        y_edges=y_edges,
    )
