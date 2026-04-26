from pathlib import Path
from array import array

import numpy as np
import ROOT


ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)


def _set_efficiency_palette():
    stops = array("d", [0.00, 0.20, 0.45, 0.70, 1.00])
    red = array("d", [0.08, 0.00, 0.10, 0.95, 0.80])
    green = array("d", [0.12, 0.28, 0.72, 0.82, 0.05])
    blue = array("d", [0.35, 0.80, 0.55, 0.08, 0.05])
    ROOT.TColor.CreateGradientColorTable(len(stops), stops, red, green, blue, 255)
    ROOT.gStyle.SetNumberContours(255)


def _finite_points(eff, yerr=None):
    mask = np.isfinite(eff.y)
    x = np.asarray(eff.x[mask], dtype=float)
    ex = np.asarray(eff.xerr[mask], dtype=float)
    y = np.asarray(eff.y[mask], dtype=float)
    if yerr is None:
        ey_low = np.asarray(eff.yerr_low[mask], dtype=float)
        ey_high = np.asarray(eff.yerr_high[mask], dtype=float)
    else:
        err = np.asarray(yerr[mask], dtype=float)
        ey_low = err
        ey_high = err
    return x, ex, y, ey_low, ey_high


def _graph_asymm(eff, color, marker, yerr=None):
    x, ex, y, ey_low, ey_high = _finite_points(eff, yerr)
    graph = ROOT.TGraphAsymmErrors(
        len(x), x, y, ex, ex, ey_low, ey_high
    )
    graph.SetLineColor(color)
    graph.SetMarkerColor(color)
    graph.SetLineWidth(2)
    graph.SetMarkerStyle(marker)
    return graph


def efficiency_plot(path, title, branch, real_eff=None, emu_eff=None, emu_error=None, real_error=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    canvas = ROOT.TCanvas("c_eff", "c_eff", 900, 650)
    canvas.SetGrid()
    canvas.SetRightMargin(0.26)

    axis = ROOT.TH1D(
        "axis",
        f"{title};{branch.name};Efficiency",
        branch.bins,
        branch.x_min,
        branch.x_max,
    )
    axis.SetMinimum(branch.y_min)
    axis.SetMaximum(branch.y_max)
    axis.Draw("AXIS")

    legend = ROOT.TLegend(0.76, 0.72, 0.98, 0.90)
    legend.SetBorderSize(0)

    objects = [axis, legend]
    if real_eff is not None:
        real_graph = _graph_asymm(real_eff, ROOT.kBlack, 20, real_error)
        real_graph.Draw("E1P SAME")
        legend.AddEntry(real_graph, "Real response", "lep")
        objects.append(real_graph)
    if emu_eff is not None:
        emu_graph = _graph_asymm(emu_eff, ROOT.kRed + 1, 21, emu_error)
        emu_graph.Draw("E1P SAME")
        legend.AddEntry(emu_graph, "Emulated", "lep")
        objects.append(emu_graph)

    legend.Draw()
    canvas.SaveAs(str(path))


def ratio_plot(path, branch, x, ratio_values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.isfinite(ratio_values)
    x_values = np.asarray(x[mask], dtype=float)
    y_values = np.asarray(ratio_values[mask], dtype=float)
    xerr = np.zeros(len(x_values), dtype=float)
    yerr = np.zeros(len(y_values), dtype=float)

    canvas = ROOT.TCanvas("c_ratio", "c_ratio", 900, 500)
    canvas.SetGrid()
    axis = ROOT.TH1D(
        "axis_ratio",
        f"Train/test bootstrap mean over central emulated efficiency;{branch.name};Bootstrap mean / central",
        branch.bins,
        branch.x_min,
        branch.x_max,
    )
    axis.SetMinimum(0.8)
    axis.SetMaximum(1.2)
    axis.Draw("AXIS")

    line = ROOT.TLine(branch.x_min, 1.0, branch.x_max, 1.0)
    line.SetLineStyle(2)
    line.Draw("SAME")

    graph = ROOT.TGraphErrors(len(x_values), x_values, y_values, xerr, yerr)
    graph.SetLineColor(ROOT.kRed + 1)
    graph.SetMarkerColor(ROOT.kRed + 1)
    graph.SetMarkerStyle(21)
    graph.SetLineWidth(2)
    graph.Draw("P SAME")
    canvas.SaveAs(str(path))


def _hist2d(name, title, values, x_edges, y_edges):
    hist = ROOT.TH2D(name, title, len(x_edges) - 1, x_edges, len(y_edges) - 1, y_edges)
    for ix in range(values.shape[0]):
        for iy in range(values.shape[1]):
            value = values[ix, iy]
            if np.isfinite(value):
                # ROOT renders exact 0.0 content like an empty cell for COLZ plots.
                # Use a tiny display-only value so true zero-efficiency bins get
                # the low-end palette color; empty bins are masked separately.
                hist.SetBinContent(ix + 1, iy + 1, max(float(value), 1.0e-12))
    hist.SetMinimum(0.0)
    hist.SetMaximum(1.0)
    return hist


def _empty_bin_boxes(denominator, x_edges, y_edges):
    boxes = []
    for ix in range(denominator.shape[0]):
        for iy in range(denominator.shape[1]):
            if denominator[ix, iy] > 0:
                continue
            box = ROOT.TBox(x_edges[ix], y_edges[iy], x_edges[ix + 1], y_edges[iy + 1])
            box.SetFillColor(ROOT.kWhite)
            box.SetLineColor(ROOT.kWhite)
            boxes.append(box)
    return boxes


def _draw_2d(hist, denominator, x_edges, y_edges):
    hist.Draw("COLZ0")
    boxes = _empty_bin_boxes(denominator, x_edges, y_edges)
    for box in boxes:
        box.Draw("SAME")
    return boxes


def efficiency2d_plot(path, x_branch, y_branch, real_eff, emu_eff):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _set_efficiency_palette()

    canvas = ROOT.TCanvas("c_eff2d", "c_eff2d", 1200, 520)
    canvas.Divide(2, 1)

    real = _hist2d(
        "h_real_2d",
        f"Real response;{x_branch.name};{y_branch.name};Efficiency",
        real_eff.values,
        real_eff.x_edges,
        real_eff.y_edges,
    )
    emu = _hist2d(
        "h_emu_2d",
        f"Emulated response;{x_branch.name};{y_branch.name};Efficiency",
        emu_eff.values,
        emu_eff.x_edges,
        emu_eff.y_edges,
    )

    canvas.cd(1)
    ROOT.gPad.SetRightMargin(0.16)
    real_boxes = _draw_2d(real, real_eff.denominator, real_eff.x_edges, real_eff.y_edges)
    canvas.cd(2)
    ROOT.gPad.SetRightMargin(0.16)
    emu_boxes = _draw_2d(emu, emu_eff.denominator, emu_eff.x_edges, emu_eff.y_edges)
    canvas.SaveAs(str(path))
    return real_boxes, emu_boxes
