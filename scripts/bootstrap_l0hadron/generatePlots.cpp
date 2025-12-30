#include "TCanvas.h"
#include "TFile.h"
#include "TH1D.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TLegend.h"
#include "TGraphErrors.h"
#include <experimental/filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <system_error>

namespace fs = std::experimental::filesystem;

namespace
{
static int subsetNumberFromFilename(const fs::path& path)
{
    if(path.extension() != ".root") return -1;
    const auto stem = path.stem().string();
    const std::string prefix = "test_subset_";
    const std::string suffix = "_output";
    if(stem.size() <= prefix.size() + suffix.size()) return -1;
    if(stem.compare(0, prefix.size(), prefix) != 0) return -1;
    if(stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) != 0) return -1;

    const auto start = prefix.size();
    const auto count = stem.size() - prefix.size() - suffix.size();
    if(count == 0) return -1;

    int value = 0;
    for(size_t i = 0; i < count; i++)
    {
        const char c = stem[start + i];
        if(c < '0' || c > '9') return -1;
        value = value * 10 + (c - '0');
    }
    return value;
}

struct EffHists
{
    TH1D denom;
    TH1D realNum;
    TH1D emuNum;
};

struct RunningStats
{
    int n = 0;
    double mean = 0.0;
    double m2 = 0.0;

    void add(double x)
    {
        n++;
        const double delta = x - mean;
        mean += delta / n;
        const double delta2 = x - mean;
        m2 += delta * delta2;
    }

    double variance() const
    {
        return (n > 1) ? (m2 / (n - 1)) : 0.0;
    }

    double sd() const
    {
        return std::sqrt(variance());
    }
};

EffHists fillHists(TTree* tree, int numBins, double xMin, double xMax)
{
    TH1::AddDirectory(kFALSE);
    EffHists h{
        TH1D("hDenom", "", numBins, xMin, xMax),
        TH1D("hRealNum", "", numBins, xMin, xMax),
        TH1D("hEmuNum", "", numBins, xMin, xMax),
    };

    TTreeReader reader(tree);
    TTreeReaderValue<Double_t> d0_pt(reader, "d0_pt");
    TTreeReaderValue<Double_t> d0_l0_hadron_tos(reader, "d0_l0_hadron_tos");
    TTreeReaderValue<Float_t> d0_l0_hadron_tos_emu_xgb(reader, "d0_l0_hadron_tos_emu_xgb");

    while(reader.Next())
    {
        h.denom.Fill(*d0_pt);
        h.realNum.Fill(*d0_pt, *d0_l0_hadron_tos);
        h.emuNum.Fill(*d0_pt, *d0_l0_hadron_tos_emu_xgb);
    }

    return h;
}

std::unique_ptr<TH1D> efficiencyHist(const TH1D& num, const TH1D& denom, const char* name)
{
    std::unique_ptr<TH1D> eff(static_cast<TH1D*>(num.Clone(name)));
    eff->Reset("ICES");

    for(int bin = 1; bin <= denom.GetNbinsX(); bin++)
    {
        const double total = denom.GetBinContent(bin);
        if(total <= 0.0) continue;
        eff->SetBinContent(bin, num.GetBinContent(bin) / total);
    }

    return eff;
}

std::vector<std::string> listTestOutputFiles(const std::string& genDir)
{
    std::vector<std::string> files;
    std::error_code ec;
    for(fs::directory_iterator it(genDir, ec), end; it != end && !ec; it.increment(ec))
    {
        if(subsetNumberFromFilename(it->path()) < 0) continue;
        files.push_back(it->path().string());
    }

    std::sort(files.begin(), files.end(), [](const std::string& a, const std::string& b) {
        const int na = subsetNumberFromFilename(fs::path(a));
        const int nb = subsetNumberFromFilename(fs::path(b));
        if(na != -1 && nb != -1 && na != nb) return na < nb;
        return a < b;
    });
    return files;
}

void savePerFilePlot(const std::string& outPath, TH1D& realEff, TH1D& emuEff)
{
    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);

    realEff.SetTitle("L0Hadron TOS efficiency;d0_{pT} [GeV];Efficiency");
    realEff.SetMinimum(0.0);
    realEff.SetMaximum(1.05);
    realEff.SetLineColor(kBlack);
    realEff.SetLineWidth(2);

    emuEff.SetLineColor(kRed + 1);
    emuEff.SetLineWidth(2);

    TCanvas c("c", "c", 900, 600);
    c.SetGrid();
    realEff.Draw("HIST");
    emuEff.Draw("HIST SAME");

    TLegend leg(0.55, 0.15, 0.88, 0.3);
    leg.SetBorderSize(0);
    leg.AddEntry(&realEff, "Real response", "l");
    leg.AddEntry(&emuEff, "Emulated", "l");
    leg.Draw();

    c.SaveAs(outPath.c_str());
}
} // namespace

int main()
{
    const std::string genDir = "../../gen";
    const std::string plotDir = "plots";

    //Histogram Parameters
    constexpr int numBins = 20;
    constexpr double xMin = 0.0;
    constexpr double xMax = 20.0;

    std::error_code ec;
    fs::create_directories(plotDir, ec);
    if(ec) return 1;

    const auto files = listTestOutputFiles(genDir);
    if(files.empty()) return 1;

    std::vector<RunningStats> realStats(numBins);
    std::vector<RunningStats> emuStats(numBins);
    for(const auto& path : files)
    {
        std::unique_ptr<TFile> file(TFile::Open(path.c_str(), "READ"));
        auto* tree = file->Get<TTree>("DecayTree");

        auto h = fillHists(tree, numBins, xMin, xMax);
        auto realEff = efficiencyHist(h.realNum, h.denom, "hRealEff");
        auto emuEff = efficiencyHist(h.emuNum, h.denom, "hEmuEff");

        const auto stem = fs::path(path).stem().string();
        savePerFilePlot(plotDir + "/" + stem + "_eff.png", *realEff, *emuEff);

        for(int bin = 1; bin <= numBins; bin++)
        {
            const double total = h.denom.GetBinContent(bin);
            if(total <= 0.0) continue;

            realStats[bin - 1].add(realEff->GetBinContent(bin));
            emuStats[bin - 1].add(emuEff->GetBinContent(bin));
        }
    }

    TH1D axis("axis", "L0Hadron TOS efficiency;d0_{pT} [GeV];Efficiency", numBins, xMin, xMax);

    std::vector<double> x;
    std::vector<double> ex;
    std::vector<double> yReal;
    std::vector<double> eReal;
    std::vector<double> yEmu;
    std::vector<double> eEmu;
    x.reserve(numBins);
    ex.reserve(numBins);
    yReal.reserve(numBins);
    eReal.reserve(numBins);
    yEmu.reserve(numBins);
    eEmu.reserve(numBins);

    for(int bin = 1; bin <= numBins; bin++)
    {
        const double low = axis.GetBinLowEdge(bin);
        const double high = low + axis.GetBinWidth(bin);
        const auto& rs = realStats[bin - 1];
        const auto& es = emuStats[bin - 1];

        if(rs.n == 0 && es.n == 0) continue;
        x.push_back(axis.GetBinCenter(bin));
        ex.push_back(0.0);
        yReal.push_back(rs.mean);
        eReal.push_back(rs.sd());
        yEmu.push_back(es.mean);
        eEmu.push_back(es.sd());
    }

    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);

    TCanvas c("c", "c", 900, 600);
    c.SetGrid();
    axis.SetMinimum(0.0);
    axis.SetMaximum(1.05);
    axis.Draw("AXIS");

    auto* grReal = new TGraphErrors(static_cast<int>(x.size()), x.data(), yReal.data(), ex.data(), eReal.data());
    auto* grEmu = new TGraphErrors(static_cast<int>(x.size()), x.data(), yEmu.data(), ex.data(), eEmu.data());
    grReal->SetLineColor(kBlack);
    grReal->SetLineWidth(2);
    grReal->SetMarkerStyle(20);
    grReal->SetMarkerColor(kBlack);
    grEmu->SetLineColor(kRed + 1);
    grEmu->SetLineWidth(2);
    grEmu->SetMarkerStyle(21);
    grEmu->SetMarkerColor(kRed + 1);
    grReal->Draw("E1P SAME");
    grEmu->Draw("E1P SAME");

    TLegend leg(0.55, 0.15, 0.88, 0.33);
    leg.SetBorderSize(0);
    leg.AddEntry(grReal, "Real response", "lep");
    leg.AddEntry(grEmu, "Emulated", "lep");
    leg.Draw();

    c.SaveAs((plotDir + "/efficiency_plot_combined.png").c_str());

    return 0;
}
