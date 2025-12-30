#include "TFile.h"
#include "TTree.h"
#include "TDirectory.h"
#include "TMemFile.h"
#include <experimental/filesystem>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <system_error>

namespace fs = std::experimental::filesystem;

namespace
{
struct ReducedTreeHandle 
{
    std::unique_ptr<TMemFile> memFile;
    TTree* tree = nullptr;
};

ReducedTreeHandle getReducedTree(const std::string& fileName)
{
    ReducedTreeHandle out;
    std::shared_ptr<TFile> file(TFile::Open(fileName.c_str(), "READ"));
    TDirectoryFile* dir = (TDirectoryFile*)file->Get("TupleB0");
    auto tree = dir->Get<TTree>("DecayTree");

    //Keep all branches that exist in the sample tree
    std::shared_ptr<TFile> branchesFile(TFile::Open("../../samples/run2-rdx-train_xgb.root", "READ"));
    TDirectoryFile* branchesDir = (TDirectoryFile*)branchesFile->Get("TupleB0");
    auto branchesTree = branchesDir->Get<TTree>("DecayTree");
    TObjArray* branches = branchesTree->GetListOfBranches();

    tree->SetBranchStatus("*", 0);
    for(int i = 0; i < branches->GetEntries(); i++)
    {
        auto* branch = static_cast<TBranch*>(branches->At(i));
        const char* name = branch->GetName();
        if(tree->GetBranch(name)) tree->SetBranchStatus(name, 1);
    }

    //Additional necessary branches
    if(tree->GetBranch("FitVar_q2")) tree->SetBranchStatus("FitVar_q2", 1);
    if(tree->GetBranch("FitVar_Mmiss2")) tree->SetBranchStatus("FitVar_Mmiss2", 1);
    if(tree->GetBranch("FitVar_El")) tree->SetBranchStatus("FitVar_El", 1);


    out.memFile = std::make_unique<TMemFile>("reducedTree.root", "RECREATE");
    TDirectory::TContext inMemory(out.memFile.get());
    out.tree = tree->CloneTree(-1, "fast");
    return out;
}
} //namespace

int main(int argc, char** argv)
{
    std::mt19937 rng(12345);

    std::error_code ec;
    fs::create_directories(fs::path("subsets"), ec);
    if(ec) return 1;

    //Inputs
    const std::string inputPath = (argc > 1) ? argv[1] : "/home/rishabh/lhcb-ntuples-gen/ntuples/0.9.4-trigger_emulation/Dst_D0-mc/Dst_D0--21_04_21--mc--MC_2016_Beam6500GeV-2016-MagDown-Nu1.6-25ns-Pythia8_Sim09j_Trig0x6139160F_Reco16_Turbo03a_Filtered_11574021_D0TAUNU.SAFESTRIPTRIG.DST.root";

    //Hyperparameters
    constexpr int numSubsets = 2;
    constexpr double trainFrac = 0.5;

    auto reduced = ::getReducedTree(inputPath);
    if(!reduced.tree) return 1;
    TTree* reducedTree = reduced.tree;

    long numEntries = reducedTree->GetEntries();
    long subsetSize = (numEntries + numSubsets - 1) / numSubsets;

    std::vector<std::unique_ptr<TFile>> trainFiles;
    std::vector<std::unique_ptr<TFile>> testFiles;
    std::vector<TTree*> trainSubsets;
    std::vector<TTree*> testSubsets;

    trainFiles.reserve(numSubsets);
    testFiles.reserve(numSubsets);
    trainSubsets.reserve(numSubsets);
    testSubsets.reserve(numSubsets);

    std::vector<std::vector<long>> trainIndexes;
    trainIndexes.resize(numSubsets);
    for(int k = 0; k < numSubsets; k++)
    {
        trainIndexes[k].reserve(subsetSize);
    }
    for(long i = 0; i < numEntries; i++)
    {
        long k = i % numSubsets;
        trainIndexes[k].push_back(i);
    }
    for(int k = 0; k < numSubsets; k++)
    {
        std::shuffle(trainIndexes[k].begin(), trainIndexes[k].end(), rng);
        long numTrain = long(trainIndexes[k].size() * trainFrac);
        trainIndexes[k] = std::vector<long>(trainIndexes[k].begin(), trainIndexes[k].begin() + numTrain);
    }
	
    for(int i = 0; i < numSubsets; i++)
    {
        const auto trainPath = (fs::path("subsets") / ("train_subset_" + std::to_string(i + 1) + ".root")).string();
        trainFiles.emplace_back(TFile::Open(trainPath.c_str(), "RECREATE"));
        trainFiles.back()->cd();
        TTree* out = reducedTree->CloneTree(0);
        out->SetName("DecayTree");
        trainSubsets.push_back(out);

        const auto testPath = (fs::path("subsets") / ("test_subset_" + std::to_string(i + 1) + ".root")).string();
        testFiles.emplace_back(TFile::Open(testPath.c_str(), "RECREATE"));
        testFiles.back()->cd();
        out = reducedTree->CloneTree(0);
        out->SetName("DecayTree");
        testSubsets.push_back(out);
    }

    for(long i = 0; i < numEntries; i++)
    {
        reducedTree->GetEntry(i);
        long k = i % numSubsets;
        if(std::find(trainIndexes[k].begin(), trainIndexes[k].end(), i) != trainIndexes[k].end())
        {
            trainSubsets[k]->Fill();
        }
        else
        {
            testSubsets[k]->Fill();
        }
    }

    for(int i = 0; i < numSubsets; i++)
    {
        if(trainFiles[i] && trainSubsets[i])
        {
            trainFiles[i]->cd();
            trainSubsets[i]->Write();
            trainFiles[i]->Close();
        }
        if(testFiles[i] && testSubsets[i])
        {
            testFiles[i]->cd();
            testSubsets[i]->Write();
            testFiles[i]->Close();
        }
    }
    return 0;
}
