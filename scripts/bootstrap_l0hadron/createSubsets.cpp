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
#include <iostream>
#include <numeric>

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
    std::string subsetDir = "subsets";

    std::error_code ec;
    fs::create_directories(subsetDir, ec);
    if(ec) return 1;

    //Inputs
    const std::string inputPath = (argc > 1) ? argv[1] : "/home/rishabh/lhcb-ntuples-gen/ntuples/0.9.4-trigger_emulation/Dst_D0-mc/Dst_D0--21_04_21--mc--MC_2016_Beam6500GeV-2016-MagDown-Nu1.6-25ns-Pythia8_Sim09j_Trig0x6139160F_Reco16_Turbo03a_Filtered_11574021_D0TAUNU.SAFESTRIPTRIG.DST.root";

    //Hyperparameters
    constexpr int numSubsets = 50;
    constexpr double trainFrac = 0.5;
    constexpr bool resample = true;

    auto reduced = ::getReducedTree(inputPath);
    if(!reduced.tree) return 1;
    TTree* reducedTree = reduced.tree;

    long numEntries = reducedTree->GetEntries();
    long subsetSize = resample ? numEntries : (numEntries + numSubsets - 1) / numSubsets;

    std::vector<std::unique_ptr<TFile>> trainFiles;
    std::vector<std::unique_ptr<TFile>> testFiles;
    std::vector<TTree*> trainSubsets;
    std::vector<TTree*> testSubsets;

    trainFiles.reserve(numSubsets);
    testFiles.reserve(numSubsets);
    trainSubsets.reserve(numSubsets);
    testSubsets.reserve(numSubsets);

    std::vector<long> shuffledIndexes(numEntries);
    std::vector<std::vector<long>> indexes(numSubsets);

    std::iota(shuffledIndexes.begin(), shuffledIndexes.end(), 0);

    std::shuffle(shuffledIndexes.begin(), shuffledIndexes.end(), rng);
    std::uniform_int_distribution<long> dist(1, subsetSize - 1);
    for(int k = 0; k < numSubsets; k++)
    {
        if(resample)
        {
            for(int i = 0; i < subsetSize; i++)
            {
                indexes[k].push_back(dist(rng));
            }
        }
        else
        {
            const long start = subsetSize * k;
            if(start >= numEntries)
            {
                indexes[k].clear();
                continue;
            }
            const long end = std::min(numEntries, start + subsetSize);
            indexes[k] = std::vector<long>(shuffledIndexes.begin() + start, shuffledIndexes.begin() + end);
        }
        std::sort(indexes[k].begin(), indexes[k].end()); //For performance; reading from a TTree is slow out of order
    }

    std::cout << "Generated subsets. Now creating trees" << std::endl;
	
    for(int i = 0; i < numSubsets; i++)
    {
        const auto trainPath = (fs::path(subsetDir) / ("train_subset_" + std::to_string(i + 1) + ".root")).string();
        trainFiles.emplace_back(TFile::Open(trainPath.c_str(), "RECREATE"));
        trainFiles.back()->cd();
        TTree* out = reducedTree->CloneTree(0);
        out->SetName("DecayTree");
        trainSubsets.push_back(out);

        const auto testPath = (fs::path(subsetDir) / ("test_subset_" + std::to_string(i + 1) + ".root")).string();
        testFiles.emplace_back(TFile::Open(testPath.c_str(), "RECREATE"));
        testFiles.back()->cd();
        out = reducedTree->CloneTree(0);
        out->SetName("DecayTree");
        testSubsets.push_back(out);
    }

    for(int i = 0; i < numSubsets; i++)
    {
        for(long j = 0; j < indexes[i].size(); j++)
        {
            reducedTree->GetEntry(indexes[i][j]);
            if(j < indexes[i].size() * trainFrac) trainSubsets[i]->Fill();
            else testSubsets[i]->Fill();
        }
        std::cout << "Tree #" << (i + 1) << " finished" << std::endl;
    }

    std::cout << "All trees finished. Now writing to memory" << std::endl;

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

    std::cout << "All trees created and written to memory" << std::endl;
    return 0;
}
