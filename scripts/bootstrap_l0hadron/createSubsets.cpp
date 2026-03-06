#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TError.h"
#include <experimental/filesystem>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <system_error>
#include <iostream>
#include <numeric>
#include <cstring>
#include <cstdlib>

namespace fs = std::experimental::filesystem;

namespace
{
struct ReducedTreeHandle 
{
    std::unique_ptr<TChain> chain;
    TTree* tree = nullptr;
};

std::vector<std::string> collectInputFiles(const std::string& inputPath)
{
    std::vector<std::string> fileNames;
    std::error_code ec;
    fs::path path(inputPath);
    if(fs::is_directory(path, ec))
    {
        fs::directory_iterator it(path, ec);
        fs::directory_iterator end;
        for(; it != end && !ec; it.increment(ec))
        {
            std::error_code fileEc;
            if(!fs::is_regular_file(it->path(), fileEc) || fileEc) continue;
            if(it->path().extension() != ".root") continue;
            const std::string fileName = it->path().filename().string();
            if(fileName.find("aux") != std::string::npos) continue;
            fileNames.push_back(it->path().string());
        }
        std::sort(fileNames.begin(), fileNames.end());
    }
    else if(!ec)
    {
        fileNames.push_back(inputPath);
    }
    return fileNames;
}

ReducedTreeHandle getReducedTree(const std::string& inputPath, const std::string& branchesPath)
{
    ReducedTreeHandle out;
    auto inputFiles = collectInputFiles(inputPath);

    out.chain = std::make_unique<TChain>("TupleB0/DecayTree");
    for(const auto& fileName : inputFiles) out.chain->Add(fileName.c_str());
    if(out.chain->GetEntries() == 0) return out;

    //Keep all branches that exist in the sample tree
    std::shared_ptr<TFile> branchesFile(TFile::Open(branchesPath.c_str(), "READ"));
    TDirectoryFile* branchesDir = (TDirectoryFile*)branchesFile->Get("TupleB0");
    auto branchesTree = branchesDir->Get<TTree>("DecayTree");
    TObjArray* branches = branchesTree->GetListOfBranches();

    out.chain->SetBranchStatus("*", 0);
    for(int i = 0; i < branches->GetEntries(); i++)
    {
        auto* branch = static_cast<TBranch*>(branches->At(i));
        const char* name = branch->GetName();
        if(out.chain->GetBranch(name)) out.chain->SetBranchStatus(name, 1);
    }

    //Additional necessary branches
    if(out.chain->GetBranch("FitVar_q2")) out.chain->SetBranchStatus("FitVar_q2", 1);
    if(out.chain->GetBranch("FitVar_Mmiss2")) out.chain->SetBranchStatus("FitVar_Mmiss2", 1);
    if(out.chain->GetBranch("FitVar_El")) out.chain->SetBranchStatus("FitVar_El", 1);


    out.tree = out.chain.get();
    return out;
}
} //namespace

int main(int argc, char** argv)
{
    std::mt19937 rng(12345);
    const fs::path scriptDir = fs::absolute(fs::path(argv[0])).parent_path();
    const fs::path repoRoot = scriptDir / ".." / "..";
    const std::string defaultSubsetDir = (scriptDir / "subsets").string();
    const std::string branchesPath = (repoRoot / "samples" / "run2-rdx-train_xgb.root").string();
    const std::string defaultInputPath = (repoRoot / "samples" / "run2-rdx-sample.root").string();

    //Inputs / Hyperparameters
    const std::string inputPath = (argc > 2) ? argv[1] : defaultInputPath;
    const int numSubsets = (argc > 2) ? std::atoi(argv[2]) : ((argc == 2) ? std::atoi(argv[1]) : 100);
    const double trainFrac = (argc > 3) ? std::atof(argv[3]) : 0.5;
    const std::string subsetDir = (argc > 4) ? argv[4] : defaultSubsetDir;
    constexpr bool resample = true;

    std::error_code ec;
    fs::create_directories(subsetDir, ec);
    if(ec) return 1;

    auto reduced = ::getReducedTree(inputPath, branchesPath);
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
    std::uniform_int_distribution<long> dist(0, numEntries - 1);
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
        std::vector<bool> isTrain(indexes[i].size(), false);
        const long numTrain = static_cast<long>(indexes[i].size() * trainFrac);
        for(long j = 0; j < numTrain; j++) isTrain[j] = true;
        std::shuffle(isTrain.begin(), isTrain.end(), rng);

        for(long j = 0; j < indexes[i].size(); j++)
        {
            reducedTree->GetEntry(indexes[i][j]);
            if(isTrain[j]) trainSubsets[i]->Fill();
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
            reducedTree->CopyAddresses(trainSubsets[i], true);
        }
        if(testFiles[i] && testSubsets[i])
        {
            testFiles[i]->cd();
            testSubsets[i]->Write();
            reducedTree->CopyAddresses(testSubsets[i], true);
        }
    }

    if(auto* clones = reducedTree->GetListOfClones()) clones->Clear("nodelete");

    std::cout << "All trees created and written to memory" << std::endl;
    return 0;
}
