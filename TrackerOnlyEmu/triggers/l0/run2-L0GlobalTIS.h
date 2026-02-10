// Stolen from:
//   https://gitlab.cern.ch/lhcb-slb/B02DplusTauNu/-/blob/master/tuple_processing_chain/emulate_L0Hadron_TOS_RLc.py
// Last Change: Wed May 19, 2021 at 12:27 AM +0200
//
#ifndef _RUN2_L0_HADRON_
#define _RUN2_L0_HADRON_

#include <map>

#include <TFile.h>
#include <TH2F.h>
#include <TMath.h>
#include <TString.h>

using std::map;
using std::vector;

///////////////////
// Configurables //
///////////////////

const map<int, TString> YEAR_HISTO_REL = {
    // { 2015, "0" },
    { 2016, "0" },
    { 2017, "1" },
    { 2018, "2" },
};

const TString RESP_HISTO_PREFIX = "Jpsi_data_eff";

//////////////////
// L0Global TIS //
//////////////////

map<int, TH2F*> readL0GlobalTisResp( TFile* ntp ) {
  map<int, TH2F*> resp;

  for ( auto const& m : YEAR_HISTO_REL ) {
    auto histo_name = RESP_HISTO_PREFIX + m.second;
    resp[m.first]   = static_cast<TH2F*>( ntp->Get( histo_name ) );
  }

  return resp;
}

// returns a float- an eff/prob for passing B L0Global TIS, as measured using TISTOS method in JpsiK data
float l0GlobalTisTriggerEmu( double PZ, double PT, int year,
                             map<int, TH2F*> respHistos, bool adhoc_correction ) {
  auto hist = respHistos[year];

  // ad-hoc correction to mimic high log(pT) behavior seen in rdx fullsim MC, found in lhcb-ntuples-gen/scripts/l0_global_tis_highpT_adhoc_correction.py using D*+munu fullsim MC
  map<int, map<int, vector<float>>> ADHOC_DSTMUNU_HIGHPT_CORRECTION;
  ADHOC_DSTMUNU_HIGHPT_CORRECTION[2016][1] = {9.576293217377373, 0.38961711525917053, -0.04307553315370772}; // D*munu high log(pT) bin mean, eff from JpsiK data, slope correction
  ADHOC_DSTMUNU_HIGHPT_CORRECTION[2016][2] = {9.645412819275125, 0.40560808777809143, 0.1920121662958972};
  ADHOC_DSTMUNU_HIGHPT_CORRECTION[2016][3] = {9.724147903446033, 0.41952842473983765, 0.21584737936752765};
  ADHOC_DSTMUNU_HIGHPT_CORRECTION[2016][4] = {9.80807058838877, 0.46126940846443176, 0.25353211968587264};
  ADHOC_DSTMUNU_HIGHPT_CORRECTION[2017][1] = {9.573918210303054, 0.3488537669181824, 0.19981855562035766};
  ADHOC_DSTMUNU_HIGHPT_CORRECTION[2017][2] = {9.648051795507696, 0.3812873661518097, 0.1692287430651749};
  ADHOC_DSTMUNU_HIGHPT_CORRECTION[2017][3] = {9.724531930069968, 0.42026373744010925, 0.20335972471697286};
  ADHOC_DSTMUNU_HIGHPT_CORRECTION[2017][4] = {9.80744115981552, 0.4522625207901001, 0.2293965047341547};
  ADHOC_DSTMUNU_HIGHPT_CORRECTION[2018][1] = {9.575316536210408, 0.36407339572906494, -0.15023259431100983};
  ADHOC_DSTMUNU_HIGHPT_CORRECTION[2018][2] = {9.64728926210228, 0.3955422341823578, 0.15679897820527733};
  ADHOC_DSTMUNU_HIGHPT_CORRECTION[2018][3] = {9.72602227412706, 0.4256914556026459, 0.24396464174205812};
  ADHOC_DSTMUNU_HIGHPT_CORRECTION[2018][4] = {9.807297853596634, 0.45863404870033264, 0.2866896902756077};

  if ( PZ > 0 ) {
    auto binPZ = hist->GetXaxis()->FindBin( TMath::Log( PZ ) );
    auto binPT = hist->GetYaxis()->FindBin( TMath::Log( PT ) );
    if (adhoc_correction && binPT == hist->GetNbinsY()) { // only apply correction to high log(pT) bin (w = w_uncor(a_i(log(pT)-m_i)+e_i)/e_i, but w_uncor = e_i, see l0_global_tis_highpT_adhoc_correction.py for notation)
      // std::cout << "...correcting L0 Global TIS measurement for high B log(pT)..." << std::endl;
      vector<float> adhoc = ADHOC_DSTMUNU_HIGHPT_CORRECTION[year][binPZ];
      return adhoc[2]*(TMath::Log(PT)-adhoc[0])+adhoc[1];
    }
    return hist->GetBinContent( binPZ, binPT );
  }

  return 0;
}

// alternative weight using 3D info for eff- [log(pz), log(pt), ntracks (new)]
float l0GlobalTisTrigger3DEmu( double PZ, double PT, int nTracks, int year,
                             map<int, TH2F*> respHistos, bool adhoc_correction ) {
  auto hist = respHistos[year];

  // ad-hoc correction to mimic high log(pT) behavior seen in rdx fullsim MC, found in lhcb-ntuples-gen/scripts/l0_global_tis_highpT_adhoc_correction.py using D*+munu fullsim MC
  // map<int, map<int, vector<float>>> ADHOC_DSTMUNU_HIGHPT_CORRECTION;
  // ADHOC_DSTMUNU_HIGHPT_CORRECTION[2016][1] = {9.576293217377373, 0.38961711525917053, -0.04307553315370772}; // D*munu high log(pT) bin mean, eff from JpsiK data, slope correction
  // ADHOC_DSTMUNU_HIGHPT_CORRECTION[2016][2] = {9.645412819275125, 0.40560808777809143, 0.1920121662958972};
  // ADHOC_DSTMUNU_HIGHPT_CORRECTION[2016][3] = {9.724147903446033, 0.41952842473983765, 0.21584737936752765};
  // ADHOC_DSTMUNU_HIGHPT_CORRECTION[2016][4] = {9.80807058838877, 0.46126940846443176, 0.25353211968587264};
  // ADHOC_DSTMUNU_HIGHPT_CORRECTION[2017][1] = {9.573918210303054, 0.3488537669181824, 0.19981855562035766};
  // ADHOC_DSTMUNU_HIGHPT_CORRECTION[2017][2] = {9.648051795507696, 0.3812873661518097, 0.1692287430651749};
  // ADHOC_DSTMUNU_HIGHPT_CORRECTION[2017][3] = {9.724531930069968, 0.42026373744010925, 0.20335972471697286};
  // ADHOC_DSTMUNU_HIGHPT_CORRECTION[2017][4] = {9.80744115981552, 0.4522625207901001, 0.2293965047341547};
  // ADHOC_DSTMUNU_HIGHPT_CORRECTION[2018][1] = {9.575316536210408, 0.36407339572906494, -0.15023259431100983};
  // ADHOC_DSTMUNU_HIGHPT_CORRECTION[2018][2] = {9.64728926210228, 0.3955422341823578, 0.15679897820527733};
  // ADHOC_DSTMUNU_HIGHPT_CORRECTION[2018][3] = {9.72602227412706, 0.4256914556026459, 0.24396464174205812};
  // ADHOC_DSTMUNU_HIGHPT_CORRECTION[2018][4] = {9.807297853596634, 0.45863404870033264, 0.2866896902756077};

  if ( PZ > 0 ) {
    auto binPZ = hist->GetXaxis()->FindBin( TMath::Log( PZ ) );
    auto binPT = hist->GetYaxis()->FindBin( TMath::Log( PT ) );
    auto binNTracks = hist->GetZaxis()->FindBin( nTracks );
    // if (adhoc_correction && binPT == hist->GetNbinsY()) { // only apply correction to high log(pT) bin (w = w_uncor(a_i(log(pT)-m_i)+e_i)/e_i, but w_uncor = e_i, see l0_global_tis_highpT_adhoc_correction.py for notation)
    //   // std::cout << "...correcting L0 Global TIS measurement for high B log(pT)..." << std::endl;
    //   vector<float> adhoc = ADHOC_DSTMUNU_HIGHPT_CORRECTION[year][binPZ];
    //   return adhoc[2]*(TMath::Log(PT)-adhoc[0])+adhoc[1];
    // }
    return hist->GetBinContent( binPZ, binPT, binNTracks );
  }

  return 0;
}

#endif
