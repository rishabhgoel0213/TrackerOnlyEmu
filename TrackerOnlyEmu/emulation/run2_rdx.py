#!/usr/bin/env python3
#
# Author: Yipeng Sun
# License: BSD 2-clause
# Last Change: Wed Dec 15, 2021 at 06:08 PM +0100

from itertools import combinations
from ROOT import gInterpreter

from TrackerOnlyEmu.loader import load_file, load_cpp
from TrackerOnlyEmu.executor import ExecDirective as EXEC
from TrackerOnlyEmu.utils import func_call_gen


#################
# L0 Hadron TOS #
#################
# Configurables ################################################################

XGB_TRAIN_BRANCHES = [
    'nTracks',  # To model the NumSPDHits < 450 cut
    'd0_P',
    'd0_PT',
    'k_P',
    'k_PT',
    # 'k_TRUEPT',
    'k_L0Calo_HCAL_realET',
    'k_L0Calo_HCAL_xProjection',
    'k_L0Calo_HCAL_yProjection',
    'k_L0Calo_HCAL_region',
    'pi_P',
    'pi_PT',
    # 'pi_TRUEPT',
    'pi_L0Calo_HCAL_realET',
    'pi_L0Calo_HCAL_xProjection',
    'pi_L0Calo_HCAL_yProjection',
    'pi_L0Calo_HCAL_region',
    # 'spi_P',
    # 'spi_PT',
    # 'spi_TRUEPT',
    # 'spi_L0Calo_HCAL_realET',
    # 'spi_L0Calo_HCAL_xProjection',
    # 'spi_L0Calo_HCAL_yProjection',
    # 'spi_L0Calo_HCAL_region',
]


#################
# L0 Global TIS #
#################
# Main #########################################################################

def run2_rdx_l0_global_tis_directive_gen(Bmeson, year, adhoc_tis_correction=True, ntracks=False):
    load_cpp('<triggers/l0/run2-L0GlobalTIS.h>')

    if not ntracks: gInterpreter.Declare('auto histoResp = new TFile("{}");'.format(load_file('<triggers/l0/l0_tis_efficiency.root>')))
    else: gInterpreter.Declare('auto histoResp = new TFile("{}");'.format(load_file('<triggers/l0/l0_tis_efficiency3D.root>')))

    epilogue = '''
    auto hResp = readL0GlobalTisResp(histoResp);
    '''
    gInterpreter.Declare(epilogue)

    # NOTE: For RDX, we use TRUE B momentum due to missing neutrinos
    if not ntracks: return [
        EXEC('Define', '{}_pz'.format(Bmeson), '{}_TRUEP_Z'.format(Bmeson), True),
        EXEC('Define', '{}_pt'.format(Bmeson), '{}_TRUEPT'.format(Bmeson), True),
        EXEC('Define', '{}_l0_global_tis_emu'.format(Bmeson), 'l0GlobalTisTriggerEmu({}, {}, {}, hResp, {})'.format('{}_pz'.format(Bmeson), '{}_pt'.format(Bmeson), year, str(adhoc_tis_correction).lower()), True),
    ]
    else: return [
        EXEC('Define', '{}_pz'.format(Bmeson), '{}_TRUEP_Z'.format(Bmeson), True),
        EXEC('Define', '{}_pt'.format(Bmeson), '{}_TRUEPT'.format(Bmeson), True),
        EXEC('Define', 'ntracks', 'nTracks', True),
        EXEC('Define', '{}_l0_global_tis_emu'.format(Bmeson), 'l0GlobalTisTrigger3DEmu({}, {}, {}, {}, hResp, {})'.format('{}_pz'.format(Bmeson), '{}_pt'.format(Bmeson), 'ntracks', year, str(adhoc_tis_correction).lower()), True),
    ]


#########
# HLT 1 #
#########
# Configurables ################################################################

GEC_SEL_BRANCHES = [
    'NumVeloClusters',
    'NumITClusters',
    'NumOTClusters',
]

GLOBAL_CORR_BRANCHES = {
    'particle': [
        'TRACK_nTTHits'
    ],
    'global': [
        'NumVeloClusters',
        'NumITClusters',
        'NumOTClusters',
    ]
}

TRACK_SEL_BRANCHES = [
    'PT',
    'P',
    'TRACK_CHI2NDOF',
    'IPCHI2_OWNPV',
    'TRACK_GhostProb'
]

TWO_TRACK_SPEC_BRANCHES = {
    'PT':  'PT',
    'P': 'P',
    'TRCHI2DOF': 'TRACK_CHI2NDOF',
    'BPVIPCHI2': 'IPCHI2_OWNPV',
    'TRGHOSTPROB': 'TRACK_GhostProb',
    'PX': 'PX',
    'PY': 'PY',
}

TWO_TRACK_COMB_SPEC_BRANCHES = {
    'VDCHI2': 'VDCHI2_OWNPV_COMB',
    'SUMPT': 'SUMPT_COMB',
    'DOCA': 'DOCA_COMB',
    'VCHI2': 'VERTEX_CHI2_COMB',
    'BPVETA': 'ETA_COMB',
    'BPVCORRM': 'MCORR_OWNPV_COMB',
    'BPVDIRA': 'DIRA_OWNPV_COMB',
    'MVA': 'Matrixnet_Hlt1TwoTrackMVAEmulations',
}


# Helpers ######################################################################

def global_corr_gen(particle):
    params = [particle+'_'+b for b in GLOBAL_CORR_BRANCHES['particle']]
    params += GLOBAL_CORR_BRANCHES['global']
    return 'hlt1GlobalPass({})'.format(', '.join(params))


def track_spec_gen(particles, branches):
    specs = []

    for p in particles:
        spec = []
        for name, val in branches.items():
            spec.append('{' + '"{}"'.format(name) + ', ' + p + '_' + val  + '}')

        specs.append('{' + ', '.join(spec) + '}')

    return 'vector<map<string, double> >{' + ', '.join(specs) + '}'


def comb_spec_gen(particle, branches, suffixs):
    specs = []

    for suf1, suf2 in combinations(suffixs, 2):
        spec = []
        for name, val in branches.items():
            spec.append(
                '{' + '"{}"'.format(name) + ', ' + particle + '_' + val +
                '_' + '{}_{}'.format(suf1, suf2) + '}')

        specs.append('{' + ', '.join(spec) + '}')

    return 'vector<map<string, double> >{' + ', '.join(specs) + '}'


# Main #########################################################################

def run2_rdx_hlt1_directive_gen(Bmeson, year):
    load_cpp('<triggers/hlt1/run2-Hlt1GEC.h>')
    load_cpp('<triggers/hlt1/run2-Hlt1TrackMVA.h>')
    load_cpp('<triggers/hlt1/run2-Hlt1TwoTrackMVA.h>')
    load_cpp('<triggers/kinematics.h>')

    return [
        # Various corrections
        EXEC('Define', 'pass_gec',
             func_call_gen('hlt1GEC', GEC_SEL_BRANCHES), True),
        EXEC('Define', 'k_pass_hlt1_corr', global_corr_gen('k'), True),
        EXEC('Define', 'pi_pass_hlt1_corr', global_corr_gen('pi'), True),

        # Hlt1TrackMVA emulation
        EXEC('Define', 'k_hlt1_trackmva_tos_emu',
             func_call_gen(
                 'hlt1TrackMVATriggerEmu',
                 ['k_'+n for n in TRACK_SEL_BRANCHES] +
                 ['k_pass_hlt1_corr', year]), True),
        EXEC('Define', 'pi_hlt1_trackmva_tos_emu',
             func_call_gen(
                 'hlt1TrackMVATriggerEmu',
                 ['pi_'+n for n in TRACK_SEL_BRANCHES] +
                 ['pi_pass_hlt1_corr', year]), True),
        EXEC('Define', 'd0_hlt1_trackmva_tos_emu',
             'k_hlt1_trackmva_tos_emu || pi_hlt1_trackmva_tos_emu', True),

        # Hlt1TwoTrackMVA emulation
        EXEC('Define', 'vec_pass_hlt1_corr',
             'vector<bool>{ k_pass_hlt1_corr, pi_pass_hlt1_corr }'),
        EXEC('Define', 'track_spec',
             track_spec_gen(['k', 'pi'], TWO_TRACK_SPEC_BRANCHES)),
        EXEC('Define', 'comb_spec',
             comb_spec_gen(Bmeson,
                           TWO_TRACK_COMB_SPEC_BRANCHES, range(1, 4))),
        EXEC('Define', 'd0_hlt1_twotrackmva_tos_emu',
             'hlt1TwoTrackMVATriggerEmu(track_spec, comb_spec, vec_pass_hlt1_corr, {})'.format(year),
             True),
    ]
