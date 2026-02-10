#!/usr/bin/env python3
#
# Author: Yipeng Sun
# Last Change: Sun Oct 31, 2021 at 03:40 AM +0100

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # Don't hijack argparse!
ROOT.PyConfig.DisableRootLogon = True  # Don't read .rootlogon.py

from argparse import ArgumentParser
from itertools import combinations
from ROOT import RDataFrame

from TrackerOnlyEmu.executor import ExecDirective as EXEC
from TrackerOnlyEmu.executor import process_directives
from TrackerOnlyEmu.emulation.run2_rdx import run2_rdx_hlt1_directive_gen


#################################
# Command line arguments parser #
#################################

def parse_input():
    parser = ArgumentParser(
        description='Emulate Hlt1{Track,TwoTrack}MVA triggers.')

    parser.add_argument('input', help='''
specify input ntuple file.
''')

    parser.add_argument('output', help='''
specify output ntuple file.
''')

    parser.add_argument('-t', '--tree', default='TupleB0/DecayTree', help='''
specify tree name.
''')

    parser.add_argument('-y', '--year', default='2016', help='''
specify year.''')

    parser.add_argument('-B', '--Bmeson', default='b0', help='''
specify the name of the B meson in the tree.''')

    parser.add_argument('--debug', action='store_true', help='''
enable debug mode.
''')

    return parser.parse_args()


##################
# Apply triggers #
##################

if __name__ == '__main__':
    args = parse_input()

    directives = run2_rdx_hlt1_directive_gen(args.Bmeson, args.year)

    directives_debug = [
        # Reference variables
        EXEC('Define', 'k_l0_global_dec', 'k_L0Global_Dec', True),
        EXEC('Define', 'k_hlt1_trackmva_tos', 'k_Hlt1TrackMVADecision_TOS',
             True),
        EXEC('Define', 'pi_l0_global_dec', 'pi_L0Global_Dec', True),
        EXEC('Define', 'pi_hlt1_trackmva_tos', 'pi_Hlt1TrackMVADecision_TOS',
             True),
        EXEC('Define', 'd0_l0_global_dec', 'd0_L0Global_Dec', True),
        EXEC('Define', 'd0_hlt1_trackmva_tos', 'd0_Hlt1TrackMVADecision_TOS',
             True),
        EXEC('Define', 'd0_hlt1_twotrackmva_tos',
             'd0_Hlt1TwoTrackMVADecision_TOS', True),

        # Fit variables
        EXEC('Define', 'q2', 'FitVar_q2 / 1e6', True),
        EXEC('Define', 'mmiss2', 'FitVar_Mmiss2 / 1e6', True),
        EXEC('Define', 'el', 'FitVar_El / 1e3', True),

        # Kinematic variables
        EXEC('Define', 'k_pt', 'k_PT / 1e3', True),
        EXEC('Define', 'k_p', 'k_P / 1e3', True),
        EXEC('Define', 'pi_pt', 'pi_PT / 1e3', True),
        EXEC('Define', 'pi_p', 'pi_P / 1e3', True),
        EXEC('Define', 'mu_pt', 'mu_PT / 1e3', True),
        EXEC('Define', 'mu_p', 'mu_P / 1e3', True),
        EXEC('Define', 'k_pi_pt', 'computePt(k_PX+pi_PX, k_PY+pi_PY) / 1e3', True),
        EXEC('Define', 'd0_pt', 'd0_PT / 1e3', True),

        # Track quality variables
        EXEC('Define', 'k_chi2ndof', 'k_TRACK_CHI2NDOF', True),
        EXEC('Define', 'k_ipchi2', 'k_IPCHI2_OWNPV', True),
        EXEC('Define', 'k_ghost', 'k_TRACK_GhostProb', True),
        EXEC('Define', 'pi_chi2ndof', 'pi_TRACK_CHI2NDOF', True),
        EXEC('Define', 'pi_ipchi2', 'pi_IPCHI2_OWNPV', True),
        EXEC('Define', 'pi_ghost', 'pi_TRACK_GhostProb', True),
        EXEC('Define', 'mu_chi2ndof', 'mu_TRACK_CHI2NDOF', True),
        EXEC('Define', 'mu_ipchi2', 'mu_IPCHI2_OWNPV', True),
        EXEC('Define', 'mu_ghost', 'mu_TRACK_GhostProb', True),

        # Angular variables
        EXEC('Define', 'k_theta', 'theta(k_PZ, k_P)', True),
        EXEC('Define', 'k_phi', 'phi(k_PX, k_PY)', True),
        EXEC('Define', 'pi_theta', 'theta(pi_PZ, pi_P)', True),
        EXEC('Define', 'pi_phi', 'phi(pi_PX, pi_PY)', True),
        EXEC('Define', 'mu_theta', 'theta(mu_PZ, mu_P)', True),
        EXEC('Define', 'mu_phi', 'phi(mu_PX, mu_PY)', True),
    ]

    directives_mva = []
    for i, j in combinations(range(1, 5), 2):
        directives_mva.append(
            EXEC('Define', 'mva_score_{}_{}'.format(i, j),
                 '{}_Matrixnet_Hlt1TwoTrackMVAEmulations_{}_{}'.format(
                     args.Bmeson, i, j), True))
        directives_mva.append(
            EXEC('Define', 'mva_dira_{}_{}'.format(i, j),
                 '{}_DIRA_OWNPV_COMB_{}_{}'.format(args.Bmeson, i, j), True))
        directives_mva.append(
            EXEC('Define', 'mva_doca_{}_{}'.format(i, j),
                 '{}_DOCA_COMB_{}_{}'.format(args.Bmeson, i, j), True))
        directives_mva.append(
            EXEC('Define', 'mva_eta_{}_{}'.format(i, j),
                 '{}_ETA_COMB_{}_{}'.format(args.Bmeson, i, j), True))
        directives_mva.append(
            EXEC('Define', 'mva_ip_chi2_{}_{}'.format(i, j),
                 '{}_IPCHI2_OWNPV_COMB_{}_{}'.format(args.Bmeson, i, j), True))
        directives_mva.append(
            EXEC('Define', 'mva_pt_{}_{}'.format(i, j),
                 '{}_PT_COMB_{}_{}'.format(args.Bmeson, i, j), True))
        directives_mva.append(
            EXEC('Define', 'mva_p_{}_{}'.format(i, j),
                 '{}_P_COMB_{}_{}'.format(args.Bmeson, i, j), True))
        directives_mva.append(
            EXEC('Define', 'mva_vd_chi2_{}_{}'.format(i, j),
                 '{}_VDCHI2_OWNPV_COMB_{}_{}'.format(args.Bmeson, i, j), True))
        directives_mva.append(
            EXEC('Define', 'mva_vertex_chi2_{}_{}'.format(i, j),
                 '{}_VERTEX_CHI2_COMB_{}_{}'.format(args.Bmeson, i, j), True))
        directives_mva.append(
            EXEC('Define', 'mva_vertex_ndof_{}_{}'.format(i, j),
                 '{}_VERTEX_NDOF_COMB_{}_{}'.format(args.Bmeson, i, j), True))
        directives_mva.append(
            EXEC('Define', 'mva_mcorr_{}_{}'.format(i, j),
                 '{}_MCORR_OWNPV_COMB_{}_{}'.format(args.Bmeson, i, j), True))

    if args.debug:
        directives += directives_debug
        directives += directives_mva
        # Apply the nSPDHits cut
        # directives.append(
        #     EXEC('Filter', instruct='NumSPDHits < 450'))
        directives.append(
            EXEC('Define', 'nspd_hits', 'NumSPDHits', True))

    init_frame = RDataFrame(args.tree, args.input)
    dfs, output_br_names = process_directives(directives, init_frame)

    # Always keep run and event numbers
    output_br_names.push_back('runNumber')
    output_br_names.push_back('eventNumber')

    # Output
    dfs[-1].Snapshot(args.tree, args.output, output_br_names)
