#!/usr/bin/env python3
#
# Author: Yipeng Sun
# Last Change: Sun Oct 31, 2021 at 04:55 AM +0100

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # Don't hijack argparse!
ROOT.PyConfig.DisableRootLogon = True  # Don't read .rootlogon.py

import pickle

from argparse import ArgumentParser
from ROOT import RDataFrame
from ROOT.RDF import RSnapshotOptions

from TrackerOnlyEmu.loader import load_file
from TrackerOnlyEmu.executor import ExecDirective as EXEC
from TrackerOnlyEmu.executor import process_directives
from TrackerOnlyEmu.utils import get_df_vars
from TrackerOnlyEmu.emulation.run2_rdx import (
    run2_rdx_l0_global_tis_directive_gen,
    run2_rdx_hlt1_directive_gen,
    XGB_TRAIN_BRANCHES,
)


#################################
# Command line arguments parser #
#################################

def parse_input():
    parser = ArgumentParser(
        description='Emulate all triggers for run 2 RDX.')

    parser.add_argument('input', help='''
specify input ntuple file.
''')

    parser.add_argument('output', help='''
specify output ntuple file.
''')

    parser.add_argument('-t', '--tree', default='TupleB0/DecayTree', help='''
specify tree name.
''')

    parser.add_argument('-y', '--year', default='2016', help='specify year.')

    parser.add_argument('-c', '--adhoc_tis_correction', action='store_true', help='specify if ad-hoc L0 Global TIS correction at high B pT should be used')

    parser.add_argument('-n', '--ntracks', action='store_true', help='specify if L0 Global TIS eff should consider nTracks for val (3D=[log(pz),log(pt),nTracks])')

    parser.add_argument('-B', '--Bmeson', default='b0', help='''
specify the name of the B meson in the tree.''')

    parser.add_argument('-l', '--load', default='<triggers/l0/xgb4-2016.pickle>', help='''
specify the trained regressor to load.''')

    parser.add_argument('--debug', action='store_true', help='''
enable debug mode.
''')

    return parser.parse_args()


##################
# Apply triggers #
##################

if __name__ == '__main__':
    args = parse_input()

    if args.adhoc_tis_correction: print('Note: using ad-hoc correction for L0 Global TIS measurement at high B log(pT)')
    if args.ntracks: print('Note: Using 3D eff histo in [log(pz),log(pt),nTracks] for L0 Global TIS eff')

    # L0Global TIS
    directives = run2_rdx_l0_global_tis_directive_gen(args.Bmeson, args.year, args.adhoc_tis_correction, args.ntracks)
    # HLT 1
    directives += run2_rdx_hlt1_directive_gen(args.Bmeson, args.year)

    init_frame = RDataFrame(args.tree, args.input)
    dfs, _ = process_directives(directives, init_frame)

    # L0Hadron TOS
    input_vars = get_df_vars(init_frame, XGB_TRAIN_BRANCHES)
    regressor = pickle.load(open(load_file(args.load), 'rb'))

    # Collect the previous output branches
    hlt1_brs = [
        'k_hlt1_trackmva_tos_emu', 'pi_hlt1_trackmva_tos_emu',
        'd0_hlt1_trackmva_tos_emu', 'd0_hlt1_twotrackmva_tos_emu',
    ]
    out_np = dfs[-1].AsNumpy(columns=['runNumber', 'eventNumber']+hlt1_brs)
    for br in hlt1_brs:  # Convert to int otherwise RDataFrame doesn't like it
        out_np[br] = out_np[br].astype(int)

    out_tmp = {k+'_tmp': v for k, v in out_np.items()}
    # Add the L0Global TIS branch
    l0global_tis_br = f'{args.Bmeson}_l0_global_tis_emu'
    out_tmp[l0global_tis_br] = get_df_vars(dfs[-1], l0global_tis_br)
    # Add the L0Hadron TOS branch
    out_tmp['d0_l0_hadron_tos_emu'] = regressor.predict_proba(input_vars).T[1]
    out_df = ROOT.RDF.MakeNumpyDataFrame(out_tmp)

    directives_post = [
        EXEC('Define', br, f'static_cast<Bool_t>({br}_tmp)', True)
        for br in hlt1_brs
    ] + [
        EXEC('Define', 'runNumber', 'static_cast<UInt_t>(runNumber_tmp)', True),
        EXEC('Define', 'eventNumber',
             'static_cast<ULong64_t>(eventNumber_tmp)', True),
    ]

    # Output
    out_dfs, output_br_names = process_directives(directives_post, out_df)
    output_br_names.push_back('d0_l0_hadron_tos_emu')
    output_br_names.push_back(l0global_tis_br)

    output_opts = RSnapshotOptions()
    output_opts.fMode = 'UPDATE'
    out_dfs[-1].Snapshot(args.tree, args.output, output_br_names, output_opts)
