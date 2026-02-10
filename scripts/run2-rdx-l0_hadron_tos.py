#!/usr/bin/env python3
#
# Author: Yipeng Sun
# Last Change: Tue Nov 09, 2021 at 03:15 PM +0100
# Based on the script 'regmva.py' shared by Patrick Owen

import pickle
import sys
import numpy as np

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True  # Don't hijack argparse!
ROOT.PyConfig.DisableRootLogon = True  # Don't read .rootlogon.py

from argparse import ArgumentParser
from copy import deepcopy
from ROOT import gInterpreter, RDataFrame
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier

from TrackerOnlyEmu.loader import load_cpp, load_file
from TrackerOnlyEmu.executor import ExecDirective as EXEC
from TrackerOnlyEmu.executor import process_directives
from TrackerOnlyEmu.utils import Timer
from TrackerOnlyEmu.utils import gen_output_dict, get_df_vars
from TrackerOnlyEmu.emulation.run2_rdx import XGB_TRAIN_BRANCHES


#################
# Configurables #
#################

def bdt_prepare():
    load_cpp('<triggers/l0/run2-L0Hadron.h>')

    gInterpreter.Declare('auto histoResp = new TFile("{}");'.format(
        load_file('<triggers/l0/hcal_et_response.root>')))
    gInterpreter.Declare('auto histoCluster = new TFile("{}");'.format(
        load_file('<triggers/l0/hcal_two_part_clusters.root>')))

    epilogue = '''
    auto hResp = readSinglePartResp(histoResp);

    auto hSharedIn  = static_cast<TH1D*>(histoCluster->Get("shared_with_radial_inner"));
    auto hSharedOut = static_cast<TH1D*>(histoCluster->Get("shared_with_radial_outer"));

    auto hMissIn  = static_cast<TH1D*>(histoCluster->Get("missing_with_radial_inner"));
    auto hMissOut = static_cast<TH1D*>(histoCluster->Get("missing_with_radial_outer"));
    '''
    gInterpreter.Declare(epilogue)


REGRESSOR_CONFIG = dict()

# XGB
REGRESSOR_CONFIG['xgb'] = {
    'train_brs': XGB_TRAIN_BRANCHES,
    'output_brs': [
        'runNumber',
        'eventNumber',
    ],
    'debug_brs': [
        'd0_l0_hadron_tos',
        'k_pt',
        'pi_pt',
        'd0_pt',
        'k_p',
        'pi_p',
        'd0_p',
        'nspdhits',
        'q2', 'mmiss2', 'el'
    ],
    'reg_br': 'd0_L0HadronDecision_TOS',
    'prep': lambda: 0,
    'predict': lambda xgb, input_vars: {
        'd0_l0_hadron_tos_emu_xgb': xgb.predict_proba(input_vars).T[1],
    },
    'dir': lambda args: [],
    'dir_debug': lambda args: [
        # Reference variables
        EXEC('Define', 'd0_l0_hadron_tos',
             'static_cast<Double_t>(d0_L0HadronDecision_TOS)', True),

        # Global variables
        EXEC('Define', 'nspdhits', 'NumSPDHits', True),

        # Fit variables
        EXEC('Define', 'q2', 'FitVar_q2 / 1e6', True),
        EXEC('Define', 'mmiss2', 'FitVar_Mmiss2 / 1e6', True),
        EXEC('Define', 'el', 'FitVar_El / 1e3', True),

        # Kinematic variables
        EXEC('Define', 'd0_pt', 'd0_PT / 1e3', True),
        EXEC('Define', 'k_pt', 'k_PT / 1e3', True),
        EXEC('Define', 'pi_pt', 'pi_PT / 1e3', True),
        EXEC('Define', 'd0_p', 'd0_P / 1e3', True),
        EXEC('Define', 'k_p', 'k_P / 1e3', True),
        EXEC('Define', 'pi_p', 'pi_P / 1e3', True),
    ],
    'dir_post': lambda args: [],
}

# BDT
REGRESSOR_CONFIG['bdt'] = {
    'train_brs': [
        'd0_PT',
        'd0_P',
        'k_L0Calo_HCAL_realET',  # NOTE: realET is not Trigger ET!
        'pi_L0Calo_HCAL_realET',
        'rdiff_k_pi'  # NOTE: This has to be computed by us!
    ],
    'output_brs': [
        'runNumber',
        'eventNumber',
        'd0_et_diff',
        'd0_et_emu_no_bdt',
    ],
    'debug_brs': [
        'd0_l0_hadron_tos',
        'd0_l0_hadron_tos_emu_no_bdt',
        'k_pt',
        'pi_pt',
        'd0_pt',
        'k_p',
        'pi_p',
        'd0_p',
        'd0_trg_et',
        'nspdhits',
        'k_L0Calo_HCAL_TriggerET',
        'pi_L0Calo_HCAL_TriggerET',
    ],
    'reg_br': 'd0_et_diff',
    'prep': bdt_prepare,
    'predict': lambda bdt, input_vars: {
        'd0_et_diff_pred': bdt.predict(input_vars)
    },
    'dir': lambda args: [
        EXEC('Define', 'k_et_smeared',
             'singlePartEt(k_P, k_PT, k_L0Calo_HCAL_realET, hResp)', True),
        EXEC('Define', 'pi_et_smeared',
             'singlePartEt(pi_P, pi_PT, pi_L0Calo_HCAL_realET, hResp)', True),
        # Used as a BDT input
        EXEC('Define', 'rdiff_k_pi', 'rDiff({})'.format(
            ', '.join([p+'_L0Calo_HCAL_'+d+'Projection'
                       for p in ['k', 'pi'] for d in ['x', 'y']])), True),

        # Trigger emulation based on physical considerations
        EXEC('Define', 'shared_k_pi',
             'isShared(rdiff_k_pi, k_L0Calo_HCAL_region, pi_L0Calo_HCAL_region, hSharedIn, hSharedOut)',
             True),
        EXEC('Define', 'miss_k_pi',
             'missingFraction(rdiff_k_pi, k_L0Calo_HCAL_region, pi_L0Calo_HCAL_region, hMissIn, hMissOut)',
             True),
        EXEC('Define', 'k_et_emu_no_bdt',
             'twoPartEt(k_et_smeared, pi_et_smeared, shared_k_pi, miss_k_pi)', True),
        EXEC('Define', 'pi_et_emu_no_bdt',
             'twoPartEt(pi_et_smeared, k_et_smeared, shared_k_pi, miss_k_pi)', True),
        EXEC('Define', 'd0_et_emu_no_bdt',
             'TMath::Max(k_et_emu_no_bdt, pi_et_emu_no_bdt)', True),
        EXEC('Define', 'd0_l0_hadron_tos_emu_no_bdt',
             'static_cast<Int_t>(l0HadronTriggerEmu(d0_et_emu_no_bdt, {}))'.format(args.year), True),
        EXEC('Define', 'd0_trg_et',
             'capHcalResp(k_L0Calo_HCAL_TriggerET, pi_L0Calo_HCAL_TriggerET)', True),

        # Regression variable
        EXEC('Define', 'd0_et_diff', 'd0_trg_et - d0_et_emu_no_bdt', True),
    ],
    'dir_debug': lambda args: [
        # Reference variables
        EXEC('Define', 'd0_l0_hadron_tos',
             'static_cast<Double_t>(d0_L0HadronDecision_TOS)', True),

        # Global variables
        EXEC('Define', 'nspdhits', 'NumSPDHits', True),

        # Fit variables
        EXEC('Define', 'q2', 'FitVar_q2 / 1e6', True),
        EXEC('Define', 'mmiss2', 'FitVar_Mmiss2 / 1e6', True),
        EXEC('Define', 'el', 'FitVar_El / 1e3', True),

        # Kinematic variables
        EXEC('Define', 'd0_pt', 'd0_PT / 1e3', True),
        EXEC('Define', 'k_pt', 'k_PT / 1e3', True),
        EXEC('Define', 'pi_pt', 'pi_PT / 1e3', True),
        EXEC('Define', 'd0_p', 'd0_P / 1e3', True),
        EXEC('Define', 'k_p', 'k_P / 1e3', True),
        EXEC('Define', 'pi_p', 'pi_P / 1e3', True),
    ],
    'dir_post': lambda args: [
        EXEC('Define', 'd0_et_emu_bdt', 'capHcalResp(d0_et_emu_no_bdt + d0_et_diff_pred)', True),
        EXEC('Define', 'd0_et_trg_pred_diff', 'd0_et_diff - d0_et_diff_pred', True),
        EXEC('Define', 'd0_l0_hadron_tos_emu_bdt',
             'static_cast<Double_t>(l0HadronTriggerEmu(d0_et_emu_bdt, {}))'.format(args.year), True),
    ],
}

# Old BDT, needed to deal w/ old training inputs
REGRESSOR_CONFIG['bdt_old'] = deepcopy(REGRESSOR_CONFIG['bdt'])
REGRESSOR_CONFIG['bdt_old']['dir'] = lambda args: [
    EXEC('Define', 'd0_trg_et',
         'capHcalResp(k_L0Calo_HCAL_TriggerET, pi_L0Calo_HCAL_TriggerET)', True),
    EXEC('Define', 'd0_et_diff', 'd0_trg_et - d0_et_emu_no_bdt', True),
]


#################################
# Command line arguments parser #
#################################

def parse_input():
    parser = ArgumentParser(
        description='Train a regression BDT/XGB for a better emulation of L0Hadron TOS trigger.')

    parser.add_argument('input', help='''
specify input ntuple file.''')

    parser.add_argument('output', help='''
specify output ntuple file.''')

    parser.add_argument('-t', '--tree', default='TupleB0/DecayTree', help='''
specify tree name.''')

    parser.add_argument('-y', '--year', default='2016', help='''
specify year.''')

    parser.add_argument('--debug', action='store_true', help='''
enable debug mode.
''')

    parser.add_argument('-m', '--mode', choices=['bdt', 'xgb', 'bdt_old'],
                        default='xgb', help='''
specify which regressor to use.''')

    parser.add_argument('--load', default=None, help='''
optionally specify serialized BDT to load.''')

    parser.add_argument('--dump', default=None, help='''
optionally specify output to pickled BDT object.''')

    parser.add_argument('--max-depth', default=4, type=int, help='''
optionally specify the max_depth parameter for the BDT.''')

    parser.add_argument('--ntrees', default=300, type=int, help='''
optionally specify the n_estimators parameter for the BDT.''')

    return parser.parse_args()


#############
# Train BDT #
#############

if __name__ == '__main__':
    args = parse_input()
    config = REGRESSOR_CONFIG[args.mode]

    config['prep']()
    train_brs = config['train_brs']
    reg_br = config['reg_br']
    output_brs = config['output_brs']

    directives = config['dir'](args)
    if args.debug:
        directives += config['dir_debug'](args)
        output_brs += config['debug_brs']

    init_frame = RDataFrame(args.tree, args.input)
    dfs, _ = process_directives(directives, init_frame)

    input_vars = get_df_vars(dfs[-1], train_brs)
    regression_var = get_df_vars(dfs[-1], reg_br)

    if not args.load:
        print(f'Start training a {args.mode} with {args.ntrees} trees and max-depth {args.max_depth}.')
        if args.mode in ['bdt', 'bdt_old']:
            regressor = AdaBoostRegressor(
                DecisionTreeRegressor(max_depth=args.max_depth),
                n_estimators=args.ntrees, random_state=np.random.RandomState(1))

        if args.mode == 'xgb':
            regressor = XGBClassifier(
                n_estimators=args.ntrees, max_depth=args.max_depth,
                use_label_encoder=False, eval_metric='mlogloss', reg_lambda=0.5)

        with Timer() as t:
            regressor.fit(input_vars, regression_var)
        print(f'BDT fitted. It takes a total of {t():,.2f} sec')

        if args.dump:
            print(f'Export trained {args.mode} to {args.dump}...')
            with open(args.dump, 'wb') as f:
                pickle.dump(regressor, f)

    else:
        if args.mode == 'bdt_old':
            print('bdt_old mode cannot be used for BDT application')
            sys.exit(255)

        print(f'Load already serialized {args.mode}...')
        regressor = pickle.load(open(load_file(args.load), 'rb'))

    # Output the ntuple
    print('Generate output ntuple...')
    output = gen_output_dict(input_vars, train_brs)
    output.update(dfs[-1].AsNumpy(columns=output_brs))
    output.update(config['predict'](regressor, input_vars))
    output_df = ROOT.RDF.MakeNumpyDataFrame(output)

    out_dfs, _ = process_directives(config['dir_post'](args), output_df)
    out_dfs[-1].Snapshot(args.tree, args.output)
