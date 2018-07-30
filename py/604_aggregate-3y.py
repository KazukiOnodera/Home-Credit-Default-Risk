#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 09:52:37 2018

@author: Kazuki

based on
https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features/code
"""

import numpy as np
import pandas as pd
import gc
import os
from multiprocessing import Pool, cpu_count
NTHREAD = cpu_count()
import utils, utils_agg
utils.start(__file__)
#==============================================================================
PREF = 'f604_'

KEY = 'SK_ID_CURR'

month_start = -12*3 # -96
month_end   = -12*2 # -96

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
bure = utils.read_pickles('../data/bureau', [KEY, 'SK_ID_BUREAU', 'CREDIT_ACTIVE'])
bb = utils.read_pickles('../data/bureau_balance')
bb = bb[bb['MONTHS_BALANCE'].between(month_start, month_end)]

bb = pd.merge(bb, bure, on='SK_ID_BUREAU', how='left')



train = utils.load_train([KEY])
test = utils.load_test([KEY])

# =============================================================================
# 
# =============================================================================

def aggregate(args):
    print(args)
    k, v, prefix = args
    
    df = (bb[bb[k]==v])
    
    df_agg = df.groupby(KEY).agg(utils_agg.bb_num_aggregations)
    df_agg.columns = pd.Index([prefix + '_' + e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    # std / mean
    col_std = [c for c in df_agg.columns if c.endswith('_std')]
    for c in col_std:
        df_agg[f'{c}-d-mean'] = df_agg[c]/df_agg[c.replace('_std', '_mean')]
    
    # max / min
    col_max = [c for c in df_agg.columns if c.endswith('_max')]
    for c in col_max:
        df_agg[f'{c}-d-min'] = df_agg[c]/df_agg[c.replace('_max', '_min')]
    
    df_agg[f'{prefix}_BURE_COUNT'] = df.groupby(KEY).size()
    
    gr2size = df.groupby([KEY, 'SK_ID_BUREAU']).size()
    gr2size.name = 'CURR-BUREAU_cnt'
    gr1 = gr2size.groupby(KEY)
    gr1size = gr1.agg({**{'CURR-BUREAU_cnt': ['min', 'max', 'mean', 'sum', 'var', 'size']}})
    gr1size.columns = pd.Index([prefix + '_' + e[0] + "_" + e[1] for e in gr1size.columns.tolist()])
    
    df_agg = pd.concat([df_agg, gr1size], axis=1)
    
    # merge
    df_agg.reset_index(inplace=True)
    tmp = pd.merge(train, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF), '../feature/train')
    
    tmp = pd.merge(test, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')
    
    return

# =============================================================================
# main
# =============================================================================


argss = [
        ['CREDIT_ACTIVE', 'Active', 'Active'],
        ['CREDIT_ACTIVE', 'Closed', 'Closed']
        ]

pool = Pool(NTHREAD)
callback = pool.map(aggregate, argss)
pool.close()



#==============================================================================
utils.end(__file__)
