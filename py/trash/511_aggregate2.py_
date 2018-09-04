#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 08:25:44 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import gc
import os
from multiprocessing import Pool, cpu_count
NTHREAD = cpu_count()
import utils_agg
import utils
utils.start(__file__)
#==============================================================================
PREF = 'f511_'

KEY = 'SK_ID_CURR'

day_start = -365*10 # -2922
day_end   = -365*0 # -2922

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
bure = utils.read_pickles('../data/bureau')
bure = bure[bure['DAYS_CREDIT'].between(day_start, day_end)]


col_cat = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']

train = utils.load_train([KEY])
test = utils.load_test([KEY])

# =============================================================================
# 
# =============================================================================
def aggregate():
    
    df = utils.get_dummies(bure)
    
    li = []
    for c1 in df.columns:
        for c2 in col_cat:
            if c1.startswith(c2+'_'):
                li.append(c1)
                break
    
    cat_aggregations = {}
    for cat in li:
        cat_aggregations[cat] = ['mean', 'sum']
    
    df_agg = df.groupby('SK_ID_BUREAU').agg({**utils_agg.bure_num_aggregations, **cat_aggregations})
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    df_agg[KEY] = df.groupby('SK_ID_BUREAU')[KEY].mean()
    
    num_agg = {}
    for s in ['min', 'mean', 'max', 'std']:
        col = [c for c in df_agg.columns if c.endswith(f'_{s}')]
        for c in col:
            if c == 'min':
                num_agg[c] = ['mean', 'max', 'std']
            elif c == 'max':
                num_agg[c] = ['min', 'mean', 'std']
            else:
                num_agg[c] = ['min', 'mean', 'max', 'std']
    
    df_agg = df_agg.groupby(KEY).agg({**num_agg})
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
#    # std / mean
#    col_std = [c for c in df_agg.columns if c.endswith('_std')]
#    for c in col_std:
#        df_agg[f'{c}-d-mean'] = df_agg[c]/df_agg[c.replace('_std', '_mean')]
#    
#    # max / min
#    col_max = [c for c in df_agg.columns if c.endswith('_max')]
#    for c in col_max:
#        df_agg[f'{c}-d-min'] = df_agg[c]/df_agg[c.replace('_max', '_min')]
    
    df_agg['BURE_COUNT'] = df.groupby(KEY).size()
    df_agg.reset_index(inplace=True)
    
    tmp = pd.merge(train, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF), '../feature/train')
    
    tmp = pd.merge(test, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')
    
    return


# =============================================================================
# main
# =============================================================================

aggregate()




#==============================================================================
utils.end(__file__)


