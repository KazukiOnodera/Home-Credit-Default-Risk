#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 21:37:00 2018

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
import utils
utils.start(__file__)
#==============================================================================
PREF = 'pos_204_'

KEY = 'SK_ID_CURR'

month_start = -12*3 # -96
month_end   = -12*2 # -96

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
pos = utils.read_pickles('../data/POS_CASH_balance')
pos = pos[pos['MONTHS_BALANCE'].between(month_start, month_end)]


num_aggregations = {
    # TODO: optimize stats
    'MONTHS_BALANCE': ['min', 'max', 'mean', 'size'],
    'SK_DPD': ['max', 'mean'],
    'SK_DPD_DEF': ['max', 'mean'],
    
    'CNT_INSTALMENT_diff':  ['min', 'max', 'mean', 'var'],
    'CNT_INSTALMENT_ratio': ['min', 'max', 'mean', 'var'],
    
    'SK_DPD_diff':          ['max', 'mean', 'var', 'sum'],
    'SK_DPD_diff_over0':    ['max', 'mean', 'var', 'sum'],
    'SK_DPD_diff_over5':    ['max', 'mean', 'var', 'sum'],
    'SK_DPD_diff_over10':   ['max', 'mean', 'var', 'sum'],
    'SK_DPD_diff_over15':   ['max', 'mean', 'var', 'sum'],
    'SK_DPD_diff_over20':   ['max', 'mean', 'var', 'sum'],
    'SK_DPD_diff_over25':   ['max', 'mean', 'var', 'sum'],
}

col_cat = ['NAME_CONTRACT_STATUS']

train = utils.load_train([KEY])
test = utils.load_test([KEY])

# =============================================================================
# 
# =============================================================================
def aggregate():
    
    df = utils.get_dummies(pos)
    
    li = []
    for c1 in df.columns:
        for c2 in col_cat:
            if c1.startswith(c2+'_'):
                li.append(c1)
                break
    
    cat_aggregations = {}
    for cat in li:
        cat_aggregations[cat] = ['mean', 'sum']
    
    df_agg = df.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    df_agg['POS_COUNT'] = df.groupby('SK_ID_CURR').size()
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
