#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 21:36:19 2018

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
import utils_agg
import utils
utils.start(__file__)
#==============================================================================
PREF = 'f203_'

KEY = 'SK_ID_CURR'

month_start = -12*2 # -96
month_end   = -12*1 # -96

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
pos = utils.read_pickles('../data/POS_CASH_balance')
pos = pos[pos['MONTHS_BALANCE'].between(month_start, month_end)]


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
    
    df_agg = df.groupby(KEY).agg({**utils_agg.pos_num_aggregations, **cat_aggregations})
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    # std / mean
    col_std = [c for c in df_agg.columns if c.endswith('_std')]
    for c in col_std:
        df_agg[f'{c}-d-mean'] = df_agg[c]/df_agg[c.replace('_std', '_mean')]
    
    # max / min
    col_max = [c for c in df_agg.columns if c.endswith('_max')]
    for c in col_max:
        try:
            df_agg[f'{c}-d-min'] = df_agg[c]/df_agg[c.replace('_max', '_min')]
        except:
            pass
    
    df_agg['POS_COUNT'] = df.groupby(KEY).size()
    df_agg.reset_index(inplace=True)
    
    utils.remove_feature(df_agg, var_limit=0, corr_limit=0.98, sample_size=19999)
    
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
