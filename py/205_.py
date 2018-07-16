#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 00:54:23 2018

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
PREF = 'f205_'

KEY = 'SK_ID_CURR'

month_start = -12*10 # -96
month_end   = -12*0 # -96

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
col = ['SK_ID_PREV', 'SK_ID_CURR', 'MONTHS_BALANCE', 'CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE']
pos = utils.read_pickles('../data/POS_CASH_balance', col).sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'])
pos = pos[pos['MONTHS_BALANCE'].between(month_start, month_end)]

pos['CNT_INSTALMENT_FUTURE_diff'] = pos.groupby('SK_ID_PREV')['CNT_INSTALMENT_FUTURE'].diff()

train = utils.load_train([KEY])
test = utils.load_test([KEY])

pos_num_aggregations = {
    'CNT_INSTALMENT_FUTURE_diff': ['min', 'mean', 'max', 'var'],
        }
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
    
    df_agg = df.groupby(KEY).agg({**pos_num_aggregations})
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
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
