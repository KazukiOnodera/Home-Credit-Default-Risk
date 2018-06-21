#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 00:47:12 2018

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
PREF = 'ins_304_'

KEY = 'SK_ID_CURR'

day_start = -365*3  # min: -2922
day_end   = -365*2  # min: -2922

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
ins = utils.read_pickles('../data/installments_payments')
ins = ins[ins['DAYS_INSTALMENT'].between(day_start, day_end)]



num_aggregations = {
    # TODO: optimize stats
    'NUM_INSTALMENT_VERSION': ['nunique'],
    'DPD': ['max', 'mean', 'sum', 'nunique'],
    'DBD': ['max', 'mean', 'sum', 'nunique'],
    'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
    'AMT_PAYMENT':    ['min', 'max', 'mean', 'sum'],
    'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'],
    'amt_ratio': ['min', 'mean', 'var'],
    'amt_delta': ['min', 'mean', 'var'],
    'days_weighted_delay': ['min', 'max', 'mean', 'sum'],
    'delayed_day_over0': ['min', 'max', 'mean', 'var'],
    'delayed_money_0': ['min', 'max', 'mean', 'var'],
    'delayed_money_ratio_0': ['min', 'max', 'mean', 'var'],
    'not-delayed_day_0': ['min', 'max', 'mean', 'var'],
    'not-delayed_money_0': ['min', 'max', 'mean', 'var'],
    'not-delayed_money_ratio_0': ['min', 'max', 'mean', 'var'],
    'delayed_day_over5': ['min', 'max', 'mean', 'var'],
    'delayed_money_5': ['min', 'max', 'mean', 'var'],
    'delayed_money_ratio_5': ['min', 'max', 'mean', 'var'],
    'not-delayed_day_5': ['min', 'max', 'mean', 'var'],
    'not-delayed_money_5': ['min', 'max', 'mean', 'var'],
    'not-delayed_money_ratio_5': ['min', 'max', 'mean', 'var'],
    'delayed_day_over10': ['min', 'max', 'mean', 'var'],
    'delayed_money_10': ['min', 'max', 'mean', 'var'],
    'delayed_money_ratio_10': ['min', 'max', 'mean', 'var'],
    'not-delayed_day_10': ['min', 'max', 'mean', 'var'],
    'not-delayed_money_10': ['min', 'max', 'mean', 'var'],
    'not-delayed_money_ratio_10': ['min', 'max', 'mean', 'var'],
    'delayed_day_over15': ['min', 'max', 'mean', 'var'],
    'delayed_money_15': ['min', 'max', 'mean', 'var'],
    'delayed_money_ratio_15': ['min', 'max', 'mean', 'var'],
    'not-delayed_day_15': ['min', 'max', 'mean', 'var'],
    'not-delayed_money_15': ['min', 'max', 'mean', 'var'],
    'not-delayed_money_ratio_15': ['min', 'max', 'mean', 'var'],
    'delayed_day_over20': ['min', 'max', 'mean', 'var'],
    'delayed_money_20': ['min', 'max', 'mean', 'var'],
    'delayed_money_ratio_20': ['min', 'max', 'mean', 'var'],
    'not-delayed_day_20': ['min', 'max', 'mean', 'var'],
    'not-delayed_money_20': ['min', 'max', 'mean', 'var'],
    'not-delayed_money_ratio_20': ['min', 'max', 'mean', 'var'],
    'delayed_day_over25': ['min', 'max', 'mean', 'var'],
    'delayed_money_25': ['min', 'max', 'mean', 'var'],
    'delayed_money_ratio_25': ['min', 'max', 'mean', 'var'],
    'not-delayed_day_25': ['min', 'max', 'mean', 'var'],
    'not-delayed_money_25': ['min', 'max', 'mean', 'var'],
    'not-delayed_money_ratio_25': ['min', 'max', 'mean', 'var'],
    'delayed_day_over30': ['min', 'max', 'mean', 'var'],
    'delayed_money_30': ['min', 'max', 'mean', 'var'],
    'delayed_money_ratio_30': ['min', 'max', 'mean', 'var'],
    'not-delayed_day_30': ['min', 'max', 'mean', 'var'],
    'not-delayed_money_30': ['min', 'max', 'mean', 'var'],
    'not-delayed_money_ratio_30': ['min', 'max', 'mean', 'var'],
    'delayed_day_over35': ['min', 'max', 'mean', 'var'],
    'delayed_money_35': ['min', 'max', 'mean', 'var'],
    'delayed_money_ratio_35': ['min', 'max', 'mean', 'var'],
    'not-delayed_day_35': ['min', 'max', 'mean', 'var'],
    'not-delayed_money_35': ['min', 'max', 'mean', 'var'],
    'not-delayed_money_ratio_35': ['min', 'max', 'mean', 'var'],
    'delayed_day_over40': ['min', 'max', 'mean', 'var'],
    'delayed_money_40': ['min', 'max', 'mean', 'var'],
    'delayed_money_ratio_40': ['min', 'max', 'mean', 'var'],
    'not-delayed_day_40': ['min', 'max', 'mean', 'var'],
    'not-delayed_money_40': ['min', 'max', 'mean', 'var'],
    'not-delayed_money_ratio_40': ['min', 'max', 'mean', 'var'],
    'delayed_day_over45':         ['min', 'max', 'mean', 'var'],
    'delayed_money_45':           ['min', 'max', 'mean', 'var'],
    'delayed_money_ratio_45':     ['min', 'max', 'mean', 'var'],
    'not-delayed_day_45':         ['min', 'max', 'mean', 'var'],
    'not-delayed_money_45':       ['min', 'max', 'mean', 'var'],
    'not-delayed_money_ratio_45': ['min', 'max', 'mean', 'var'],
}



#col_cat = ['NAME_CONTRACT_STATUS']

train = utils.load_train([KEY])
test = utils.load_test([KEY])

# =============================================================================
# 
# =============================================================================
def aggregate():
    
    df = ins
    
#    li = []
#    for c1 in df.columns:
#        for c2 in col_cat:
#            if c1.startswith(c2+'_'):
#                li.append(c1)
#                break
#    
#    cat_aggregations = {}
#    for cat in li:
#        cat_aggregations[cat] = ['mean', 'sum']
    
    df_agg = df.groupby('SK_ID_CURR').agg({**num_aggregations})
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    df_agg['INS_COUNT'] = df.groupby('SK_ID_CURR').size()
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
