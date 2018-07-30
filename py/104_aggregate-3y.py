#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 16:49:12 2018

@author: kazuki.onodera

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

# setting
PREF = 'f104_'

KEY = 'SK_ID_CURR'

day_start = -365*3 # min: -2922
day_end   = -365*2 # min: -2922

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
prev = utils.read_pickles('../data/previous_application')
prev = prev[prev['DAYS_DECISION'].between(day_start, day_end)]

#prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION'], inplace=True, ascending=[True, False])


col_cat = ['NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'NAME_CASH_LOAN_PURPOSE',
           'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON',
           'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO',
           'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY', 'NAME_YIELD_GROUP',
           'PRODUCT_COMBINATION']

train = utils.load_train([KEY])
test = utils.load_test([KEY])

# =============================================================================
# 
# =============================================================================
def aggregate(args):
    print(args)
    k, v, prefix = args
    
    df = utils.get_dummies(prev[prev[k]==v])
    
    li = []
    for c1 in df.columns:
        for c2 in col_cat:
            if c1.startswith(c2+'_'):
                li.append(c1)
                break
    
    cat_aggregations = {}
    for cat in li:
        cat_aggregations[cat] = ['mean', 'sum']
    
    df_agg = df.groupby('SK_ID_CURR').agg({**utils_agg.prev_num_aggregations, **cat_aggregations})
    df_agg.columns = pd.Index([prefix + e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    # std / mean
    col_std = [c for c in df_agg.columns if c.endswith('_std')]
    for c in col_std:
        df_agg[f'{c}-d-mean'] = df_agg[c]/df_agg[c.replace('_std', '_mean')]
    
    # max / min
    col_max = [c for c in df_agg.columns if c.endswith('_max')]
    for c in col_max:
        df_agg[f'{c}-d-min'] = df_agg[c]/df_agg[c.replace('_max', '_min')]
    
    df_agg[prefix+'PREV_COUNT'] = df.groupby('SK_ID_CURR').size()
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


argss = [
        ['NAME_CONTRACT_STATUS', 'Approved', 'approved_'],
        ['NAME_CONTRACT_STATUS', 'Refused', 'refused_'],
        ['NAME_YIELD_GROUP', 'high', 'nyg-high_'],
        ['NAME_YIELD_GROUP', 'middle', 'nyg-middle_'],
        ['NAME_YIELD_GROUP', 'low_normal', 'nyg-low_normal_'],
        ['NAME_YIELD_GROUP', 'low_action', 'nyg-low_action_'],
        ['active',    1, 'active_'],
        ['completed', 1, 'completed_'],
        ]

pool = Pool(NTHREAD)
callback = pool.map(aggregate, argss)
pool.close()

#==============================================================================
utils.end(__file__)
