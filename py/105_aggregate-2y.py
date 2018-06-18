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
import utils
#utils.start(__file__)
#==============================================================================

# setting
PREF = 'prev_105_'

KEY = 'SK_ID_CURR'

day_start = -365*2 # min: -2922
day_end   = -365*1 # min: -2922

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
prev = utils.read_pickles('../data/previous_application')
prev = prev[prev['DAYS_DECISION'].between(day_start, day_end)]

#prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION'], inplace=True, ascending=[True, False])

num_aggregations = {
    # TODO: optimize stats
    'AMT_ANNUITY':             ['min', 'max', 'mean'],
    'AMT_APPLICATION':         ['min', 'max', 'mean'],
    'AMT_CREDIT':              ['min', 'max', 'mean'],
    'APP_CREDIT_PERC':         ['min', 'max', 'mean', 'var'],
    'AMT_DOWN_PAYMENT':        ['min', 'max', 'mean'],
    'AMT_GOODS_PRICE':         ['min', 'max', 'mean'],
    'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
    'RATE_DOWN_PAYMENT':       ['min', 'max', 'mean'],
    'DAYS_DECISION':           ['min', 'max', 'mean'],
    'CNT_PAYMENT':             ['mean', 'sum'],
}

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
    
    df_agg = df.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    df_agg.columns = pd.Index([prefix + e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
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
        ['NAME_CONTRACT_STATUS', 'Approved', 'approved_'],
        ['NAME_CONTRACT_STATUS', 'Approved', 'refused_'],
        ['active',    1, 'active_'],
        ['completed', 1, 'completed_'],
        ]

pool = Pool(NTHREAD)
callback = pool.map(aggregate, argss)
pool.close()

#==============================================================================
utils.end(__file__)
