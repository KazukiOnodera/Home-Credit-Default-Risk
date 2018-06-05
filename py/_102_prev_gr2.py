#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 17:19:26 2018

@author: kazuki.onodera

previous_application

"""

import numpy as np
import pandas as pd
import gc
from multiprocessing import Pool
from glob import glob
import utils
utils.start(__file__)
#==============================================================================

KEY = 'SK_ID_CURR'
PREF = 'prev_102'
NTHREAD = 16

col_num = ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 
           'AMT_GOODS_PRICE', 'HOUR_APPR_PROCESS_START',
           'FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY',
           'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED',
           'CNT_PAYMENT',
           'DAYS_DECISION', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
           'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION',
           'NFLAG_INSURED_ON_APPROVAL']

col_cat = ['NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START',
           'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE',
           'CODE_REJECT_REASON', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE',
           'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE',
           'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION']

col_group = ['SK_ID_PREV', 'NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START',
           'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE',
           'CODE_REJECT_REASON', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE',
           'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE',
           'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION']

# =============================================================================
# feature
# =============================================================================
prev = utils.read_pickles('../data/previous_application')

base = prev[[KEY]].drop_duplicates().set_index(KEY)

train = utils.load_train([KEY])
test = utils.load_test([KEY])

def nunique(x):
    return len(set(x))

def multi_gr2(k):
    gr2 = prev.groupby([KEY, k])
    gc.collect()
    print(k)
    keyname = 'gby-'+'-'.join([KEY, k])
    # size
    gr1 = gr2.size().groupby(KEY)
    name = f'{PREF}_{keyname}_size'
    base[f'{name}_min']  = gr1.min()
    base[f'{name}_max']  = gr1.max()
    base[f'{name}_max-min']  = base[f'{name}_max'] - base[f'{name}_min']
    base[f'{name}_mean'] = gr1.mean()
    base[f'{name}_std']  = gr1.std()
    base[f'{name}_sum']  = gr1.sum()
    base[f'{name}_nunique']     = gr1.size()
    for v in col_num:
        
        # min
        gr1 = gr2[v].min().groupby(KEY)
        name = f'{PREF}_{keyname}_{v}_min'
        base[f'{name}_max']     = gr1.max()
        base[f'{name}_mean']    = gr1.mean()
        base[f'{name}_std']     = gr1.std()
        base[f'{name}_sum']     = gr1.sum()
        base[f'{name}_nunique'] = gr1.apply(nunique)
        
        # max
        gr1 = gr2[v].max().groupby(KEY)
        name = f'{PREF}_{keyname}_{v}_max'
        base[f'{name}_min']  = gr1.min()
        base[f'{name}_mean'] = gr1.mean()
        base[f'{name}_std']  = gr1.std()
        base[f'{name}_sum']  = gr1.sum()
        base[f'{name}_nunique'] = gr1.apply(nunique)
        
        # mean
        gr1 = gr2[v].mean().groupby(KEY)
        name = f'{PREF}_{keyname}_{v}_mean'
        base[f'{name}_min']  = gr1.min()
        base[f'{name}_max']  = gr1.max()
        base[f'{name}_max-min']  = base[f'{name}_max'] - base[f'{name}_min']
        base[f'{name}_mean'] = gr1.mean()
        base[f'{name}_std']  = gr1.std()
        base[f'{name}_sum']  = gr1.sum()
        base[f'{name}_nunique'] = gr1.apply(nunique)
        
        # std
        gr1 = gr2[v].std().groupby(KEY)
        name = f'{PREF}_{keyname}_{v}_std'
        base[f'{name}_min']  = gr1.min()
        base[f'{name}_max']  = gr1.max()
        base[f'{name}_max-min']  = base[f'{name}_max'] - base[f'{name}_min']
        base[f'{name}_mean'] = gr1.mean()
        base[f'{name}_std']  = gr1.std()
        base[f'{name}_sum']  = gr1.sum()
        base[f'{name}_nunique'] = gr1.apply(nunique)
        
        # sum
        gr1 = gr2[v].sum().groupby(KEY)
        name = f'{PREF}_{keyname}_{v}_sum'
        base[f'{name}_min']  = gr1.min()
        base[f'{name}_max']  = gr1.max()
        base[f'{name}_max-min']  = base[f'{name}_max'] - base[f'{name}_min']
        base[f'{name}_mean'] = gr1.mean()
        base[f'{name}_std']  = gr1.std()
        base[f'{name}_nunique'] = gr1.apply(nunique)
    
    base.reset_index(inplace=True)
    df = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_pickles(df, f'../data/102_{k}_train', utils.SPLIT_SIZE)
    
    df = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_pickles(df, f'../data/102_{k}_test', utils.SPLIT_SIZE)
    print(f'finish {k}')
    return

# =============================================================================
# gr2
# =============================================================================
pool = Pool(NTHREAD)
callback = pool.map(multi_gr2, col_group)
pool.close()


#==============================================================================
utils.end(__file__)


