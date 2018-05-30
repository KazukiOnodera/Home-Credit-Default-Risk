#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 22:01:36 2018

@author: kazuki.onodera
"""
import pandas as pd
import gc
from glob import glob
from multiprocessing import Pool
from itertools import combinations
import utils
utils.start(__file__)
#==============================================================================

KEY = 'SK_ID_CURR'
PREF = 'prev'
NTHREAD = 15

col_cat = ['NAME_CONTRACT_TYPE', 
           'FLAG_LAST_APPL_PER_CONTRACT',
           'NFLAG_LAST_APPL_IN_DAY',
           'NAME_CASH_LOAN_PURPOSE', 
           'NAME_CONTRACT_STATUS', 
           'NAME_PAYMENT_TYPE',
           'CODE_REJECT_REASON', 
           'NAME_TYPE_SUITE', 
           'NAME_CLIENT_TYPE',
           'NAME_GOODS_CATEGORY',
           'NAME_PORTFOLIO', 
           'NAME_PRODUCT_TYPE',
           'CHANNEL_TYPE', 
           'NAME_SELLER_INDUSTRY', 
           'NAME_YIELD_GROUP', 
           'PRODUCT_COMBINATION']


col_cat_comb = list(combinations(col_cat, 2))

# =============================================================================
# feature
# =============================================================================
prev = utils.read_pickles('../data/previous_application')

base = prev[[KEY]].drop_duplicates().set_index(KEY)

col_cat = []
for cc in col_cat_comb:
    c1, c2 = cc
    prev[f'{c1}-{c2}'] = prev[c1].map(str)+'-'+prev[c2].map(str)
    col_cat.append(f'{c1}-{c2}')

train = utils.load_train([KEY])
test = utils.load_test([KEY])

# =============================================================================
# cat
# =============================================================================

def multi(c1):
    global base
    print(c1)
    df_sum = pd.crosstab(prev[KEY], prev[c1])
    df_sum.columns = [f'{PREF}_{c1}_{str(c2).replace(" ", "-")}_sum' for c2 in df_sum.columns]
    df_norm = pd.crosstab(prev[KEY], prev[c1], normalize='index')
    df_norm.columns = [f'{PREF}_{c1}_{str(c2).replace(" ", "-")}_norm' for c2 in df_norm.columns]
    df = pd.concat([df_sum, df_norm], axis=1)
    
    col = df.columns.tolist()
    base = pd.concat([base, df], axis=1)
    base[col] = base[col].fillna(-1)
    base.reset_index(inplace=True)
    
    df = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_pickles(df, f'../data/110_{c1}_train', utils.SPLIT_SIZE)
    gc.collect()
    
    df = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_pickles(df,  f'../data/110_{c1}_test',  utils.SPLIT_SIZE)
    gc.collect()

# =============================================================================
# 
# =============================================================================
pool = Pool(NTHREAD)
callback = pool.map(multi, col_cat)
pool.close()


#==============================================================================
utils.end(__file__)


