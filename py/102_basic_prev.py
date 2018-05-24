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
import utils
utils.start(__file__)
#==============================================================================

KEY = 'SK_ID_CURR'
PREF = 'prev_'

col_num = ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 
           'AMT_GOODS_PRICE', 'FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY',
           'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED',
           'DAYS_DECISION', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
           'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION',
           'NFLAG_INSURED_ON_APPROVAL']

col_cat = ['NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START',
           'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE',
           'CODE_REJECT_REASON', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE',
           'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE',
           'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION']

# =============================================================================
# feature
# =============================================================================
prev = utils.read_pickles('../data/previous_application')

base = prev[[KEY]].drop_duplicates().set_index(KEY)

gr = prev.groupby(KEY)

def nunique(x):
    return len(set(x))

# stats
for c in col_num:
    gc.collect()
    print(c)
    base[f'{PREF}{c}_min'] = gr[c].min()
    base[f'{PREF}{c}_max'] = gr[c].max()
    base[f'{PREF}{c}_max-min'] = base[f'{PREF}{c}_max'] - base[f'{PREF}{c}_min']
    base[f'{PREF}{c}_mean'] = gr[c].mean()
    base[f'{PREF}{c}_std'] = gr[c].std()
    base[f'{PREF}{c}_sum'] = gr[c].sum()
    base[f'{PREF}{c}_nunique'] = gr[c].apply(nunique)
    

for c in col_cat:
    gc.collect()
    print(c)
    df = pd.crosstab(prev[KEY], prev[c])
    df.columns = ['{PREF}'+c.replace(' ', '-')+'_sum' for c in df.columns]
    base = pd.concat([base, df])


base.reset_index(inplace=True)

# =============================================================================
# merge
# =============================================================================

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left')


test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left')

utils.to_pickles(train, '../data/102_train', utils.SPLIT_SIZE)
utils.to_pickles(test,  '../data/102_test',  utils.SPLIT_SIZE)



#==============================================================================
utils.end(__file__)


