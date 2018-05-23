#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 17:19:26 2018

@author: kazuki.onodera

previous_application

"""

import numpy as np
import pandas as pd
import utils
utils.start(__file__)
#==============================================================================
KEY = 'SK_ID_CURR'

# =============================================================================
# feature
# =============================================================================
prev = utils.read_pickles('../data/previous_application')

# NAME_CONTRACT_TYPE
df = pd.crosstab(prev[KEY], prev['NAME_CONTRACT_TYPE'])
df.columns = ['prev_sum_'+c.replace(' ', '-') for c in df.columns]


gr = prev.groupby(KEY)

def nunique(x):
    return len(set(x))

# stats
col_numeric = ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 
               'AMT_GOODS_PRICE', 'FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY',
               'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED',
               'DAYS_DECISION', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
               'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION',
               'NFLAG_INSURED_ON_APPROVAL']
for c in col_numeric:
    print(c)
    df[f'{c}_min'] = gr[c].min()
    df[f'{c}_max'] = gr[c].max()
    df[f'{c}_max-min'] = df[f'{c}_max'] - df[f'{c}_min']
    df[f'{c}_mean'] = gr[c].mean()
    df[f'{c}_std'] = gr[c].std()
    df[f'{c}_sum'] = gr[c].sum()
    df[f'{c}_nunique'] = gr[c].apply(nunique)
    


col_cat = ['NAME_CONTRACT_TYPE', 'WEEKDAY_APPR_PROCESS_START',
           'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE',
           'CODE_REJECT_REASON', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE',
           'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE',
           'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION']
for c in col_cat:
    print(c)
    df[f'{c}_nunique'] = gr[c].apply(nunique)
    


df.reset_index(inplace=True)

# =============================================================================
# merge
# =============================================================================

train = utils.load_train([KEY])
train = pd.merge(train, df, on=KEY, how='left')


test = utils.load_test([KEY])
test = pd.merge(test, df, on=KEY, how='left')

utils.to_pickles(train, '../data/102_train', utils.SPLIT_SIZE)
utils.to_pickles(test,  '../data/102_test',  utils.SPLIT_SIZE)



#==============================================================================
utils.end(__file__)


