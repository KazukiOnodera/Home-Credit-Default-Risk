#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:45:03 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import gc
import utils
utils.start(__file__)
#==============================================================================
KEY = 'SK_ID_CURR'
PREF = 'pos_'


col_num = ['MONTHS_BALANCE', 'CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE',
           'NAME_CONTRACT_STATUS', 'SK_DPD', 'SK_DPD_DEF']

col_cat = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']


# =============================================================================
# feature
# =============================================================================
pos = utils.read_pickles('../data/POS_CASH_balance')
base = pos[[KEY]].drop_duplicates().set_index(KEY)

#### newest ####
for T in range(-1, -6, -1):
    print(T)
    pos_ = pos[pos['MONTHS_BALANCE']==T]
    
    gr = pos_.groupby(KEY)
    base[f'pos-T{T}_size'] = gr.size()
    base[f'pos-T{T}_CNT_INSTALMENT_FUTURE_sum'] = gr['CNT_INSTALMENT_FUTURE'].sum()
    base[f'pos-T{T}_CNT_INSTALMENT_sum']        = gr['CNT_INSTALMENT'].sum()
    base[f'pos-T{T}_CNT_INSTALMENT_ratio'] = base[f'pos-T{T}_CNT_INSTALMENT_FUTURE_sum'] / base[f'pos-T{T}_CNT_INSTALMENT_sum']
    
    c1 = 'NAME_CONTRACT_STATUS'
    df = pd.crosstab(pos_[KEY], pos_[c1])
    df.columns = [f'pos-T{T}_{c2.replace(" ", "-")}_sum' for c2 in df.columns]
    col = df.columns.tolist()
    base = pd.concat([base, df], axis=1)
    base[col] = base[col].fillna(-1)
    
    base['pos-T{T}_SK_DPD_sum']           = gr['SK_DPD'].sum()
    base['pos-T{T}_SK_DPD_DEF_sum']       = gr['SK_DPD_DEF'].sum()
    base['pos-T{T}_CNT_INSTALMENT_ratio'] = base['pos-T{T}_SK_DPD_sum'] / base['pos-T{T}_SK_DPD_DEF_sum']
    
    base.fillna(-1, inplace=True)

#### comp MONTHS_BALANCE ####
comp = pos[pos['NAME_CONTRACT_STATUS']=='Completed']
df = comp.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE']).drop_duplicates('SK_ID_CURR', keep='last')
df.set_index('SK_ID_CURR', inplace=True)
df = df[['MONTHS_BALANCE']]
base['pos-comp_last_month'] = df
base['pos-comp_cnt'] = comp.groupby(KEY).size()

gr = pos.groupby(KEY)

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

for c1 in col_cat:
    gc.collect()
    print(c1)
    df = pd.crosstab(pos[KEY], pos[c1])
    df.columns = [f'{PREF}{c1}_{c2.replace(" ", "-")}_sum' for c2 in df.columns]
    col = df.columns.tolist()
    base = pd.concat([base, df], axis=1)
    base[col] = base[col].fillna(-1)


base.reset_index(inplace=True)

# =============================================================================
# merge
# =============================================================================

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)


test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)

utils.to_pickles(train, '../data/104_train', utils.SPLIT_SIZE)
utils.to_pickles(test,  '../data/104_test',  utils.SPLIT_SIZE)






#==============================================================================
utils.end(__file__)


