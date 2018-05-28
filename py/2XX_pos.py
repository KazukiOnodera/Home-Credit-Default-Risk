#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:48:56 2018

@author: kazuki.onodera


monthごとにサマる


"""

import os
import pandas as pd
import gc
from multiprocessing import Pool
from glob import glob
import utils
utils.start(__file__)
#==============================================================================
KEY = 'SK_ID_CURR'
PREF = 'pos'

NTHREAD = 2

col_num = ['MONTHS_BALANCE', 'CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE',
           'SK_DPD', 'SK_DPD_DEF']

col_cat = ['NAME_CONTRACT_STATUS']

col_group = ['SK_ID_PREV', 'NAME_CONTRACT_STATUS']

# =============================================================================
# feature
# =============================================================================
pos = utils.read_pickles('../data/POS_CASH_balance')
base = pos[[KEY]].drop_duplicates().set_index(KEY)

# =============================================================================
# other features
# =============================================================================
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
    
    base[f'pos-T{T}_SK_DPD_sum']           = gr['SK_DPD'].sum()
    base[f'pos-T{T}_SK_DPD_DEF_sum']       = gr['SK_DPD_DEF'].sum()
    base[f'pos-T{T}_CNT_INSTALMENT_ratio'] = base[f'pos-T{T}_SK_DPD_sum'] / base[f'pos-T{T}_SK_DPD_DEF_sum']
    
    base.fillna(-1, inplace=True)

#### comp MONTHS_BALANCE ####
comp = pos[pos['NAME_CONTRACT_STATUS']=='Completed']
df = comp.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE']).drop_duplicates('SK_ID_CURR', keep='last')
df.set_index('SK_ID_CURR', inplace=True)
df = df[['MONTHS_BALANCE']]
base['pos-comp_last_month'] = df
base['pos-comp_cnt'] = comp.groupby(KEY).size()

if base.columns.duplicated().sum() != 0:
    raise Exception( base.columns[base.columns.duplicated()] )


# =============================================================================
# merge
# =============================================================================
base.reset_index(inplace=True)

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)


test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)

utils.to_pickles(train, '../data/2XX_train', utils.SPLIT_SIZE)
utils.to_pickles(test,  '../data/2XX_test',  utils.SPLIT_SIZE)







#==============================================================================
utils.end(__file__)


