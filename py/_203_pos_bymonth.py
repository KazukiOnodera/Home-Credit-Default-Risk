#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 02:05:08 2018

@author: Kazuki
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
# load
# =============================================================================
pos = utils.read_pickles('../data/POS_CASH_balance').sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])
base = pos[[KEY]].drop_duplicates().set_index(KEY)


# =============================================================================
# 
# =============================================================================
gr = pos.groupby(KEY)
for i in range(1,6):
    pos[f'SK_DPD_diff{i}']     = gr.SK_DPD.diff(i)
    pos[f'SK_DPD_DEF_diff{i}'] = gr.SK_DPD_DEF.diff(i)
    pos[f'CNT_INSTALMENT_FUTURE_diff{i}'] = gr['CNT_INSTALMENT_FUTURE'].diff(i)


gr1_size = gr.size()
gr2_size = pos.groupby([KEY, 'SK_ID_PREV']).size()

for T in range(-1, -6, -1):
    print(T)
    pos_ = pos[pos['MONTHS_BALANCE']==T]
    
    gr = pos_.groupby(KEY)
    base[f'pos_T{T}_size'] = gr.size()
    base[f'pos_T{T}_CNT_INSTALMENT_FUTURE_sum'] = gr['CNT_INSTALMENT_FUTURE'].sum()
    base[f'pos_T{T}_CNT_INSTALMENT_sum']        = gr['CNT_INSTALMENT'].sum()
    base[f'pos_T{T}_CNT_INSTALMENT_ratio'] = base[f'pos_T{T}_CNT_INSTALMENT_FUTURE_sum'] / base[f'pos_T{T}_CNT_INSTALMENT_sum']
    
    c1 = 'NAME_CONTRACT_STATUS'
    df = pd.crosstab(pos_[KEY], pos_[c1])
    df.columns = [f'pos-T{T}_{c2.replace(" ", "-")}_sum' for c2 in df.columns]
    col = df.columns.tolist()
    base = pd.concat([base, df], axis=1)
    base[col] = base[col].fillna(-1)
    
    base[f'pos_T{T}_SK_DPD_sum']           = gr['SK_DPD'].sum()
    base[f'pos_T{T}_SK_DPD_DEF_sum']       = gr['SK_DPD_DEF'].sum()
    base[f'pos_T{T}_CNT_INSTALMENT_ratio'] = base[f'pos_T{T}_SK_DPD_sum'] / base[f'pos_T{T}_SK_DPD_DEF_sum']
    
#   for j in range(1,6):
    j = 1
    base[f'pos_T{T}_SK_DPD_diff{j}_sum']           = gr[f'SK_DPD_diff{j}'].sum()
    base[f'pos_T{T}_SK_DPD_DEF_diff{j}_sum']       = gr[f'SK_DPD_DEF_diff{j}'].sum()
    base[f'pos_T{T}_CNT_INSTALMENT_diff{j}_ratio'] = base[f'pos_T{T}_SK_DPD_diff{j}_sum'] / base[f'pos_T{T}_SK_DPD_DEF_diff{j}_sum']
    
    base.fillna(-1, inplace=True)


comp = pos[pos['NAME_CONTRACT_STATUS']=='Completed']
df = comp.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE']).drop_duplicates('SK_ID_CURR', keep='last')
df.set_index('SK_ID_CURR', inplace=True)
base['pos_comp_last_month'] = df['MONTHS_BALANCE']
base['pos_comp_cnt'] = comp.groupby(KEY).size()

df = pos[pos['CNT_INSTALMENT_FUTURE']==0].loc[pos['NAME_CONTRACT_STATUS']=='Active']
base[f'{PREF}_CNT_INSTALMENT_FUTURE==0&Active_sum'] = df.groupby(KEY).size()
base[f'{PREF}_CNT_INSTALMENT_FUTURE==0&Active_ratio'] = base[f'{PREF}_CNT_INSTALMENT_FUTURE==0&Active_sum']/gr1_size




# =============================================================================
# merge
# =============================================================================
base.reset_index(inplace=True)
if base.columns.duplicated().sum() != 0:
    raise Exception( base.columns[base.columns.duplicated()] )

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)


test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)

utils.to_pickles(train, '../data/203_train', utils.SPLIT_SIZE)
utils.to_pickles(test,  '../data/203_test',  utils.SPLIT_SIZE)







#==============================================================================
utils.end(__file__)



