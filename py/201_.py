#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 22:07:57 2018

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
pos = utils.read_pickles('../data/POS_CASH_balance')
pos.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'], inplace=True, ascending=[True, False])
# TODO: what if same MONTHS_BALANCE?

base = pos[[KEY]].drop_duplicates().set_index(KEY)


# =============================================================================
# latest
# =============================================================================
latest = pos[pos['MONTHS_BALANCE']==pos.groupby('SK_ID_CURR')['MONTHS_BALANCE'].transform(max)]
c1 = 'NAME_CONTRACT_STATUS'
df_sum = pd.crosstab(latest[KEY], latest[c1])
df_sum.columns = [f'{PREF}_latest_{c1}_{str(c2).replace(" ", "-")}_sum' for c2 in df_sum.columns]
df_norm = pd.crosstab(latest[KEY], latest[c1], normalize='index')
df_norm.columns = [f'{PREF}_latest_{c1}_{str(c2).replace(" ", "-")}_norm' for c2 in df_norm.columns]
df = pd.concat([df_sum, df_norm], axis=1)
col = df.columns.tolist()
base = pd.concat([base, df], axis=1)
base[col] = base[col].fillna(-1)

base[f'{PREF}_latest_CNT_INSTALMENT_min'] = latest.groupby(KEY).CNT_INSTALMENT.min()
base[f'{PREF}_latest_CNT_INSTALMENT_mean'] = latest.groupby(KEY).CNT_INSTALMENT.mean()
base[f'{PREF}_latest_CNT_INSTALMENT_max'] = latest.groupby(KEY).CNT_INSTALMENT.max()
base[f'{PREF}_latest_CNT_INSTALMENT_max-min'] = base[f'{PREF}_latest_CNT_INSTALMENT_max'] - base[f'{PREF}_latest_CNT_INSTALMENT_min']



# =============================================================================
# binary features
# =============================================================================
for i in range(1, 11):
    pos[f'SK_DPD_over{i}'] = (pos.SK_DPD>=i)*1

for c in ['Completed', 'Active', 'Signed', 'Returned to the store',
       'Approved', 'Demand', 'Amortized debt', 'Canceled', 'XNA']:
    pos[f'is_{c}'] = (pos.NAME_CONTRACT_STATUS==c)*1



# =============================================================================
# merge
# =============================================================================
base.reset_index(inplace=True)
if base.columns.duplicated().sum() != 0:
    raise Exception( base.columns[base.columns.duplicated()] )

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)
utils.to_pickles(train, '../data/201_train', utils.SPLIT_SIZE)
del train; gc.collect()


test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)
utils.to_pickles(test,  '../data/201_test',  utils.SPLIT_SIZE)
del test; gc.collect()




#==============================================================================
utils.end(__file__)

