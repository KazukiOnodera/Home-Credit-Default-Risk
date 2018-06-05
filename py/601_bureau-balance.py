#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 20:43:28 2018

@author: kazuki.onodera
"""

import os
import pandas as pd
import gc
from multiprocessing import Pool
from glob import glob
import utils
utils.start(__file__)
#==============================================================================
KEY = 'SK_ID_BUREAU'
PREF = 'bb_601'

bb = utils.read_pickles('../data/bureau_balance')

c1 = 'STATUS'
df_sum = pd.crosstab(bb[KEY], bb[c1])
df_sum.columns = [f'{PREF}_{c1}_{str(c2).replace(" ", "-")}_sum' for c2 in df_sum.columns]
df_norm = pd.crosstab(bb[KEY], bb[c1], normalize='index')
df_norm.columns = [f'{PREF}_{c1}_{str(c2).replace(" ", "-")}_norm' for c2 in df_norm.columns]
df = pd.concat([df_sum, df_norm], axis=1)

# month STATUS
col_cat = []
for i in (range(0, -5, -1)):
    tmp = bb[bb.MONTHS_BALANCE==i].set_index(KEY).drop('MONTHS_BALANCE', axis=1)
    df[f'{PREF}_MONTH_{i}'] = tmp['STATUS']
    df[f'{PREF}_MONTH_{i}'].fillna('-1', inplace=True)
    col_cat.append(f'{PREF}_MONTH_{i}')


# STATUS stats
gr = bb.groupby(KEY)
df[f'{PREF}_MONTH_min'] = gr.MONTHS_BALANCE.min()
df[f'{PREF}_MONTH_max'] = gr.MONTHS_BALANCE.max()
df[f'{PREF}_MONTH_min-max'] = df[f'{PREF}_MONTH_min'] - df[f'{PREF}_MONTH_max']

df.reset_index(inplace=True)



# merge
bre = utils.read_pickles('../data/bureau',  ['SK_ID_CURR', KEY])

merged = pd.merge(bre, df, on=KEY, how='left')

merged[col_cat] = merged[col_cat].fillna('-2')

# =============================================================================
# SK_ID_CURR
# =============================================================================
KEY = 'SK_ID_CURR'
base = merged[[KEY]].drop_duplicates().set_index(KEY)

col_num = ['bb_STATUS_0_sum', 'bb_STATUS_1_sum',
           'bb_STATUS_2_sum', 'bb_STATUS_3_sum', 'bb_STATUS_4_sum',
           'bb_STATUS_5_sum', 'bb_STATUS_C_sum', 'bb_STATUS_X_sum',
           'bb_STATUS_0_norm', 'bb_STATUS_1_norm', 'bb_STATUS_2_norm',
           'bb_STATUS_3_norm', 'bb_STATUS_4_norm', 'bb_STATUS_5_norm',
           'bb_STATUS_C_norm', 'bb_STATUS_X_norm',
           'bb_MONTH_min', 'bb_MONTH_max', 'bb_MONTH_min-max'
           ]

col_cat = ['bb_MONTH_0', 'bb_MONTH_-1', 'bb_MONTH_-2', 'bb_MONTH_-3', 'bb_MONTH_-4']


def nunique(x):
    return len(set(x))

# =============================================================================
# gr1
# =============================================================================
gr = merged.groupby(KEY)

# stats
keyname = 'gby-'+KEY
base[f'{PREF}_{keyname}_size'] = gr.size()
for c in col_num:
    gc.collect()
    print(c)
    base[f'{PREF}_{keyname}_{c}_min'] = gr[c].min()
    base[f'{PREF}_{keyname}_{c}_max'] = gr[c].max()
    base[f'{PREF}_{keyname}_{c}_max-min'] = base[f'{PREF}_{keyname}_{c}_max'] - base[f'{PREF}_{keyname}_{c}_min']
    base[f'{PREF}_{keyname}_{c}_mean'] = gr[c].mean()
    base[f'{PREF}_{keyname}_{c}_std'] = gr[c].std()
    base[f'{PREF}_{keyname}_{c}_sum'] = gr[c].sum()
    base[f'{PREF}_{keyname}_{c}_nunique'] = gr[c].apply(nunique)


# =============================================================================
# cat
# =============================================================================
for c1 in col_cat:
    gc.collect()
    print(c1)
    df_sum = pd.crosstab(merged[KEY], merged[c1])
    df_sum.columns = [f'{c1}_{str(c2).replace(" ", "-")}_sum' for c2 in df_sum.columns]
    df_norm = pd.crosstab(merged[KEY], merged[c1], normalize='index')
    df_norm.columns = [f'{c1}_{str(c2).replace(" ", "-")}_norm' for c2 in df_norm.columns]
    df = pd.concat([df_sum, df_norm], axis=1)
    col = df.columns.tolist()
    base = pd.concat([base, df], axis=1)
    base[col] = base[col].fillna(-1)




# =============================================================================
# merge
# =============================================================================
base.reset_index(inplace=True)

train = utils.load_train([KEY])
train = pd.merge(train, base, on=KEY, how='left').drop(KEY, axis=1)


test = utils.load_test([KEY])
test = pd.merge(test, base, on=KEY, how='left').drop(KEY, axis=1)

utils.to_pickles(train, '../data/601_train', utils.SPLIT_SIZE)
utils.to_pickles(test,  '../data/601_test',  utils.SPLIT_SIZE)







#==============================================================================
utils.end(__file__)


