#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 00:23:35 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import os
import utils
utils.start(__file__)
#==============================================================================

PREF = 'prev_101_'

KEY = 'SK_ID_CURR'

os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# 
# =============================================================================
prev = utils.read_pickles('../data/previous_application')

gr = prev.groupby(KEY)

col = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
       'AMT_CREDIT-dby-AMT_ANNUITY', 'DAYS_BIRTH']
#col = ['AMT_CREDIT']
train = utils.load_train([KEY]+col)
test = utils.load_test([KEY]+col)

train.columns = [KEY] + ['app_'+c for c in train.columns[1:]]
test.columns  = [KEY] + ['app_'+c for c in test.columns[1:]]

col_init = train.columns.tolist()

# =============================================================================
# feature
# =============================================================================
base = gr['remain_debt'].sum()
base.name = 'remain_debt_sum'
base = base.to_frame()
#base['remain_debt_mean'] = gr['remain_debt'].mean()

base['remain_year_sum'] = gr['remain_year'].sum()
base['remain_debt_per_year'] = base['remain_debt_sum'] / base['remain_year_sum']

base['active_cnt'] = gr['active'].sum()
base['completed_cnt'] = gr['completed'].sum()

base['remain_year_min']  = gr['remain_year'].min()
base['remain_year_mean'] = gr['remain_year'].mean()
base['remain_year_max']  = gr['remain_year'].max()

base['elapsed_year_min']  = gr['elapsed_year'].min()
base['elapsed_year_mean'] = gr['elapsed_year'].mean()
base['elapsed_year_max']  = gr['elapsed_year'].max()

base['DAYS_DECISION_min'] = gr['DAYS_DECISION'].min()
base['DAYS_DECISION_max'] = gr['DAYS_DECISION'].max()

base['remain_debt_min']  = gr['remain_debt'].min()
base['remain_debt_mean'] = gr['remain_debt'].mean()
base['remain_debt_max']  = gr['remain_debt'].max()


# future payment
col = prev.head().filter(regex='^AMT_ANNUITY_').columns
col_rem = []
for c in col:
    base[f'{c}_sum'] = gr[c].sum()
    col_rem.append(f'{c}_sum')



base.reset_index(inplace=True)

# =============================================================================
# merge
# =============================================================================

def mk_feature(df):
    # future payment
    #rem_max = df['app_AMT_CREDIT-dby-AMT_ANNUITY'].max() # train:45y test:32y
    df['remain_year_curr'] = df['app_AMT_CREDIT-dby-AMT_ANNUITY']
    for i,c in enumerate( col_rem ): # TODO: 45y?
        c1 = f'AMT_ANNUITY_rem-p-app_{i+1}y'
        c2 = f'AMT_ANNUITY_rem-p-app_{i+1}y-dby-AMT_INCOME_TOTAL'
        df[c1] = df[c] + df['remain_year_curr'].map(lambda x: min(x, 1)) * df['app_AMT_ANNUITY']
        df[c2] = df[c1] / df['app_AMT_INCOME_TOTAL']
    #    df.loc[train[c1]==0, c1] = np.nan
        df['remain_year_curr'] -= 1
        df['remain_year_curr'] = df['remain_year_curr'].map(lambda x: max(x, 0))
    
    del df['remain_year_curr']
    
    df['DAYS_DECISION_min-m-DAYS_BIRTH'] = df['DAYS_DECISION_min'] - df['app_DAYS_BIRTH']
    df['DAYS_DECISION_max-m-DAYS_BIRTH'] = df['DAYS_DECISION_max'] - df['app_DAYS_BIRTH']
    df['remain_year_max-p-DAYS_BIRTH'] = (df['remain_year_max'] * 365) - df['app_DAYS_BIRTH']
    df['remain_year_min-p-DAYS_BIRTH'] = (df['remain_year_min'] * 365) - df['app_DAYS_BIRTH']



train2 = pd.merge(train, base, on=KEY, how='left')
train2[col_rem] = train2[col_rem].fillna(0)
train2['debt_sum_rem-p-app'] = train2['app_AMT_CREDIT'] + train2['remain_debt_sum']

mk_feature(train2)





test2 = pd.merge(test, base, on=KEY, how='left')
test2[col_rem] = test2[col_rem].fillna(0)
test2['debt_sum_rem-p-app'] = test2['app_AMT_CREDIT'] + test2['remain_debt_sum']

mk_feature(test2)

# =============================================================================
# output
# =============================================================================
train2.drop(col_init+col_rem, axis=1, inplace=True)
test2.drop(col_init+col_rem, axis=1, inplace=True)
utils.to_feature(train2.add_prefix(PREF), '../feature/train')
utils.to_feature(test2.add_prefix(PREF),  '../feature/test')

#==============================================================================
utils.end(__file__)


