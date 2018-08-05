#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 15:39:40 2018

@author: Kazuki

TOTAL PAYMENT
ins + cc

"""


import numpy as np
import pandas as pd
import gc
import os
from multiprocessing import Pool
#NTHREAD = cpu_count()
from itertools import combinations
import utils
#utils.start(__file__)
#==============================================================================
PREF = 'f752_'

KEY = 'SK_ID_CURR'

month_start = -12*1 # -96
month_end   = -12*0 # -96

os.system(f'rm ../feature/t*_{PREF}*')

# =============================================================================
# ins
# =============================================================================
df = pd.read_csv('../input/installments_payments.csv.zip',
                 usecols=['SK_ID_CURR', 'DAYS_ENTRY_PAYMENT', 'AMT_PAYMENT'])

df['MONTHS_BALANCE'] = (df['DAYS_ENTRY_PAYMENT']/30).map(np.floor)
df = df[df['MONTHS_BALANCE'].between(month_start, month_end)]

df = df.groupby([KEY, 'MONTHS_BALANCE']).sum().reset_index()
df.drop(['DAYS_ENTRY_PAYMENT'], axis=1, inplace=True)

df.sort_values([KEY, 'MONTHS_BALANCE'], inplace=True)
df.reset_index(drop=True, inplace=True)
ins = df

# =============================================================================
# cc
# =============================================================================
df = utils.read_pickles('../data/credit_card_balance', 
                         ['SK_ID_CURR', 'MONTHS_BALANCE', 'AMT_DRAWINGS_CURRENT',
                          'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_TOTAL_RECEIVABLE'])
df = df[df['MONTHS_BALANCE'].between(month_start, month_end)]
df = df.groupby([KEY, 'MONTHS_BALANCE']).sum().reset_index()
cc = df


# =============================================================================
# 
# =============================================================================

df = pd.merge(ins, cc, on=['SK_ID_CURR', 'MONTHS_BALANCE'], how='outer').fillna(0)
df_ = df[['SK_ID_CURR', 'MONTHS_BALANCE']]

col = ['AMT_PAYMENT', 'AMT_DRAWINGS_CURRENT',
       'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_TOTAL_RECEIVABLE']
col_comb = []
for i in range(2, 5):
    col_comb += list(combinations(col, i))

for c in col_comb:
    df_['-'.join(c)+'_sum'] = df[list(c)].sum(1)
df = df_

num_aggregations = {}
for c in df.columns[2:]:
    num_aggregations[c] = ['min', 'mean', 'max', 'std', 'sum']

train = utils.load_train([KEY])
test = utils.load_test([KEY])

# =============================================================================
# 
# =============================================================================
def aggregate():
    
    df_agg = df.groupby(KEY).agg({**num_aggregations})
    df_agg.columns = pd.Index([e[0] + "_" + e[1] for e in df_agg.columns.tolist()])
    
    col_std = [c for c in df_agg.columns if c.endswith('_std')]
    for c in col_std:
        df_agg[f'{c}-d-mean'] = df_agg[c]/df_agg[c.replace('_std', '_mean')]
    
    df_agg['INSCC_COUNT'] = df.groupby(KEY).size()
    df_agg.reset_index(inplace=True)
    
    tmp = pd.merge(train, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF), '../feature/train')
    
    tmp = pd.merge(test, df_agg, on=KEY, how='left').drop(KEY, axis=1)
    utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')
    
    return


# =============================================================================
# main
# =============================================================================

aggregate()



#==============================================================================
utils.end(__file__)



