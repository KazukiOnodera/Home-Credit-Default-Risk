#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 02:50:47 2018

@author: Kazuki
"""

import numpy as np
import pandas as pd
import gc
import os
from multiprocessing import Pool, cpu_count
NTHREAD = cpu_count()
import utils_agg
import utils
#utils.start(__file__)
#==============================================================================
PREF = 'f412_'

KEY = 'SK_ID_CURR'

SHIFT = 1

month   =  12*1

#os.system(f'rm ../feature/t*_{PREF}*')
# =============================================================================
# curr
# =============================================================================
col_app_money = ['app_AMT_INCOME_TOTAL', 'app_AMT_CREDIT', 'app_AMT_ANNUITY', 'app_AMT_GOODS_PRICE']
col_app_day = ['app_DAYS_BIRTH', 'app_DAYS_EMPLOYED', 'app_DAYS_REGISTRATION', 'app_DAYS_ID_PUBLISH', 'app_DAYS_LAST_PHONE_CHANGE']

def get_trte():
    usecols = ['SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
    usecols += ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE']
    rename_di = {
                 'AMT_INCOME_TOTAL':       'app_AMT_INCOME_TOTAL', 
                 'AMT_CREDIT':             'app_AMT_CREDIT', 
                 'AMT_ANNUITY':            'app_AMT_ANNUITY',
                 'AMT_GOODS_PRICE':        'app_AMT_GOODS_PRICE',
                 'DAYS_BIRTH':             'app_DAYS_BIRTH', 
                 'DAYS_EMPLOYED':          'app_DAYS_EMPLOYED', 
                 'DAYS_REGISTRATION':      'app_DAYS_REGISTRATION', 
                 'DAYS_ID_PUBLISH':        'app_DAYS_ID_PUBLISH', 
                 'DAYS_LAST_PHONE_CHANGE': 'app_DAYS_LAST_PHONE_CHANGE',
                 }
    trte = pd.concat([pd.read_csv('../input/application_train.csv.zip', usecols=usecols).rename(columns=rename_di), 
                      pd.read_csv('../input/application_test.csv.zip',  usecols=usecols).rename(columns=rename_di)],
                      ignore_index=True)
    return trte

# =============================================================================
# 
# =============================================================================
df = pd.read_csv('../input/credit_card_balance.csv.zip')
df.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'], inplace=True)
df.reset_index(drop=True, inplace=True)

df = pd.merge(df, get_trte(), on='SK_ID_CURR', how='left')


df['emp_month'] = df['app_DAYS_EMPLOYED']/30

col = ['SK_ID_CURR', 'AMT_BALANCE',
       'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT',
       'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
       'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY',
       'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT',
       'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE',
       'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT',
       'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT',
       'CNT_INSTALMENT_MATURE_CUM', 'NAME_CONTRACT_STATUS', 'SK_DPD',
       'SK_DPD_DEF']

df_before = df.loc[df.MONTHS_BALANCE.between(df['emp_month']-month, df['emp_month']), col]
df_after  = df.loc[df.MONTHS_BALANCE.between(df['emp_month'], df['emp_month']+month), col]


df_before = df_before.groupby(KEY).mean()#.add_prefix('bf_')
df_after  = df_after.groupby(KEY).mean()#.add_prefix('af_')

df_ratio = pd.DataFrame()
for c in df_before.columns:
    df_ratio[c+'_ratio'] = df_before[c] / df_after[c]


df_ratio.reset_index(inplace=True)
df_ratio.replace(np.inf, np.nan, inplace=True)
df_ratio.replace(-np.inf, np.nan, inplace=True)

# =============================================================================
# output
# =============================================================================
train = utils.load_train([KEY])
test = utils.load_test([KEY])

tmp = pd.merge(train, df_ratio, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(tmp.add_prefix(PREF), '../feature/train')

tmp = pd.merge(test, df_ratio, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')




#==============================================================================
utils.end(__file__)

