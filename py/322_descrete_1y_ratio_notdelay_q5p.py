#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 22:36:17 2018

@author: Kazuki

params:
    {
    day: 1y
    amt: ratio
    delay: False
    descrete_method: quantile, 5%
    }

"""

import numpy as np
import pandas as pd
import gc
import os
from multiprocessing import Pool
#NTHREAD = cpu_count()
import utils
utils.start(__file__)
#==============================================================================
PREF = 'f322_'

KEY = 'SK_ID_CURR'

MONEY_FEATURE = 'amt_ratio'

day_start = -365*1  # min: -2922
day_end   = -365*0  # min: -2922

os.system(f'rm ../feature/t*_{PREF}*')

# =============================================================================
# 
# =============================================================================
def get_trte():
    usecols = ['SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
#    usecols += ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE']
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

df = pd.read_csv('../input/installments_payments.csv.zip')
df = df[df['DAYS_INSTALMENT'].between(day_start, day_end)].sort_values(['SK_ID_PREV', 'SK_ID_CURR', 'DAYS_ENTRY_PAYMENT'])
df = pd.merge(df, get_trte(), on='SK_ID_CURR', how='left')

# app
df['AMT_PAYMENT-d-app_AMT_INCOME_TOTAL'] = df['AMT_PAYMENT'] / df['app_AMT_INCOME_TOTAL']
df['AMT_PAYMENT-d-app_AMT_CREDIT']      = df['AMT_PAYMENT'] / df['app_AMT_CREDIT']
df['AMT_PAYMENT-d-app_AMT_ANNUITY']     = df['AMT_PAYMENT'] / df['app_AMT_ANNUITY']
df['AMT_PAYMENT-d-app_AMT_GOODS_PRICE'] = df['AMT_PAYMENT'] / df['app_AMT_GOODS_PRICE']

df['amt_ratio'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT']
df['delay'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']

df = df[df['delay']<=0]
df.delay = df.delay.astype(int)

df['delay_qcut'] = pd.qcut(df['delay'], 20, duplicates='drop')
df['delay_qcut'] = df['delay_qcut'].map(lambda x: f'{int(x.left)}=<{int(x.right)}')

df['pay_qcut'] = pd.qcut(df[MONEY_FEATURE], 100, duplicates='drop')
df['pay_qcut'] = df['pay_qcut'].map(lambda x: f'{x.left}=<{x.right}')

df['all1'] = 1

feature = pd.pivot_table(df, index='SK_ID_CURR', 
                         columns=['pay_qcut', 'delay_qcut'], values='all1',
                         aggfunc='sum')

feature.columns = pd.Index(['pay'+e[0] + "_delay" + e[1] for e in feature.columns.tolist()])
feature.reset_index(inplace=True)

print(feature.shape)

# =============================================================================
# output
# =============================================================================
train = utils.load_train([KEY])
test = utils.load_test([KEY])

tmp = pd.merge(train, feature, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(tmp.add_prefix(PREF), '../feature/train')

tmp = pd.merge(test, feature, on=KEY, how='left').drop(KEY, axis=1)
utils.to_feature(tmp.add_prefix(PREF),  '../feature/test')



#==============================================================================
utils.end(__file__)


