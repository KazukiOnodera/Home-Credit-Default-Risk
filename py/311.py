#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 11:35:25 2018

@author: Kazuki
"""


import numpy as np
import pandas as pd
import gc
import os
from multiprocessing import Pool
#NTHREAD = cpu_count()
import utils_agg
import utils
#utils.start(__file__)
#==============================================================================
PREF = 'f311_'

KEY = 'SK_ID_CURR'

# =============================================================================

df = pd.read_csv('../input/installments_payments.csv.zip',
                 usecols=['SK_ID_PREV', 'SK_ID_CURR', 'DAYS_ENTRY_PAYMENT', 'AMT_PAYMENT'])

df['month'] = (df['DAYS_ENTRY_PAYMENT']/30).map(np.floor)
df = df.groupby([KEY, 'SK_ID_PREV', 'month']).sum().reset_index()
df.drop('DAYS_ENTRY_PAYMENT', axis=1, inplace=True)

df.sort_values(['SK_ID_PREV', 'month'], inplace=True)
df.reset_index(drop=True, inplace=True)



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

df = pd.merge(df, get_trte(), on='SK_ID_CURR', how='left')

prev = pd.read_csv('../input/previous_application.csv.zip', 
                   usecols=['SK_ID_PREV', 'CNT_PAYMENT', 'AMT_ANNUITY'])
prev['CNT_PAYMENT'].replace(0, np.nan, inplace=True)
df = pd.merge(df, prev, on='SK_ID_PREV', how='left')
del prev; gc.collect()






