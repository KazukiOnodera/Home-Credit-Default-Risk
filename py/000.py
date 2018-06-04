#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 11:11:57 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import os, utils
utils.start(__file__)
#==============================================================================

os.system('rm -rf ../data')
os.system('mkdir ../data')

df = pd.read_csv('../input/application_train.csv.zip')
utils.to_pickles(df, '../data/train', utils.SPLIT_SIZE)
utils.to_pickles(df[['TARGET']], '../data/label', utils.SPLIT_SIZE)


df = pd.read_csv('../input/application_test.csv.zip')
utils.to_pickles(df, '../data/test', utils.SPLIT_SIZE)
df[['SK_ID_CURR']].to_pickle('../data/sub.p')


df = pd.read_csv('../input/bureau.csv.zip')
utils.to_pickles(df, '../data/bureau', utils.SPLIT_SIZE)

df = pd.read_csv('../input/bureau_balance.csv.zip')
utils.to_pickles(df, '../data/bureau_balance', utils.SPLIT_SIZE)

df = pd.read_csv('../input/credit_card_balance.csv.zip')
utils.to_pickles(df, '../data/credit_card_balance', utils.SPLIT_SIZE)

df = pd.read_csv('../input/installments_payments.csv.zip')
utils.to_pickles(df, '../data/installments_payments', utils.SPLIT_SIZE)

df = pd.read_csv('../input/previous_application.csv.zip')
df['FLAG_LAST_APPL_PER_CONTRACT'] = (df['FLAG_LAST_APPL_PER_CONTRACT']=='Y')*1
for c in ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
    df.loc[df[c]==365243, c] = np.nan
utils.to_pickles(df, '../data/previous_application', utils.SPLIT_SIZE)

df = pd.read_csv('../input/POS_CASH_balance.csv.zip')
utils.to_pickles(df, '../data/POS_CASH_balance', utils.SPLIT_SIZE)

#==============================================================================
utils.end(__file__)

