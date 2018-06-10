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
os.system('rm -rf ../feature')
os.system('mkdir ../data')
os.system('mkdir ../feature')


df = pd.read_csv('../input/application_train.csv.zip')
df['CODE_GENDER'] = 1 - (df['CODE_GENDER']=='F')*1 # 4 'XNA' are converted to 'M'
df['FLAG_OWN_CAR'] = (df['FLAG_OWN_CAR']=='Y')*1
df['FLAG_OWN_REALTY'] = (df['FLAG_OWN_REALTY']=='Y')*1
df['EMERGENCYSTATE_MODE'] = (df['EMERGENCYSTATE_MODE']=='Yes')*1
utils.to_pickles(df, '../data/train', utils.SPLIT_SIZE)
utils.to_pickles(df[['TARGET']], '../data/label', utils.SPLIT_SIZE)


df = pd.read_csv('../input/application_test.csv.zip')
df['CODE_GENDER'] = 1 - (df['CODE_GENDER']=='F')*1 # no 'XNA'
df['FLAG_OWN_CAR'] = (df['FLAG_OWN_CAR']=='Y')*1
df['FLAG_OWN_REALTY'] = (df['FLAG_OWN_REALTY']=='Y')*1
df['EMERGENCYSTATE_MODE'] = (df['EMERGENCYSTATE_MODE']=='Yes')*1
utils.to_pickles(df, '../data/test', utils.SPLIT_SIZE)
df[['SK_ID_CURR']].to_pickle('../data/sub.p')

# =============================================================================
# prev
# =============================================================================
df = pd.read_csv('../input/previous_application.csv.zip')
df['FLAG_LAST_APPL_PER_CONTRACT'] = (df['FLAG_LAST_APPL_PER_CONTRACT']=='Y')*1
for c in ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 
          'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
    c_ = c + '_without_365243'
    df[c_] = df[c]
    df.loc[df[c_]==365243, c_] = np.nan
df['amt_cre-by-app'] = df['AMT_CREDIT'] / df['AMT_APPLICATION']
df['amt_ann-by-app'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

utils.to_pickles(df, '../data/previous_application', utils.SPLIT_SIZE)


# =============================================================================
# 
# =============================================================================
df = pd.read_csv('../input/POS_CASH_balance.csv.zip')
utils.to_pickles(df, '../data/POS_CASH_balance', utils.SPLIT_SIZE)

# =============================================================================
# bureau
# =============================================================================
df = pd.read_csv('../input/bureau.csv.zip')
df['days_end-cre'] = df['DAYS_CREDIT_ENDDATE'] - df['DAYS_CREDIT']
df['days_fact-cre'] = df['DAYS_ENDDATE_FACT'] - df['DAYS_CREDIT']
df['days_fact-end'] = df['DAYS_ENDDATE_FACT'] - df['DAYS_CREDIT_ENDDATE']
df['days_update-cre'] = df['DAYS_CREDIT_UPDATE'] - df['DAYS_CREDIT']
df['days_update-end'] = df['DAYS_CREDIT_UPDATE'] - df['DAYS_CREDIT_ENDDATE']
df['days_update-fact'] = df['DAYS_CREDIT_UPDATE'] - df['DAYS_ENDDATE_FACT']

df['ant_cre-debt'] = df['AMT_CREDIT_SUM'] - df['AMT_CREDIT_SUM_DEBT']
df['ant_cre-debt-by-limit'] = df['ant_cre-debt'] / df['AMT_CREDIT_SUM_LIMIT']
df['ant_debt-p-limit'] = df['AMT_CREDIT_SUM_DEBT'] + df['AMT_CREDIT_SUM_LIMIT']
df['ant_cre-by-debt-p-limit'] = df['AMT_CREDIT_SUM'] / df['ant_debt-p-limit']

df['AMT_CREDIT_SUM-by-days_end-cre'] = df['AMT_CREDIT_SUM'] / df['days_end-cre']
df['AMT_CREDIT_SUM-by-days_fact-cre'] = df['AMT_CREDIT_SUM'] / df['days_fact-cre']
df['AMT_CREDIT_SUM-by-days_fact-end'] = df['AMT_CREDIT_SUM'] / df['days_fact-end']
df['AMT_CREDIT_SUM-by-days_update-cre'] = df['AMT_CREDIT_SUM'] / df['days_update-cre']
df['AMT_CREDIT_SUM-by-days_update-end'] = df['AMT_CREDIT_SUM'] / df['days_update-end']

utils.to_pickles(df, '../data/bureau', utils.SPLIT_SIZE)


# =============================================================================
# bureau_balance
# =============================================================================
df = pd.read_csv('../input/bureau_balance.csv.zip')
utils.to_pickles(df, '../data/bureau_balance', utils.SPLIT_SIZE)

# =============================================================================
# ins
# =============================================================================
df = pd.read_csv('../input/installments_payments.csv.zip')
df['days_delayed_payment'] = df['DAYS_INSTALMENT'] - df['DAYS_ENTRY_PAYMENT']
df['amt_ratio'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT']
df['amt_delta'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
df['days_weighted_delay'] = df['amt_ratio'] * df['days_delayed_payment']
decay = 0.0003 # decay rate per a day
feature = f'days_weighted_delay_tsw3' # Time Series Weight
df[feature] = df['days_weighted_delay'] * (1 + (df['DAYS_ENTRY_PAYMENT']*decay) )
utils.to_pickles(df, '../data/installments_payments', utils.SPLIT_SIZE)


# =============================================================================
# credit card
# =============================================================================
df = pd.read_csv('../input/credit_card_balance.csv.zip')
utils.to_pickles(df, '../data/credit_card_balance', utils.SPLIT_SIZE)


#==============================================================================
utils.end(__file__)

