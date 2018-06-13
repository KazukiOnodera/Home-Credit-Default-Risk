#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 11:11:57 2018

@author: kazuki.onodera

-dby- -> devide by
-x- -> *
-p- -> +
-m- -> -

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


# =============================================================================
# application
# =============================================================================
df = pd.read_csv('../input/application_train.csv.zip')

def f1(df):
    df['CODE_GENDER'] = 1 - (df['CODE_GENDER']=='F')*1 # 4 'XNA' are converted to 'M'
    df['FLAG_OWN_CAR'] = (df['FLAG_OWN_CAR']=='Y')*1
    df['FLAG_OWN_REALTY'] = (df['FLAG_OWN_REALTY']=='Y')*1
    df['EMERGENCYSTATE_MODE'] = (df['EMERGENCYSTATE_MODE']=='Yes')*1
    
    df['AMT_CREDIT-dby-AMT_INCOME_TOTAL'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['AMT_ANNUITY-dby-AMT_INCOME_TOTAL'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['AMT_GOODS_PRICE-dby-AMT_INCOME_TOTAL'] = df['AMT_GOODS_PRICE'] / df['AMT_INCOME_TOTAL']
    df['AMT_CREDIT-dby-AMT_ANNUITY']    = df['AMT_CREDIT'] / df['AMT_ANNUITY'] # how long should user pay?(year)
    df['AMT_GOODS_PRICE-dby-AMT_ANNUITY']    = df['AMT_GOODS_PRICE'] / df['AMT_ANNUITY']# how long should user pay?(year)
    df['AMT_GOODS_PRICE-dby-AMT_CREDIT']    = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    df['AMT_GOODS_PRICE-m-AMT_CREDIT']    = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
    
    df['AMT_GOODS_PRICE-m-AMT_CREDIT-dby-AMT_INCOME_TOTAL'] = df['AMT_GOODS_PRICE-m-AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    df['age'] = df['DAYS_BIRTH'] / -365
    df['AMT_CREDIT-dby-AMT_ANNUITY-p-age'] = df['AMT_CREDIT-dby-AMT_ANNUITY'] + df['age'] # age when user finish the loan
    df.loc[df['DAYS_EMPLOYED']==365243, 'DAYS_EMPLOYED'] = np.nan
    df['DAYS_EMPLOYED-m-DAYS_BIRTH']            = df['DAYS_EMPLOYED'] - df['DAYS_BIRTH']
    df['DAYS_REGISTRATION-m-DAYS_BIRTH']        = df['DAYS_REGISTRATION'] - df['DAYS_BIRTH']
    df['DAYS_ID_PUBLISH-m-DAYS_BIRTH']          = df['DAYS_ID_PUBLISH'] - df['DAYS_BIRTH']
    df['DAYS_LAST_PHONE_CHANGE-m-DAYS_BIRTH']   = df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_BIRTH']
    
    df['DAYS_REGISTRATION-m-DAYS_EMPLOYED']        = df['DAYS_REGISTRATION'] - df['DAYS_EMPLOYED']
    df['DAYS_ID_PUBLISH-m-DAYS_EMPLOYED']          = df['DAYS_ID_PUBLISH'] - df['DAYS_EMPLOYED']
    df['DAYS_LAST_PHONE_CHANGE-m-DAYS_EMPLOYED']   = df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_EMPLOYED']
    
    df['DAYS_ID_PUBLISH-m-DAYS_REGISTRATION']          = df['DAYS_ID_PUBLISH'] - df['DAYS_REGISTRATION']
    df['DAYS_LAST_PHONE_CHANGE-m-DAYS_REGISTRATION']   = df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_REGISTRATION']
    
    df['DAYS_LAST_PHONE_CHANGE-m-DAYS_ID_PUBLISH']   = df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_ID_PUBLISH']
    
    df['cnt_adults'] = df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN']
    df.loc[df['CNT_CHILDREN']==0, 'CNT_CHILDREN'] = np.nan
    df['AMT_INCOME_TOTAL-by-CNT_CHILDREN'] = df['AMT_INCOME_TOTAL']  / df['CNT_CHILDREN']
    df['AMT_CREDIT-by-CNT_CHILDREN']       = df['AMT_CREDIT']        / df['CNT_CHILDREN']
    df['AMT_ANNUITY-by-CNT_CHILDREN']      = df['AMT_ANNUITY']       / df['CNT_CHILDREN']
    df['AMT_GOODS_PRICE-by-CNT_CHILDREN']  = df['AMT_GOODS_PRICE']   / df['CNT_CHILDREN']

f1(df)
utils.to_pickles(df, '../data/train', utils.SPLIT_SIZE)
utils.to_pickles(df[['TARGET']], '../data/label', utils.SPLIT_SIZE)


df = pd.read_csv('../input/application_test.csv.zip')
f1(df)
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

df['days_fdue-m-fdrw'] = df['DAYS_FIRST_DUE_without_365243'] - df['DAYS_FIRST_DRAWING_without_365243']
df['days_ldue1-m-fdrw'] = df['DAYS_LAST_DUE_1ST_VERSION_without_365243'] - df['DAYS_FIRST_DRAWING_without_365243']
df['days_ldue-m-fdrw'] = df['DAYS_LAST_DUE_without_365243'] - df['DAYS_FIRST_DRAWING_without_365243']
df['days_trm-m-fdrw'] = df['DAYS_TERMINATION_without_365243'] - df['DAYS_FIRST_DRAWING_without_365243']

df['days_ldue1-m-fdue'] = df['DAYS_LAST_DUE_1ST_VERSION_without_365243'] - df['DAYS_FIRST_DUE_without_365243']
df['days_ldue-m-fdue'] = df['DAYS_LAST_DUE_without_365243'] - df['DAYS_FIRST_DUE_without_365243']
df['days_trm-m-fdue'] = df['DAYS_TERMINATION_without_365243'] - df['DAYS_FIRST_DUE_without_365243']

df['days_ldue-m-ldue1'] = df['DAYS_LAST_DUE_without_365243'] - df['DAYS_LAST_DUE_1ST_VERSION_without_365243']
df['days_trm-m-ldue1'] = df['DAYS_TERMINATION_without_365243'] - df['DAYS_LAST_DUE_1ST_VERSION_without_365243']

df['days_trm-m-ldue'] = df['DAYS_TERMINATION_without_365243'] - df['DAYS_LAST_DUE_without_365243']


df['AMT_APPLICATION-dby-AMT_ANNUITY'] = df['AMT_APPLICATION'] / df['AMT_ANNUITY']
df['AMT_CREDIT-dby-AMT_ANNUITY'] = df['AMT_CREDIT'] / df['AMT_ANNUITY'] # year
df['AMT_DOWN_PAYMENT-dby-AMT_APPLICATION'] = df['AMT_DOWN_PAYMENT'] / df['AMT_APPLICATION']
df['AMT_CREDIT-dby-AMT_APPLICATION'] = df['AMT_CREDIT'] / df['AMT_APPLICATION']

df['remain_year'] = df['AMT_CREDIT-dby-AMT_ANNUITY'] + (df['DAYS_FIRST_DUE']/365) # TODO: DAYS_FIRST_DUE?
df['remain_debt'] = df['remain_year'] * df['AMT_ANNUITY']
df.loc[df['remain_debt']<0, 'remain_debt'] = np.nan # TODO: np.nan?



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

