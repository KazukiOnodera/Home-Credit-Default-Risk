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
from multiprocessing import Pool, cpu_count
NTHREAD = cpu_count()
import os, utils
utils.start(__file__)
#==============================================================================

os.system('rm -rf ../data')
os.system('rm -rf ../feature')
os.system('rm -rf ../unuse_feature')
os.system('mkdir ../data')
os.system('mkdir ../feature')
os.system('mkdir ../unuse_feature')

def multi(p):
    if p==0:
        # =============================================================================
        # application
        # =============================================================================
        
        def f1(df):
            df['CODE_GENDER'] = 1 - (df['CODE_GENDER']=='F')*1 # 4 'XNA' are converted to 'M'
            df['FLAG_OWN_CAR'] = (df['FLAG_OWN_CAR']=='Y')*1
            df['FLAG_OWN_REALTY'] = (df['FLAG_OWN_REALTY']=='Y')*1
            df['EMERGENCYSTATE_MODE'] = (df['EMERGENCYSTATE_MODE']=='Yes')*1
            
            df['credit-dby-income']       = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
            df['annuity-dby-income']      = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
            df['goods_price-dby-income']  = df['AMT_GOODS_PRICE'] / df['AMT_INCOME_TOTAL']
            
#            df['income-dby-credit']       = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
#            df['income-dby-annuity']      = df['AMT_INCOME_TOTAL'] / df['AMT_ANNUITY']
#            df['income-dby-goods_price']  = df['AMT_INCOME_TOTAL'] / df['AMT_GOODS_PRICE']
            
            df['credit-dby-annuity']      = df['AMT_CREDIT'] / df['AMT_ANNUITY'] # how long should user pay?(year)
            df['goods_price-dby-annuity'] = df['AMT_GOODS_PRICE'] / df['AMT_ANNUITY']# how long should user pay?(year)
            df['goods_price-dby-credit']  = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
            df['goods_price-m-credit']    = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
            
            df['goods_price-m-credit-dby-income'] = df['goods_price-m-credit'] / df['AMT_INCOME_TOTAL']
            
            df['age_finish_payment'] = df['DAYS_BIRTH'].abs() + (df['credit-dby-annuity']*30)
#            df['age_finish_payment'] = (df['DAYS_BIRTH']/-365) + df['credit-dby-annuity']
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
            df['income_per_adult'] = df['AMT_INCOME_TOTAL'] / df['cnt_adults']
            df.loc[df['CNT_CHILDREN']==0, 'CNT_CHILDREN'] = np.nan
            df['income-by-CNT_CHILDREN'] = df['AMT_INCOME_TOTAL']  / df['CNT_CHILDREN']
            df['credit-by-CNT_CHILDREN']       = df['AMT_CREDIT']        / df['CNT_CHILDREN']
            df['annuity-by-CNT_CHILDREN']      = df['AMT_ANNUITY']       / df['CNT_CHILDREN']
            df['goods_price-by-CNT_CHILDREN']  = df['AMT_GOODS_PRICE']   / df['CNT_CHILDREN']
            
            # =========
            # https://www.kaggle.com/jsaguiar/updated-0-792-lb-lightgbm-with-simple-features/code
            # =========
            df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
            df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
            df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
            
            # =========
            # https://www.kaggle.com/poohtls/fork-of-fork-lightgbm-with-simple-features/code
            # =========
            docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
            live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
            inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
            
#            df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
#            df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
            df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
            df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
            df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
            df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
#            df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
            df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
            df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
            df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
            df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
            df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
            df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
            df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
            df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
            df['NEW_PHONE_TO_EMPLOYED_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
#            df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

        
        df = pd.read_csv('../input/application_train.csv.zip')
        f1(df)
        utils.to_pickles(df, '../data/train', utils.SPLIT_SIZE)
        utils.to_pickles(df[['TARGET']], '../data/label', utils.SPLIT_SIZE)
        
        
        df = pd.read_csv('../input/application_test.csv.zip')
        f1(df)
        utils.to_pickles(df, '../data/test', utils.SPLIT_SIZE)
        df[['SK_ID_CURR']].to_pickle('../data/sub.p')
    
    elif p==1:
        # =============================================================================
        # prev
        # =============================================================================
        usecols = ['SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
        rename_di = {'AMT_INCOME_TOTAL': 'app_AMT_INCOME_TOTAL', 
                     'AMT_CREDIT': 'app_AMT_CREDIT', 
                     'AMT_ANNUITY': 'app_AMT_ANNUITY',
                     'AMT_GOODS_PRICE': 'app_AMT_GOODS_PRICE'}
        trte = pd.concat([pd.read_csv('../input/application_train.csv.zip', usecols=usecols).rename(columns=rename_di), 
                          pd.read_csv('../input/application_test.csv.zip',  usecols=usecols).rename(columns=rename_di)],
                          ignore_index=True)
        
        df = pd.merge(pd.read_csv('../input/previous_application.csv.zip'),
                     trte, on='SK_ID_CURR', how='left')
        df['FLAG_LAST_APPL_PER_CONTRACT'] = (df['FLAG_LAST_APPL_PER_CONTRACT']=='Y')*1
        
        for c in ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 
                  'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
            df.loc[df[c]==365243, c] = np.nan
        
        df['days_fdue-m-fdrw'] = df['DAYS_FIRST_DUE'] - df['DAYS_FIRST_DRAWING']
        df['days_ldue1-m-fdrw'] = df['DAYS_LAST_DUE_1ST_VERSION'] - df['DAYS_FIRST_DRAWING']
        df['days_ldue-m-fdrw'] = df['DAYS_LAST_DUE'] - df['DAYS_FIRST_DRAWING'] # total span
        df['days_trm-m-fdrw'] = df['DAYS_TERMINATION'] - df['DAYS_FIRST_DRAWING']
        
        df['days_ldue1-m-fdue'] = df['DAYS_LAST_DUE_1ST_VERSION'] - df['DAYS_FIRST_DUE']
        df['days_ldue-m-fdue'] = df['DAYS_LAST_DUE'] - df['DAYS_FIRST_DUE']
        df['days_trm-m-fdue'] = df['DAYS_TERMINATION'] - df['DAYS_FIRST_DUE']
        
        df['days_ldue-m-ldue1'] = df['DAYS_LAST_DUE'] - df['DAYS_LAST_DUE_1ST_VERSION']
        df['days_trm-m-ldue1'] = df['DAYS_TERMINATION'] - df['DAYS_LAST_DUE_1ST_VERSION']
        
        df['days_trm-m-ldue'] = df['DAYS_TERMINATION'] - df['DAYS_LAST_DUE']
        
        # money
        df['total_debt'] = df['AMT_ANNUITY'] * df['CNT_PAYMENT']
        df['AMT_CREDIT-dby-total_debt'] = df['AMT_CREDIT'] / df['total_debt']
        df['AMT_GOODS_PRICE-dby-total_debt'] = df['AMT_GOODS_PRICE'] / df['total_debt']
        df['AMT_GOODS_PRICE-dby-AMT_CREDIT'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
        
        # app
        df['AMT_ANNUITY-dby-app_AMT_INCOME_TOTAL']     = df['AMT_ANNUITY']     / df['app_AMT_INCOME_TOTAL']
        df['AMT_APPLICATION-dby-app_AMT_INCOME_TOTAL'] = df['AMT_APPLICATION'] / df['app_AMT_INCOME_TOTAL']
        df['AMT_CREDIT-dby-app_AMT_INCOME_TOTAL']      = df['AMT_CREDIT']      / df['app_AMT_INCOME_TOTAL']
        df['AMT_GOODS_PRICE-dby-app_AMT_INCOME_TOTAL'] = df['AMT_GOODS_PRICE'] / df['app_AMT_INCOME_TOTAL']
        
        df['AMT_ANNUITY-dby-app_AMT_CREDIT']     = df['AMT_ANNUITY']     / df['app_AMT_CREDIT']
        df['AMT_APPLICATION-dby-app_AMT_CREDIT'] = df['AMT_APPLICATION'] / df['app_AMT_CREDIT']
        df['AMT_CREDIT-dby-app_AMT_CREDIT']      = df['AMT_CREDIT']      / df['app_AMT_CREDIT']
        df['AMT_GOODS_PRICE-dby-app_AMT_CREDIT'] = df['AMT_GOODS_PRICE'] / df['app_AMT_CREDIT']
        
        df['AMT_ANNUITY-dby-app_AMT_ANNUITY']     = df['AMT_ANNUITY']     / df['app_AMT_ANNUITY']
        df['AMT_APPLICATION-dby-app_AMT_ANNUITY'] = df['AMT_APPLICATION'] / df['app_AMT_ANNUITY']
        df['AMT_CREDIT-dby-app_AMT_ANNUITY']      = df['AMT_CREDIT']      / df['app_AMT_ANNUITY']
        df['AMT_GOODS_PRICE-dby-app_AMT_ANNUITY'] = df['AMT_GOODS_PRICE'] / df['app_AMT_ANNUITY']
        
        df['AMT_ANNUITY-dby-app_AMT_GOODS_PRICE']     = df['AMT_ANNUITY']     / df['app_AMT_GOODS_PRICE']
        df['AMT_APPLICATION-dby-app_AMT_GOODS_PRICE'] = df['AMT_APPLICATION'] / df['app_AMT_GOODS_PRICE']
        df['AMT_CREDIT-dby-app_AMT_GOODS_PRICE']      = df['AMT_CREDIT']      / df['app_AMT_GOODS_PRICE']
        df['AMT_GOODS_PRICE-dby-app_AMT_GOODS_PRICE'] = df['AMT_GOODS_PRICE'] / df['app_AMT_GOODS_PRICE']
        
        
        df['cnt_paid'] = df.apply(lambda x: min( np.ceil(x['DAYS_FIRST_DUE']/-30), x['CNT_PAYMENT'] ), axis=1)
        df['cnt_paid_ratio'] = df['cnt_paid'] / df['CNT_PAYMENT']
        df['cnt_unpaid'] = df['CNT_PAYMENT'] - df['cnt_paid']
        df['amt_paid'] = df['AMT_ANNUITY'] * df['cnt_paid']
#        df['amt_paid_ratio'] = df['amt_paid'] / df['total_debt'] # same as cnt_paid_ratio
        df['amt_unpaid'] = df['total_debt'] - df['amt_paid']
        
        df['active'] = (df['cnt_unpaid']>0)*1
        df['completed'] = (df['cnt_unpaid']==0)*1
        
        # future payment
        rem_max = df['cnt_unpaid'].max() # 80
        df['cnt_unpaid_tmp'] = df['cnt_unpaid']
        for i in range(int( rem_max )):
            c = f'prev_future_payment_{i+1}m'
            df[c] = df['cnt_unpaid_tmp'].map(lambda x: min(x, 1)) * df['AMT_ANNUITY']
            df.loc[df[c]==0, c] = np.nan
            df['cnt_unpaid_tmp'] -= 1
            df['cnt_unpaid_tmp'] = df['cnt_unpaid_tmp'].map(lambda x: max(x, 0))
#        df['prev_future_payment_max'] = df.filter(regex='^prev_future_payment_').max(1)
        
        del df['cnt_unpaid_tmp']
        
        
        # past payment
        rem_max = df['cnt_paid'].max() # 72
        df['cnt_paid_tmp'] = df['cnt_paid']
        for i in range(int( rem_max )):
            c = f'prev_past_payment_{i+1}m'
            df[c] = df['cnt_paid_tmp'].map(lambda x: min(x, 1)) * df['AMT_ANNUITY']
            df.loc[df[c]==0, c] = np.nan
            df['cnt_paid_tmp'] -= 1
            df['cnt_paid_tmp'] = df['cnt_paid_tmp'].map(lambda x: max(x, 0))
#        df['prev_past_payment_max'] = df.filter(regex='^prev_past_payment_').max(1)
        
        del df['cnt_paid_tmp']
        
        df['APP_CREDIT_PERC'] = df['AMT_APPLICATION'] / df['AMT_CREDIT']
        
        #df.filter(regex='^amt_future_payment_')
        
        utils.to_pickles(df, '../data/previous_application', utils.SPLIT_SIZE)
    
    elif p==2:
        # =============================================================================
        # 
        # =============================================================================
        df = pd.read_csv('../input/POS_CASH_balance.csv.zip')
        
        df['CNT_INSTALMENT_diff'] = df['CNT_INSTALMENT'] - df['CNT_INSTALMENT_FUTURE']
        df['CNT_INSTALMENT_ratio'] = df['CNT_INSTALMENT_FUTURE'] / df['CNT_INSTALMENT']
        
        df['SK_DPD_diff'] = df['SK_DPD'] - df['SK_DPD_DEF']
        df['SK_DPD_diff_over0'] = (df['SK_DPD_diff']>0)*1
        df['SK_DPD_diff_over5']  = (df['SK_DPD_diff']>5)*1
        df['SK_DPD_diff_over10'] = (df['SK_DPD_diff']>10)*1
        df['SK_DPD_diff_over15'] = (df['SK_DPD_diff']>15)*1
        df['SK_DPD_diff_over20'] = (df['SK_DPD_diff']>20)*1
        df['SK_DPD_diff_over25'] = (df['SK_DPD_diff']>25)*1
        
        utils.to_pickles(df, '../data/POS_CASH_balance', utils.SPLIT_SIZE)
    
    elif p==3:
        # =============================================================================
        # ins
        # =============================================================================
        df = pd.read_csv('../input/installments_payments.csv.zip')
        df['days_delayed_payment'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
        df['amt_ratio'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT']
        df['amt_delta'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
        df['days_weighted_delay'] = df['amt_ratio'] * df['days_delayed_payment']
        
        # Days past due and days before due (no negative values)
        df['DPD'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
        df['DBD'] = df['DAYS_INSTALMENT'] - df['DAYS_ENTRY_PAYMENT']
        df['DPD'] = df['DPD'].apply(lambda x: x if x > 0 else 0)
        df['DBD'] = df['DBD'].apply(lambda x: x if x > 0 else 0)
        
        decay = 0.0003 # decay rate per a day
        feature = f'days_weighted_delay_tsw3' # Time Series Weight
        df[feature] = df['days_weighted_delay'] * (1 + (df['DAYS_ENTRY_PAYMENT']*decay) )
        
        for i in range(0, 50, 5):
            c1 = f'delayed_day_over{i}'
            df[c1] = (df['days_delayed_payment']>i)*1
            
            c2 = f'delayed_money_{i}'
            df[c2] = df[c1] * df.AMT_PAYMENT
            
            c3 = f'delayed_money_ratio_{i}'
            df[c3] = df[c1] * df.amt_ratio
            
            c1 = f'not-delayed_day_{i}'
            df[c1] = (df['days_delayed_payment']<=i)*1
            
            c2 = f'not-delayed_money_{i}'
            df[c2] = df[c1] * df.AMT_PAYMENT
            
            c3 = f'not-delayed_money_ratio_{i}'
            df[c3] = df[c1] * df.amt_ratio
        
        utils.to_pickles(df, '../data/installments_payments', utils.SPLIT_SIZE)
    
    elif p==4:
        # =============================================================================
        # credit card
        # =============================================================================
        df = pd.read_csv('../input/credit_card_balance.csv.zip')
        utils.to_pickles(df, '../data/credit_card_balance', utils.SPLIT_SIZE)
    
    elif p==5:
        # =============================================================================
        # bureau
        # =============================================================================
        df = pd.read_csv('../input/bureau.csv.zip')
        df['DAYS_CREDIT_ENDDATE-m-DAYS_CREDIT'] = df['DAYS_CREDIT_ENDDATE'] - df['DAYS_CREDIT']
        df['DAYS_ENDDATE_FACT-m-DAYS_CREDIT'] = df['DAYS_ENDDATE_FACT'] - df['DAYS_CREDIT']
        df['DAYS_ENDDATE_FACT-m-DAYS_CREDIT_ENDDATE'] = df['DAYS_ENDDATE_FACT'] - df['DAYS_CREDIT_ENDDATE']
        df['DAYS_CREDIT_UPDATE-m-DAYS_CREDIT'] = df['DAYS_CREDIT_UPDATE'] - df['DAYS_CREDIT']
        df['DAYS_CREDIT_UPDATE-m-DAYS_CREDIT_ENDDATE'] = df['DAYS_CREDIT_UPDATE'] - df['DAYS_CREDIT_ENDDATE']
        df['DAYS_CREDIT_UPDATE-m-DAYS_ENDDATE_FACT'] = df['DAYS_CREDIT_UPDATE'] - df['DAYS_ENDDATE_FACT']
        
        df['AMT_CREDIT_SUM-m-AMT_CREDIT_SUM_DEBT'] = df['AMT_CREDIT_SUM'] - df['AMT_CREDIT_SUM_DEBT']
        df['AMT_CREDIT_SUM_DEBT-dby-AMT_CREDIT_SUM'] = df['AMT_CREDIT_SUM_DEBT'] / df['AMT_CREDIT_SUM']
        df['AMT_CREDIT_SUM-m-AMT_CREDIT_SUM_DEBT-dby-AMT_CREDIT_SUM_LIMIT'] = df['AMT_CREDIT_SUM-m-AMT_CREDIT_SUM_DEBT'] / df['AMT_CREDIT_SUM_LIMIT']
        df['AMT_CREDIT_SUM_DEBT-dby-AMT_CREDIT_SUM_LIMIT'] = df['AMT_CREDIT_SUM_DEBT'] / df['AMT_CREDIT_SUM_LIMIT']
        df['AMT_CREDIT_SUM_DEBT-p-AMT_CREDIT_SUM_LIMIT'] = df['AMT_CREDIT_SUM_DEBT'] + df['AMT_CREDIT_SUM_LIMIT']
        df['AMT_CREDIT_SUM-dby-debt-p-AMT_CREDIT_SUM_DEBT-p-AMT_CREDIT_SUM_LIMIT'] = df['AMT_CREDIT_SUM'] / df['AMT_CREDIT_SUM_DEBT-p-AMT_CREDIT_SUM_LIMIT']
        
        #df['AMT_CREDIT_SUM-by-days_end-cre'] = df['AMT_CREDIT_SUM'] / df['days_end-m-cre']
        #df['AMT_CREDIT_SUM-by-days_fact-cre'] = df['AMT_CREDIT_SUM'] / df['days_fact-m-cre']
        #df['AMT_CREDIT_SUM-by-days_fact-end'] = df['AMT_CREDIT_SUM'] / df['days_fact-m-end']
        #df['AMT_CREDIT_SUM-by-days_update-cre'] = df['AMT_CREDIT_SUM'] / df['days_update-m-cre']
        #df['AMT_CREDIT_SUM-by-days_update-end'] = df['AMT_CREDIT_SUM'] / df['days_update-m-end']
        
        utils.to_pickles(df, '../data/bureau', utils.SPLIT_SIZE)
    
    elif p==5:
        # =============================================================================
        # bureau_balance
        # =============================================================================
        df = pd.read_csv('../input/bureau_balance.csv.zip')
        utils.to_pickles(df, '../data/bureau_balance', utils.SPLIT_SIZE)
    
    else:
        return

# =============================================================================
# main
# =============================================================================
pool = Pool(NTHREAD)
callback = pool.map(multi, range(10))
pool.close()

#==============================================================================
utils.end(__file__)

