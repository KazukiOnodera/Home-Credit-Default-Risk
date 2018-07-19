#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 11:11:57 2018

@author: kazuki.onodera

-d- -> /
-x- -> *
-p- -> +
-m- -> -

"""

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
NTHREAD = cpu_count()
from itertools import combinations
import os, utils, gc
utils.start(__file__)
#==============================================================================

folders = ['../data', '../feature', '../feature_unused', 
#           '../feature_var0', '../feature_corr1'
           ]
for fol in folders:
    os.system(f'rm -rf {fol}')
    os.system(f'mkdir {fol}')

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

def prep_prev(df):
    df['AMT_APPLICATION'].replace(0, np.nan, inplace=True)
    df['AMT_CREDIT'].replace(0, np.nan, inplace=True)
    df['CNT_PAYMENT'].replace(0, np.nan, inplace=True)
    df['AMT_DOWN_PAYMENT'].replace(np.nan, 0, inplace=True)
    df.loc[df['NAME_CONTRACT_STATUS']!='Approved', 'AMT_DOWN_PAYMENT'] = np.nan
    df['RATE_DOWN_PAYMENT'].replace(np.nan, 0, inplace=True)
    df.loc[df['NAME_CONTRACT_STATUS']!='Approved', 'RATE_DOWN_PAYMENT'] = np.nan
#    df['xxx'].replace(0, np.nan, inplace=True)
#    df['xxx'].replace(0, np.nan, inplace=True)
    return

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
            
            df['AMT_CREDIT-d-AMT_INCOME_TOTAL']   = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
            df['AMT_ANNUITY-d-AMT_INCOME_TOTAL']  = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
            df['AMT_GOODS_PRICE-d-AMT_INCOME_TOTAL']  = df['AMT_GOODS_PRICE'] / df['AMT_INCOME_TOTAL']
            
            df['AMT_CREDIT-d-AMT_ANNUITY']  = df['AMT_CREDIT'] / df['AMT_ANNUITY'] # how long should user pay?(month)
            df['AMT_GOODS_PRICE-d-AMT_ANNUITY'] = df['AMT_GOODS_PRICE'] / df['AMT_ANNUITY']# how long should user pay?(month)
            df['AMT_GOODS_PRICE-d-AMT_CREDIT']  = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
            df['AMT_GOODS_PRICE-m-AMT_CREDIT']  = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
            
            df['AMT_GOODS_PRICE-m-AMT_CREDIT-d-AMT_INCOME_TOTAL'] = df['AMT_GOODS_PRICE-m-AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
            
            df['age_finish_payment'] = df['DAYS_BIRTH'].abs() + (df['AMT_CREDIT-d-AMT_ANNUITY']*30)
#            df['age_finish_payment'] = (df['DAYS_BIRTH']/-365) + df['credit-d-annuity']
            df.loc[df['DAYS_EMPLOYED']==365243, 'DAYS_EMPLOYED'] = np.nan
            df['DAYS_EMPLOYED-m-DAYS_BIRTH']                 = df['DAYS_EMPLOYED'] - df['DAYS_BIRTH']
            df['DAYS_REGISTRATION-m-DAYS_BIRTH']             = df['DAYS_REGISTRATION'] - df['DAYS_BIRTH']
            df['DAYS_ID_PUBLISH-m-DAYS_BIRTH']               = df['DAYS_ID_PUBLISH'] - df['DAYS_BIRTH']
            df['DAYS_LAST_PHONE_CHANGE-m-DAYS_BIRTH']        = df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_BIRTH']
            df['DAYS_REGISTRATION-m-DAYS_EMPLOYED']          = df['DAYS_REGISTRATION'] - df['DAYS_EMPLOYED']
            df['DAYS_ID_PUBLISH-m-DAYS_EMPLOYED']            = df['DAYS_ID_PUBLISH'] - df['DAYS_EMPLOYED']
            df['DAYS_LAST_PHONE_CHANGE-m-DAYS_EMPLOYED']     = df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_EMPLOYED']
            df['DAYS_ID_PUBLISH-m-DAYS_REGISTRATION']        = df['DAYS_ID_PUBLISH'] - df['DAYS_REGISTRATION']
            df['DAYS_LAST_PHONE_CHANGE-m-DAYS_REGISTRATION'] = df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_REGISTRATION']
            df['DAYS_LAST_PHONE_CHANGE-m-DAYS_ID_PUBLISH']   = df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_ID_PUBLISH']
            
            col = ['DAYS_EMPLOYED-m-DAYS_BIRTH',
                   'DAYS_REGISTRATION-m-DAYS_BIRTH',
                   'DAYS_ID_PUBLISH-m-DAYS_BIRTH',
                   'DAYS_LAST_PHONE_CHANGE-m-DAYS_BIRTH',
                   'DAYS_REGISTRATION-m-DAYS_EMPLOYED',
                   'DAYS_ID_PUBLISH-m-DAYS_EMPLOYED',
                   'DAYS_LAST_PHONE_CHANGE-m-DAYS_EMPLOYED',
                   'DAYS_ID_PUBLISH-m-DAYS_REGISTRATION',
                   'DAYS_LAST_PHONE_CHANGE-m-DAYS_REGISTRATION',
                   'DAYS_LAST_PHONE_CHANGE-m-DAYS_ID_PUBLISH'
                   ]
            col_comb = list(combinations(col, 2))
            
            for i,j in col_comb:
                df[f'{i}-d-{j}'] = df[i] / df[j]
            
            
            df['DAYS_EMPLOYED-d-DAYS_BIRTH']                 = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
            df['DAYS_REGISTRATION-d-DAYS_BIRTH']             = df['DAYS_REGISTRATION'] / df['DAYS_BIRTH']
            df['DAYS_ID_PUBLISH-d-DAYS_BIRTH']               = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
            df['DAYS_LAST_PHONE_CHANGE-d-DAYS_BIRTH']        = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
            df['DAYS_REGISTRATION-d-DAYS_EMPLOYED']          = df['DAYS_REGISTRATION'] / df['DAYS_EMPLOYED']
            df['DAYS_ID_PUBLISH-d-DAYS_EMPLOYED']            = df['DAYS_ID_PUBLISH'] / df['DAYS_EMPLOYED']
            df['DAYS_LAST_PHONE_CHANGE-d-DAYS_EMPLOYED']     = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
            df['DAYS_ID_PUBLISH-d-DAYS_REGISTRATION']        = df['DAYS_ID_PUBLISH'] / df['DAYS_REGISTRATION']
            df['DAYS_LAST_PHONE_CHANGE-d-DAYS_REGISTRATION'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_REGISTRATION']
            df['DAYS_LAST_PHONE_CHANGE-d-DAYS_ID_PUBLISH']   = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_ID_PUBLISH']
            
            df['OWN_CAR_AGE-d-DAYS_BIRTH'] = (df['OWN_CAR_AGE']*(-365)) / df['DAYS_BIRTH']
            df['OWN_CAR_AGE-m-DAYS_BIRTH'] = df['DAYS_BIRTH'] + (df['OWN_CAR_AGE']*365)
            df['OWN_CAR_AGE-d-DAYS_EMPLOYED'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
            df['OWN_CAR_AGE-m-DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'] + (df['OWN_CAR_AGE']*365)
            
            
            df['cnt_adults'] = df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN']
            df['CNT_CHILDREN-d-CNT_FAM_MEMBERS'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
            df['income_per_adult'] = df['AMT_INCOME_TOTAL'] / df['cnt_adults']
#            df.loc[df['CNT_CHILDREN']==0, 'CNT_CHILDREN'] = np.nan
            df['AMT_INCOME_TOTAL-d-CNT_CHILDREN']       = df['AMT_INCOME_TOTAL']  / (df['CNT_CHILDREN']+0.000001)
            df['AMT_CREDIT-d-CNT_CHILDREN']       = df['AMT_CREDIT']        / (df['CNT_CHILDREN']+0.000001)
            df['AMT_ANNUITY-d-CNT_CHILDREN']      = df['AMT_ANNUITY']       / (df['CNT_CHILDREN']+0.000001)
            df['AMT_GOODS_PRICE-d-CNT_CHILDREN']  = df['AMT_GOODS_PRICE']   / (df['CNT_CHILDREN']+0.000001)
            df['AMT_INCOME_TOTAL-d-cnt_adults']       = df['AMT_INCOME_TOTAL']  / df['cnt_adults']
            df['AMT_CREDIT-d-cnt_adults']       = df['AMT_CREDIT']        / df['cnt_adults']
            df['AMT_ANNUITY-d-cnt_adults']      = df['AMT_ANNUITY']       / df['cnt_adults']
            df['AMT_GOODS_PRICE-d-cnt_adults']  = df['AMT_GOODS_PRICE']   / df['cnt_adults']
            df['AMT_INCOME_TOTAL-d-CNT_FAM_MEMBERS'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
            df['AMT_CREDIT-d-CNT_FAM_MEMBERS']       = df['AMT_CREDIT']       / df['CNT_FAM_MEMBERS']
            df['AMT_ANNUITY-d-CNT_FAM_MEMBERS']      = df['AMT_ANNUITY']      / df['CNT_FAM_MEMBERS']
            df['AMT_GOODS_PRICE-d-CNT_FAM_MEMBERS']  = df['AMT_GOODS_PRICE']  / df['CNT_FAM_MEMBERS']
            
            # EXT_SOURCE_x
            df['EXT_SOURCES_prod']  = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
            df['EXT_SOURCES_sum']  = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis=1)
            df['EXT_SOURCES_sum']  = df['EXT_SOURCES_sum'].fillna(df['EXT_SOURCES_sum'].mean())
            df['EXT_SOURCES_mean']  = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
            df['EXT_SOURCES_mean']  = df['EXT_SOURCES_mean'].fillna(df['EXT_SOURCES_mean'].mean())
            df['EXT_SOURCES_std']   = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
            df['EXT_SOURCES_std']   = df['EXT_SOURCES_std'].fillna(df['EXT_SOURCES_std'].mean())
            
            df['EXT_SOURCES_1-2-3']  = df['EXT_SOURCE_1'] - df['EXT_SOURCE_2'] - df['EXT_SOURCE_3']
            df['EXT_SOURCES_2-1-3']  = df['EXT_SOURCE_2'] - df['EXT_SOURCE_1'] - df['EXT_SOURCE_3']
            df['EXT_SOURCES_1-2']    = df['EXT_SOURCE_1'] - df['EXT_SOURCE_2']
            df['EXT_SOURCES_2-3']    = df['EXT_SOURCE_2'] - df['EXT_SOURCE_3']
            df['EXT_SOURCES_1-3']    = df['EXT_SOURCE_1'] - df['EXT_SOURCE_3']
            
            
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
            
            df['alldocs_kurt'] = df[docs].kurtosis(axis=1)
            df['alldocs_skew'] = df[docs].skew(axis=1)
            df['alldocs_mean'] = df[docs].mean(axis=1)
            df['alldocs_sum']  = df[docs].sum(axis=1)
            df['alldocs_std']  = df[docs].std(axis=1)
            df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
            df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
            df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
            df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
            df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
            df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
            df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
            df['NEW_PHONE_TO_EMPLOYED_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
            
            
            # =============================================================================
            # Maxwell features
            # =============================================================================
            bdg_avg  = df.filter(regex='_AVG$').columns
            bdg_mode = df.filter(regex='_MODE$').columns
            bdg_medi = df.filter(regex='_MEDI$').columns[:len(bdg_avg)] # ignore FONDKAPREMONT_MODE...
            
            df['building_score_avg_mean'] = df[bdg_avg].mean(1)
            df['building_score_avg_std']  = df[bdg_avg].std(1)
            df['building_score_avg_sum']  = df[bdg_avg].sum(1)
            
            df['building_score_mode_mean'] = df[bdg_mode].mean(1)
            df['building_score_mode_std']  = df[bdg_mode].std(1)
            df['building_score_mode_sum']  = df[bdg_mode].sum(1)
            
            df['building_score_medi_mean'] = df[bdg_medi].mean(1)
            df['building_score_medi_std']  = df[bdg_medi].std(1)
            df['building_score_medi_sum']  = df[bdg_medi].sum(1)
            
            
            df['maxwell_feature_1'] = (df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']) ** (1 / 2)
            
            
            
            df.replace(np.inf, np.nan, inplace=True) # TODO: any other plan?
            df.replace(-np.inf, np.nan, inplace=True)
            return
        
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
        """
        df = utils.read_pickles('../data/previous_application')
        """
        trte = get_trte()
        
        df = pd.merge(pd.read_csv('../input/previous_application.csv.zip'),
                     trte, on='SK_ID_CURR', how='left')
        prep_prev(df)
        df['FLAG_LAST_APPL_PER_CONTRACT'] = (df['FLAG_LAST_APPL_PER_CONTRACT']=='Y')*1
        
        # day
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
        df['AMT_CREDIT-d-total_debt'] = df['AMT_CREDIT'] / df['total_debt']
        df['AMT_GOODS_PRICE-d-total_debt'] = df['AMT_GOODS_PRICE'] / df['total_debt']
        df['AMT_GOODS_PRICE-d-AMT_CREDIT'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
        
        # app & money
        df['AMT_ANNUITY-d-app_AMT_INCOME_TOTAL']     = df['AMT_ANNUITY']     / df['app_AMT_INCOME_TOTAL']
        df['AMT_APPLICATION-d-app_AMT_INCOME_TOTAL'] = df['AMT_APPLICATION'] / df['app_AMT_INCOME_TOTAL']
        df['AMT_CREDIT-d-app_AMT_INCOME_TOTAL']      = df['AMT_CREDIT']      / df['app_AMT_INCOME_TOTAL']
        df['AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL'] = df['AMT_GOODS_PRICE'] / df['app_AMT_INCOME_TOTAL']
        
        df['AMT_ANNUITY-m-app_AMT_INCOME_TOTAL']     = df['AMT_ANNUITY']     - df['app_AMT_INCOME_TOTAL']
        df['AMT_APPLICATION-m-app_AMT_INCOME_TOTAL'] = df['AMT_APPLICATION'] - df['app_AMT_INCOME_TOTAL']
        df['AMT_CREDIT-m-app_AMT_INCOME_TOTAL']      = df['AMT_CREDIT']      - df['app_AMT_INCOME_TOTAL']
        df['AMT_GOODS_PRICE-m-app_AMT_INCOME_TOTAL'] = df['AMT_GOODS_PRICE'] - df['app_AMT_INCOME_TOTAL']
        
        df['AMT_ANNUITY-d-app_AMT_CREDIT']     = df['AMT_ANNUITY']     / df['app_AMT_CREDIT']
        df['AMT_APPLICATION-d-app_AMT_CREDIT'] = df['AMT_APPLICATION'] / df['app_AMT_CREDIT']
        df['AMT_CREDIT-d-app_AMT_CREDIT']      = df['AMT_CREDIT']      / df['app_AMT_CREDIT']
        df['AMT_GOODS_PRICE-d-app_AMT_CREDIT'] = df['AMT_GOODS_PRICE'] / df['app_AMT_CREDIT']
        
        df['AMT_ANNUITY-m-app_AMT_CREDIT']     = df['AMT_ANNUITY']     - df['app_AMT_CREDIT']
        df['AMT_APPLICATION-m-app_AMT_CREDIT'] = df['AMT_APPLICATION'] - df['app_AMT_CREDIT']
        df['AMT_CREDIT-m-app_AMT_CREDIT']      = df['AMT_CREDIT']      - df['app_AMT_CREDIT']
        df['AMT_GOODS_PRICE-m-app_AMT_CREDIT'] = df['AMT_GOODS_PRICE'] - df['app_AMT_CREDIT']
        
        df['AMT_ANNUITY-m-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL']     = (df['AMT_ANNUITY']     - df['app_AMT_CREDIT']) / df['app_AMT_INCOME_TOTAL']
        df['AMT_APPLICATION-m-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL'] = (df['AMT_APPLICATION'] - df['app_AMT_CREDIT']) / df['app_AMT_INCOME_TOTAL']
        df['AMT_CREDIT-m-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL']      = (df['AMT_CREDIT']      - df['app_AMT_CREDIT']) / df['app_AMT_INCOME_TOTAL']
        df['AMT_GOODS_PRICE-m-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL'] = (df['AMT_GOODS_PRICE'] - df['app_AMT_CREDIT']) / df['app_AMT_INCOME_TOTAL']
        
        
        df['AMT_ANNUITY-d-app_AMT_ANNUITY']     = df['AMT_ANNUITY']     / df['app_AMT_ANNUITY']
        df['AMT_APPLICATION-d-app_AMT_ANNUITY'] = df['AMT_APPLICATION'] / df['app_AMT_ANNUITY']
        df['AMT_CREDIT-d-app_AMT_ANNUITY']      = df['AMT_CREDIT']      / df['app_AMT_ANNUITY']
        df['AMT_GOODS_PRICE-d-app_AMT_ANNUITY'] = df['AMT_GOODS_PRICE'] / df['app_AMT_ANNUITY']
        
        df['AMT_ANNUITY-m-app_AMT_ANNUITY']     = df['AMT_ANNUITY']     - df['app_AMT_ANNUITY']
        df['AMT_APPLICATION-m-app_AMT_ANNUITY'] = df['AMT_APPLICATION'] - df['app_AMT_ANNUITY']
        df['AMT_CREDIT-m-app_AMT_ANNUITY']      = df['AMT_CREDIT']      - df['app_AMT_ANNUITY']
        df['AMT_GOODS_PRICE-m-app_AMT_ANNUITY'] = df['AMT_GOODS_PRICE'] - df['app_AMT_ANNUITY']
        
        df['AMT_ANNUITY-m-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL']     = (df['AMT_ANNUITY']     - df['app_AMT_ANNUITY']) / df['app_AMT_INCOME_TOTAL']
        df['AMT_APPLICATION-m-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL'] = (df['AMT_APPLICATION'] - df['app_AMT_ANNUITY']) / df['app_AMT_INCOME_TOTAL']
        df['AMT_CREDIT-m-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL']      = (df['AMT_CREDIT']      - df['app_AMT_ANNUITY']) / df['app_AMT_INCOME_TOTAL']
        df['AMT_GOODS_PRICE-m-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL'] = (df['AMT_GOODS_PRICE'] - df['app_AMT_ANNUITY']) / df['app_AMT_INCOME_TOTAL']
        
        df['AMT_ANNUITY-d-app_AMT_GOODS_PRICE']     = df['AMT_ANNUITY']     / df['app_AMT_GOODS_PRICE']
        df['AMT_APPLICATION-d-app_AMT_GOODS_PRICE'] = df['AMT_APPLICATION'] / df['app_AMT_GOODS_PRICE']
        df['AMT_CREDIT-d-app_AMT_GOODS_PRICE']      = df['AMT_CREDIT']      / df['app_AMT_GOODS_PRICE']
        df['AMT_GOODS_PRICE-d-app_AMT_GOODS_PRICE'] = df['AMT_GOODS_PRICE'] / df['app_AMT_GOODS_PRICE']
        
        df['AMT_ANNUITY-m-app_AMT_GOODS_PRICE']     = df['AMT_ANNUITY']     - df['app_AMT_GOODS_PRICE']
        df['AMT_APPLICATION-m-app_AMT_GOODS_PRICE'] = df['AMT_APPLICATION'] - df['app_AMT_GOODS_PRICE']
        df['AMT_CREDIT-m-app_AMT_GOODS_PRICE']      = df['AMT_CREDIT']      - df['app_AMT_GOODS_PRICE']
        df['AMT_GOODS_PRICE-m-app_AMT_GOODS_PRICE'] = df['AMT_GOODS_PRICE'] - df['app_AMT_GOODS_PRICE']
        
        df['AMT_ANNUITY-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL']     = (df['AMT_ANNUITY']     - df['app_AMT_GOODS_PRICE']) / df['app_AMT_INCOME_TOTAL']
        df['AMT_APPLICATION-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL'] = (df['AMT_APPLICATION'] - df['app_AMT_GOODS_PRICE']) / df['app_AMT_INCOME_TOTAL']
        df['AMT_CREDIT-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL']      = (df['AMT_CREDIT']      - df['app_AMT_GOODS_PRICE']) / df['app_AMT_INCOME_TOTAL']
        df['AMT_GOODS_PRICE-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL'] = (df['AMT_GOODS_PRICE'] - df['app_AMT_GOODS_PRICE']) / df['app_AMT_INCOME_TOTAL']
        
        # app & day
        col_prev = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 
                    'DAYS_LAST_DUE', 'DAYS_TERMINATION']
        for c1 in col_prev:
            for c2 in col_app_day:
#                print(f"'{c1}-m-{c2}',")
                df[f'{c1}-m-{c2}'] = df[c1] - df[c2]
                df[f'{c1}-d-{c2}'] = df[c1] / df[c2]
        
        df['cnt_paid'] = df.apply(lambda x: min( np.ceil(x['DAYS_FIRST_DUE']/-30), x['CNT_PAYMENT'] ), axis=1)
        df['cnt_paid_ratio'] = df['cnt_paid'] / df['CNT_PAYMENT']
        df['cnt_unpaid'] = df['CNT_PAYMENT'] - df['cnt_paid']
        df['amt_paid'] = df['AMT_ANNUITY'] * df['cnt_paid']
#        df['amt_paid_ratio'] = df['amt_paid'] / df['total_debt'] # same as cnt_paid_ratio
        df['amt_unpaid'] = df['total_debt'] - df['amt_paid']
        
        df['active'] = (df['cnt_unpaid']>0)*1
        df['completed'] = (df['cnt_unpaid']==0)*1
        
        # future payment
        df_tmp = pd.DataFrame()
        print('future payment')
        rem_max = df['cnt_unpaid'].max() # 80
#        rem_max = 1
        df['cnt_unpaid_tmp'] = df['cnt_unpaid']
        for i in range(int( rem_max )):
            c = f'future_payment_{i+1}m'
            df_tmp[c] = df['cnt_unpaid_tmp'].map(lambda x: min(x, 1)) * df['AMT_ANNUITY']
            df_tmp.loc[df_tmp[c]==0, c] = np.nan
            df['cnt_unpaid_tmp'] -= 1
            df['cnt_unpaid_tmp'] = df['cnt_unpaid_tmp'].map(lambda x: max(x, 0))
#        df['prev_future_payment_max'] = df.filter(regex='^prev_future_payment_').max(1)
        
        del df['cnt_unpaid_tmp']
        df = pd.concat([df, df_tmp], axis=1)
        
        
        # past payment
        df_tmp = pd.DataFrame()
        print('past payment')
        rem_max = df['cnt_paid'].max() # 72
        df['cnt_paid_tmp'] = df['cnt_paid']
        for i in range(int( rem_max )):
            c = f'past_payment_{i+1}m'
            df_tmp[c] = df['cnt_paid_tmp'].map(lambda x: min(x, 1)) * df['AMT_ANNUITY']
            df_tmp.loc[df_tmp[c]==0, c] = np.nan
            df['cnt_paid_tmp'] -= 1
            df['cnt_paid_tmp'] = df['cnt_paid_tmp'].map(lambda x: max(x, 0))
#        df['prev_past_payment_max'] = df.filter(regex='^prev_past_payment_').max(1)
        
        del df['cnt_paid_tmp']
        df = pd.concat([df, df_tmp], axis=1)
        
        df['APP_CREDIT_PERC'] = df['AMT_APPLICATION'] / df['AMT_CREDIT']
        
        #df.filter(regex='^amt_future_payment_')
        
        df.replace(np.inf, np.nan, inplace=True) # TODO: any other plan?
        df.replace(-np.inf, np.nan, inplace=True)
        
        utils.to_pickles(df, '../data/previous_application', utils.SPLIT_SIZE)
    
    elif p==2:
        # =============================================================================
        # POS
        # =============================================================================
        """
        df = utils.read_pickles('../data/POS_CASH_balance')
        """
        df = pd.read_csv('../input/POS_CASH_balance.csv.zip')
        
        # data cleansing!!!
        ## drop signed. sample SK_ID_PREV==1769939
        df = df[df.NAME_CONTRACT_STATUS!='Signed']
        
        ## Zombie NAME_CONTRACT_STATUS=='Completed' and CNT_INSTALMENT_FUTURE!=0. 1134377
        df.loc[(df.NAME_CONTRACT_STATUS=='Completed') & (df.CNT_INSTALMENT_FUTURE!=0), 'NAME_CONTRACT_STATUS'] = 'Active'
        
        ## CNT_INSTALMENT_FUTURE=0 and Active. sample SK_ID_PREV==1998905, 2174168
        df.loc[(df.CNT_INSTALMENT_FUTURE==0) & (df.NAME_CONTRACT_STATUS=='Active'), 'NAME_CONTRACT_STATUS'] = 'Completed'
        
        ## remove duplicated CNT_INSTALMENT_FUTURE=0. sample SK_ID_PREV==2601827
        df_0 = df[df['CNT_INSTALMENT_FUTURE']==0]
        df_1 = df[df['CNT_INSTALMENT_FUTURE']>0]
        df_0['NAME_CONTRACT_STATUS'] = 'Completed'
        df_0.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'], ascending=[True, False], inplace=True)
        df_0.drop_duplicates('SK_ID_PREV', keep='last', inplace=True)
        df = pd.concat([df_0, df_1], ignore_index=True)
        del df_0, df_1; gc.collect()
        
        # TODO: end in active. 1002879
#        df['CNT_INSTALMENT_FUTURE_min'] = df.groupby('SK_ID_PREV').CNT_INSTALMENT_FUTURE.transform('min')
#        df['MONTHS_BALANCE_max'] = df.groupby('SK_ID_PREV').MONTHS_BALANCE.transform('max')
#        df.loc[(df.CNT_INSTALMENT_FUTURE_min!=0) & (df.MONTHS_BALANCE_max!=-1)]
        
        df.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'], inplace=True)
        df['CNT_INSTALMENT_FUTURE_diff'] = df[['SK_ID_PREV', 'CNT_INSTALMENT_FUTURE']].groupby('SK_ID_PREV')['CNT_INSTALMENT_FUTURE'].diff()
        df['CNT_INSTALMENT_FUTURE_diff_diff'] = df[['SK_ID_PREV', 'CNT_INSTALMENT_FUTURE_diff']].groupby('SK_ID_PREV')['CNT_INSTALMENT_FUTURE_diff'].diff()
        df['CNT_INSTALMENT_FUTURE_pct_change'] = df[['SK_ID_PREV', 'CNT_INSTALMENT_FUTURE']].groupby('SK_ID_PREV')['CNT_INSTALMENT_FUTURE'].pct_change()
        df['CNT_INSTALMENT-m-CNT_INSTALMENT_FUTURE'] = df['CNT_INSTALMENT'] - df['CNT_INSTALMENT_FUTURE']
        df['CNT_INSTALMENT_FUTURE-d-CNT_INSTALMENT'] = df['CNT_INSTALMENT_FUTURE'] / df['CNT_INSTALMENT']
        
        df['SK_DPD_diff'] = df['SK_DPD'] - df['SK_DPD_DEF']
#        df['SK_DPD_diff_over0'] = (df['SK_DPD_diff']>0)*1
#        df['SK_DPD_diff_over5']  = (df['SK_DPD_diff']>5)*1
#        df['SK_DPD_diff_over10'] = (df['SK_DPD_diff']>10)*1
#        df['SK_DPD_diff_over15'] = (df['SK_DPD_diff']>15)*1
#        df['SK_DPD_diff_over20'] = (df['SK_DPD_diff']>20)*1
#        df['SK_DPD_diff_over25'] = (df['SK_DPD_diff']>25)*1
        
        df.replace(np.inf, np.nan, inplace=True) # TODO: any other plan?
        df.replace(-np.inf, np.nan, inplace=True)
        
        utils.to_pickles(df, '../data/POS_CASH_balance', utils.SPLIT_SIZE)
    
    elif p==3:
        # =============================================================================
        # ins
        # =============================================================================
        """
        df = utils.read_pickles('../data/installments_payments')
        """
        df = pd.read_csv('../input/installments_payments.csv.zip')
        
        trte = get_trte()
        df = pd.merge(df, trte, on='SK_ID_CURR', how='left')
        
        prev = pd.read_csv('../input/previous_application.csv.zip', 
                           usecols=['SK_ID_PREV', 'CNT_PAYMENT', 'AMT_ANNUITY'])
        prev['CNT_PAYMENT'].replace(0, np.nan, inplace=True)
#        prep_prev(prev)
        df = pd.merge(df, prev, on='SK_ID_PREV', how='left')
        
        del trte, prev; gc.collect()
        
        df['month'] = (df['DAYS_ENTRY_PAYMENT']/30).map(np.floor)
        
        # app
        df['DAYS_ENTRY_PAYMENT-m-app_DAYS_BIRTH']             = df['DAYS_ENTRY_PAYMENT'] - df['app_DAYS_BIRTH']
        df['DAYS_ENTRY_PAYMENT-m-app_DAYS_EMPLOYED']          = df['DAYS_ENTRY_PAYMENT'] - df['app_DAYS_EMPLOYED']
        df['DAYS_ENTRY_PAYMENT-m-app_DAYS_REGISTRATION']      = df['DAYS_ENTRY_PAYMENT'] - df['app_DAYS_REGISTRATION']
        df['DAYS_ENTRY_PAYMENT-m-app_DAYS_ID_PUBLISH']        = df['DAYS_ENTRY_PAYMENT'] - df['app_DAYS_ID_PUBLISH']
        df['DAYS_ENTRY_PAYMENT-m-app_DAYS_LAST_PHONE_CHANGE'] = df['DAYS_ENTRY_PAYMENT'] - df['app_DAYS_LAST_PHONE_CHANGE']
        
        df['AMT_PAYMENT-d-app_AMT_INCOME_TOTAL'] = df['AMT_PAYMENT'] / df['app_AMT_INCOME_TOTAL']
        df['AMT_PAYMENT-d-app_AMT_CREDIT']      = df['AMT_PAYMENT'] / df['app_AMT_CREDIT']
        df['AMT_PAYMENT-d-app_AMT_ANNUITY']     = df['AMT_PAYMENT'] / df['app_AMT_ANNUITY']
        df['AMT_PAYMENT-d-app_AMT_GOODS_PRICE'] = df['AMT_PAYMENT'] / df['app_AMT_GOODS_PRICE']
        
        # prev
        df['NUM_INSTALMENT_ratio'] = df['NUM_INSTALMENT_NUMBER'] / df['CNT_PAYMENT']
        df['AMT_PAYMENT-d-AMT_ANNUITY'] = df['AMT_PAYMENT'] / df['AMT_ANNUITY']
        
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
        
        df_tmp = pd.DataFrame()
        for i in range(0, 50, 5):
            c1 = f'delayed_day_over{i}'
            df_tmp[c1] = (df['days_delayed_payment']>i)*1
            
            c2 = f'delayed_money_{i}'
            df_tmp[c2] = df_tmp[c1] * df.AMT_PAYMENT
            
            c3 = f'delayed_money_ratio_{i}'
            df_tmp[c3] = df_tmp[c1] * df.amt_ratio
            
            c1 = f'not-delayed_day_{i}'
            df_tmp[c1] = (df['days_delayed_payment']<=i)*1
            
            c2 = f'not-delayed_money_{i}'
            df_tmp[c2] = df_tmp[c1] * df.AMT_PAYMENT
            
            c3 = f'not-delayed_money_ratio_{i}'
            df_tmp[c3] = df_tmp[c1] * df.amt_ratio
        
        df = pd.concat([df, df_tmp], axis=1)
        
        df.replace(np.inf, np.nan, inplace=True) # TODO: any other plan?
        df.replace(-np.inf, np.nan, inplace=True)
        
        utils.to_pickles(df, '../data/installments_payments', utils.SPLIT_SIZE)
        
        utils.to_pickles(df[df['days_delayed_payment']>0].reset_index(drop=True), 
                         '../data/installments_payments_delay', utils.SPLIT_SIZE)
        
        utils.to_pickles(df[df['days_delayed_payment']<=0].reset_index(drop=True),
                         '../data/installments_payments_notdelay', utils.SPLIT_SIZE)
    
    elif p==4:
        # =============================================================================
        # credit card
        # =============================================================================
        """
        df = utils.read_pickles('../data/credit_card_balance')
        """
        
        df = pd.read_csv('../input/credit_card_balance.csv.zip')
        
        trte = get_trte()
        df = pd.merge(df, trte, on='SK_ID_CURR', how='left')
        
        df[col_app_day] = df[col_app_day]/30
        
        # app
        df['AMT_BALANCE-d-app_AMT_INCOME_TOTAL']    = df['AMT_BALANCE'] / df['app_AMT_INCOME_TOTAL']
        df['AMT_BALANCE-d-app_AMT_CREDIT']          = df['AMT_BALANCE'] / df['app_AMT_CREDIT']
        df['AMT_BALANCE-d-app_AMT_ANNUITY']         = df['AMT_BALANCE'] / df['app_AMT_ANNUITY']
        df['AMT_BALANCE-d-app_AMT_GOODS_PRICE']     = df['AMT_BALANCE'] / df['app_AMT_GOODS_PRICE']
        
        df['AMT_DRAWINGS_CURRENT-d-app_AMT_INCOME_TOTAL']    = df['AMT_DRAWINGS_CURRENT'] / df['app_AMT_INCOME_TOTAL']
        df['AMT_DRAWINGS_CURRENT-d-app_AMT_CREDIT']          = df['AMT_DRAWINGS_CURRENT'] / df['app_AMT_CREDIT']
        df['AMT_DRAWINGS_CURRENT-d-app_AMT_ANNUITY']         = df['AMT_DRAWINGS_CURRENT'] / df['app_AMT_ANNUITY']
        df['AMT_DRAWINGS_CURRENT-d-app_AMT_GOODS_PRICE']     = df['AMT_DRAWINGS_CURRENT'] / df['app_AMT_GOODS_PRICE']
        
        for c in col_app_day:
            print(f'MONTHS_BALANCE-m-{c}')
            df[f'MONTHS_BALANCE-m-{c}'] = df['MONTHS_BALANCE'] - df[c]
        
        
        df['AMT_BALANCE-d-AMT_CREDIT_LIMIT_ACTUAL'] = df['AMT_BALANCE'] / df['AMT_CREDIT_LIMIT_ACTUAL']
        df['AMT_BALANCE-d-AMT_DRAWINGS_CURRENT']    = df['AMT_BALANCE'] / df['AMT_DRAWINGS_CURRENT']
        
        df['AMT_DRAWINGS_CURRENT-d-AMT_CREDIT_LIMIT_ACTUAL'] = df['AMT_DRAWINGS_CURRENT'] / df['AMT_CREDIT_LIMIT_ACTUAL']
        
        df['SK_DPD_diff'] = df['SK_DPD'] - df['SK_DPD_DEF']
        df['SK_DPD_diff_over0'] = (df['SK_DPD_diff']>0)*1
        df['SK_DPD_diff_over5']  = (df['SK_DPD_diff']>5)*1
        df['SK_DPD_diff_over10'] = (df['SK_DPD_diff']>10)*1
        df['SK_DPD_diff_over15'] = (df['SK_DPD_diff']>15)*1
        df['SK_DPD_diff_over20'] = (df['SK_DPD_diff']>20)*1
        df['SK_DPD_diff_over25'] = (df['SK_DPD_diff']>25)*1
        
        
        col = df.columns[3:15]
        df.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'], inplace=True)
        for c in col:
            print(c)
            df[f'{c}_diff'] = df[['SK_ID_PREV', c]].groupby('SK_ID_PREV')[c].diff()
            df[f'{c}_diff_diff'] = df[['SK_ID_PREV', f'{c}_diff']].groupby('SK_ID_PREV')[f'{c}_diff'].diff()
            df[f'{c}_pctchange'] = df[['SK_ID_PREV', c]].groupby('SK_ID_PREV')[c].pct_change()
            df[f'{c}_pctchange_pctchange'] = df[['SK_ID_PREV', f'{c}_pctchange']].groupby('SK_ID_PREV')[f'{c}_pctchange'].pct_change()
        
        df.replace(np.inf, np.nan, inplace=True) # TODO: any other plan?
        df.replace(-np.inf, np.nan, inplace=True)
        
        utils.to_pickles(df, '../data/credit_card_balance', utils.SPLIT_SIZE)
    
    elif p==5:
        # =============================================================================
        # bureau
        # =============================================================================        
        df = pd.read_csv('../input/bureau.csv.zip')
        
        trte = get_trte()
        df = pd.merge(df, trte, on='SK_ID_CURR', how='left')
        
        col_bure_money = ['AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 
                          'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE']
        col_bure_day   = ['DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT']
        
        # app
        for c1 in col_bure_money:
            for c2 in col_app_money:
#                print(f"'{c1}-d-{c2}',")
                df[f'{c1}-d-{c2}'] = df[c1] / df[c2]
                
        for c1 in col_bure_day:
            for c2 in col_app_day:
#                print(f"'{c1}-m-{c2}',")
                df[f'{c1}-m-{c2}'] = df[c1] - df[c2]
                df[f'{c1}-d-{c2}'] = df[c1] / df[c2]
        
        df['DAYS_CREDIT_ENDDATE-m-DAYS_CREDIT'] = df['DAYS_CREDIT_ENDDATE'] - df['DAYS_CREDIT']
        df['DAYS_ENDDATE_FACT-m-DAYS_CREDIT'] = df['DAYS_ENDDATE_FACT'] - df['DAYS_CREDIT']
        df['DAYS_ENDDATE_FACT-m-DAYS_CREDIT_ENDDATE'] = df['DAYS_ENDDATE_FACT'] - df['DAYS_CREDIT_ENDDATE']
        df['DAYS_CREDIT_UPDATE-m-DAYS_CREDIT'] = df['DAYS_CREDIT_UPDATE'] - df['DAYS_CREDIT']
        df['DAYS_CREDIT_UPDATE-m-DAYS_CREDIT_ENDDATE'] = df['DAYS_CREDIT_UPDATE'] - df['DAYS_CREDIT_ENDDATE']
        df['DAYS_CREDIT_UPDATE-m-DAYS_ENDDATE_FACT'] = df['DAYS_CREDIT_UPDATE'] - df['DAYS_ENDDATE_FACT']
        
        df['AMT_CREDIT_SUM-m-AMT_CREDIT_SUM_DEBT'] = df['AMT_CREDIT_SUM'] - df['AMT_CREDIT_SUM_DEBT']
        df['AMT_CREDIT_SUM_DEBT-d-AMT_CREDIT_SUM'] = df['AMT_CREDIT_SUM_DEBT'] / df['AMT_CREDIT_SUM']
        df['AMT_CREDIT_SUM-m-AMT_CREDIT_SUM_DEBT-d-AMT_CREDIT_SUM_LIMIT'] = df['AMT_CREDIT_SUM-m-AMT_CREDIT_SUM_DEBT'] / df['AMT_CREDIT_SUM_LIMIT']
        df['AMT_CREDIT_SUM_DEBT-d-AMT_CREDIT_SUM_LIMIT'] = df['AMT_CREDIT_SUM_DEBT'] / df['AMT_CREDIT_SUM_LIMIT']
        df['AMT_CREDIT_SUM_DEBT-p-AMT_CREDIT_SUM_LIMIT'] = df['AMT_CREDIT_SUM_DEBT'] + df['AMT_CREDIT_SUM_LIMIT']
        df['AMT_CREDIT_SUM-d-debt-p-AMT_CREDIT_SUM_DEBT-p-AMT_CREDIT_SUM_LIMIT'] = df['AMT_CREDIT_SUM'] / df['AMT_CREDIT_SUM_DEBT-p-AMT_CREDIT_SUM_LIMIT']
        
        #df['AMT_CREDIT_SUM-d-days_end-cre'] = df['AMT_CREDIT_SUM'] / df['days_end-m-cre']
        #df['AMT_CREDIT_SUM-d-days_fact-cre'] = df['AMT_CREDIT_SUM'] / df['days_fact-m-cre']
        #df['AMT_CREDIT_SUM-d-days_fact-end'] = df['AMT_CREDIT_SUM'] / df['days_fact-m-end']
        #df['AMT_CREDIT_SUM-d-days_update-cre'] = df['AMT_CREDIT_SUM'] / df['days_update-m-cre']
        #df['AMT_CREDIT_SUM-d-days_update-end'] = df['AMT_CREDIT_SUM'] / df['days_update-m-end']
        
        df.replace(np.inf, np.nan, inplace=True) # TODO: any other plan?
        df.replace(-np.inf, np.nan, inplace=True)
        
        utils.to_pickles(df, '../data/bureau', utils.SPLIT_SIZE)
    
    elif p==6:
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

