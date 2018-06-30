#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 21:47:53 2018

@author: kazuki.onodera
"""

import gc
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append('/home/kazuki_onodera/Python')
import lgbmextension as ex
import lightgbm as lgb
import multiprocessing
from glob import glob
import count
import os
import utils
#utils.start(__file__)
#==============================================================================

SEED = 71

param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.01,
         'max_depth': -1,
         'num_leaves': 255,
         'max_bin': 255,
         'colsample_bytree': 0.9,
         'subsample': 0.9,
         'nthread': multiprocessing.cpu_count(),
         'bagging_freq': 1,
#         'verbose':-1,
         'seed': SEED
         }

use_files = ['train_other_']

# =============================================================================
# set features
# =============================================================================

files = utils.get_use_files(use_files, True)


X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET


if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

# =============================================================================
# lgb
# =============================================================================

dtrain = lgb.Dataset(X, y)
model = lgb.train(param, dtrain, 500)

imp = ex.getImp(model)
col = imp[imp['split']==0]['index'].tolist()
for c in col:
    os.system(f'touch "../unused_feature/{c}.f"')

features_bure = imp[imp['split']!=0]['index'].tolist()

best_score = 0

features_best = ['app_001_AMT_ANNUITY', 'app_001_AMT_CREDIT', 'app_001_AMT_GOODS_PRICE', 'app_001_APARTMENTS_AVG', 'app_001_CODE_GENDER', 'app_001_COMMONAREA_AVG', 'app_001_DAYS_BIRTH', 'app_001_DAYS_EMPLOYED', 'app_001_DAYS_EMPLOYED-m-DAYS_BIRTH', 'app_001_DAYS_ID_PUBLISH', 'app_001_DAYS_ID_PUBLISH-m-DAYS_BIRTH', 'app_001_DAYS_LAST_PHONE_CHANGE', 'app_001_DAYS_REGISTRATION-m-DAYS_BIRTH', 'app_001_DEF_30_CNT_SOCIAL_CIRCLE', 'app_001_ENTRANCES_MEDI', 'app_001_EXT_SOURCE_1', 'app_001_EXT_SOURCE_2', 'app_001_EXT_SOURCE_3', 'app_001_FLAG_DOCUMENT_3', 'app_001_FLAG_DOCUMENT_5', 'app_001_FLAG_DOCUMENT_6', 'app_001_FLAG_OWN_CAR', 'app_001_FLAG_WORK_PHONE', 'app_001_NAME_CONTRACT_TYPE', 'app_001_NAME_EDUCATION_TYPE', 'app_001_NAME_INCOME_TYPE', 'app_001_NEW_DOC_IND_KURT', 'app_001_ORGANIZATION_TYPE', 'app_001_OWN_CAR_AGE', 'app_001_REGION_RATING_CLIENT', 'app_001_REGION_RATING_CLIENT_W_CITY', 'app_001_YEARS_BUILD_MEDI', 'app_001_annuity-dby-income', 'app_001_cnt_adults', 'app_001_credit-dby-annuity', 'app_001_credit-dby-income', 'app_001_goods_price-by-CNT_CHILDREN', 'app_001_goods_price-dby-annuity', 'app_001_goods_price-m-credit', 'app_001_goods_price-m-credit-dby-income', 'app_001_income-by-CNT_CHILDREN', 'app_001_income_per_adult', 'bure_501_AMT_CREDIT_MAX_OVERDUE_mean', 'bure_501_AMT_CREDIT_MAX_OVERDUE_sum', 'bure_501_AMT_CREDIT_SUM-m-AMT_CREDIT_SUM_DEBT_min', 'bure_501_AMT_CREDIT_SUM_DEBT-dby-AMT_CREDIT_SUM_max', 'bure_501_AMT_CREDIT_SUM_DEBT-dby-AMT_CREDIT_SUM_mean', 'bure_501_AMT_CREDIT_SUM_DEBT-dby-AMT_CREDIT_SUM_var', 'bure_501_DAYS_CREDIT_ENDDATE_max', 'bure_501_DAYS_CREDIT_UPDATE_max', 'bure_501_DAYS_CREDIT_UPDATE_min', 'bure_501_DAYS_ENDDATE_FACT_var', 'bure_501_credit-dby-annuity_mean', 'bure_503_DAYS_CREDIT_max', 'bure_505_Active_AMT_CREDIT_SUM_DEBT-dby-AMT_CREDIT_SUM_max', 'bure_505_Active_AMT_CREDIT_SUM_DEBT-dby-AMT_CREDIT_SUM_mean', 'bure_505_Active_AMT_CREDIT_SUM_sum', 'bure_505_Active_DAYS_CREDIT_ENDDATE-m-DAYS_CREDIT_min', 'bure_505_Active_DAYS_CREDIT_mean', 'bure_505_Closed_DAYS_CREDIT_ENDDATE-m-DAYS_CREDIT_min', 'bure_505_Closed_DAYS_CREDIT_UPDATE_mean', 'bure_505_Closed_credit-dby-income_mean', 'cre_401_AMT_PAYMENT_CURRENT_sum', 'cre_401_CNT_DRAWINGS_CURRENT_var', 'cre_402_AMT_BALANCE-dby-AMT_CREDIT_LIMIT_ACTUAL_max', 'ins_301_AMT_PAYMENT_sum', 'ins_301_DBD_max', 'ins_301_NUM_INSTALMENT_ratio_mean', 'ins_301_NUM_INSTALMENT_ratio_var', 'ins_301_days_weighted_delay_mean', 'ins_302_DAYS_ENTRY_PAYMENT_mean', 'ins_302_DAYS_ENTRY_PAYMENT_sum', 'ins_302_NUM_INSTALMENT_ratio_mean', 'ins_302_NUM_INSTALMENT_ratio_min', 'ins_302_NUM_INSTALMENT_ratio_var', 'ins_302_days_weighted_delay_max', 'ins_302_not-delayed_money_ratio_0_mean', 'pos_201_CNT_INSTALMENT_diff_mean', 'pos_201_CNT_INSTALMENT_ratio_mean', 'pos_201_MONTHS_BALANCE_mean', 'pos_201_MONTHS_BALANCE_min', 'pos_201_NAME_CONTRACT_STATUS_Active_sum', 'pos_201_NAME_CONTRACT_STATUS_Completed_sum', 'pos_201_SK_DPD_DEF_mean', 'pos_202_CNT_INSTALMENT_diff_mean', 'pos_202_CNT_INSTALMENT_ratio_mean', 'pos_202_CNT_INSTALMENT_ratio_var', 'pos_202_MONTHS_BALANCE_mean', 'pos_202_SK_DPD_DEF_mean', 'pos_202_SK_DPD_mean', 'pos_203_CNT_INSTALMENT_diff_mean', 'pos_203_CNT_INSTALMENT_ratio_mean', 'pos_204_SK_DPD_DEF_max', 'prev_101_active_AMT_GOODS_PRICE-dby-total_debt_min', 'prev_101_approved_AMT_ANNUITY-dby-app_AMT_ANNUITY_max', 'prev_101_approved_AMT_ANNUITY-dby-app_AMT_ANNUITY_mean', 'prev_101_approved_AMT_GOODS_PRICE-dby-total_debt_mean', 'prev_101_approved_AMT_GOODS_PRICE-dby-total_debt_min', 'prev_101_approved_APP_CREDIT_PERC_var', 'prev_101_approved_amt_paid_mean', 'prev_101_approved_amt_paid_sum', 'prev_101_completed_AMT_DOWN_PAYMENT_mean', 'prev_101_completed_AMT_GOODS_PRICE-dby-total_debt_max', 'prev_101_completed_APP_CREDIT_PERC_var', 'prev_101_completed_HOUR_APPR_PROCESS_START_mean', 'prev_105_DAYS_DECISION_ref_max', 'prev_105_amt_unpaid_sum-p-app', 'prev_105_approved_ratio', 'bb_601_Active_STATUS_0_mean', 'bb_601_Closed_MONTHS_BALANCE_var', 'bb_601_Closed_BURE_COUNT']
 #  0.7915954560368597 (ets 0.02)

categorical_feature = ['app_001_NAME_CONTRACT_TYPE',
                     'app_001_CODE_GENDER',
                     'app_001_FLAG_OWN_CAR',
                     'app_001_FLAG_OWN_REALTY',
                     'app_001_NAME_TYPE_SUITE',
                     'app_001_NAME_INCOME_TYPE',
                     'app_001_NAME_EDUCATION_TYPE',
                     'app_001_NAME_FAMILY_STATUS',
                     'app_001_NAME_HOUSING_TYPE',
                     'app_001_OCCUPATION_TYPE',
                     'app_001_WEEKDAY_APPR_PROCESS_START',
                     'app_001_ORGANIZATION_TYPE',
                     'app_001_FONDKAPREMONT_MODE',
                     'app_001_HOUSETYPE_MODE',
                     'app_001_WALLSMATERIAL_MODE',
                     'app_001_EMERGENCYSTATE_MODE']


X = pd.concat([
                pd.read_feather('../feature/train_'+f+'.f') for f in tqdm(features_best, mininterval=60)
               ]+[X], axis=1)

dtrain = lgb.Dataset(X[features_best], y, 
                     categorical_feature=list( set(X.columns)&set(categorical_feature)) )
gc.collect()

ret = lgb.cv(param, dtrain, 9999, nfold=5,
             early_stopping_rounds=50, verbose_eval=10,
             seed=SEED)

best_score = ret['auc-mean'][-1]
print(f'best_score: {best_score}')
utils.send_line(f'best_score: {best_score}')


# =============================================================================
# stepwise
# =============================================================================

features_curr = features_best[:]

for c in features_bure:
    print()
    gc.collect()
    
    features_new = features_curr[:]
    if c in features_new:
        print(f'drop {c}')
        features_new.remove(c)
    else:
        features_new.append(c)
        print(f'add {c}')
    
    dtrain = lgb.Dataset(X[features_new], y,
                         categorical_feature=list( set(features_new)&set(categorical_feature)) )
    ret = lgb.cv(param, dtrain, 9999, nfold=5,
                 early_stopping_rounds=50, verbose_eval=None,
                 seed=SEED)
    score = ret['auc-mean'][-1]
    print(f"auc-mean {score}")
    
    if best_score < score:
        print(f'UPDATE!    SCORE:{score:+.5f}    DIFF:{score-best_score:+.5f}')
        print(f'features: {features_new}')
        best_score = score
        features_curr = features_new
        utils.send_line(f'{c}: {best_score}')

#==============================================================================
utils.end(__file__)



