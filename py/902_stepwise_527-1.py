#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 23:19:07 2018

@author: Kazuki
"""

from glob import glob
import pandas as pd
import sys
sys.path.append('/home/kazuki_onodera/Python')
import lgbmextension as ex
import lightgbm as lgb
from time import sleep
import os
import gc
import utils
#utils.start(__file__)
#==============================================================================

SEED = 71

param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.01,
         'max_depth': -1,
         'num_leaves': 127,
         'max_bin': 100,
         'colsample_bytree': 0.5,
         'subsample': 0.5,
         'nthread': 64,
         'bagging_freq': 1,
         
         'seed': SEED, 
         'verbose': -1
         }

categorical_feature = ['NAME_CONTRACT_TYPE',
                     'CODE_GENDER',
                     'FLAG_OWN_CAR',
                     'FLAG_OWN_REALTY',
                     'NAME_TYPE_SUITE',
                     'NAME_INCOME_TYPE',
                     'NAME_EDUCATION_TYPE',
                     'NAME_FAMILY_STATUS',
                     'NAME_HOUSING_TYPE',
                     'OCCUPATION_TYPE',
                     'WEEKDAY_APPR_PROCESS_START',
                     'ORGANIZATION_TYPE',
                     'FONDKAPREMONT_MODE',
                     'HOUSETYPE_MODE',
                     'WALLSMATERIAL_MODE',
                     'EMERGENCYSTATE_MODE']

# =============================================================================
# wait
# =============================================================================

#while True:
#    if os.path.isfile('SUCCESS_901'):
#        break
#    else:
#        sleep(60*1)
#
#utils.reset_time()

# =============================================================================
# 
# =============================================================================
imp = pd.read_csv('LOG/imp_901_cv_527-1.py.csv').set_index('index')
feature_all = imp[imp['split'] != 0].index.tolist()

folders = sorted(glob('../data/*_train'))

def read_pickle(folder, usecols):
    df = pd.read_pickle(folder+'/000.p')
    col = list( set(usecols) & set(df.columns))
    if len(col)>0:
        df = utils.read_pickles(folder, col)
        return df
    else:
        print(f'{folder} desnt have valid columns')
        return pd.DataFrame()


X = pd.concat([
                read_pickle(f, feature_all) for f in (folders)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET


# =============================================================================
# LGB
# =============================================================================
#use_features = feature_all[:1000]
use_features = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'ORGANIZATION_TYPE', 'EXT_SOURCE_1', 'ins_gby-SK_ID_CURR-SK_ID_PREV_AMT_PAYMENT_min_mean', 'OCCUPATION_TYPE', 'DAYS_BIRTH', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'AMT_CREDIT', 'DAYS_EMPLOYED', 'NAME_EDUCATION_TYPE', 'OWN_CAR_AGE', 'DAYS_ID_PUBLISH', 'prev_gby-SK_ID_CURR-CODE_REJECT_REASON_DAYS_FIRST_DRAWING_sum_mean', 'bureau_gby-SK_ID_CURR-CREDIT_TYPE_DAYS_CREDIT_mean_mean', 'bureau_gby-SK_ID_CURR-CREDIT_TYPE_DAYS_CREDIT_ENDDATE_max_min', 'pos_gby-SK_ID_CURR-SK_ID_PREV_CNT_INSTALMENT_FUTURE_mean_mean', 'FLAG_DOCUMENT_3', 'bureau_gby-SK_ID_CURR_DAYS_CREDIT_max', 'pos_gby-SK_ID_CURR-SK_ID_PREV_CNT_INSTALMENT_FUTURE_mean_std', 'bureau_gby-SK_ID_CURR-CREDIT_CURRENCY_DAYS_CREDIT_mean_min', 'ins_gby-SK_ID_CURR-NUM_INSTALMENT_NUMBER_AMT_PAYMENT_mean_min', 'bureau_gby-SK_ID_CURR-CREDIT_ACTIVE_AMT_CREDIT_SUM_std_min', 'ins_gby-SK_ID_CURR-SK_ID_PREV_DAYS_INSTALMENT_sum_std', 'bureau_gby-SK_ID_CURR-CREDIT_CURRENCY_DAYS_CREDIT_mean_max', 'ins_gby-SK_ID_CURR-NUM_INSTALMENT_VERSION_AMT_PAYMENT_max_min', 'ins_gby-SK_ID_CURR-NUM_INSTALMENT_NUMBER_DAYS_ENTRY_PAYMENT_sum_max', 'bureau_gby-SK_ID_CURR-CREDIT_ACTIVE_AMT_CREDIT_MAX_OVERDUE_mean_sum', 'ins_gby-SK_ID_CURR-NUM_INSTALMENT_VERSION_AMT_PAYMENT_mean_std', 'bureau_gby-SK_ID_CURR_DAYS_CREDIT_std', 'bureau_gby-SK_ID_CURR-CREDIT_ACTIVE_AMT_CREDIT_SUM_DEBT_sum_std', 'prev_gby-SK_ID_CURR-CHANNEL_TYPE_DAYS_LAST_DUE_1ST_VERSION_min_max', 'ins_gby-SK_ID_CURR-NUM_INSTALMENT_VERSION_DAYS_ENTRY_PAYMENT_sum_mean', 'bureau_gby-SK_ID_CURR-CREDIT_ACTIVE_AMT_CREDIT_MAX_OVERDUE_min_max', 'pos_gby-SK_ID_CURR-NAME_CONTRACT_STATUS_SK_DPD_DEF_sum_std']

# benchmark
dtrain = lgb.Dataset(X[use_features], y, 
                     categorical_feature=list( set(categorical_feature)&set(use_features) ))
ret = lgb.cv(param, dtrain, 9999, nfold=5,
             early_stopping_rounds=50, verbose_eval=None,
             seed=SEED)

best_score = ret['auc-mean'][-1]
print(f'benchmark: {best_score}')
print(f'features: {use_features}')


for c in feature_all[1200:]:
    print()
    gc.collect()
    
    use_features_ = use_features[:]
    if c in use_features_:
        print(f'drop {c}')
        use_features_.remove(c)
    else:
        use_features_.append(c)
        print(f'add {c}')
    
    dtrain = lgb.Dataset(X[use_features_], y, 
                         categorical_feature=list( set(categorical_feature)&set(use_features_) ))
    ret = lgb.cv(param, dtrain, 9999, nfold=5,
                 early_stopping_rounds=50, verbose_eval=None,
                 seed=SEED)
    score = ret['auc-mean'][-1]
    print(f"auc-mean {score}")
    
    if best_score < score:
        print(f'UPDATE!    SCORE:{score:+.5f}    DIFF:{score-best_score:+.5f}')
        print(f'features: {use_features_}')
        best_score = score
        use_features = use_features_



#==============================================================================
utils.end(__file__)


