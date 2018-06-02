#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 16:42:14 2018

@author: Kazuki
"""

import gc
import pandas as pd
import sys
sys.path.append('/home/kazuki_onodera/Python')
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import multiprocessing
import utils
utils.start(__file__)
#==============================================================================

SEED = 71

X = utils.read_pickles('../data/101_train')
y = utils.read_pickles('../data/label').TARGET


if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')


param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.05,
         'max_depth': -1,
         'num_leaves': 511,
         'max_bin': 511,
         'colsample_bytree': 0.5,
         'subsample': 0.5,
         'nthread': multiprocessing.cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         'seed': SEED
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



dtrain = lgb.Dataset(X, y, categorical_feature=list( set(X.columns)&set(categorical_feature)) )

ret = lgb.cv(param, dtrain, 9999, nfold=5,
             early_stopping_rounds=50, verbose_eval=False,
             seed=SEED)
print(f"BENCHMARK CV auc-mean {ret['auc-mean'][-1]}")


le = LabelEncoder()

for c in X.columns:
    
    if c not in categorical_feature:
        print(f'categorize {c}')
        
        X_ = X.copy()
        X_[c].fillna('na dayo', inplace=True)
        X_[c] = le.fit_transform( X_[c].astype(str) )
        
        dtrain = lgb.Dataset(X_, y, categorical_feature=list( set(X.columns)&set(categorical_feature+[c])) )
        
        ret = lgb.cv(param, dtrain, 9999, nfold=5,
                     early_stopping_rounds=50, verbose_eval=False,
#                     categorical_feature=list( set(X.columns)&set(categorical_feature)),
                     seed=SEED)
        print(f"CV auc-mean {ret['auc-mean'][-1]}")
        gc.collect()



#==============================================================================
utils.end(__file__)




