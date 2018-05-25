#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 21:21:36 2018

@author: Kazuki
"""


import numpy as np
import pandas as pd
import sys
sys.path.append('/home/kazuki_onodera/Python')
import lgbmextension as ex
import lightgbm as lgb
import gc
import utils
#utils.start(__file__)
#==============================================================================

SEED = 71


X = pd.concat([utils.read_pickles('../data/101_train'), 
               utils.read_pickles('../data/102_train')], axis=1)
y = utils.read_pickles('../data/label').TARGET

param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.05,
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

for c in X.columns:
    print(f'drop {c}')
    gc.collect()
    categorical_feature_ = categorical_feature[:]
    if c in categorical_feature_:
        categorical_feature_.remove(c)
    dtrain = lgb.Dataset(X.drop(c, axis=1), y, 
                         categorical_feature=categorical_feature_)
    ret = lgb.cv(param, dtrain, 9999, nfold=5,
#                 categorical_feature=categorical_feature,
                 early_stopping_rounds=50, verbose_eval=None,
                 seed=SEED)
    print(f"auc-mean {ret['auc-mean'][-1]}")



#==============================================================================
utils.end(__file__)


