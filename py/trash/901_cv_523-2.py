#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 20:07:35 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('/home/kazuki_onodera/Python')
import lgbmextension as ex
import lightgbm as lgb
import utils
utils.start(__file__)
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

dtrain = lgb.Dataset(X, y, categorical_feature=categorical_feature)



yhat, imp, ret = ex.stacking(X, y, param, 9999, esr=50, seed=SEED,
                             categorical_feature=categorical_feature)


imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)


#==============================================================================
utils.end(__file__)


