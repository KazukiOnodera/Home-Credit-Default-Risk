#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 22:55:34 2018

@author: kazuki.onodera
"""

from glob import glob
from os import system
import pandas as pd
import sys
sys.path.append('/home/kazuki_onodera/Python')
import lgbmextension as ex
import lightgbm as lgb
import multiprocessing
import utils
utils.start(__file__)
system('rm SUCCESS_901')
#==============================================================================

SEED = 71

folders = sorted(glob('../data/*_train'))

X = pd.concat([
#               utils.read_pickles('../data/101_train'), 
#               utils.read_pickles('../data/102_train'), 
#               utils.read_pickles('../data/103_train'), 
#               utils.read_pickles('../data/104_train')
        utils.read_pickles(f) for f in (folders)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET

print(f'X.shape {X.shape}')

param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.01,
         'max_depth': -1,
         'num_leaves': 511,
         'max_bin': 100,
         'colsample_bytree': 0.1,
         'subsample': 0.5,
         'nthread': multiprocessing.cpu_count(),
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

ret = lgb.cv(param, dtrain, 9999, nfold=5,
             early_stopping_rounds=50, verbose_eval=10,
             seed=SEED)
print(f"CV auc-mean {ret['auc-mean'][-1]}")

dtrain = lgb.Dataset(X, y, categorical_feature=categorical_feature)
model = lgb.train(param, dtrain, len(ret['auc-mean']))

imp = ex.getImp(model)


imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)

system('touch SUCCESS_901')

#==============================================================================
utils.end(__file__)


