#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 17:11:08 2018

@author: Kazuki
"""

import gc
import pandas as pd
import sys
sys.path.append('/home/kazuki_onodera/Python')
import lgbmextension as ex
import lightgbm as lgb
import multiprocessing
from glob import glob
import utils
utils.start(__file__)
#==============================================================================

# setting

SEED = 71

#imp_file = 'LOG/imp_901_cv_610-1.py.csv'
imp_file = None

#==============================================================================
if imp_file is None:
    remove_names = []
else:
    imp = pd.read_csv(imp_file).set_index('index')
    remove_names = imp[imp['split']==0].index.tolist()

folders = sorted(glob('../feature/train*.pkl'))
folders_ = []
if len(remove_names)>0:
    for i in remove_names:
        for j in folders:
#            if i not in j:
            if not j.endswith(i):
                folders_.append(j)
    folders = folders_

folders = sorted(glob('../feature/train*.pkl'))
X = pd.concat([
                pd.read_pickle(f) for f in (folders)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET


if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.01,
         'max_depth': -1,
         'num_leaves': 255,
         'max_bin': 255,
         'colsample_bytree': 0.5,
         'subsample': 0.5,
         'nthread': multiprocessing.cpu_count(),
         'bagging_freq': 1,
#         'verbose':-1,
         'seed': SEED
         }


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


dtrain = lgb.Dataset(X, y, categorical_feature=list( set(X.columns)&set(categorical_feature)) )
gc.collect()

ret = lgb.cv(param, dtrain, 9999, nfold=5,
             early_stopping_rounds=50, verbose_eval=10,
             seed=SEED)
print(f"CV auc-mean {ret['auc-mean'][-1]}")


dtrain = lgb.Dataset(X, y, categorical_feature=list( set(X.columns)&set(categorical_feature)) )
model = lgb.train(param, dtrain, len(ret['auc-mean']))

imp = ex.getImp(model)


imp.to_csv(f'LOG/imp_{__file__}.csv', index=False)

#==============================================================================
utils.end(__file__)




