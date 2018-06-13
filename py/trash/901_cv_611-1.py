#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 20:25:05 2018

@author: Kazuki
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
import utils
utils.start(__file__)
#==============================================================================

# setting

SEED = 71

#imp_file = 'LOG/imp_901_cv_610-2.py.csv'
imp_file = None

#==============================================================================
utils.check_feature()

if imp_file is None:
    remove_names = []
else:
    imp = pd.read_csv(imp_file).set_index('index')
    remove_names = imp[imp['split']==0].index.tolist()

files = sorted(glob('../feature/train*.f'))
remove_files = []
if len(remove_names)>0:
    for i in files:
        for j in remove_names:
            if i.endswith(j+'.f'):
                remove_files.append(i)
                break
    
    print(f'remove {len(remove_files)} files')
    files = sorted(list( set(files)-set(remove_files) ))

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=100)
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


