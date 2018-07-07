#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 09:02:36 2018

@author: Kazuki
"""


import gc
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
sys.path.append('/home/kazuki_onodera/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count, Pool
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
         'max_depth': 6,
         'num_leaves': 63,
         'max_bin': 255,
         
         'min_child_weight': 10,
         'min_data_in_leaf': 150,
         'reg_lambda': 0.5,  # L2 regularization term on weights.
         'reg_alpha': 0.5,  # L1 regularization term on weights.
         
         'colsample_bytree': 0.9,
         'subsample': 0.9,
#         'nthread': 32,
         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         'seed': SEED
         }


categorical_feature = ['f001_NAME_CONTRACT_TYPE',
                     'f001_CODE_GENDER',
                     'f001_FLAG_OWN_CAR',
                     'f001_FLAG_OWN_REALTY',
                     'f001_NAME_TYPE_SUITE',
                     'f001_NAME_INCOME_TYPE',
                     'f001_NAME_EDUCATION_TYPE',
                     'f001_NAME_FAMILY_STATUS',
                     'f001_NAME_HOUSING_TYPE',
                     'f001_OCCUPATION_TYPE',
                     'f001_WEEKDAY_APPR_PROCESS_START',
                     'f001_ORGANIZATION_TYPE',
                     'f001_FONDKAPREMONT_MODE',
                     'f001_HOUSETYPE_MODE',
                     'f001_WALLSMATERIAL_MODE',
                     'f001_EMERGENCYSTATE_MODE']


use_files = ['train_f0']


# =============================================================================
# load
# =============================================================================

files = utils.get_use_files(use_files, True)

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET


# nejumi
files = sorted(glob('../feature_nejumi/*train*'))
X['CNT_PAYMENT'] = np.load(files[0])
X['nejumi_v2'] = np.load(files[1])

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

X = X.rank(method='dense')
gc.collect()


# =============================================================================
# cv bench
# =============================================================================
dtrain = lgb.Dataset(X.iloc[:,:-2], y, 
                     categorical_feature=list( set(X.columns)&set(categorical_feature)) )
gc.collect()

ret = lgb.cv(param, dtrain, 9999, nfold=5,
             early_stopping_rounds=100, verbose_eval=50,
             seed=SEED)

result = f"CV auc-mean(bench): {ret['auc-mean'][-1]}\nbest round {len(ret['auc-mean'])}"
print(result)

utils.send_line(result)



# =============================================================================
# cv with CNT_PAYMENT
# =============================================================================
dtrain = lgb.Dataset(X.iloc[:,:-1], y, 
                     categorical_feature=list( set(X.columns)&set(categorical_feature)) )
gc.collect()

ret = lgb.cv(param, dtrain, 9999, nfold=5,
             early_stopping_rounds=100, verbose_eval=50,
             seed=SEED)

result = f"CV auc-mean(with CNT_PAYMENT): {ret['auc-mean'][-1]}\nbest round {len(ret['auc-mean'])}"
print(result)

utils.send_line(result)


# =============================================================================
# cv with CNT_PAYMENT + nejumi_v3
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=list( set(X.columns)&set(categorical_feature)) )
gc.collect()

ret = lgb.cv(param, dtrain, 9999, nfold=5,
             early_stopping_rounds=100, verbose_eval=50,
             seed=SEED)

result = f"CV auc-mean(with CNT_PAYMENT + nejumi_v3): {ret['auc-mean'][-1]}\nbest round {len(ret['auc-mean'])}"
print(result)

utils.send_line(result)


#==============================================================================
utils.end(__file__)
