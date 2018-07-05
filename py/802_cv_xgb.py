#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 20:53:43 2018

@author: kazuki.onodera
"""

import gc
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append('/home/kazuki_onodera/PythonLibrary')
import xgbextension as ex
import xgboost as xgb
import multiprocessing
from glob import glob
import count
import os
import utils
#utils.start(__file__)
#==============================================================================

SEED = 71

params = {
          'booster': 'gbtree',  # gbtree, gblinear or dart
          'silent': 1,  # 0:printing mode 1:silent mode.
          'nthread': multiprocessing.cpu_count(),
          'eta': 0.02,
          'gamma': 0.1,
          'max_depth': 6,
          'min_child_weight': 100,
          # 'max_delta_step': 0,
          'subsample': 0.9,
          'colsample_bytree': 0.8,
          'colsample_bylevel': 0.8,
          'lambda': 0.1,  # L2 regularization term on weights.
          'alpha': 0.1,  # L1 regularization term on weights.
          'tree_method': 'auto',
          # 'sketch_eps': 0.03,
          'scale_pos_weight': 1,
          # 'updater': 'grow_colmaker,prune',
          # 'refresh_leaf': 1,
          # 'process_type': 'default',
          'grow_policy': 'depthwise',
          # 'max_leaves': 0,
          'max_bin': 256,
          # 'predictor': 'cpu_predictor',
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
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

use_files = []



# =============================================================================
# load
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

X = pd.get_dummies(X, columns=categorical_feature, drop_first=True)

X = X.rank(method='dense')


# =============================================================================
# cv
# =============================================================================
dtrain = xgb.DMatrix(X, y)
gc.collect()

ret = xgb.cv(params, dtrain, 9999, nfold=5,
             early_stopping_rounds=50, verbose_eval=10,
             seed=SEED)

result = f"CV auc-mean {ret['auc-mean'][-1]}"
print(result)
utils.send_line(result)


#==============================================================================
utils.end(__file__)

utils.stop_instance()

