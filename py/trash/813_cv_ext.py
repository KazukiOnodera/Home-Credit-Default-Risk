#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 10:25:31 2018

@author: Kazuki
"""

import gc, os
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count, Pool
from glob import glob
from sklearn.metrics import roc_auc_score
import utils, utils_best
#utils.start(__file__)
#==============================================================================

SEED = 71

new_features = ['f014', 'f015', 'f016']

COMMENT = new_features


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


loader = utils_best.Loader('LB804')

# =============================================================================
# load
# =============================================================================
X_old = loader.train()
y = utils.read_pickles('../data/label').TARGET

files_tr = utils.get_use_files(new_features, True)


X_ = pd.concat([pd.read_feather(f) for f in tqdm(files_tr, mininterval=60)
                ], axis=1)
X_new = pd.concat([X_old, X_], axis=1).drop(['f001_EXT_SOURCE_1', 'f001_EXT_SOURCE_2','f001_EXT_SOURCE_3'], axis=1)
del X_

if X_new.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X_new.columns[X_new.columns.duplicated()] }')
print('no dup :) ')
print(f'X_new.shape {X_new.shape}')

gc.collect()

CAT = list( set(X_new.columns) & set(loader.category()) )

## =============================================================================
## cv old
## =============================================================================
#dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
#gc.collect()
#
#ret, models = lgb.cv(param, dtrain, 99999, nfold=7,
#                     early_stopping_rounds=100, verbose_eval=50,
#                     seed=111)
#
#y_pred = ex.eval_oob(X, y, models, 111)
#
#result = f"CV auc-mean({COMMENT}): {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
#print(result)
#utils.send_line(result)
#
#imp = ex.getImp(models)


# =============================================================================
# cv loop
# =============================================================================

dtrain = lgb.Dataset(X_new, y, categorical_feature=CAT, free_raw_data=False)
gc.collect()

y_pred = pd.Series(0, index=y.index)

for i in range(5):
    ret, models = lgb.cv(param, dtrain, 99999, nfold=7,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=i)
    
    y_pred += ex.eval_oob(X_new, y, models, i).rank()

y_pred /= y_pred.max()

auc_mean = roc_auc_score(y, y_pred)
result = f"CV auc-mean(ext imp): {auc_mean}"
print(result)
utils.send_line(result)

#==============================================================================
utils.end(__file__)
#utils.stop_instance()




