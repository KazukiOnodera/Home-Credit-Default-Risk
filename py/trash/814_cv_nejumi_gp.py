#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 14:01:34 2018

@author: kazuki.onodera
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
#import count
import utils, utils_best
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


# =============================================================================
# load
# =============================================================================
X = pd.read_pickle('../data/X_train_nejumi_gp.pkl.gz')
y = utils.read_pickles('../data/label').TARGET


if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()


# =============================================================================
# cv
# =============================================================================
dtrain = lgb.Dataset(X, y )
gc.collect()

ret, models = lgb.cv(param, dtrain, 99999, nfold=7,
                     early_stopping_rounds=100, verbose_eval=50,
                     seed=111)

y_pred = ex.eval_oob(X, y, models, 111)

result = f"CV auc-mean: {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
print(result)
utils.send_line(result)

imp = ex.getImp(models)


# =============================================================================
# cv loop
# =============================================================================
from sklearn.metrics import roc_auc_score

dtrain = lgb.Dataset(X, y, free_raw_data=False)
gc.collect()

y_pred = pd.Series(0, index=y.index)

for i in range(5):
    ret, models = lgb.cv(param, dtrain, 99999, nfold=7,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=i)
    
    y_pred += ex.eval_oob(X, y, models, i).rank()

y_pred /= y_pred.max()

auc_mean = roc_auc_score(y, y_pred)
result = f"CV auc-mean(nejumi gp): {auc_mean}"
print(result)
utils.send_line(result)


#==============================================================================
utils.end(__file__)
#utils.stop_instance()



