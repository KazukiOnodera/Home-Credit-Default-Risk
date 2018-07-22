#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 13:33:43 2018

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
import count
import utils, utils_cat, utils_best
utils.start(__file__)
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
X = utils_best.load_best_train()
y = utils.read_pickles('../data/label').TARGET

weight = pd.read_feather('../feature/train_f750_y_pred.f')

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

CAT = list( set(X.columns)&set(utils_cat.ALL) )

# =============================================================================
# cv
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=CAT ,
                     weight=weight.values.flatten())
gc.collect()

ret = lgb.cv(param, dtrain, 9999, nfold=7,
             early_stopping_rounds=100, verbose_eval=50,
             seed=SEED)

result = f"CV auc-mean(weight as weight): {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
print(result)

utils.send_line(result)


# =============================================================================
# cv
# =============================================================================

dtrain = lgb.Dataset(pd.concat([X, weight], axis=1), y, 
                     categorical_feature=CAT)
gc.collect()

ret = lgb.cv(param, dtrain, 9999, nfold=7,
             early_stopping_rounds=100, verbose_eval=50,
             seed=SEED)

result = f"CV auc-mean(weight as feature): {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
print(result)

utils.send_line(result)



#==============================================================================
utils.end(__file__)
utils.stop_instance()



