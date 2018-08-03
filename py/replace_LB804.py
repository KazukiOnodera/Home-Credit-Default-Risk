#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 11:56:51 2018

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

#new_feature = 'f110'

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
X = loader.train()
y = utils.read_pickles('../data/label').TARGET

col_var = [c for c in X.columns if c.endswith('_var')]

files = ('../feature/train_' + X.columns + '.f').tolist()

files_read = []
for file in files:
    if not os.path.isfile(file):
        file = file.replace('_var.f', '_std.f')
        if not os.path.isfile(file):
            print(file)
        else:
            files_read.append(file)
    else:
        files_read.append(file)


X_ = pd.concat([pd.read_feather(f) for f in files_read], axis=1)


CAT = list( set(X.columns) & set(loader.category()) )

# =============================================================================
# cv
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
gc.collect()

ret, models = lgb.cv(param, dtrain, 99999, nfold=7,
                     early_stopping_rounds=100, verbose_eval=50,
                     seed=111)

result = f"CV auc-mean(X): {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
print(result)

utils.send_line(result)

imp = ex.getImp(models)


# =============================================================================
# cv
# =============================================================================
dtrain = lgb.Dataset(X_, y, categorical_feature=CAT )
gc.collect()

ret, models = lgb.cv(param, dtrain, 99999, nfold=7,
                     early_stopping_rounds=100, verbose_eval=50,
                     seed=111)

result = f"CV auc-mean(X_): {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
print(result)

utils.send_line(result)

imp = ex.getImp(models)




#==============================================================================
utils.end(__file__)


