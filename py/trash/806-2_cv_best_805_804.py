#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 18:53:50 2018

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


loader805 = utils_best.Loader('CV805_LB803')
loader804 = utils_best.Loader('LB804')

# =============================================================================
# load
# =============================================================================
X_805 = loader805.train()
X_804 = loader804.train()

col = X_804.columns.difference(X_805.columns)
X = pd.concat([X_805, X_804[col]], axis=1)

y = utils.read_pickles('../data/label').TARGET


#col = [c for c in X.columns if new_feature in c]
#X.drop(col, axis=1, inplace=True)
#files = glob(f'../feature/train_{new_feature}*')
#X_ = pd.concat([pd.read_feather(f) for f in tqdm(files, mininterval=60)
#                ], axis=1)
#
#X = pd.concat([X, X_], axis=1)
#del X_

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

CAT = list( set(X.columns) & set(loader805.category() + loader805.category() ) )

# =============================================================================
# cv
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
gc.collect()

seed = np.random.randint(9999)
ret, models = lgb.cv(param, dtrain, 9999, nfold=6,
                     early_stopping_rounds=100, verbose_eval=50,
                     seed=seed)

result = f"CV auc-mean: {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
print(result)

utils.send_line(result)

imp = ex.getImp(models)

#==============================================================================
utils.end(__file__)




