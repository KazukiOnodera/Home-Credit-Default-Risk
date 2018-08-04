#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 11:08:20 2018

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


loader = utils_best.Loader('LB804')

# =============================================================================
# load
# =============================================================================
X = loader.train()
y = utils.read_pickles('../data/label').TARGET


if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

CAT = list( set(X.columns) & set(loader.category()) )

# =============================================================================
# cv
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
gc.collect()

ret, models = lgb.cv(param, dtrain, 99999, nfold=7,
                     early_stopping_rounds=100, verbose_eval=50,
                     seed=111)

result = f"CV auc-mean: {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
print(result)

utils.send_line(result)

imp = ex.getImp(models)

# =============================================================================
# predict
# =============================================================================
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

X_test = loader.test()

sub_train = pd.DataFrame(index=X.index)
sub_test  = pd.DataFrame(index=X_test.index)

sub_train['y'] = y
sub_train['y_pred'] = 0
sub_test['y_pred'] = 0

skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=111)
for (train_index, valid_index),model in zip(skf.split(X, y), models):
    X_val, y_val = X.iloc[valid_index], y.iloc[valid_index]
    y_pred = model.predict(X_val)
    print(roc_auc_score(y_val.values, y_pred))
    
    sub_train.iloc[valid_index, -1] = y_pred
    sub_test['y_pred'] += model.predict(X_test)

sub_test['y_pred'] /= 7

# =============================================================================
# save
# =============================================================================
sub_train.to_csv('../data/LB804_train_pred.csv', index=False)
sub_test.to_csv('../data/LB804_test_pred.csv', index=False)


#==============================================================================
utils.end(__file__)
#utils.stop_instance()



