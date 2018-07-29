#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 12:40:39 2018

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
from multiprocessing import cpu_count
from sklearn.model_selection import StratifiedKFold
from glob import glob
import count
import utils
#utils.start(__file__)
#==============================================================================

PREF = 'f398_'

SEED = 71

NFOLD = 5

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
# load train
# =============================================================================

files = []
for i in range(318, 323):
    files += glob(f'../feature/train_f{i}*')

X_train = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET


if X_train.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X_train.columns[X_train.columns.duplicated()] }')
print('no dup :) ')
print(f'X_train.shape {X_train.shape}')

gc.collect()

X_train['cat'] = 1

sub_train = pd.DataFrame(index=X_train.index)

CAT = ['cat']
# =============================================================================
# load test
# =============================================================================

files = []
for i in range(318, 323):
    files += glob(f'../feature/test_f{i}*')

X_test = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)[X_train.columns[:-1]]

gc.collect()

X_test['cat'] = 1

sub_test = pd.DataFrame(index=X_test.index)

# =============================================================================
# predict
# =============================================================================
skf = StratifiedKFold(n_splits=NFOLD)

sub_train['y_pred'] = 0
sub_test['y_pred'] = 0
for train_index, valid_index in skf.split(X_train, y):
    dtrain = lgb.Dataset(X_train.iloc[train_index], y.iloc[train_index],
                         categorical_feature=CAT)
    dvalid = lgb.Dataset(X_train.iloc[valid_index], y.iloc[valid_index],
                         categorical_feature=CAT)
    
    model = lgb.train(params=param, train_set=dtrain, num_boost_round=9999, 
                  valid_sets=[dtrain, dvalid], 
                  valid_names=['train','valid'], 
                  early_stopping_rounds=100, 
                  #evals_result=evals_result, 
                  verbose_eval=50
                  )
    
    sub_train.iloc[valid_index, -1] = model.predict(X_train.iloc[valid_index])
    sub_test['y_pred'] += model.predict(X_test)

sub_test['y_pred'] /= NFOLD

print('train:', sub_train.y_pred.describe())
print('test:', sub_test.y_pred.describe())

# =============================================================================
# save
# =============================================================================

utils.to_feature(sub_train.add_prefix(PREF), '../feature/train')
utils.to_feature(sub_test.add_prefix(PREF),  '../feature/test')





#==============================================================================
utils.end(__file__)




