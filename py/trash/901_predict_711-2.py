#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 00:28:10 2018

@author: Kazuki
"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import gc, os
import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
#import lgbextension as ex
import xgboost as xgb
from multiprocessing import cpu_count
from glob import glob
import utils, utils_cat
utils.start(__file__)
#==============================================================================

SEED = 71

LOOP = 1

NROUND = 2249

SUBMIT_FILE_PATH = '../output/711-2.csv.gz'

COMMENT = f'CV auc-1st fold: 0.803395 round: {NROUND}'

EXE_SUBMIT = True

params = {
          'booster': 'gbtree',  # gbtree, gblinear or dart
          'silent': 1,  # 0:printing mode 1:silent mode.
          'nthread': cpu_count(),
          'eta': 0.01,
          'gamma': 0.1,
          'max_depth': 6,
          'min_child_weight': 100,
          # 'max_delta_step': 0,
          'subsample': 0.9,
          'colsample_bytree': 0.8,
          'colsample_bylevel': 0.8,
          'lambda': 0.1,  # L2 regularization term on weights.
          'alpha': 0.1,  # L1 regularization term on weights.
          'tree_method': 'hist',
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
#          'seed': SEED
          }


use_files = []
np.random.seed(SEED)

# =============================================================================
# load train, test
# =============================================================================

files = utils.get_use_files(use_files, True)

X_train = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET

# maxwell
maxwell = pd.read_feather('../feature_someone/Maxwell_train.f')
X_train = pd.concat([X_train, maxwell], axis=1); del maxwell; gc.collect()


if X_train.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X_train.columns[X_train.columns.duplicated()] }')
print('no dup :) ')
print(f'X_train.shape {X_train.shape}')
gc.collect()

COL = X_train.columns.tolist()

CAT = list( set(X_train.columns)&set(utils_cat.ALL))
print(f'category: {CAT}')



# test
files = utils.get_use_files(use_files, False)

X_test = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
                ], axis=1)

# maxwell
maxwell = pd.read_feather('../feature_someone/Maxwell_test.f')
X_test = pd.concat([X_test, maxwell], axis=1)[COL]; gc.collect()

train_row = X_train.shape[0]

XX = pd.concat([X_train, X_test], ignore_index=True); del X_train, X_test; gc.collect()

XX = pd.get_dummies(XX, columns=CAT, drop_first=True)

X_train, X_test = XX.iloc[:train_row], XX.iloc[train_row:]
del XX; gc.collect()

dtrain = xgb.DMatrix(X_train, y)
dtest = xgb.DMatrix(X_test)

del X_train, X_test; gc.collect()


# =============================================================================
# training
# =============================================================================
models = []
for i in range(LOOP):
    print(f'LOOP: {i}')
    gc.collect()
    params.update({'seed':np.random.randint(9999)})
    model = xgb.train(params, dtrain, NROUND)
#    model.save_model(f'lgb{i}.model')
    models.append(model)


del dtrain; gc.collect()

"""

models = []
for i in range(LOOP):
    bst = lgb.Booster(model_file=f'lgb{i}.model')
    models.append(bst)

imp = ex.getImp(models)

"""


# =============================================================================
# test
# =============================================================================
sub = pd.read_pickle('../data/sub.p')

gc.collect()

label_name = 'TARGET'

sub[label_name] = 0
for model in models:
    y_pred = model.predict(dtest)
    sub[label_name] += pd.Series(y_pred).rank()
sub[label_name] /= LOOP
sub[label_name] /= sub[label_name].max()
sub['SK_ID_CURR'] = sub['SK_ID_CURR'].map(int)

sub.to_csv(SUBMIT_FILE_PATH, index=False, compression='gzip')

# =============================================================================
# submission
# =============================================================================
if EXE_SUBMIT:
    print('submit')
    utils.submit(SUBMIT_FILE_PATH, COMMENT)


#==============================================================================
utils.end(__file__)


