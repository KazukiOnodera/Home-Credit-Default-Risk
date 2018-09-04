#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 15:36:39 2018

@author: kazuki.onodera
"""

import gc, os
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count, Pool
#from glob import glob
from sklearn.metrics import roc_auc_score
import utils_cat
import utils
utils.start(__file__)
#==============================================================================

SEED = 71

LOOP = 3

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
         
         'colsample_bytree': 0.7,
         'subsample': 0.5,
#         'nthread': 16,
         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         'seed': SEED
         }



imp_files = ['LOG/imp_181_imp.py.csv', 
         'LOG/imp_182_imp.py.csv', 
         'LOG/imp_183_imp.py.csv', 
         'LOG/imp_184_imp.py.csv', 
         'LOG/imp_185_imp.py.csv', ]

# =============================================================================
# load
# =============================================================================
files_tr = []
files_te = []

for p in imp_files:
    imp = pd.read_csv(p)
    imp = imp[imp.split>2]
    files_tr += ('../feature/train_' + imp.feature + '.f').tolist()
    files_te += ('../feature/test_' + imp.feature + '.f').tolist()

files_tr = sorted(set(files_tr))
files_te = sorted(set(files_te))
print('features:', len(files_tr))



X_tr = pd.concat([
                pd.read_feather(f) for f in tqdm(files_tr, mininterval=60)
               ], axis=1)
X_tr['y'] = 0

X_te = pd.concat([
                pd.read_feather(f) for f in tqdm(files_te, mininterval=60)
               ], axis=1)
X_te['y'] = 1

train_len = X_tr.shape[0]

X = pd.concat([X_tr, X_te], ignore_index=True); del X_tr, X_te
y = X['y']; del X['y']

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

CAT = list( set(X.columns)&set(utils_cat.ALL))

# =============================================================================
# training with cv
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=CAT, 
                     free_raw_data=False)

model_all = []
y_pred = pd.Series(0, index=y.index)
for i in range(LOOP):
    gc.collect()
    param['seed'] = i
    ret, models = lgb.cv(param, dtrain, 9999, nfold=NFOLD,
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=i)
    model_all += models
    y_pred += ex.eval_oob(X, y, models, i)
    
    auc_mean = roc_auc_score(y, y_pred)
    result = f"CV auc-mean(loop {i}): {auc_mean} {ret['auc-mean'][-1]}"
    print(result)
    utils.send_line(result)
    
y_pred /= LOOP

auc_mean = roc_auc_score(y, y_pred)
result = f"CV auc-mean: {auc_mean}"
print(result)
utils.send_line(result)


y_pred.name = 'f190_adv'
y_pred = y_pred.to_frame()

# =============================================================================
# output
# =============================================================================
train = y_pred.iloc[:train_len,]
test  = y_pred.iloc[train_len:,]

utils.to_feature(train, '../feature/train')
utils.to_feature(test, '../feature/test')


#==============================================================================
utils.end(__file__)
utils.stop_instance()
