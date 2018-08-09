#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 19:15:14 2018

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
from sklearn.model_selection import StratifiedKFold
#from glob import glob
import count
import utils, utils_cat
utils.start(__file__)
#==============================================================================

PREF = 'f011_'

os.system(f'rm ../feature/t*_{PREF}*')

SEED = 71
FOLD = 5
label_name = 'f001_EXT_SOURCE_1'



params = {
         'objective': 'regression',
         'metric': 'rmse',
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
         'nthread': 21,
#         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         'seed': SEED
         }


np.random.seed(SEED)
# =============================================================================
# load
# =============================================================================

prefixes = [
        'f001', 
        'f002', 
             ]

files = utils.get_use_files(prefixes, True)

X_train = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y_train = X_train[label_name]
X_train.drop(label_name, axis=1, inplace=True)


CAT = list( set(X_train.columns) & set(utils_cat.ALL) )

if X_train.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X_train.columns[X_train.columns.duplicated()] }')
print('no dup :) ')
print(f'X_train.shape {X_train.shape}')

gc.collect()



files = utils.get_use_files(prefixes, False)

X_test = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y_test = X_test[label_name]
X_test.drop(label_name, axis=1, inplace=True)



X_train_train, y_train_train = X_train[~y_train.isnull()], y_train[~y_train.isnull()]
X_train_test,  y_train_test  = X_train[y_train.isnull()],  y_train[y_train.isnull()]

X_test_train, y_test_train = X_test[~y_test.isnull()], y_test[~y_test.isnull()]
X_test_test,  y_test_test  = X_test[y_test.isnull()],  y_test[y_test.isnull()]

# =============================================================================
# cv
# =============================================================================
X = pd.concat([X_train_train, X_test_train])
y = pd.concat([y_train_train, y_test_train])

dtrain = lgb.Dataset(X, y, categorical_feature=CAT)
gc.collect()

ret, models = lgb.cv(params, dtrain, 99999, nfold=FOLD, stratified=False,
                     early_stopping_rounds=100, verbose_eval=50,
                     seed=SEED)


# =============================================================================
# 
# =============================================================================
#NROUND = int(len(ret['rmse-mean'])*1.3) # 12234
#print(f'NROUND: {NROUND}')
#
#dtrain = lgb.Dataset(X, y, categorical_feature=CAT)
#
#model = lgb.train(params, dtrain, NROUND)

train_ind = y_train.isnull()
test_ind  = y_test.isnull()

y_train.loc[train_ind] = 0
y_test.loc[test_ind] = 0

for model in models:
    y_train.loc[train_ind] += pd.Series(model.predict(X_train_test)).values
    y_test.loc[test_ind]   += pd.Series(model.predict(X_test_test)).values
    

y_train.loc[train_ind] /= len(models)
y_test.loc[test_ind]   /= len(models)

y_train = y_train.to_frame()
y_test = y_test.to_frame()
y_train.columns = [label_name.replace('f001_', '')]
y_test.columns = [label_name.replace('f001_', '')]

# =============================================================================
# otuput
# =============================================================================
utils.to_feature(y_train.add_prefix(PREF), '../feature/train')
utils.to_feature(y_test.add_prefix(PREF),  '../feature/test')

#==============================================================================
utils.end(__file__)

