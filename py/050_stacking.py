#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 14:06:35 2018

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
#utils.start(__file__)
#==============================================================================

PREF = 'f050'

os.system(f'rm ../feature/t*_{PREF}*')

SEED = 71
FOLD = 5
LOOP = 10

params = {
         'objective': 'binary',
         'metric': 'auc',
#         'metric': 'binary_logloss',
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


np.random.seed(SEED)

# =============================================================================
# load
# =============================================================================

use_files = [
        'f001', 
        'f002', 
             ]

files = utils.get_use_files(use_files, True)

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/label').TARGET


CAT = list( set(X.columns) & set(utils_cat.ALL) )

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()



files = utils.get_use_files(use_files, False)

X_test = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)

# =============================================================================
# stack
# =============================================================================

sub_train = pd.DataFrame(index=X.index)
sub_train['y_pred'] = 0

sub_test = pd.DataFrame(index=X_test.index)
sub_test['y_pred'] = 0

for j in range(LOOP):
    
    skf = StratifiedKFold(n_splits=FOLD, shuffle=True, 
                          random_state=np.random.randint(9999))
    
    for i,(train_index, test_index) in enumerate(skf.split(X, y)):
    
        dtrain = lgb.Dataset(X.iloc[train_index], y.iloc[train_index], 
                             categorical_feature=CAT)
        dvalid = lgb.Dataset(X.iloc[test_index], y.iloc[test_index], 
                             categorical_feature=CAT)
        gc.collect()
        
        model = lgb.train(params=params, train_set=dtrain, num_boost_round=9999, 
                          valid_sets=[dtrain, dvalid], 
                          valid_names=['train','valid'], 
    #                      categorical_feature=CAT, 
                          early_stopping_rounds=100,
                          verbose_eval=50,
    #                      seed=SEED
                          )
        
        sub_train.iloc[test_index, -1] += model.predict(X.iloc[test_index])
        sub_test['y_pred'] += model.predict(X_test)

sub_train['y_pred'] /= LOOP
sub_test['y_pred'] /= FOLD*LOOP



# =============================================================================
# otuput
# =============================================================================
utils.to_feature(sub_train.add_prefix(PREF), '../feature/train')
utils.to_feature(sub_test.add_prefix(PREF),  '../feature/test')

#==============================================================================
utils.end(__file__)
