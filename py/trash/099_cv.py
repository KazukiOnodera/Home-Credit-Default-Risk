#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 09:02:36 2018

@author: Kazuki
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
import count
import utils, utils_cat
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
         'nthread': 32,
#         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         'seed': SEED
         }





# =============================================================================
# load
# =============================================================================

use_files = [
        'train_f001',
        'train_f002',
#        'train_f052',
#        'train_f053',
#        'train_f054',
        'train_f055',
        'train_f056',
        'train_f057',
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


# =============================================================================
# cv bench
# =============================================================================
dtrain = lgb.Dataset(X, y, 
                     categorical_feature=CAT)
gc.collect()

ret = lgb.cv(param, dtrain, 9999, nfold=10,
             early_stopping_rounds=100, verbose_eval=50,
             seed=SEED)

result = f"CV auc-mean(bench): {ret['auc-mean'][-1]}\nbest round {len(ret['auc-mean'])}"
print(result)

utils.send_line(result)


# =============================================================================
# imp
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
model = lgb.train(param, dtrain, 1000)
imp = ex.getImp(model).sort_values(['gain', 'feature'], ascending=[False, True])



# =============================================================================
# cv drop
# =============================================================================
dtrain = lgb.Dataset(X.drop(['f001_EXT_SOURCE_3', 'f001_EXT_SOURCE_1'], axis=1), y, 
                     categorical_feature=CAT)
gc.collect()

ret = lgb.cv(param, dtrain, 9999, nfold=10,
             early_stopping_rounds=100, verbose_eval=50,
             seed=SEED)

result = f"CV auc-mean(drop): {ret['auc-mean'][-1]}\nbest round {len(ret['auc-mean'])}"
print(result)






# =============================================================================
# load
# =============================================================================

use_files = [
        'train_f001', 
        'train_f003', 
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


# =============================================================================
# cv bench
# =============================================================================
dtrain = lgb.Dataset(X, y, 
                     categorical_feature=CAT)
gc.collect()

ret = lgb.cv(param, dtrain, 9999, nfold=10,
             early_stopping_rounds=100, verbose_eval=50,
             seed=SEED)

result = f"CV auc-mean(ta): {ret['auc-mean'][-1]}\nbest round {len(ret['auc-mean'])}"
print(result)

utils.send_line(result)


# =============================================================================
# imp
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
model = lgb.train(param, dtrain, 1000)
imp_ta = ex.getImp(model).sort_values(['gain', 'feature'], ascending=[False, True])




#==============================================================================
utils.end(__file__)



