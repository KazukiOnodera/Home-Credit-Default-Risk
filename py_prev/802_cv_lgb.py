#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 12:39:16 2018

@author: kazuki.onodera
"""

import gc, os
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append(f'/home/{os.environ.get("USER")}/PythonLibrary')
import lgbextension as ex
import lightgbm as lgb
from multiprocessing import cpu_count
#from glob import glob
from sklearn.model_selection import GroupKFold
import count
import utils_cat
import utils
utils.start(__file__)
#==============================================================================

NFOLD = 5

SEED = 71

HEAD = 500

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
imp = pd.read_csv('LOG/801_imp_lgb.py.csv')
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']
imp.sort_values('total', ascending=False, inplace=True)

use_files = (imp.head(HEAD).feature + '.f').tolist()

files = utils.get_use_files(use_files, True)

X = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/prev_label').TARGET


if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

sub_train = utils.read_pickles('../data/prev_train', ['SK_ID_CURR']).set_index('SK_ID_CURR')
sub_train['y'] = y.values
sub_train['cnt'] = sub_train.index.value_counts()
sub_train['w'] = 1 / sub_train.cnt.values

group_kfold = GroupKFold(n_splits=NFOLD)
sub_train['g'] = sub_train.index % NFOLD

CAT = list( set(X.columns)&set(utils_cat.ALL))


# =============================================================================
# cv
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
gc.collect()

ret = lgb.cv(param, dtrain, 9999, folds=group_kfold.split(X, sub_train['y'], 
                                                          sub_train['g']), 
             early_stopping_rounds=100, verbose_eval=50,
             seed=SEED)

result = f"CV auc-mean: {ret['auc-mean'][-1]}"
print(result)

utils.send_line(result)


#==============================================================================
utils.end(__file__)
utils.stop_instance()

