#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 00:30:46 2018

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
from glob import glob
import count
import utils, utils_cat
#utils.start(__file__)
#==============================================================================

PREF = 'f750_'

SEED = 71

HEAD = 600

param = {
         'objective': 'binary',
         'metric': 'auc',
         'learning_rate': 0.02,
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
paths = glob('LOG/imp_803*.csv')

for i,path in enumerate(paths):
    imp = pd.read_csv(path)
    imp['split'] /= imp['split'].max()
    imp['gain'] /= imp['gain'].max()
    imp['total'] = imp['split'] + imp['gain']
    imp.sort_values('total', ascending=False, inplace=True)
    imp.set_index('feature', inplace=True)
    if i==0:
        imp_all = imp
    else:
        imp_all['total'] += imp['total']

imp_all.sort_values('total', ascending=False, inplace=True)
imp_all.reset_index(inplace=True)


# train
use_files = (imp_all.head(HEAD).feature + '.f').tolist()

files = utils.get_use_files(use_files, True)

X_train = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)

# test
files = utils.get_use_files(use_files, False)

X_test = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)


# =============================================================================
# drop feature
# =============================================================================

imp = pd.read_csv('LOG/imp_750_adversarial.py.csv')
imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']
imp.sort_values('total', ascending=False, inplace=True)

# =============================================================================
# concat
# =============================================================================


X_train['target'] = 0
X_test['target']  = 1

feature_drop = imp.head(150).feature.tolist()


X = pd.concat([X_train, X_test], ignore_index=True)
y = X['target']
X.drop(['target']+feature_drop, axis=True, inplace=True)


if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

n_train = X_train.shape[0]

# =============================================================================
# cv
# =============================================================================

CAT = list( set(X.columns)&set(utils_cat.ALL))

dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
gc.collect()

ret = lgb.cv(param, dtrain, 9999, nfold=5,
             early_stopping_rounds=100, verbose_eval=50,
             seed=SEED)

result = f"CV auc-mean({HEAD}): {ret['auc-mean'][-1]}"
print(result)

utils.send_line(result)


model = lgb.train(params=param, train_set=dtrain, num_boost_round=100)
imp = ex.getImp(model)


# =============================================================================
# stepwise
# =============================================================================

for c in imp.feature:
    X_ = X.drop(c, axis=1)
    CAT = list( set(X_.columns)&set(utils_cat.ALL))
    dtrain = lgb.Dataset(X_, y, categorical_feature=CAT )
    gc.collect()
    
    ret = lgb.cv(param, dtrain, 9999, nfold=5,
                 early_stopping_rounds=100, verbose_eval=50,
                 seed=SEED)
    
    result = f"CV auc-mean(drop {c}): {ret['auc-mean'][-1]}"
    print(result)
    
    utils.send_line(result)



#==============================================================================
utils.end(__file__)
#utils.stop_instance()




