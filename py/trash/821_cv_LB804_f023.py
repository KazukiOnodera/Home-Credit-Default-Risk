#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 00:48:25 2018

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
from sklearn.model_selection import GroupKFold

from glob import glob
#import count
import utils, utils_best
utils.start(__file__)
#==============================================================================

SEED = 71

NFOLD = 6

new_features = ['f023']

COMMENT = new_features


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

col = []
files = []
for new_feature in new_features:
    col += [c for c in X.columns if new_feature in c]
    files += glob(f'../feature/train_{new_feature}*')

print('files:', len(files))

X.drop(col, axis=1, inplace=True)
X_ = pd.concat([pd.read_feather(f) for f in tqdm(files, mininterval=60)
                ], axis=1)
X = pd.concat([X, X_], axis=1)
del X_

if X.columns.duplicated().sum()>0:
    raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
print('no dup :) ')
print(f'X.shape {X.shape}')

gc.collect()

CAT = list( set(X.columns) & set(loader.category()) )
CAT += [new_features[0]+'_user_id']

tmp = pd.read_csv('../data/user_id_v4.csv.zip').set_index('SK_ID_CURR') # TODO: change
sub_train = pd.read_csv('../input/application_train.csv.zip', usecols=['SK_ID_CURR']).set_index('SK_ID_CURR')
sub_train['user_id'] = tmp.user_id
sub_train['g'] = sub_train.user_id % NFOLD

sub_train['y'] = y.values
#sub_train['cnt'] = sub_train.index.value_counts()
#sub_train['w'] = 1 / sub_train.cnt.values

group_kfold = GroupKFold(n_splits=NFOLD)
folds = group_kfold.split(X, sub_train['y'], sub_train['g'])

# =============================================================================
# cv
# =============================================================================
dtrain = lgb.Dataset(X, y, categorical_feature=CAT )
gc.collect()

ret, models = lgb.cv(param, dtrain, 9999, folds=folds, 
                     early_stopping_rounds=100, verbose_eval=50,
                     seed=SEED)

y_pred = ex.eval_oob(X, y, models, SEED)

result = f"CV auc-mean({COMMENT}): {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
print(result)
utils.send_line(result)

imp = ex.getImp(models)

imp['split'] /= imp['split'].max()
imp['gain'] /= imp['gain'].max()
imp['total'] = imp['split'] + imp['gain']

imp.sort_values('total', ascending=False, inplace=True)
imp.reset_index(drop=True, inplace=True)

imp.to_csv('LOG/imp_f023.csv', index=False)

"""

imp = pd.read_csv('LOG/imp_f023.csv')

"""
# =============================================================================
# 
# =============================================================================

X = loader.train()
y = utils.read_pickles('../data/label').TARGET

files = []
for new_feature in imp[imp.split!=0][imp.feature.str.startswith(new_features[0])].feature:
    files += glob(f'../feature/train_{new_feature}*')

X_ = pd.concat([pd.read_feather(f) for f in tqdm(files, mininterval=60)
                ], axis=1)

for i in range(30,1000,30):
    X_new = pd.concat([X, X_.iloc[:,:i] ], axis=1)
    CAT = list( set(X_new.columns) & set(loader.category()+[new_features[0]+'_user_id']) )
    
    dtrain = lgb.Dataset(X_new, y, categorical_feature=CAT )
    gc.collect()
    
    folds = group_kfold.split(X, sub_train['y'], sub_train['g'])
    ret, models = lgb.cv(param, dtrain, 9999, folds=folds, 
                         early_stopping_rounds=100, verbose_eval=50,
                         seed=SEED)
    
    y_pred = ex.eval_oob(X, y, models, SEED)
    
    result = f"CV auc-mean({COMMENT, i}): {ret['auc-mean'][-1]} + {ret['auc-stdv'][-1]}"
    print(result)
    utils.send_line(result)


#==============================================================================
utils.end(__file__)



