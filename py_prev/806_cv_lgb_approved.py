#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 20:27:14 2018

@author: Kazuki
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

HEADS = list(range(300, 1000, 100))

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
         'nthread': 16,
#         'nthread': cpu_count(),
         'bagging_freq': 1,
         'verbose':-1,
         'seed': SEED
         }

# =============================================================================
# load
# =============================================================================
imp = pd.read_csv('LOG/imp_801_imp_lgb.py.csv').sort_values('total', ascending=False)

files = ('../feature_prev/train_' + imp.head(max(HEADS)).feature + '.f').tolist()


X_all = pd.concat([
                pd.read_feather(f) for f in tqdm(files, mininterval=60)
               ], axis=1)
y = utils.read_pickles('../data/prev_label').TARGET

val = utils.read_pickles('../data/prev_train', ['NAME_CONTRACT_STATUS'])
ind = val[val['NAME_CONTRACT_STATUS']=='Approved'].index
y = y[ind]


sub_train = utils.read_pickles('../data/prev_train', ['SK_ID_CURR']).set_index('SK_ID_CURR').iloc[ind]
sub_train['y'] = y.values
sub_train['cnt'] = sub_train.index.value_counts()
sub_train['w'] = 1 / sub_train.cnt.values

group_kfold = GroupKFold(n_splits=NFOLD)
sub_train['g'] = sub_train.index % NFOLD


for HEAD in HEADS:
    
    X = X_all.iloc[ind, :HEAD]
    
    if X.columns.duplicated().sum()>0:
        raise Exception(f'duplicated!: { X.columns[X.columns.duplicated()] }')
    print('no dup :) ')
    print(f'X.shape {X.shape}')
    
    gc.collect()
    
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
    
    result = f"CV auc-mean({HEAD}): {ret['auc-mean'][-1]}"
    print(result)
    
    utils.send_line(result)


#==============================================================================
utils.end(__file__)
#utils.stop_instance()



